"""Stage 5: every estimator recovers a planted value; the simulation reproduces the variance and
correlations it is given; recalibration fixes known distortions and leaves good data alone."""

import json
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from nbaprops.db.base import session_scope
from nbaprops.db.models import ModelRun, Projection
from nbaprops.features import build as fb
from nbaprops.models import calibrate, simulate
from nbaprops.models import minutes as mm
from nbaprops.models import project as pj
from nbaprops.models import rates as rm
from tests import feature_fixtures as ff

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def league():
    df = fb.training_frame(ff.build(n_regular=82))
    return df[df.season == "2024-25"], df[df.season == "2025-26"], df


@pytest.fixture(scope="module")
def evaluated(league):
    train, test, everything = league
    return pj.evaluate(train, test, everything, n_sims=2000)


# --- rates: estimators recover planted values ---------------------------------


def test_nb_alpha_recovers_planted_dispersion():
    mu = RNG.uniform(2, 25, 40_000)
    y = RNG.poisson(mu * RNG.gamma(1 / 0.10, 0.10, mu.shape))
    assert rm.nb_alpha(y, mu) == pytest.approx(0.10, abs=0.01)


def test_teammate_shared_variance_recovers_planted_shock():
    rows = []
    for g in range(3000):
        shock = RNG.gamma(1 / 0.02, 0.02)  # one shock per team-game, variance 0.02
        for p in range(8):
            mu = 5 + 3 * p
            rows.append({"game_id": g, "team_id": 1, "mu": mu, "y": RNG.poisson(mu * shock)})
    df = pd.DataFrame(rows)
    assert rm.shared_variance_teammates(df, "y", "mu") == pytest.approx(0.02, abs=0.004)


def _rate_frame(n, effect):
    df = pd.DataFrame({c: np.nan for c in fb.FEATURE_COLUMNS}, index=range(n))
    df["usage_l10"] = RNG.uniform(0.12, 0.32, n)
    df["absent_usage_sum"] = RNG.choice([0.0, 0.3], n, p=[0.8, 0.2])
    for s in rm.STATS:
        df[f"{s}_pm_shrunk"] = {"pts": 0.5, "reb": 0.2, "ast": 0.12, "fg3m": 0.05}[s]
    df["y_minutes"] = RNG.uniform(15, 38, n)
    lift = np.exp(effect * df["absent_usage_sum"])
    for s in rm.STATS:
        df[f"y_{s}"] = RNG.poisson(df[f"{s}_pm_shrunk"] * df["y_minutes"] * lift)
    df["game_date"] = pd.Timestamp("2024-10-22") + pd.to_timedelta(np.arange(n) // 20, "D")
    df["game_id"] = (np.arange(n) // 20).astype(str)
    df["team_id"] = (np.arange(n) // 10) % 2
    return df


def test_rate_model_learns_a_real_effect_and_gates_out_noise():
    learned = rm.fit(_rate_frame(12_000, effect=1.0))  # 30% more production when usage frees up
    assert learned.rounds["pts"] > 0
    probe = _rate_frame(2, effect=0)
    probe["absent_usage_sum"] = [0.0, 0.3]
    r = learned.rates(probe)["pts"]
    assert r[1] / r[0] == pytest.approx(np.exp(0.3), rel=0.08)
    noise = rm.fit(_rate_frame(12_000, effect=0.0))
    assert noise.rounds["pts"] == 0  # nothing to learn: anchor only


def test_threes_rate_never_exceeds_points():
    model = rm.fit(_rate_frame(4000, effect=0.0))
    probe = _rate_frame(3, effect=0)
    probe["fg3m_pm_shrunk"] = 0.5  # absurd: threes alone would exceed points
    r = model.rates(probe)
    assert (3 * r["fg3m"] <= r["pts"]).all()


def test_rate_model_round_trip(tmp_path):
    model = rm.fit(_rate_frame(4000, effect=1.0))
    model.save(tmp_path / "r.json")
    again = rm.RateModel.load(tmp_path / "r.json")
    probe = _rate_frame(50, effect=1.0)
    for s in rm.STATS:
        assert again.rates(probe)[s] == pytest.approx(model.rates(probe)[s])


# --- simulation --------------------------------------------------------------


def _manual_rates(alpha=0.08, v_env=0.0, v_player=0.0):
    booster = lgb.train(
        {"objective": "poisson", "verbose": -1},
        lgb.Dataset(np.zeros((10, len(fb.FEATURE_COLUMNS))), np.ones(10)),
        num_boost_round=1,
    )
    return rm.RateModel(
        boosters={s: booster for s in rm.STATS},
        rounds={s: 0 for s in rm.STATS},
        league_rates={"pts": 0.5, "reb": 0.2, "ast": 0.12, "fg3m": 0.05},
        alpha={s: [alpha] * 3 for s in rm.SIM_STATS},
        usage_edges=[-np.inf, 0.18, 0.24, np.inf],
        v_env={s: v_env for s in rm.SIM_STATS},
        v_player=v_player,
        usage_median=0.2,
    )


def _game_rows(n=10):
    df = pd.DataFrame({c: np.nan for c in fb.FEATURE_COLUMNS}, index=range(n))
    df["is_home"] = [1] * (n // 2) + [0] * (n - n // 2)
    df["started_share_l10"] = [1.0, 1.0, 1.0, 0.0, 0.0] * (n // 5)
    df["usage_l10"] = 0.2
    df["rating_total"], df["rating_team_margin"] = 225.0, 0.0
    for s, r in {"pts": 0.6, "reb": 0.2, "ast": 0.15, "fg3m": 0.06}.items():
        df[f"{s}_pm_shrunk"] = r
    return df


def _fixed_minutes(n, mins=32.0, sd=0.01):
    return mm.MinutesDist(
        np.full(n, mins), np.full(n, sd), np.full(n, sd), np.zeros(n), np.full(n, 10.0)
    )


def test_points_variance_matches_target_with_shared_shocks_included():
    rows = _game_rows()
    rates = _manual_rates(alpha=0.08, v_env=0.02, v_player=0.01)
    params = simulate.ScenarioParams(sd_total=18, sd_margin=13, rho_starter=0, rho_bench=0)
    sims = simulate.simulate_game(rows, _fixed_minutes(len(rows)), rates, params, 60_000, 1)
    mu = 32 * 0.6
    target = mu + 0.08 * mu**2  # the data's total variance, shared parts included
    assert sims["pts"].mean(axis=1) == pytest.approx(np.full(len(rows), mu), rel=0.02)
    assert sims["pts"].var(axis=1) == pytest.approx(np.full(len(rows), target), rel=0.08)


def test_shared_environment_correlates_teammates_only_when_present():
    rows = _game_rows()
    params = simulate.ScenarioParams(18, 13, 0, 0)
    with_env = simulate.simulate_game(
        rows, _fixed_minutes(10), _manual_rates(v_env=0.03), params, 40_000, 2
    )
    without = simulate.simulate_game(
        rows, _fixed_minutes(10), _manual_rates(v_env=0.0), params, 40_000, 2
    )
    assert np.corrcoef(with_env["pts"][0], with_env["pts"][1])[0, 1] > 0.05
    assert abs(np.corrcoef(without["pts"][0], without["pts"][1])[0, 1]) < 0.02


def test_blowouts_cut_starters_and_lift_bench_while_keeping_each_minutes_distribution():
    rows = _game_rows()
    dist = mm.MinutesDist(
        np.full(10, 30.0), np.full(10, 4.0), np.full(10, 3.0), np.zeros(10), np.full(10, 10.0)
    )
    params = simulate.ScenarioParams(18, 13, rho_starter=-0.4, rho_bench=0.3)
    sims = simulate.simulate_game(rows, dist, _manual_rates(), params, 50_000, 3)
    blowout = np.abs(sims["_home_margin"])
    assert np.corrcoef(sims["minutes"][0], blowout)[0, 1] < -0.2  # starter
    assert np.corrcoef(sims["minutes"][3], blowout)[0, 1] > 0.1  # bench
    for p in (0.1, 0.5, 0.9):  # marginal minutes unchanged by the copula
        assert np.quantile(sims["minutes"][0], p) == pytest.approx(dist.quantile(p)[0], abs=0.3)


def test_lines_are_internally_consistent():
    rows = _game_rows()
    sims = simulate.simulate_game(
        rows, _fixed_minutes(10), _manual_rates(v_player=0.01), simulate.ScenarioParams(), 20_000, 4
    )
    assert (sims["pts"] >= 3 * sims["fg3m"]).all()
    assert (sims["pra"] == sims["pts"] + sims["reb"] + sims["ast"]).all()
    assert sims["fg3m"].mean(axis=1) == pytest.approx(np.full(10, 32 * 0.06), rel=0.05)


def test_market_lines_center_the_scenario_and_seeds_reproduce():
    rows = _game_rows()
    args = (rows, _fixed_minutes(10), _manual_rates(), simulate.ScenarioParams(), 20_000)
    a = simulate.simulate_game(*args, seed=5, total0=240.0, home_margin0=-8.0)
    assert a["_total"].mean() == pytest.approx(240, abs=0.5)
    assert a["_home_margin"].mean() == pytest.approx(-8, abs=0.3)
    b = simulate.simulate_game(*args, seed=5, total0=240.0, home_margin0=-8.0)
    assert (a["pts"] == b["pts"]).all()
    assert simulate.game_seed("run1", "g1") != simulate.game_seed("run2", "g1")


def test_scenario_fit_finds_no_blowout_effect_where_there_is_none(league):
    train, _, everything = league
    params = simulate.fit_scenario(train, mm.fit(train), everything)
    assert abs(params.rho_starter) < 0.1 and abs(params.rho_bench) < 0.1
    assert 5 < params.sd_margin < 30 and 5 < params.sd_total < 40


def test_p_over_and_push():
    draws = np.array([[10, 11, 12, 12, 14]])
    assert simulate.p_over(draws, np.array([11.5]))[0] == pytest.approx(0.6)
    assert simulate.p_over(draws, np.array([12.0]))[0] == pytest.approx(0.2)
    assert simulate.p_push(draws, np.array([12.0]))[0] == pytest.approx(0.4)


# --- recalibration -----------------------------------------------------------


def _probs(n, distort):
    true = RNG.uniform(0.05, 0.95, n)
    hit = (RNG.random(n) < true).astype(float)
    return true**distort if distort != 1 else true, hit


def test_recalibration_fixes_a_distortion():
    p, hit = _probs(20_000, distort=0.6)  # systematically too high
    cal = calibrate.Calibrator.fit({"pts": p}, {"pts": hit})
    assert "pts" in cal.curves
    p2, hit2 = _probs(20_000, distort=0.6)
    assert calibrate.brier(cal.apply("pts", p2), hit2) < calibrate.brier(p2, hit2) - 0.005
    assert calibrate.expected_calibration_error(cal.apply("pts", p2), hit2) < 0.015


def test_recalibration_leaves_calibrated_markets_alone():
    p, hit = _probs(20_000, distort=1)
    cal = calibrate.Calibrator.fit({"reb": p}, {"reb": hit})
    assert "reb" not in cal.curves
    assert cal.apply("reb", np.array([0.37]))[0] == pytest.approx(0.37)


def test_calibration_failures_only_for_real_and_significant_errors():
    p, hit = _probs(50_000, distort=1)
    assert calibrate.calibration_failures(p, hit) == []
    biased, hit = _probs(50_000, distort=0.6)
    assert calibrate.calibration_failures(biased, hit)


def test_calibrator_round_trip(tmp_path):
    p, hit = _probs(20_000, distort=0.6)
    cal = calibrate.Calibrator.fit({"ast": p}, {"ast": hit})
    cal.save(tmp_path / "c.json")
    again = calibrate.Calibrator.load(tmp_path / "c.json")
    assert again.apply("ast", p[:100]) == pytest.approx(cal.apply("ast", p[:100]))


# --- end to end --------------------------------------------------------------


def test_evaluation_report_on_the_league(evaluated):
    report, _ = evaluated
    assert set(report["markets"]) == set(simulate.MARKETS)
    for market, v in report["markets"].items():
        assert v["mae_improvement"] > -0.02, market  # this league's truth is ~season average
        assert v["ece_raw"] < 0.03, market
        assert v["calibration_failures"] == [], market
    assert report["passes_calibration_target"]


def test_bundle_round_trip_reproduces_projections(league, tmp_path):
    train, test, everything = league
    bundle = pj.fit_bundle(
        train, test[test.game_date < test.game_date.median()], everything, "stage5-test", n_sims=500
    )
    bundle.save(tmp_path / "b")
    again = pj.Bundle.load(tmp_path / "b")
    day = test[test.game_date == test.game_date.max()]
    now = datetime(2026, 4, 1, tzinfo=UTC)
    a = pj.project_frame(day, bundle, 1000, everything, now)
    b = pj.project_frame(day, again, 1000, everything, now)
    pd.testing.assert_frame_equal(a, b)
    for probs in a["p_over"]:
        values = list(probs.values())
        assert all(0 < v < 1 for v in values)
        assert values == sorted(values, reverse=True)  # higher line, lower P(over)


@pytest.mark.db
def test_projections_stored_under_a_registered_run(league, tmp_path):
    from tests.test_revalidation import _reset

    _reset(with_schedule=False)
    train, test, everything = league
    bundle = pj.fit_bundle(train, test.head(3000), everything, "stage5-db", n_sims=300)
    day = test[test.game_date == test.game_date.max()]
    proj = pj.project_frame(day, bundle, 500, everything, datetime(2026, 4, 1, tzinfo=UTC))
    with session_scope() as s:
        from nbaprops.pipeline import schedule

        games = day.drop_duplicates("game_id")
        schedule.upsert_games(
            s,
            [
                {
                    "game_id": g.game_id,
                    "season": g.season,
                    "season_type": "regular",
                    "game_date": g.game_date.date(),
                    "tip_time": None,
                    "home_team_id": g.team_id if g.is_home else g.opp_team_id,
                    "away_team_id": g.opp_team_id if g.is_home else g.team_id,
                    "status": "final",
                }
                for g in games.itertuples()
            ],
        )
        pj.register_run(s, bundle)
        n = pj.store_projections(s, bundle.run_id, proj)
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(Projection)) == n == len(proj)
        run = s.get(ModelRun, "stage5-db")
        assert run.status == "challenger" and run.config["kelly_fraction"] == 0.25
        row = s.scalars(select(Projection).limit(1)).one()
        assert isinstance(row.p_over, dict) and row.market in simulate.MARKETS
    with session_scope() as s:  # rerunning the slate updates rather than duplicates
        pj.store_projections(s, bundle.run_id, proj)
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(Projection)) == n
    # Re-projecting one game after a player is ruled out replaces that game's rows: his old
    # projection must not linger (pricing would skip the game for "injury news after it").
    from nbaprops.db.models import PlayerAlias

    out = int(proj["player_id"].iloc[0])
    game = proj.loc[proj["player_id"] == out, "game_id"].iloc[0]
    rest = proj[(proj["game_id"] == game) & (proj["player_id"] != out)]
    with session_scope() as s:
        pj.store_projections(s, bundle.run_id, rest)
    with session_scope() as s:
        left = set(s.scalars(select(Projection.player_id).where(Projection.game_id == game)))
        assert out not in left and left == set(rest["player_id"].astype(int))
        others = s.scalar(
            select(func.count()).select_from(Projection).where(Projection.game_id != game)
        )
        assert others == int((proj["game_id"] != game).sum())  # other games untouched
        assert s.get(PlayerAlias, "player") is None  # no alias from a placeholder name


def test_command_line_evaluate_and_train(league, tmp_path):
    _, _, everything = league
    path = tmp_path / "f.parquet"
    everything.to_parquet(path, index=False)
    pj.main(
        [
            "evaluate",
            "--features",
            str(path),
            "--train",
            "2024-25",
            "--test",
            "2025-26",
            "--sims",
            "300",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    report = json.loads(next((tmp_path / "out").glob("eval-*.json")).read_text())
    assert "markets" in report
    pj.main(
        [
            "train",
            "--features",
            str(path),
            "--train",
            "2024-25",
            "--calibrate",
            "2025-26",
            "--sims",
            "300",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    folder = next((tmp_path / "out").glob("stage5-*"))
    assert {p.name for p in folder.iterdir()} >= {
        "minutes.json",
        "rates.json",
        "scenario.json",
        "calibrator.json",
        "meta.json",
    }


def test_market_lines_move_projections_relative_to_ratings():
    rows = _game_rows()  # ratings: total 225, even game
    args = (rows, _fixed_minutes(10), _manual_rates(), simulate.ScenarioParams(), 40_000)
    base = simulate.simulate_game(*args, seed=9)
    market = simulate.simulate_game(*args, seed=9, total0=255.0, home_margin0=0.0)
    lift = 255 / 225  # each team expected to score ~13% more than the ratings say
    assert market["pts"].mean() / base["pts"].mean() == pytest.approx(lift, rel=0.02)
    assert market["reb"].mean() / base["reb"].mean() == pytest.approx(lift, rel=0.02)
    wild = simulate.simulate_game(*args, seed=9, total0=400.0, home_margin0=0.0)
    assert wild["pts"].mean() / base["pts"].mean() == pytest.approx(1.15, rel=0.02)  # capped
    favored = simulate.simulate_game(*args, seed=9, total0=225.0, home_margin0=10.0)
    home, away = favored["pts"][:5].mean(), favored["pts"][5:].mean()
    assert home > base["pts"][:5].mean() > away  # a home favorite's players project higher


def test_line_grid_reaches_far_rungs():
    draws = RNG.poisson(20, size=(1, 20_000))
    grid = simulate.line_grid(draws)[0]
    assert grid[0] <= np.quantile(draws, 0.01) and grid[-1] >= np.quantile(draws, 0.99)
