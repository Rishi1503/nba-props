import numpy as np
import pandas as pd
import pytest

from nbaprops.features import build as fb
from nbaprops.models import minutes as mm
from tests import feature_fixtures as ff


@pytest.fixture(scope="module")
def league():
    df = fb.training_frame(ff.build(n_regular=82))
    return df[df.season == "2024-25"], df[df.season == "2025-26"], df


@pytest.fixture(scope="module")
def model(league):
    train, _, _ = league
    return mm.fit(train)


# --- distribution math -------------------------------------------------------


def _dist(p_exit=0.0):
    n = 3
    return mm.MinutesDist(
        q50=np.array([30.0, 20.0, 8.0]),
        sd_left=np.array([4.0, 3.0, 2.0]),
        sd_right=np.array([2.0, 3.0, 4.0]),
        p_exit=np.full(n, p_exit),
        exit_cap=np.array([10.0, 7.0, 3.0]),
    )


def test_quantiles_reproduce_the_model_quantiles_without_exits():
    d = _dist()
    assert d.quantile(0.5) == pytest.approx(d.q50, abs=1e-3)
    assert d.quantile(0.1) == pytest.approx(d.q50 - mm.Z90 * d.sd_left, abs=1e-3)
    assert d.quantile(0.9) == pytest.approx(d.q50 + mm.Z90 * d.sd_right, abs=1e-3)


def test_cdf_and_quantile_are_inverse_and_respect_bounds():
    d = _dist(p_exit=0.05)
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        assert d.cdf(d.quantile(p)) == pytest.approx(np.full(3, p), abs=1e-6)
    assert (d.cdf(-1) == 0).all() and (d.cdf(mm.MAX_MINUTES) == 1).all()


def test_mean_matches_monte_carlo():
    d = _dist(p_exit=0.04)
    draws = d.sample(200_000, np.random.default_rng(1))
    assert draws.min() >= 0 and draws.max() <= mm.MAX_MINUTES
    # Analytic mean ignores the tiny clipped mass of the 8-minute player below 0
    assert draws.mean(axis=1)[:2] == pytest.approx(d.mean()[:2], abs=0.05)


def test_exit_mass_sits_below_the_cap():
    d = _dist(p_exit=0.10)
    assert (d.cdf(d.exit_cap) >= 0.10).all()
    assert (d.quantile(0.04) < d.exit_cap).all()


# --- early exits -------------------------------------------------------------


def _hist(minutes, start="2025-11-01"):
    dates = pd.date_range(start, periods=len(minutes), freq="2D")
    return pd.DataFrame(
        {"player_id": 1, "game_date": dates, "y_minutes": minutes, "min_avg_l10": 32.0}
    )


def test_exit_rate_uses_only_earlier_games():
    h = _hist([33, 31, 4, 32, 34])  # an exit in game 3
    rates = mm.exit_rates(h, h, league_rate=0.02)
    prior = 40 * 0.02
    assert rates[0] == pytest.approx(0.02)  # no earlier games: the league rate
    assert rates[1] == pytest.approx(prior / 41)  # one normal game nudges it down
    assert rates[2] == pytest.approx(prior / 42)  # its own exit does not count yet
    assert rates[3] == pytest.approx((1 + prior) / 43)  # counted from the next game on


def test_exit_definition():
    m = pd.Series([4.0, 14.0, 10.0, 4.0])
    avg = pd.Series([32.0, 32.0, 32.0, 12.0])
    assert list(mm.is_early_exit(m, avg)) == [True, False, True, False]


# --- learning ----------------------------------------------------------------


def test_learns_a_real_effect_and_beats_the_baseline():
    """Teammate absences add minutes; the 10-game average can't see that, the model can."""
    rng = np.random.default_rng(3)
    n = 6000
    base = rng.uniform(12, 36, n)
    absent = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({c: np.nan for c in mm.FEATURES}, index=range(n))
    df["min_ewm"] = df["min_avg_l10"] = df["min_avg_l5"] = df["min_avg_l3"] = base
    df["std_min_avg"] = base
    df["absent_starters"] = absent
    df["absent_min_sum"] = absent * 32.0
    for c in ("games_missed", "new_team", "late_season", "clinched_top6", "b2b"):
        df[c] = 0
    df["career_games"] = 200
    df["started_share_l10"] = (base > 26).astype(float)
    df["y_minutes"] = base + 6 * absent + rng.normal(0, 2.0, n)
    df["player_id"] = rng.integers(0, 300, n)
    df["game_date"] = pd.Timestamp("2024-10-22") + pd.to_timedelta(rng.integers(0, 170, n), "D")
    train, test = df.iloc[:4500], df.iloc[4500:]
    report = mm.evaluate(mm.fit(train), test)
    assert report["buckets"]["key_absence"]["improvement"] > 0.5
    assert report["buckets"]["all"]["improvement"] > 0.2
    assert 0.74 <= report["buckets"]["all"]["coverage_80"] <= 0.86


def test_never_much_worse_than_baseline_and_well_calibrated(model, league):
    _, test, everything = league
    report = mm.evaluate(model, test, history=everything)
    allb = report["buckets"]["all"]
    assert allb["improvement"] >= -0.02  # pure noise around a fixed share: ~baseline
    assert 0.75 <= allb["coverage_80"] <= 0.85
    assert report["passes_coverage_target"] in (True, False)
    assert {"starters", "bench", "back_to_back", "key_absence"} <= set(report["buckets"])


def test_quantiles_ordered_and_bounded(model, league):
    _, test, everything = league
    _, s = mm.predict(model, test, history=everything)
    assert (s["q10"] < s["q50"]).all() and (s["q50"] < s["q90"]).all()
    assert s["q90"].max() <= mm.MAX_MINUTES and s["q10"].min() >= 0


# --- rules -------------------------------------------------------------------


def test_low_confidence_rules_flag_and_widen(model, league):
    _, test, everything = league
    row = test.iloc[[200]].copy()
    row["games_missed"], row["new_team"], row["career_games"] = 0, 0, 300
    row["late_season"], row["clinched_top6"] = 0, 0
    _, base = mm.predict(model, row, history=everything)
    assert not base["low_confidence"].iloc[0]
    for col, val in (("games_missed", 4), ("new_team", 1), ("career_games", 2)):
        r = row.copy()
        r[col] = val
        d, s = mm.predict(model, r, history=everything)
        assert s["low_confidence"].iloc[0], col
        width = s["q90"].iloc[0] - s["q10"].iloc[0]
        assert width > base["q90"].iloc[0] - base["q10"].iloc[0]
    starter = row.copy()
    starter["late_season"], starter["clinched_top6"], starter["started_share_l10"] = 1, 1, 0.9
    assert mm.predict(model, starter, history=everything)[1]["low_confidence"].iloc[0]


# --- persistence and determinism ---------------------------------------------


def test_save_load_round_trip(model, league, tmp_path):
    _, test, everything = league
    path = tmp_path / "m.json"
    model.save(path)
    again = mm.MinutesModel.load(path)
    a = mm.predict(model, test, history=everything)[1]
    b = mm.predict(again, test, history=everything)[1]
    pd.testing.assert_frame_equal(a, b)


def test_training_is_deterministic(league):
    train, test, everything = league
    a = mm.predict(mm.fit(train), test.head(300), history=everything)[1]
    b = mm.predict(mm.fit(train), test.head(300), history=everything)[1]
    pd.testing.assert_frame_equal(a, b)
