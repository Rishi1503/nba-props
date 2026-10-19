"""Stage 6: walk-forward backtest, Layer A metrics, holdout guard, comparison, CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from nbaprops.backtest import metrics
from nbaprops.backtest import run as bt
from nbaprops.backtest import walkforward as wf
from nbaprops.features import build as fb
from nbaprops.models import minutes as mm
from tests import feature_fixtures as ff

FAST = {"retrain_days": 14, "n_sims": 300, "first_train_season": "2024-25"}


@pytest.fixture(scope="module")
def frame():
    df = fb.training_frame(ff.build())
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@pytest.fixture(scope="module")
def result(frame):
    return wf.run_season(frame, "2025-26", wf.Settings(**FAST))


# --- walk-forward mechanics ---------------------------------------------------------


def test_blocks_cover_every_date_once_and_skip_empty_windows():
    dates = pd.Series(pd.to_datetime(["2025-10-21", "2025-10-23", "2025-11-20", "2025-11-21"]))
    b = wf.blocks(dates, 7)
    assert b[0][0] == pd.Timestamp("2025-10-21")
    covered = [d for d in dates if any(s <= d < e for s, e in b)]
    assert len(covered) == len(dates)
    assert all(any(s <= d < e for d in dates) for s, e in b)  # no empty windows
    assert all((e - s).days == 7 for s, e in b)


def test_models_only_ever_train_on_earlier_games(result):
    for blk in result.blocks:
        assert pd.Timestamp(blk["trained_through"]) < pd.Timestamp(blk["block"])


def test_later_outcomes_change_no_earlier_prediction(frame, result):
    """The decisive leakage test: rewrite every outcome in the last block and rerun. Every
    earlier prediction must be bit-for-bit identical (models, calibrator, exit history)."""
    last = result.blocks[-1]["block"]
    changed = frame.copy()
    late = changed["game_date"] >= pd.Timestamp(last)
    for c in ("y_minutes", "y_pts", "y_reb", "y_ast", "y_fg3m", "y_pra"):
        changed.loc[late, c] = changed.loc[late, c] * 2 + 7
    again = wf.run_season(changed, "2025-26", wf.Settings(**FAST))
    key = ["row_key", "market", "line"]
    a = result.lines[result.lines["block"] != last].set_index(key).sort_index()
    b = again.lines[again.lines["block"] != last].set_index(key).sort_index()
    pd.testing.assert_frame_equal(a[["p_raw", "p_cal", "p_base"]], b[["p_raw", "p_cal", "p_base"]])
    assert not a.empty


def test_first_block_is_uncalibrated_and_later_blocks_use_only_earlier_lines(result):
    first = result.lines[result.lines["block"] == result.blocks[0]["block"]]
    from nbaprops.models.calibrate import P_MAX, P_MIN

    assert np.allclose(first["p_cal"], first["p_raw"].clip(P_MIN, P_MAX), atol=1e-12)
    assert result.blocks[0]["calibrated_markets"] == []


def test_every_played_game_in_the_season_is_predicted_once(frame, result):
    played = frame[
        (frame["season"] == "2025-26")
        & frame["y_minutes"].notna()
        & (frame["season_type"] != "preseason")
    ]
    pts = result.rows[result.rows["market"] == "pts"]
    assert len(pts) == len(played) and pts["row_key"].is_unique


def test_cache_reuses_models_and_reproduces_results(frame, result, tmp_path):
    s = wf.Settings(**FAST, cache_dir=tmp_path)
    first = wf.run_season(frame, "2025-26", s)
    second = wf.run_season(frame, "2025-26", s)
    assert not any(b["cached"] for b in first.blocks) and all(b["cached"] for b in second.blocks)
    key = ["row_key", "market", "line"]
    pd.testing.assert_frame_equal(
        first.lines.set_index(key).sort_index()[["p_raw", "p_cal"]],
        result.lines.set_index(key).sort_index()[["p_raw", "p_cal"]],
    )
    pd.testing.assert_frame_equal(
        second.lines.set_index(key).sort_index()[["p_raw", "p_cal"]],
        result.lines.set_index(key).sort_index()[["p_raw", "p_cal"]],
    )


def test_cache_key_changes_with_features_and_data(frame):
    train = frame[frame["season"] == "2024-25"]
    k1 = wf._cache_key("2025-11-01", ["a", "b"], wf._fingerprint(train))
    assert k1 == wf._cache_key("2025-11-01", ["b", "a"], wf._fingerprint(train))
    assert k1 != wf._cache_key("2025-11-01", ["a"], wf._fingerprint(train))
    edited = train.copy()
    edited.iloc[0, edited.columns.get_loc("y_pts")] += 1  # a corrected box score
    assert k1 != wf._cache_key("2025-11-01", ["a", "b"], wf._fingerprint(edited))


# --- Layer A metrics ------------------------------------------------------------------


def test_pit_detects_shape_problems():
    rng = np.random.default_rng(1)
    n, sims = 6000, 2000
    truth = rng.poisson(8, n)
    right = rng.poisson(8, (n, sims))
    assert metrics.pit_summary(wf._pit(right, truth, rng))["verdict"].startswith("flat")
    narrow = np.clip(np.round(rng.normal(8, 0.8, (n, sims))), 0, None)  # too confident
    assert "too narrow" in metrics.pit_summary(wf._pit(narrow, truth, rng))["verdict"]
    wide = rng.poisson(8, (n, sims)) + rng.integers(-6, 7, (n, sims))
    assert "too wide" in metrics.pit_summary(wf._pit(wide, truth, rng))["verdict"]
    low = rng.poisson(6.5, (n, sims))
    v = metrics.pit_summary(wf._pit(low, truth, rng))["verdict"]
    assert "exceed" in v and "narrow" not in v  # a shift is not a width problem
    both = np.clip(np.round(rng.normal(6.5, 0.8, (n, sims))), 0, None)
    assert "exceed" in metrics.pit_summary(wf._pit(both, truth, rng))["verdict"]


def test_layer_a_report_is_complete(result):
    rep = metrics.layer_a(result)
    assert set(rep["markets"]) == {"pts", "reb", "ast", "fg3m", "pra"}
    for v in rep["markets"].values():
        assert v["accuracy"]["n"] > 0 and v["pit"]["n"] > 0
        for k in (
            "brier_raw",
            "brier_calibrated",
            "log_loss_raw",
            "log_loss_calibrated",
            "ece_calibrated",
            "log_loss_baseline",
            "reliability",
            "calibration_failures",
        ):
            assert k in v["probabilities"]
        assert set(v["breakdowns"]) == {"month", "low_confidence", "roster_change", "postseason"}
    assert set(rep["targets"]["markets"]) == set(rep["markets"])
    assert 0.6 < rep["minutes"]["coverage_80"] < 0.95


def test_log_loss_known_values():
    assert metrics.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert metrics.log_loss([1.0], [1]) < 1e-3  # clipped, not zero-division


def test_stage5_targets_fail_when_any_market_fails():
    good = {
        "accuracy": {"mae_improvement": 0.02},
        "probabilities": {"ece_calibrated": 0.01, "calibration_failures": []},
    }
    bad = {
        "accuracy": {"mae_improvement": -0.01},
        "probabilities": {"ece_calibrated": 0.01, "calibration_failures": []},
    }
    assert metrics.stage5_targets({"markets": {"pts": good}})["passes"]
    out = metrics.stage5_targets({"markets": {"pts": good, "reb": bad}})
    assert not out["passes"] and not out["markets"]["reb"]["beats_baseline_mae"]


# --- comparison ---------------------------------------------------------------------


def test_compare_identical_runs_finds_no_difference(result):
    out = metrics.compare(result, result)
    for v in out["markets"].values():
        assert v["log_loss"]["mean_diff"] == 0 and "no significant" in v["log_loss"]["verdict"]


def test_compare_detects_a_better_run(result):
    better = wf.Result(
        result.season, result.lines.copy(), result.rows.copy(), result.minutes, result.blocks
    )
    hit = better.lines["hit"].to_numpy()
    better.lines["p_cal"] = 0.8 * better.lines["p_cal"] + 0.2 * hit  # leans toward truth
    better.rows["median"] = 0.5 * better.rows["median"] + 0.5 * better.rows["y"]
    out = metrics.compare(result, better, "old", "new")
    for v in out["markets"].values():
        assert v["log_loss"]["verdict"] == "new better" and v["mae"]["verdict"] == "new better"


def test_models_without_the_roster_group_never_see_it(frame):
    drop = set(fb.FEATURE_GROUPS["roster"])
    feats = [c for c in fb.FEATURE_COLUMNS if c not in drop]
    train = frame[frame["season"] == "2024-25"]
    m = mm.fit(train, features=[f for f in mm.FEATURES if f in feats])
    assert not drop & set(m.features)
    traded = frame[
        (frame["player_id"] == ff.TRADED)
        & (frame["stint_change"] == 1)
        & (frame["stint_games"] > 0)
    ]
    assert np.allclose(
        mm.anchor(traded, use_stint=False),
        traded["min_ewm"].fillna(traded["std_min_avg"]).to_numpy(),
    )
    assert not np.allclose(mm.anchor(traded), mm.anchor(traded, use_stint=False))


# --- holdout guard and command line ----------------------------------------------------


def test_holdout_can_be_used_once(tmp_path):
    ledger = tmp_path / "ledger.json"
    assert bt.check_holdout(bt.HOLDOUT_SEASON, None, ledger)
    bt.record_holdout(bt.HOLDOUT_SEASON, "bt-1", True, None, True, ledger)
    with pytest.raises(bt.HoldoutSpent):
        bt.check_holdout(bt.HOLDOUT_SEASON, None, ledger)
    assert bt.check_holdout(bt.HOLDOUT_SEASON, "fixed a data bug", ledger) is False
    assert bt.check_holdout("2024-25", None, ledger)  # tuning seasons are unrestricted
    bt.record_holdout("2024-25", "bt-2", True, None, True, ledger)
    assert [e["run_id"] for e in json.loads(ledger.read_text())] == ["bt-1"]


def test_command_line_end_to_end(frame, tmp_path):
    parquet = tmp_path / "train.parquet"
    frame.to_parquet(parquet)
    common = [
        "--season",
        bt.HOLDOUT_SEASON,
        "--features",
        str(parquet),
        "--retrain-days",
        "14",
        "--n-sims",
        "300",
        "--first-train-season",
        "2024-25",
        "--cache",
        str(tmp_path / "cache"),
        "--reports",
        str(tmp_path / "reports"),
    ]
    rep = bt.main(common)
    assert rep["holdout"] == "clean"
    assert (tmp_path / "reports" / "backtest" / f"{rep['run_id']}.md").exists()
    with pytest.raises(bt.HoldoutSpent):
        bt.main(common)
    again = bt.main([*common, "--rerun-holdout", "test of the guard", "--ablate", "roster"])
    assert again["holdout"].startswith("NOT CLEAN")
    assert set(again["ablation"]["markets"]) == {"pts", "reb", "ast", "fg3m", "pra"}
    md = (tmp_path / "reports" / "backtest" / f"{again['run_id']}.md").read_text()
    assert "With vs without the roster inputs" in md and "NOT CLEAN" in md
    ledger = json.loads((tmp_path / "reports" / "holdout_ledger.json").read_text())
    assert [e["clean"] for e in ledger] == [True, False]


def test_production_needs_a_passing_holdout(frame, tmp_path):
    parquet = tmp_path / "train.parquet"
    frame.to_parquet(parquet)
    with pytest.raises(SystemExit):
        bt.main(
            [
                "--season",
                "2024-25",
                "--features",
                str(parquet),
                "--retrain-days",
                "14",
                "--n-sims",
                "300",
                "--first-train-season",
                "2024-25",
                "--production",
                "--reports",
                str(tmp_path / "r"),
                "--cache",
                str(tmp_path / "c"),
            ]
        )


def test_production_bundle_calibrates_on_out_of_sample_lines(frame, result):
    b = bt.production_bundle(frame, result.lines, "prod-test", "2024-25")
    assert b.meta["train_seasons"] == ["2024-25", "2025-26"]
    assert b.minutes.trained_through == str(frame["game_date"].max().date())
    assert b.meta["calibration"] == "walk-forward out-of-sample predictions"


# --- pricing DraftKings' exact lines inside the walk-forward (Layer B input) -------------


def test_exact_line_prices_agree_with_the_walk_forwards_own_test_lines(frame, result):
    tl = result.lines[result.lines["market"] == "pts"].copy()
    tl[["game_id", "player_id"]] = tl["row_key"].str.split(":", expand=True)
    tl["player_id"] = tl["player_id"].astype("int64")
    sample = tl[tl["row_key"].isin(tl["row_key"].drop_duplicates().iloc[:40])]
    half = sample[["game_id", "player_id", "market", "line"]]
    whole = half.assign(line=np.floor(half["line"])).drop_duplicates()  # x.0 lines
    s = wf.Settings(**FAST, price_lines=pd.concat([half, whole], ignore_index=True))
    again = wf.run_season(frame, "2025-26", s)
    pr = again.priced.merge(
        sample[["game_id", "player_id", "line", "p_cal"]],
        on=["game_id", "player_id", "line"],
        how="inner",
    )
    assert len(pr) == len(sample)
    assert np.allclose(pr["p_over"], pr["p_cal"]) and (pr["p_push"] == 0).all()
    w = again.priced[np.isclose(again.priced["line"] % 1, 0)]
    assert len(w) and (w["p_push"] >= 0).all()
    assert np.allclose(w["p_over"] + w["p_under"] + w["p_push"], 1.0)
    # over a whole line L = over L + 0.5; push = P(over L - 0.5) - P(over L + 0.5)
    look = sample.set_index(["game_id", "player_id", "line"])["p_cal"]
    checked = 0
    for r in w.itertuples():
        up, down = (r.game_id, r.player_id, r.line + 0.5), (r.game_id, r.player_id, r.line - 0.5)
        if up in look.index and down in look.index:
            assert r.p_over == pytest.approx(look[up])
            assert r.p_push == pytest.approx(max(look[down] - look[up], 0.0))
            checked += 1
    assert checked > 10


def _synthetic_market(frame):
    """DraftKings and two consensus books on the first ten dates of 2025-26: consensus prices
    the season-to-date average, DraftKings adds noise. Mechanics only, not edge."""
    from scipy.stats import norm

    from nbaprops.pricing.odds_math import to_american

    rng = np.random.default_rng(3)
    t = frame[(frame["season"] == "2025-26") & frame["y_minutes"].notna()]
    dates = sorted(t["game_date"].unique())[:10]
    t = t[t["game_date"].isin(dates) & t["std_pts_avg"].notna()]
    games = (
        t.groupby("game_id")["game_date"]
        .first()
        .reset_index()
        .assign(
            tip_time=lambda d: (
                pd.to_datetime(d["game_date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=23)
            ),
            status="final",
        )
    )
    rows = []
    for r in t.itertuples():
        mean = r.std_pts_avg
        line = np.floor(mean) + 0.5
        p = float(np.clip(1 - norm.cdf(line, mean, max(np.sqrt(mean) * 1.3, 1)), 0.1, 0.9))
        tip = pd.Timestamp(r.game_date).tz_localize("UTC") + pd.Timedelta(hours=23)
        for book, q in (
            ("fanduel", p),
            ("bovada", p),
            ("draftkings", float(np.clip(p + rng.normal(0, 0.06), 0.08, 0.92))),
        ):
            for ts, close in (
                (tip - pd.Timedelta(hours=10), False),
                (tip - pd.Timedelta(minutes=20), True),
            ):
                rows.append(
                    {
                        "id": len(rows),
                        "snapshot_ts": ts,
                        "book": book,
                        "game_id": r.game_id,
                        "player_id": int(r.player_id),
                        "market": "pts",
                        "line_type": "main",
                        "line": line,
                        "over_price": int(round(to_american(1 / (q * 1.045)))),
                        "under_price": int(round(to_american(1 / ((1 - q) * 1.045)))),
                        "is_withdrawn": False,
                        "is_close": close,
                    }
                )
    outcomes = t.assign(
        minutes=t["y_minutes"],
        dnp=False,
        pts=t["y_pts"],
        reb=t["y_reb"],
        ast=t["y_ast"],
        fg3m=t["y_fg3m"],
    )[["game_id", "player_id", "minutes", "dnp", "pts", "reb", "ast", "fg3m"]]
    return pd.DataFrame(rows), games, outcomes


def test_layer_b_end_to_end_on_a_synthetic_market(frame):
    from nbaprops.backtest import layer_b
    from nbaprops.pricing.engine import Rules

    odds, games, outcomes = _synthetic_market(frame)
    lines = layer_b.lines_to_price(odds, games)
    assert len(lines) and lines["line"].notna().all()
    res = wf.run_season(frame, "2025-26", wf.Settings(**FAST, price_lines=lines))
    rules = Rules()
    out = layer_b.section(res.priced, odds, games, outcomes, rules)
    bets = out["bets"]
    assert bets["candidates"] > 0 and bets["flagged"] > 0
    _, picks = layer_b.replay(res.priced, odds, games, outcomes, rules)
    assert (picks["stake"] <= rules.max_bet_pct + 1e-12).all()
    assert (picks.groupby("game_id")["stake"].sum() <= rules.max_game_pct + 1e-9).all()
    assert (picks.groupby("game_date")["stake"].sum() <= rules.max_daily_pct + 1e-9).all()
    assert not picks.duplicated(["game_id", "player_id"]).any()
    assert (picks["ev"] >= rules.ev_min - 1e-12).all() and picks["agree"].all()
    settled = picks[picks["result"] != "void"]
    assert bets["overall"]["profit"] == pytest.approx(settled["profit"].sum(), abs=1e-4)
    assert out["blend"]["rows"] > 0 and out["blend"]["second_half"]


@pytest.mark.db
def test_command_line_layer_b_without_logged_odds_says_so(frame, tmp_path):
    from tests.test_revalidation import _reset

    _reset(with_schedule=True)  # no 2025-26 odds in the database
    parquet = tmp_path / "train.parquet"
    frame.to_parquet(parquet)
    rep = bt.main(
        [
            "--season",
            "2024-25",
            "--features",
            str(parquet),
            "--retrain-days",
            "14",
            "--n-sims",
            "300",
            "--first-train-season",
            "2024-25",
            "--layer-b",
            "--reports",
            str(tmp_path / "r"),
            "--cache",
            str(tmp_path / "c"),
        ]
    )
    assert "skipped" in rep["layer_b"]
    md = next((tmp_path / "r" / "backtest").glob("*.md")).read_text()
    assert "Betting replay (Layer B)" in md and "no logged odds" in md


def test_production_refuses_a_reused_holdout(frame, tmp_path, monkeypatch):
    parquet = tmp_path / "train.parquet"
    frame.to_parquet(parquet)
    ledger = tmp_path / "r" / "holdout_ledger.json"
    bt.record_holdout(bt.HOLDOUT_SEASON, "earlier", True, None, True, ledger)
    monkeypatch.setattr(
        bt.metrics,
        "stage5_targets",
        lambda rep: {"markets": {m: {"passes": True} for m in rep["markets"]}, "passes": True},
    )
    with pytest.raises(SystemExit, match="clean"):
        bt.main(
            [
                "--season",
                bt.HOLDOUT_SEASON,
                "--features",
                str(parquet),
                "--retrain-days",
                "14",
                "--n-sims",
                "300",
                "--first-train-season",
                "2024-25",
                "--production",
                "--rerun-holdout",
                "looking again",
                "--reports",
                str(tmp_path / "r"),
                "--cache",
                str(tmp_path / "c"),
            ]
        )


def test_cache_key_changes_when_model_code_changes(monkeypatch):
    k1 = wf._cache_key("2025-11-01", ["a"], "fp")
    monkeypatch.setattr(wf, "code_version", lambda: "edited")
    assert wf._cache_key("2025-11-01", ["a"], "fp") != k1
    monkeypatch.undo()
    assert len(wf.code_version()) == 12 and wf._cache_key("2025-11-01", ["a"], "fp") == k1


def test_walk_forward_grids_match_its_own_calibrated_lines(frame, result):
    tl = result.lines[result.lines["market"] == "pts"].copy()
    tl[["game_id", "player_id"]] = tl["row_key"].str.split(":", expand=True)
    tl["player_id"] = tl["player_id"].astype("int64")
    keys = tl[["game_id", "player_id"]].drop_duplicates().iloc[:30].assign(market="pts")
    again = wf.run_season(frame, "2025-26", wf.Settings(**FAST, grid_keys=keys))
    assert len(again.grids) == len(keys)
    checked = 0
    for g in again.grids.itertuples():
        ps = [g.p_over[k] for k in sorted(g.p_over, key=float)]
        assert all(a >= b for a, b in zip(ps, ps[1:], strict=False))  # decreasing in the line
        mine = tl[(tl["game_id"] == g.game_id) & (tl["player_id"] == g.player_id)]
        for r in mine.itertuples():
            key = f"{r.line:.1f}"
            if key in g.p_over:
                assert g.p_over[key] == pytest.approx(r.p_cal, abs=6e-5)  # stored to 4 dp
                checked += 1
    assert checked > 50


def test_layer_b_prices_milestones_and_reports_parlays_and_growth(frame):
    from nbaprops.backtest import layer_b
    from nbaprops.pricing.engine import Rules

    odds, games, outcomes = _synthetic_market(frame)
    dk_main = odds[(odds["book"] == "draftkings")]
    ms = dk_main.assign(
        line_type="milestone",
        line=dk_main["line"] + 3.0,
        over_price=260,
        under_price=None,
        id=dk_main["id"] + 10**6,
    )
    odds = pd.concat([odds, ms], ignore_index=True)
    s = wf.Settings(
        **FAST,
        price_lines=layer_b.lines_to_price(odds, games),
        grid_keys=layer_b.milestone_keys(odds, games),
    )
    res = wf.run_season(frame, "2025-26", s)
    assert len(res.grids) > 0
    out = layer_b.section(res.priced, odds, games, outcomes, Rules(), grids=res.grids)
    cands, _ = layer_b.replay(res.priced, odds, games, outcomes, Rules(), grids=res.grids)
    priced_ms = cands[cands["line_type"] == "milestone"]
    assert len(priced_ms) > 0  # milestones went through the same builder as live pricing
    assert priced_ms["has_consensus"].all()  # anchored on the consensus main line
    assert set(out) >= {"bets", "blend", "satellites", "growth", "by_line_type"}
    assert out["growth"]["singles_only"]["days"] > 0
