"""Stage 9: weekly review, market shut-off, go-live checklist, paper-data tuning, live
calibration, the candidate log, real bets and the ramp."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import BankrollEntry, Game, MarketControl, OpsRun, Pick
from nbaprops.ops import golive, livecal, review, tune
from nbaprops.ops.place import AlreadyPlaced, place, ramp
from nbaprops.pricing import engine
from tests.test_slate import CFG, DAY, NOW, seed

START = date(2026, 11, 2)  # a Monday
TODAY = START + timedelta(days=35)  # Monday Dec 7: four complete weeks behind it
CLV = {"pts": 0.03, "reb": -0.01}


def history(per_day: int = 10, days: int = 35) -> None:
    """Settled paper picks: fair 60%, wins exactly 3 in 5, all at -150 (ROI exactly 0)."""
    seed()
    with session_scope() as s:
        players = (
            s.execute(
                text("SELECT player_id FROM players ORDER BY player_id LIMIT :n"), {"n": per_day}
            )
            .scalars()
            .all()
        )
        k = 0
        for d in range(days):
            day = START + timedelta(days=d)
            gid = f"09{d:08d}"
            s.add(
                Game(
                    game_id=gid,
                    season="2026-27",
                    season_type="regular",
                    game_date=day,
                    tip_time=datetime(day.year, day.month, day.day, 23, 0, tzinfo=UTC),
                    home_team_id=1610612765,
                    away_team_id=1610612738,
                    status="final",
                )
            )
            s.flush()
            for i, pid in enumerate(players):
                market = "pts" if i % 2 == 0 else "reb"
                win = k % 5 < 3
                # 10 AM and 3 PM New York: EST (UTC-5), since the clocks changed on Nov 1
                hour = 15 if i < per_day // 2 else 20
                s.add(
                    Pick(
                        run_id="r1",
                        game_id=gid,
                        player_id=pid,
                        market=market,
                        line=Decimal("20.5"),
                        side="over",
                        tier="high_conf",
                        mode="paper",
                        flagged_ts=datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC),
                        dk_price=-150,
                        fair_prob=Decimal("0.6"),
                        ev=Decimal("0.06"),
                        stake=Decimal("10"),
                        result="win" if win else "loss",
                        pnl=Decimal("6.67") if win else Decimal("-10"),
                        clv=Decimal(str(CLV[market])),
                    )
                )
                k += 1
        for d in range(1, 15):
            t = datetime.combine(TODAY - timedelta(days=d), datetime.min.time(), UTC) + timedelta(
                hours=18
            )
            s.add(OpsRun(kind="cycle", started_ts=t, finished_ts=t, ok=True, actions=[], errors=[]))


# --- weekly review and the market rule -----------------------------------------------------


@pytest.mark.db
def test_weekly_review_numbers_and_splits():
    history()
    with session_scope() as s:
        rep = review.weekly_review(s, TODAY)
    season = rep["season"]
    assert season["bets"] == 350 and season["staked_bets"] == 350
    assert season["hit_rate"] == pytest.approx(0.6) and season["breakeven"] == pytest.approx(0.6)
    assert season["avg_clv"] == pytest.approx(0.01)
    by = rep["season_splits"]["by_market"]
    assert by["pts"]["avg_clv"] == pytest.approx(0.03) and by["reb"]["avg_clv"] == pytest.approx(
        -0.01
    )
    assert set(rep["season_splits"]["by_flag_time"]) == {"10 AM", "3 PM"}
    assert [w["avg_clv"] for w in rep["weekly_clv"]] == [pytest.approx(0.01)] * 4
    assert rep["market_rule"] == [{"market": "reb", "picks": 175, "avg_clv": -0.01}]
    assert "Switched off" in review.render(rep)


@pytest.mark.db
def test_market_rule_switches_off_once_and_pricing_respects_it():
    history()
    with session_scope() as s:
        rep = review.weekly_review(s, TODAY)
        assert review.apply_market_rule(s, rep) == ["reb"]
        assert review.apply_market_rule(s, rep) == []  # already off
    seed()  # the Stage 7 slate: every prop is points
    with session_scope() as s:
        s.add(MarketControl(market="pts", disabled=True, reason="test", changed_ts=NOW))
    from nbaprops.pricing import slate

    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, write=False)
    assert res.picks.empty and any("switched off: pts" in m for m in res.skipped)


# --- go-live -----------------------------------------------------------------------------


@pytest.mark.db
def test_go_live_ready_when_every_criterion_passes():
    history()
    with session_scope() as s:
        result = golive.checklist(s, TODAY)
    assert result["ready"], golive.render(result)
    assert "READY" in golive.render(result)


@pytest.mark.db
@pytest.mark.parametrize("breaker", ["few_picks", "manual_fix", "failed_cycle", "deposit"])
def test_go_live_blocks_on_each_failure(breaker):
    history(per_day=4) if breaker == "few_picks" else history()
    with session_scope() as s:
        y = datetime.combine(TODAY - timedelta(days=1), datetime.min.time(), UTC) + timedelta(
            hours=20
        )
        if breaker == "manual_fix":
            s.add(
                OpsRun(
                    kind="manual_fix",
                    started_ts=y,
                    finished_ts=y,
                    ok=False,
                    actions=[],
                    errors=[],
                    note="fixed a parser by hand",
                )
            )
        if breaker == "failed_cycle":
            t = y - timedelta(days=2)
            s.add(
                OpsRun(
                    kind="cycle",
                    started_ts=t,
                    finished_ts=t,
                    ok=False,
                    actions=[],
                    errors=["odds: down"],
                )
            )
        if breaker == "deposit":
            s.add(
                BankrollEntry(entry_date=START, deposits=Decimal("500"), bankroll=Decimal("1500"))
            )
    with session_scope() as s:
        result = golive.checklist(s, TODAY)
    failed = [c["name"] for c in result["criteria"] if not c["ok"]]
    assert not result["ready"]
    expect = {
        "few_picks": "300+",
        "manual_fix": "Pipeline",
        "failed_cycle": "Pipeline",
        "deposit": "Bankroll",
    }[breaker]
    assert len(failed) == 1 and failed[0].startswith(expect), failed


# --- candidate log: what was logged replays to the same decision ---------------------------


@pytest.mark.db
def test_logged_candidates_replay_to_the_live_decision():
    from nbaprops.pricing import slate

    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG)
    with session_scope() as s:
        log = pd.read_sql_query(text("SELECT * FROM candidate_log"), s.connection())
    assert 0 < len(log) <= res.candidates  # main lines and milestones near the threshold
    kept = log["flag"] | (
        log["line_type"].isin(["main", "milestone"]) & (log["ev"] >= CFG.candidate_log_min_ev)
    )
    assert kept.all()  # every logged row meets the storage rule
    two = log[log["kind"] == "two_way"].reset_index(drop=True)
    inputs = pd.json_normalize(two["inputs"])
    assert set(engine.DECISION_COLUMNS) <= set(inputs.columns)
    again = engine.decide(
        pd.concat([two[engine.KEY[:-1]], two[["line"]], inputs], axis=1),
        engine.Rules.from_config(CFG),
    )
    assert list(again["flag"]) == list(two["flag"])
    assert np.allclose(again["ev"], two["ev"])


# --- tuning on paper data ------------------------------------------------------------------


def logged_frame(n_days=12, per_day=40, seed_=0):
    """Synthetic logged candidates: DraftKings +100/-120 (no-vig over 0.478), a 50/50
    consensus, model probabilities spread around the market; outcomes drawn from the model."""
    rng = np.random.default_rng(seed_)
    rows = []
    for d in range(n_days):
        day = START + timedelta(days=d)
        for i in range(per_day):
            p = float(np.clip(rng.normal(0.55, 0.08), 0.2, 0.9))
            rows.append(
                {
                    "game_id": f"g{d}",
                    "player_id": i,
                    "market": "pts",
                    "line": 20.5,
                    "game_date": day,
                    "logged_ts": datetime(day.year, day.month, day.day, 14, tzinfo=UTC),
                    "p_over": p,
                    "p_under": 1 - p,
                    "p_push": 0.0,
                    "p_over_cons": 0.5,
                    "p_over_novig": 0.5 / (0.5 + 120 / 220),
                    "p_under_novig": (120 / 220) / (0.5 + 120 / 220),
                    "over_price": 100,
                    "under_price": -120,
                    "low_confidence": False,
                    "actual": 25.0 if rng.random() < p else 15.0,
                }
            )
    df = pd.DataFrame(rows)
    closes = pd.Series(0.5, index=pd.MultiIndex.from_frame(df[engine.KEY]))
    return df, closes


def test_a_stricter_ev_floor_only_removes_picks():
    from dataclasses import replace

    df, _ = logged_frame()
    loose = tune.replay(df, replace(engine.Rules(), ev_min=0.02))
    strict = tune.replay(df, replace(engine.Rules(), ev_min=0.06))
    key = ["game_id", "player_id"]
    assert set(map(tuple, strict[key].values)) <= set(map(tuple, loose[key].values))
    assert len(strict) < len(loose)


def test_tuning_chooses_early_and_judges_late():
    df, closes = logged_frame()
    out = tune.tune_logged(df, closes, engine.Rules())
    assert len(out["grid_first_60pct"]) == len(tune.WEIGHTS) * len(tune.EV_MINS)
    assert "chosen" in out and "later_40pct" in out["chosen"]
    assert out["current"]["model_weight"] == 0.3
    late_dates = sorted(df["game_date"].unique())[int(len(df["game_date"].unique()) * 0.6) :]
    assert out["split_at"] == str(late_dates[0])
    assert out["blend"]["rows"] > 0 and out["blend"]["chosen_weight"] is not None


# --- live calibration ---------------------------------------------------------------------


def test_live_curves_compose_with_the_previous_one():
    new = ([0.0, 0.5, 1.0], [0.0, 0.4, 1.0])
    first = livecal.compose(new, None)
    assert np.interp(0.5, first["x"], first["y"]) == pytest.approx(0.4)
    second = livecal.compose(new, first)  # new(old(x)): 0.5 -> 0.4 -> 0.32
    assert np.interp(0.5, second["x"], second["y"]) == pytest.approx(0.32)


def test_live_calibration_keeps_scenario_blends_exact():
    from nbaprops.pricing.slate import apply_live_calibration

    grids = pd.DataFrame(
        [
            {
                "market": "pts",
                "p_over": {"20.5": 0.6},
                "scenarios": {
                    "player_id": 1,
                    "p_play": 0.25,
                    "in": {"20.5": 0.4},
                    "out": {"20.5": 0.8},
                },
            },
            {"market": "reb", "p_over": {"8.5": 0.5}, "scenarios": None},
        ]
    )
    out = apply_live_calibration(grids, {"pts": {"x": [0, 1], "y": [0, 0.5]}})
    sc = out.loc[0, "scenarios"]
    assert sc["in"]["20.5"] == pytest.approx(0.2) and sc["out"]["20.5"] == pytest.approx(0.4)
    assert out.loc[0, "p_over"]["20.5"] == pytest.approx(0.25 * 0.2 + 0.75 * 0.4)
    assert out.loc[1, "p_over"] == {"8.5": 0.5}  # no curve for rebounds: unchanged


# --- real bets and the ramp ---------------------------------------------------------------


@pytest.mark.db
def test_placing_a_real_bet_and_the_ramp():
    history(per_day=2, days=3)
    with session_scope() as s:
        paper = s.scalars(select(Pick).where(Pick.mode == "paper")).first()
        real = place(s, paper.pick_id, -140, 7.5, NOW)
        assert real.mode == "real" and real.price_placed == -140 and float(real.stake) == 7.5
        with pytest.raises(AlreadyPlaced):
            place(s, paper.pick_id, -140, 7.5, NOW)
        assert ramp(s, 200, 0.5)[0] == 0.5  # 1 of 200 real bets
        assert ramp(s, 1, 0.5)[0] == 0.5  # past the ramp, but no CLV yet: held
        real.clv = Decimal("0.02")
        s.flush()
        assert ramp(s, 1, 0.5) == (1.0, "full stakes: real CLV +2.00% after 1 bets")


def test_pick_alerts_show_the_ramped_real_amount():
    from nbaprops.ops.cycle import format_picks

    picks = pd.DataFrame(
        [
            {
                "player_id": 1,
                "market": "pts",
                "side": "over",
                "line": 20.5,
                "dk_price": -150,
                "fair_prob": 0.66,
                "ev": 0.1,
                "stake": 20.0,
                "tier": "high_conf",
            }
        ]
    )
    msg = format_picks(picks, "10:00 AM picks", 0.5, "ramp: 3 of 200 real bets placed")
    assert "stake 20.00 (bet 10.00)" in msg and "ramp: 3 of 200" in msg


@pytest.mark.db
def test_every_cycle_is_recorded():
    from nbaprops.ops import alerts, cycle
    from tests.test_ops_cycle import QUIET, Fake

    seed()
    cycle.run_cycle(session_scope, QUIET, Fake().deps(), alerts.RecordingNotifier())
    cycle.run_cycle(session_scope, QUIET, Fake(fail={"odds"}).deps(), alerts.RecordingNotifier())
    with session_scope() as s:
        runs = s.execute(text("SELECT ok, errors FROM ops_runs ORDER BY id")).all()
    assert [r.ok for r in runs] == [True, False] and runs[1].errors == ["odds: odds broke"]


@pytest.mark.db
def test_the_review_splits_unders_on_role_changes_and_rising_lines():
    """Unders on players whose key teammate was out (and whose line rose since the morning)
    are reported on their own, so CLV can say whether the market knows better there."""
    import json

    history()
    with session_scope() as s:
        ids = s.execute(text("SELECT pick_id FROM picks ORDER BY pick_id LIMIT 40")).scalars().all()
        s.execute(
            text("""UPDATE picks SET side = 'under', context = CAST(:c AS jsonb)
                    WHERE pick_id = ANY(:i)"""),
            {"c": json.dumps({"key_out": True}), "i": ids},
        )
        s.execute(
            text("""
            INSERT INTO odds_snapshots (snapshot_ts, provider, book, game_id, player_id,
                player_name_raw, market, line_type, line, over_price, under_price, is_close,
                is_withdrawn)
            SELECT g.tip_time - interval '10 hours', 'test', 'draftkings', p.game_id,
                   p.player_id, 'x', p.market, 'main', p.line - 2, -110, -110, false, false
            FROM picks p JOIN games g USING (game_id) WHERE p.pick_id = ANY(:i)"""),
            {"i": ids},
        )
    with session_scope() as s:
        rep = review.weekly_review(s, TODAY)
    sr = rep["season_splits"]
    assert sr["by_side_role"]["under, key teammate out"]["bets"] == 40
    assert sr["by_side_role"]["over, untagged"]["bets"] == 310
    assert sr["by_side_line_move"]["under, line rose 1+"]["bets"] == 40
    assert "under, key teammate out" in review.render(rep)
