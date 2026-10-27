"""Layer B: betting replay, grading, CLV and the blend-weight check."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nbaprops.backtest import layer_b
from nbaprops.pricing import engine
from nbaprops.pricing.odds_math import to_american

G, A, B = "0022600010", 101, 102
TIP = datetime(2026, 10, 21, 23, 0, tzinfo=UTC)  # 7:00 PM New York (EDT)
GAMES = pd.DataFrame([{"game_id": G, "game_date": date(2026, 10, 21), "tip_time": TIP}])


def odds_frame(specs):
    """(time, book, player, line, over, under[, line_type, withdrawn])"""
    out = []
    for i, (ts, book, pid, line, over, under, *rest) in enumerate(specs):
        out.append(
            {
                "id": i,
                "snapshot_ts": ts,
                "book": book,
                "game_id": G,
                "player_id": pid,
                "market": "pts",
                "line_type": rest[0] if rest else "main",
                "line": line,
                "over_price": over,
                "under_price": under,
                "is_withdrawn": rest[1] if len(rest) > 1 else False,
                "is_close": False,
            }
        )
    return pd.DataFrame(out)


MORNING = datetime(2026, 10, 21, 13, 0, tzinfo=UTC)
LATE = TIP - timedelta(minutes=20)
ODDS = odds_frame(
    [
        (MORNING, "draftkings", A, 24.5, 100, -120),
        (MORNING, "fanduel", A, 24.5, -110, -110),
        (MORNING, "bovada", A, 24.5, -110, -110),
        (MORNING, "draftkings", A, 29.5, 150, -180, "alternate"),
        (MORNING + timedelta(minutes=30), "draftkings", A, 29.5, None, None, "alternate", True),
        (MORNING, "draftkings", B, 10.5, -110, -110),
        (MORNING, "fanduel", B, 10.5, -110, -110),
        (MORNING, "bovada", B, 10.5, -110, -110),
        # the market moves toward the over on A before tip
        (LATE, "fanduel", A, 24.5, -150, 120),
        (LATE, "bovada", A, 24.5, -150, 120),
        (LATE, "draftkings", A, 24.5, -130, 110),
    ]
)
MODEL = pd.DataFrame(
    [
        {
            "game_id": G,
            "player_id": A,
            "market": "pts",
            "line": 24.5,
            "p_over": 0.70,
            "p_under": 0.30,
            "p_push": 0.0,
            "low_confidence": False,
        },
        {
            "game_id": G,
            "player_id": A,
            "market": "pts",
            "line": 29.5,
            "p_over": 0.60,
            "p_under": 0.40,
            "p_push": 0.0,
            "low_confidence": False,
        },  # pulled before flag time
        {
            "game_id": G,
            "player_id": B,
            "market": "pts",
            "line": 10.5,
            "p_over": 0.55,
            "p_under": 0.45,
            "p_push": 0.0,
            "low_confidence": False,
        },
    ]
)


def outcomes(a_pts=30, a_dnp=False):
    return pd.DataFrame(
        [
            {
                "game_id": G,
                "player_id": A,
                "minutes": 0 if a_dnp else 35,
                "dnp": a_dnp,
                "pts": 0 if a_dnp else a_pts,
                "reb": 5,
                "ast": 4,
                "fg3m": 2,
            },
            {
                "game_id": G,
                "player_id": B,
                "minutes": 30,
                "dnp": False,
                "pts": 8,
                "reb": 9,
                "ast": 1,
                "fg3m": 0,
            },
        ]
    )


def test_replay_matches_hand_calculation():
    cands, picks = layer_b.replay(MODEL, ODDS, GAMES, outcomes(), engine.Rules())
    assert set(zip(cands["player_id"], cands["line"], strict=True)) == {(A, 24.5), (B, 10.5)}
    assert len(picks) == 1  # B: fair 0.515 at -110 is -EV; the pulled alternate isn't priced
    p = picks.iloc[0]
    assert (p["player_id"], p["side"], p["price"]) == (A, "over", 100)
    assert p["ev"] == pytest.approx(0.12) and p["stake"] == pytest.approx(0.02)
    cons_close = 0.6 / (0.6 + 100 / 220)  # FanDuel and Bovada both -150/+120 at the close
    assert p["p_close_fair"] == pytest.approx(cons_close)
    assert p["clv"] == pytest.approx(cons_close - 0.5)  # took +100 (implied 0.5)
    assert p["result"] == "win" and p["profit"] == pytest.approx(0.02)


def test_a_player_who_doesnt_play_is_void():
    _, picks = layer_b.replay(MODEL, ODDS, GAMES, outcomes(a_dnp=True), engine.Rules())
    assert picks.iloc[0]["result"] == "void" and picks.iloc[0]["profit"] == 0
    s = layer_b.summarize(picks)
    assert s["bets"] == 0 and s["voided"] == 1


def test_grading_push_loss_and_combined_stats():
    picks = pd.DataFrame(
        [
            {
                "game_id": G,
                "player_id": A,
                "market": "pts",
                "line": 30.0,
                "side": "over",
                "stake": 1.0,
                "decimal": 2.0,
            },  # 30 on 30: push
            {
                "game_id": G,
                "player_id": A,
                "market": "pts",
                "line": 24.5,
                "side": "under",
                "stake": 1.0,
                "decimal": 1.8,
            },  # loss
            {
                "game_id": G,
                "player_id": A,
                "market": "pra",
                "line": 38.5,
                "side": "over",
                "stake": 1.0,
                "decimal": 1.9,
            },  # 30 + 5 + 4 = 39: win
        ]
    )
    g = layer_b.grade(picks, outcomes())
    assert list(g["result"]) == ["push", "loss", "win"]
    assert list(g["profit"]) == pytest.approx([0.0, -1.0, 0.9])
    assert g["actual"].iloc[2] == 39


def test_flag_time_is_ten_am_new_york_through_daylight_saving_and_before_tip():
    games = pd.DataFrame(
        [
            {"game_id": "oct", "game_date": date(2026, 10, 21), "tip_time": TIP},
            {
                "game_id": "nov",
                "game_date": date(2026, 11, 10),
                "tip_time": datetime(2026, 11, 11, 0, 30, tzinfo=UTC),
            },
            {
                "game_id": "early",
                "game_date": date(2026, 12, 25),
                "tip_time": datetime(2026, 12, 25, 15, 0, tzinfo=UTC),
            },  # 10:00 AM EST tip
        ]
    )
    t = layer_b.flag_times(games)
    assert t["oct"] == pd.Timestamp("2026-10-21 14:00", tz="UTC")  # EDT
    assert t["nov"] == pd.Timestamp("2026-11-10 15:00", tz="UTC")  # EST after Nov 1
    assert t["early"] == pd.Timestamp("2026-12-25 14:50", tz="UTC")  # 10 min before tip


def test_summary_and_report():
    _, picks = layer_b.replay(MODEL, ODDS, GAMES, outcomes(), engine.Rules())
    s = layer_b.summarize(picks)
    assert s["bets"] == 1 and s["roi_staked"] == pytest.approx(1.0) and "warning" in s
    rep = layer_b.report(picks, picks)
    assert set(rep) >= {"overall", "by_market", "by_ev", "by_source", "high_confidence"}
    assert rep["by_ev"]["8%-"]["bets"] == 1 and rep["by_source"]["consensus"]["bets"] == 1


def test_blend_curve_picks_the_more_informative_source_out_of_sample():
    """Consensus close = truth; the model is truth plus noise. The chosen weight should lean
    on the consensus, and the second half (never used to choose) should confirm it."""
    rng = np.random.default_rng(7)
    rows, games, outs, model = [], [], [], []
    for d in range(40):
        gid = f"g{d:03d}"
        day = date(2026, 11, 1) + timedelta(days=d)
        tip = datetime(day.year, day.month, day.day, 23, 0, tzinfo=UTC)
        games.append({"game_id": gid, "game_date": day, "tip_time": tip})
        for pid in range(25):
            p = rng.uniform(0.3, 0.7)
            for book in ("fanduel", "bovada"):
                over = int(round(to_american(1 / (p * 1.045))))
                under = int(round(to_american(1 / ((1 - p) * 1.045))))
                rows.append(
                    {
                        "id": len(rows),
                        "snapshot_ts": tip - timedelta(minutes=20),
                        "book": book,
                        "game_id": gid,
                        "player_id": pid,
                        "market": "pts",
                        "line_type": "main",
                        "line": 10.5,
                        "over_price": over,
                        "under_price": under,
                        "is_withdrawn": False,
                        "is_close": True,
                    }
                )
            hit = rng.random() < p
            outs.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "minutes": 30,
                    "dnp": False,
                    "pts": 11 if hit else 10,
                    "reb": 0,
                    "ast": 0,
                    "fg3m": 0,
                }
            )
            model.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "market": "pts",
                    "line": 10.5,
                    "p_over": float(np.clip(p + rng.normal(0, 0.15), 0.02, 0.98)),
                }
            )
    out = layer_b.blend_curve(
        pd.DataFrame(model), pd.DataFrame(rows), pd.DataFrame(games), pd.DataFrame(outs)
    )
    assert out["rows"] == 1000 and out["first_half_rows"] + out["second_half_rows"] == 1000
    assert out["chosen_weight"] <= 0.3
    sh = out["second_half"]
    assert sh["consensus_only"] < sh["model_only"]
    assert sh["chosen_weight"] <= sh["model_only"]


# --- parlay tiers and bankroll growth ------------------------------------------------------


def _picks(results, day="2026-11-01", stake=0.02):
    return pd.DataFrame(
        [
            {
                "game_id": f"g{i}",
                "player_id": i,
                "market": "pts",
                "line": 20.5,
                "side": "over",
                "decimal": 2.0,
                "p_fair": 0.62,
                "p_push": 0.0,
                "stake": stake,
                "result": r,
                "game_date": day,
                "profit": stake if r == "win" else -stake if r == "loss" else 0.0,
            }
            for i, r in enumerate(results)
        ]
    )


def test_satellites_are_graded_the_draftkings_way():
    rules = engine.Rules()
    for results, factor in (
        (["win", "win"], 4.0),
        (["win", "push"], 2.0),
        (["win", "void"], 2.0),
        (["win", "loss"], 0.0),
        (["push", "void"], 1.0),
    ):
        sats = layer_b.replay_satellites(_picks(results), rules)
        assert len(sats) == 1, results
        s = sats.iloc[0]
        expect = s["stake"] * (factor - 1) if factor else -s["stake"]
        assert s["profit"] == pytest.approx(expect), results
        assert s["result"] == ("loss" if factor == 0 else "push" if factor == 1 else "win")


def test_bankroll_growth_compounds_and_shares_the_daily_cap():
    rules = engine.Rules()
    picks = pd.concat([_picks(["win"], "2026-11-01", 0.02), _picks(["loss"], "2026-11-02", 0.02)])
    sats = pd.DataFrame([{"game_date": "2026-11-01", "stake": 0.01, "profit": 0.03}])
    g = layer_b.bankroll_growth(picks, sats, rules)
    assert g["singles_only"]["final_bankroll"] == pytest.approx(round(1.02 * 0.98, 4))
    assert g["singles_plus_satellites"]["final_bankroll"] == pytest.approx(round(1.05 * 0.98, 4))
    assert g["singles_only"]["max_drawdown"] == pytest.approx(0.02)
    big = pd.DataFrame([{"game_date": "2026-11-01", "stake": 0.18, "profit": 0.18}])
    capped = layer_b.bankroll_growth(_picks(["win"], "2026-11-01", 0.02), big, rules)
    # 0.20 staked > 0.10 cap: that day's result scales by 0.10 / 0.20
    assert capped["singles_plus_satellites"]["final_bankroll"] == pytest.approx(1 + 0.20 * 0.5)


def test_the_replay_runs_the_daily_schedule_with_first_flag_wins():
    moved = pd.concat(
        [
            ODDS,
            odds_frame(
                [
                    # 1:00 PM New York: DraftKings moves B's over to +120, +EV for the model
                    (datetime(2026, 10, 21, 17, 0, tzinfo=UTC), "draftkings", B, 10.5, 120, -145),
                ]
            ).assign(id=lambda d: d["id"] + 1000),
        ],
        ignore_index=True,
    )
    model = MODEL.copy()
    model.loc[model["player_id"] == B, ["p_over", "p_under"]] = [0.60, 0.40]
    no_cap = engine.Rules(max_edge=10.0)  # this example's edge is above the cap
    _, picks = layer_b.replay(model, moved, GAMES, outcomes(), no_cap)
    by = picks.set_index("player_id")
    assert by.loc[A, "flag_time"] == "10:00"  # A flagged at 10, never again
    assert by.loc[B, "flag_time"] == "15:00" and by.loc[B, "price"] == 120
    assert len(picks) == 2
    only_morning = layer_b.replay(model, moved, GAMES, outcomes(), no_cap, "10:00")[1]
    assert list(only_morning["player_id"]) == [A]
    rep = layer_b.report(picks, picks)
    assert set(rep["by_flag_time"]) == {"10:00", "15:00"} and "standard" in rep
