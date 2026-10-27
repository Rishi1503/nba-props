"""Market rebuild, consensus, and the pricing engine. Expected values are worked by hand."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nbaprops.pricing import engine, market
from nbaprops.pricing.odds_math import (
    ev_decimal,
    ev_with_push,
    kelly_decimal,
    kelly_with_push,
)

T0 = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)
G, A, B = "0022600010", 101, 102


def rows(*specs):
    """(minutes after T0, book, player, line, over, under, line_type, withdrawn)"""
    out = []
    for i, s in enumerate(specs):
        mins, book, pid, line, over, under, *rest = s
        lt = rest[0] if rest else "main"
        gone = rest[1] if len(rest) > 1 else False
        out.append(
            {
                "id": i,
                "snapshot_ts": T0 + timedelta(minutes=mins),
                "book": book,
                "game_id": G,
                "player_id": pid,
                "market": "pts",
                "line_type": lt,
                "line": line,
                "over_price": over,
                "under_price": under,
                "is_withdrawn": gone,
                "is_close": False,
            }
        )
    return pd.DataFrame(out)


def model(pid=A, line=24.5, p_over=0.70, p_push=0.0, low=False):
    return pd.DataFrame(
        [
            {
                "game_id": G,
                "player_id": pid,
                "market": "pts",
                "line": line,
                "p_over": p_over,
                "p_under": 1 - p_over - p_push,
                "p_push": p_push,
                "low_confidence": low,
                "game_date": "2026-10-21",
            }
        ]
    )


# --- market rebuild ---------------------------------------------------------------------


def test_board_keeps_the_latest_quote_per_prop_and_drops_withdrawn():
    odds = rows(
        (0, "draftkings", A, 24.5, -110, -110),
        (30, "draftkings", A, 25.5, -110, -110),  # main line moved: same prop
        (0, "draftkings", A, 29.5, 150, -180, "alternate"),
        (40, "draftkings", A, 29.5, None, None, "alternate", True),  # pulled
        (0, "fanduel", A, 24.5, -110, -110),
        (90, "fanduel", A, 26.5, -110, -110),  # after the cutoff below
    )
    b = market.board(odds, T0 + timedelta(minutes=60))
    got = sorted(zip(b["book"], b["line_type"], b["line"], strict=True))
    assert got == [("draftkings", "main", 25.5), ("fanduel", "main", 24.5)]
    early = market.board(odds, T0 + timedelta(minutes=35))
    assert ("draftkings", "alternate", 29.5) in set(
        zip(early["book"], early["line_type"], early["line"], strict=True)
    )


def test_board_per_game_uses_each_games_own_time():
    odds = pd.concat(
        [
            rows((0, "draftkings", A, 24.5, -110, -110), (60, "draftkings", A, 25.5, -110, -110)),
            rows(
                (0, "draftkings", B, 10.5, -110, -110), (60, "draftkings", B, 11.5, -110, -110)
            ).assign(game_id="other", id=lambda d: d.id + 10),
        ]
    )
    b = market.board_per_game(
        odds, {G: T0 + timedelta(minutes=30), "other": T0 + timedelta(minutes=90)}
    )
    assert dict(zip(b["game_id"], b["line"], strict=True)) == {G: 24.5, "other": 11.5}


def test_consensus_is_the_median_of_two_or_more_books_at_the_same_line():
    odds = rows(
        (0, "fanduel", A, 24.5, -110, -110),  # 0.5
        (0, "bovada", A, 24.5, -150, 120),  # 0.6 / (0.6 + 0.4545...) = 0.56897
        (0, "unibet", A, 24.5, -110, None),  # one side only: not usable
        (0, "draftkings", A, 24.5, -300, 240),  # never part of the consensus
        (0, "fanduel", B, 10.5, -110, -110),  # only one book: no consensus
    )
    c = market.consensus(market.board(odds, T0 + timedelta(hours=1)))
    assert list(c["player_id"]) == [A] and int(c["n_books"].iloc[0]) == 2
    assert c["p_over_cons"].iloc[0] == pytest.approx((0.5 + 0.6 / (0.6 + 100 / 220)) / 2)


def test_a_book_quoting_one_line_as_main_and_alternate_counts_once():
    odds = rows(
        (0, "fanduel", A, 24.5, -110, -110),
        (0, "fanduel", A, 24.5, -110, -110, "alternate"),
        (0, "bovada", A, 24.5, -110, -110),
    )
    c = market.consensus(market.board(odds, T0 + timedelta(hours=1)))
    assert int(c["n_books"].iloc[0]) == 2


def test_fair_close_prefers_consensus_then_draftkings_own_no_vig():
    close = market.board(
        rows(
            (0, "fanduel", A, 24.5, -150, 120),
            (0, "bovada", A, 24.5, -150, 120),
            (0, "draftkings", A, 24.5, -110, -110),
            (0, "draftkings", B, 10.5, -150, 120),  # no consensus for B
        ),
        T0 + timedelta(hours=1),
    )
    picks = pd.DataFrame(
        {
            "game_id": G,
            "player_id": [A, A, B, 999],
            "market": "pts",
            "line": [24.5, 24.5, 10.5, 5.5],
            "side": ["over", "under", "over", "over"],
        }
    )
    p = market.fair_close(close, picks)
    cons = 0.6 / (0.6 + 100 / 220)
    assert p[:3] == pytest.approx([cons, 1 - cons, cons])  # B: DraftKings' own no-vig
    assert np.isnan(p[3])


# --- odds math with pushes ---------------------------------------------------------------


def test_push_aware_ev_and_kelly_reduce_to_the_usual_formulas_without_a_push():
    assert ev_with_push(0.56, 0.44, 2.0) == pytest.approx(ev_decimal(0.56, 2.0))
    assert kelly_with_push(0.56, 0.44, 2.0) == pytest.approx(kelly_decimal(0.56, 2.0))
    # with a push: p=0.48, q=0.42, b=1 -> EV 0.06, Kelly (0.48-0.42)/(1*0.90)
    assert ev_with_push(0.48, 0.42, 2.0) == pytest.approx(0.06)
    assert kelly_with_push(0.48, 0.42, 2.0) == pytest.approx(0.06 / 0.90)
    assert kelly_with_push(0.40, 0.50, 2.0) == 0.0


# --- pricing engine ------------------------------------------------------------------------

RULES = engine.Rules()
# for tests of other rules whose examples carry edges above the sanity cap (tested on its own)
NO_CAP = engine.Rules(max_edge=10.0)
BOARD = rows(
    (0, "draftkings", A, 24.5, 100, -120),  # no-vig over = 0.5 / (0.5 + 0.54545) = 0.47826
    (0, "fanduel", A, 24.5, -110, -110),
    (0, "bovada", A, 24.5, -110, -110),  # consensus over = 0.5
)
NOW = T0 + timedelta(hours=1)


def test_flagged_pick_matches_hand_calculation():
    c = engine.candidates(model(p_over=0.70), market.board(BOARD, NOW), RULES).iloc[0]
    # fair = 0.3 * 0.70 + 0.7 * 0.50 = 0.56; at +100 (d = 2): EV = 0.56 - 0.44 = 0.12
    assert c["side"] == "over" and c["has_consensus"]
    assert c["p_fair"] == pytest.approx(0.56) and c["ev"] == pytest.approx(0.12)
    assert c["p_dk"] == pytest.approx(0.5 / (0.5 + 120 / 220))
    assert c["flag"] and c["high_confidence"]  # model 0.70 >= 0.70
    assert c["kelly"] == pytest.approx(0.02)  # quarter Kelly 0.03, capped at 2%


def test_the_ev_floor():
    # p_model 0.52: fair = 0.156 + 0.35 = 0.506, EV = 0.012 < 3%: not flagged
    low = engine.candidates(model(p_over=0.52), market.board(BOARD, NOW), RULES).iloc[0]
    assert low["ev"] == pytest.approx(0.012) and not low["flag"]
    # p_model 0.60: fair = 0.18 + 0.35 = 0.53, EV = 0.06 >= 3%: flagged
    high = engine.candidates(model(p_over=0.60), market.board(BOARD, NOW), RULES).iloc[0]
    assert high["ev"] == pytest.approx(0.06) and high["flag"] and not high["high_confidence"]


def test_model_only_shrinks_toward_draftkings_and_needs_five_percent():
    board = market.board(BOARD[BOARD["book"].isin(["draftkings", "fanduel"])], NOW)
    c = engine.candidates(model(p_over=0.70), board, NO_CAP).iloc[0]
    dk = 0.5 / (0.5 + 120 / 220)
    fair = 0.70 + 0.5 * (dk - 0.70)
    assert not c["has_consensus"] and c["p_fair"] == pytest.approx(fair)
    assert c["ev"] == pytest.approx(fair - (1 - fair)) and c["flag"]
    weak = engine.candidates(model(p_over=0.56), board, RULES).iloc[0]
    assert 0.0 < weak["ev"] < RULES.ev_min_model_only and not weak["flag"]


def test_no_flag_when_the_consensus_sees_the_other_side():
    board = market.board(
        rows(
            (0, "draftkings", A, 24.5, 130, -160),  # no-vig over 0.4140
            (0, "fanduel", A, 24.5, 150, -180),
            (0, "bovada", A, 24.5, 150, -180),  # cons 0.3913
        ),
        NOW,
    )
    c = engine.candidates(model(p_over=0.80), board, RULES).iloc[0]
    assert c["side"] == "over" and c["ev"] > RULES.ev_min  # the blend still likes it...
    assert not c["agree"] and not c["flag"]  # ...but the market doesn't agree


def test_low_confidence_minutes_block_the_flag():
    c = engine.candidates(model(p_over=0.70, low=True), market.board(BOARD, NOW), RULES).iloc[0]
    assert c["ev"] == pytest.approx(0.12) and not c["flag"]


def test_whole_number_line_counts_the_push():
    board = market.board(
        rows(
            (0, "draftkings", A, 25.0, 100, -120),
            (0, "fanduel", A, 25.0, -110, -110),
            (0, "bovada", A, 25.0, -110, -110),
        ),
        NOW,
    )
    c = engine.candidates(model(line=25.0, p_over=0.55, p_push=0.10), board, RULES).iloc[0]
    # win = 0.3 * 0.55 + 0.7 * (0.5 * 0.9) = 0.48; lose = 1 - 0.1 - 0.48 = 0.42
    assert c["p_fair"] == pytest.approx(0.48) and c["p_lose"] == pytest.approx(0.42)
    assert c["ev"] == pytest.approx(0.06)
    assert c["kelly"] == pytest.approx(0.25 * 0.06 / 0.90)


def test_sizing_one_pick_per_player_and_per_game_and_daily_caps():
    c = pd.DataFrame(
        {
            "game_id": ["g1"] * 4 + [f"g{i}" for i in range(2, 9)],
            "player_id": [1, 1, 2, 3] + list(range(10, 17)),
            "market": ["pts", "reb", "pts", "pts"] + ["pts"] * 7,
            "ev": [0.10, 0.20, 0.10, 0.10] + [0.10] * 7,
            "kelly": [0.02] * 11,
            "flag": True,
            "game_date": "2026-10-21",
        }
    )
    picks = engine.select_and_size(c, 1000.0, RULES)
    p1 = picks[picks["player_id"] == 1]
    assert list(p1["market"]) == ["reb"]  # best EV for that player in that game
    g1 = picks[picks["game_id"] == "g1"]["stake"]
    assert len(g1) == 3
    # game cap: 3 x 20 = 60 > 40 -> scaled to 40 in that game; then the day: 40 + 7 x 20 = 180
    # > 100 -> everything scaled by 100/180
    assert g1.sum() == pytest.approx(40 * 100 / 180)
    assert picks["stake"].sum() == pytest.approx(100.0)
    assert picks["stake"].max() <= 20.0 + 1e-9


# --- milestones ---------------------------------------------------------------------------


def poisson_grid(mean, lo=5, hi=45):
    from scipy.stats import poisson

    return {f"{k + 0.5:.1f}": float(poisson.sf(k, mean)) for k in range(lo, hi)}


def test_shifted_tail_without_anchor_reads_the_grid():
    g = poisson_grid(22)
    assert engine.shifted_tail(g, 19.5) == pytest.approx(g["19.5"])


def test_shifted_tail_matches_the_anchor_exactly_and_moves_the_tail():
    g = poisson_grid(22)
    # consensus says 24.5 is a coin flip: the model (median ~22) is shifted up by ~2.5
    assert engine.shifted_tail(g, 24.5, 24.5, 0.5) == pytest.approx(0.5, abs=1e-9)
    up = engine.shifted_tail(g, 29.5, 24.5, 0.5)
    assert up > g["29.5"]  # the tail moved with the location
    assert engine.shifted_tail(g, 29.5, 24.5, 0.6) > up  # a stronger consensus lifts it more


def _ms_board(with_consensus=True, price=150):
    specs = [(0, "draftkings", A, 19.5, price, None, "milestone")]
    if with_consensus:
        specs += [(0, "fanduel", A, 24.5, -110, -110), (0, "bovada", A, 24.5, -110, -110)]
    return market.board(rows(*specs), NOW)


def _grids(mean=22, low=False):
    return pd.DataFrame(
        [
            {
                "game_id": G,
                "player_id": A,
                "market": "pts",
                "p_over": poisson_grid(mean),
                "low_confidence": low,
                "game_date": "2026-10-21",
            }
        ]
    )


def test_milestone_with_consensus_uses_the_shifted_distribution():
    c = engine.milestone_candidates(_grids(), _ms_board(), RULES).iloc[0]
    expect = engine.shifted_tail(poisson_grid(22), 19.5, 24.5, 0.5)
    assert c["has_consensus"] and c["anchor_line"] == 24.5
    assert c["p_fair"] == pytest.approx(expect)
    assert c["ev"] == pytest.approx(expect * 1.5 - (1 - expect))  # +150 -> d = 2.5
    # flagged only between the EV floor and the sanity cap (above it: held back)
    assert c["flag"] == (RULES.ev_min <= c["ev"] <= RULES.max_edge) and c["side"] == "over"
    assert c["held_back"] == (c["ev"] > RULES.max_edge)
    assert c["line_type"] == "milestone" and c["p_push"] == 0


def test_milestone_without_consensus_is_model_only_and_needs_five_percent():
    c = engine.milestone_candidates(_grids(), _ms_board(False), RULES).iloc[0]
    p = poisson_grid(22)["19.5"]
    fair = p + 0.5 * (1 / 2.5 - p)  # halfway toward DraftKings' implied probability
    assert not c["has_consensus"] and c["p_fair"] == pytest.approx(fair)
    assert c["flag"] == (RULES.ev_min_model_only <= c["ev"] <= RULES.max_edge)


def test_milestone_blocked_by_low_confidence_and_skipped_without_a_projection():
    assert not engine.milestone_candidates(_grids(low=True), _ms_board(), RULES).iloc[0]["flag"]
    other = _grids().assign(player_id=B)
    assert engine.milestone_candidates(other, _ms_board(), RULES).empty


# --- sizing across runs in one day ------------------------------------------------------


def test_earlier_picks_count_against_the_caps_and_block_repeat_players():
    c = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g3"],
            "player_id": [1, 2, 3, 4],
            "market": "pts",
            "ev": 0.1,
            "kelly": 0.02,
            "flag": True,
            "game_date": "2026-10-21",
        }
    )
    committed = pd.DataFrame(
        {
            "game_id": ["g1", "g9"],
            "player_id": [1, 9],
            "stake": [30.0, 60.0],
            "game_date": "2026-10-21",
        }
    )
    picks = engine.select_and_size(c, 1000.0, RULES, committed=committed)
    assert 1 not in set(picks["player_id"])  # already picked earlier today
    g1 = picks[picks["game_id"] == "g1"]["stake"].sum()
    assert g1 <= 40 - 30 + 1e-9  # the game cap counts the earlier 30
    assert picks["stake"].sum() + 90 <= 100 + 1e-9  # the day cap counts the earlier 90
    full = pd.DataFrame(
        {"game_id": ["g9"], "player_id": [9], "stake": [100.0], "game_date": "2026-10-21"}
    )
    assert engine.select_and_size(c, 1000.0, RULES, committed=full).empty  # day already full


def test_standard_flags_are_tracked_at_zero_stake_when_the_strategy_says_so():
    from dataclasses import replace

    c = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": [1, 2],
            "market": "pts",
            "ev": [0.06, 0.10],
            "kelly": 0.02,
            "flag": True,
            "high_confidence": [False, True],
            "game_date": "2026-10-21",
        }
    )
    hc_only = engine.select_and_size(c, 1000.0, replace(RULES, stake_standard_flags=False))
    by = hc_only.set_index("player_id")
    assert by.loc[1, "stake"] == 0 and by.loc[1, "tracked_only"]  # logged, not bet
    assert by.loc[2, "stake"] == pytest.approx(20.0) and not by.loc[2, "tracked_only"]
    both = engine.select_and_size(c, 1000.0, replace(RULES, stake_standard_flags=True))
    assert (both["stake"] == 20.0).all()


def test_lines_beyond_the_models_range_are_not_priced():
    board = market.board(
        rows(
            (0, "draftkings", A, 9.5, -2000, 1000, "alternate"),
            (0, "draftkings", A, 24.5, 100, -120),
        ),
        NOW,
    )
    m = pd.concat([model(line=9.5, p_over=1.0), model(line=24.5, p_over=0.70)])
    c = engine.candidates(m, board, RULES)
    assert list(c["line"]) == [24.5]  # 9.5 reads as a certainty: skipped
    g = poisson_grid(22, lo=5, hi=45)
    assert np.isnan(engine.shifted_tail(g, 60.5)) and np.isnan(engine.shifted_tail(g, 2.5))
    far = _ms_board().assign(
        line=lambda d: np.where(d["line_type"] == "milestone", 59.5, d["line"])
    )
    assert engine.milestone_candidates(_grids(), far, RULES).empty


# --- Questionable teammates ------------------------------------------------------------


def scen_model(p_mix, p_in, p_out):
    return model(p_over=p_mix).assign(
        p_over_in=p_in,
        p_under_in=1 - p_in,
        p_push_in=0.0,
        p_over_out=p_out,
        p_under_out=1 - p_out,
        p_push_out=0.0,
    )


def test_robust_rule_flags_bets_that_hold_up_either_way():
    from dataclasses import replace

    c = engine.candidates(scen_model(0.70, 0.72, 0.66), market.board(BOARD, NOW), RULES).iloc[0]
    # fair if he plays: 0.3 * 0.72 + 0.35 = 0.566 -> EV +0.132; if he sits: 0.548 -> +0.096
    assert c["ev_if_plays"] == pytest.approx(0.132) and c["ev_if_sits"] == pytest.approx(0.096)
    assert c["flag"]
    # block never flags a Questionable teammate
    blocked = engine.candidates(
        scen_model(0.70, 0.72, 0.66),
        market.board(BOARD, NOW),
        replace(RULES, questionable_teammates="block"),
    ).iloc[0]
    assert not blocked["flag"]


def test_robust_rule_rejects_a_bet_that_depends_on_the_news():
    from dataclasses import replace

    board = market.board(BOARD, NOW)
    m = scen_model(0.62, 0.40, 0.84)  # blend fair 0.536 -> EV +0.072, but -0.06 if he plays
    robust = engine.candidates(m, board, RULES).iloc[0]
    assert robust["ev"] == pytest.approx(0.072) and robust["ev_if_plays"] == pytest.approx(-0.06)
    assert not robust["flag"]
    assert engine.candidates(m, board, replace(RULES, questionable_teammates="mixture")).iloc[0][
        "flag"
    ]


def test_rows_without_a_questionable_teammate_are_unaffected_by_the_rule():
    from dataclasses import replace

    for rule in ("robust", "mixture", "block"):
        c = engine.candidates(
            model(p_over=0.70),
            market.board(BOARD, NOW),
            replace(RULES, questionable_teammates=rule),
        ).iloc[0]
        assert c["flag"] and np.isnan(c["ev_if_plays"])


def test_milestone_scenarios_share_the_blends_shift():
    g = _grids().assign(
        scenarios=[{"player_id": 9, "p_play": 0.5, "in": poisson_grid(19), "out": poisson_grid(25)}]
    )
    c = engine.milestone_candidates(g, _ms_board(), NO_CAP).iloc[0]
    shift = engine.anchor_shift(poisson_grid(22), 24.5, 0.5)
    f_in = engine.shifted_tail(poisson_grid(19), 19.5, shift=shift)
    f_out = engine.shifted_tail(poisson_grid(25), 19.5, shift=shift)
    assert c["ev_if_plays"] == pytest.approx(f_in * 1.5 - (1 - f_in))
    assert c["ev_if_sits"] == pytest.approx(f_out * 1.5 - (1 - f_out))
    assert c["flag"] == (c["ev"] >= RULES.ev_min and min(c["ev_if_plays"], c["ev_if_sits"]) >= 0)


def test_selection_prefers_growth_over_a_longshots_higher_ev():
    """Same player, two flagged props, each at its own quarter-Kelly stake: a +450 alternate
    with EV +16% and a -110 line with EV +10%. Growth scales roughly with EV^2 / (d - 1), so
    the longshot needs proportionally more edge; here the -110 line is the pick."""
    from nbaprops.pricing.odds_math import kelly_with_push

    rows = []
    for line, d, ev in ((34.5, 5.5, 0.16), (24.5, 1 + 100 / 110, 0.10)):
        p = (1 + ev) / d  # p * (d - 1) - (1 - p) = ev
        rows.append(
            {
                "game_id": "g1",
                "player_id": 1,
                "market": "pts",
                "line": line,
                "decimal": d,
                "p_fair": p,
                "p_lose": 1 - p,
                "ev": ev,
                "kelly": min(kelly_with_push(p, 1 - p, d, 0.25), 0.02),
                "flag": True,
                "high_confidence": False,
                "game_date": "d",
            }
        )
    c = pd.DataFrame(rows)
    g = engine.expected_growth(c)
    assert c["ev"].iloc[0] > c["ev"].iloc[1] and g[1] > g[0]  # more EV, less growth
    pick = engine.select_and_size(c, 1000.0, RULES)
    assert list(pick["line"]) == [24.5]


def test_a_nearby_consensus_line_is_read_across_to_draftkings_line():
    board = market.board(
        rows(
            (0, "draftkings", A, 25.5, 100, -120),
            (0, "fanduel", A, 24.5, -110, -110),
            (0, "bovada", A, 24.5, -110, -110),
        ),
        NOW,
    )
    g = poisson_grid(26)
    m = model(line=25.5, p_over=g["25.5"])
    alone = engine.candidates(m, board, RULES).iloc[0]
    assert not alone["has_consensus"]  # no two books at 25.5: model-only
    grids = pd.DataFrame([{"game_id": G, "player_id": A, "market": "pts", "p_over": g}])
    c = engine.candidates(m, board, RULES, grids).iloc[0]
    assert c["has_consensus"] and c["cons_source"] == "shifted"
    expect = engine.shifted_tail(g, 25.5, 24.5, 0.5)
    assert c["p_over_cons"] == pytest.approx(expect)
    far = market.board(
        rows(
            (0, "draftkings", A, 25.5, 100, -120),
            (0, "fanduel", A, 22.5, -110, -110),
            (0, "bovada", A, 22.5, -110, -110),
        ),
        NOW,
    )
    assert not engine.candidates(m, far, RULES, grids).iloc[0]["has_consensus"]  # 3 points


def test_a_high_confidence_pick_keeps_the_players_slot_when_standard_flags_arent_staked():
    """Same player, two flagged props: a standard longshot with more growth and a
    high-confidence favorite. If standard flags aren't staked, the favorite must win the
    player's one slot, or we'd bet nothing on him."""
    from dataclasses import replace

    c = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "player_id": 1,
                "market": "ast",
                "line": 8.5,
                "decimal": 6.38,
                "p_fair": 0.227,
                "p_lose": 0.773,
                "ev": 0.448,
                "kelly": 0.02,
                "flag": True,
                "high_confidence": False,
                "game_date": "d",
            },
            {
                "game_id": "g1",
                "player_id": 1,
                "market": "ast",
                "line": 4.5,
                "decimal": 1.284,
                "p_fair": 0.827,
                "p_lose": 0.173,
                "ev": 0.063,
                "kelly": 0.02,
                "flag": True,
                "high_confidence": True,
                "game_date": "d",
            },
        ]
    )
    g = engine.expected_growth(c)
    assert g[0] > g[1]  # the longshot has more growth on paper
    hc_only = engine.select_and_size(c, 1000.0, replace(RULES, stake_standard_flags=False))
    assert list(hc_only["line"]) == [4.5] and hc_only["stake"].iloc[0] > 0
    both = engine.select_and_size(c, 1000.0, replace(RULES, stake_standard_flags=True))
    assert list(both["line"]) == [8.5]  # staking everything: growth decides


def test_the_sanity_cap_holds_back_implausible_edges():
    """Oct 26, 2026: main-line edges of +18% to +37% (Jaylin Williams, Haliburton). An edge
    that large usually means the model is missing news, not that DraftKings is wrong."""
    from dataclasses import replace

    # model 0.90: fair = 0.27 + 0.35 = 0.62 at +100: EV 0.24, above the 0.15 cap
    c = engine.candidates(model(p_over=0.90), market.board(BOARD, NOW), RULES).iloc[0]
    assert c["ev"] == pytest.approx(0.24)
    assert c["held_back"] and not c["flag"] and not c["high_confidence"]
    wide = engine.candidates(
        model(p_over=0.90), market.board(BOARD, NOW), replace(RULES, max_edge=0.30)
    ).iloc[0]
    assert wide["flag"] and not wide["held_back"]


def test_far_milestone_rungs_are_never_flagged():
    """The far tail is priced from the model alone, and it overstated longshots ("45+ points"
    and "25+ points" parlay legs at +118%). Only rungs inside the validated band count."""
    from dataclasses import replace

    no_cap = replace(RULES, max_edge=10.0)
    far_board = market.board(
        rows(
            (0, "draftkings", A, 34.5, 900, None, "milestone"),
            (0, "fanduel", A, 24.5, -110, -110),
            (0, "bovada", A, 24.5, -110, -110),
        ),
        NOW,
    )
    far = engine.milestone_candidates(_grids(), far_board, no_cap).iloc[0]
    assert far["p_model"] < RULES.milestone_p_min
    assert not far["flag"] and not far["held_back"]  # out of band: never a pick, even uncapped
    near = engine.milestone_candidates(_grids(), _ms_board(), no_cap).iloc[0]
    assert RULES.milestone_p_min <= near["p_model"] <= RULES.milestone_p_max
    assert near["flag"]  # inside the band, edge allowed by this cap: flagged
    capped = engine.milestone_candidates(_grids(), _ms_board(), RULES).iloc[0]
    assert capped["held_back"] and not capped["flag"]  # same rung, default cap: held back


def test_the_cap_and_band_are_validated_in_the_config():
    import yaml

    from nbaprops.config import StrategyConfig

    base = yaml.safe_load(open("config/strategy.yaml"))
    StrategyConfig.model_validate(base)  # the shipped config is valid
    with pytest.raises(ValueError, match="max_edge"):
        StrategyConfig.model_validate({**base, "max_edge": base["ev_min_model_only"]})
    with pytest.raises(ValueError, match="milestone_p_min"):
        StrategyConfig.model_validate({**base, "milestone_p_min": 0.9, "milestone_p_max": 0.5})


def test_the_board_says_how_many_props_were_held_back():
    from datetime import date, datetime

    from nbaprops.pricing import run as prun
    from nbaprops.pricing.slate import SlateResult

    res = SlateResult(date(2026, 10, 26), datetime(2026, 10, 26, 20, 0), "r", held_back=3)
    assert "3 props held back" in prun.render(res)
