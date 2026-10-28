"""Parlay builder: exact push-aware EV, boosts, same-game joints, selection."""

import numpy as np
import pytest

from nbaprops.pricing import parlays as pl
from nbaprops.pricing.engine import Rules
from nbaprops.pricing.odds_math import boosted_decimal

RULES = Rules()


def leg(ref, game, p, d, push=0.0, line=24.5, side="over", market="pts", player=None):
    return pl.Leg(ref, game, player if player is not None else ref, market, line, side, d, p, push)


def test_independent_ev_is_the_product_formula():
    a, b = leg(0, "g1", 0.60, 2.0), leg(1, "g2", 0.55, 1.9)
    ev, p_all, p_lose = pl.payout_ev((a, b))
    assert ev == pytest.approx(0.60 * 2.0 * 0.55 * 1.9 - 1)
    # P(any leg loses) = 1 - P(no leg loses) = 1 - 0.60 * 0.55 (no pushes here)
    assert p_all == pytest.approx(0.33) and p_lose == pytest.approx(1 - 0.60 * 0.55)
    # pushes: a pushed leg drops out (factor 1): E = (0.5*2 + 0.1) * (0.6*2) - 1
    c = leg(2, "g3", 0.50, 2.0, push=0.10, line=25.0)
    ev, _, _ = pl.payout_ev((c, a))
    assert ev == pytest.approx((0.5 * 2 + 0.1) * (0.6 * 2) - 1)


def test_enumeration_equals_product_formula_on_random_legs():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(2, 5)
        legs = []
        for i in range(n):
            push = rng.choice([0.0, rng.uniform(0, 0.15)])
            legs.append(
                leg(i, f"g{i}", rng.uniform(0.3, 0.8) * (1 - push), rng.uniform(1.4, 3.0), push)
            )
        ev, _, _ = pl.payout_ev(legs)
        assert ev == pytest.approx(np.prod([lg.p_win * lg.decimal + lg.p_push for lg in legs]) - 1)


def test_the_plans_boost_example():
    """Plan, Stage 7: a 4-leg parlay at +284 with a 30% boost and legs that truly hit 75% each
    has about +49% EV, against about +22% unboosted."""
    d_leg = 3.84**0.25
    legs = [leg(i, f"g{i}", 0.75, d_leg) for i in range(4)]
    plain, _, _ = pl.payout_ev(legs)
    boosted, _, _ = pl.payout_ev(legs, boost=0.30)
    assert plain == pytest.approx(0.75**4 * 3.84 - 1) and round(plain, 2) == 0.22
    assert boosted == pytest.approx(0.75**4 * boosted_decimal(3.84, 0.30) - 1)
    assert round(boosted, 2) == 0.48 and boosted > 0.48  # "about +49%"
    assert boosted_decimal(4.0, 0.5) == pytest.approx(5.5)  # +300 boosted 50% pays +450


def test_boost_applies_to_profit_after_a_pushed_leg():
    a = leg(0, "g1", 0.0, 2.0, push=1.0, line=25.0)  # certain push
    b = leg(1, "g2", 1.0, 2.0)  # certain win
    ev, _, _ = pl.payout_ev((a, b), boost=0.5)
    assert ev == pytest.approx(1 + (2.0 - 1) * 1.5 - 1)  # pays as a boosted single


def test_same_game_joint_preserves_the_simulations_dependence():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 10, 20000).astype(float)
    a = leg(0, "g", 0.55, 1.9, line=4.5, player=1)
    b = leg(1, "g", 0.55, 1.9, line=4.5, player=2)
    raw = {("g", 1, "pts", 4.5, "over"): 0.5, ("g", 2, "pts", 4.5, "over"): 0.5}
    together = {("g", 1): {"pts": x}, ("g", 2): {"pts": x}}  # perfectly correlated
    assert pl.same_game_joint((a, b), together, raw) == pytest.approx(0.55)  # capped at min
    apart = {("g", 1): {"pts": x}, ("g", 2): {"pts": rng.permutation(x)}}
    assert pl.same_game_joint((a, b), apart, raw) == pytest.approx(0.55 * 0.55, rel=0.05)


def test_satellites_pick_the_best_disjoint_pairs_from_different_games():
    legs = [
        leg(0, "g1", 0.62, 2.0),
        leg(1, "g2", 0.60, 2.0),
        leg(2, "g3", 0.58, 2.0),
        leg(3, "g4", 0.57, 2.0),
        leg(4, "g1", 0.70, 2.0, player=40),
    ]
    sats = pl.satellites(legs, RULES, 1000.0, max_n=2)
    assert 1 <= len(sats) <= 2
    refs = [lg.pick_ref for s in sats for lg in s.legs]
    assert len(refs) == len(set(refs))  # a leg is in at most one parlay
    for s in sats:
        assert s.legs[0].game_id != s.legs[1].game_id
        assert s.ev >= RULES.ev_min and 0 < s.stake <= RULES.max_parlay_pct * 1000 + 1e-9
    assert not pl.satellites(legs, RULES, 1000.0, max_n=2, used={0, 1, 2, 3, 4})


def test_satellites_skip_negative_or_thin_edges():
    legs = [leg(0, "g1", 0.50, 1.95), leg(1, "g2", 0.50, 1.95)]
    assert pl.satellites(legs, RULES, 1000.0, max_n=2) == []


def test_boost_parlay_respects_games_legs_and_max_stake():
    legs = [leg(i, f"g{i % 4}", 0.72, 1.45, player=i) for i in range(6)]  # games repeat
    p = pl.boost_parlay(legs, boost=0.30, max_stake=10.0, rules=RULES, bankroll=1000.0)
    assert p is not None and 3 <= len(p.legs) <= 4
    assert len({lg.game_id for lg in p.legs}) == len(p.legs)
    assert p.stake <= 10.0 and p.boost_pct == 0.30 and p.ev >= RULES.ev_min


def test_same_game_parlay_at_draftkings_price_with_haircut():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 10, 20000).astype(float)
    a = leg(0, "g", 0.55, 1.9, line=4.5, player=1)
    b = leg(1, "g", 0.55, 1.9, line=4.5, player=2)
    draws = {("g", 1): {"pts": x}, ("g", 2): {"pts": x}}
    raw = {("g", 1, "pts", 4.5, "over"): 0.5, ("g", 2, "pts", 4.5, "over"): 0.5}
    p = pl.same_game_parlay([a, b], 150, draws, raw, RULES, 1000.0)
    assert p.same_game and p.ev == pytest.approx(0.55 * 2.5 - 1 - 0.03)
    assert 0 < p.stake <= RULES.max_parlay_pct * 1000
    with pytest.raises(ValueError):
        pl.same_game_parlay([a, leg(2, "h", 0.5, 1.9, player=3)], 150, draws, raw, RULES, 1000.0)
    with pytest.raises(ValueError):
        pl.same_game_parlay(
            [a, leg(3, "g", 0.5, 1.9, line=5.0, player=3)], 150, draws, raw, RULES, 1000.0
        )


def test_a_boost_can_turn_a_negative_ev_single_into_the_pick():
    import pandas as pd

    cands = pd.DataFrame(
        [
            # fair 0.52 at -110: EV -0.7% unboosted; with a 50% boost d = 1 + 0.909 * 1.5
            {
                "game_id": "g1",
                "player_id": 1,
                "market": "pts",
                "line": 24.5,
                "side": "over",
                "decimal": 1 + 100 / 110,
                "p_fair": 0.52,
                "p_push": 0.0,
                "low_confidence": False,
            },
            {
                "game_id": "g2",
                "player_id": 2,
                "market": "pts",
                "line": 20.5,
                "side": "over",
                "decimal": 1 + 100 / 110,
                "p_fair": 0.60,
                "p_push": 0.0,
                "low_confidence": True,
            },
        ]
    )
    row, par = pl.boost_single(cands, 0.5, 25.0, RULES, 1000.0)
    d = 1 + (100 / 110) * 1.5
    assert row["player_id"] == 1  # the stronger prop is low-confidence: excluded
    assert par.ev == pytest.approx(0.52 * (d - 1) - 0.48) and par.ev > 0.2
    assert par.decimal == pytest.approx(d) and par.boost_pct == 0.5
    assert 0 < par.stake <= 25.0 and par.tier == "boost" and len(par.legs) == 1
    assert pl.boost_single(cands, 0.5, 25.0, RULES, 1000.0, exclude_players={("g1", 1)}) is None


# --- alt-line parlays (a paper experiment) -------------------------------------------------


def _alt_cands():
    import pandas as pd

    def rung(gid, pid, market, line, p_model, p_fair, dec, **kw):
        return {
            "game_id": gid,
            "player_id": pid,
            "market": market,
            "line": line,
            "line_type": "milestone",
            "side": "over",
            "p_model": p_model,
            "p_fair": p_fair,
            "decimal": dec,
            "price": -150,
            "ev": p_fair * dec - 1,
            "has_consensus": True,
            "held_back": False,
            "low_confidence": False,
            **kw,
        }

    return pd.DataFrame(
        [
            rung("g1", 1, "reb", 5.5, 0.76, 0.74, 1.45),  # good leg
            rung("g1", 1, "reb", 7.5, 0.66, 0.62, 1.70),  # same player, worse rung: deduped
            rung("g2", 2, "fg3m", 1.5, 0.75, 0.73, 1.45),  # good leg
            rung("g3", 3, "ast", 3.5, 0.75, 0.74, 1.43),  # assists: shaded 1.5 points
            rung("g4", 4, "pts", 19.5, 0.78, 0.76, 1.45),  # points rungs: not a validated market
            rung("g5", 5, "reb", 6.5, 0.60, 0.60, 1.80),  # model below 65%
            rung("g6", 6, "reb", 4.5, 0.80, 0.80, 1.45, held_back=True),
            rung("g7", 7, "reb", 3.5, 0.80, 0.78, 1.40, low_confidence=True),
        ]
    )


def test_alt_combo_legs_only_from_validated_stats_inside_the_band():
    from nbaprops.pricing import parlays
    from nbaprops.pricing.engine import Rules

    pool = parlays.alt_combo_pool(_alt_cands(), Rules())
    assert sorted(zip(pool["player_id"], pool["market"], strict=True)) == [
        (1, "reb"),
        (2, "fg3m"),
        (3, "ast"),
    ]
    assert float(pool.set_index("player_id").loc[1, "line"]) == 5.5  # one rung per player
    ast = pool.set_index("player_id").loc[3]
    assert ast["p_leg"] == pytest.approx(0.74 - 0.015)  # assists shaded down


def test_alt_combos_pay_about_even_money_from_different_games():
    from nbaprops.pricing import parlays
    from nbaprops.pricing.engine import Rules

    rules = Rules()
    pool = parlays.alt_combo_pool(_alt_cands(), rules)
    built = parlays.alt_combos(pool, rules, 1000.0, 2)
    assert built
    for p in built:
        assert p.tier == "alt_combo" and len(p.legs) >= 2
        assert len({lg.game_id for lg in p.legs}) == len(p.legs)
        assert rules.alt_combo_decimal_min <= p.decimal <= rules.alt_combo_decimal_max
        assert p.ev >= rules.ev_min and 0 < p.stake <= rules.max_parlay_pct * 1000 + 1e-9
    assert parlays.alt_combos(pool, rules, 1000.0, 0) == []


def test_alt_combo_legs_reuse_a_flagged_pick_and_skip_busy_players():
    import pandas as pd

    from nbaprops.pricing import slate
    from nbaprops.pricing.engine import Rules

    cands = _alt_cands()
    # this run already flagged player 1's exact rung; player 2 was picked at 10 AM
    picks = cands.iloc[[0]].assign(stake=0.0, tracked_only=True).reset_index(drop=True)
    done = pd.DataFrame({"game_id": ["g2"], "player_id": [2], "stake": [10.0]})
    extra = cands.iloc[[0]].assign(game_id="g8", player_id=8, market="fg3m", line=2.5)
    picks2, built = slate._alt_combo_parlays(
        pd.concat([cands, extra], ignore_index=True), picks, done, Rules(), 1000.0, 2
    )
    players = {lg.player_id for p in built for lg in p.legs}
    assert built and 2 not in players
    for p in built:
        for lg in p.legs:
            row = picks2.iloc[lg.pick_ref]
            assert int(row["player_id"]) == lg.player_id and float(row["line"]) == lg.line
            if lg.player_id == 1:
                assert lg.pick_ref == 0  # the flagged row backs the leg
            else:
                assert row["tier_override"] == "boost_leg" and row["stake"] == 0.0


@pytest.mark.db
def test_the_database_accepts_the_alt_combo_tier():
    from datetime import UTC, date, datetime
    from decimal import Decimal

    from nbaprops.db.base import session_scope
    from nbaprops.db.models import Parlay as ParlayRow

    with session_scope() as s:
        s.add(
            ParlayRow(
                created_ts=datetime.now(UTC),
                slate_date=date(2026, 11, 1),
                tier="alt_combo",
                mode="paper",
                same_game=False,
                joint_prob=Decimal("0.52"),
                dk_price=105,
                ev=Decimal("0.06"),
                stake=Decimal("5"),
            )
        )
        s.flush()
        s.rollback()
