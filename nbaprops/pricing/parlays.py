"""Parlay builder (plan, Stage 7 "Parlay builder").

Every parlay is built only from flagged singles, one leg per player, and a leg appears in at
most one parlay. Two tiers:

  satellite  2 legs from different games; quarter Kelly on the parlay's own EV, capped at
             max_parlay_pct; at most max_satellites_per_slate per slate.
  boost      3-4 legs when a DraftKings profit boost is entered; quarter Kelly on the boosted
             EV, up to the boost's max stake. Different games, unless the joint probability
             comes from the simulation (same_game_joint).

Payout with pushes is exact: DraftKings drops a pushed leg and pays the rest, so each leg's
payout factor is d (win), 1 (push) or 0 (loss), and the parlay pays the product (all pushes
return the stake). For independent legs E[payout] = prod(p_win * d + p_push); with a boost
the profit part is multiplied by (1 + b), which needs the outcome enumeration below.

Same-game legs: joint hit rate from the shared simulation, kept "lift-preserving": the
calibrated marginals multiplied by the simulation's dependence ratio
joint_raw / prod(raw marginals), then sgp_ev_haircut points off EV (joint tails are the least
reliable part of the model). Same-game parlays need DraftKings' own price (it applies its own
correlation adjustment), entered manually.

Selection maximizes expected log-growth over a disjoint set of candidates, ranked by EV.
"""

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from nbaprops.pricing.odds_math import boosted_decimal, kelly_with_push, to_american


@dataclass(frozen=True)
class Leg:
    pick_ref: int  # index of the flagged single this leg comes from
    game_id: str
    player_id: int
    market: str
    line: float
    side: str
    decimal: float
    p_win: float  # fair probabilities
    p_push: float = 0.0

    @property
    def p_lose(self) -> float:
        return max(0.0, 1.0 - self.p_win - self.p_push)


@dataclass
class Parlay:
    tier: str
    legs: tuple[Leg, ...]
    decimal: float  # unboosted payout; boosted_decimal for boost parlays
    p_all_win: float
    p_any_lose: float
    ev: float
    boost_pct: float | None = None
    same_game: bool = False
    kelly: float = 0.0
    stake: float = 0.0
    growth: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def american(self) -> int:
        return int(round(to_american(self.decimal)))


def legs_from_picks(picks) -> list[Leg]:
    """Flagged singles (a DataFrame from engine.select_and_size) as parlay legs."""
    out = []
    for i, r in enumerate(picks.itertuples(index=False)):
        out.append(
            Leg(
                i,
                r.game_id,
                int(r.player_id),
                r.market,
                float(r.line),
                r.side,
                float(r.decimal),
                float(r.p_fair),
                float(getattr(r, "p_push", 0.0) or 0),
            )
        )
    return out


def payout_ev(legs, boost: float = 0.0) -> tuple[float, float, float]:
    """(EV per $1, P(every leg wins), P(any leg loses)) for independent legs, enumerating
    win/push/loss per leg. With boost b the profit of the surviving legs is scaled by 1 + b."""
    exp_payout = 0.0
    for outcome in product((0, 1, 2), repeat=len(legs)):  # 0 win, 1 push, 2 loss
        p, mult = 1.0, 1.0
        for leg, o in zip(legs, outcome, strict=True):
            p *= (leg.p_win, leg.p_push, leg.p_lose)[o]
            if o == 0:
                mult *= leg.decimal
            elif o == 2:
                mult = 0.0
        if p == 0.0 or mult == 0.0:
            continue
        payout = 1.0 + (mult - 1.0) * (1.0 + boost) if mult > 1.0 else mult
        exp_payout += p * payout
    p_all = float(np.prod([leg.p_win for leg in legs]))
    p_any_lose = 1.0 - float(np.prod([1.0 - leg.p_lose for leg in legs]))
    return exp_payout - 1.0, p_all, p_any_lose


def same_game_joint(legs, draws: dict, raw: dict) -> float:
    """Lift-preserving joint hit probability for legs in one game. `draws`:
    {(game_id, player_id): {market: simulated values}} from the same simulated games;
    `raw`: {(game_id, player_id, market, line, side): raw simulated marginal}."""
    hits = []
    for leg in legs:
        d = np.asarray(draws[(leg.game_id, leg.player_id)][leg.market])
        hits.append(d > leg.line if leg.side == "over" else d < leg.line)
    joint_raw = float(np.mean(np.all(hits, axis=0)))
    prod_raw = float(
        np.prod([raw[(lg.game_id, lg.player_id, lg.market, lg.line, lg.side)] for lg in legs])
    )
    lift = joint_raw / prod_raw if prod_raw > 0 else 1.0
    joint = float(np.prod([lg.p_win for lg in legs])) * lift
    return float(min(joint, min(lg.p_win for lg in legs)))


def _growth(p_win: float, p_lose: float, decimal: float, f: float) -> float:
    """Expected log-growth of staking fraction f (push mass contributes 0: conservative)."""
    if f <= 0:
        return 0.0
    return p_win * np.log1p((decimal - 1) * f) + p_lose * np.log1p(-f)


def _finish(p: Parlay, fraction: float, cap: float, bankroll: float) -> Parlay:
    p.kelly = min(kelly_with_push(p.p_all_win, p.p_any_lose, p.decimal, fraction), cap)
    p.stake = p.kelly * bankroll
    p.growth = _growth(p.p_all_win, p.p_any_lose, p.decimal, p.kelly)
    return p


def _best_disjoint(cands: list[Parlay], k: int, used: set[int]) -> list[Parlay]:
    """Highest total growth over at most k candidates with no shared legs (exact search
    over the top candidates by EV; slates have few flagged legs)."""
    pool = [
        c
        for c in sorted(cands, key=lambda c: -c.ev)[:60]
        if c.growth > 0 and not {lg.pick_ref for lg in c.legs} & used
    ]
    best, best_g = [], 0.0
    for n in range(1, k + 1):
        for combo in combinations(pool, n):
            refs = [lg.pick_ref for c in combo for lg in c.legs]
            if len(refs) != len(set(refs)):
                continue
            g = sum(c.growth for c in combo)
            if g > best_g:
                best, best_g = list(combo), g
    return best


def satellites(
    legs: list[Leg], rules, bankroll: float, max_n: int, used: set[int] | None = None
) -> list[Parlay]:
    """Two-leg parlays from different games with EV >= ev_min, sized by quarter Kelly
    (capped at max_parlay_pct), chosen for the highest total expected growth."""
    used = used or set()
    cands = []
    for a, b in combinations(legs, 2):
        if a.game_id == b.game_id:
            continue  # same-game pairs need the simulation and DraftKings' own SGP price
        ev, p_all, p_lose = payout_ev((a, b))
        if ev < rules.ev_min:
            continue
        p = Parlay("satellite", (a, b), a.decimal * b.decimal, p_all, p_lose, ev)
        cands.append(_finish(p, rules.kelly_fraction, rules.max_parlay_pct, bankroll))
    return _best_disjoint(cands, max_n, used)


def boost_parlay(
    legs: list[Leg],
    boost: float,
    max_stake: float,
    rules,
    bankroll: float,
    used: set[int] | None = None,
    min_legs: int = 3,
    max_legs: int = 4,
) -> Parlay | None:
    """The single best boosted parlay (a DraftKings boost applies to one bet): 3-4 legs from
    different games, boosted EV >= ev_min, quarter Kelly on the boosted price, at most the
    boost's max stake."""
    used = used or set()
    pool = [lg for lg in legs if lg.pick_ref not in used]
    cands = []
    for n in range(min_legs, max_legs + 1):
        for combo in combinations(pool, n):
            if len({lg.game_id for lg in combo}) < n:
                continue
            ev, p_all, p_lose = payout_ev(combo, boost)
            if ev < rules.ev_min:
                continue
            d = boosted_decimal(float(np.prod([lg.decimal for lg in combo])), boost)
            p = Parlay("boost", combo, d, p_all, p_lose, ev, boost_pct=boost)
            p = _finish(p, rules.kelly_fraction, 1.0, bankroll)
            p.stake = min(p.stake, max_stake)
            p.kelly = p.stake / bankroll if bankroll else 0.0
            p.growth = _growth(p.p_all_win, p.p_any_lose, p.decimal, p.kelly)
            cands.append(p)
    best = _best_disjoint(cands, 1, used)
    return best[0] if best else None


def same_game_parlay(
    legs: list[Leg],
    dk_price: int,
    draws: dict,
    raw: dict,
    rules,
    bankroll: float,
    boost: float = 0.0,
) -> Parlay:
    """Price a same-game parlay at DraftKings' quoted SGP price (entered manually). Legs must
    be on half-point lines (DraftKings re-prices pushed SGP legs its own way)."""
    if len({lg.game_id for lg in legs}) != 1:
        raise ValueError("a same-game parlay's legs must share one game")
    if any(float(lg.line) % 1 == 0 for lg in legs):
        raise ValueError("same-game parlay legs must be on half-point lines")
    from nbaprops.pricing.odds_math import to_decimal

    joint = same_game_joint(legs, draws, raw)
    d = boosted_decimal(to_decimal(dk_price), boost) if boost else to_decimal(dk_price)
    ev = joint * d - 1.0 - rules.sgp_ev_haircut
    p = Parlay(
        "boost" if boost else "satellite",
        tuple(legs),
        d,
        joint,
        1 - joint,
        ev,
        boost_pct=boost or None,
        same_game=True,
        notes=[f"EV includes the {rules.sgp_ev_haircut:.0%} same-game haircut"],
    )
    if ev <= 0:
        return p
    p.kelly = min(rules.kelly_fraction * ev / (d - 1), rules.max_parlay_pct)
    p.stake = p.kelly * bankroll
    return p


def boost_single(
    cands,
    boost: float,
    max_stake: float,
    rules,
    bankroll: float,
    exclude_players: set | None = None,
):
    """The best single to put a DraftKings profit boost on: every candidate prop (flagged or
    not, since the boost itself can create the edge) repriced at the boosted odds, low-
    confidence players and players already picked today excluded, boosted EV >= ev_min,
    highest expected growth. Returns (candidate row, Parlay with one leg) or None."""
    from nbaprops.pricing.odds_math import ev_with_push

    exclude_players = exclude_players or set()
    best = None
    for i, r in enumerate(cands.itertuples(index=False)):
        if bool(r.low_confidence) or (r.game_id, int(r.player_id)) in exclude_players:
            continue
        d = boosted_decimal(float(r.decimal), boost)
        p_win, p_push = float(r.p_fair), float(getattr(r, "p_push", 0.0) or 0.0)
        p_lose = max(0.0, 1.0 - p_win - p_push)
        ev = ev_with_push(p_win, p_lose, d)
        if ev < rules.ev_min:
            continue
        f = min(kelly_with_push(p_win, p_lose, d, rules.kelly_fraction), max_stake / bankroll)
        g = _growth(p_win, p_lose, d, f)
        if best is None or g > best[0]:
            leg = Leg(
                i,
                r.game_id,
                int(r.player_id),
                r.market,
                float(r.line),
                r.side,
                float(r.decimal),
                p_win,
                p_push,
            )
            par = Parlay(
                "boost",
                (leg,),
                d,
                p_win,
                p_lose,
                ev,
                boost_pct=boost,
                kelly=f,
                stake=f * bankroll,
                growth=g,
                notes=["boosted single"],
            )
            best = (g, i, par)
    if best is None:
        return None
    return cands.iloc[best[1]], best[2]
