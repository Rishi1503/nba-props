"""Pure odds math used by the pricing engine, parlay builder, and CLV tracking.

Conventions: American odds are ints (-110, +250). Decimal odds are floats (1.909, 3.5).
Probabilities are floats in (0, 1). Every formula here matches the plan's Stage 7 and
the Implementation reference.
"""

from collections.abc import Iterable
from math import prod

# --- Conversions -------------------------------------------------------------


def _check_american(american: int | float) -> None:
    if -100 < american < 100:
        raise ValueError(f"invalid American odds {american}: must be <= -100 or >= +100")


def to_decimal(american: int | float) -> float:
    """American -> decimal. -250 -> 1.4, +150 -> 2.5."""
    _check_american(american)
    if american < 0:
        return 1.0 + 100.0 / abs(american)
    return 1.0 + american / 100.0


def to_american(decimal: float) -> float:
    """Decimal -> American (float; round for display). 1.4 -> -250, 2.5 -> +150."""
    if decimal <= 1.0:
        raise ValueError(f"invalid decimal odds {decimal}: must be > 1")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def implied_prob(american: int | float) -> float:
    """Implied probability including vig. -110 -> 0.5238."""
    return 1.0 / to_decimal(american)


def breakeven_prob(american: int | float) -> float:
    """Hit rate needed to break even at this price (same as implied probability)."""
    return implied_prob(american)


# --- De-vig ------------------------------------------------------------------


def devig_two_way(
    over: int | float, under: int | float, method: str = "multiplicative"
) -> tuple[float, float]:
    """Remove the vig from a two-way market. Returns (p_over, p_under) summing to 1.

    multiplicative: scale implied probabilities so they sum to 1.
    power: find k with p_over**k + p_under**k = 1; shifts more vig onto the longshot.
    """
    p_o, p_u = implied_prob(over), implied_prob(under)
    if method == "multiplicative":
        total = p_o + p_u
        return p_o / total, p_u / total
    if method == "power":
        lo, hi = 1.0, 10.0
        if p_o + p_u < 1.0:  # negative overround (rare, stale lines): solve on k < 1
            lo, hi = 0.01, 1.0
        for _ in range(200):
            k = (lo + hi) / 2
            s = p_o**k + p_u**k
            if abs(s - 1.0) < 1e-12:
                break
            # s decreases as k increases (probabilities < 1)
            if s > 1.0:
                lo = k
            else:
                hi = k
        a, b = p_o**k, p_u**k
        return a / (a + b), b / (a + b)
    raise ValueError(f"unknown de-vig method {method!r}")


# --- Value and sizing --------------------------------------------------------


def blend(p_model: float, p_market: float | None, model_weight: float) -> float:
    """Fair probability: weighted blend of model and market (the de-vigged consensus of
    FanDuel, Bovada and Unibet; see pricing/market.py).

    With no market probability, returns the model probability unchanged; the caller
    applies model-only shrinkage and the higher EV threshold.
    """
    if p_market is None:
        return p_model
    return model_weight * p_model + (1.0 - model_weight) * p_market


def shrink_toward(p_model: float, p_anchor: float, shrink: float) -> float:
    """Pull p_model a fraction `shrink` of the way toward p_anchor (0 = no change)."""
    return p_model + shrink * (p_anchor - p_model)


def ev_decimal(p: float, decimal: float) -> float:
    """Expected profit per $1 staked at decimal odds."""
    return p * (decimal - 1.0) - (1.0 - p)


def ev(p: float, american: int | float) -> float:
    """Expected profit per $1 staked at American odds. ev(0.75, -250) -> 0.05."""
    return ev_decimal(p, to_decimal(american))


def kelly_decimal(p: float, decimal: float, fraction: float = 1.0) -> float:
    """Kelly stake as a fraction of bankroll; 0 when EV <= 0."""
    edge = ev_decimal(p, decimal)
    if edge <= 0:
        return 0.0
    return fraction * edge / (decimal - 1.0)


def kelly(p: float, american: int | float, fraction: float = 1.0) -> float:
    return kelly_decimal(p, to_decimal(american), fraction)


def ev_with_push(p_win: float, p_lose: float, decimal: float) -> float:
    """Expected profit per $1 when a push (whole-number line) returns the stake:
    p_win (d - 1) - p_lose. Equals ev_decimal when p_win + p_lose = 1."""
    return p_win * (decimal - 1.0) - p_lose


def kelly_with_push(p_win: float, p_lose: float, decimal: float, fraction: float = 1.0) -> float:
    """Kelly fraction with a push outcome. Maximizing p log(1 + b f) + q log(1 - f), with
    b = d - 1, gives f = (b p - q) / (b (p + q)); with no push (p + q = 1) this is the usual
    (b p - q) / b. 0 when EV <= 0."""
    b = decimal - 1.0
    edge = b * p_win - p_lose
    if edge <= 0 or p_win + p_lose <= 0:
        return 0.0
    return fraction * edge / (b * (p_win + p_lose))


# --- Parlays and boosts ------------------------------------------------------


def parlay_decimal(decimals: Iterable[float]) -> float:
    """Multi-game parlay payout: product of leg decimal odds."""
    legs = list(decimals)
    if not legs:
        raise ValueError("parlay needs at least one leg")
    return prod(legs)


def boosted_decimal(decimal: float, boost: float) -> float:
    """Apply a DraftKings-style profit boost: profit is multiplied by (1 + boost)."""
    if boost < 0:
        raise ValueError("boost must be >= 0")
    return 1.0 + (decimal - 1.0) * (1.0 + boost)


def independent_joint_prob(probs: Iterable[float]) -> float:
    """Joint hit probability for legs in different games (assumed independent)."""
    return prod(probs)


# --- Closing line value ------------------------------------------------------


def clv(p_close_fair: float, decimal_taken: float) -> float:
    """CLV = fair closing probability minus the probability implied by the price taken.

    Positive means the bet was placed at a better price than the fair close.
    """
    return p_close_fair - 1.0 / decimal_taken
