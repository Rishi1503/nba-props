"""Real bets and the go-live ramp (plan, Stage 9 "After go-live").

`place` records that you bet a pick: a separate real-money row with the price you got and
your stake (the paper pick stays, so paper tracking continues). The ramp: half the engine's
stakes for the first 200 real bets, then full quarter Kelly if real-money CLV holds
(non-negative); otherwise stay at half and say so.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select, text

from nbaprops.db.models import Pick


class AlreadyPlaced(RuntimeError):
    pass


def place(session, pick_id: int, price: int, stake: float, now: datetime) -> Pick:
    paper = session.get(Pick, pick_id)
    if paper is None:
        raise KeyError(f"no pick {pick_id}")
    exists = session.scalar(
        select(Pick.pick_id).where(
            Pick.game_id == paper.game_id, Pick.player_id == paper.player_id, Pick.mode == "real"
        )
    )
    if exists:
        raise AlreadyPlaced(
            f"a real bet on this player in this game is already recorded (pick {exists})"
        )
    real = Pick(
        **{
            c: getattr(paper, c)
            for c in (
                "run_id",
                "game_id",
                "player_id",
                "market",
                "line",
                "line_type",
                "side",
                "tier",
                "dk_price",
                "fair_prob",
                "p_model",
                "p_consensus",
                "kelly",
                "ev",
            )
        },
        mode="real",
        flagged_ts=now,
        price_placed=int(price),
        stake=Decimal(str(round(stake, 2))),
    )
    session.add(real)
    session.flush()
    return real


def ramp(session, ramp_bets: int, ramp_factor: float) -> tuple[float, str]:
    """(stake multiplier for real bets, why)."""
    n, clv = session.execute(
        text("""SELECT count(*), avg(clv) FROM picks WHERE mode = 'real'
                                     AND tier <> 'boost_leg'""")
    ).one()
    if n < ramp_bets:
        return ramp_factor, f"ramp: {n} of {ramp_bets} real bets placed, half stakes"
    if clv is None or float(clv) < 0:
        return ramp_factor, f"ramp held: real CLV {float(clv or 0):+.2%} after {n} bets"
    return 1.0, f"full stakes: real CLV {float(clv):+.2%} after {n} bets"
