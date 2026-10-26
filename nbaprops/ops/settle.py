"""Morning settlement (plan, Stage 8 8:00 AM job; Stage 9 CLV definition).

Grades every unsettled pick whose game is final and whose box scores are loaded (a game with
no box scores yet waits; a player with no line in a loaded box score didn't play: void).
Idempotent: settled picks are never touched again, so it can run on every cycle.

  pick:   result (win/loss/push/void), P&L at the price taken (price_placed for real bets,
          else the flagged DraftKings price), DraftKings' closing price at the same line and
          side, and CLV = fair close probability - 1 / decimal taken (fair close: consensus
          no-vig close at the line, else DraftKings' own; milestones have none).
  parlay: once every leg is graded; a pushed or void leg drops out and the rest pay, and a
          boost multiplies the profit. A same-game parlay with a pushed or void leg is
          re-priced by DraftKings itself, so it is marked void here for you to correct.
  real-money P&L is added to the bankroll ledger (dated the settlement day); paper P&L never
  touches the bankroll.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import select, text

from nbaprops.db.models import BankrollEntry, Parlay, ParlayLeg, Pick
from nbaprops.pricing.market import TARGET_BOOK, board_per_game, fair_close, two_way
from nbaprops.pricing.odds_math import boosted_decimal, to_decimal


@dataclass
class SettleResult:
    picks: int = 0
    parlays: int = 0
    record: dict = field(default_factory=lambda: {"win": 0, "loss": 0, "push": 0, "void": 0})
    pnl: dict = field(default_factory=lambda: {"paper": 0.0, "real": 0.0})
    clv: list = field(default_factory=list)

    def summary(self) -> str:
        r = self.record
        clv = f", average CLV {np.mean(self.clv):+.2%}" if self.clv else ""
        return (
            f"Settled {self.picks} picks ({r['win']}-{r['loss']}, {r['push']} push, "
            f"{r['void']} void) and {self.parlays} parlays. P&L paper "
            f"{self.pnl['paper']:+.2f}, real {self.pnl['real']:+.2f}{clv}."
        )


def _ready_picks(session) -> pd.DataFrame:
    """Unsettled picks on final games that have box scores loaded."""
    return pd.read_sql_query(
        text("""
        SELECT p.*, g.tip_time, g.game_date FROM picks p JOIN games g USING (game_id)
        WHERE p.result IS NULL AND g.status = 'final'
          AND EXISTS (SELECT 1 FROM player_game_logs l WHERE l.game_id = p.game_id)"""),
        session.connection(),
    )


def _outcome(session, game_id: str, player_id: int, market: str) -> float | None:
    row = (
        session.execute(
            text("""SELECT minutes, dnp, pts, reb, ast, fg3m FROM player_game_logs
                                  WHERE game_id = :g AND player_id = :p"""),
            {"g": game_id, "p": player_id},
        )
        .mappings()
        .first()
    )
    if row is None or row["dnp"] or not row["minutes"]:
        return None
    if market == "pra":
        return float(row["pts"] + row["reb"] + row["ast"])
    return float(row[market])


def _closes(session, picks: pd.DataFrame) -> pd.DataFrame:
    """The board just before each pick's tip."""
    ids = picks["game_id"].unique().tolist()
    odds = pd.read_sql_query(
        text("""SELECT * FROM odds_snapshots
                                     WHERE game_id = ANY(:g) AND player_id IS NOT NULL"""),
        session.connection(),
        params={"g": ids},
    )
    if odds.empty:
        return odds
    odds["line"] = odds["line"].astype(float)
    odds["player_id"] = odds["player_id"].astype("int64")
    tips = picks.drop_duplicates("game_id").set_index("game_id")["tip_time"]
    times = {
        g: pd.Timestamp(t).tz_convert("UTC") - timedelta(microseconds=1)
        for g, t in tips.items()
        if pd.notna(t)
    }
    return board_per_game(odds, times)


CLV_SHIFT_MAX = 3.0  # points a closing consensus line may be read across for CLV


def read_across(session, picks: pd.DataFrame, close: pd.DataFrame) -> np.ndarray:
    """Fair close for picks whose line nobody quotes at the close, because the market moved
    past it. Leaving those blank would bias CLV down: they are the picks the market moved
    toward. The closing consensus main line is read across to the pick's line with the
    pick's own projection (market location, model shape), up to CLV_SHIFT_MAX points."""
    from nbaprops.pricing import engine

    out = picks["fair_close"].to_numpy(float).copy()
    todo = np.flatnonzero(np.isnan(out))
    if not len(todo):
        return out
    anchors = engine.consensus_anchors(close, engine.Rules())
    grids = pd.read_sql_query(
        text("""
        SELECT run_id, game_id, player_id, market, p_over FROM projections
        WHERE game_id = ANY(:g)"""),
        session.connection(),
        params={"g": picks["game_id"].iloc[todo].unique().tolist()},
    )
    gl = grids.set_index(["run_id", "game_id", "player_id", "market"])["p_over"]
    for i in todo:
        r = picks.iloc[i]
        k = (r["game_id"], int(r["player_id"]), r["market"])
        gk = (r["run_id"], *k)
        if k not in anchors or gk not in gl.index:
            continue
        a_line, a_p = anchors[k]
        if abs(a_line - float(r["line"])) > CLV_SHIFT_MAX:
            continue
        p_over = engine.shifted_tail(gl.loc[gk], float(r["line"]), a_line, a_p)
        if np.isfinite(p_over):
            out[i] = p_over if r["side"] == "over" else 1 - p_over
    return out


POSTPONED_AFTER = timedelta(hours=36)  # DraftKings voids bets on games that aren't played


def void_unplayed(session, now: datetime, res: "SettleResult") -> None:
    """Picks on games still not final well after their scheduled tip (postponed or
    suspended) are void, as DraftKings settles them. A game only counts as unplayed once
    later games' stats are in while it still has none: a game whose stats simply haven't
    been loaded yet (a missed morning) waits. Before that check, two days without
    `stats-day` voided every pick on perfectly normal games."""
    ids = (
        session.execute(
            text("""
        SELECT p.pick_id FROM picks p JOIN games g USING (game_id)
        WHERE p.result IS NULL AND g.status <> 'final' AND g.tip_time < :cut
          AND NOT EXISTS (SELECT 1 FROM player_game_logs l WHERE l.game_id = g.game_id)
          AND EXISTS (SELECT 1 FROM player_game_logs l2 JOIN games g2 ON g2.game_id = l2.game_id
                      WHERE g2.game_date > g.game_date + 1)"""),
            {"cut": now - POSTPONED_AFTER},
        )
        .scalars()
        .all()
    )
    for pid in ids:
        pick = session.get(Pick, int(pid))
        pick.result, pick.pnl = "void", Decimal("0")
        res.picks += 1
        res.record["void"] += 1
    session.flush()


def unvoid_missing_stats(session) -> tuple[int, int]:
    """Reopen picks voided only because their game's stats weren't loaded yet (the old
    36-hour rule), and the parlays they're in. A real did-not-play void needs the game's
    stats, so it is never touched; a real postponement gets voided again by the current
    rule once later games' stats are in."""
    ids = list(
        session.execute(
            text("""
        SELECT p.pick_id FROM picks p
        WHERE p.result = 'void'
          AND NOT EXISTS (SELECT 1 FROM player_game_logs l WHERE l.game_id = p.game_id)""")
        )
        .scalars()
        .all()
    )
    if not ids:
        return 0, 0
    pars = list(
        session.execute(
            text("""
        SELECT DISTINCT l.parlay_id FROM parlay_legs l JOIN parlays p USING (parlay_id)
        WHERE l.pick_id = ANY(:ids) AND p.result IS NOT NULL"""),
            {"ids": ids},
        )
        .scalars()
        .all()
    )
    session.execute(
        text("""UPDATE picks SET result = NULL, pnl = NULL, closing_price = NULL,
                            clv = NULL WHERE pick_id = ANY(:ids)"""),
        {"ids": ids},
    )
    if pars:
        session.execute(
            text("UPDATE parlays SET result = NULL, pnl = NULL WHERE parlay_id = ANY(:p)"),
            {"p": pars},
        )
    return len(ids), len(pars)


def settle(session, today: date, now: datetime) -> SettleResult:
    res = SettleResult()
    void_unplayed(session, now, res)
    picks = _ready_picks(session)
    if len(picks):
        picks["line"] = picks["line"].astype(float)
        picks["player_id"] = picks["player_id"].astype("int64")
        close = _closes(session, picks)
        picks["fair_close"] = (
            fair_close(close, picks) if len(close) else np.full(len(picks), np.nan)
        )
        if len(close):
            picks["fair_close"] = read_across(session, picks, close)
        dk = two_way(close[close["book"] == TARGET_BOOK]) if len(close) else pd.DataFrame()
        for r in picks.itertuples(index=False):
            actual = _outcome(session, r.game_id, int(r.player_id), r.market)
            taken = int(r.price_placed) if pd.notna(r.price_placed) else int(r.dk_price)
            d = to_decimal(taken)
            stake = float(r.stake)
            if actual is None:
                result, pnl = "void", 0.0
            elif actual == r.line:
                result, pnl = "push", 0.0
            elif (actual > r.line) == (r.side == "over"):
                result, pnl = "win", stake * (d - 1)
            else:
                result, pnl = "loss", -stake
            closing = None
            if len(dk):
                q = dk[
                    (dk["game_id"] == r.game_id)
                    & (dk["player_id"] == r.player_id)
                    & (dk["market"] == r.market)
                    & (dk["line"] == r.line)
                ]
                if len(q):
                    closing = int(q.iloc[-1][f"{r.side}_price"])
            clv = None if pd.isna(r.fair_close) else float(r.fair_close) - 1 / d
            pick = session.get(Pick, int(r.pick_id))
            pick.result, pick.pnl = result, Decimal(str(round(pnl, 2)))
            pick.closing_price = closing
            pick.clv = None if clv is None else Decimal(str(round(clv, 4)))
            res.picks += 1
            res.record[result] += 1
            res.pnl[r.mode] += pnl
            if clv is not None and r.tier != "boost_leg":
                res.clv.append(clv)
        session.flush()
    _settle_parlays(session, res)
    _ledger(session, today, now, res.pnl["real"], res.picks + res.parlays)
    return res


def _settle_parlays(session, res: SettleResult) -> None:
    open_ = session.scalars(select(Parlay).where(Parlay.result.is_(None))).all()
    for par in open_:
        legs = session.scalars(
            select(Pick)
            .join(ParlayLeg, ParlayLeg.pick_id == Pick.pick_id)
            .where(ParlayLeg.parlay_id == par.parlay_id)
        ).all()
        if not legs or any(lg.result is None for lg in legs):
            continue  # wait for every leg
        stake = float(par.stake)
        boost = float(par.boost_pct or 0)
        if par.same_game:
            if any(lg.result in ("push", "void") for lg in legs):
                result, pnl = "void", 0.0  # DraftKings re-prices SGPs itself: correct by hand
            elif all(lg.result == "win" for lg in legs):
                taken = par.price_placed if par.price_placed is not None else par.dk_price
                d = to_decimal(int(taken))
                result, pnl = "win", stake * ((boosted_decimal(d, boost) if boost else d) - 1)
            else:
                result, pnl = "loss", -stake
        else:
            factor = 1.0
            for lg in legs:
                if lg.result == "loss":
                    factor = 0.0
                    break
                if lg.result == "win":
                    factor *= to_decimal(
                        int(lg.price_placed if lg.price_placed is not None else lg.dk_price)
                    )
            if factor == 0.0:
                result, pnl = "loss", -stake
            elif factor == 1.0:
                result, pnl = ("void" if all(lg.result == "void" for lg in legs) else "push"), 0.0
            else:
                result, pnl = "win", stake * (factor - 1) * (1 + boost)
        par.result, par.pnl = result, Decimal(str(round(pnl, 2)))
        res.parlays += 1
        res.pnl[par.mode] += pnl
    session.flush()


def _ledger(session, today: date, now: datetime, real_pnl: float, settled: int) -> None:
    """Add real-money P&L to today's ledger row (created from the latest balance)."""
    if not settled or abs(real_pnl) < 0.005:
        return
    row = session.get(BankrollEntry, today)
    if row is None:
        prev = session.scalar(
            select(BankrollEntry)
            .where(BankrollEntry.entry_date < today)
            .order_by(BankrollEntry.entry_date.desc())
            .limit(1)
        )
        if prev is None:
            return  # no bankroll recorded yet: nothing to update
        row = BankrollEntry(
            entry_date=today,
            bankroll=prev.bankroll,
            settled_pnl=Decimal("0"),
            deposits=Decimal("0"),
            withdrawals=Decimal("0"),
            note=f"settled {now:%H:%M} UTC",
        )
        session.add(row)
    pnl = Decimal(str(round(real_pnl, 2)))
    row.settled_pnl = (row.settled_pnl or Decimal("0")) + pnl
    row.bankroll = row.bankroll + pnl
    session.flush()
