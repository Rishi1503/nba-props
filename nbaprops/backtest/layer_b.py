"""Layer B: betting results from logged odds (plan, Stage 6).

`replay` prices every DraftKings prop the model covers at each game's flag time (default
10:00 AM New York on game day, the first daily run; never later than 10 minutes before tip),
applies the Stage 7 rules through pricing.engine, grades picks from box scores, and measures
CLV against the fair close (pricing.market.fair_close). Stakes are fractions of a fixed
bankroll of 1, so ROI isn't distorted by compounding.

`blend_curve` answers "does blending model and market reduce log loss?" honestly: the
weight is chosen on the first half of the dates and reported on the second half.

Guardrails from the plan: fewer than 500 settled bets get a warning that ROI is not yet
meaningful, and nothing here tunes a threshold on the data it reports.
"""

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nbaprops.backtest.metrics import log_loss
from nbaprops.pricing import engine, parlays
from nbaprops.pricing.market import board_per_game, consensus, fair_close

NY = ZoneInfo("America/New_York")
MIN_BETS = 500  # plan: ROI from fewer bets is not meaningful
EV_BUCKETS = [(0.03, 0.05), (0.05, 0.08), (0.08, np.inf)]


def flag_times(games: pd.DataFrame, at: str = "10:00") -> dict:
    hh, mm_ = (int(x) for x in at.split(":"))
    out = {}
    for g in games.itertuples(index=False):
        if pd.isna(g.tip_time):
            continue
        tip = pd.Timestamp(g.tip_time).tz_convert("UTC")
        local = pd.Timestamp(g.game_date).replace(hour=hh, minute=mm_).tz_localize(NY)
        out[g.game_id] = min(local.tz_convert("UTC"), tip - timedelta(minutes=10))
    return out


def close_times(games: pd.DataFrame) -> dict:
    """The close is the board just before tip."""
    return {
        g.game_id: pd.Timestamp(g.tip_time).tz_convert("UTC") - timedelta(microseconds=1)
        for g in games.itertuples(index=False)
        if pd.notna(g.tip_time)
    }


def grade(picks: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """win / loss / push / void (didn't play, as DraftKings voids), and profit per pick."""
    o = outcomes.set_index(["game_id", "player_id"])
    res, profit, actual = [], [], []
    for p in picks.itertuples(index=False):
        key = (p.game_id, p.player_id)
        row = o.loc[key] if key in o.index else None
        played = row is not None and not bool(row.get("dnp", False)) and (row["minutes"] or 0) > 0
        if not played:
            res.append("void")
            profit.append(0.0)
            actual.append(np.nan)
            continue
        val = (row["pts"] + row["reb"] + row["ast"]) if p.market == "pra" else row[p.market]
        actual.append(float(val))
        if val == p.line:
            r = "push"
        elif (val > p.line) == (p.side == "over"):
            r = "win"
        else:
            r = "loss"
        res.append(r)
        profit.append(p.stake * (p.decimal - 1) if r == "win" else -p.stake if r == "loss" else 0.0)
    return picks.assign(result=res, profit=profit, actual=actual)


SCHEDULE = "10:00,15:00,18:00"  # plan, Stage 8: the daily pricing runs (New York)


def replay(
    model: pd.DataFrame,
    odds: pd.DataFrame,
    games: pd.DataFrame,
    outcomes: pd.DataFrame,
    rules: engine.Rules,
    flag_at: str = SCHEDULE,
    grids: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all candidates, graded picks with CLV). Replays each scheduled pricing run in order
    (comma-separated New York times; never later than 10 minutes before tip): each sees the
    board at its own time, and players already picked are skipped with their exposure
    counted, as live (first flag wins). Candidates come from engine.build_candidates, the
    function live pricing uses; `grids` add milestones and shifted consensus."""
    has_grids = grids is not None and len(grids)
    keep_games = set(model["game_id"]) if len(model) else set()
    if has_grids:
        keep_games |= set(grids["game_id"])
    games = games[games["game_id"].isin(keep_games)]
    gd = games.set_index("game_id")["game_date"]
    m = model.assign(game_date=model["game_id"].map(gd)) if len(model) else model
    g = grids.assign(game_date=grids["game_id"].map(gd)) if has_grids else None
    all_cands, picks = [], pd.DataFrame()
    for at in [t.strip() for t in flag_at.split(",") if t.strip()]:
        cands = engine.build_candidates(m, g, board_per_game(odds, flag_times(games, at)), rules)
        if cands.empty:
            continue
        cands = cands.assign(flag_time=at)
        all_cands.append(cands)
        committed = picks[["game_id", "player_id", "stake", "game_date"]] if len(picks) else None
        new = engine.select_and_size(cands, 1.0, rules, committed=committed)
        picks = pd.concat([picks, new], ignore_index=True) if len(new) else picks
    cands = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame()
    if picks.empty:
        return cands, picks
    close_board = board_per_game(odds, close_times(games))
    picks["p_close_fair"] = fair_close(close_board, picks, rules.min_consensus_books)
    picks["clv"] = picks["p_close_fair"] - 1 / picks["decimal"]
    return cands, grade(picks, outcomes)


def replay_satellites(picks: pd.DataFrame, rules: engine.Rules) -> pd.DataFrame:
    """Satellite parlays built each day from that day's flagged legs (live rules), graded the
    DraftKings way: a pushed or void leg drops out and the rest pay."""
    out = []
    for day, d in picks.groupby("game_date"):
        d = d.reset_index(drop=True)
        for p in parlays.satellites(
            parlays.legs_from_picks(d), rules, 1.0, rules.max_satellites_per_slate
        ):
            factor = 1.0
            for lg in p.legs:
                r = d.loc[lg.pick_ref, "result"]
                factor *= lg.decimal if r == "win" else 0.0 if r == "loss" else 1.0
            result = "loss" if factor == 0 else "push" if factor == 1 else "win"
            out.append(
                {
                    "game_date": day,
                    "ev": p.ev,
                    "stake": p.stake,
                    "decimal": p.decimal,
                    "result": result,
                    "profit": p.stake * (factor - 1) if factor else -p.stake,
                }
            )
    return pd.DataFrame(out)


def bankroll_growth(picks: pd.DataFrame, sats: pd.DataFrame, rules: engine.Rules) -> dict:
    """Compounded bankroll for singles alone vs singles plus satellites (stakes are fractions
    of the bankroll at bet time). With satellites the daily cap covers both, as live."""

    def path(include_sats: bool) -> dict:
        days = sorted(
            set(picks["game_date"])
            | (set(sats["game_date"]) if include_sats and len(sats) else set())
        )
        bank, peak, worst = 1.0, 1.0, 0.0
        for day in days:
            s = picks[(picks["game_date"] == day) & (picks["result"] != "void")]
            stake, profit = float(s["stake"].sum()), float(s["profit"].sum())
            if include_sats and len(sats):
                p = sats[sats["game_date"] == day]
                stake += float(p["stake"].sum())
                profit += float(p["profit"].sum())
            if stake > rules.max_daily_pct:
                profit *= rules.max_daily_pct / stake
            bank *= 1 + profit
            peak = max(peak, bank)
            worst = max(worst, 1 - bank / peak)
        return {
            "final_bankroll": round(bank, 4),
            "max_drawdown": round(worst, 4),
            "days": len(days),
        }

    return {"singles_only": path(False), "singles_plus_satellites": path(True)}


def _clustered_mean(x: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, c = x[ok], np.asarray(clusters)[ok]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    resid = pd.Series(x - x.mean()).groupby(c).sum()
    return float(x.mean()), float(np.sqrt((resid**2).sum()) / len(x))


def summarize(picks: pd.DataFrame) -> dict:
    """Counts, EV, CLV and ROI, with standard errors clustered by date (same-night results
    move together)."""
    s = picks[picks["result"] != "void"]
    n = int(len(s))
    out = {"bets": n, "voided": int((picks["result"] == "void").sum())}
    if not n:
        return out
    # ROI over staked picks; hit rate and CLV over every settled pick (zero-stake standard
    # flags are tracked, not bet)
    bet = s[s["stake"] > 0]
    staked = float(bet["stake"].sum())
    days = s["game_date"].astype(str).to_numpy()
    roi, roi_se = _clustered_mean(
        (bet["profit"] / bet["stake"]).to_numpy(), bet["game_date"].astype(str).to_numpy()
    )
    clv, clv_se = _clustered_mean(s["clv"].to_numpy(float), days)
    wins, losses = int((s["result"] == "win").sum()), int((s["result"] == "loss").sum())
    out.update(
        {
            "staked": round(staked, 4),
            "staked_bets": int(len(bet)),
            "profit": round(float(s["profit"].sum()), 4),
            "roi_staked": round(float(s["profit"].sum()) / staked, 4) if staked else None,
            "roi_per_bet": round(roi, 4),
            "roi_per_bet_se": round(roi_se, 4),
            "hit_rate": round(wins / (wins + losses), 4) if wins + losses else None,
            "breakeven_rate": round(float((1 / s["decimal"]).mean()), 4),
            "avg_ev": round(float(s["ev"].mean()), 4),
            "avg_clv": round(clv, 4),
            "avg_clv_se": round(clv_se, 4),
            "clv_coverage": round(float(s["clv"].notna().mean()), 4),
            "pushes": int((s["result"] == "push").sum()),
        }
    )
    if n < MIN_BETS:
        out["warning"] = (
            f"only {n} settled bets: ROI is not meaningful below {MIN_BETS} "
            "(plan, Stage 6); judge by CLV"
        )
    return out


def report(picks: pd.DataFrame, cands: pd.DataFrame) -> dict:
    if picks.empty:
        return {"overall": {"bets": 0}, "candidates": int(len(cands))}
    out = {
        "candidates": int(len(cands)),
        "flagged": int(cands["flag"].sum()),
        "overall": summarize(picks),
        "by_market": {},
        "by_ev": {},
        "by_source": {},
    }
    for m, g in picks.groupby("market"):
        out["by_market"][m] = summarize(g)
    for lo, hi in EV_BUCKETS:
        g = picks[(picks["ev"] >= lo) & (picks["ev"] < hi)]
        out["by_ev"][f"{lo:.0%}-{'' if hi == np.inf else f'{hi:.0%}'}"] = summarize(g)
    out["by_source"]["consensus"] = summarize(picks[picks["has_consensus"]])
    out["by_source"]["model_only"] = summarize(picks[~picks["has_consensus"]])
    out["high_confidence"] = summarize(picks[picks["high_confidence"]])
    out["standard"] = summarize(picks[~picks["high_confidence"].astype(bool)])
    if "flag_time" in picks:
        out["by_flag_time"] = {t: summarize(g) for t, g in picks.groupby("flag_time")}
    return out


def blend_curve(
    model: pd.DataFrame,
    odds: pd.DataFrame,
    games: pd.DataFrame,
    outcomes: pd.DataFrame,
    min_books: int = 2,
    weights: tuple = tuple(np.round(np.arange(0, 1.01, 0.1), 2)),
) -> dict:
    """Log loss of P(over) at closing consensus lines for blends of model and consensus.
    The weight is chosen on the first half of the dates and judged on the second half."""
    games = games[games["game_id"].isin(model["game_id"])]
    close = board_per_game(odds, close_times(games))
    cons = consensus(close, min_books)
    j = cons.merge(model, on=["game_id", "player_id", "market", "line"], how="inner")
    if j.empty:
        return {"rows": 0}
    o = outcomes.set_index(["game_id", "player_id"])
    keep, hit = [], []
    for r in j.itertuples(index=False):
        key = (r.game_id, r.player_id)
        if key not in o.index or bool(o.loc[key].get("dnp", False)):
            keep.append(False)
            hit.append(np.nan)
            continue
        row = o.loc[key]
        val = (row["pts"] + row["reb"] + row["ast"]) if r.market == "pra" else row[r.market]
        keep.append(val != r.line)  # pushes say nothing about over vs under
        hit.append(float(val > r.line))
    j = j.assign(hit=hit)[keep]
    if j.empty:
        return {"rows": 0}
    j = j.assign(game_date=j["game_id"].map(games.set_index("game_id")["game_date"]))
    cut = j["game_date"].sort_values().iloc[len(j) // 2]
    first, second = j[j["game_date"] < cut], j[j["game_date"] >= cut]

    def loss(df, w):
        return log_loss(w * df["p_over"] + (1 - w) * df["p_over_cons"], df["hit"])

    curve = {str(w): round(loss(first, w), 5) for w in weights} if len(first) else {}
    best = float(min(curve, key=curve.get)) if curve else 0.3
    return {
        "rows": int(len(j)),
        "first_half_rows": int(len(first)),
        "second_half_rows": int(len(second)),
        "curve_first_half": curve,
        "chosen_weight": best,
        "second_half": {
            "model_only": round(loss(second, 1.0), 5),
            "consensus_only": round(loss(second, 0.0), 5),
            "chosen_weight": round(loss(second, best), 5),
            "plan_default_0.3": round(loss(second, 0.3), 5),
        }
        if len(second)
        else {},
    }


# --- wiring for the backtest command ---------------------------------------------------


def _times(flag_at: str) -> list[str]:
    return [t.strip() for t in flag_at.split(",") if t.strip()]


def lines_to_price(
    odds: pd.DataFrame, games: pd.DataFrame, flag_at: str = SCHEDULE, min_books: int = 2
) -> pd.DataFrame:
    """The lines Layer B needs the model to price: every DraftKings two-way line on each
    game's board at each scheduled run (lines move during the day), plus every consensus line
    at the close (for blend_curve)."""
    from nbaprops.pricing.market import TARGET_BOOK, two_way

    cols = ["game_id", "player_id", "market", "line"]
    parts = []
    for at in _times(flag_at):
        flag = board_per_game(odds, flag_times(games, at))
        parts.append(two_way(flag[flag["book"] == TARGET_BOOK])[cols])
    parts.append(consensus(board_per_game(odds, close_times(games)), min_books)[cols])
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    out["player_id"] = out["player_id"].astype("int64")
    return out.reset_index(drop=True)


def section(
    priced: pd.DataFrame,
    odds: pd.DataFrame,
    games: pd.DataFrame,
    outcomes: pd.DataFrame,
    rules: engine.Rules,
    flag_at: str = SCHEDULE,
    grids: pd.DataFrame | None = None,
) -> dict:
    """The Layer B part of a backtest report."""
    if odds.empty or (priced.empty and (grids is None or grids.empty)):
        return {"skipped": "no logged odds for this season yet (Stage 2 collects them)"}
    cands, picks = replay(priced, odds, games, outcomes, rules, flag_at, grids)
    out = {
        "flag_times_new_york": flag_at,
        "rules": rules.__dict__,
        "bets": report(picks, cands),
        "blend": blend_curve(priced, odds, games, outcomes, rules.min_consensus_books)
        if len(priced)
        else {"rows": 0},
    }
    if len(picks):
        sats = replay_satellites(picks, rules)
        out["satellites"] = summarize_parlays(sats)
        out["growth"] = bankroll_growth(picks, sats, rules)
        out["by_line_type"] = {lt: summarize(g) for lt, g in picks.groupby("line_type")}
    return out


def summarize_parlays(sats: pd.DataFrame) -> dict:
    if sats.empty:
        return {"parlays": 0}
    settled = sats[sats["result"] != "push"]
    staked = float(sats["stake"].sum())
    out = {
        "parlays": int(len(sats)),
        "hit_rate": round(float((settled["result"] == "win").mean()), 4) if len(settled) else None,
        "avg_ev": round(float(sats["ev"].mean()), 4),
        "roi_staked": round(float(sats["profit"].sum()) / staked, 4) if staked else None,
    }
    if len(sats) < MIN_BETS:
        out["warning"] = f"only {len(sats)} parlays: results are not yet meaningful"
    return out


def milestone_keys(
    odds: pd.DataFrame, games: pd.DataFrame, flag_at: str = SCHEDULE
) -> pd.DataFrame:
    """Players and stats with a DraftKings milestone on any scheduled run's board: the
    walk-forward stores full grids for these so Layer B can price milestones like live."""
    from nbaprops.pricing.market import TARGET_BOOK

    parts = []
    for at in _times(flag_at):
        b = board_per_game(odds, flag_times(games, at))
        parts.append(b[(b["book"] == TARGET_BOOK) & (b["line_type"] == "milestone")])
    ms = pd.concat(parts, ignore_index=True)
    out = ms[["game_id", "player_id", "market"]].drop_duplicates().reset_index(drop=True)
    out["player_id"] = out["player_id"].astype("int64")
    return out


def load_market(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(odds snapshots, games, outcomes) for one season from Postgres."""
    from sqlalchemy import text

    from nbaprops.db.base import session_scope

    with session_scope() as s:
        conn = s.connection()
        params = {"s": season}
        games = pd.read_sql_query(
            text("SELECT game_id, game_date, tip_time, status FROM games WHERE season = :s"),
            conn,
            params=params,
        )
        odds = pd.read_sql_query(
            text(
                "SELECT o.* FROM odds_snapshots o JOIN games g USING (game_id) "
                "WHERE g.season = :s AND o.player_id IS NOT NULL"
            ),
            conn,
            params=params,
        )
        outcomes = pd.read_sql_query(
            text(
                "SELECT l.game_id, l.player_id, l.minutes, l.dnp, l.pts, l.reb, l.ast, l.fg3m "
                "FROM player_game_logs l JOIN games g USING (game_id) WHERE g.season = :s"
            ),
            conn,
            params=params,
        )
    odds["line"] = odds["line"].astype(float)
    return odds, games, outcomes
