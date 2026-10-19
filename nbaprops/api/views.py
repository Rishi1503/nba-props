"""What the dashboard reads (plan, Stage 10), as plain functions over a session so each one
is tested directly. All values are JSON-ready (floats, ints, ISO strings)."""

import csv
import io
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from nbaprops.ops import review as rv
from nbaprops.pricing.odds_math import to_decimal


def _q(session, sql: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), session.connection(), params=params)


def jsonable(obj):
    """Nested report structures (dicts and lists of NumPy and pandas values) as plain JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, pd.Timestamp | datetime | date):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if hasattr(obj, "__float__") and not isinstance(obj, int | str):
        return float(obj)
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: NaN -> None, numpy and Decimal -> float/int, times -> ISO strings."""
    out = []
    for r in df.to_dict("records"):
        clean = {}
        for k, v in r.items():
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
                clean[k] = None
            elif isinstance(v, (pd.Timestamp, datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, np.bool_ | bool):
                clean[k] = bool(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif hasattr(v, "__float__") and not isinstance(v, int | str | dict | list):
                clean[k] = float(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


# --- slate, picks, parlays ------------------------------------------------------------------


def slate(session, day: date, now: datetime) -> dict:
    from nbaprops.features.build import latest_statuses

    games = _q(
        session,
        """
        SELECT g.game_id, g.tip_time, g.status, h.abbreviation AS home, a.abbreviation AS away
        FROM games g JOIN teams h ON h.team_id = g.home_team_id
        JOIN teams a ON a.team_id = g.away_team_id
        WHERE g.game_date = :d ORDER BY g.tip_time""",
        d=day,
    )
    inj = _q(
        session,
        """SELECT i.report_ts, i.game_id, i.team_id, i.player_id, i.status,
                                p.full_name FROM injury_snapshots i
                         LEFT JOIN players p USING (player_id)
                         WHERE i.game_id = ANY(:g)""",
        g=games["game_id"].tolist(),
    )
    statuses = latest_statuses(inj, now) if len(inj) else inj
    if len(statuses):
        names = inj.drop_duplicates("player_id").set_index("player_id")["full_name"]
        statuses = statuses.assign(name=statuses["player_id"].map(names))
    projected = _q(
        session,
        """SELECT DISTINCT p.game_id, p.player_id, pl.full_name AS name
                               FROM projections p LEFT JOIN players pl USING (player_id)
                               WHERE p.game_id = ANY(:g) ORDER BY name""",
        g=games["game_id"].tolist(),
    )
    out = []
    for g in _records(games):
        mine = projected[projected["game_id"] == g["game_id"]]
        g["players"] = _records(mine[["player_id", "name"]])
        rows = statuses[statuses["game_id"] == g["game_id"]] if len(statuses) else statuses
        g["injuries"] = _records(rows[["player_id", "name", "status"]]) if len(rows) else []
        g["lineups"] = None  # no reliable free lineup feed (plan, Stage 2b)
        out.append(g)
    return {"date": str(day), "games": out}


def _current_dk(session, keys: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """DraftKings' current price for each pick's prop (same line type; main lines may move)."""
    from nbaprops.pricing.market import board

    if keys.empty:
        return pd.DataFrame()
    odds = _q(
        session,
        """SELECT * FROM odds_snapshots WHERE game_id = ANY(:g)
                          AND book = 'draftkings' AND player_id IS NOT NULL""",
        g=keys["game_id"].unique().tolist(),
    )
    if odds.empty:
        return pd.DataFrame()
    odds["line"] = odds["line"].astype(float)
    return board(odds, now)


def picks(session, day: date, now: datetime, mode: str = "paper") -> list[dict]:
    from nbaprops.pricing.slate import injury_changes

    df = _q(
        session,
        """
        SELECT p.*, pl.full_name AS player, g.tip_time,
               r.pick_id AS real_pick_id, r.price_placed AS real_price, r.stake AS real_stake,
               r.result AS real_result, r.pnl AS real_pnl
        FROM picks p JOIN games g USING (game_id) LEFT JOIN players pl USING (player_id)
        LEFT JOIN picks r ON r.game_id = p.game_id AND r.player_id = p.player_id
                          AND r.mode = 'real' AND p.mode = 'paper'
        WHERE g.game_date = :d AND p.mode = :m AND p.tier <> 'boost_leg'
        ORDER BY p.ev DESC""",
        d=day,
        m=mode,
    )
    if df.empty:
        return []
    df["line"] = df["line"].astype(float)
    # badges: injury news on that game since the flag, and where DraftKings is now
    flagged = df.groupby("game_id")["flagged_ts"].min().rename("as_of_ts").reset_index()
    news = injury_changes(session, flagged, now)
    df["injury_news"] = df["game_id"].map(news)
    cur = _current_dk(session, df, now)
    now_line, now_price = [], []
    for r in df.itertuples(index=False):
        q = (
            cur[
                (cur["game_id"] == r.game_id)
                & (cur["player_id"] == r.player_id)
                & (cur["market"] == r.market)
                & (cur["line_type"] == r.line_type)
            ]
            if len(cur)
            else cur
        )
        if r.line_type != "main" and len(q):
            q = q[q["line"] == r.line]
        if len(q):
            row = q.iloc[-1]
            now_line.append(float(row["line"]))
            now_price.append(row[f"{r.side}_price"])
        else:
            now_line.append(None)
            now_price.append(None)
    df["line_now"], df["price_now"] = now_line, now_price
    df["moved"] = [
        (ln is not None and ln != r.line)
        or (p is not None and not pd.isna(p) and int(p) != int(r.dk_price))
        for ln, p, r in zip(now_line, now_price, df.itertuples(index=False), strict=True)
    ]
    keep = [
        "pick_id",
        "game_id",
        "player_id",
        "player",
        "market",
        "line",
        "line_type",
        "side",
        "tier",
        "mode",
        "dk_price",
        "fair_prob",
        "p_model",
        "p_consensus",
        "ev",
        "stake",
        "kelly",
        "flagged_ts",
        "tip_time",
        "result",
        "pnl",
        "clv",
        "closing_price",
        "injury_news",
        "line_now",
        "price_now",
        "moved",
        "real_pick_id",
        "real_price",
        "real_stake",
        "real_result",
        "real_pnl",
    ]
    return _records(df[keep])


def parlays(session, day: date, mode: str = "paper") -> list[dict]:
    par = _q(
        session,
        """SELECT * FROM parlays WHERE slate_date = :d AND mode = :m
                         ORDER BY ev DESC""",
        d=day,
        m=mode,
    )
    legs = _q(
        session,
        """
        SELECT l.parlay_id, p.pick_id, p.player_id, pl.full_name AS player, p.market, p.line,
               p.side, p.dk_price, p.fair_prob, p.result
        FROM parlay_legs l JOIN picks p USING (pick_id) LEFT JOIN players pl USING (player_id)
        WHERE l.parlay_id = ANY(:ids)""",
        ids=par["parlay_id"].tolist(),
    )
    out = []
    for p in _records(par):
        mine = legs[legs["parlay_id"] == p["parlay_id"]]
        p["legs"] = _records(mine.drop(columns="parlay_id"))
        d = to_decimal(int(p["dk_price"]))
        boost = p.get("boost_pct") or 0
        p["payout_decimal"] = round(1 + (d - 1) * (1 + boost), 4)
        out.append(p)
    return out


# --- player detail --------------------------------------------------------------------


def pmf_from_grid(grid: dict) -> list[dict]:
    """Histogram from a survival grid {"24.5": P(X > 24.5)}: P(X = k) = S(k - .5) - S(k + .5)."""
    xs = sorted(float(k) for k in grid)
    s = {x: float(grid[f"{x:.1f}"]) for x in xs}
    out = []
    for lo, hi in zip(xs, xs[1:], strict=False):
        out.append({"value": int(lo + 0.5), "p": round(max(s[lo] - s[hi], 0.0), 5)})
    return out


def player_projection(session, player_id: int, day: date) -> dict:
    from nbaprops.ops import bundles

    champ = bundles.champion(session)
    proj = _q(
        session,
        """
        SELECT p.*, r.status FROM projections p JOIN games g USING (game_id)
        JOIN model_runs r USING (run_id)
        WHERE p.player_id = :p AND g.game_date = :d
        ORDER BY (p.run_id = :c) DESC, p.as_of_ts DESC""",
        p=player_id,
        d=day,
        c=champ or "",
    )
    proj = proj.drop_duplicates("market", keep="first")
    name = session.execute(
        text("SELECT full_name FROM players WHERE player_id = :p"), {"p": player_id}
    ).scalar()
    markets = {}
    for r in proj.itertuples(index=False):
        markets[r.market] = {
            "mean": float(r.mean),
            "sd": float(r.sd),
            "run_id": r.run_id,
            "as_of": r.as_of_ts.isoformat(),
            "low_confidence": r.low_confidence,
            "histogram": pmf_from_grid(r.p_over),
            "questionable_scenarios": r.scenarios is not None,
        }
    log = _q(
        session,
        """
        SELECT g.game_date, l.minutes, l.pts, l.reb, l.ast, l.fg3m, l.dnp,
               t.abbreviation AS team
        FROM player_game_logs l JOIN games g USING (game_id)
        JOIN teams t ON t.team_id = l.team_id
        WHERE l.player_id = :p AND g.game_date < :d ORDER BY g.game_date DESC LIMIT 10""",
        p=player_id,
        d=day,
    )
    lines = _q(
        session,
        """
        SELECT o.snapshot_ts, o.market, o.line, o.over_price, o.under_price, o.is_withdrawn
        FROM odds_snapshots o JOIN games g USING (game_id)
        WHERE o.player_id = :p AND g.game_date = :d AND o.book = 'draftkings'
          AND o.line_type = 'main' ORDER BY o.snapshot_ts""",
        p=player_id,
        d=day,
    )
    return {
        "player_id": player_id,
        "name": name,
        "date": str(day),
        "markets": markets,
        "recent_games": _records(log),
        "line_history": _records(lines),
    }


# --- performance -------------------------------------------------------------------------


def performance(
    session,
    start: date,
    end: date,
    mode: str = "paper",
    market: str | None = None,
    tier: str | None = None,
) -> dict:
    from nbaprops.models.calibrate import reliability

    df = rv.load_settled(session, start, end, mode)
    if market:
        df = df[df["market"] == market]
    if tier:
        df = df[df["tier"] == tier]
    out = {
        "filters": {
            "start": str(start),
            "end": str(end),
            "mode": mode,
            "market": market,
            "tier": tier,
        },
        "summary": rv.summarize(df),
    }
    if df.empty:
        return out
    settled = df[df["result"] != "void"]
    out["record"] = {k: int((settled["result"] == k).sum()) for k in ("win", "loss", "push")}
    out["by_market"] = {m: rv.summarize(g) for m, g in df.groupby("market")}
    out["by_ev"] = rv._ev_buckets(df)
    s = settled[settled["result"].isin(["win", "loss"])]
    out["calibration"] = (
        reliability(s["fair_prob"].to_numpy(float), (s["result"] == "win").to_numpy(float))
        if len(s)
        else []
    )
    daily = (
        settled.groupby("game_date")
        .agg(clv=("clv", "mean"), bets=("clv", "size"), pnl=("pnl", "sum"))
        .reset_index()
    )
    daily["cum_pnl"] = daily["pnl"].cumsum()
    out["daily"] = _records(daily)
    return out


# --- other pages ---------------------------------------------------------------------------


def backtests(session) -> list[dict]:
    return _records(
        _q(
            session,
            """SELECT run_id, created_ts, season, holdout,
                                          report->'layer_a'->'targets'->>'passes' AS passes
                                   FROM backtest_reports ORDER BY created_ts DESC""",
        )
    )


def backtest(session, run_id: str) -> dict | None:
    r = session.execute(
        text("SELECT report FROM backtest_reports WHERE run_id = :r"), {"r": run_id}
    ).scalar()
    return r


def export_csv(session, year: int, mode: str = "real") -> str:
    """Every settled bet in the calendar year (plan: tax records), singles and parlays."""
    singles = _q(
        session,
        """
        SELECT g.game_date AS date, 'single' AS kind, pl.full_name AS player, p.market, p.side,
               p.line, COALESCE(p.price_placed, p.dk_price) AS odds, p.stake, p.result, p.pnl
        FROM picks p JOIN games g USING (game_id) LEFT JOIN players pl USING (player_id)
        WHERE p.mode = :m AND p.result IS NOT NULL AND p.tier <> 'boost_leg'
          AND EXTRACT(YEAR FROM g.game_date) = :y""",
        m=mode,
        y=year,
    )
    par = _q(
        session,
        """
        SELECT slate_date AS date, tier || ' parlay' AS kind, NULL AS player, NULL AS market,
               NULL AS side, NULL AS line, COALESCE(price_placed, dk_price) AS odds, stake,
               result, pnl FROM parlays
        WHERE mode = :m AND result IS NOT NULL AND EXTRACT(YEAR FROM slate_date) = :y""",
        m=mode,
        y=year,
    )
    rows = pd.concat([singles, par], ignore_index=True).sort_values("date", kind="stable")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        ["date", "kind", "player", "market", "side", "line", "odds", "stake", "result", "pnl"]
    )
    for r in rows.itertuples(index=False):
        w.writerow(
            [
                r.date,
                r.kind,
                r.player or "",
                r.market or "",
                r.side or "",
                "" if pd.isna(r.line) else r.line,
                r.odds,
                r.stake,
                r.result,
                r.pnl,
            ]
        )
    return buf.getvalue()


def bankroll(session) -> list[dict]:
    return _records(_q(session, "SELECT * FROM bankroll_ledger ORDER BY entry_date"))


def settings(session, cfg) -> dict:
    controls = _q(session, "SELECT market, disabled, reason, changed_ts FROM market_controls")
    return {
        "strategy": cfg.model_dump(),
        "markets": _records(controls),
        "as_of": datetime.now(UTC).isoformat(),
    }


def bets(session, start: date, end: date, mode: str = "real") -> dict:
    """The bet tracker: singles and parlays in a date range, settled or open, in date order,
    with running P&L over the settled ones."""
    singles = _q(
        session,
        """
        SELECT p.pick_id AS id, 'single' AS kind, g.game_date AS date, pl.full_name AS player,
               p.market, p.side, p.line, COALESCE(p.price_placed, p.dk_price) AS odds,
               p.stake, p.result, p.pnl, p.clv, p.tier
        FROM picks p JOIN games g USING (game_id) LEFT JOIN players pl USING (player_id)
        WHERE p.mode = :m AND p.tier <> 'boost_leg' AND g.game_date BETWEEN :a AND :b
          AND p.stake > 0""",
        m=mode,
        a=start,
        b=end,
    )
    par = _q(
        session,
        """
        SELECT parlay_id AS id, tier || ' parlay' AS kind, slate_date AS date, NULL AS player,
               NULL AS market, NULL AS side, NULL AS line, COALESCE(price_placed, dk_price) AS odds,
               stake, result, pnl, NULL AS clv, tier
        FROM parlays WHERE mode = :m AND slate_date BETWEEN :a AND :b""",
        m=mode,
        a=start,
        b=end,
    )
    rows = pd.concat([singles, par], ignore_index=True)
    if rows.empty:
        return {"bets": [], "totals": {"staked": 0.0, "pnl": 0.0, "open": 0}}
    rows = rows.sort_values(["date", "kind", "id"], kind="stable").reset_index(drop=True)
    pnl = rows["pnl"].astype(float).fillna(0.0)
    rows["running_pnl"] = pnl.cumsum().round(2)
    settled = rows["result"].notna()
    return {
        "bets": _records(rows),
        "totals": {
            "staked": round(float(rows.loc[settled, "stake"].astype(float).sum()), 2),
            "pnl": round(float(pnl.sum()), 2),
            "open": int((~settled).sum()),
            "record": {
                k: int((rows["result"] == k).sum()) for k in ("win", "loss", "push", "void")
            },
        },
    }
