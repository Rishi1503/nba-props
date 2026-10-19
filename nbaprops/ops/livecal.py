"""Monthly live recalibration: a second calibration layer fit on live results.

The champion's calibrator was fit on backtest predictions; live results may drift (and the
weekly retrain carries the calibrator over). Each month, per market, an isotonic curve is fit
on the logged model P(over) at DraftKings' lines against what happened, in time order, and
kept only if it beats no adjustment on the later 30% (models.calibrate.Calibrator.fit).

Logged probabilities already include the current live curve, so a new curve is composed
with the old one (new(old(x))) rather than replacing it.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from nbaprops.db.models import LiveCalibration
from nbaprops.models.calibrate import Calibrator

GRID = np.round(np.linspace(0, 1, 201), 4)


def compose(new: tuple, old: dict | None) -> dict:
    """The curve x -> new(old(x)) on a dense grid (identity when there's no old curve)."""
    x = GRID
    inner = np.interp(x, old["x"], old["y"]) if old else x
    return {"x": x.tolist(), "y": np.round(np.interp(inner, new[0], new[1]), 6).tolist()}


def fit_live(session, start: date, end: date, now: datetime) -> dict:
    """{market: n used} for markets whose curve was updated."""
    d = pd.read_sql_query(
        text("""
        SELECT c.market, c.line, c.inputs->>'p_over' AS p_over, c.logged_ts, g.game_date,
               CASE c.market WHEN 'pts' THEN l.pts WHEN 'reb' THEN l.reb WHEN 'ast' THEN l.ast
                    WHEN 'fg3m' THEN l.fg3m WHEN 'pra' THEN l.pts + l.reb + l.ast END AS actual,
               c.game_id, c.player_id
        FROM candidate_log c JOIN games g USING (game_id)
        JOIN player_game_logs l ON l.game_id = c.game_id AND l.player_id = c.player_id
        WHERE c.kind = 'two_way' AND g.status = 'final' AND NOT l.dnp AND l.minutes > 0
          AND g.game_date BETWEEN :a AND :b AND c.logged_ts < g.tip_time"""),
        session.connection(),
        params={"a": start, "b": end},
    )
    if d.empty:
        return {}
    d["p_over"] = d["p_over"].astype(float)
    d = d.dropna(subset=["p_over"])
    # one row per prop: the last pricing before tip
    d = d.sort_values("logged_ts").drop_duplicates(
        ["game_id", "player_id", "market", "line"], keep="last"
    )
    d = d[d["actual"].astype(float) != d["line"]].sort_values(["game_date", "logged_ts"])
    hit = (d["actual"].astype(float) > d["line"]).astype(float)
    cal = Calibrator.fit(
        {m: g["p_over"].to_numpy() for m, g in d.groupby("market")},
        {m: hit[g.index].to_numpy() for m, g in d.groupby("market")},
    )
    old = {
        m: c for m, c in session.execute(text("SELECT market, curve FROM live_calibration")).all()
    }
    updated = {}
    for m, curve in cal.curves.items():
        new = compose(curve, old.get(m))
        n = int((d["market"] == m).sum())
        session.execute(
            insert(LiveCalibration)
            .values(market=m, curve=new, n=n, fitted_ts=now)
            .on_conflict_do_update(
                index_elements=["market"], set_={"curve": new, "n": n, "fitted_ts": now}
            )
        )
        updated[m] = n
    return updated
