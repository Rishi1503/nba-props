"""Retune on paper data only (plan, Stage 9: never on the holdout season).

Replays the logged pricing runs (candidate_log) with the engine's own `decide` and
`select_and_size`, first flag wins across each day's runs, for a grid of blend weights and
EV floors. The setting with the best average CLV on the earlier 60% of dates is then judged,
with the current config, on the later 40%: a recommendation is only as good as it does on
dates it never saw. Also reports the blend log-loss check on the logged lines. Never changes
the config: applying a result is your call.
"""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from nbaprops.backtest.metrics import log_loss
from nbaprops.pricing import engine
from nbaprops.pricing.market import board_per_game, fair_close

WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
EV_MINS = (0.02, 0.03, 0.04, 0.05, 0.06)
MIN_BETS = 30


def load_logged(session, start: date, end: date) -> pd.DataFrame:
    """Two-way candidates on finished games, with each prop's outcome."""
    df = pd.read_sql_query(
        text("""
        SELECT c.*, g.game_date, g.tip_time,
               CASE c.market WHEN 'pts' THEN l.pts WHEN 'reb' THEN l.reb WHEN 'ast' THEN l.ast
                    WHEN 'fg3m' THEN l.fg3m WHEN 'pra' THEN l.pts + l.reb + l.ast END AS actual
        FROM candidate_log c JOIN games g USING (game_id)
        JOIN player_game_logs l ON l.game_id = c.game_id AND l.player_id = c.player_id
        WHERE c.kind = 'two_way' AND g.status = 'final' AND g.game_date BETWEEN :a AND :b
          AND NOT l.dnp AND l.minutes > 0"""),
        session.connection(),
        params={"a": start, "b": end},
    )
    if df.empty:
        return df
    inputs = pd.json_normalize(df["inputs"]).set_index(df.index)
    df = pd.concat(
        [df.drop(columns=["inputs", "side", "ev", "p_fair", "flag", "has_consensus"]), inputs],
        axis=1,
    )
    df["actual"] = df["actual"].astype(float)
    return df


def close_probs(session, logged: pd.DataFrame) -> pd.Series:
    """Fair closing P(over) per logged prop (consensus close at the line, else DraftKings')."""
    ids = logged["game_id"].unique().tolist()
    odds = pd.read_sql_query(
        text("SELECT * FROM odds_snapshots WHERE game_id = ANY(:g) AND player_id IS NOT NULL"),
        session.connection(),
        params={"g": ids},
    )
    props = logged[engine.KEY].drop_duplicates().assign(side="over")
    if odds.empty:
        return pd.Series(np.nan, index=pd.MultiIndex.from_frame(props[engine.KEY]))
    odds["line"] = odds["line"].astype(float)
    odds["player_id"] = odds["player_id"].astype("int64")
    tips = logged.drop_duplicates("game_id").set_index("game_id")["tip_time"]
    close = board_per_game(
        odds,
        {g: pd.Timestamp(t).tz_convert("UTC") - timedelta(microseconds=1) for g, t in tips.items()},
    )
    p = fair_close(close, props)
    return pd.Series(p, index=pd.MultiIndex.from_frame(props[engine.KEY]))


def replay(logged: pd.DataFrame, rules: engine.Rules) -> pd.DataFrame:
    """Picks the engine would have made under `rules`, run by run, first flag wins."""
    picks = pd.DataFrame()
    for _, run in logged.sort_values("logged_ts").groupby("logged_ts", sort=True):
        res = engine.decide(run.reset_index(drop=True), rules)
        committed = picks[["game_id", "player_id", "stake", "game_date"]] if len(picks) else None
        new = engine.select_and_size(res, 1.0, rules, committed=committed)
        if len(new):
            picks = pd.concat([picks, new], ignore_index=True)
    return picks


def score(picks: pd.DataFrame, closes: pd.Series) -> dict:
    if picks.empty:
        return {"bets": 0}
    key = list(zip(*(picks[c] for c in engine.KEY), strict=True))
    p_over = closes.reindex(key).to_numpy(float)
    p_side = np.where(picks["side"] == "over", p_over, 1 - p_over)
    clv = p_side - 1 / picks["decimal"].to_numpy(float)
    won = np.where(
        picks["side"] == "over", picks["actual"] > picks["line"], picks["actual"] < picks["line"]
    )
    push = picks["actual"] == picks["line"]
    profit = np.where(push, 0.0, np.where(won, picks["decimal"] - 1, -1.0))
    ok = np.isfinite(clv)
    return {
        "bets": int(len(picks)),
        "avg_clv": round(float(clv[ok].mean()), 4) if ok.any() else None,
        "roi_flat": round(float(profit.mean()), 4),
    }


def tune(session, start: date, end: date, rules: engine.Rules) -> dict:
    logged = load_logged(session, start, end)
    if logged.empty:
        return {"rows": 0, "note": "no logged candidates on finished games yet"}
    return tune_logged(logged, close_probs(session, logged), rules)


def tune_logged(logged: pd.DataFrame, closes: pd.Series, rules: engine.Rules) -> dict:
    """The tuning itself, on logged candidates with outcomes and closing probabilities."""
    dates = sorted(logged["game_date"].unique())
    cut = dates[int(len(dates) * 0.6)] if len(dates) > 1 else dates[-1]
    early, late = logged[logged["game_date"] < cut], logged[logged["game_date"] >= cut]
    grid = []
    for w in WEIGHTS:
        for ev in EV_MINS:
            r = replace(rules, model_weight=w, ev_min=ev)
            grid.append({"model_weight": w, "ev_min": ev, **score(replay(early, r), closes)})
    eligible = [g for g in grid if g["bets"] >= MIN_BETS and g.get("avg_clv") is not None]
    best = max(eligible, key=lambda g: g["avg_clv"]) if eligible else None
    out = {
        "rows": int(len(logged)),
        "dates": [str(dates[0]), str(dates[-1])],
        "split_at": str(cut),
        "grid_first_60pct": grid,
        "current": {
            "model_weight": rules.model_weight,
            "ev_min": rules.ev_min,
            "later_40pct": score(replay(late, rules), closes),
        },
    }
    if best:
        chosen = replace(rules, model_weight=best["model_weight"], ev_min=best["ev_min"])
        out["chosen"] = {
            "model_weight": best["model_weight"],
            "ev_min": best["ev_min"],
            "later_40pct": score(replay(late, chosen), closes),
        }
    else:
        out["note"] = f"no setting had {MIN_BETS}+ bets on the earlier dates: keep collecting"
    out["blend"] = blend_check(logged, cut)
    return out


def blend_check(logged: pd.DataFrame, cut) -> dict:
    """Log loss of P(over) for blends of model and consensus, weight chosen on the earlier
    dates and judged on the later ones (props with a consensus, pushes excluded)."""
    d = logged[logged["p_over_cons"].notna() & (logged["actual"] != logged["line"])]
    d = d.drop_duplicates(engine.KEY, keep="last")
    if d.empty:
        return {"rows": 0}
    hit = (d["actual"] > d["line"]).astype(float)
    first, second = d["game_date"] < cut, d["game_date"] >= cut

    def loss(mask, w):
        p = w * d.loc[mask, "p_over"] + (1 - w) * d.loc[mask, "p_over_cons"]
        return log_loss(p, hit[mask])

    curve = (
        {str(w): round(loss(first, w), 5) for w in np.round(np.arange(0, 1.01, 0.1), 2)}
        if first.any()
        else {}
    )
    best = float(min(curve, key=curve.get)) if curve else None
    return {
        "rows": int(len(d)),
        "curve_first": curve,
        "chosen_weight": best,
        "later": {
            "model_only": round(loss(second, 1.0), 5),
            "consensus_only": round(loss(second, 0.0), 5),
            "chosen": round(loss(second, best), 5),
        }
        if second.any() and best is not None
        else {},
    }
