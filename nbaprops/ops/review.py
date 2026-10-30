"""Weekly review (plan, Stage 9; Sunday night): how the paper picks are doing, and why.

For the week and the season to date: CLV, ROI and hit rate (with the breakeven rate for the
prices taken), split by stat, edge size, flag time, over/under, tier, and consensus vs
model-only; predicted EV against realized CLV per edge bucket (if the biggest edges
underperform, raise the threshold or shrink harder); price slippage on real bets; parlays by
tier; and the weekly CLV series the go-live rule needs. CLV standard errors are clustered
by date, since same-night results move together.

`apply_market_rule` switches off any market whose season CLV is negative after 100+ picks
(plan) and says so; `market enable` on the command line turns one back on.
"""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from nbaprops.db.models import MarketControl
from nbaprops.pricing.odds_math import to_decimal

ET = ZoneInfo("America/New_York")
EV_BUCKETS = [(0.03, 0.05), (0.05, 0.08), (0.08, np.inf)]
MARKET_RULE_MIN_PICKS = 100


def season_start(d: date) -> date:
    return date(d.year if d.month >= 9 else d.year - 1, 9, 1)


def load_settled(session, start: date, end: date, mode: str = "paper") -> pd.DataFrame:
    """Settled picks on games from `start` to `end` (boost-only legs excluded)."""
    df = pd.read_sql_query(
        text("""
        SELECT p.*, g.game_date, g.tip_time, op.open_line FROM picks p JOIN games g USING (game_id)
        LEFT JOIN LATERAL (
            SELECT o.line AS open_line FROM odds_snapshots o
            WHERE o.game_id = p.game_id AND o.player_id = p.player_id AND o.market = p.market
              AND o.book = 'draftkings' AND o.line_type = 'main'
            ORDER BY o.snapshot_ts LIMIT 1) op ON p.line_type = 'main'
        WHERE p.result IS NOT NULL AND p.mode = :m AND p.tier <> 'boost_leg'
          AND g.game_date BETWEEN :a AND :b"""),
        session.connection(),
        params={"m": mode, "a": start, "b": end},
    )
    for c in ("stake", "pnl", "clv", "ev", "fair_prob", "p_model", "p_consensus", "line"):
        df[c] = df[c].astype(float)
    if len(df):
        hour = pd.to_datetime(df["flagged_ts"], utc=True).dt.tz_convert(ET).dt.hour
        df["flag_time"] = np.select(
            [hour == 10, hour == 15, hour == 18], ["10 AM", "3 PM", "6 PM"], "late change"
        )
        taken = df["price_placed"].fillna(df["dk_price"]).astype(int)
        df["decimal"] = [to_decimal(x) for x in taken]
        df["role"] = [_role(c) for c in df["context"]]
        lead = (
            pd.to_datetime(df["tip_time"], utc=True) - pd.to_datetime(df["flagged_ts"], utc=True)
        ).dt.total_seconds() / 60
        # the record rule from 2026-10-30: picks made within 2 hours of tip-off
        df["timing"] = np.where(lead <= 120, "within 2h of tip", "earlier in the day")
        move = df["line"] - df["open_line"].astype(float)
        df["line_move"] = np.select(
            [df["open_line"].isna(), move >= 1, move <= -1],
            ["no opening line", "line rose 1+", "line fell 1+"],
            "line steady",
        )
    return df


def _role(context) -> str:
    """Whether a key teammate was out when the pick was flagged (features.context)."""
    if not isinstance(context, dict):
        return "untagged"
    return "key teammate out" if context.get("key_out") else "no key absence"


def _clustered(x: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x)
    x, c = x[ok], np.asarray(clusters)[ok]
    if not len(x):
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    resid = pd.Series(x - x.mean()).groupby(c).sum()
    return float(x.mean()), float(np.sqrt((resid**2).sum()) / len(x))


def summarize(df: pd.DataFrame) -> dict:
    s = df[df["result"] != "void"]
    out = {"bets": int(len(s)), "voided": int((df["result"] == "void").sum())}
    if s.empty:
        return out
    days = s["game_date"].astype(str).to_numpy()
    staked = s[s["stake"] > 0]
    wins, losses = int((s["result"] == "win").sum()), int((s["result"] == "loss").sum())
    clv, clv_se = _clustered(s["clv"].to_numpy(float), days)
    out.update(
        {
            "staked_bets": int(len(staked)),
            "staked": round(float(staked["stake"].sum()), 2),
            "profit": round(float(staked["pnl"].sum()), 2),
            "roi": round(float(staked["pnl"].sum() / staked["stake"].sum()), 4)
            if len(staked)
            else None,
            "hit_rate": round(wins / (wins + losses), 4) if wins + losses else None,
            "breakeven": round(float((1 / s["decimal"]).mean()), 4),
            "avg_ev": round(float(s["ev"].mean()), 4),
            "avg_clv": None if np.isnan(clv) else round(clv, 4),
            "clv_se": None if np.isnan(clv_se) else round(clv_se, 4),
            "clv_coverage": round(float(s["clv"].notna().mean()), 4),
        }
    )
    return out


def _split(df: pd.DataFrame, col: str) -> dict:
    return {str(k): summarize(g) for k, g in df.groupby(col)} if len(df) else {}


def _ev_buckets(df: pd.DataFrame) -> dict:
    out = {}
    for lo, hi in EV_BUCKETS:
        g = df[(df["ev"] >= lo) & (df["ev"] < hi) & (df["result"] != "void")]
        label = f"{lo:.0%}-{'' if hi == np.inf else f'{hi:.0%}'}"
        out[label] = {
            **summarize(g),
            "predicted_ev": round(float(g["ev"].mean()), 4) if len(g) else None,
        }
    return out


def weekly_clv(df: pd.DataFrame, end: date, weeks: int = 4) -> list[dict]:
    """Average CLV per complete week (Monday-Sunday) ending on or before `end`."""
    last_sunday = end - timedelta(days=(end.weekday() + 1) % 7)
    out = []
    for i in range(weeks - 1, -1, -1):
        hi = last_sunday - timedelta(days=7 * i)
        lo = hi - timedelta(days=6)
        g = df[(df["game_date"] >= lo) & (df["game_date"] <= hi) & (df["result"] != "void")]
        clv = g["clv"].dropna()
        out.append(
            {
                "week": f"{lo} to {hi}",
                "bets": int(len(g)),
                "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
            }
        )
    return out


def parlay_summary(session, start: date, end: date, mode: str = "paper") -> dict:
    p = pd.read_sql_query(
        text("""SELECT tier, result, stake, pnl, ev FROM parlays
                                   WHERE slate_date BETWEEN :a AND :b AND mode = :m
                                     AND result IS NOT NULL"""),
        session.connection(),
        params={"a": start, "b": end, "m": mode},
    )
    out = {}
    for tier, g in p.groupby("tier"):
        settled = g[~g["result"].isin(["push", "void"])]
        stake = float(g["stake"].astype(float).sum())
        out[tier] = {
            "parlays": int(len(g)),
            "hit_rate": round(float((settled["result"] == "win").mean()), 4)
            if len(settled)
            else None,
            "avg_ev": round(float(g["ev"].astype(float).mean()), 4),
            "roi": round(float(g["pnl"].astype(float).sum()) / stake, 4) if stake else None,
        }
    return out


def slippage(session, start: date, end: date) -> dict:
    """Real bets: the price you got vs DraftKings' price when flagged."""
    r = load_settled(session, start, end, "real")
    r = r[r["price_placed"].notna()]
    if r.empty:
        return {"bets": 0}
    flagged = np.array([1 / to_decimal(int(x)) for x in r["dk_price"]])
    got = np.array([1 / to_decimal(int(x)) for x in r["price_placed"]])
    return {
        "bets": int(len(r)),
        "avg_implied_prob_cost": round(float((got - flagged).mean()), 4),
        "worse_than_flagged": int((got > flagged + 1e-9).sum()),
    }


def weekly_review(session, end: date, mode: str = "paper") -> dict:
    week = load_settled(session, end - timedelta(days=6), end, mode)
    season = load_settled(session, season_start(end), end, mode)
    rep = {
        "week_ending": str(end),
        "mode": mode,
        "week": summarize(week),
        "season": summarize(season),
        "weekly_clv": weekly_clv(season, end),
    }
    for name, df in (("week_splits", week), ("season_splits", season)):
        rep[name] = {
            "by_market": _split(df, "market"),
            "by_flag_time": _split(df, "flag_time") if len(df) else {},
            "by_side": _split(df, "side"),
            "by_tier": _split(df, "tier"),
            "by_source": _split(
                df.assign(source=np.where(df["p_consensus"].notna(), "consensus", "model-only")),
                "source",
            )
            if len(df)
            else {},
            "ev_vs_clv": _ev_buckets(df) if len(df) else {},
            # unders on players whose role grew: does the market know better there?
            "by_side_role": _split(df.assign(k=df["side"] + ", " + df["role"]), "k")
            if len(df)
            else {},
            "by_side_line_move": _split(df.assign(k=df["side"] + ", " + df["line_move"]), "k")
            if len(df)
            else {},
            "by_timing": _split(df, "timing") if len(df) else {},
        }
    rep["parlays"] = parlay_summary(session, season_start(end), end, mode)
    rep["slippage"] = slippage(session, season_start(end), end)
    rep["market_rule"] = market_rule_candidates(season)
    return rep


def market_rule_candidates(season: pd.DataFrame) -> list[dict]:
    """Markets with 100+ settled picks and negative season CLV."""
    out = []
    for m, g in season[season["result"] != "void"].groupby("market") if len(season) else []:
        clv = g["clv"].dropna()
        if len(g) >= MARKET_RULE_MIN_PICKS and len(clv) and clv.mean() < 0:
            out.append({"market": m, "picks": int(len(g)), "avg_clv": round(float(clv.mean()), 4)})
    return out


def apply_market_rule(session, rep: dict, now: datetime | None = None) -> list[str]:
    """Switch off the markets the rule names; returns the ones newly switched off."""
    now = now or datetime.now(UTC)
    done = []
    for c in rep["market_rule"]:
        current = session.get(MarketControl, c["market"])
        if current is not None and current.disabled:
            continue
        session.execute(
            insert(MarketControl)
            .values(
                market=c["market"],
                disabled=True,
                changed_ts=now,
                reason=f"season CLV {c['avg_clv']:+.2%} after {c['picks']} picks",
            )
            .on_conflict_do_update(
                index_elements=["market"],
                set_={
                    "disabled": True,
                    "changed_ts": now,
                    "reason": f"season CLV {c['avg_clv']:+.2%}",
                },
            )
        )
        done.append(c["market"])
    return done


def render(rep: dict) -> str:
    """A phone-sized summary; the full report is the JSON."""

    def line(label, s):
        if not s.get("bets"):
            return f"{label}: no settled picks"
        clv = "n/a" if s.get("avg_clv") is None else f"{s['avg_clv']:+.2%}"
        roi = "n/a" if s.get("roi") is None else f"{s['roi']:+.1%}"
        hit = "n/a" if s.get("hit_rate") is None else f"{s['hit_rate']:.1%}"
        return (
            f"{label}: {s['bets']} picks, CLV {clv}, ROI {roi} on {s.get('staked_bets', 0)} "
            f"staked, hit {hit} vs breakeven {s['breakeven']:.1%}"
        )

    out = [
        f"Weekly review ({rep['mode']}), week ending {rep['week_ending']}",
        line("Week", rep["week"]),
        line("Season", rep["season"]),
    ]
    weeks = [w["avg_clv"] for w in rep["weekly_clv"]]
    out.append("CLV by week: " + ", ".join("n/a" if w is None else f"{w:+.2%}" for w in weeks))
    for m, s in rep["season_splits"]["by_market"].items():
        out.append("  " + line(m, s))
    for b, s in rep["season_splits"]["ev_vs_clv"].items():
        if s.get("bets"):
            clv = "n/a" if s.get("avg_clv") is None else f"{s['avg_clv']:+.2%}"
            out.append(f"  EV {b}: predicted {s['predicted_ev']:+.1%}, realized CLV {clv}")
    sr = rep["season_splits"]
    for key in ("within 2h of tip", "earlier in the day"):
        if key in sr.get("by_timing", {}):
            out.append("  picks made " + line(key, sr["by_timing"][key]))
    for key in ("under, key teammate out", "under, no key absence"):
        if key in sr.get("by_side_role", {}):
            out.append("  " + line(key, sr["by_side_role"][key]))
    for key in ("under, line rose 1+", "under, line steady"):
        if key in sr.get("by_side_line_move", {}):
            out.append("  " + line(key, sr["by_side_line_move"][key]))
    names = {
        "satellite": "Parlays (from singles)",
        "boost": "Boosts",
        "alt_combo": "Alt-line parlays (experiment)",
    }
    for tier, p in rep.get("parlays", {}).items():
        hit = "n/a" if p.get("hit_rate") is None else f"{p['hit_rate']:.1%}"
        roi = "n/a" if p.get("roi") is None else f"{p['roi']:+.1%}"
        out.append(
            f"{names.get(tier, tier)}: {p['parlays']} settled, hit {hit}, "
            f"ROI {roi}, average predicted edge {p['avg_ev']:+.1%}"
        )
    if rep["market_rule"]:
        out.append(
            "Switched off (CLV negative after 100+ picks): "
            + ", ".join(c["market"] for c in rep["market_rule"])
        )
    if rep["season"].get("bets", 0) < 300:
        out.append("Fewer than 300 picks: these numbers are still noisy (plan, Stage 9).")
    return "\n".join(out)
