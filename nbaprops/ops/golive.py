"""Go-live checklist (plan, Stage 9), checked by code. Real money goes in only when every
item passes:

  1. at least 300 settled paper picks, over at least four weeks (whichever comes later)
  2. average CLV positive, and positive in at least 3 of the last 4 complete weeks
  3. expected calibration error under 3 points on live picks, no bucket off by more than 8
     (buckets need 20+ picks, so noise alone can't fail one)
  4. the pipeline ran 14 straight days: every cycle succeeded and no manual fix was recorded
     (a day with no cycles at all breaks the streak)
  5. a bankroll is recorded and fixed: no deposits or withdrawals since the first entry

Kill rule, checked beside it: average CLV with a day-clustered confidence interval (picks on
one day share games and news). Stop early at 150+ picks over 3+ weeks if the 99% interval is
entirely below zero; stop at 300+ picks over 4+ weeks if the 95% interval is; and stop at
600+ picks if average CLV is still not above zero (an edge worth betting would show by then).
"""

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from nbaprops.models.calibrate import expected_calibration_error, reliability
from nbaprops.ops.review import ET, load_settled, season_start, weekly_clv

MIN_PICKS = 300
MIN_WEEKS = 4
ECE_MAX = 0.03
BUCKET_MAX = 0.08
BUCKET_MIN_N = 20
CLEAN_DAYS = 14
# kill rule: (picks, weeks, z) stages; the last stage stops if average CLV isn't positive
KILL_EARLY = (150, 3, 2.576)
KILL_FULL = (300, 4, 1.96)
KILL_NO_EDGE = 600


@dataclass
class Criterion:
    name: str
    ok: bool
    value: str
    detail: str = ""


def _picks(session, today: date) -> pd.DataFrame:
    df = load_settled(session, season_start(today), today, "paper")
    return df[df["result"] != "void"]


def check_volume(df: pd.DataFrame, today: date) -> Criterion:
    n = len(df)
    first = min(df["game_date"]) if n else None
    weeks = (today - first).days / 7 if first else 0.0
    ok = n >= MIN_PICKS and weeks >= MIN_WEEKS
    return Criterion("300+ paper picks over 4+ weeks", ok, f"{n} picks over {weeks:.1f} weeks")


def check_clv(df: pd.DataFrame, today: date) -> Criterion:
    clv = df["clv"].dropna()
    weeks = weekly_clv(df, today, 4)
    pos = sum(1 for w in weeks if w["avg_clv"] is not None and w["avg_clv"] > 0)
    avg = float(clv.mean()) if len(clv) else float("nan")
    ok = bool(len(clv)) and avg > 0 and pos >= 3
    weekly = ", ".join("n/a" if w["avg_clv"] is None else f"{w['avg_clv']:+.2%}" for w in weeks)
    return Criterion(
        "CLV positive, and in 3 of 4 weeks",
        ok,
        f"average {avg:+.2%}; {pos} of 4 weeks positive" if len(clv) else "no CLV yet",
        f"weeks: {weekly}",
    )


def check_calibration(df: pd.DataFrame) -> Criterion:
    s = df[df["result"].isin(["win", "loss"])]  # pushes say nothing about the probability
    if s.empty:
        return Criterion("Calibration within 3 points, no bucket off by 8+", False, "no picks yet")
    p = s["fair_prob"].to_numpy(float)
    hit = (s["result"] == "win").to_numpy(float)
    ece = expected_calibration_error(p, hit)
    worst = max(
        (
            abs(b["predicted"] - b["observed"])
            for b in reliability(p, hit)
            if b["n"] >= BUCKET_MIN_N
        ),
        default=0.0,
    )
    ok = ece < ECE_MAX and worst <= BUCKET_MAX
    return Criterion(
        "Calibration within 3 points, no bucket off by 8+",
        ok,
        f"ECE {ece:.1%}, worst bucket {worst:.1%}",
    )


def check_pipeline(session, today: date) -> Criterion:
    runs = pd.read_sql_query(
        text("""SELECT kind, started_ts, ok FROM ops_runs
                                      WHERE started_ts >= :since"""),
        session.connection(),
        params={"since": today - timedelta(days=60)},
    )
    if runs.empty:
        return Criterion("Pipeline clean 14 straight days", False, "no runs recorded")
    runs["day"] = pd.to_datetime(runs["started_ts"], utc=True).dt.tz_convert(ET).dt.date
    streak, day = 0, today - timedelta(days=1)  # today is still in progress
    while True:
        r = runs[runs["day"] == day]
        cycles = r[r["kind"] == "cycle"]
        if cycles.empty or not cycles["ok"].all() or (r["kind"] == "manual_fix").any():
            break
        streak += 1
        day -= timedelta(days=1)
    return Criterion(
        "Pipeline clean 14 straight days", streak >= CLEAN_DAYS, f"{streak} clean days in a row"
    )


def check_bankroll(session) -> Criterion:
    rows = pd.read_sql_query(
        text("""SELECT entry_date, deposits, withdrawals, bankroll
                                      FROM bankroll_ledger ORDER BY entry_date"""),
        session.connection(),
    )
    if rows.empty:
        return Criterion("Bankroll recorded and fixed", False, "no bankroll recorded")
    later = rows.iloc[1:]
    moved = (
        later["deposits"].astype(float).abs().sum() + later["withdrawals"].astype(float).abs().sum()
    )
    return Criterion(
        "Bankroll recorded and fixed",
        moved == 0,
        f"{float(rows['bankroll'].iloc[0]):,.2f} set {rows['entry_date'].iloc[0]}"
        + ("" if moved == 0 else "; changed since (deposits or withdrawals)"),
    )


def kill_check(df: pd.DataFrame, today: date) -> dict:
    """Whether the paper record says to stop: the model is losing to closing lines, or has had
    a large sample without showing an edge. CLV is per pick; the interval is clustered by day."""
    d = df[df["clv"].notna()]
    n = int(len(d))
    first = min(d["game_date"]) if n else None
    weeks = (today - first).days / 7 if first else 0.0
    out = {
        "verdict": "too early",
        "picks": n,
        "weeks": round(weeks, 1),
        "avg_clv": None,
        "low": None,
        "high": None,
        "reason": "",
    }
    if not n:
        out["reason"] = "no CLV yet"
        return out
    clv = d["clv"].astype(float)
    avg = float(clv.mean())
    by_day = (clv - avg).groupby(d["game_date"]).sum()
    se = float(np.sqrt((by_day**2).sum())) / n if len(by_day) > 1 else float("inf")
    out["avg_clv"] = round(avg, 5)

    def interval(z):
        return avg - z * se, avg + z * se

    lo95, hi95 = interval(1.96)
    if np.isfinite(se):
        out["low"], out["high"] = round(lo95, 5), round(hi95, 5)
    early_n, early_w, early_z = KILL_EARLY
    full_n, full_w, full_z = KILL_FULL
    if n < early_n or weeks < early_w:
        out["reason"] = f"needs {early_n}+ picks over {early_w}+ weeks"
    elif n >= KILL_NO_EDGE and avg <= 0:
        out["verdict"] = "STOP"
        out["reason"] = f"{n} picks and average CLV is still not positive: no edge to bet"
    elif n >= full_n and weeks >= full_w and interval(full_z)[1] < 0:
        out["verdict"] = "STOP"
        out["reason"] = "95% interval for average CLV is entirely below zero"
    elif interval(early_z)[1] < 0:
        out["verdict"] = "STOP"
        out["reason"] = "99% interval for average CLV is entirely below zero (early stop)"
    else:
        out["verdict"] = "keep going"
        out["reason"] = "not clearly losing to closing lines yet"
    return out


def checklist(session, today: date) -> dict:
    df = _picks(session, today)
    items = [
        check_volume(df, today),
        check_clv(df, today),
        check_calibration(df),
        check_pipeline(session, today),
        check_bankroll(session),
    ]
    kill = kill_check(df, today)
    return {
        "date": str(today),
        "ready": all(c.ok for c in items) and kill["verdict"] != "STOP",
        "criteria": [asdict(c) for c in items],
        "kill": kill,
    }


def render(result: dict) -> str:
    out = [f"Go-live check {result['date']}: {'READY' if result['ready'] else 'not yet'}"]
    k = result.get("kill")
    if k:
        span = (
            f"; average CLV {k['avg_clv']:+.2%} (95% {k['low']:+.2%} to {k['high']:+.2%})"
            if k["avg_clv"] is not None and k["low"] is not None
            else ""
        )
        out.append(
            f"Kill rule: {k['verdict'].upper()} - {k['picks']} picks over {k['weeks']} weeks"
            f"{span}. {k['reason']}"
        )
        if k["verdict"] == "STOP":
            out.append(
                "The model is not beating DraftKings' closing lines. Stop paper trading and "
                "don't bet real money; more tuning on the same data would be fitting noise."
            )
    for c in result["criteria"]:
        out.append(f"[{'x' if c['ok'] else ' '}] {c['name']}: {c['value']}")
    if not result["ready"]:
        out.append("Real money waits until every box is checked (plan, Stage 9).")
    return "\n".join(out)


def ece_note() -> str:
    """Why 300 picks isn't enough to prove a hit rate (plan, Stage 9 sample size)."""
    se = np.sqrt(0.75 * 0.25 / 300)
    return f"At 300 picks a 75% hit rate is +/-{1.96 * se:.1%}; CLV, not hit rate, gates go-live."
