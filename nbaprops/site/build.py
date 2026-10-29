"""Build the bankroll page: the paper record as if every recommended bet had been placed.

    python -m nbaprops.site.build --out site/dist

Writes index.html (self-contained: the data is embedded) and hash.txt, a fingerprint of the
data, so publishing can skip an upload when nothing changed (picks, results, the verdict).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
TEMPLATE = Path(__file__).with_name("page.html")


def _f(x) -> float:
    return float(x) if x is not None else 0.0


def snapshot(session, now: datetime) -> dict:
    from nbaprops.api import views
    from nbaprops.ops import golive
    from nbaprops.ops import review as rv

    today = now.astimezone(ET).date()
    season = rv.season_start(today)
    ledger = views.bankroll(session)
    start_bank = _f(ledger[0]["bankroll"]) if ledger else 1000.0
    start_date = str(ledger[0]["entry_date"]) if ledger else str(season)
    days = list(reversed(views.record(session, season, today, "paper")))  # oldest first

    curve = [{"date": start_date, "bankroll": round(start_bank, 2), "pnl": 0.0}]
    bank, staked, singles = start_bank, 0.0, {"win": 0, "loss": 0, "push": 0}
    tiers: dict[str, dict] = {}
    out_days = []
    for d in days:
        bets = [p for p in d["picks"] if _f(p["stake"]) > 0]
        settled = [p for p in bets if p["result"] in ("win", "loss", "push")]
        pars = [p for p in d["parlays"] if _f(p["stake"]) > 0]
        done_pars = [p for p in pars if p["result"] in ("win", "loss", "push")]
        day_pnl = sum(_f(p["pnl"]) for p in settled) + sum(_f(p["pnl"]) for p in done_pars)
        staked += sum(_f(p["stake"]) for p in settled) + sum(_f(p["stake"]) for p in done_pars)
        for p in settled:
            singles[p["result"]] += 1
        for p in done_pars:
            t = tiers.setdefault(p["tier"], {"won": 0, "lost": 0, "pnl": 0.0})
            t["won" if p["result"] == "win" else "lost"] += p["result"] != "push"
            t["pnl"] = round(t["pnl"] + _f(p["pnl"]), 2)
        if settled or done_pars:
            bank += day_pnl
            curve.append({"date": d["date"], "bankroll": round(bank, 2), "pnl": round(day_pnl, 2)})
        out_days.append(
            {
                "date": d["date"],
                "pnl": round(day_pnl, 2),
                "open": len(bets) - len(settled) + len(pars) - len(done_pars),
                "picks": [
                    {
                        k: p[k]
                        for k in (
                            "player",
                            "market",
                            "side",
                            "line",
                            "dk_price",
                            "stake",
                            "result",
                            "pnl",
                        )
                    }
                    for p in bets
                ],
                "parlays": [
                    {
                        "tier": p["tier"],
                        "dk_price": p["dk_price"],
                        "stake": p["stake"],
                        "result": p["result"],
                        "pnl": p["pnl"],
                        "legs": [
                            {k: leg[k] for k in ("player", "market", "side", "line", "result")}
                            for leg in p["legs"]
                        ],
                    }
                    for p in pars
                ],
            }
        )
    clv = rv.summarize(rv.load_settled(session, season, today, "paper"))
    kill = golive.checklist(session, today)["kill"]
    profit = round(bank - start_bank, 2)
    data = {
        "start": {"date": start_date, "bankroll": round(start_bank, 2)},
        "bankroll": round(bank, 2),
        "profit": profit,
        "roi": round(profit / staked, 4) if staked else None,
        "staked": round(staked, 2),
        "singles": singles,
        "parlays": tiers,
        "avg_clv": clv.get("avg_clv"),
        "clv_bets": clv.get("bets", 0),
        "curve": curve,
        "days": list(reversed(out_days))[:21],  # newest first, three weeks
        "today": str(today),
        "kill": {k: kill.get(k) for k in ("verdict", "picks", "weeks", "avg_clv", "reason")},
    }
    return json.loads(json.dumps(data, default=str))


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def render(data: dict, now: datetime) -> str:
    page = {**data, "updated": now.astimezone(ET).strftime("%b %-d, %-I:%M %p ET")}
    blob = json.dumps(page).replace("</", "<\\/")  # safe inside <script>
    return TEMPLATE.read_text().replace("/*__DATA__*/null", blob)


def build(out: Path, now: datetime | None = None) -> str:
    from nbaprops.db.base import session_scope

    now = now or datetime.now(UTC)
    with session_scope() as s:
        data = snapshot(s, now)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(data, now))
    fp = fingerprint(data)
    (out / "hash.txt").write_text(fp + "\n")
    return fp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("site/dist"))
    args = ap.parse_args()
    print(f"built {args.out / 'index.html'} (data {build(args.out)})")
