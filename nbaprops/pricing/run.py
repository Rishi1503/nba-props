"""Pricing command: `python -m nbaprops.pricing.run --date 2026-10-20`.

Prices the slate with the active model run's projections and the latest odds, prints the
ranked board, and writes picks and parlays (paper mode by default; --dry-run writes nothing).
Boosts typed in from the DraftKings app: --boost 0.30:25 (a 30% parlay profit boost with a
$25 max stake) or --boost-single 0.25:50 (a boost on one straight bet).
"""

import argparse
from datetime import UTC, date, datetime

from nbaprops.config import load_strategy
from nbaprops.db.base import session_scope
from nbaprops.pricing import slate
from nbaprops.settings import get_settings


def _boost(text: str) -> tuple[float, float]:
    pct, max_stake = text.split(":")
    return float(pct), float(max_stake)


def render(res: slate.SlateResult) -> str:
    out = [
        f"Slate {res.game_date} as of {res.as_of:%H:%M} UTC, run {res.run_id}, "
        f"bankroll {res.bankroll:,.2f}; {res.candidates} props priced."
    ]
    out += [f"  skipped: {m}" for m in res.skipped]
    if res.blocked_questionable:
        out.append(
            f"  {res.blocked_questionable} props blocked only by the Questionable-teammate "
            "rule (not +EV whichever way the injury news goes)"
        )
    if res.blocked_low_confidence:
        out.append(
            f"  {res.blocked_low_confidence} props cleared every rule except low "
            "confidence (minutes uncertainty, incl. teammates of a Questionable player)"
        )
    if len(res.picks):
        out.append("")
        out.append(
            f"{'player':>8} {'market':6} {'line':>5} {'side':5} {'price':>6} "
            f"{'fair':>6} {'EV':>7} {'stake':>8}  tier"
        )
        for r in res.picks.sort_values("ev", ascending=False).itertuples():
            out.append(
                f"{int(r.player_id):>8} {r.market:6} {r.line:>5} {r.side:5} {int(r.price):>+6d} "
                f"{r.p_fair:>6.3f} {r.ev:>+7.1%} {r.stake:>8.2f}  "
                f"{'high-conf' if r.high_confidence else 'standard'}"
                f"{'' if r.has_consensus else ' (model only)'}"
            )
    for p in res.parlays:
        legs = " + ".join(f"{lg.player_id} {lg.market} {lg.side} {lg.line}" for lg in p.legs)
        out.append(
            f"{p.tier}: {legs} at {p.american:+d}"
            f"{f' boosted {p.boost_pct:.0%}' if p.boost_pct else ''}: "
            f"EV {p.ev:+.1%}, stake {p.stake:.2f}"
        )
    if not len(res.picks) and not res.parlays:
        out.append("No picks.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> slate.SlateResult:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--mode", choices=["paper", "real"], default="paper")
    ap.add_argument("--boost", type=_boost, action="append", default=[])
    ap.add_argument("--boost-single", type=_boost, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_strategy(get_settings().strategy_config)
    with session_scope() as s:
        boosts = [(*b, "parlay") for b in args.boost] + [(*b, "single") for b in args.boost_single]
        res = slate.price_slate(
            s, args.date, datetime.now(UTC), cfg, args.mode, boosts, write=not args.dry_run
        )
    print(render(res))
    return res


if __name__ == "__main__":
    main()
