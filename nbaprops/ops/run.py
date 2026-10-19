"""Stage 8 command line. Run `python -m nbaprops.ops.run --help`.

cycle          what the scheduler runs every 15 minutes (odds, settlement, projection and
               pricing at 10 AM / 3 PM / 6 PM, late-change reruns, health checks)
settle         grade finished picks and parlays now
retrain        weekly: refit the models through yesterday as a challenger (shadow)
import-bundle  store a saved bundle in the database: import-bundle PATH [--champion]
promote        make a stored run the champion (and roll back the same way)
status         the champion, challengers, and today's picks
review         weekly review (Sunday night): report, market shut-off rule, go-live status
golive         the go-live checklist
tune           retune the blend weight and EV floor on paper data (a report; never applied)
recalibrate    monthly: fit the live calibration layer on recent results
place          record a real bet: place PICK_ID --price -110 --stake 20
market         switch a market off or back on: market pts --off "reason" / market pts --on
note-fix       record a manual fix (resets the go-live pipeline streak): note-fix "what"
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from nbaprops.config import load_strategy
from nbaprops.db.base import session_scope
from nbaprops.ops import alerts, bundles
from nbaprops.ops.cycle import ET, real_deps, run_cycle
from nbaprops.settings import get_settings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nbaprops.ops.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cycle")
    sub.add_parser("settle")
    sub.add_parser("retrain")
    imp = sub.add_parser("import-bundle")
    imp.add_argument("path", type=Path)
    imp.add_argument("--champion", action="store_true")
    pro = sub.add_parser("promote")
    pro.add_argument("run_id")
    sub.add_parser("status")
    rv = sub.add_parser("review")
    rv.add_argument("--mode", choices=["paper", "real"], default="paper")
    sub.add_parser("golive")
    tu = sub.add_parser("tune")
    tu.add_argument("--days", type=int, default=60)
    rc = sub.add_parser("recalibrate")
    rc.add_argument("--days", type=int, default=90)
    pl = sub.add_parser("place")
    pl.add_argument("pick_id", type=int)
    pl.add_argument("--price", type=int, required=True)
    pl.add_argument("--stake", type=float, required=True)
    mk = sub.add_parser("market")
    mk.add_argument("market", choices=["pts", "reb", "ast", "fg3m", "pra"])
    onoff = mk.add_mutually_exclusive_group(required=True)
    onoff.add_argument("--off", metavar="REASON")
    onoff.add_argument("--on", action="store_true")
    nf = sub.add_parser("note-fix")
    nf.add_argument("note")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = datetime.now(UTC)
    today = now.astimezone(ET).date()
    notifier = alerts.from_env()
    cfg = load_strategy(get_settings().strategy_config)

    if args.cmd == "cycle":
        from nbaprops.pipeline.odds import PropLineProvider

        provider = PropLineProvider(get_settings().propline_api_key)
        rep = run_cycle(session_scope, now, real_deps(provider, cfg), notifier)
        print(
            f"cycle: {', '.join(rep.actions)}; {rep.new_picks} new picks"
            + (f"; errors: {rep.errors}" if rep.errors else "")
        )
        if rep.errors:
            raise SystemExit(1)  # mark the scheduled run failed (the alert already went out)
    elif args.cmd == "settle":
        from nbaprops.ops.settle import settle

        with session_scope() as s:
            print(settle(s, today, now).summary())
    elif args.cmd == "retrain":
        from nbaprops.ops.retrain import retrain

        with session_scope() as s:
            run_id = retrain(s, today)
        msg = (
            f"Weekly retrain stored challenger {run_id} (projects in shadow)."
            if run_id
            else "Weekly retrain skipped: no champion model to start from."
        )
        notifier.send(msg)
        print(msg)
    elif args.cmd == "import-bundle":
        from nbaprops.models.project import Bundle

        bundle = Bundle.load(args.path)
        with session_scope() as s:
            bundles.save(s, bundle, champion=args.champion)
        print(f"stored {bundle.run_id}" + (" as champion" if args.champion else ""))
    elif args.cmd == "promote":
        with session_scope() as s:
            bundles.promote(s, args.run_id)
        print(f"{args.run_id} is now the champion")
    elif args.cmd == "status":
        with session_scope() as s:
            print("champion:", bundles.champion(s))
            print("challengers:", bundles.stored_challengers(s))
            n = s.execute(
                text("""SELECT count(*), coalesce(sum(stake), 0) FROM picks p
                                  JOIN games g USING (game_id) WHERE g.game_date = :d"""),
                {"d": today},
            ).one()
            print(f"today ({today}): {n[0]} picks, {float(n[1]):.2f} staked")

    elif args.cmd in ("review", "golive", "tune", "recalibrate", "place", "market", "note-fix"):
        stage9(args, now, today, cfg, notifier)


def stage9(args, now, today, cfg, notifier) -> None:
    import json
    from datetime import timedelta

    from nbaprops.db.models import MarketControl, OpsRun
    from nbaprops.ops import golive, livecal, review, tune
    from nbaprops.ops.place import place
    from nbaprops.pricing.engine import Rules

    with session_scope() as s:
        if args.cmd == "review":
            rep = review.weekly_review(s, today, args.mode)
            off = review.apply_market_rule(s, rep, now) if args.mode == "paper" else []
            check = golive.checklist(s, today)
            msg = review.render(rep) + "\n\n" + golive.render(check)
            if off:
                msg += "\nNow switched off: " + ", ".join(off)
            Path("reports/review").mkdir(parents=True, exist_ok=True)
            Path(f"reports/review/{today}-{args.mode}.json").write_text(
                json.dumps({"review": rep, "golive": check}, indent=2, default=str)
            )
            notifier.send(msg)
            print(msg)
        elif args.cmd == "golive":
            print(golive.render(golive.checklist(s, today)))
        elif args.cmd == "tune":
            out = tune.tune(s, today - timedelta(days=args.days), today, Rules.from_config(cfg))
            print(json.dumps(out, indent=2, default=str))
        elif args.cmd == "recalibrate":
            done = livecal.fit_live(s, today - timedelta(days=args.days), today, now)
            msg = (
                "Live calibration updated: " + ", ".join(f"{m} ({n})" for m, n in done.items())
                if done
                else "Live calibration: no market improved on recent results; unchanged."
            )
            notifier.send(msg)
            print(msg)
        elif args.cmd == "place":
            real = place(s, args.pick_id, args.price, args.stake, now)
            print(f"recorded real bet {real.pick_id}: {args.stake:.2f} at {args.price:+d}")
        elif args.cmd == "market":
            row = s.get(MarketControl, args.market) or MarketControl(market=args.market)
            row.disabled, row.reason, row.changed_ts = bool(args.off), args.off, now
            s.merge(row)
            print(f"{args.market} {'off' if args.off else 'on'}")
        elif args.cmd == "note-fix":
            s.add(
                OpsRun(
                    kind="manual_fix",
                    started_ts=now,
                    finished_ts=now,
                    ok=False,
                    actions=[],
                    errors=[],
                    note=args.note,
                )
            )
            print("noted; the go-live pipeline streak restarts")


if __name__ == "__main__":
    main()
