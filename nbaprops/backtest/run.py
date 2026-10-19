"""Stage 6 command: `python -m nbaprops.backtest.run --season 2025-26`.

Runs the walk-forward backtest for a season (after a calibration warm-up on the season
before), writes a Layer A report (JSON + Markdown) under reports/backtest/, and optionally:

  --ablate roster   rerun without the roster inputs and report the paired difference
  --layer-b         betting replay on the season's logged odds (Postgres), using the
                    walk-forward's probabilities at DraftKings' exact lines
  --production      retrain on every season and save a production bundle, calibrated on the
                    walk-forward's out-of-sample predictions (only after the holdout passes)

The holdout season (HOLDOUT_SEASON) is protected: its first evaluation is recorded in
reports/holdout_ledger.json (commit it). A second evaluation is refused unless
--rerun-holdout "<reason>" is given, and that report is marked as not a clean holdout.
Thresholds tuned after looking at a holdout make it a tuning season.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nbaprops.backtest import layer_b, metrics
from nbaprops.backtest import walkforward as wf
from nbaprops.features.build import FEATURE_COLUMNS, FEATURE_GROUPS, prev_season
from nbaprops.models import calibrate, simulate
from nbaprops.models import minutes as mm
from nbaprops.models import rates as rm
from nbaprops.models.project import Bundle, git_sha

log = logging.getLogger(__name__)

HOLDOUT_SEASON = "2025-26"
FIRST_SEASON = "2021-22"  # priors warm-up only (Stage 1)
REPORTS = Path("reports")
LEDGER = REPORTS / "holdout_ledger.json"


class HoldoutSpent(RuntimeError):
    pass


# --- holdout ledger -------------------------------------------------------------------


def read_ledger(path: Path = LEDGER) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def check_holdout(season: str, rerun_reason: str | None, path: Path = LEDGER) -> bool:
    """True if this evaluation is a clean holdout. Raises if it was already used and no reason
    is given for using it again."""
    if season != HOLDOUT_SEASON:
        return True
    used = [e for e in read_ledger(path) if e["season"] == season]
    if not used:
        return True
    if not rerun_reason:
        first = used[0]
        raise HoldoutSpent(
            f"The {season} holdout was already evaluated on {first['date']} "
            f"(run {first['run_id']}). Re-running it after changes turns it into a tuning "
            'season. If you must, pass --rerun-holdout "<reason>"; the report will be '
            "marked as not a clean holdout."
        )
    return False


def record_holdout(
    season: str, run_id: str, clean: bool, reason: str | None, passes: bool, path: Path = LEDGER
) -> None:
    if season != HOLDOUT_SEASON:
        return
    entries = read_ledger(path)
    entries.append(
        {
            "season": season,
            "run_id": run_id,
            "date": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "clean": clean,
            "reason": reason,
            "passes": passes,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


# --- data ----------------------------------------------------------------------------


def seasons_through(season: str) -> list[str]:
    out = [season]
    while out[-1] > FIRST_SEASON:
        out.append(prev_season(out[-1]))
    return sorted(out)


def load_frame(season: str, parquet: Path | None) -> pd.DataFrame:
    if parquet:
        df = pd.read_parquet(parquet)
    else:
        from nbaprops.db.base import session_scope
        from nbaprops.features.build import build_training_set

        with session_scope() as s:
            df = build_training_set(s, seasons_through(season))
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[df["season"] <= season]


# --- production bundle -------------------------------------------------------------------


def production_bundle(
    df: pd.DataFrame,
    oos_lines: pd.DataFrame,
    run_id: str,
    first_train_season: str,
    features: list[str] | None = None,
) -> Bundle:
    """Every model on every training season through the latest game; the calibrator on the
    walk-forward's out-of-sample predictions (the only honest calibration data once the
    models have trained on everything)."""
    feats = list(features or FEATURE_COLUMNS)
    train = df[
        (df["season"] >= first_train_season)
        & (df["season_type"] != "preseason")
        & df["y_minutes"].notna()
    ]
    minutes_model = mm.fit(train, features=[f for f in mm.FEATURES if f in feats])
    rate_model = rm.fit(train, features=feats)
    scenario = simulate.fit_scenario(train, minutes_model, df)
    ol = oos_lines.sort_values(["game_date", "row_key"], kind="stable")
    cal = calibrate.Calibrator.fit(
        {m: g["p_raw"].to_numpy() for m, g in ol.groupby("market")},
        {m: g["hit"].to_numpy() for m, g in ol.groupby("market")},
    )
    meta = {
        "trained_through": minutes_model.trained_through,
        "train_seasons": sorted(train["season"].unique().tolist()),
        "calibration": "walk-forward out-of-sample predictions",
        "calibration_seasons": sorted(pd.to_datetime(ol["game_date"]).dt.year.unique().tolist()),
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    return Bundle(run_id, minutes_model, rate_model, scenario, cal, meta)


# --- report ----------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    a = report["layer_a"]
    lines = [
        f"# Backtest {report['run_id']}",
        "",
        f"Season {a['season']}; {a['blocks']} weekly blocks; "
        f"holdout: {report['holdout']}; git {report['git_sha'][:7]}",
        "",
    ]
    t = a["targets"]
    lines += [
        f"**Stage 5 targets out of sample: {'PASS' if t['passes'] else 'FAIL'}**",
        "",
        "| Market | MAE vs baseline | Log loss | Baseline log loss (same lines) | ECE "
        "| Decile failures | PIT | Passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, v in a["markets"].items():
        acc, pr, pit = v["accuracy"], v["probabilities"], v["pit"]
        lines.append(
            f"| {m} | {acc.get('mae_improvement', 0):+.1%} | "
            f"{pr.get('log_loss_model_same_rows', pr['log_loss_calibrated']):.4f} | "
            f"{pr.get('log_loss_baseline', float('nan')):.4f} | {pr['ece_calibrated']:.3f} | "
            f"{len(pr['calibration_failures'])} | {pit.get('verdict', '')} | "
            f"{'yes' if t['markets'][m]['passes'] else 'no'} |"
        )
    mn = a["minutes"]
    lines += [
        "",
        f"Minutes: MAE {mn['mae_model']} vs {mn['mae_baseline']} "
        f"({mn['mae_improvement']:+.1%}); 80% interval coverage {mn['coverage_80']:.1%}.",
        "",
    ]
    lb = report.get("layer_b")
    if lb:
        lines += ["## Betting replay (Layer B)", ""]
        if "skipped" in lb:
            lines += [lb["skipped"], ""]
        else:
            o = lb["bets"]["overall"]
            if o.get("bets"):
                lines += [
                    f"{o['bets']} settled bets ({o['voided']} void). Hit rate "
                    f"{o['hit_rate']:.1%} vs breakeven {o['breakeven_rate']:.1%}; average EV "
                    f"{o['avg_ev']:+.1%}; average CLV {o['avg_clv']:+.2%} "
                    f"(SE {o['avg_clv_se']:.2%}); ROI per bet {o['roi_per_bet']:+.1%} "
                    f"(SE {o['roi_per_bet_se']:.1%}).",
                    "",
                ]
                if o.get("warning"):
                    lines += [f"**{o['warning']}**", ""]
            else:
                lines += ["No bets were flagged.", ""]
            sat = lb.get("satellites", {})
            if sat.get("parlays"):
                lines += [
                    f"Satellites: {sat['parlays']} parlays, hit rate {sat['hit_rate']:.1%}, "
                    f"average EV {sat['avg_ev']:+.1%}, ROI {sat['roi_staked']:+.1%}.",
                    "",
                ]
            gr = lb.get("growth")
            if gr:
                a, b = gr["singles_only"], gr["singles_plus_satellites"]
                lines += [
                    f"Bankroll growth: singles alone {a['final_bankroll']:.3f}x (max "
                    f"drawdown {a['max_drawdown']:.1%}); with satellites "
                    f"{b['final_bankroll']:.3f}x ({b['max_drawdown']:.1%}).",
                    "",
                ]
            bl = lb["blend"]
            if bl.get("second_half"):
                sh = bl["second_half"]
                lines += [
                    f"Blend check (weight chosen on the first half of dates, judged on the "
                    f"second): chosen model weight {bl['chosen_weight']}; second-half log loss "
                    f"model {sh['model_only']}, consensus {sh['consensus_only']}, chosen blend "
                    f"{sh['chosen_weight']}, plan default 0.3 {sh['plan_default_0.3']}.",
                    "",
                ]
    if report.get("ablation"):
        lines += [
            "## With vs without the roster inputs",
            "",
            "Negative differences favor the model WITH roster inputs.",
            "",
            "| Market | Log loss diff | z | MAE diff | z |",
            "|---|---|---|---|---|",
        ]
        for m, v in report["ablation"]["markets"].items():
            ll, mae = v["log_loss"], v["mae"]
            lines.append(
                f"| {m} | {ll.get('mean_diff', 0):+.5f} | {ll.get('z', 0)} | "
                f"{mae.get('mean_diff', 0):+.4f} | {mae.get('z', 0)} |"
            )
    lines += [
        "",
        "Small samples: per-slice numbers in the JSON report are descriptive, not "
        "evidence. Bet thresholds are tuned on paper-trading data, never on this report.",
    ]
    return "\n".join(lines) + "\n"


def store_report(report: dict) -> bool:
    """Also keep the report in Postgres for the website (best effort: a backtest run on a
    laptop without a database still writes its files)."""
    try:
        from sqlalchemy.dialects.postgresql import insert

        from nbaprops.db.base import session_scope
        from nbaprops.db.models import BacktestReport

        payload = json.loads(json.dumps(report, default=str))
        with session_scope() as s:
            s.execute(
                insert(BacktestReport)
                .values(
                    run_id=report["run_id"],
                    created_ts=datetime.now(UTC),
                    season=report["layer_a"]["season"],
                    holdout=report["holdout"],
                    report=payload,
                )
                .on_conflict_do_update(index_elements=["run_id"], set_={"report": payload})
            )
        return True
    except Exception as exc:  # no database here: files only
        log.warning("report not stored in the database: %s", exc)
        return False


def write_report(report: dict, folder: Path = REPORTS / "backtest") -> Path:
    store_report(report)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    (folder / f"{report['run_id']}.md").write_text(render_markdown(report))
    return path


# --- main ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--season", required=True)
    ap.add_argument("--features", type=Path, help="training-frame Parquet instead of Postgres")
    ap.add_argument("--retrain-days", type=int, default=7)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--first-train-season", default="2022-23")
    ap.add_argument("--ablate", choices=sorted(FEATURE_GROUPS))
    ap.add_argument("--production", action="store_true")
    ap.add_argument("--layer-b", action="store_true")
    ap.add_argument(
        "--flag-at",
        default=layer_b.SCHEDULE,
        help="New York times of the daily pricing runs, comma-separated",
    )
    ap.add_argument("--rerun-holdout", metavar="REASON")
    ap.add_argument("--cache", type=Path, default=Path("artifacts/backtest_cache"))
    ap.add_argument("--reports", type=Path, default=REPORTS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ledger = args.reports / "holdout_ledger.json"
    clean = check_holdout(args.season, args.rerun_holdout, ledger)
    df = load_frame(args.season, args.features)
    run_id = f"bt-{args.season}-{datetime.now(UTC):%Y%m%d%H%M}"
    settings = wf.Settings(
        retrain_days=args.retrain_days,
        n_sims=args.n_sims,
        first_train_season=args.first_train_season,
        cache_dir=args.cache,
        label=run_id,
    )
    market = None
    if args.layer_b:
        market = layer_b.load_market(args.season)
        odds, games, _ = market
        if not odds.empty:
            settings.price_lines = layer_b.lines_to_price(odds, games, args.flag_at)
            # grids for every priced player (milestones and shifted consensus), as live
            settings.grid_keys = (
                pd.concat(
                    [
                        settings.price_lines[["game_id", "player_id", "market"]],
                        layer_b.milestone_keys(odds, games, args.flag_at),
                    ]
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )
    target, warm = wf.run(df, args.season, settings)
    report = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "created": datetime.now(UTC).isoformat(),
        "settings": {
            "retrain_days": args.retrain_days,
            "n_sims": args.n_sims,
            "first_train_season": args.first_train_season,
        },
        "holdout": ("clean" if clean else f"NOT CLEAN (rerun: {args.rerun_holdout})")
        if args.season == HOLDOUT_SEASON
        else "not the holdout season",
        "layer_a": metrics.layer_a(target),
        "warmup": metrics.layer_a(warm) if warm is not None else None,
        "blocks": target.blocks,
    }
    if market is not None:
        from nbaprops.config import load_strategy
        from nbaprops.pricing.engine import Rules
        from nbaprops.settings import get_settings

        rules = Rules.from_config(load_strategy(get_settings().strategy_config))
        report["layer_b"] = layer_b.section(
            target.priced, *market, rules, args.flag_at, target.grids
        )
    if args.ablate:
        drop = set(FEATURE_GROUPS[args.ablate])
        without = wf.Settings(
            **{
                **settings.__dict__,
                "features": [c for c in FEATURE_COLUMNS if c not in drop],
                "label": f"{run_id}-no-{args.ablate}",
            }
        )
        seed = warm.lines if warm is not None else None
        other = wf.run_season(df, args.season, without, seed)
        report["ablation"] = metrics.compare(
            other, target, f"without {args.ablate}", f"with {args.ablate}"
        )
    passes = report["layer_a"]["targets"]["passes"]
    record_holdout(args.season, run_id, clean, args.rerun_holdout, passes, ledger)
    path = write_report(report, args.reports / "backtest")
    print(render_markdown(report))
    print(f"Report: {path}")
    if args.production:
        if args.season != HOLDOUT_SEASON or not passes or not clean:
            raise SystemExit(
                "Production retrain needs a clean, passing holdout evaluation (season "
                f"{HOLDOUT_SEASON}); the report above was still written."
            )
        oos = pd.concat([x.lines for x in (warm, target) if x is not None], ignore_index=True)
        bundle = production_bundle(
            df, oos, f"prod-{datetime.now(UTC):%Y%m%d%H%M}", args.first_train_season
        )
        folder = Path("artifacts/models") / bundle.run_id
        bundle.save(folder)
        print(f"Production bundle: {folder} (register it as a challenger; Stage 11 promotes)")
    return report


if __name__ == "__main__":
    main()
