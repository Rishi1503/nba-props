"""Stage 5 (part 4): put minutes, rates, simulation and recalibration together.

evaluate  train on --train seasons, simulate the --test (tuning) season, fit the calibrator
          on its first half and judge everything on the second half, so the calibration
          check isn't graded on its own training data. Reports, per market: MAE of the
          simulated median vs the plan's season-average baseline, Brier and calibration
          (raw and recalibrated) vs a normal-distribution baseline.
train     fit a bundle (minutes, rates, scenario, calibrator) and save it.
slate     project a day's games from Postgres with a saved bundle, store `projections`
          rows under the bundle's run_id, and register the run in `model_runs`.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from nbaprops.models import calibrate, simulate
from nbaprops.models import minutes as mm
from nbaprops.models import rates as rm
from nbaprops.models.simulate import MARKETS

TEST_LINE_OFFSETS = np.array([-2, -1, 0, 1, 2]) + 0.5  # around each median (plan, Stage 6)


@dataclass
class Bundle:
    run_id: str
    minutes: mm.MinutesModel
    rates: rm.RateModel
    scenario: simulate.ScenarioParams
    calibrator: calibrate.Calibrator
    meta: dict

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.minutes.save(folder / "minutes.json")
        self.rates.save(folder / "rates.json")
        self.scenario.save(folder / "scenario.json")
        self.calibrator.save(folder / "calibrator.json")
        (folder / "meta.json").write_text(
            json.dumps({**self.meta, "run_id": self.run_id}, indent=2, default=str)
        )

    @classmethod
    def load(cls, folder: Path) -> "Bundle":
        meta = json.loads((folder / "meta.json").read_text())
        return cls(
            meta["run_id"],
            mm.MinutesModel.load(folder / "minutes.json"),
            rm.RateModel.load(folder / "rates.json"),
            simulate.ScenarioParams.load(folder / "scenario.json"),
            calibrate.Calibrator.load(folder / "calibrator.json"),
            meta,
        )


# --- simulation over many games ----------------------------------------------


def _subdist(d: mm.MinutesDist, idx: np.ndarray) -> mm.MinutesDist:
    return mm.MinutesDist(
        d.q50[idx], d.sd_left[idx], d.sd_right[idx], d.p_exit[idx], d.exit_cap[idx]
    )


def _centers(g: pd.DataFrame) -> tuple[float | None, float | None]:
    """Market total and home margin when logged (plan: the market centers the simulation)."""
    if "market_total" not in g or "market_team_margin" not in g:
        return None, None
    total = g["market_total"].astype(float).dropna()
    home = g[g["is_home"].astype(int) == 1]["market_team_margin"].astype(float).dropna()
    return (
        float(total.iloc[0]) if len(total) else None,
        float(home.iloc[0]) if len(home) else None,
    )


def simulate_frame(
    frame: pd.DataFrame,
    minutes_model: mm.MinutesModel,
    rates: rm.RateModel,
    scenario: simulate.ScenarioParams,
    run_id: str,
    n_sims: int,
    history: pd.DataFrame | None = None,
    keep_draws: bool = False,
):
    """Simulate every game in `frame`. Returns (summary per row and market, test-line table,
    and the raw draws per game if keep_draws)."""
    frame = frame.reset_index(drop=True)
    dist, msum = mm.predict(minutes_model, frame, history)
    summaries, lines_out, draws_out = [], [], {}
    for game_id, g in frame.groupby("game_id", sort=False):
        idx = g.index.to_numpy()
        total0, margin0 = _centers(g)
        sims = simulate.simulate_game(
            g,
            _subdist(dist, idx),
            rates,
            scenario,
            n_sims,
            simulate.game_seed(run_id, str(game_id)),
            total0,
            margin0,
        )
        if keep_draws:
            draws_out[game_id] = (idx, sims)
        for market in MARKETS:
            d = sims[market]
            med = np.median(d, axis=1)
            summaries.append(
                pd.DataFrame(
                    {
                        "row": idx,
                        "market": market,
                        "mean": d.mean(axis=1),
                        "median": med,
                        "sd": d.std(axis=1),
                        "low_confidence": msum["low_confidence"].to_numpy()[idx],
                    }
                )
            )
            lines = np.floor(med)[:, None] + TEST_LINE_OFFSETS[None, :]
            lines = np.maximum(lines, 0.5)
            probs = simulate.p_over(d, lines)
            block = pd.DataFrame(
                {
                    "row": np.repeat(idx, lines.shape[1]),
                    "market": market,
                    "line": lines.ravel(),
                    "p_raw": probs.ravel(),
                }
            )
            if f"y_{market}" in g:
                y = np.repeat(g[f"y_{market}"].to_numpy(float), lines.shape[1])
                block["hit"] = (y > block["line"].to_numpy()).astype(float)
            lines_out.append(block)
    summary = pd.concat(summaries, ignore_index=True)
    test_lines = pd.concat(lines_out, ignore_index=True)
    return summary, test_lines, draws_out


# --- baselines (plan, Stage 5 step 5) -----------------------------------------


def baseline_mean(rows: pd.DataFrame, market: str) -> np.ndarray:
    """Season-to-date per-game average, falling back to last-10 rate x last-10 minutes."""
    parts = ["pts", "reb", "ast"] if market == "pra" else [market]
    season = sum(rows[f"std_{p}_avg"] for p in parts)
    recent = sum(rows[f"{p}_pm_l10"] for p in parts) * rows["min_avg_l10"]
    return season.fillna(recent).to_numpy(float)


def baseline_scale(rows: pd.DataFrame, market: str) -> float:
    """k in sd = k * sqrt(mean), fitted on training rows."""
    b = baseline_mean(rows, market)
    y = rows[f"y_{market}"].to_numpy(float)
    m = np.isfinite(b) & (b > 0)
    return float(np.sqrt(np.mean((y[m] - b[m]) ** 2 / b[m]))) if m.any() else 1.5


def baseline_p_over(mean: np.ndarray, k: float, line: np.ndarray) -> np.ndarray:
    sd = k * np.sqrt(np.maximum(mean, 0.25))
    return 1 - ndtr((line - mean) / sd)


# --- evaluation --------------------------------------------------------------


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    history: pd.DataFrame,
    n_sims: int = 2000,
    run_id: str = "eval",
) -> tuple[dict, Bundle]:
    minutes_model = mm.fit(train)
    rate_model = rm.fit(train)
    scenario = simulate.fit_scenario(train, minutes_model, history)
    test = test.dropna(subset=["y_minutes"]).reset_index(drop=True)
    summary, tl, _ = simulate_frame(
        test, minutes_model, rate_model, scenario, run_id, n_sims, history
    )
    dates = test["game_date"].to_numpy()
    cut = np.quantile(dates.astype("datetime64[ns]").astype(np.int64), 0.5)
    first_half = dates.astype("datetime64[ns]").astype(np.int64) < cut
    tl["first_half"] = first_half[tl["row"].to_numpy()]
    tl["game_date"] = test["game_date"].to_numpy()[tl["row"].to_numpy()]
    tl = tl.sort_values(["game_date", "row"], kind="stable")  # calibrator needs time order
    fit_part, judge = tl[tl["first_half"]], tl[~tl["first_half"]]
    cal = calibrate.Calibrator.fit(
        {m: g["p_raw"].to_numpy() for m, g in fit_part.groupby("market")},
        {m: g["hit"].to_numpy() for m, g in fit_part.groupby("market")},
    )

    report = {
        "rows": int(len(test)),
        "n_sims": n_sims,
        "scenario": scenario.__dict__,
        "rate_rounds": rate_model.rounds,
        "v_env": rate_model.v_env,
        "v_player": rate_model.v_player,
        "markets": {},
    }
    judge_rows = ~first_half
    for market in MARKETS:
        s = summary[summary["market"] == market].set_index("row").sort_index()
        y = test[f"y_{market}"].to_numpy(float)
        base = baseline_mean(test, market)
        m = judge_rows & np.isfinite(base)
        mae_model = float(np.mean(np.abs(s["median"].to_numpy()[m] - y[m])))
        mae_base = float(np.mean(np.abs(base[m] - y[m])))
        j = judge[judge["market"] == market]
        p_raw, hit = j["p_raw"].to_numpy(), j["hit"].to_numpy()
        p_cal = cal.apply(market, p_raw)
        k = baseline_scale(train.dropna(subset=[f"y_{market}"]), market)
        b_mean = base[j["row"].to_numpy()]
        ok = np.isfinite(b_mean)
        p_base = baseline_p_over(b_mean[ok], k, j["line"].to_numpy()[ok])
        table = calibrate.reliability(p_cal, hit)
        # Errors are correlated within a player (5 lines per game share one outcome, and his
        # rate-estimation error persists across his games), so cluster by player.
        players = test["player_id"].to_numpy()[j["row"].to_numpy()]
        failures = calibrate.calibration_failures(p_cal, hit, clusters=players)
        report["markets"][market] = {
            "n_rows": int(m.sum()),
            "mae_model": round(mae_model, 3),
            "mae_baseline": round(mae_base, 3),
            "mae_improvement": round(1 - mae_model / mae_base, 4),
            "brier_raw": round(calibrate.brier(p_raw, hit), 5),
            "brier_calibrated": round(calibrate.brier(p_cal, hit), 5),
            "brier_baseline": round(calibrate.brier(p_base, hit[ok]), 5),
            "ece_raw": round(calibrate.expected_calibration_error(p_raw, hit), 4),
            "ece_calibrated": round(calibrate.expected_calibration_error(p_cal, hit), 4),
            "max_decile_error_calibrated": round(
                max((abs(r["predicted"] - r["observed"]) for r in table), default=0.0), 4
            ),
            "calibration_failures": failures,
            "reliability_calibrated": table,
        }
    mk = report["markets"]
    report["passes_mae_target"] = all(v["mae_improvement"] > 0 for v in mk.values())
    report["passes_calibration_target"] = all(not v["calibration_failures"] for v in mk.values())
    bundle = Bundle(
        run_id,
        minutes_model,
        rate_model,
        scenario,
        cal,
        {"trained_through": minutes_model.trained_through, "kind": "eval"},
    )
    return report, bundle


def fit_bundle(
    train: pd.DataFrame,
    calibrate_on: pd.DataFrame,
    history: pd.DataFrame,
    run_id: str,
    n_sims: int = 2000,
) -> Bundle:
    """Production bundle: models on `train`, calibrator on the whole calibration season."""
    minutes_model = mm.fit(train)
    rate_model = rm.fit(train)
    scenario = simulate.fit_scenario(train, minutes_model, history)
    calib = calibrate_on.dropna(subset=["y_minutes"]).reset_index(drop=True)
    _, tl, _ = simulate_frame(calib, minutes_model, rate_model, scenario, run_id, n_sims, history)
    tl["game_date"] = calib["game_date"].to_numpy()[tl["row"].to_numpy()]
    tl = tl.sort_values(["game_date", "row"], kind="stable")  # calibrator needs time order
    cal = calibrate.Calibrator.fit(
        {m: g["p_raw"].to_numpy() for m, g in tl.groupby("market")},
        {m: g["hit"].to_numpy() for m, g in tl.groupby("market")},
    )
    meta = {
        "trained_through": minutes_model.trained_through,
        "train_seasons": sorted(train["season"].unique().tolist()),
        "calibration_seasons": sorted(calib["season"].unique().tolist()),
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    return Bundle(run_id, minutes_model, rate_model, scenario, cal, meta)


# --- live slates -------------------------------------------------------------


def git_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def register_run(session, bundle: Bundle, metrics: dict | None = None) -> None:
    from sqlalchemy.dialects.postgresql import insert

    from nbaprops.config import load_strategy
    from nbaprops.db.models import ModelRun
    from nbaprops.settings import get_settings

    cfg = load_strategy(get_settings().strategy_config).model_dump()
    session.execute(
        insert(ModelRun)
        .values(
            run_id=bundle.run_id,
            git_sha=bundle.meta.get("git_sha", git_sha())[:40],
            trained_through=date.fromisoformat(bundle.minutes.trained_through),
            config=cfg,
            params={
                "scenario": bundle.scenario.__dict__,
                "rate_rounds": bundle.rates.rounds,
                "minutes_rounds": {str(k): v for k, v in bundle.minutes.best_rounds.items()},
            },
            metrics=metrics or {},
            status="challenger",
        )
        .on_conflict_do_nothing()
    )


def _draws_by_player(
    frame: pd.DataFrame, bundle: Bundle, n_sims: int, history: pd.DataFrame | None
) -> tuple[dict, dict]:
    """{(game_id, player_id): {market: draws}} and {(game_id, player_id): low_confidence}."""
    frame = frame.reset_index(drop=True)
    summary, _, draws = simulate_frame(
        frame,
        bundle.minutes,
        bundle.rates,
        bundle.scenario,
        bundle.run_id,
        n_sims,
        history,
        keep_draws=True,
    )
    out = {}
    for idx, sims in draws.values():
        for k, i in enumerate(idx):
            key = (frame.at[i, "game_id"], int(frame.at[i, "player_id"]))
            out[key] = {m: sims[m][k] for m in MARKETS}
    low = summary[summary["market"] == "pts"].set_index("row")["low_confidence"]
    flags = {
        (frame.at[i, "game_id"], int(frame.at[i, "player_id"])): bool(v) for i, v in low.items()
    }
    return out, flags


def mix_questionable(
    base: dict, alt: dict, keys: pd.DataFrame, frame: pd.DataFrame, run_id: str, n_sims: int
) -> dict:
    """For each team with a key Questionable player, every teammate's draw comes from the
    "he plays" simulation with probability p_play, else from the "he sits" one (the same
    choice for the whole team in a given run). His own draws stay "he plays": his props void
    if he sits."""
    if alt is None or not len(keys):
        return base
    team_of = {
        (g, int(p)): t
        for g, p, t in zip(frame["game_id"], frame["player_id"], frame["team_id"], strict=True)
    }
    mixed = dict(base)
    for k in keys.itertuples(index=False):
        rng = np.random.default_rng(
            simulate.game_seed(f"{run_id}:questionable", f"{k.game_id}:{k.team_id}")
        )
        plays = rng.random(n_sims) < k.p_play
        for key, draws in base.items():
            if key[0] != k.game_id or team_of.get(key) != k.team_id or key[1] == k.player_id:
                continue
            if key not in alt:
                continue
            mixed[key] = {m: np.where(plays, draws[m], alt[key][m]) for m in MARKETS}
    return mixed


def _grid(lines: np.ndarray, p: np.ndarray) -> dict:
    return {f"{ln:.1f}": round(float(pv), 4) for ln, pv in zip(lines, p, strict=True)}


def project_frame(
    frame: pd.DataFrame,
    bundle: Bundle,
    n_sims: int,
    history: pd.DataFrame | None,
    as_of: datetime,
    alt_frame: pd.DataFrame | None = None,
    questionable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (game, player, market) with calibrated P(over) on a line grid. With
    `alt_frame` and `questionable` (from `questionable_scenarios`), teammates of a key
    Questionable player are mixed over whether he plays, and their row also carries both
    scenario grids ("in": he plays, "out": he sits) at the same lines, so pricing can require
    a bet to hold up either way. low_confidence is the minutes model's own flag only."""
    frame = frame.reset_index(drop=True)
    base, flags = _draws_by_player(frame, bundle, n_sims, history)
    draws, alt = base, None
    q_of: dict = {}
    if alt_frame is not None and questionable is not None and len(questionable):
        alt, _ = _draws_by_player(alt_frame, bundle, n_sims, history)
        draws = mix_questionable(base, alt, questionable, frame, bundle.run_id, n_sims)
        q_of = {
            (k.game_id, k.team_id): (int(k.player_id), float(k.p_play))
            for k in questionable.itertuples(index=False)
        }
    team_of = {
        (g, int(p)): t
        for g, p, t in zip(frame["game_id"], frame["player_id"], frame["team_id"], strict=True)
    }
    cal = bundle.calibrator
    rows = []
    for (game_id, player_id), by_market in draws.items():
        q = q_of.get((game_id, team_of[(game_id, player_id)]))
        mixed = (
            q is not None and player_id != q[0] and alt is not None and (game_id, player_id) in alt
        )
        for market, d in by_market.items():
            lines = np.array(simulate.line_grid(d[None, :])[0])
            p = cal.apply(market, (d[None, :] > lines[:, None]).mean(1))
            scen = None
            if mixed:
                # Calibrate each scenario (the calibrator learned on games where everyone's
                # participation was known), then mix: the stored blend is then exactly
                # p_play * in + (1 - p_play) * out, and can be re-blended with a measured rate.
                p_in = cal.apply(
                    market, (base[(game_id, player_id)][market][None, :] > lines[:, None]).mean(1)
                )
                p_out = cal.apply(
                    market, (alt[(game_id, player_id)][market][None, :] > lines[:, None]).mean(1)
                )
                p = q[1] * p_in + (1 - q[1]) * p_out
                scen = {
                    "player_id": q[0],
                    "p_play": q[1],
                    "in": _grid(lines, p_in),
                    "out": _grid(lines, p_out),
                }
            rows.append(
                {
                    "game_id": game_id,
                    "player_id": player_id,
                    "market": market,
                    "as_of_ts": as_of,
                    "mean": float(d.mean()),
                    "sd": float(d.std()),
                    "p_over": _grid(lines, p),
                    "low_confidence": flags[(game_id, player_id)],
                    "scenarios": scen,
                }
            )
    return pd.DataFrame(rows)


def store_projections(session, run_id: str, projections: pd.DataFrame) -> int:
    from sqlalchemy import delete
    from sqlalchemy.dialects.postgresql import insert

    from nbaprops.db.models import Projection
    from nbaprops.pipeline.reference import ensure_player

    if not len(projections):
        return 0
    # A projection run replaces each game's set: a player ruled out since the last run must
    # not keep his old row (pricing would see injury news "after" it and skip the game).
    games = [str(g) for g in projections["game_id"].unique()]
    session.execute(
        delete(Projection).where(Projection.run_id == run_id, Projection.game_id.in_(games))
    )
    for pid in projections["player_id"].unique():
        ensure_player(session, int(pid), "")  # no made-up name: it would become an alias
    recs = [{**r, "run_id": run_id} for r in projections.to_dict("records")]
    stmt = insert(Projection).values(recs)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["run_id", "game_id", "player_id", "market"],
            set_={
                c: stmt.excluded[c]
                for c in ("as_of_ts", "mean", "sd", "p_over", "low_confidence", "scenarios")
            },
        )
    )
    return len(recs)


def project_slate(
    session,
    bundle: Bundle,
    game_date: date,
    as_of: datetime,
    n_sims: int = 10000,
    extra_players: list[tuple[int, int]] | None = None,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Project a day's games (or only `game_ids`, for a late-news rerun) and store them."""
    from nbaprops.features.build import (
        build_feature_scenarios,
        build_training_set,
        prev_season,
    )
    from nbaprops.pipeline.stats import season_for_date

    frame, alt, questionable = build_feature_scenarios(session, game_date, as_of, extra_players)
    if game_ids is not None:
        frame = frame[frame["game_id"].isin(game_ids)]
        if alt is not None:
            alt = alt[alt["game_id"].isin(game_ids)]
        questionable = questionable[questionable["game_id"].isin(game_ids)]
    if not len(frame):
        return pd.DataFrame()
    season = season_for_date(game_date)
    # Finished games for early-exit rates (without this every player would get the league rate)
    history = build_training_set(session, [prev_season(season), season])
    projections = project_frame(frame, bundle, n_sims, history, as_of, alt, questionable)
    register_run(session, bundle)
    store_projections(session, bundle.run_id, projections)
    return projections


# --- command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="nbaprops.models.project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "train"):
        p = sub.add_parser(name)
        p.add_argument("--features", default="data/features/train.parquet")
        p.add_argument("--train", nargs="+", default=["2022-23", "2023-24"])
        p.add_argument("--sims", type=int, default=2000)
        p.add_argument("--out", default="artifacts/stage5")
    sub.choices["evaluate"].add_argument("--test", nargs="+", default=["2024-25"])
    sub.choices["train"].add_argument("--calibrate", nargs="+", default=["2024-25"])
    sl = sub.add_parser("slate")
    sl.add_argument("--bundle", required=True)
    sl.add_argument("--date", help="YYYY-MM-DD (default: today in New York)")
    sl.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args(argv)

    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    if args.cmd in ("evaluate", "train"):
        df = pd.read_parquet(args.features)
        df = df[df["season_type"] != "preseason"]
        train = df[df["season"].isin(args.train)]
        other = args.test if args.cmd == "evaluate" else args.calibrate
        held = df[df["season"].isin(other)]
        history = df[df["season"].isin(args.train + other)]
        out = Path(args.out)
        if args.cmd == "evaluate":
            report, _ = evaluate(train, held, history, args.sims, f"eval-{stamp}")
            out.mkdir(parents=True, exist_ok=True)
            (out / f"eval-{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
            brief = {
                m: {
                    k: v[k]
                    for k in (
                        "mae_improvement",
                        "brier_calibrated",
                        "brier_baseline",
                        "ece_calibrated",
                        "calibration_failures",
                    )
                }
                for m, v in report["markets"].items()
            }
            print(
                json.dumps(
                    {
                        "markets": brief,
                        "passes_mae_target": report["passes_mae_target"],
                        "passes_calibration_target": report["passes_calibration_target"],
                    },
                    indent=2,
                )
            )
        else:
            bundle = fit_bundle(train, held, history, f"stage5-{stamp}", args.sims)
            bundle.save(out / bundle.run_id)
            print(f"saved {out / bundle.run_id}")
    else:
        from zoneinfo import ZoneInfo

        from nbaprops.db.base import session_scope

        bundle = Bundle.load(Path(args.bundle))
        now = datetime.now(UTC)
        day = (
            date.fromisoformat(args.date)
            if args.date
            else now.astimezone(ZoneInfo("America/New_York")).date()
        )
        with session_scope() as s:
            proj = project_slate(s, bundle, day, now, args.sims)
        print(f"{len(proj)} projections stored under {bundle.run_id}")


if __name__ == "__main__":
    main()
