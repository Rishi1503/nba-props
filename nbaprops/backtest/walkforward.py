"""Walk-forward backtest (plan, Stage 6).

A season is replayed in blocks of `retrain_days`. Before each block, every model is retrained
on games strictly before the block's first day, exactly as weekly production retraining
would, and the block's games are simulated with those models. Nothing in a block can see
that block or anything after it:

  - features are leakage-proof by construction (Stage 3: state after earlier games only);
  - models train on rows dated before the block (`train_cutoff`);
  - early-exit history is cut at the block's end, and only counts games before each row;
  - the calibrator is fit only on out-of-sample predictions for games before the block.

The calibrator is seeded with the previous season's out-of-sample predictions (a "warm-up"
walk-forward), so the first weeks of the target season are calibrated the way production
would be. Those same out-of-sample predictions are what the production model is calibrated on.

Trained models are cached per block (keyed by cutoff, feature list and a fingerprint of the
training rows), so reruns and with/without comparisons don't retrain what hasn't changed.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbaprops.features.build import FEATURE_COLUMNS, prev_season
from nbaprops.models import calibrate, simulate
from nbaprops.models import minutes as mm
from nbaprops.models import rates as rm
from nbaprops.models.project import (
    baseline_mean,
    baseline_p_over,
    baseline_scale,
    simulate_frame,
)
from nbaprops.models.simulate import MARKETS

log = logging.getLogger(__name__)

ROSTER_CHANGE_COLS = ("departed_min_sum", "arrived_min_sum", "returning_min_sum", "absent_n")


@dataclass
class Settings:
    retrain_days: int = 7  # plan: retrain weekly, matching production
    n_sims: int = 2000
    first_train_season: str = "2022-23"  # 2021-22 only warms up priors (Stage 1)
    features: list[str] | None = None  # None = every input; a subset for comparisons
    cache_dir: Path | None = None
    label: str = "wf"
    pit_seed: int = 20261020
    # DraftKings lines to price as well (game_id, player_id, market, line), for Layer B
    price_lines: pd.DataFrame | None = None
    # (game_id, player_id, market) needing a full probability grid (milestones), for Layer B
    grid_keys: pd.DataFrame | None = None


@dataclass
class Result:
    season: str
    lines: pd.DataFrame  # out-of-sample test lines: one row per (player-game, market, line)
    rows: pd.DataFrame  # one row per (player-game, market): projection, outcome, PIT
    minutes: pd.DataFrame  # one row per player-game: minutes quantiles and outcome
    blocks: list[dict] = field(default_factory=list)  # what was trained before each block
    # model probabilities at requested lines: p_over, p_under, p_push (calibrated)
    priced: pd.DataFrame = field(default_factory=pd.DataFrame)
    # calibrated half-point grids {"24.5": p, ...}, stored exactly as live projections are
    grids: pd.DataFrame = field(default_factory=pd.DataFrame)


# --- blocks --------------------------------------------------------------------


def blocks(dates: pd.Series, retrain_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Half-open [start, end) windows covering every date, starting at the first one; windows
    with no games are skipped (the All-Star break, the gap before the playoffs)."""
    days = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if days.empty:
        return []
    out, start, last = [], days.min(), days.max()
    unique = set(days)
    while start <= last:
        end = start + timedelta(days=retrain_days)
        if any(start <= d < end for d in unique):
            out.append((start, end))
        start = end
    return out


# --- one block -------------------------------------------------------------------


def _fingerprint(train: pd.DataFrame) -> str:
    """Changes whenever the training rows do (new games, corrected box scores)."""
    cols = ["y_minutes", "y_pts", "y_reb", "y_ast", "y_fg3m"]
    sums = train[[c for c in cols if c in train]].sum().round(3).tolist()
    return f"{len(train)}:{train['game_date'].max()}:{sums}"


# Source files whose code decides what a trained model is. The feature builder is included
# because its output isn't in the data fingerprint (which covers outcomes only).
MODEL_SOURCES = (
    "models/minutes.py",
    "models/rates.py",
    "models/simulate.py",
    "models/gate.py",
    "features/build.py",
)


def code_version() -> str:
    """Hash of the model and feature code: any change invalidates cached models."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for rel in MODEL_SOURCES:
        h.update((root / rel).read_bytes())
    return h.hexdigest()[:12]


def _cache_key(cutoff, features: list[str], fingerprint: str) -> str:
    raw = json.dumps([str(cutoff), sorted(features), fingerprint, code_version()], default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def train_models(
    train: pd.DataFrame, history: pd.DataFrame, features: list[str], cache: Path | None, cutoff
) -> tuple:
    """Minutes, rates and scenario models on `train`, from the cache when unchanged."""
    minute_feats = [f for f in mm.FEATURES if f in features]
    key = _cache_key(cutoff, features, _fingerprint(train))
    folder = cache / key if cache else None
    if folder and (folder / "scenario.json").exists():
        return (
            mm.MinutesModel.load(folder / "minutes.json"),
            rm.RateModel.load(folder / "rates.json"),
            simulate.ScenarioParams.load(folder / "scenario.json"),
            True,
        )
    minutes_model = mm.fit(train, features=minute_feats)
    rate_model = rm.fit(train, features=features)
    scenario = simulate.fit_scenario(train, minutes_model, history)
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        minutes_model.save(folder / "minutes.json")
        rate_model.save(folder / "rates.json")
        scenario.save(folder / "scenario.json")  # written last: marks a complete entry
    return minutes_model, rate_model, scenario, False


def _calibrator(prior_lines: pd.DataFrame) -> calibrate.Calibrator:
    """Fit only on out-of-sample predictions for games already played, in time order."""
    if prior_lines.empty:
        return calibrate.Calibrator()
    pl = prior_lines.sort_values(["game_date", "row_key"], kind="stable")
    return calibrate.Calibrator.fit(
        {m: g["p_raw"].to_numpy() for m, g in pl.groupby("market")},
        {m: g["hit"].to_numpy() for m, g in pl.groupby("market")},
    )


def _pit(draws: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized probability integral transform for counts: uniform on [F(y-1), F(y)].
    Flat across rows when the simulated distribution has the right shape."""
    below = (draws < y[:, None]).mean(axis=1)
    at_or_below = (draws <= y[:, None]).mean(axis=1)
    return below + rng.random(len(y)) * (at_or_below - below)


def price_at_lines(
    lines: pd.DataFrame,
    f: pd.DataFrame,
    draws: dict,
    msum: pd.DataFrame,
    calibrator: calibrate.Calibrator,
) -> pd.DataFrame:
    """Calibrated over/under/push at arbitrary lines from the block's own draws. Stats are
    whole numbers, so over a whole line L is over L + 0.5 and the push is
    P(over L - 0.5) - P(over L + 0.5); the calibrator is applied at those half-points."""
    rows = f[["game_id", "player_id"]].reset_index().rename(columns={"index": "r"})
    pl = lines.merge(rows, on=["game_id", "player_id"], how="inner")
    if pl.empty:
        return pd.DataFrame()
    where = {}
    for gid, (idx, _) in draws.items():
        for k, r in enumerate(idx):
            where[r] = (gid, k)
    out = []
    for market, part in pl.groupby("market"):
        line = part["line"].to_numpy(float)
        whole = np.isclose(line % 1, 0)
        hi = np.where(whole, line + 0.5, line)
        lo = np.where(whole, line - 0.5, line)
        d = np.stack([draws[where[r][0]][1][market][where[r][1]] for r in part["r"]])
        c_hi = calibrator.apply(market, (d > hi[:, None]).mean(axis=1))
        c_lo = calibrator.apply(market, (d > lo[:, None]).mean(axis=1))
        push = np.where(whole, np.clip(c_lo - c_hi, 0, 1), 0.0)
        out.append(
            part.drop(columns="r").assign(
                p_over=c_hi,
                p_push=push,
                p_under=np.clip(1 - c_hi - push, 0, 1),
                low_confidence=msum["low_confidence"].to_numpy()[part["r"].to_numpy()],
            )
        )
    return pd.concat(out, ignore_index=True)


def grids_at(
    keys: pd.DataFrame,
    f: pd.DataFrame,
    draws: dict,
    msum: pd.DataFrame,
    calibrator: calibrate.Calibrator,
) -> pd.DataFrame:
    """Calibrated P(over) grids (1st-99th percentile half-lines) for the requested players,
    the same shape as the live `projections.p_over` column."""
    rows = f[["game_id", "player_id"]].reset_index().rename(columns={"index": "r"})
    k = keys.drop_duplicates().merge(rows, on=["game_id", "player_id"], how="inner")
    if k.empty:
        return pd.DataFrame()
    where = {r: (gid, j) for gid, (idx, _) in draws.items() for j, r in enumerate(idx)}
    out = []
    for g in k.itertuples(index=False):
        gid, j = where[g.r]
        d = np.asarray(draws[gid][1][g.market][j])
        lines = np.asarray(simulate.line_grid(d[None, :])[0])
        p = calibrator.apply(g.market, (d[None, :] > lines[:, None]).mean(axis=1))
        out.append(
            {
                "game_id": g.game_id,
                "player_id": g.player_id,
                "market": g.market,
                "p_over": {f"{ln:.1f}": round(float(v), 4) for ln, v in zip(lines, p, strict=True)},
                "low_confidence": bool(msum["low_confidence"].to_numpy()[g.r]),
            }
        )
    return pd.DataFrame(out)


def run_block(
    frame: pd.DataFrame,
    models: tuple,
    calibrator: calibrate.Calibrator,
    k_baseline: dict,
    history: pd.DataFrame,
    settings: Settings,
    block_label: str,
) -> tuple:
    """Simulate one block day by day (bounded memory); returns (lines, rows, minutes,
    priced lines)."""
    minutes_model, rate_model, scenario = models
    out_lines, out_rows, out_min, out_priced, out_grids = [], [], [], [], []
    for day, day_frame in frame.groupby("game_date", sort=True):
        f = day_frame.reset_index(drop=True)
        run_id = f"{settings.label}:{block_label}"
        summary, tl, draws = simulate_frame(
            f,
            minutes_model,
            rate_model,
            scenario,
            run_id,
            settings.n_sims,
            history,
            keep_draws=True,
        )
        _, msum = mm.predict(minutes_model, f, history)
        key = f["game_id"].astype(str) + ":" + f["player_id"].astype(str)
        # lines
        tl["row_key"] = key.to_numpy()[tl["row"].to_numpy()]
        for c in ("game_id", "player_id", "team_id", "game_date", "season_type"):
            tl[c] = f[c].to_numpy()[tl["row"].to_numpy()]
        tl["p_cal"] = np.nan
        tl["p_base"] = np.nan
        for market, idx in tl.groupby("market").groups.items():
            part = tl.loc[idx]
            tl.loc[idx, "p_cal"] = calibrator.apply(market, part["p_raw"].to_numpy())
            b = baseline_mean(f, market)[part["row"].to_numpy()]
            tl.loc[idx, "p_base"] = np.where(
                np.isfinite(b),
                baseline_p_over(np.nan_to_num(b), k_baseline[market], part["line"].to_numpy()),
                np.nan,
            )
        out_lines.append(tl)
        if settings.price_lines is not None and len(settings.price_lines):
            out_priced.append(price_at_lines(settings.price_lines, f, draws, msum, calibrator))
        if settings.grid_keys is not None and len(settings.grid_keys):
            out_grids.append(grids_at(settings.grid_keys, f, draws, msum, calibrator))
        # rows with PIT
        rng = np.random.default_rng([settings.pit_seed, int(pd.Timestamp(day).strftime("%Y%m%d"))])
        pits = {m: np.full(len(f), np.nan) for m in MARKETS}
        for idx, sims in draws.values():
            for m in MARKETS:
                y = f[f"y_{m}"].to_numpy(float)[idx]
                pits[m][idx] = _pit(sims[m], y, rng)
        changed = np.zeros(len(f), dtype=bool)
        for c in ROSTER_CHANGE_COLS:
            if c in f:
                changed |= f[c].fillna(0).to_numpy() > 0
        for m in MARKETS:
            s = summary[summary["market"] == m].set_index("row").sort_index()
            out_rows.append(
                pd.DataFrame(
                    {
                        "row_key": key,
                        "game_id": f["game_id"],
                        "player_id": f["player_id"],
                        "team_id": f["team_id"],
                        "game_date": f["game_date"],
                        "season_type": f["season_type"],
                        "market": m,
                        "mean": s["mean"].to_numpy(),
                        "median": s["median"].to_numpy(),
                        "sd": s["sd"].to_numpy(),
                        "low_confidence": s["low_confidence"].to_numpy(),
                        "y": f[f"y_{m}"].to_numpy(float),
                        "pit": pits[m],
                        "baseline": baseline_mean(f, m),
                        "roster_change": changed,
                        # rotation minutes missing from the player's team (pre-game known)
                        "teammates_out_min": f["absent_min_sum"].fillna(0).to_numpy(float)
                        if "absent_min_sum" in f
                        else 0.0,
                        "same_pos_out_min": f["same_pos_absent_min"].fillna(0).to_numpy(float)
                        if "same_pos_absent_min" in f
                        else 0.0,
                    }
                )
            )
        out_min.append(
            pd.DataFrame(
                {
                    "row_key": key,
                    "player_id": f["player_id"],
                    "game_date": f["game_date"],
                    "q10": msum["q10"].to_numpy(),
                    "q50": msum["q50"].to_numpy(),
                    "q90": msum["q90"].to_numpy(),
                    "y": f["y_minutes"].to_numpy(float),
                    "baseline": f["min_avg_l10"].to_numpy(float),
                    "low_confidence": msum["low_confidence"].to_numpy(),
                }
            )
        )
    priced = pd.concat(out_priced, ignore_index=True) if out_priced else pd.DataFrame()
    grids = pd.concat(out_grids, ignore_index=True) if out_grids else pd.DataFrame()
    return (
        pd.concat(out_lines, ignore_index=True),
        pd.concat(out_rows, ignore_index=True),
        pd.concat(out_min, ignore_index=True),
        priced,
        grids,
    )


# --- one season --------------------------------------------------------------------


def run_season(
    df: pd.DataFrame, season: str, settings: Settings, seed_lines: pd.DataFrame | None = None
) -> Result:
    """Walk-forward over `season`. `df` is the full training frame (every season loaded);
    `seed_lines` are earlier out-of-sample lines that start the calibrator."""
    features = list(settings.features or FEATURE_COLUMNS)
    df = df[(df["season_type"] != "preseason") & df["y_minutes"].notna()]
    target = df[df["season"] == season]
    eligible = df[df["season"] >= settings.first_train_season]
    prior = seed_lines.copy() if seed_lines is not None else pd.DataFrame()
    lines, rows, mins, priced, grid_parts, info = [], [], [], [], [], []
    for start, end in blocks(target["game_date"], settings.retrain_days):
        train = eligible[eligible["game_date"] < start]
        frame = target[(target["game_date"] >= start) & (target["game_date"] < end)]
        if train.empty or frame.empty:
            continue
        history = df[df["game_date"] < end]  # exit rates only count games before each row
        models = train_models(train, history, features, settings.cache_dir, start)
        cal = _calibrator(prior)
        k = {m: baseline_scale(train.dropna(subset=[f"y_{m}"]), m) for m in MARKETS}
        label = f"{start:%Y-%m-%d}"
        bl, br, bm, bp, bg = run_block(frame, models[:3], cal, k, history, settings, label)
        priced.append(bp)
        grid_parts.append(bg)
        bl["block"] = label
        lines.append(bl)
        rows.append(br)
        mins.append(bm)
        prior = pd.concat([prior, bl], ignore_index=True)
        info.append(
            {
                "block": label,
                "train_rows": int(len(train)),
                "rows": int(len(frame)),
                "trained_through": str(train["game_date"].max().date()),
                "cached": models[3],
                "minutes_rounds": models[0].best_rounds,
                "rate_rounds": models[1].rounds,
                "calibrated_markets": sorted(cal.curves),
            }
        )
        log.info("%s block %s: %d rows (train %d)", season, label, len(frame), len(train))
    if not lines:
        raise ValueError(f"no games to backtest in {season}")
    return Result(
        season,
        pd.concat(lines, ignore_index=True),
        pd.concat(rows, ignore_index=True),
        pd.concat(mins, ignore_index=True),
        info,
        pd.concat(priced, ignore_index=True) if priced else pd.DataFrame(),
        pd.concat(grid_parts, ignore_index=True) if grid_parts else pd.DataFrame(),
    )


def run(
    df: pd.DataFrame, season: str, settings: Settings, warmup: bool = True
) -> tuple[Result, Result | None]:
    """Target season, preceded by the previous season as a calibration warm-up when there is
    training data before it. Returns (target, warm-up or None)."""
    warm = None
    before = prev_season(season)
    if warmup and before > settings.first_train_season and (df["season"] == before).any():
        warm = run_season(df, before, settings)
    seed = warm.lines if warm is not None else None
    return run_season(df, season, settings, seed), warm
