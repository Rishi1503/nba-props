"""Layer A: projection accuracy (plan, Stage 6), from a walk-forward Result.

Per market: MAE and RMSE against the baseline; calibration of P(over) on test lines near each
projection (raw and recalibrated, with failures judged by player-clustered standard errors);
Brier score and log loss against the baseline's probabilities; and a PIT histogram, whose
shape says whether the simulated distribution is too narrow (U-shape), too wide (hump) or
shifted (sloped). Breakdowns show where the model is weak: by month, by low-confidence flag,
around roster changes, and in the postseason.

`compare` pairs two runs on the same games, with standard errors clustered by player: it
backs the with/without-roster comparison now and champion-challenger promotion later.
"""

import numpy as np
import pandas as pd

from nbaprops.models import calibrate
from nbaprops.models.simulate import MARKETS

P_EPS = 1e-4
ECE_TARGET = 0.03  # plan, Stage 5: every market within 3 points
PIT_BINS = 10


def log_loss(p: np.ndarray, hit: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), P_EPS, 1 - P_EPS)
    y = np.asarray(hit, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ll_rows(p: np.ndarray, hit: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), P_EPS, 1 - P_EPS)
    y = np.asarray(hit, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pit_summary(pit: np.ndarray) -> dict:
    """Histogram plus a plain-language verdict. Tolerances are deliberately loose: with real
    seasons (tens of thousands of rows) anything flagged is a real shape problem."""
    u = np.asarray(pit, float)
    u = u[np.isfinite(u)]
    if not len(u):
        return {"n": 0}
    counts, _ = np.histogram(u, bins=PIT_BINS, range=(0, 1))
    share = counts / counts.sum()
    lo, hi = float(share[0]), float(share[-1])
    outer = lo + hi
    mean = float(u.mean())
    # Each tail bin should hold ~10%. Too narrow = BOTH tails heavy; too wide = both light.
    # A shift piles outcomes into ONE tail, so it is judged from the mean instead: the fixes
    # differ (widen vs re-center), so the diagnosis must tell them apart.
    issues = []
    if abs(mean - 0.5) > 0.03:
        issues.append(
            "shifted low: outcomes tend to exceed projections"
            if mean > 0.5
            else "shifted high: outcomes tend to fall short of projections"
        )
    if min(lo, hi) > 0.13:
        issues.append("too narrow: outcomes land in both tails too often (U-shape)")
    elif max(lo, hi) < 0.07:
        issues.append("too wide: outcomes cluster in the middle (hump)")
    verdict = "; ".join(issues) or "flat: the distribution's shape is right"
    return {
        "n": int(len(u)),
        "share": [round(float(s), 4) for s in share],
        "outer_share": round(outer, 4),
        "mean": round(mean, 4),
        "verdict": verdict,
    }


def _accuracy(r: pd.DataFrame) -> dict:
    ok = np.isfinite(r["baseline"].to_numpy(float))
    rr = r[ok]
    if rr.empty:
        return {"n": 0}
    err_m = rr["median"] - rr["y"]
    err_b = rr["baseline"] - rr["y"]
    mae_m, mae_b = float(err_m.abs().mean()), float(err_b.abs().mean())
    rmse_m = float(np.sqrt(((rr["mean"] - rr["y"]) ** 2).mean()))
    rmse_b = float(np.sqrt((err_b**2).mean()))
    return {
        "n": int(len(rr)),
        "mae_model": round(mae_m, 4),
        "mae_baseline": round(mae_b, 4),
        "mae_improvement": round(1 - mae_m / mae_b, 4) if mae_b else None,
        "rmse_model": round(rmse_m, 4),
        "rmse_baseline": round(rmse_b, 4),
    }


def _probabilities(ln: pd.DataFrame) -> dict:
    hit = ln["hit"].to_numpy(float)
    out = {"n_lines": int(len(ln)), "n_player_games": int(ln["row_key"].nunique())}
    for name, col in (("raw", "p_raw"), ("calibrated", "p_cal")):
        p = ln[col].to_numpy(float)
        out[f"brier_{name}"] = round(calibrate.brier(p, hit), 5)
        out[f"log_loss_{name}"] = round(log_loss(p, hit), 5)
        out[f"ece_{name}"] = round(calibrate.expected_calibration_error(p, hit), 4)
    b = ln[np.isfinite(ln["p_base"].to_numpy(float))]
    if len(b):
        out["log_loss_baseline"] = round(log_loss(b["p_base"], b["hit"]), 5)
        out["log_loss_model_same_rows"] = round(log_loss(b["p_cal"], b["hit"]), 5)
        out["brier_baseline"] = round(
            calibrate.brier(b["p_base"].to_numpy(), b["hit"].to_numpy()), 5
        )
    p_cal = ln["p_cal"].to_numpy(float)
    out["reliability"] = calibrate.reliability(p_cal, hit)
    out["calibration_failures"] = calibrate.calibration_failures(
        p_cal, hit, clusters=ln["player_id"].to_numpy()
    )
    return out


def _breakdowns(r: pd.DataFrame, ln: pd.DataFrame) -> dict:
    """MAE improvement and calibrated log loss by slice, so weak spots are visible."""
    slices = {
        "month": (r["game_date"].dt.strftime("%Y-%m"), ln["game_date"].dt.strftime("%Y-%m")),
        "low_confidence": (
            r["low_confidence"].astype(bool),
            ln["row_key"].map(r.set_index("row_key")["low_confidence"]).astype(bool),
        ),
        "roster_change": (
            r["roster_change"].astype(bool),
            ln["row_key"].map(r.set_index("row_key")["roster_change"]).astype(bool),
        ),
        "postseason": (
            r["season_type"].isin(["playoffs", "playin"]),
            ln["season_type"].isin(["playoffs", "playin"]),
        ),
    }
    out = {}
    for name, (rk, lk) in slices.items():
        part = {}
        for key in sorted(pd.unique(rk.astype(str))):
            rs, ls = r[rk.astype(str) == key], ln[lk.astype(str) == key]
            acc = _accuracy(rs)
            part[key] = {
                "rows": int(len(rs)),
                "mae_improvement": acc.get("mae_improvement"),
                "log_loss": round(log_loss(ls["p_cal"], ls["hit"]), 5) if len(ls) else None,
            }
        out[name] = part
    return out


def minutes_report(mins: pd.DataFrame) -> dict:
    ok = mins[np.isfinite(mins["baseline"].to_numpy(float))]
    mae_m = float((ok["q50"] - ok["y"]).abs().mean())
    mae_b = float((ok["baseline"] - ok["y"]).abs().mean())
    inside = (mins["y"] >= mins["q10"]) & (mins["y"] <= mins["q90"])
    return {
        "n": int(len(ok)),
        "mae_model": round(mae_m, 4),
        "mae_baseline": round(mae_b, 4),
        "mae_improvement": round(1 - mae_m / mae_b, 4) if mae_b else None,
        "coverage_80": round(float(inside.mean()), 4),
    }


def layer_a(res) -> dict:
    """Full Layer A report for one walk-forward Result."""
    report = {
        "season": res.season,
        "markets": {},
        "minutes": minutes_report(res.minutes),
        "blocks": len(res.blocks),
    }
    for m in MARKETS:
        r = res.rows[res.rows["market"] == m]
        ln = res.lines[res.lines["market"] == m]
        report["markets"][m] = {
            "accuracy": _accuracy(r),
            "probabilities": _probabilities(ln),
            "pit": pit_summary(r["pit"].to_numpy()),
            "breakdowns": _breakdowns(r, ln),
        }
    report["targets"] = stage5_targets(report)
    return report


def stage5_targets(report: dict) -> dict:
    """Plan, Stage 5 'done when', applied out of sample: each market beats its baseline on
    MAE, its recalibrated ECE is under 3 points, and no decile fails the clustered check."""
    per = {}
    for m, v in report["markets"].items():
        acc, pr = v["accuracy"], v["probabilities"]
        per[m] = {
            "beats_baseline_mae": bool(
                acc.get("mae_improvement") is not None and acc["mae_improvement"] > 0
            ),
            "ece_under_target": bool(pr["ece_calibrated"] < ECE_TARGET),
            "no_decile_failures": not pr["calibration_failures"],
        }
        per[m]["passes"] = all(per[m].values())
    return {"markets": per, "passes": all(v["passes"] for v in per.values())}


# --- paired comparison of two runs --------------------------------------------------


def _paired(a: np.ndarray, b: np.ndarray, clusters: np.ndarray) -> dict:
    """Mean of (b - a) with a standard error clustered by `clusters`."""
    d = np.asarray(b, float) - np.asarray(a, float)
    n = len(d)
    if n < 2:
        return {"n": int(n)}
    resid = pd.Series(d - d.mean()).groupby(np.asarray(clusters)).sum()
    se = float(np.sqrt((resid**2).sum()) / n)
    z = float(d.mean() / se) if se > 0 else 0.0
    return {
        "n": int(n),
        "mean_diff": round(float(d.mean()), 6),
        "se": round(se, 6),
        "z": round(z, 2),
    }


def compare(a, b, label_a: str = "a", label_b: str = "b") -> dict:
    """Paired comparison of two Results on the same games. Negative log-loss and MAE
    differences mean `b` is better. 'better'/'worse' need |z| > 2 (clustered by player)."""
    out = {"a": label_a, "b": label_b, "markets": {}}
    for m in MARKETS:
        la = a.lines[a.lines["market"] == m]
        lb = b.lines[b.lines["market"] == m]
        j = la.merge(lb, on=["row_key", "line"], suffixes=("_a", "_b"))
        ll = _paired(
            _ll_rows(j["p_cal_a"], j["hit_a"]),
            _ll_rows(j["p_cal_b"], j["hit_b"]),
            j["player_id_a"].to_numpy(),
        )
        ra = a.rows[a.rows["market"] == m]
        rb = b.rows[b.rows["market"] == m]
        k = ra.merge(rb, on="row_key", suffixes=("_a", "_b"))
        mae = _paired(
            (k["median_a"] - k["y_a"]).abs(),
            (k["median_b"] - k["y_b"]).abs(),
            k["player_id_a"].to_numpy(),
        )

        def verdict(x: dict) -> str:
            if "z" not in x:
                return "too few rows"
            if x["z"] < -2:
                return f"{label_b} better"
            if x["z"] > 2:
                return f"{label_b} worse"
            return "no significant difference"

        out["markets"][m] = {
            "log_loss": {**ll, "verdict": verdict(ll)},
            "mae": {**mae, "verdict": verdict(mae)},
        }
    return out
