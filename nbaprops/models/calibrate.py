"""Stage 5 (part 3): recalibration (plan, Stage 5 and design review).

Simulated probabilities are rarely calibrated out of the box, especially far from the median.
One isotonic regression per market maps raw P(over) to observed frequency. It is fit on the
tuning season (plan) and refit monthly on paper-trading results (Stage 9). Every probability
the pricing engine sees goes through `apply`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_SAMPLES = 500  # below this, a market keeps its raw probabilities
FIT_BINS = 50  # isotonic is fit on probability-binned averages: smoother, less overfit
P_MIN, P_MAX = 0.001, 0.999


@dataclass
class Calibrator:
    curves: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, p_raw: dict[str, np.ndarray], hit: dict[str, np.ndarray], holdout: float = 0.3
    ) -> "Calibrator":
        """Arrays must be in time order. Each market's curve is fit on the earlier part and
        kept only if it beats raw probabilities (Brier) on the later `holdout` part; if kept,
        it is refit on everything. Otherwise that market passes through unchanged."""
        curves = {}
        for market, p in p_raw.items():
            y = np.asarray(hit[market], dtype=float)
            p = np.asarray(p, dtype=float)
            if len(p) < MIN_SAMPLES:
                continue
            cut = int(len(p) * (1 - holdout))
            trial = _isotonic(p[:cut], y[:cut])
            later_p, later_y = p[cut:], y[cut:]
            if brier(np.interp(later_p, *trial), later_y) < brier(later_p, later_y):
                curves[market] = _isotonic(p, y)
        return cls(curves)

    def apply(self, market: str, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if market not in self.curves:
            return np.clip(p, P_MIN, P_MAX)
        x, y = self.curves[market]
        return np.clip(extend(p, np.asarray(x, float), np.asarray(y, float)), P_MIN, P_MAX)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "calibrator", "version": 1, "curves": self.curves}))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        d = json.loads(path.read_text())
        return cls({k: (v[0], v[1]) for k, v in d["curves"].items()})


def extend(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The fitted curve inside [x[0], x[-1]]; outside it, the correction at the edge fades
    linearly to none at 0 and at 1. The curve is fit on test lines near each player's typical
    output (raw probabilities of roughly 0.35-0.65 for points), so plain interpolation pinned
    every line beyond that range at the edge value: "over 10.5 points" for a 25-point scorer,
    truly ~97%, came out 62%, and long-shot alt-line rungs looked like huge edges. Continuous
    at the edges and never reorders probabilities (the slopes stay positive)."""
    out = np.interp(p, x, y)
    lo, hi = float(x[0]), float(x[-1])
    above, below = p > hi, p < lo
    if hi < 1:
        out[above] = p[above] + (y[-1] - hi) * (1 - p[above]) / (1 - hi)
    if lo > 0:
        out[below] = p[below] + (y[0] - lo) * p[below] / lo
    return out


def _isotonic(p: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(p)
    groups = np.array_split(order, min(FIT_BINS, len(p)))
    px = np.array([p[g].mean() for g in groups])
    py = np.array([y[g].mean() for g in groups])
    w = np.array([len(g) for g in groups], dtype=float)
    iso = IsotonicRegression(y_min=P_MIN, y_max=P_MAX, out_of_bounds="clip")
    iso.fit(px, py, sample_weight=w)
    return list(map(float, iso.X_thresholds_)), list(map(float, iso.y_thresholds_))


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[dict]:
    """Calibration table: predicted vs observed rate per probability decile."""
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            out.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(m.sum()),
                    "predicted": round(float(p[m].mean()), 4),
                    "observed": round(float(hit[m].mean()), 4),
                }
            )
    return out


def expected_calibration_error(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    """Count-weighted mean |predicted - observed| over deciles (plan, Stage 9 uses this)."""
    table = reliability(p, hit, bins)
    n = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in table) / n) if n else 0.0


def brier(p: np.ndarray, hit: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(hit, float)) ** 2))


def calibration_failures(
    p: np.ndarray,
    hit: np.ndarray,
    tolerance: float = 0.03,
    z_crit: float = 3.0,
    clusters: np.ndarray | None = None,
) -> list[dict]:
    """Deciles off by more than `tolerance` AND by more than z_crit standard errors.
    z_crit = 3 because ~50 deciles are checked at once (5 markets x 10 bins): at z = 2,
    about one would fail by chance alone even for a perfectly calibrated model.
    With a few hundred results per decile, noise alone is about +/-3 points, so a raw
    tolerance check would fail a perfect model; at full-season sample sizes this is the
    plan's "every decile within 3 points".

    `clusters`: the unit within which errors are correlated. Evaluations pass player ids:
    several lines on one player-game share an outcome, and a player's rate-estimation error
    persists across his games, so 500 rows from 16 players are far fewer than 500 independent
    results (on the synthetic league that turned z = 4.6 into z = 2.0). The standard error is
    clustered (each cluster's summed residual counts once) and never below the
    independent-rows value. Small leagues therefore get a weaker check; real seasons (~450
    players) keep it strong. Without clusters, this is the usual binomial standard error."""
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    bad = []
    for r in reliability(p, hit):
        lo = float(r["bin"].split("-")[0])
        hi = lo + 0.1
        m = (p >= lo) & (p < hi) if hi < 1 - 1e-9 else (p >= lo) & (p <= 1)
        err = abs(r["predicted"] - r["observed"])
        if clusters is None:
            se = np.sqrt(max(r["predicted"] * (1 - r["predicted"]), 1e-4) / r["n"])
        else:
            # Never below the independent-rows error: clustering can only add uncertainty.
            resid = pd.Series(hit[m] - p[m]).groupby(np.asarray(clusters)[m]).sum()
            se = max(
                float(np.sqrt((resid**2).sum())) / r["n"],
                np.sqrt(max(r["predicted"] * (1 - r["predicted"]), 1e-4) / r["n"]),
            )
        if err > tolerance and err / se > z_crit:
            bad.append({**r, "z": round(float(err / se), 2)})
    return bad
