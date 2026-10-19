"""The held-out gate every model passes before its trees are used.

Trees must beat their simple anchor on held-out games by at least `min_gain` (relative) AND
by more than `z` standard errors of the paired per-game difference. The second condition
matters: with a few hundred held-out games, a 0.5% gain happens by luck (on pure-noise test
data the trees passed a gain-only gate and then lost about 5% to their anchor out of sample).
With full seasons of games, real effects clear it easily.
"""

import numpy as np

Z_REQUIRED = 2.0


def beats_anchor(
    loss_anchor: np.ndarray, loss_model: np.ndarray, min_gain: float, z: float = Z_REQUIRED
) -> bool:
    """Per-game losses on the same held-out games, anchor vs model."""
    la = np.asarray(loss_anchor, dtype=float)
    lm = np.asarray(loss_model, dtype=float)
    d = la - lm
    n = len(d)
    if n < 2 or la.mean() <= 0:
        return False
    se = d.std(ddof=1) / np.sqrt(n)
    return bool(d.mean() / la.mean() >= min_gain and d.mean() > z * se)


def pinball_per_row(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    d = np.asarray(y, float) - np.asarray(pred, float)
    return np.maximum(q * d, (q - 1) * d)


def poisson_dev_per_row(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))
