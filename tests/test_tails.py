"""Tail calibration on a league where the truth is known: outcomes are Poisson, and the grids
carry either the exact Poisson probabilities (calibrated) or the same probabilities stretched
away from 50% (an overconfident model)."""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from nbaprops.backtest import tails


def league(stretch: float = 1.0, players: int = 300, games: int = 25, seed: int = 7):
    rng = np.random.default_rng(seed)
    mus = rng.uniform(2, 12, players)
    grids, rows = [], []
    for pid, mu in enumerate(mus):
        lines = np.arange(0.5, np.ceil(mu * 2.5) + 0.5, 1.0)
        p = poisson.sf(np.floor(lines), mu)  # P(X > line) for half-point lines
        p = np.clip(0.5 + stretch * (p - 0.5), 0.001, 0.999)
        grid = {f"{ln:.1f}": float(v) for ln, v in zip(lines, p, strict=True)}
        for g in range(games):
            gid = f"g{pid}-{g}"
            grids.append({"game_id": gid, "player_id": pid, "market": "reb", "p_over": grid})
            rows.append(
                {"game_id": gid, "player_id": pid, "market": "reb", "y": float(rng.poisson(mu))}
            )
    return pd.DataFrame(grids), pd.DataFrame(rows)


def test_a_calibrated_model_passes_every_band():
    pts = tails.favored_points(*league())
    out = tails.tail_report(pts)
    assert out["passes"], out["failures"]
    high = [b for b in out["bands"]["reb over"] if b["band"] == "0.85-0.90"][0]
    assert abs(high["predicted"] - high["observed"]) < 0.02


def test_an_overconfident_model_fails_in_the_tails():
    out = tails.tail_report(tails.favored_points(*league(stretch=1.35)))
    assert not out["passes"]
    worst = max(out["failures"], key=lambda b: float(b["band"][:4]))
    assert worst["observed"] < worst["predicted"]  # it said 90%, it won less often
    assert float(worst["band"][:4]) >= 0.75


def test_the_favored_side_is_scored():
    grids = pd.DataFrame(
        [{"game_id": "g", "player_id": 1, "market": "pts", "p_over": {"10.5": 0.2, "4.5": 0.8}}]
    )
    rows = pd.DataFrame([{"game_id": "g", "player_id": 1, "market": "pts", "y": 7.0}])
    pts = tails.favored_points(grids, rows).sort_values("line")
    assert list(pts["side"]) == ["over", "under"]  # 4.5: over at 80%; 10.5: under at 80%
    assert list(pts["q"]) == [0.8, 0.8] and list(pts["hit"]) == [1.0, 1.0]
    assert "Tail calibration" in "\n".join(tails.render(tails.tail_report(pts)))


def test_the_calibrator_no_longer_pins_lines_beyond_its_fitted_range():
    """2024-25 tail check: 200,000+ points lines piled up at P = 0.62 (truly ~0.84) because
    interpolation went flat beyond the fitted range. Inside the range nothing changes."""
    from nbaprops.models.calibrate import Calibrator

    x, y = [0.35, 0.5, 0.65], [0.33, 0.5, 0.62]
    cal = Calibrator({"pts": (x, y)})
    inside = np.linspace(0.35, 0.65, 31)
    assert np.allclose(cal.apply("pts", inside), np.interp(inside, x, y))
    far = cal.apply("pts", np.array([0.80, 0.90, 0.97, 0.20, 0.05]))
    assert len(set(np.round(far, 4))) == 5  # no pile-up at one value
    assert far[2] > 0.95 and far[4] < 0.06  # a raw 97% stays ~97%, a raw 5% ~5%
    edges = cal.apply("pts", np.array([0.65 - 1e-9, 0.65 + 1e-9, 0.35 - 1e-9, 0.35 + 1e-9]))
    assert abs(edges[0] - edges[1]) < 1e-6 and abs(edges[2] - edges[3]) < 1e-6  # continuous
    out = cal.apply("pts", np.linspace(0, 1, 2001))
    assert np.all(np.diff(out) >= -1e-12)  # never reorders probabilities


def test_a_narrow_calibrator_no_longer_creates_a_tail_pile_up():
    """The same league, with probabilities passed through a calibrator fit only on 0.35-0.65
    (as the backtest's near-median test lines are): the tails stay calibrated."""
    from nbaprops.models.calibrate import Calibrator

    grids, rows = league()
    cal = Calibrator({"reb": ([0.35, 0.65], [0.35, 0.65])})
    grids["p_over"] = [
        {k: float(cal.apply("reb", np.array([v]))[0]) for k, v in g.items()}
        for g in grids["p_over"]
    ]
    out = tails.tail_report(tails.favored_points(grids, rows))
    assert out["passes"], out["failures"]
