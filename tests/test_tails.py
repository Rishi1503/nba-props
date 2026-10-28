"""Tail calibration on a league where the truth is known: outcomes are Poisson, and the grids
carry either the exact Poisson probabilities (calibrated) or the same probabilities stretched
away from 50% (an overconfident model)."""

import numpy as np
import pandas as pd
import pytest
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


def test_absence_bias_finds_the_players_stepping_up():
    """Players at the missing starter's position beat their projection by 10%; everyone else is
    projected right. The diagnostic must point at exactly that slice."""
    from nbaprops.backtest import metrics

    rng = np.random.default_rng(5)
    n = 6000
    out_min = rng.choice([0.0, 8.0, 34.0], n)
    same = (out_min > 15) & (rng.random(n) < 0.5)
    mean = rng.uniform(8, 25, n)
    y = rng.normal(mean * np.where(same, 1.10, 1.0), 3.0)
    rows = pd.DataFrame(
        {
            "market": "pts",
            "player_id": rng.integers(0, 400, n),
            "mean": mean,
            "y": y,
            "teammates_out_min": out_min,
            "same_pos_out_min": np.where(same, 30.0, 0.0),
        }
    )
    b = {x["bucket"]: x for x in metrics.absence_bias(rows)["pts"]}
    assert abs(b["none"]["bias_pct"]) < 0.02
    assert b["15+ min, same position"]["bias_pct"] > 0.07 and b["15+ min, same position"]["z"] > 4
    assert abs(b["15+ min, other position"]["bias_pct"]) < 0.02


def test_one_definition_of_a_key_teammate_out():
    from nbaprops.features.context import key_out, role_context

    star = {
        "absent_min_sum": 34.2,
        "absent_usage_sum": 0.28,
        "absent_starters": 1,
        "same_pos_absent_min": 34.2,
    }
    assert role_context(star) == {
        "team_out_min": 34.2,
        "team_out_usage": 0.28,
        "starters_out": 1,
        "same_pos_out_min": 34.2,
        "key_out": True,
    }
    assert key_out(1, 28.0)  # stored as a percentage: same answer
    assert not key_out(0, 0.30)  # no starter out: bench pieces, not a role change
    assert not key_out(1, 0.12)  # a low-usage starter
    assert role_context({"absent_usage_sum": float("nan")})["key_out"] is False


def test_absence_bias_has_the_key_teammate_out_slices():
    from nbaprops.backtest import metrics

    rng = np.random.default_rng(9)
    n = 8000
    key = rng.random(n) < 0.3
    same = key & (rng.random(n) < 0.4)
    mean = rng.uniform(8, 25, n)
    rows = pd.DataFrame(
        {
            "market": "pts",
            "player_id": rng.integers(0, 400, n),
            "mean": mean,
            "y": rng.normal(mean * np.where(same, 1.12, 1.0), 3.0),
            "teammates_out_min": np.where(key, 34.0, 0.0),
            "same_pos_out_min": np.where(same, 34.0, 0.0),
            "teammates_out_usage": np.where(key, 0.27, 0.0),
            "starters_out": np.where(key, 1.0, 0.0),
            "started_share": rng.random(n),
        }
    )
    b = {x["bucket"]: x for x in metrics.absence_bias(rows)["pts"]}
    assert b["key teammate out, same position"]["bias_pct"] > 0.09
    assert {"key teammate out, starters", "key teammate out, bench"} <= set(b)


# --- alt-rung correction (points/PRA favored overs) ----------------------------------------


def _underrated_overs(n=60000, gap=0.08, seed=4):
    """Favored overs that hit `gap` more often than the model says, across a season."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.55, 0.95, n)
    truth = np.clip(q + gap * (1 - (q - 0.55) / 0.4), 0, 1)  # the gap closes toward 95%
    dates = pd.to_datetime("2024-10-22") + pd.to_timedelta(rng.integers(0, 170, n), unit="D")
    return pd.DataFrame(
        {
            "market": rng.choice(["pts", "pra"], n),
            "player_id": rng.integers(0, 500, n),
            "game_date": dates,
            "side": "over",
            "line": 10.5,
            "q": q,
            "hit": (rng.random(n) < truth).astype(float),
        }
    )


def test_the_rung_curve_learns_the_underrating_and_passes_out_of_sample():
    from nbaprops.backtest import tails

    pts = _underrated_overs()
    curve = tails.fit_rung_curve(pts, "pts")
    assert tails.apply_rung_curve(curve, [0.60])[0] == pytest.approx(0.67, abs=0.025)
    assert tails.apply_rung_curve(curve, [0.40])[0] == 0.40  # unders' side untouched
    before = tails.tail_report(pts[pts["market"] == "pts"])
    assert not before["passes"]  # the problem as the 2024-25 check found it
    check = tails.rung_check(pts)
    assert check["passes"] and set(check["curves"]) == {"pts", "pra"}
    md = "\n".join(tails.render_rung_check(check))
    assert "PASS" in md and "failing bands" in md


def test_the_rung_check_fails_without_enough_data():
    from nbaprops.backtest import tails

    check = tails.rung_check(_underrated_overs(n=500))
    assert not check["passes"]


def test_rung_curves_load_from_the_file_or_not_at_all(tmp_path):
    import json
    from types import SimpleNamespace

    from nbaprops.pricing import engine

    f = tmp_path / "rungs.json"
    f.write_text(json.dumps({"curves": {"pts": [[0.5, 0.7, 1.0], [0.5, 0.78, 1.0]]}}))
    assert engine.load_rung_curves(f) == (("pts", (0.5, 0.7, 1.0), (0.5, 0.78, 1.0)),)
    assert engine.load_rung_curves(tmp_path / "missing.json") == ()
    assert engine.load_rung_curves(None) == ()
    rules = engine.Rules.from_config(SimpleNamespace(rung_calibration=str(f)))
    assert rules.rung_curves and rules.rung_curves[0][0] == "pts"
