"""Tail calibration: are the model's 55-95% probabilities right?

The Stage 5 targets check calibration at test lines near each player's typical output, where
probabilities sit around 30-65%. Staked picks (70%+) and DraftKings' alt-line rungs ("20+
points" at -200) live further out. Here every line of every calibrated grid (1st-99th
percentile half-lines, exactly as live pricing stores them) is scored against the actual
result, on the favored side of each line (over if P(over) >= 50%, else under), in 5-point
bands from 55% to 95%, separately for overs and unders.

A band fails when it is off by more than `tolerance` AND by more than `z_crit` standard
errors, clustered by player (dozens of lines on one player-game share one outcome, and a
player's rate error persists across his games), never below the independent-rows error:
the same bar as the Stage 5 decile check.
"""

import numpy as np
import pandas as pd

EDGES = np.round(np.arange(0.55, 0.951, 0.05), 2)  # 0.55-0.60 ... 0.90-0.95
MIN_BAND_ROWS = 200


def favored_points(grids: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (player-game, market, line) on the favored side, inside the bands:
    market, player_id, side, line, q (the model's probability for that side), hit."""
    cols = ["game_id", "player_id", "market", "y"] + (["game_date"] if "game_date" in rows else [])
    g = grids.merge(
        rows[cols].drop_duplicates(["game_id", "player_id", "market"]),
        on=["game_id", "player_id", "market"],
        how="inner",
    )
    g = g[np.isfinite(g["y"].astype(float))]
    dates = g["game_date"] if "game_date" in g else pd.Series(pd.NaT, index=g.index)
    recs = [
        (m, pid, dt, float(k), float(p), float(y))
        for m, pid, dt, grid, y in zip(
            g["market"], g["player_id"], dates, g["p_over"], g["y"], strict=True
        )
        for k, p in grid.items()
    ]
    d = pd.DataFrame(recs, columns=["market", "player_id", "game_date", "line", "p_over", "y"])
    over = d["p_over"] >= 0.5
    d["side"] = np.where(over, "over", "under")
    d["q"] = np.where(over, d["p_over"], 1 - d["p_over"])
    d["hit"] = np.where(over, d["y"] > d["line"], d["y"] < d["line"]).astype(float)
    keep = (d["q"] >= EDGES[0]) & (d["q"] < EDGES[-1])
    return d.loc[
        keep, ["market", "player_id", "game_date", "side", "line", "q", "hit"]
    ].reset_index(drop=True)


# --- alt-rung correction -----------------------------------------------------------------
# Favored points and PRA overs at 60-85% hit 4-12 points more often than the model said (the
# 2024-25 tail check), while main lines are calibrated. One calibration can't serve both
# (the hybrid tried), so alt rungs get their own correction, used only when pricing them.
RUNG_MARKETS = ("pts", "pra")


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) fit, pool-adjacent-violators."""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w, strict=True)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            wt = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / wt, wt, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


def fit_rung_curve(points: pd.DataFrame, market: str) -> tuple[list, list] | None:
    """Model probability -> observed rate for favored overs of `market`, by 5-point band,
    made monotone and pinned to (0.5, 0.5) and (1, 1); None without enough data."""
    d = points[(points["market"] == market) & (points["side"] == "over")]
    xs, ys, ws = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        b = d[(d["q"] >= lo) & (d["q"] < hi)]
        if len(b) >= MIN_BAND_ROWS:
            xs.append(float(b["q"].mean()))
            ys.append(float(b["hit"].mean()))
            ws.append(float(len(b)))
    if len(xs) < 3:
        return None
    y = np.clip(_pav(np.array(ys), np.array(ws)), 0.5, 0.995)
    x = [0.5, *xs, 1.0]
    y = [0.5, *np.maximum.accumulate(y).tolist(), 1.0]
    return [round(v, 5) for v in x], [round(v, 5) for v in y]


def apply_rung_curve(curve, p):
    """The correction for favored overs (p >= 0.5); probabilities below 0.5 are unchanged."""
    p = np.asarray(p, dtype=float)
    x, y = curve
    return np.where(p >= 0.5, np.interp(p, x, y), p)


def rung_check(points: pd.DataFrame) -> dict:
    """Fit on the first half of the season, check on the second half (never seen by the
    fit), and refit on everything for production. Passes when every corrected band of the
    second half passes the tail test."""
    out = {"markets": {}, "passes": False, "curves": {}}
    d = points[points["side"] == "over"].dropna(subset=["game_date"])
    if d.empty:
        return out
    split = pd.to_datetime(d["game_date"]).median()
    first = d[pd.to_datetime(d["game_date"]) < split]
    second = d[pd.to_datetime(d["game_date"]) >= split]
    ok_all = True
    for m in RUNG_MARKETS:
        curve = fit_rung_curve(first, m)
        test = second[second["market"] == m]
        if curve is None or test.empty:
            out["markets"][m] = {"passes": False, "reason": "not enough data"}
            ok_all = False
            continue
        raw = tail_report(test)
        fixed = tail_report(test.assign(q=apply_rung_curve(curve, test["q"])))
        out["markets"][m] = {"passes": bool(fixed["passes"]), "before": raw, "after": fixed}
        ok_all &= bool(fixed["passes"])
        full = fit_rung_curve(d, m)
        if full is not None:
            out["curves"][m] = full
    out["passes"] = ok_all and bool(out["curves"])
    out["split_date"] = str(split.date())
    return out


def render_rung_check(check: dict) -> list[str]:
    lines = [
        "## Alt-rung correction (points and PRA favored overs)",
        "",
        f"Fit on games before {check.get('split_date', '?')}, checked on the games after it "
        f"(never seen by the fit): **{'PASS' if check['passes'] else 'FAIL'}**",
        "",
    ]
    for m, v in check["markets"].items():
        if "after" not in v:
            lines.append(f"- {m}: {v.get('reason', 'not checked')}")
            continue
        n_before = len(v["before"]["failures"])
        n_after = len(v["after"]["failures"])
        lines.append(f"- {m}: failing bands {n_before} before the correction, {n_after} after")
    lines += [
        "",
        "| Market | Corrected band | Lines | Corrected | Observed | z | |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, v in check["markets"].items():
        for b in v.get("after", {}).get("bands", {}).get(f"{m} over", []):
            lines.append(
                f"| {m} | {b['band']} | {b['n']} | {b['predicted']:.3f} | {b['observed']:.3f} "
                f"| {b['z']:+.2f} | {'FAIL' if b['fails'] else ''} |"
            )
    lines.append("")
    return lines


def tail_report(points: pd.DataFrame, tolerance: float = 0.03, z_crit: float = 3.0) -> dict:
    bands: dict[str, list[dict]] = {}
    for (market, side), g in points.groupby(["market", "side"]):
        out = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
            m = (g["q"] >= lo) & (g["q"] < hi)
            n = int(m.sum())
            if n < MIN_BAND_ROWS:
                continue
            pred, obs = float(g.loc[m, "q"].mean()), float(g.loc[m, "hit"].mean())
            resid = (g.loc[m, "hit"] - g.loc[m, "q"]).groupby(g.loc[m, "player_id"]).sum()
            se = max(
                float(np.sqrt((resid**2).sum())) / n,
                float(np.sqrt(max(pred * (1 - pred), 1e-4) / n)),
            )
            z = (obs - pred) / se
            out.append(
                {
                    "band": f"{lo:.2f}-{hi:.2f}",
                    "n": n,
                    "players": int(g.loc[m, "player_id"].nunique()),
                    "predicted": round(pred, 4),
                    "observed": round(obs, 4),
                    "z": round(z, 2),
                    "fails": bool(abs(obs - pred) > tolerance and abs(z) > z_crit),
                }
            )
        bands[f"{market} {side}"] = out
    failures = [{"group": k, **b} for k, bs in bands.items() for b in bs if b["fails"]]
    return {
        "bands": bands,
        "failures": failures,
        "passes": not failures,
        "tolerance": tolerance,
        "z_crit": z_crit,
    }


def render(tails: dict) -> list[str]:
    lines = [
        "",
        "## Tail calibration (55-95%, the favored side of every grid line)",
        "",
        f"**{'PASS' if tails['passes'] else 'FAIL'}**: a band fails if it is off by more "
        f"than {tails['tolerance']:.0%} and by more than {tails['z_crit']:g} "
        "player-clustered standard errors.",
        "",
        "| Market | Side | Band | Lines | Players | Predicted | Observed | z | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group, bands in tails["bands"].items():
        market, side = group.split()
        for b in bands:
            lines.append(
                f"| {market} | {side} | {b['band']} | {b['n']} | {b['players']} | "
                f"{b['predicted']:.3f} | {b['observed']:.3f} | {b['z']:+.2f} | "
                f"{'FAIL' if b['fails'] else ''} |"
            )
    return lines
