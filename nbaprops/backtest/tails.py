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
    g = grids.merge(
        rows[["game_id", "player_id", "market", "y"]],
        on=["game_id", "player_id", "market"],
        how="inner",
    )
    g = g[np.isfinite(g["y"].astype(float))]
    recs = [
        (m, pid, float(k), float(p), float(y))
        for m, pid, grid, y in zip(g["market"], g["player_id"], g["p_over"], g["y"], strict=True)
        for k, p in grid.items()
    ]
    d = pd.DataFrame(recs, columns=["market", "player_id", "line", "p_over", "y"])
    over = d["p_over"] >= 0.5
    d["side"] = np.where(over, "over", "under")
    d["q"] = np.where(over, d["p_over"], 1 - d["p_over"])
    d["hit"] = np.where(over, d["y"] > d["line"], d["y"] < d["line"]).astype(float)
    keep = (d["q"] >= EDGES[0]) & (d["q"] < EDGES[-1])
    return d.loc[keep, ["market", "player_id", "side", "line", "q", "hit"]].reset_index(drop=True)


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
