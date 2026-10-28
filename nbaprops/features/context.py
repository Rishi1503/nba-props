"""Role-change context: who is missing from a player's team today, in the pre-game inputs the
model already uses. One definition for the live pick tags, the backtest slices and the
weekly review, so "key teammate out" means the same thing everywhere."""

from __future__ import annotations

import math

KEY_OUT_USAGE = 0.22  # usage share missing: a star, or two significant rotation players


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def usage_share(x) -> float:
    """Usage as a fraction (0.28), whether stored as a fraction or a percentage (28.0)."""
    v = _num(x)
    return v / 100 if v > 1.5 else v


def key_out(starters_out, usage_out) -> bool:
    return _num(starters_out) >= 1 and usage_share(usage_out) >= KEY_OUT_USAGE


def role_context(row) -> dict:
    """Tags for one player-game from its feature row (a mapping or a namedtuple)."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    usage = usage_share(get("absent_usage_sum", 0))
    starters = int(_num(get("absent_starters", 0)))
    return {
        "team_out_min": round(_num(get("absent_min_sum", 0)), 1),
        "team_out_usage": round(usage, 3),
        "starters_out": starters,
        "same_pos_out_min": round(_num(get("same_pos_absent_min", 0)), 1),
        "key_out": key_out(starters, usage),
    }
