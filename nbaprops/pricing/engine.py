"""Pricing engine core (plan, Stage 7): probabilities in, flagged and sized picks out.

Pure functions over DataFrames, shared by the Stage 6 betting backtest (Layer B) and live
pricing, so backtests and live picks can't drift apart.

For each DraftKings two-way prop the model has a probability for:
  1. consensus = median de-vigged P(over) of FanDuel/Bovada/Unibet at that line (2+ books);
  2. fair = model_weight * model + (1 - model_weight) * consensus; with no consensus, the
     model pulled `model_only_shrink` of the way toward DraftKings' own no-vig probability;
  3. EV against DraftKings' price on each side, counting a push on whole-number lines as a
     returned stake; the better side is the candidate;
  4. flagged when EV >= ev_min (ev_min_model_only without a consensus), the model and the
     consensus both price that side above DraftKings' no-vig probability, and the player's
     minutes projection isn't low-confidence;
  5. high-confidence mode: flagged with the model's (recalibrated) probability >= hc_prob_min.

Sizing: quarter Kelly (push-aware), capped per bet, then per game and per day by scaling.
One pick per player per game: his best-EV flagged prop.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbaprops.pricing.market import CONSENSUS_BOOKS, TARGET_BOOK, consensus, two_way
from nbaprops.pricing.odds_math import (
    ev_with_push,
    kelly_with_push,
    shrink_toward,
    to_decimal,
)

KEY = ["game_id", "player_id", "market", "line"]
# The model is only trusted inside its simulated range: stored grids span the 1st-99th
# percentile, and a line beyond them reads as a certainty the model can't back up.
P_TRUSTED = (0.01, 0.99)


@dataclass(frozen=True)
class Rules:
    model_weight: float = 0.3
    min_consensus_books: int = 2
    ev_min: float = 0.03
    ev_min_model_only: float = 0.05
    model_only_shrink: float = 0.5
    hc_prob_min: float = 0.70
    max_edge: float = 0.15  # EV above this is held back: the model is likely missing news
    milestone_p_min: float = 0.20  # milestone rungs flagged only inside this model-probability
    milestone_p_max: float = 0.92  # band (the far tail priced from the model alone overstated)
    kelly_fraction: float = 0.25
    max_bet_pct: float = 0.02
    max_game_pct: float = 0.04
    max_daily_pct: float = 0.10
    max_parlay_pct: float = 0.01
    max_satellites_per_slate: int = 2
    sgp_ev_haircut: float = 0.03
    stake_standard_flags: bool = True  # config default is False (high-confidence singles)
    questionable_teammates: str = "robust"  # robust | mixture | block (see strategy.yaml)
    alt_combo_markets: tuple = ("reb", "fg3m", "ast")
    alt_combo_shade: tuple = ()  # (market, probability shaved off) pairs, alt-parlay legs only
    alt_combo_p_min: float = 0.65
    alt_combo_decimal_min: float = 1.8
    alt_combo_decimal_max: float = 2.6
    alt_combo_max_legs: int = 3
    max_alt_combos_per_slate: int = 2
    # (market, xs, ys) curves correcting favored-over probabilities of alt rungs only
    rung_curves: tuple = ()
    # (market, shift) added to the model's chance of the over (the under moves the other way):
    # the 2024-25 tail check had assist overs about 1.1 points high on both sides of every line
    over_shift: tuple = (("ast", -0.012),)

    @classmethod
    def from_config(cls, cfg) -> "Rules":
        vals = {k: getattr(cfg, k) for k in cls.__dataclass_fields__ if hasattr(cfg, k)}
        vals["rung_curves"] = load_rung_curves(getattr(cfg, "rung_calibration", None))
        if "alt_combo_markets" in vals:
            vals["alt_combo_markets"] = tuple(vals["alt_combo_markets"])
        if "alt_combo_shade" in vals:
            vals["alt_combo_shade"] = tuple(sorted(dict(vals["alt_combo_shade"]).items()))
        if "over_shift" in vals:
            vals["over_shift"] = tuple(sorted(dict(vals["over_shift"]).items()))
        return cls(**vals)


def candidates(
    model: pd.DataFrame, board: pd.DataFrame, rules: Rules, grids: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`model`: game_id, player_id, market, line, p_over, p_under, p_push (whole lines),
    low_confidence (+ any passthrough columns such as game_date). `board`: live quotes from
    pricing.market.board. One row per DraftKings prop the model covers, best side chosen."""
    dk = two_way(board[board["book"] == TARGET_BOOK]).drop_duplicates(KEY, keep="last")
    m = dk.merge(model, on=KEY, how="inner", suffixes=("", "_model"))
    lo, hi = P_TRUSTED
    m = m[m["p_over"].between(lo, hi) & m["p_under"].between(lo, hi)]
    if m.empty:
        return m
    cons = consensus(board, rules.min_consensus_books)[[*KEY, "p_over_cons", "n_books"]]
    m = m.merge(cons, on=KEY, how="left")
    m["cons_source"] = np.where(m["p_over_cons"].notna(), "exact", "")
    if grids is not None and len(grids):
        # No two books at DraftKings' exact line: read the books' nearby consensus line across
        # with the model's shape (anchored as for milestones). Half-point lines only.
        anchors = consensus_anchors(board, rules)
        gl = grids.drop_duplicates(["game_id", "player_id", "market"], keep="last").set_index(
            ["game_id", "player_id", "market"]
        )["p_over"]
        for i in np.flatnonzero(m["p_over_cons"].isna().to_numpy()):
            r = m.iloc[i]
            k = (r["game_id"], int(r["player_id"]), r["market"])
            line = float(r["line"])
            if k not in anchors or k not in gl.index or line % 1 == 0:
                continue
            a_line, a_p = anchors[k]
            if abs(a_line - line) > SHIFT_MAX:
                continue
            p = shifted_tail(gl.loc[k], line, a_line, a_p)
            if np.isfinite(p):
                m.iloc[i, m.columns.get_loc("p_over_cons")] = p
                m.iloc[i, m.columns.get_loc("cons_source")] = "shifted"
    return decide(m, rules)


# The columns `decide` needs: one row per DraftKings two-way prop. The candidate log stores
# exactly these, so paper-data tuning replays decisions with the engine's own rules.
DECISION_COLUMNS = [
    "p_over",
    "p_under",
    "p_push",
    "p_over_cons",
    "p_over_novig",
    "p_under_novig",
    "over_price",
    "under_price",
    "low_confidence",
]
SCENARIO_COLUMNS = [
    "p_over_in",
    "p_under_in",
    "p_push_in",
    "p_over_out",
    "p_under_out",
    "p_push_out",
]


def shift_overs(m: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Rules.over_shift: move the model's chance of the over by a measured amount per market
    (and the under the other way, so over + under + push still sums to one)."""
    if not rules.over_shift or "market" not in m:
        return m
    m = m.copy()
    for market, d in rules.over_shift:
        mask = (m["market"] == market).to_numpy()
        if not mask.any():
            continue
        for tag in ("", "_in", "_out"):
            o, u = f"p_over{tag}", f"p_under{tag}"
            if o in m and u in m:
                po = m.loc[mask, o].astype(float)
                pu = m.loc[mask, u].astype(float)
                moved = np.clip(po + d, 0.0, (po + pu).to_numpy()) - po  # stays within the pair
                m.loc[mask, o] = po + moved
                m.loc[mask, u] = pu - moved
    return m


def _over_shift(rules: Rules, market: str) -> float:
    return dict(rules.over_shift).get(market, 0.0)


def decide(m: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """The flag decision for two-way props, from DECISION_COLUMNS (+ SCENARIO_COLUMNS for
    teammates of a key Questionable player). Shared by live pricing and paper-data tuning."""
    m = shift_overs(m.reset_index(drop=True), rules)
    if "p_push" not in m:
        m = m.assign(p_push=0.0)
    has_cons = m["p_over_cons"].notna().to_numpy()
    push = m["p_push"].fillna(0.0).to_numpy(float) if "p_push" in m else np.zeros(len(m))
    out = {}
    for side in ("over", "under"):
        p_model = m[f"p_{side}"].to_numpy(float)
        # Market no-vig probabilities ignore pushes; scale them to "win" probabilities.
        dk_side = m[f"p_{side}_novig"].to_numpy(float) * (1 - push)
        cons_raw = m["p_over_cons"].to_numpy(float)
        cons_side = (cons_raw if side == "over" else 1 - cons_raw) * (1 - push)
        fair = np.where(
            has_cons,
            rules.model_weight * p_model + (1 - rules.model_weight) * np.nan_to_num(cons_side),
            [
                shrink_toward(a, b, rules.model_only_shrink)
                for a, b in zip(p_model, dk_side, strict=True)
            ],
        )
        lose = np.clip(1 - push - fair, 0, 1)
        dec = np.array([to_decimal(int(x)) for x in m[f"{side}_price"]])
        ev = np.array([ev_with_push(w, ll, d) for w, ll, d in zip(fair, lose, dec, strict=True)])
        agree = (p_model > dk_side) & np.where(has_cons, np.nan_to_num(cons_side) > dk_side, True)
        out[side] = {
            "p_model": p_model,
            "p_dk": dk_side,
            "p_cons": np.where(has_cons, cons_side, np.nan),
            "p_fair": fair,
            "p_lose": lose,
            "decimal": dec,
            "ev": ev,
            "agree": agree,
            "price": m[f"{side}_price"].to_numpy(),
        }
    take_over = out["over"]["ev"] >= out["under"]["ev"]
    # Teammates of a key Questionable player: the same side's fair EV if he plays / sits.
    # The consensus is one market price either way; only the model probability changes.
    scen = {}
    if "p_over_in" in m:
        cons_over = m["p_over_cons"].to_numpy(float)
        for tag in ("in", "out"):
            ev_s = np.full(len(m), np.nan)
            push_s = m[f"p_push_{tag}"].fillna(0.0).to_numpy(float)
            for side, mask in (("over", take_over), ("under", ~take_over)):
                p_model_s = m[f"p_{side}_{tag}"].to_numpy(float)
                # market no-vig probabilities ignore pushes: scale by this scenario's push
                cons_s = (cons_over if side == "over" else 1 - cons_over) * (1 - push_s)
                dk_s = m[f"p_{side}_novig"].to_numpy(float) * (1 - push_s)
                fair_s = np.where(
                    has_cons,
                    rules.model_weight * p_model_s
                    + (1 - rules.model_weight) * np.nan_to_num(cons_s),
                    p_model_s + rules.model_only_shrink * (dk_s - p_model_s),
                )
                lose_s = np.clip(1 - push_s - fair_s, 0, 1)
                ev_s = np.where(mask, fair_s * (out[side]["decimal"] - 1) - lose_s, ev_s)
            scen[tag] = ev_s
    res = m[[c for c in m.columns if c not in ("p_over", "p_under")]].copy()
    res["side"] = np.where(take_over, "over", "under")
    for k in ("p_model", "p_dk", "p_cons", "p_fair", "p_lose", "decimal", "ev", "agree", "price"):
        res[k] = np.where(take_over, out["over"][k], out["under"][k])
    res["p_push"] = push
    res["has_consensus"] = has_cons
    threshold = np.where(has_cons, rules.ev_min, rules.ev_min_model_only)
    low = res["low_confidence"].fillna(False).astype(bool).to_numpy()
    res["flag"] = (res["ev"].to_numpy() >= threshold) & res["agree"].to_numpy() & ~low
    res["held_back"] = res["flag"] & (res["ev"].to_numpy() > rules.max_edge)
    res["flag"] &= ~res["held_back"]
    res["ev_if_plays"] = scen.get("in", np.full(len(res), np.nan))
    res["ev_if_sits"] = scen.get("out", np.full(len(res), np.nan))
    res["flag"] &= questionable_ok(res, rules)
    res["high_confidence"] = res["flag"] & (res["p_model"] >= rules.hc_prob_min)
    res["kelly"] = np.minimum(
        [
            kelly_with_push(w, ll, d, rules.kelly_fraction)
            for w, ll, d in zip(res["p_fair"], res["p_lose"], res["decimal"], strict=True)
        ],
        rules.max_bet_pct,
    )
    return res


SHIFT_MAX = 1.5  # read a nearby consensus line across at most this many points


def consensus_anchors(board: pd.DataFrame, rules: Rules) -> dict:
    """{(game_id, player_id, market): (line, p_over)}: the consensus books' most-quoted main
    line (2+ books, both sides) and its median no-vig probability."""
    main = two_way(board[board["book"].isin(CONSENSUS_BOOKS) & (board["line_type"] == "main")])
    anchors = {}
    if main.empty:
        return anchors
    g = main.groupby(KEY).agg(p=("p_over_novig", "median"), n=("book", "nunique")).reset_index()
    g = g[g["n"] >= rules.min_consensus_books]
    g = g.assign(dist=(g["p"] - 0.5).abs()).sort_values(["n", "dist"], ascending=[False, True])
    for r in g.drop_duplicates(["game_id", "player_id", "market"]).itertuples(index=False):
        anchors[(r.game_id, int(r.player_id), r.market)] = (float(r.line), float(r.p))
    return anchors


def _grid_fn(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(sorted(float(k) for k in grid))
    ps = np.array([grid[k] for k in sorted(grid, key=float)], dtype=float)
    return xs, ps


def anchor_shift(grid: dict, anchor_line: float, anchor_p: float) -> float:
    """How far to move the distribution so that P(over anchor_line) = anchor_p."""
    xs, ps = _grid_fn(grid)
    rp, rx = ps[::-1], xs[::-1]  # increasing in p; drop ties for np.interp
    keep = np.concatenate([[True], np.diff(rp) > 0])
    return anchor_line - float(np.interp(anchor_p, rp[keep], rx[keep]))


def shifted_tail(
    grid: dict,
    line: float,
    anchor_line: float | None = None,
    anchor_p: float | None = None,
    shift: float | None = None,
) -> float:
    """P(over `line`) from a stored half-point grid {"24.5": p, ...}, after shifting the
    distribution so that P(over anchor_line) = anchor_p (the consensus main line's no-vig
    probability): the market sets the location, the model the tail shape. At anchor_p = 0.5
    this is the plan's "median matches the consensus main line". `shift` reuses a shift
    computed on another grid (scenario grids move with their blend). NaN beyond the
    simulated range."""
    xs, ps = _grid_fn(grid)
    if shift is None:
        shift = anchor_shift(grid, anchor_line, anchor_p) if anchor_line is not None else 0.0
    x = line - shift
    if x < xs[0] or x > xs[-1]:
        return float("nan")  # beyond the simulated range: not priceable
    return float(np.interp(x, xs, ps))


def load_rung_curves(path) -> tuple:
    """The alt-rung correction written by `make backtest-rungs`, or () without one."""
    import json
    from pathlib import Path

    if not path:
        return ()
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    if not p.exists():
        return ()
    curves = json.loads(p.read_text()).get("curves", {})
    return tuple(
        (m, tuple(float(v) for v in xy[0]), tuple(float(v) for v in xy[1]))
        for m, xy in sorted(curves.items())
    )


def _rung_fix(curves: dict, market: str, p: float) -> float:
    """Favored-over probability of an alt rung, corrected for `market` when a curve exists."""
    if market not in curves or not np.isfinite(p) or p < 0.5:
        return p
    xs, ys = curves[market]
    return float(np.interp(p, xs, ys))


def milestone_candidates(grids: pd.DataFrame, board: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """One-sided DraftKings milestones ("20+" stored as 19.5). `grids`: game_id, player_id,
    market, p_over (the stored half-point grid), low_confidence (+ passthrough columns).

    With a consensus main line (the consensus books' most-quoted main line, 2+ books), the
    model distribution is shifted to match it and read at the milestone; EV floor ev_min.
    Without one, the model alone, shrunk toward DraftKings' implied probability, at
    ev_min_model_only. Milestones are half-point lines: no pushes."""
    ms = board[
        (board["book"] == TARGET_BOOK)
        & (board["line_type"] == "milestone")
        & board["over_price"].notna()
    ].drop_duplicates(KEY, keep="last")
    if ms.empty:
        return pd.DataFrame()
    anchors = consensus_anchors(board, rules)
    # one grid per player-market: the latest projection (callers pass them in time order)
    gl = grids.drop_duplicates(["game_id", "player_id", "market"], keep="last").set_index(
        ["game_id", "player_id", "market"]
    )
    curves = {m: (xs, ys) for m, xs, ys in rules.rung_curves}
    rows = []
    for q in ms.itertuples(index=False):
        k = (q.game_id, q.player_id, q.market)
        if k not in gl.index:
            continue
        info = gl.loc[k]
        grid = info["p_over"]
        d = to_decimal(int(q.over_price))
        p_model = shifted_tail(grid, q.line)
        anchor = anchors.get(k)
        shift = anchor_shift(grid, *anchor) if anchor else 0.0
        if np.isnan(p_model) or np.isnan(shifted_tail(grid, q.line, shift=shift)):
            continue

        sh = _over_shift(rules, q.market)

        def fair_from(
            g: dict, anchor=anchor, shift=shift, d=d, line=q.line, market=q.market, sh=sh
        ) -> float:
            if anchor:
                p = _rung_fix(curves, market, shifted_tail(g, line, shift=shift))
                return float(np.clip(p + sh, 0.0, 1.0))
            fixed = float(np.clip(_rung_fix(curves, market, shifted_tail(g, line)) + sh, 0, 1))
            return shrink_toward(fixed, 1 / d, rules.model_only_shrink)

        p_model = float(np.clip(_rung_fix(curves, q.market, p_model) + sh, 0.0, 1.0))

        p_fair = fair_from(grid)
        thr = rules.ev_min if anchor else rules.ev_min_model_only
        ev = p_fair * (d - 1) - (1 - p_fair)
        sc = info["scenarios"] if "scenarios" in info.index else None
        ev_in = ev_out = np.nan
        if isinstance(sc, dict):
            f_in, f_out = fair_from(sc["in"]), fair_from(sc["out"])
            ev_in, ev_out = f_in * (d - 1) - (1 - f_in), f_out * (d - 1) - (1 - f_out)
        low = bool(info.get("low_confidence", False))
        # inside the validated band: the model's probability and the one the bet is priced at
        in_band = (
            rules.milestone_p_min <= p_model <= rules.milestone_p_max
            and rules.milestone_p_min <= p_fair <= rules.milestone_p_max
        )
        held = ev >= thr and not low and in_band and ev > rules.max_edge
        flag = ev >= thr and not low and in_band and not held
        rows.append(
            {
                **{
                    c: getattr(q, c)
                    for c in ("game_id", "player_id", "market", "line", "book", "line_type")
                },
                **{
                    c: info[c]
                    for c in grids.columns
                    if c not in ("game_id", "player_id", "market", "p_over")
                },
                "side": "over",
                "price": int(q.over_price),
                "decimal": d,
                "p_model": p_model,
                "p_cons": anchor[1] if anchor else np.nan,
                "p_dk": 1 / d,
                "p_fair": p_fair,
                "p_push": 0.0,
                "p_lose": 1 - p_fair,
                "ev": ev,
                "has_consensus": bool(anchor),
                "agree": True,
                "flag": flag,
                "held_back": held,
                "high_confidence": flag and p_model >= rules.hc_prob_min,
                "kelly": min(
                    kelly_with_push(p_fair, 1 - p_fair, d, rules.kelly_fraction), rules.max_bet_pct
                ),
                "anchor_line": anchor[0] if anchor else np.nan,
                "ev_if_plays": ev_in,
                "ev_if_sits": ev_out,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["flag"] = out["flag"] & questionable_ok(out, rules)
        out["high_confidence"] = out["high_confidence"] & out["flag"]
    return out


def questionable_ok(res: pd.DataFrame, rules: Rules) -> np.ndarray:
    """Rows without a key Questionable teammate pass. With one: `robust` needs the bet +EV
    whether he plays or sits, `mixture` accepts the blend alone, `block` rejects."""
    has = np.isfinite(res["ev_if_plays"].to_numpy(float)) & np.isfinite(
        res["ev_if_sits"].to_numpy(float)
    )
    if rules.questionable_teammates == "mixture":
        return np.ones(len(res), dtype=bool)
    if rules.questionable_teammates == "block":
        return ~has
    robust = (res["ev_if_plays"].fillna(0).to_numpy() >= 0) & (
        res["ev_if_sits"].fillna(0).to_numpy() >= 0
    )
    return ~has | robust


def expected_growth(c: pd.DataFrame) -> np.ndarray:
    """p_win log(1 + b f) + p_lose log(1 - f) at each candidate's capped Kelly fraction
    (pushes contribute 0). Frames without prices (hand-built) fall back to EV."""
    if not {"kelly", "decimal", "p_fair"} <= set(c.columns):
        return c["ev"].to_numpy(float)
    f = c["kelly"].to_numpy(float)
    b = c["decimal"].to_numpy(float) - 1
    lose = c["p_lose"].to_numpy(float) if "p_lose" in c else 1 - c["p_fair"].to_numpy(float)
    return c["p_fair"].to_numpy(float) * np.log1p(b * f) + lose * np.log1p(-np.minimum(f, 0.99))


def select_and_size(
    cands: pd.DataFrame,
    bankroll: float,
    rules: Rules,
    day_col: str = "game_date",
    committed: pd.DataFrame | None = None,
    daily: bool = True,
) -> pd.DataFrame:
    """Flagged candidates -> picks: best EV per player per game, quarter-Kelly stakes capped
    per bet (in `candidates`), then scaled to the per-game and (if `daily`) per-day caps.

    `committed`: picks already made today (game_id, player_id, stake, and `day_col`) from an
    earlier run. Those players get no new pick, and the caps count their stakes, so three
    runs a day can't triple the exposure. With daily=False the caller applies the daily cap
    across singles and parlays together."""
    # One pick per player per game: the prop with the highest expected log-growth at its
    # Kelly stake. Choosing by EV alone favors long odds across a player's many alternate
    # lines, where the model's tails are least reliable (a winner's curse).
    f = cands[cands["flag"]].copy()
    f["growth"] = expected_growth(f)
    # A player's one slot goes to a pick we would actually stake: when standard flags aren't
    # staked, a high-confidence pick must not lose the slot to a zero-stake standard flag
    # with more growth on paper (we'd end up betting nothing on that player).
    f["_stakeable"] = (
        True
        if rules.stake_standard_flags
        else f.get("high_confidence", pd.Series(True, index=f.index)).astype(bool)
    )
    f = f.sort_values(["_stakeable", "growth", "ev"], ascending=False, kind="stable")
    f = f.drop_duplicates(["game_id", "player_id"], keep="first").drop(columns="_stakeable").copy()
    if committed is not None and len(committed):
        taken = set(zip(committed["game_id"], committed["player_id"].astype(int), strict=True))
        f = f[[(g, int(p)) not in taken for g, p in zip(f["game_id"], f["player_id"], strict=True)]]
    if f.empty:
        return f.assign(stake=pd.Series(dtype=float))
    f["stake"] = f["kelly"] * bankroll
    # Standard flags are logged but not staked unless the strategy says so.
    f["tracked_only"] = False if rules.stake_standard_flags else ~f["high_confidence"].astype(bool)
    f.loc[f["tracked_only"], "stake"] = 0.0
    caps = [("game_id", rules.max_game_pct)] + ([(day_col, rules.max_daily_pct)] if daily else [])
    for col, cap in caps:
        used = (
            committed.groupby(col)["stake"].sum()
            if committed is not None and len(committed)
            else pd.Series(dtype=float)
        )
        left = (cap * bankroll - f[col].map(used).fillna(0.0)).clip(lower=0.0)
        total = f.groupby(col)["stake"].transform("sum")
        f["stake"] *= np.minimum(1.0, left / total.where(total > 0, 1.0))
    f = f[(f["stake"] > 0) | f["tracked_only"]]  # drop picks squeezed to 0 by the caps
    return f.reset_index(drop=True)


def build_candidates(
    model_lines: pd.DataFrame, grids: pd.DataFrame | None, board: pd.DataFrame, rules: Rules
) -> pd.DataFrame:
    """Every priceable DraftKings prop: two-way lines (model probabilities at those lines)
    plus milestones (from the stored grids). Live pricing and the Layer B backtest both call
    this, so they price the same props the same way."""
    parts = []
    if model_lines is not None and len(model_lines):
        parts.append(candidates(model_lines, board, rules, grids))
    if grids is not None and len(grids):
        parts.append(milestone_candidates(grids, board, rules))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
