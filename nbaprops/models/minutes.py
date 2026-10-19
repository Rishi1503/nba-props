"""Stage 4: minutes as a distribution, for players expected to play.

Whether a player plays at all comes from injury status (Stage 3 slates exclude Out/Doubtful).
Given that he plays, minutes are a two-part mixture:

  normal game  two half-normals joined at the median q50, half the mass on each side, with the
               left spread set by q10 and the right by q90, so all three quantiles match the
               LightGBM quantile models exactly (trained on games without an early exit)
  early exit   uniform between 0 and an exit cap (injury, fouls, ejection), with a
               per-player probability shrunk toward the league rate, from earlier games only

`MinutesDist` is the object the Stage 5 simulation samples from. Rule-based adjustments
(plan, Stage 4) widen the spread and set `low_confidence` for players back from a multi-game
absence, on a new team, rookies early in the season, and starters on clinched teams late in
the season.

Each quantile model boosts from an anchor (recency-weighted minutes plus that quantile's
typical offset), so trees only learn adjustments, and it must beat that anchor on held-out
games or it falls back to the anchor alone. With little signal it cannot do worse than a
recency-weighted average; with real signal (absences, blowouts, rest) the trees take over.

Evaluation follows the plan: MAE against the 10-game-average baseline (target: at least 8%
better), coverage of the 80% interval (target: about 80%), and the same by bucket. MAE is
scored on the distribution's median, the point prediction that minimizes absolute error.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from nbaprops.models.gate import beats_anchor, pinball_per_row

MAX_MINUTES = 53.0  # regulation plus one overtime; longer games are rare enough to clip
Z90 = 1.2815515655446004  # standard normal 90th percentile
QUANTILES = (0.1, 0.5, 0.9)

# Early exit: a rotation player (10-game average >= 20) who plays under 35% of that average
# and under 15 minutes.
EXIT_MIN_AVG = 20.0
EXIT_FRAC = 0.35
EXIT_CAP = 15.0
EXIT_PRIOR_GAMES = 40.0  # weight of the league rate when shrinking a player's exit rate

# Rule-based widening for low-confidence situations
RETURN_GAMES_MISSED = 3
ROOKIE_EARLY_GAMES = 5
WIDEN = 1.25

FEATURES = [
    "min_avg_l3",
    "min_avg_l5",
    "min_avg_l10",
    "min_avg_l20",
    "min_ewm",
    "std_min_avg",
    "prev_season_min_avg",
    "preseason_min_avg",
    "minutes_share_l10",
    "started_share_l10",
    "usage_l10",
    "absent_n",
    "absent_min_sum",
    "absent_usage_sum",
    "absent_starters",
    "same_pos_absent_min",
    "rating_team_margin",
    "rating_total",
    "b2b",
    "rest_days",
    "player_rest_days",
    "games_last_7",
    "games_missed",
    "new_team",
    "games_with_team",
    "rookie",
    "std_games",
    "career_games",
    "late_season",
    "clinched_top6",
    "eliminated",
    "is_home",
    "games_remaining",
    "is_postseason",
    "stint_games",
    "stint_min_avg_l10",
    "departed_min_sum",
    "departed_usage_sum",
    "arrived_min_sum",
    "arrived_usage_sum",
    "returning_min_sum",
    "returning_usage_sum",
    "wo_games",
    "wo_min_delta",
    "opp_absent_n",
    "opp_absent_min_sum",
    "pts_pm_l10",
    "opp_pace_l15",
]

DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "num_threads": 1,
}
MAX_ROUNDS = 2000
EARLY_STOP = 100
# Each quantile model must beat its own anchor on held-out data by this much (pinball loss),
# or it falls back to the anchor alone: the trees have to earn their complexity.
MIN_GAIN_OVER_ANCHOR = 0.005


# --- the distribution --------------------------------------------------------


@dataclass
class MinutesDist:
    """Vectorized minutes mixture, one entry per player-game."""

    q50: np.ndarray
    sd_left: np.ndarray
    sd_right: np.ndarray
    p_exit: np.ndarray
    exit_cap: np.ndarray

    def __len__(self) -> int:
        return len(self.q50)

    def _normal_mean(self) -> np.ndarray:
        return self.q50 + math.sqrt(2 / math.pi) * (self.sd_right - self.sd_left) / 2

    def mean(self) -> np.ndarray:
        return (1 - self.p_exit) * self._normal_mean() + self.p_exit * self.exit_cap / 2

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Half the mass below the median (spread sd_left), half above (sd_right)."""
        return np.where(
            x <= self.q50, ndtr((x - self.q50) / self.sd_left), ndtr((x - self.q50) / self.sd_right)
        )

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """P(minutes <= x). Draws are clipped to [0, MAX_MINUTES], so the normal part's mass
        below 0 sits at 0 and its mass above the cap sits at the cap."""
        x = np.broadcast_to(np.asarray(x, dtype=float), self.q50.shape)
        normal = np.where(x < 0, 0.0, np.where(x >= MAX_MINUTES, 1.0, self._normal_cdf(x)))
        exit_ = np.clip(x / self.exit_cap, 0.0, 1.0)
        return (1 - self.p_exit) * normal + self.p_exit * exit_

    def quantile(self, p: float, iters: int = 60) -> np.ndarray:
        lo = np.zeros(len(self))
        hi = np.full(len(self), MAX_MINUTES)
        for _ in range(iters):
            mid = (lo + hi) / 2
            below = self.cdf(mid) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return (lo + hi) / 2

    def normal_ppf(self, u: np.ndarray) -> np.ndarray:
        """Exact inverse of the normal-game part for u of shape (players, draws)."""
        z = ndtri(np.clip(u, 1e-9, 1 - 1e-9))
        q50, sl, sr = self.q50[:, None], self.sd_left[:, None], self.sd_right[:, None]
        return np.where(z < 0, q50 + z * sl, q50 + z * sr)

    def sample_correlated(self, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draws whose normal-game part follows the uniforms `u` (players, draws), so the
        simulation can tie minutes to the game scenario; early exits are drawn independently.
        Each player's minutes distribution is exactly this mixture either way."""
        normal = self.normal_ppf(u)
        exits = rng.random(u.shape) < self.p_exit[:, None]
        exit_draw = rng.random(u.shape) * self.exit_cap[:, None]
        return np.clip(np.where(exits, exit_draw, normal), 0.0, MAX_MINUTES)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n draws per player -> array (players, n)."""
        k = len(self)
        side = rng.random((k, n)) < 0.5  # below or above the median, equal mass
        z = np.abs(rng.standard_normal((k, n)))
        normal = np.where(
            side,
            self.q50[:, None] - z * self.sd_left[:, None],
            self.q50[:, None] + z * self.sd_right[:, None],
        )
        exits = rng.random((k, n)) < self.p_exit[:, None]
        exit_draw = rng.random((k, n)) * self.exit_cap[:, None]
        return np.clip(np.where(exits, exit_draw, normal), 0.0, MAX_MINUTES)


# --- early exits -------------------------------------------------------------


def is_early_exit(minutes: pd.Series, min_avg_l10: pd.Series) -> pd.Series:
    return (
        (min_avg_l10 >= EXIT_MIN_AVG) & (minutes < EXIT_FRAC * min_avg_l10) & (minutes < EXIT_CAP)
    ).fillna(False)


def exit_rates(rows: pd.DataFrame, history: pd.DataFrame, league_rate: float) -> np.ndarray:
    """Per-row early-exit probability from each player's exits in games before the row's date,
    shrunk toward the league rate. `history` needs player_id, game_date, y_minutes,
    min_avg_l10 (any finished games, including the rows themselves: only earlier dates count).
    Applies to rotation-level players; others get the league rate scaled by their minutes."""
    h = history.dropna(subset=["y_minutes"]).copy()
    h["eligible"] = h["min_avg_l10"] >= EXIT_MIN_AVG
    h["exit"] = is_early_exit(h["y_minutes"], h["min_avg_l10"]) & h["eligible"]
    h = h.sort_values(["player_id", "game_date"])
    g = h.groupby("player_id")
    h["n_elig"] = g["eligible"].cumsum()
    h["n_exit"] = g["exit"].cumsum()
    state = h[["player_id", "game_date", "n_elig", "n_exit"]]
    left = rows[["player_id", "game_date"]].reset_index().sort_values("game_date")
    state = state.sort_values("game_date")
    merged = (
        pd.merge_asof(left, state, on="game_date", by="player_id", allow_exact_matches=False)
        .set_index("index")
        .loc[rows.index]
    )
    n_elig = merged["n_elig"].fillna(0).to_numpy(float)
    n_exit = merged["n_exit"].fillna(0).to_numpy(float)
    return (n_exit + EXIT_PRIOR_GAMES * league_rate) / (n_elig + EXIT_PRIOR_GAMES)


# --- anchor: the starting point every quantile model boosts from -------------


STINT_ANCHOR_GAMES = 3.0  # after a trade, half the anchor weight on new-team minutes by game 3


def anchor(rows: pd.DataFrame, use_stint: bool = True) -> np.ndarray:
    """Best simple guess: recency-weighted minutes, then season, last season, preseason.
    After a team change, blend toward minutes with the new team as games with it accumulate
    (weight n / (n + 3)), since recent form still mixes in the old team's role."""
    a = rows["min_ewm"].astype(float)
    for col in ("std_min_avg", "prev_season_min_avg", "preseason_min_avg"):
        a = a.fillna(rows[col].astype(float))
    a = a.fillna(15.0).to_numpy()
    # use_stint=False for models trained without the roster inputs, so a with/without
    # comparison doesn't leak trade information in through the anchor.
    if use_stint and "stint_change" in rows and "stint_min_avg_l10" in rows:
        n = rows["stint_games"].fillna(0).to_numpy(float)
        stint = rows["stint_min_avg_l10"].astype(float).to_numpy()
        moved = (rows["stint_change"].fillna(0).to_numpy() == 1) & (n > 0) & ~np.isnan(stint)
        w = n / (n + STINT_ANCHOR_GAMES)
        a = np.where(moved, w * np.nan_to_num(stint) + (1 - w) * a, a)
    return a


# --- the model ---------------------------------------------------------------


@dataclass
class MinutesModel:
    boosters: dict[float, lgb.Booster]
    league_exit_rate: float
    features: list[str]
    trained_through: str
    params: dict = field(default_factory=dict)
    best_rounds: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=dict)  # quantile of (minutes - anchor) in training

    def raw_quantiles(self, rows: pd.DataFrame) -> dict[float, np.ndarray]:
        x = _matrix(rows, self.features)
        base = anchor(rows, "stint_min_avg_l10" in self.features)
        out = {}
        for q, b in self.boosters.items():
            rounds = self.best_rounds.get(q)
            trees = b.predict(x, num_iteration=rounds) if rounds else 0.0  # 0 = anchor only
            out[q] = base + self.offsets.get(q, 0.0) + trees
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "minutes",
            "version": 1,
            "league_exit_rate": self.league_exit_rate,
            "features": self.features,
            "trained_through": self.trained_through,
            "params": self.params,
            "best_rounds": {str(k): v for k, v in self.best_rounds.items()},
            "offsets": {str(k): v for k, v in self.offsets.items()},
            "boosters": {str(q): b.model_to_string() for q, b in self.boosters.items()},
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "MinutesModel":
        d = json.loads(path.read_text())
        return cls(
            boosters={float(q): lgb.Booster(model_str=s) for q, s in d["boosters"].items()},
            league_exit_rate=d["league_exit_rate"],
            features=d["features"],
            trained_through=d["trained_through"],
            params=d["params"],
            best_rounds={float(k): v for k, v in d["best_rounds"].items()},
            offsets={float(k): v for k, v in d["offsets"].items()},
        )


def _matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].astype(float).to_numpy()


def fit(
    train: pd.DataFrame,
    params: dict | None = None,
    valid_frac: float = 0.15,
    features: list[str] | None = None,
) -> MinutesModel:
    """Train the quantile models on normal games (early exits held out), with early stopping
    on the latest `valid_frac` of dates so the validation set is later than the training set."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    df = train.dropna(subset=["y_minutes"]).sort_values("game_date")
    exits = is_early_exit(df["y_minutes"], df["min_avg_l10"])
    feats = list(features or FEATURES)  # a subset, for feature-group comparisons
    use_stint = "stint_min_avg_l10" in feats
    eligible = df["min_avg_l10"] >= EXIT_MIN_AVG
    league_rate = float(exits[eligible].mean()) if eligible.any() else 0.02
    normal = df[~exits]
    cut = normal["game_date"].quantile(1 - valid_frac)
    tr, va = normal[normal["game_date"] < cut], normal[normal["game_date"] >= cut]
    boosters, rounds, offsets = {}, {}, {}
    resid = tr["y_minutes"].to_numpy() - anchor(tr, use_stint)
    loss_anchor = np.zeros(len(va))  # per held-out game, summed over the three quantiles
    loss_model = np.zeros(len(va))
    for q in QUANTILES:
        offsets[q] = float(np.quantile(resid, q))
        p = {
            **params,
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "boost_from_average": False,
        }
        dtr = lgb.Dataset(
            _matrix(tr, feats),
            tr["y_minutes"],
            init_score=anchor(tr, use_stint) + offsets[q],
            free_raw_data=False,
        )
        dva = lgb.Dataset(
            _matrix(va, feats),
            va["y_minutes"],
            init_score=anchor(va, use_stint) + offsets[q],
            reference=dtr,
        )
        b = lgb.train(
            p,
            dtr,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        boosters[q] = b
        rounds[q] = b.best_iteration
        yv = va["y_minutes"].to_numpy()
        anchor_only = anchor(va, use_stint) + offsets[q]
        boosted = anchor_only + b.predict(_matrix(va, feats), num_iteration=b.best_iteration)
        loss_anchor += pinball_per_row(yv, anchor_only, q)
        loss_model += pinball_per_row(yv, boosted, q)
    # One decision for the whole distribution: trees on some quantiles but not others would
    # adjust the tails without the center (or vice versa) and distort its shape.
    if not beats_anchor(loss_anchor, loss_model, MIN_GAIN_OVER_ANCHOR):
        rounds = {q: 0 for q in rounds}  # no significant gain on held-out games: anchor alone
    return MinutesModel(
        boosters,
        league_rate,
        feats,
        str(df["game_date"].max().date()),
        params,
        rounds,
        offsets,
    )


def predict(
    model: MinutesModel, rows: pd.DataFrame, history: pd.DataFrame | None = None
) -> tuple[MinutesDist, pd.DataFrame]:
    """Distribution plus a summary frame (mean, q10, q50, q90, p_exit, low_confidence).
    `history`: finished games for exit rates (defaults to `rows` itself, for backtests)."""
    raw = model.raw_quantiles(rows)
    q10, q50, q90 = raw[0.1], raw[0.5], raw[0.9]
    q50 = np.clip(q50, 0.5, MAX_MINUTES)
    q10 = np.minimum(q10, q50 - 0.25)  # quantile models can cross; keep them ordered
    q90 = np.maximum(q90, q50 + 0.25)
    sd_left = np.maximum((q50 - q10) / Z90, 0.5)
    sd_right = np.maximum((q90 - q50) / Z90, 0.5)

    returning = rows["games_missed"].fillna(0).to_numpy() >= RETURN_GAMES_MISSED
    new_team = rows["new_team"].fillna(0).to_numpy() == 1
    early_rookie = rows["career_games"].fillna(0).to_numpy() < ROOKIE_EARLY_GAMES
    late_starter = (
        (rows["late_season"].fillna(0).to_numpy() == 1)
        & (rows["clinched_top6"].fillna(0).to_numpy() == 1)
        & (rows["started_share_l10"].fillna(0).to_numpy() >= 0.5)
    )
    low_conf = returning | new_team | early_rookie | late_starter
    sd_left = np.where(low_conf, sd_left * WIDEN, sd_left)
    sd_right = np.where(low_conf, sd_right * WIDEN, sd_right)

    hist = rows if history is None else history
    p_exit = exit_rates(rows, hist, model.league_exit_rate)
    avg = rows["min_avg_l10"].fillna(rows["std_min_avg"]).to_numpy(float)
    avg = np.where(np.isnan(avg), q50, avg)
    p_exit = np.where(avg >= EXIT_MIN_AVG, p_exit, model.league_exit_rate * avg / EXIT_MIN_AVG)
    exit_cap = np.minimum(EXIT_CAP, np.maximum(EXIT_FRAC * q50, 2.0))

    dist = MinutesDist(q50, sd_left, sd_right, p_exit, exit_cap)
    summary = pd.DataFrame(
        {
            "mean": dist.mean(),
            "q10": dist.quantile(0.1),
            "q50": dist.quantile(0.5),
            "q90": dist.quantile(0.9),
            "p_exit": p_exit,
            "low_confidence": low_conf,
        },
        index=rows.index,
    )
    return dist, summary


# --- evaluation --------------------------------------------------------------


def baseline(rows: pd.DataFrame) -> pd.Series:
    """The plan's baseline: average minutes over the last 10 games."""
    return rows["min_avg_l10"]


def _bucket_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    starter = rows["started_share_l10"].fillna(0) >= 0.5
    return {
        "all": pd.Series(True, index=rows.index),
        "starters": starter,
        "bench": ~starter,
        "back_to_back": rows["b2b"] == 1,
        "key_absence": rows["absent_starters"].fillna(0) >= 1,
        "returning": rows["games_missed"].fillna(0) >= RETURN_GAMES_MISSED,
        "new_team": rows["new_team"] == 1,
    }


def evaluate(model: MinutesModel, test: pd.DataFrame, history: pd.DataFrame | None = None) -> dict:
    """MAE vs the 10-game baseline and 80% interval coverage, overall and by bucket.
    Rows without a 10-game baseline (a player's first game) are excluded from the comparison
    but reported."""
    test = test.dropna(subset=["y_minutes"])
    dist, s = predict(model, test, history)
    y = test["y_minutes"]
    base = baseline(test)
    has_base = base.notna()
    lo, hi = s["q10"], s["q90"]
    out = {"rows": int(len(test)), "rows_without_baseline": int((~has_base).sum()), "buckets": {}}
    for name, mask in _bucket_masks(test).items():
        m = mask & has_base
        if m.sum() == 0:
            continue
        mae_model = float((s["q50"][m] - y[m]).abs().mean())  # the median minimizes MAE
        mae_base = float((base[m] - y[m]).abs().mean())
        out["buckets"][name] = {
            "n": int(m.sum()),
            "mae_model": round(mae_model, 3),
            "mae_baseline": round(mae_base, 3),
            "improvement": round(1 - mae_model / mae_base, 4) if mae_base else None,
            "coverage_80": round(float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()), 4),
        }
    allb = out["buckets"]["all"]
    out["passes_mae_target"] = allb["improvement"] is not None and allb["improvement"] >= 0.08
    out["passes_coverage_target"] = abs(allb["coverage_80"] - 0.80) <= 0.03
    return out


# --- command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="nbaprops.models.minutes", description="Train and evaluate the Stage 4 minutes model"
    )
    ap.add_argument("--features", default="data/features/train.parquet")
    ap.add_argument("--train", nargs="+", default=["2022-23", "2023-24"])
    ap.add_argument(
        "--test",
        nargs="+",
        default=["2024-25"],
        help="tuning season; keep 2025-26 untouched until the Stage 6 holdout check",
    )
    ap.add_argument("--out", default="artifacts/minutes")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.features)
    df = df[df["season_type"] != "preseason"]
    train, test = df[df["season"].isin(args.train)], df[df["season"].isin(args.test)]
    model = fit(train)
    report = evaluate(model, test, history=df[df["season"].isin(args.train + args.test)])
    run_id = f"minutes-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out = Path(args.out)
    model.save(out / f"{run_id}.json")
    (out / f"{run_id}.report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved {out / run_id}.json")


if __name__ == "__main__":
    main()
