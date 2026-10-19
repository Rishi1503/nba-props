"""Stage 5 (part 1): per-minute rates and the variance structure the simulation needs.

Rate models: one LightGBM model per stat with a Poisson objective on the stat count and
log(actual minutes) + log(anchor rate) as the offset, so trees learn adjustments to each
player's shrunk season rate. As with minutes, a model must beat its anchor on held-out games
(Poisson deviance) or it falls back to the anchor.

Points are simulated directly from their own fitted distribution, so points variance matches
the data; threes are then drawn within the points (each three is 3 of them), so a simulated
line never has more points from threes than points. Variance pieces are estimated for pts,
reb and ast.

Variance: with actual minutes as exposure, the leftover overdispersion of each stat is
estimated as a negative binomial alpha per usage tier (plan, Stage 5). Two shared parts of
that variance are also estimated, so the simulation can add correlation without inflating
anyone's marginal variance:
  v_env    shared by teammates in the same game (scoring environment), from teammate pairs
  v_player shared by one player's points and assists, from that same-player pair
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from nbaprops.features.build import FEATURE_COLUMNS
from nbaprops.models.gate import beats_anchor, poisson_dev_per_row

STATS = ["pts", "reb", "ast", "fg3m"]
SIM_STATS = ["pts", "reb", "ast"]  # drawn from NB; fg3m is drawn within the points
USAGE_TIERS = (0.0, 1 / 3, 2 / 3, 1.0)  # usage_l10 tertiles of the training set
MIN_GAIN_OVER_ANCHOR = 0.005
MAX_ROUNDS = 2000
EARLY_STOP = 100
PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "boost_from_average": False,
    "verbose": -1,
    "seed": 11,
    "deterministic": True,
    "num_threads": 1,
}
ALPHA_FLOOR = 0.005
SHARED_CAP = 0.05


def anchor_rate(rows: pd.DataFrame, stat: str, league_rate: float) -> np.ndarray:
    """Shrunk season rate, then last-20 rate, then the league rate (per minute)."""
    r = rows[f"{stat}_pm_shrunk"].astype(float).fillna(rows[f"{stat}_pm_l20"].astype(float))
    r = r.fillna(league_rate).to_numpy()
    return np.clip(r, 1e-4, None)


@dataclass
class RateModel:
    boosters: dict[str, lgb.Booster]
    rounds: dict[str, int]
    league_rates: dict[str, float]
    alpha: dict[str, list[float]]  # NB alpha per usage tier, per simulated stat
    usage_edges: list[float]
    v_env: dict[str, float]
    v_player: float
    usage_median: float = 0.2
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # --- prediction ---------------------------------------------------------

    def rates(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        """Expected per-minute rate per stat for each row."""
        x = rows[self.features].astype(float).to_numpy()
        out = {}
        for s in STATS:
            base = anchor_rate(rows, s, self.league_rates[s])
            n = self.rounds.get(s, 0)
            adj = self.boosters[s].predict(x, num_iteration=n, raw_score=True) if n else 0.0
            out[s] = base * np.exp(adj)
        # Threes can't account for more points than the player scores.
        out["fg3m"] = np.minimum(out["fg3m"], 0.9 * out["pts"] / 3)
        return out

    def alpha_for(self, rows: pd.DataFrame, stat: str) -> np.ndarray:
        usage = rows["usage_l10"].astype(float).fillna(self.usage_median).to_numpy()
        tier = np.searchsorted(self.usage_edges[1:-1], usage, side="right")
        return np.asarray(self.alpha[stat])[tier]

    # --- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "rates",
                    "version": 1,
                    "rounds": self.rounds,
                    "league_rates": self.league_rates,
                    "alpha": self.alpha,
                    "usage_edges": self.usage_edges,
                    "v_env": self.v_env,
                    "v_player": self.v_player,
                    "usage_median": self.usage_median,
                    "features": self.features,
                    "boosters": {s: b.model_to_string() for s, b in self.boosters.items()},
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RateModel":
        d = json.loads(path.read_text())
        return cls(
            {s: lgb.Booster(model_str=m) for s, m in d["boosters"].items()},
            d["rounds"],
            d["league_rates"],
            d["alpha"],
            d["usage_edges"],
            d["v_env"],
            d["v_player"],
            d["usage_median"],
            d["features"],
        )


# --- variance estimation -----------------------------------------------------


def nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB alpha: Var = mu + alpha * mu^2."""
    mu = np.clip(mu, 1e-6, None)
    return float(max(ALPHA_FLOOR, np.sum((y - mu) ** 2 - mu) / np.sum(mu**2)))


def shared_variance_teammates(df: pd.DataFrame, y: str, mu: str) -> float:
    """Variance of a multiplicative shock shared by teammates in a game:
    E[(y_i - mu_i)(y_j - mu_j)] = mu_i mu_j v for teammates i != j."""
    num = den = 0.0
    for _, g in df.groupby(["game_id", "team_id"], sort=False):
        r = (g[y] - g[mu]).to_numpy()
        m = g[mu].to_numpy()
        num += (r.sum() ** 2 - (r**2).sum()) / 2
        den += (m.sum() ** 2 - (m**2).sum()) / 2
    return float(np.clip(num / den, 0.0, SHARED_CAP)) if den else 0.0


def shared_variance_same_player(
    df: pd.DataFrame, pairs: list[tuple[str, str]], v_env: float
) -> float:
    """Shock shared by one player's stats, net of the team environment they also share."""
    num = den = 0.0
    for a, b in pairs:
        num += float(np.sum((df[f"y_{a}"] - df[f"mu_{a}"]) * (df[f"y_{b}"] - df[f"mu_{b}"])))
        den += float(np.sum(df[f"mu_{a}"] * df[f"mu_{b}"]))
    return float(np.clip(num / den - v_env, 0.0, SHARED_CAP)) if den else 0.0


# --- training ----------------------------------------------------------------


def fit(
    train: pd.DataFrame, valid_frac: float = 0.2, features: list[str] | None = None
) -> RateModel:
    df = train[(train["y_minutes"] > 0)].dropna(subset=["y_minutes"]).sort_values("game_date")
    cut = df["game_date"].quantile(1 - valid_frac)
    tr, va = df[df["game_date"] < cut], df[df["game_date"] >= cut]
    feats = list(features or FEATURE_COLUMNS)  # a subset, for feature-group comparisons
    x_tr = tr[feats].astype(float).to_numpy()
    x_va = va[feats].astype(float).to_numpy()
    league = {s: float(tr[f"y_{s}"].sum() / tr["y_minutes"].sum()) for s in STATS}
    boosters, rounds = {}, {}
    for s in STATS:
        off_tr = np.log(tr["y_minutes"].to_numpy()) + np.log(anchor_rate(tr, s, league[s]))
        off_va = np.log(va["y_minutes"].to_numpy()) + np.log(anchor_rate(va, s, league[s]))
        dtr = lgb.Dataset(x_tr, tr[f"y_{s}"], init_score=off_tr, free_raw_data=False)
        dva = lgb.Dataset(x_va, va[f"y_{s}"], init_score=off_va, reference=dtr)
        b = lgb.train(
            {**PARAMS, "metric": "poisson"},
            dtr,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        y_va = va[f"y_{s}"].to_numpy(float)
        mu_anchor = np.exp(off_va)
        mu_model = mu_anchor * np.exp(
            b.predict(x_va, num_iteration=b.best_iteration, raw_score=True)
        )
        keep = beats_anchor(
            poisson_dev_per_row(y_va, mu_anchor),
            poisson_dev_per_row(y_va, mu_model),
            MIN_GAIN_OVER_ANCHOR,
        )
        boosters[s], rounds[s] = b, (b.best_iteration if keep else 0)

    model = RateModel(boosters, rounds, league, {}, [], {}, 0.0, features=feats)
    # Variance pieces on held-out games (in-sample residuals understate variance).
    va = va.copy()
    for s, r in model.rates(va).items():
        va[f"mu_{s}"] = r * va["y_minutes"].to_numpy()
    usage = tr["usage_l10"].astype(float)
    edges = [float(usage.quantile(q)) for q in USAGE_TIERS]
    edges[0], edges[-1] = -np.inf, np.inf
    model.usage_edges = edges
    model.usage_median = float(usage.median())
    tier = np.searchsorted(
        edges[1:-1],
        va["usage_l10"].astype(float).fillna(model.usage_median).to_numpy(),
        side="right",
    )
    for s in SIM_STATS:
        y, mu = va[f"y_{s}"].to_numpy(float), va[f"mu_{s}"].to_numpy()
        model.alpha[s] = [
            nb_alpha(y[tier == k], mu[tier == k]) if (tier == k).sum() > 50 else nb_alpha(y, mu)
            for k in range(3)
        ]
        model.v_env[s] = shared_variance_teammates(va, f"y_{s}", f"mu_{s}")
    model.v_player = shared_variance_same_player(
        va,
        [("pts", "ast")],
        float(np.mean([model.v_env["pts"], model.v_env["ast"]])),
    )
    return model
