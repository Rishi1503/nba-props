"""Stage 5 (part 2): simulate whole games.

Each of the n simulations of a game:
  1. draws a scenario: game total T ~ N(T0, sd_total) and home margin M ~ N(M0, sd_margin),
     centered on the market total and spread when logged, else the team-rating estimates.
     The rate models' expectations rest on team ratings, so when the market expects a team to
     score more than the ratings do, that team's players are shifted up accordingly;
  2. turns |M| into a blowout score B ~ N(0, 1) and ties every player's minutes to it through a
     Gaussian copula (starters lose minutes in blowouts, bench players gain), keeping each
     player's Stage 4 minutes distribution exactly;
  3. scales each team's scoring rates by its scenario points relative to expectation, and
     rebounds by the pace implied by T, shrunk so the shared variance matches the data (v_env);
  4. applies one shock per player to points and assists (v_player);
  5. draws points, rebounds and assists as negative binomial with the leftover dispersion, so
     each player's total variance still matches the data; then draws threes within the
     points, binomial on the number of threes the points could hold, with the probability set
     so the expected threes equal the rate model's.

Every sigma, correlation and variance here is estimated from data (`fit_scenario`,
`rates.fit`). Simulations are seeded from run_id and game_id, so reruns are identical.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from nbaprops.models import minutes as mm
from nbaprops.models.rates import RateModel

ALPHA_FLOOR = 0.005
MARKETS = ["pts", "reb", "ast", "fg3m", "pra"]
STARTER_SHARE = 0.5
RHO_CAP = 0.8
MAX_SHIFT = 0.15  # cap on the market-vs-rating environment shift


@dataclass
class ScenarioParams:
    sd_total: float = 18.0
    sd_margin: float = 13.0
    rho_starter: float = -0.25
    rho_bench: float = 0.15

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "ScenarioParams":
        return cls(**json.loads(path.read_text()))


def blowout_score(abs_margin: np.ndarray, m0: np.ndarray, sd: float) -> np.ndarray:
    """Normal score of |margin| under the game's own margin distribution N(m0, sd): B ~ N(0,1)
    by construction, higher = more lopsided than expected for this matchup."""
    f = ndtr((abs_margin - m0) / sd) - ndtr((-abs_margin - m0) / sd)
    return ndtri(np.clip(f, 1e-6, 1 - 1e-6))


def fit_scenario(
    frame: pd.DataFrame, minutes_model: mm.MinutesModel, history: pd.DataFrame | None = None
) -> ScenarioParams:
    """Estimate sigmas from rating errors and the minutes-blowout link from minutes PITs."""
    df = frame.dropna(
        subset=["y_team_margin", "y_game_total", "rating_team_margin", "rating_total", "y_minutes"]
    )
    # Latest 30% of dates: the minutes trees never trained on them (in-sample PITs would
    # understate the noise and distort the correlation).
    df = df[df["game_date"] >= df["game_date"].quantile(0.7)]
    games = df[df["is_home"] == 1].drop_duplicates("game_id")
    sd_total = float(np.std(games["y_game_total"] - games["rating_total"]))
    sd_margin = float(np.std(games["y_team_margin"] - games["rating_team_margin"]))
    dist, _ = mm.predict(minutes_model, df, history)
    u = dist.cdf(df["y_minutes"].to_numpy())
    # PIT of the normal-game part only: early exits are drawn independently in the simulation
    z = ndtri(np.clip(u, 0.005, 0.995))
    b = blowout_score(
        np.abs(df["y_team_margin"].to_numpy()), df["rating_team_margin"].to_numpy(), sd_margin
    )
    starter = df["started_share_l10"].fillna(0).to_numpy() >= STARTER_SHARE
    normal = ~mm.is_early_exit(df["y_minutes"], df["min_avg_l10"]).to_numpy()

    def corr(mask):
        m = mask & normal
        return (
            float(np.clip(np.corrcoef(z[m], b[m])[0, 1], -RHO_CAP, RHO_CAP))
            if m.sum() > 50
            else 0.0
        )

    return ScenarioParams(sd_total, sd_margin, corr(starter), corr(~starter))


def game_seed(run_id: str, game_id: str) -> int:
    return int(hashlib.sha256(f"{run_id}:{game_id}".encode()).hexdigest()[:16], 16)


def _nb(rng: np.random.Generator, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Negative binomial with mean mu and Var = mu + alpha mu^2 (gamma-Poisson)."""
    a = np.maximum(alpha, ALPHA_FLOOR)[:, None]
    g = rng.gamma(1.0 / a, a, size=mu.shape)
    return rng.poisson(mu * g)


def _shrunk_multiplier(raw: np.ndarray, target_var: float) -> np.ndarray:
    """1 + lam * (raw - 1), with lam set so the multiplier's variance equals target_var."""
    v = float(np.var(raw))
    lam = min(1.0, np.sqrt(target_var / v)) if v > 0 and target_var > 0 else 0.0
    return np.clip(1 + lam * (raw - 1), 0.3, 2.0)


def simulate_game(
    rows: pd.DataFrame,
    dist: mm.MinutesDist,
    rates: RateModel,
    params: ScenarioParams,
    n_sims: int,
    seed: int,
    total0: float | None = None,
    home_margin0: float | None = None,
) -> dict[str, np.ndarray]:
    """rows: every projected player in ONE game (both teams), aligned with `dist`.
    Returns arrays of shape (players, n_sims) for minutes and each market."""
    rng = np.random.default_rng(seed)
    is_home = rows["is_home"].astype(int).to_numpy() == 1
    # The rate models' expectations rest on the team-rating view of the game; the market view
    # (when logged) centers the scenario and shifts each team's environment relative to it.
    rating_total = float(rows["rating_total"].astype(float).mean())
    m = rows["rating_team_margin"].astype(float).to_numpy()
    rating_margin = float(np.nanmean(np.where(is_home, m, -m)))
    if not np.isfinite(rating_total):
        rating_total = total0 if total0 is not None and np.isfinite(total0) else 220.0
    if not np.isfinite(rating_margin):
        rating_margin = 0.0
    if total0 is None or not np.isfinite(total0):
        total0 = rating_total
    if home_margin0 is None or not np.isfinite(home_margin0):
        home_margin0 = rating_margin

    # 1. scenario
    total = total0 + params.sd_total * rng.standard_normal(n_sims)
    total = np.maximum(total, 0.5 * total0)
    margin = home_margin0 + params.sd_margin * rng.standard_normal(n_sims)
    b = blowout_score(np.abs(margin), np.full(n_sims, home_margin0), params.sd_margin)

    # 2. minutes, tied to the blowout score
    starter = rows["started_share_l10"].fillna(0).to_numpy() >= STARTER_SHARE
    rho = np.where(starter, params.rho_starter, params.rho_bench)[:, None]
    eps = rng.standard_normal((len(rows), n_sims))
    u = ndtr(rho * b[None, :] + np.sqrt(1 - rho**2) * eps)
    minutes = dist.sample_correlated(u, rng)

    # 3. team environment
    team_margin0 = np.where(is_home, home_margin0, -home_margin0)
    team_pts0 = np.maximum((total0 + team_margin0) / 2, 1.0)
    rating_team_margin = np.where(is_home, rating_margin, -rating_margin)
    rating_team_pts = np.maximum((rating_total + rating_team_margin) / 2, 1.0)
    # Market-vs-rating shift in expected environment, applied in full (it moves means), capped
    # as a guard against bad data. Only the game-to-game variation around it is shrunk.
    score_shift = np.clip(team_pts0 / rating_team_pts, 1 - MAX_SHIFT, 1 + MAX_SHIFT)[:, None]
    pace_shift = float(np.clip(total0 / rating_total, 1 - MAX_SHIFT, 1 + MAX_SHIFT))
    team_margin = np.where(is_home[:, None], margin[None, :], -margin[None, :])
    raw_score = ((total[None, :] + team_margin) / 2) / team_pts0[:, None]
    raw_pace = np.broadcast_to(total[None, :] / total0, raw_score.shape)
    env = {}
    for stat in ("pts", "ast"):
        varying = (
            np.vstack([_shrunk_multiplier(r, rates.v_env.get(stat, 0.0)) for r in raw_score])
            if len(rows)
            else raw_score
        )
        env[stat] = score_shift * varying
    env["reb"] = pace_shift * _shrunk_multiplier(raw_pace, rates.v_env.get("reb", 0.0))

    # 4. player shock
    vp = rates.v_player
    shock = rng.gamma(1 / vp, vp, size=(len(rows), n_sims)) if vp > 0 else 1.0

    # 5. stats
    r = rates.rates(rows)
    out = {"minutes": minutes}
    for stat in ("pts", "reb", "ast"):
        shared = rates.v_env.get(stat, 0.0) + (vp if stat != "reb" else 0.0)
        alpha = np.maximum(rates.alpha_for(rows, stat) - shared, ALPHA_FLOOR)
        mu = minutes * r[stat][:, None] * env[stat] * (shock if stat != "reb" else 1.0)
        out[stat] = _nb(rng, mu, alpha)
    slots = out["pts"] // 3  # threes the points could hold
    expected_threes = (minutes * r["fg3m"][:, None]).mean(axis=1)
    p3 = np.clip(expected_threes / np.maximum(slots.mean(axis=1), 1e-9), 0.0, 1.0)
    out["fg3m"] = rng.binomial(slots, p3[:, None])
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    out["_total"] = total
    out["_home_margin"] = margin
    return out


# --- reading results ---------------------------------------------------------


def p_over(draws: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """P(stat > line) per player; lines shape (players,) or (players, k)."""
    lines = np.asarray(lines, dtype=float)
    if lines.ndim == 1:
        return (draws > lines[:, None]).mean(axis=1)
    return (draws[:, None, :] > lines[:, :, None]).mean(axis=2)


def p_push(draws: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """P(stat == line); zero for half-point lines."""
    lines = np.asarray(lines, dtype=float)
    return (draws == lines[:, None]).mean(axis=1)


def line_grid(draws: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> list[list[float]]:
    """Half-point lines spanning each player's 1st-99th percentile, wide enough for far-out
    alternate and milestone rungs (milestone N+ = over N - 0.5)."""
    lo = np.floor(np.quantile(draws, lo_q, axis=1))
    hi = np.ceil(np.quantile(draws, hi_q, axis=1))
    # From just below the 1st percentile to just above the 99th, never below 0.5.
    return [list(np.arange(max(a - 1, 0), b + 1) + 0.5) for a, b in zip(lo, hi, strict=True)]


def line_probs(p_over: dict[str, float], line: float) -> tuple[float, float, float]:
    """(over, under, push) for any line from a stored half-point grid {"24.5": p, ...}.

    Stats are whole numbers, so over a whole line L is over L + 0.5, and a push is
    P(over L - 0.5) - P(over L + 0.5) (plan, Stage 5 step 4). Lines past either end of the
    grid (which spans the 1st-99th percentile) take the nearest end's value, rounded to 0 or 1.
    """
    grid = sorted((float(k), v) for k, v in p_over.items())
    lo, hi = grid[0][0], grid[-1][0]

    def over_half(x: float) -> float:
        if x < lo:
            return 1.0
        if x > hi:
            return 0.0
        return dict(grid)[x]

    if float(line).is_integer():
        over = over_half(line + 0.5)
        push = max(over_half(line - 0.5) - over, 0.0)
    else:
        over, push = over_half(line), 0.0
    return over, max(1.0 - over - push, 0.0), push
