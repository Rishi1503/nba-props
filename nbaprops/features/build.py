"""Stage 3: model inputs for every (player, game), using only information available beforehand.

How leakage is prevented: every rolling quantity is computed as the *state after* each game
(player, team, defense-by-position). A target game at date D is joined to the latest state
with date < D (`merge_asof(..., allow_exact_matches=False)`). Training rows and live slates go
through the same join, so the backtest sees exactly what production sees. Schedule facts
(rest, back-to-backs, road trips) are known in advance and joined directly.

Absences: for finished games, a rotation player who didn't play is "absent"; for upcoming
games, a rotation player listed Out or Doubtful in the latest injury report at `as_of_ts`.

Spreads and totals: `rating_*` columns come from team ratings and exist for every season, so
they are model features. `market_*` columns come from logged odds and exist only from the
2026-27 season on; they center the Stage 5 simulation and are not v1 training features.
"""

import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import text

from nbaprops.config import load_strategy
from nbaprops.settings import get_settings

log = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta", "tov"]
WINDOWS = (5, 10, 20)
PRIOR_MINUTES = 250.0  # weight of the prior, in minutes, when shrinking per-minute rates
PRIOR_MIN_SEASON_MINUTES = 300.0  # last-season minutes needed to use the player's own prior
ROTATION_LOOKBACK = 10
ROTATION_MIN_AVG = 12.0
OPP_WINDOW = 15
TEAM_WINDOW = 20
HOME_COURT_PTS = 2.0
REGULAR_SEASON_GAMES = 82
LATE_SEASON_GAMES_LEFT = 15
ABSENT_STATUSES = ("Out", "Doubtful")
# Roster changes (trades, returns from absence) fade as recent form absorbs them.
DEPARTED_FADE_GAMES = ROTATION_LOOKBACK  # weight 1.0 the first game after leaving, then -0.1/game
ARRIVED_FADE_GAMES = 10  # weight 1.0 on the debut with the new team, then -0.1/game
RETURN_MIN_MISSED = 3  # a "return" follows at least this many missed team games
RETURN_FADE_GAMES = 5  # weight 1.0 on the return game, then 0.8, 0.6, 0.4, 0.2
# With/without splits: games with an absent teammate vs games with him, same team.
WO_SHRINK_GAMES = 5.0  # deltas are scaled by n / (n + 5): full weight needs many games
WO_MIN_WITH = 3  # need at least this many games together to compare against
WO_MAX_GAMES = 82  # look back at most this many of the player's games with the team
STINT_WINDOW = 10
# Games into the current stint, capped: what matters is whether he joined recently. An
# uncapped tenure count rises forever and acts as a time index trees can overfit to.
STINT_GAMES_CAP = 30
EAST = {
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}

FEATURE_COLUMNS = [
    # recent form (per minute)
    *[f"{s}_pm_l{n}" for s in STATS for n in WINDOWS],
    "pts_pm_ewm",
    "fg3_pct_l20",
    # season to date, shrunk toward a prior
    *[f"{s}_pm_shrunk" for s in STATS],
    "prior_source",
    "std_games",
    "std_pts_avg",
    "std_reb_avg",
    "std_ast_avg",
    "std_fg3m_avg",
    # minutes and role
    "min_avg_l3",
    "min_avg_l5",
    "min_avg_l10",
    "min_avg_l20",
    "min_ewm",
    "std_min_avg",
    "prev_season_min_avg",
    "minutes_share_l10",
    "started_share_l10",
    "usage_l10",
    "preseason_min_avg",
    "career_games",
    "games_missed",
    "rookie",
    "new_team",
    "games_with_team",
    # since joining the current team (a stint resets whenever the team changes)
    "stint_games",
    "stint_change",
    "stint_min_avg_l10",
    "stint_usage_l10",
    "stint_pts_pm_l10",
    "stint_reb_pm_l10",
    "stint_ast_pm_l10",
    "stint_fg3m_pm_l10",
    # teammates
    "absent_n",
    "absent_min_sum",
    "absent_usage_sum",
    "absent_starters",
    "same_pos_absent_min",
    # roster changes around the player (weighted, fading as recent form absorbs them)
    "departed_min_sum",
    "departed_usage_sum",
    "arrived_min_sum",
    "arrived_usage_sum",
    "returning_min_sum",
    "returning_usage_sum",
    # this player with vs without today's most important absent teammate
    "wo_games",
    "wo_min_delta",
    "wo_usage_delta",
    "wo_pts_pm_delta",
    "wo_reb_pm_delta",
    "wo_ast_pm_delta",
    "wo_fg3m_pm_delta",
    # opponent absences
    "opp_absent_n",
    "opp_absent_min_sum",
    "opp_absent_usage_sum",
    "opp_absent_starters",
    "opp_same_pos_absent_min",
    # opponent and own team
    "opp_def_rtg_l15",
    "opp_def_rtg_std",
    "opp_pace_l15",
    "opp_dreb_pct_l15",
    "opp_allow_pts_pm_l15",
    "opp_allow_reb_pm_l15",
    "opp_allow_ast_pm_l15",
    "opp_allow_fg3m_pm_l15",
    "team_pace_l20",
    "team_off_rtg_l20",
    # game context
    "is_home",
    "rating_team_margin",
    "rating_total",
    "rating_team_pts",
    # schedule
    "rest_days",
    "b2b",
    "games_last_7",
    "road_trip_len",
    "opp_rest_days",
    "player_rest_days",
    # standings / late season
    "games_remaining",
    "win_pct",
    "conf_rank",
    "games_back_6",
    "games_back_10",
    "late_season",
    "clinched_top6",
    "eliminated",
    "is_postseason",
]
# Outcomes, never inputs. y_team_margin / y_game_total let Stage 5 estimate how blowouts and
# scoring environments move minutes and stats.
TARGET_COLUMNS = [
    "y_minutes",
    "y_pts",
    "y_reb",
    "y_ast",
    "y_fg3m",
    "y_pra",
    "y_team_margin",
    "y_game_total",
]


@dataclass
class FeatureInputs:
    """Plain DataFrames, loaded from Postgres by `load_inputs` or built directly in tests.

    logs:      player_game_logs + games columns (game_date, season, season_type)
    team_logs: team_game_logs + games columns
    games:     games (all statuses, including scheduled)
    players:   player_id, position
    teams:     team_id, abbreviation
    injuries:  injury_snapshots (report_ts, game_id, team_id, player_id, status); optional
    game_lines: game_line_snapshots; optional
    """

    logs: pd.DataFrame
    team_logs: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    injuries: pd.DataFrame | None = None
    game_lines: pd.DataFrame | None = None


# --- helpers -----------------------------------------------------------------


def prev_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[2:]}"


def primary_position(pos) -> str:
    if not isinstance(pos, str) or not pos:
        return "U"
    first = pos.strip()[0].upper()
    return first if first in "GFC" else "U"


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[ns]")


def _played(logs: pd.DataFrame, include_preseason: bool = False) -> pd.DataFrame:
    p = logs[(~logs["dnp"].fillna(False).astype(bool)) & (logs["minutes"].fillna(0) > 0)]
    if not include_preseason:
        p = p[p["season_type"] != "preseason"]
    return p.copy()


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str | list, right_by=None) -> pd.DataFrame:
    """Join each left row to the latest right row with an earlier date (strictly before)."""
    left = left.sort_values("game_date")
    right = right.sort_values("game_date")
    kwargs = {"by": by} if right_by is None else {"left_by": by, "right_by": right_by}
    return pd.merge_asof(left, right, on="game_date", allow_exact_matches=False, **kwargs)


# --- state tables (one row per entity per game, *after* that game) ------------


def player_state(
    logs: pd.DataFrame, team_logs: pd.DataFrame, games: pd.DataFrame | None = None
) -> pd.DataFrame:
    p = _played(logs).sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    p = p.merge(
        team_logs[["game_id", "team_id", "minutes"]].rename(columns={"minutes": "team_min"}),
        on=["game_id", "team_id"],
        how="left",
    )
    p["share"] = p["minutes"] / p["team_min"]
    p["started_f"] = p["started"].astype(float)  # NaN when unknown: excluded from the share
    out = pd.DataFrame(
        {
            "player_id": p["player_id"],
            "game_date": p["game_date"],
            "state_season": p["season"],
            "state_team_id": p["team_id"],
            "last_game_date": p["game_date"],
        }
    )
    g = p.groupby("player_id", sort=False)
    for n in WINDOWS:
        sums = (
            g[["minutes", *STATS]].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        out[f"min_avg_l{n}"] = (
            g["minutes"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        for s in STATS:
            out[f"{s}_pm_l{n}"] = sums[s] / sums["minutes"]
        if n == 20:
            out["fg3_pct_l20"] = sums["fg3m"] / sums["fg3a"].replace(0, np.nan)
    out["min_avg_l3"] = (
        g["minutes"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    ewm = g[["minutes", "pts"]].transform(lambda s: s.ewm(halflife=5, min_periods=1).mean())
    out["min_ewm"] = ewm["minutes"]
    out["pts_pm_ewm"] = ewm["pts"] / ewm["minutes"]
    out["usage_l10"] = (
        g["usage_pct"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    out["minutes_share_l10"] = (
        g["share"].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["started_share_l10"] = (
        g["started_f"].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["career_games"] = g.cumcount() + 1
    gs = p.groupby(["player_id", "season"], sort=False)
    out["std_games"] = gs.cumcount() + 1
    cums = gs[["minutes", *STATS]].cumsum()
    out["std_minutes"] = cums["minutes"]
    for s in STATS:
        out[f"std_{s}"] = cums[s]
    out["games_with_team"] = p.groupby(["player_id", "season", "team_id"]).cumcount() + 1

    # --- stints: a run of consecutive games for one team (resets on any team change) ---
    prev_team = g["team_id"].shift(1)
    start = prev_team.isna() | (prev_team != p["team_id"])
    out["stint_change"] = (start & prev_team.notna()).astype(int)  # began with a team change
    stint_id = start.astype(int).groupby(p["player_id"]).cumsum()
    keys = [p["player_id"], stint_id]
    gst = p.groupby(keys, sort=False)
    out["stint_games"] = gst.cumcount() + 1
    out["stint_change"] = out.groupby(keys)["stint_change"].transform("first")
    roll = gst[["minutes", *STATS]].rolling(STINT_WINDOW, min_periods=1).sum()
    roll = roll.reset_index(level=[0, 1], drop=True)
    out["stint_min_avg_l10"] = (
        gst["minutes"]
        .rolling(STINT_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["stint_usage_l10"] = (
        gst["usage_pct"]
        .rolling(STINT_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .astype(float)
    )
    for s in ("pts", "reb", "ast", "fg3m"):
        out[f"stint_{s}_pm_l10"] = roll[s] / roll["minutes"]
    # Role brought from the previous team: his last-10 minutes and usage just before the move.
    prev_min = out.groupby("player_id")["min_avg_l10"].shift(1)
    prev_usage = out.groupby("player_id")["usage_l10"].shift(1)
    out["prior_role_min"] = prev_min.where(start).groupby(keys).transform("first")
    out["prior_role_usage"] = prev_usage.where(start).groupby(keys).transform("first")

    # --- returns: a game after RETURN_MIN_MISSED+ missed team games, same stint ---
    if games is not None:
        frame = pd.DataFrame(
            {
                "team_id": p["team_id"],
                "game_date": p["game_date"],
                "last_game_date": g["game_date"].shift(1),
            }
        )
        missed = games_missed(games, frame)
        is_return = (missed >= RETURN_MIN_MISSED) & ~start.to_numpy()
    else:
        is_return = np.zeros(len(p), dtype=bool)
    ret = pd.Series(is_return, index=p.index)
    out["last_return_date"] = p["game_date"].where(ret).groupby(keys).ffill()
    # Role before the absence: last-10 minutes and usage as of his last game before it.
    out["return_role_min"] = prev_min.where(ret).groupby(keys).ffill()
    out["return_role_usage"] = prev_usage.where(ret).groupby(keys).ffill()
    return out


def team_state(team_logs: pd.DataFrame) -> pd.DataFrame:
    t = (
        team_logs.merge(
            team_logs[["game_id", "team_id", "oreb", "pts"]].rename(
                columns={"team_id": "opp_team_id", "oreb": "opp_oreb", "pts": "opp_pts"}
            ),
            on=["game_id", "opp_team_id"],
            how="left",
        )
        .sort_values(["team_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )
    for c in ("off_rtg", "def_rtg", "pace", "possessions", "minutes"):
        t[c] = t[c].astype(float)
    t["dreb_pct"] = t["dreb"] / (t["dreb"] + t["opp_oreb"])
    t["net"] = t["off_rtg"] - t["def_rtg"]
    out = pd.DataFrame({"team_id": t["team_id"], "game_date": t["game_date"]})
    g = t.groupby("team_id", sort=False)

    def roll(col, n):
        return g[col].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)

    for col in ("def_rtg", "pace", "dreb_pct"):
        out[f"{col}_l{OPP_WINDOW}"] = roll(col, OPP_WINDOW)
    for col in ("off_rtg", "def_rtg", "pace", "net"):
        out[f"{col}_l{TEAM_WINDOW}"] = roll(col, TEAM_WINDOW)
    out["def_rtg_std"] = t.groupby(["team_id", "season"])["def_rtg"].transform(
        lambda s: s.expanding().mean()
    )
    out["state_season"] = t["season"]
    return out


def defense_by_position_state(
    logs: pd.DataFrame, team_logs: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    """Per-minute production allowed by each defense to each position, last 15 games."""
    p = _played(logs).merge(players[["player_id", "pos"]], on="player_id", how="left")
    p["pos"] = p["pos"].fillna("U")
    p = p.merge(team_logs[["game_id", "team_id", "opp_team_id"]], on=["game_id", "team_id"])
    agg = (
        p.groupby(["opp_team_id", "pos", "game_id", "game_date"], as_index=False)[
            ["minutes", "pts", "reb", "ast", "fg3m"]
        ]
        .sum()
        .rename(columns={"opp_team_id": "def_team_id"})
        .sort_values(["def_team_id", "pos", "game_date"])
        .reset_index(drop=True)
    )
    g = agg.groupby(["def_team_id", "pos"], sort=False)
    sums = g[["minutes", "pts", "reb", "ast", "fg3m"]].rolling(OPP_WINDOW, min_periods=1).sum()
    sums = sums.reset_index(level=[0, 1], drop=True)
    out = agg[["def_team_id", "pos", "game_date"]].copy()
    for s in ("pts", "reb", "ast", "fg3m"):
        out[f"opp_allow_{s}_pm_l15"] = sums[s] / sums["minutes"]
    out["def_key"] = out["def_team_id"].astype(str) + ":" + out["pos"]
    return out.drop(columns=["def_team_id", "pos"])


def _team_game_dates(games: pd.DataFrame) -> dict[int, np.ndarray]:
    """Sorted dates of each team's finished, non-preseason games."""
    fin = games[(games["status"] == "final") & (games["season_type"] != "preseason")]
    dates: dict[int, list] = {}
    for col in ("home_team_id", "away_team_id"):
        for team, grp in fin.groupby(col):
            dates.setdefault(team, []).extend(grp["game_date"].tolist())
    return {k: np.sort(np.array(v, dtype="datetime64[ns]")) for k, v in dates.items()}


def _team_games_between(
    games: pd.DataFrame, teams, starts, ends, start_side: str = "left"
) -> np.ndarray:
    """Per row: the team's finished games with start <= date < end (start_side="left") or
    start < date < end ("right"). 0 where start is missing."""
    arrays = _team_game_dates(games)
    out = np.zeros(len(teams), dtype=int)
    for i, (team, start, end) in enumerate(zip(teams, starts, ends, strict=True)):
        arr = arrays.get(team)
        if arr is None or pd.isna(start):
            continue
        out[i] = np.searchsorted(arr, np.datetime64(end), "left") - np.searchsorted(
            arr, np.datetime64(start), start_side
        )
    return np.maximum(out, 0)


def games_missed(games: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
    """Team games (finished, non-preseason) between the player's last game and this one: the
    return-from-absence signal. 0 for a player with no earlier game."""
    return _team_games_between(
        games, t["team_id"], t["last_game_date"], t["game_date"], start_side="right"
    )


def _empty_wo(cols: list[str]) -> pd.DataFrame:
    """Typed empty result: an untyped empty frame would turn merged columns into objects."""
    types = {"game_id": "object", "player_id": "int64"}
    return pd.DataFrame({c: pd.Series(dtype=types.get(c, "float64")) for c in cols})


def with_without(logs: pd.DataFrame, rotation: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """For each target row whose team is missing someone today: how this player's minutes,
    usage and per-minute rates differed in his earlier games with this team when that
    teammate (the most important absent one) was also absent, versus games he played.

    Only earlier games count (same team, at most WO_MAX_GAMES back). Deltas need at least
    WO_MIN_WITH games together, and are scaled by n / (n + WO_SHRINK_GAMES) so a split built
    on a game or two stays near zero."""
    cols = [
        "game_id",
        "player_id",
        "wo_games",
        "wo_min_delta",
        "wo_usage_delta",
        *[f"wo_{s}_pm_delta" for s in ("pts", "reb", "ast", "fg3m")],
    ]
    if not len(rotation):
        return _empty_wo(cols)
    absent = rotation[rotation["is_absent"]]
    if not len(absent):
        return _empty_wo(cols)
    absent_by_tg = absent.groupby(["game_id", "team_id"])["player_id"].agg(set).to_dict()
    key = (
        absent.sort_values("rotation_min", ascending=False)
        .drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "player_id"]]
        .rename(columns={"player_id": "key_absent"})
    )
    rows = t[["game_id", "player_id", "team_id", "game_date"]].merge(
        key, on=["game_id", "team_id"], how="inner"
    )
    rows = rows[rows["key_absent"] != rows["player_id"]]
    played = _played(logs).sort_values("game_date")
    played_by_tg = played.groupby(["game_id", "team_id"])["player_id"].agg(set).to_dict()
    hist = {}
    for (pid, team), g in played.groupby(["player_id", "team_id"], sort=False):
        hist[(pid, team)] = (
            g["game_date"].to_numpy(dtype="datetime64[ns]"),
            g["game_id"].tolist(),
            g["minutes"].to_numpy(float),
            g["usage_pct"].to_numpy(float),
            {s: g[s].to_numpy(float) for s in ("pts", "reb", "ast", "fg3m")},
        )
    out = []
    for r in rows.itertuples(index=False):
        h = hist.get((r.player_id, r.team_id))
        rec = {c: 0.0 for c in cols[2:]} | {"game_id": r.game_id, "player_id": r.player_id}
        if h is None:
            out.append(rec)
            continue
        dates, gids, mins, usage, stats = h
        n = int(np.searchsorted(dates, np.datetime64(r.game_date), "left"))
        lo = max(0, n - WO_MAX_GAMES)
        wo = np.array(
            [r.key_absent in absent_by_tg.get((g, r.team_id), ()) for g in gids[lo:n]], dtype=bool
        )
        wi = np.array(
            [r.key_absent in played_by_tg.get((g, r.team_id), ()) for g in gids[lo:n]], dtype=bool
        )
        rec["wo_games"] = int(wo.sum())
        if wo.sum() == 0 or wi.sum() < WO_MIN_WITH:
            out.append(rec)
            continue
        shrink = wo.sum() / (wo.sum() + WO_SHRINK_GAMES)
        m = mins[lo:n]
        u = usage[lo:n]
        rec["wo_min_delta"] = shrink * (m[wo].mean() - m[wi].mean())
        u_wo, u_wi = u[wo][~np.isnan(u[wo])], u[wi][~np.isnan(u[wi])]
        rec["wo_usage_delta"] = (
            shrink * (u_wo.mean() - u_wi.mean()) if len(u_wo) and len(u_wi) else 0.0
        )
        for s, v in stats.items():
            v = v[lo:n]
            rec[f"wo_{s}_pm_delta"] = shrink * (
                v[wo].sum() / m[wo].sum() - v[wi].sum() / m[wi].sum()
            )
        out.append(rec)
    return pd.DataFrame(out, columns=cols)


# --- priors ------------------------------------------------------------------


def season_priors(logs: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player totals by season, league per-minute rates by season and position)."""
    p = _played(logs)
    totals = p.groupby(["player_id", "season"], as_index=False)[["minutes", *STATS]].sum()
    totals["games"] = p.groupby(["player_id", "season"]).size().values
    pos = totals.merge(players[["player_id", "pos"]], on="player_id", how="left")
    pos["pos"] = pos["pos"].fillna("U")
    league = pos.groupby(["season", "pos"], as_index=False)[["minutes", *STATS]].sum()
    for s in STATS:
        league[f"pos_{s}_pm"] = league[s] / league["minutes"]
    return totals, league[["season", "pos", *[f"pos_{s}_pm" for s in STATS]]]


# --- teammates ---------------------------------------------------------------


def team_absences(
    logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    players: pd.DataFrame,
    team_games: pd.DataFrame,
    statuses: pd.DataFrame | None = None,
    current_team: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (game, team) in team_games [game_id, team_id, game_date, season]:
    rotation = players with >=3 appearances and >=12 avg minutes in the team's previous 10
    games of the same season (any season type, so preseason sets the opening-night rotation),
    excluding players whose latest game before the date was for another team.

    Absent = rotation players who didn't play (finished games), or who are listed in
    `statuses` [game_id, team_id, player_id, status] as Out/Doubtful (upcoming games).

    Departed = players who met the rotation bar in the window but have left the team:
      - they have since played for another team (any game), or
      - live (`current_team`, today's rosters): they are on another team's roster now, or
      - finished games: they didn't play this game and never play for this team again
        (their next game is elsewhere). History has no trade dates, so this is the one
        deliberate look-ahead: it only separates "traded" from "absent" for players who
        never return; live, today's roster says it directly.
    Departed players are not absent; their minutes and usage count in `departed_*`, weighted
    1.0 on the first game after they left and fading by 0.1 per team game.
    Returns (team-level summary, rotation table).
    """
    played = _played(logs, include_preseason=True)
    pos = dict(zip(players["player_id"], players["pos"], strict=False))
    by_game: dict[str, list[tuple]] = defaultdict(list)
    for r in played.itertuples(index=False):
        by_game[r.game_id].append(
            (
                r.player_id,
                r.team_id,
                float(r.minutes),
                float(r.usage_pct) if pd.notna(r.usage_pct) else np.nan,
                None if pd.isna(r.started) else bool(r.started),  # unknown is not "bench"
            )
        )
    history = team_logs[["team_id", "game_id", "game_date", "season"]].sort_values("game_date")
    team_hist: dict[tuple, tuple[list, list]] = {}
    for (team, season), grp in history.groupby(["team_id", "season"]):
        team_hist[(team, season)] = (list(grp["game_date"]), list(grp["game_id"]))
    last_team = played.sort_values("game_date")[["player_id", "team_id", "game_date", "season"]]
    player_dates: dict[int, tuple[list, list, list]] = {}
    for pid, grp in last_team.groupby("player_id"):
        player_dates[pid] = (list(grp["game_date"]), list(grp["team_id"]), list(grp["season"]))

    status_map: dict[tuple, set] = defaultdict(set)
    if statuses is not None and len(statuses):
        out_rows = statuses[statuses["status"].isin(ABSENT_STATUSES)]
        for r in out_rows.itertuples(index=False):
            status_map[(r.game_id, r.team_id)].add(r.player_id)
    # Who played FOR each team in each game: a player traded to the opponent can face his old
    # team, so "played in this game" is not the same as "played for this team".
    played_for: dict[tuple, set] = defaultdict(set)
    for gid, rows in by_game.items():
        for pid, team, *_ in rows:
            played_for[(gid, team)].add(pid)

    summary, rotation_rows = [], []
    for tg in team_games.itertuples(index=False):
        dates, gids = team_hist.get((tg.team_id, tg.season), ([], []))
        window = gids[
            max(0, bisect_left(dates, tg.game_date) - ROTATION_LOOKBACK) : bisect_left(
                dates, tg.game_date
            )
        ]
        acc: dict[int, list] = defaultdict(lambda: [0, 0.0, 0.0, 0, 0, 0, -1])
        for w_idx, gid in enumerate(window):
            for pid, team, mins, usg, started in by_game.get(gid, []):
                if team != tg.team_id:
                    continue
                a = acc[pid]
                a[0] += 1
                a[1] += mins
                if not np.isnan(usg):
                    a[2] += usg
                    a[4] += 1
                if started is not None:
                    a[3] += started
                    a[5] += 1
                a[6] = w_idx  # last window game he appeared in
        finished = (tg.game_id, tg.team_id) in played_for
        need = 3 if len(window) >= 5 else max(1, len(window) // 2)
        rotation = {}
        departed_min = departed_usage = 0.0
        for pid, (apps, mins, usg, starts, n_usg, n_known, last_idx) in acc.items():
            if apps < need or mins / apps < ROTATION_MIN_AVG:
                continue
            pd_dates, pd_teams, pd_seasons = player_dates.get(pid, ([], [], []))
            i = bisect_left(pd_dates, tg.game_date)
            gone = bool(i) and pd_teams[i - 1] != tg.team_id  # has played elsewhere since
            if not gone and finished and pid not in played_for[(tg.game_id, tg.team_id)]:
                # Next game elsewhere *this season* (a trade). An offseason move is not a
                # departure: a player injured for the rest of the season is absent, not gone.
                gone = (
                    i < len(pd_teams) and pd_teams[i] != tg.team_id and pd_seasons[i] == tg.season
                )
            if not gone and not finished and current_team:
                gone = current_team.get(pid, tg.team_id) != tg.team_id  # today's roster
            role_min = mins / apps
            role_usage = usg / n_usg if n_usg else 0.0
            if gone:
                since = len(window) - 1 - last_idx  # team games since his last one here
                weight = max(0.0, (DEPARTED_FADE_GAMES - since) / DEPARTED_FADE_GAMES)
                departed_min += weight * role_min
                departed_usage += weight * role_usage
                continue
            rotation[pid] = (
                role_min,
                role_usage,
                starts / n_known if n_known else 0.0,
            )
        if finished:
            absent = {p for p in rotation if p not in played_for[(tg.game_id, tg.team_id)]}
        else:
            absent = {p for p in rotation if p in status_map.get((tg.game_id, tg.team_id), set())}
        by_pos: dict[str, float] = defaultdict(float)
        for pid in absent:
            by_pos[pos.get(pid, "U")] += rotation[pid][0]
        summary.append(
            {
                "game_id": tg.game_id,
                "team_id": tg.team_id,
                "absent_n": len(absent),
                "absent_min_sum": sum(rotation[p][0] for p in absent),
                "absent_usage_sum": sum(rotation[p][1] for p in absent),
                "absent_starters": sum(1 for p in absent if rotation[p][2] >= 0.5),
                "departed_min_sum": departed_min,
                "departed_usage_sum": departed_usage,
                **{f"absent_min_{k}": by_pos.get(k, 0.0) for k in "GFCU"},
            }
        )
        for pid, (avg_min, _, _) in rotation.items():
            rotation_rows.append(
                {
                    "game_id": tg.game_id,
                    "team_id": tg.team_id,
                    "player_id": pid,
                    "rotation_min": avg_min,
                    "is_absent": pid in absent,
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(rotation_rows)


# --- schedule and standings --------------------------------------------------


def schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, team_col, opp_col in (
        ("home", "home_team_id", "away_team_id"),
        ("away", "away_team_id", "home_team_id"),
    ):
        part = games[["game_id", "game_date", team_col, opp_col]].rename(
            columns={team_col: "team_id", opp_col: "opp_team_id"}
        )
        part["is_home"] = side == "home"
        rows.append(part)
    tg = pd.concat(rows).sort_values(["team_id", "game_date"]).reset_index(drop=True)
    g = tg.groupby("team_id", sort=False)
    tg["rest_days"] = (tg["game_date"] - g["game_date"].shift(1)).dt.days.clip(upper=7)
    tg["rest_days"] = tg["rest_days"].fillna(7)
    tg["b2b"] = (tg["rest_days"] == 1).astype(int)
    counts = []
    for _, grp in tg.groupby("team_id", sort=False):
        ds = grp["game_date"].to_numpy()
        lo = np.searchsorted(ds, ds - np.timedelta64(7, "D"), side="left")
        counts.append(pd.Series(np.arange(len(ds)) - lo, index=grp.index))
    tg["games_last_7"] = pd.concat(counts)
    away = (~tg["is_home"]).astype(int)
    run_id = (away.groupby(tg["team_id"]).diff().fillna(1) != 0).cumsum()
    tg["road_trip_len"] = away.groupby(run_id).cumsum() * away
    opp = tg[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "opp_team_id", "rest_days": "opp_rest_days"}
    )
    return tg.merge(opp, on=["game_id", "opp_team_id"], how="left")


def standings(games: pd.DataFrame, teams: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Record and conference position as of each target date (regular-season games before it)."""
    conf = {r.team_id: ("E" if r.abbreviation in EAST else "W") for r in teams.itertuples()}
    fin = games[(games["status"] == "final") & (games["season_type"] == "regular")]
    res = pd.concat(
        [
            pd.DataFrame(
                {
                    "team_id": fin["home_team_id"],
                    "season": fin["season"],
                    "game_date": fin["game_date"],
                    "win": (fin["home_score"] > fin["away_score"]).astype(int),
                }
            ),
            pd.DataFrame(
                {
                    "team_id": fin["away_team_id"],
                    "season": fin["season"],
                    "game_date": fin["game_date"],
                    "win": (fin["away_score"] > fin["home_score"]).astype(int),
                }
            ),
        ]
    ).sort_values(["team_id", "game_date"])
    res["w"] = res.groupby(["team_id", "season"])["win"].cumsum()
    res["n"] = res.groupby(["team_id", "season"]).cumcount() + 1
    dates = targets[["season", "game_date"]].drop_duplicates()
    if not len(dates):
        return pd.DataFrame()
    grid = dates.merge(pd.DataFrame({"team_id": list(conf)}), how="cross")
    rec = _asof(
        grid, res[["team_id", "season", "game_date", "w", "n"]], by=["team_id", "season"]
    ).fillna({"w": 0, "n": 0})
    rec["l"] = rec["n"] - rec["w"]
    rec["win_pct"] = np.where(rec["n"] > 0, rec["w"] / rec["n"].replace(0, 1), 0.5)
    rec["conf"] = rec["team_id"].map(conf)
    rec = rec.sort_values(
        ["season", "game_date", "conf", "win_pct", "team_id"],
        ascending=[True, True, True, False, True],
    )
    keys = ["season", "game_date", "conf"]
    rec["conf_rank"] = rec.groupby(keys).cumcount() + 1
    for seed in (6, 10):
        ref = rec[rec["conf_rank"] == seed][[*keys, "w", "l"]].rename(
            columns={"w": "ref_w", "l": "ref_l"}
        )
        rec = rec.merge(ref, on=keys, how="left")
        rec[f"games_back_{seed}"] = (
            ((rec["ref_w"] - rec["w"]) + (rec["l"] - rec["ref_l"])) / 2
        ).fillna(0.0)
        rec = rec.drop(columns=["ref_w", "ref_l"])
    rec["games_remaining"] = REGULAR_SEASON_GAMES - rec["n"]
    # Seeding signals for late-season rest and tanking (ties ignored, so both are conservative):
    # clinched: more wins than the 7th seed can still reach; eliminated: can't reach the 10th.
    for seed, col in ((7, "s7"), (10, "s10")):
        ref = rec[rec["conf_rank"] == seed][[*keys, "w", "games_remaining"]].rename(
            columns={"w": f"{col}_w", "games_remaining": f"{col}_left"}
        )
        rec = rec.merge(ref, on=keys, how="left")
    rec["clinched_top6"] = (
        (rec["conf_rank"] <= 6) & (rec["w"] > rec["s7_w"] + rec["s7_left"])
    ).astype(int)
    rec["eliminated"] = (
        (rec["conf_rank"] > 10) & (rec["w"] + rec["games_remaining"] < rec["s10_w"])
    ).astype(int)
    return rec[
        [
            "team_id",
            "season",
            "game_date",
            "win_pct",
            "conf_rank",
            "games_back_6",
            "games_back_10",
            "games_remaining",
            "clinched_top6",
            "eliminated",
        ]
    ]


# --- market lines ------------------------------------------------------------


def market_lines(game_lines: pd.DataFrame | None, as_of: datetime | None) -> pd.DataFrame:
    """Latest spread and total per game at or before `as_of` (Pinnacle first)."""
    cols = ["game_id", "market_spread_home", "market_total"]
    if game_lines is None or not len(game_lines):
        return pd.DataFrame(columns=cols)
    gl = game_lines
    if as_of is not None:
        gl = gl[pd.to_datetime(gl["snapshot_ts"], utc=True) <= pd.Timestamp(as_of)]
    rank = {"pinnacle": 0, "draftkings": 1}
    gl = gl.assign(book_rank=gl["book"].map(rank).fillna(2))
    gl = gl.sort_values(
        ["game_id", "market", "book_rank", "snapshot_ts"], ascending=[True, True, True, False]
    )
    best = gl.drop_duplicates(["game_id", "market"])
    piv = best.pivot(index="game_id", columns="market", values="point")
    return (
        pd.DataFrame(
            {
                "game_id": piv.index,
                "market_spread_home": piv.get("spreads"),
                "market_total": piv.get("totals"),
            }
        )
        .reset_index(drop=True)
        .astype({"market_spread_home": float, "market_total": float})
    )


# --- assembly ----------------------------------------------------------------


def _prepare(inp: FeatureInputs) -> FeatureInputs:
    for df in (inp.logs, inp.team_logs, inp.games):
        df["game_date"] = _dt(df["game_date"])
    inj = inp.injuries
    if inj is not None and len(inj):
        # Report rows whose name or team didn't match (they're listed by `pipeline.run aliases`)
        # can't be placed on a roster: drop them here so no step meets an empty id.
        unmatched = inj["player_id"].isna() | inj["team_id"].isna()
        if unmatched.any():
            log.warning(
                "injury report: ignoring %d rows whose player or team didn't match "
                "(match them with `pipeline.run aliases` / `alias-add`)",
                int(unmatched.sum()),
            )
            inp.injuries = inj[~unmatched].copy()
    players = inp.players.copy()
    players["pos"] = players["position"].map(primary_position)
    inp.players = players
    return inp


def assemble(
    inp: FeatureInputs,
    targets: pd.DataFrame,
    statuses: pd.DataFrame | None = None,
    as_of: datetime | None = None,
    current_team: dict[int, int] | None = None,
) -> pd.DataFrame:
    """targets: game_id, player_id, team_id, opp_team_id, is_home, game_date, season,
    season_type (+ optional y_* columns). `current_team` (today's rosters) is for live slates
    only: it describes current rosters, so it must never be used for past games."""
    t = targets.copy()
    t["game_date"] = _dt(t["game_date"])
    players = inp.players
    t = t.merge(players[["player_id", "pos"]], on="player_id", how="left")
    t["pos"] = t["pos"].fillna("U")

    # player form and season to date
    t = _asof(t, player_state(inp.logs, inp.team_logs, inp.games), by="player_id")
    same = t["state_season"] == t["season"]
    for c in ["std_games", "std_minutes", *[f"std_{s}" for s in STATS]]:
        t[c] = np.where(same, t[c], 0.0)
    t["games_with_team"] = np.where(
        same & (t["state_team_id"] == t["team_id"]), t["games_with_team"], 0
    )
    t["std_min_avg"] = t["std_minutes"] / t["std_games"].replace(0, np.nan)
    for s_ in ("pts", "reb", "ast", "fg3m"):  # season-to-date per-game averages
        t[f"std_{s_}_avg"] = t[f"std_{s_}"] / t["std_games"].replace(0, np.nan)
    t["player_rest_days"] = (t["game_date"] - t["last_game_date"]).dt.days.clip(upper=10)
    t["career_games"] = t["career_games"].fillna(0)
    t["games_missed"] = games_missed(inp.games, t)

    # priors and shrinkage
    totals, league = season_priors(inp.logs, players)
    t["prev_season"] = t["season"].map(prev_season)
    prev = totals.rename(
        columns={
            "season": "prev_season",
            "minutes": "prev_min",
            "games": "prev_games",
            **{s: f"prev_{s}" for s in STATS},
        }
    )
    t = t.merge(prev, on=["player_id", "prev_season"], how="left")
    t = t.merge(
        league.rename(columns={"season": "prev_season"}), on=["prev_season", "pos"], how="left"
    )
    own = t["prev_min"].fillna(0) >= PRIOR_MIN_SEASON_MINUTES
    has_pos = t["pos_pts_pm"].notna()
    t["prior_source"] = np.where(own, 2, np.where(has_pos, 1, 0))
    for s in STATS:
        prior = np.where(own, t[f"prev_{s}"] / t["prev_min"], t[f"pos_{s}_pm"])
        std_s, std_m = t[f"std_{s}"], t["std_minutes"]
        t[f"{s}_pm_shrunk"] = np.where(
            pd.notna(prior),
            (std_s + PRIOR_MINUTES * np.nan_to_num(prior)) / (std_m + PRIOR_MINUTES),
            std_s / std_m.replace(0, np.nan),
        )
    t["prev_season_min_avg"] = t["prev_min"] / t["prev_games"]
    t["rookie"] = (t["career_games"] == 0).astype(int) & t["prev_min"].isna().astype(int)

    # preseason minutes this season (preseason always precedes the regular-season targets)
    pre = inp.logs[(inp.logs["season_type"] == "preseason") & (inp.logs["minutes"].fillna(0) > 0)]
    pre = (
        pre.groupby(["player_id", "season"], as_index=False)["minutes"]
        .mean()
        .rename(columns={"minutes": "preseason_min_avg"})
    )
    t = t.merge(pre, on=["player_id", "season"], how="left")

    # --- since joining the current team (stints reset on any team change) ---
    continuing = (t["state_team_id"] == t["team_id"]).to_numpy()
    has_history = (t["career_games"] > 0).to_numpy()
    t["stint_games"] = np.where(continuing, t["stint_games"], 0)
    t["stint_games"] = np.minimum(t["stint_games"], STINT_GAMES_CAP)
    for c in (
        "stint_min_avg_l10",
        "stint_usage_l10",
        *[f"stint_{s_}_pm_l10" for s_ in ("pts", "reb", "ast", "fg3m")],
    ):
        t[c] = np.where(continuing, t[c], np.nan)
    # Debut with a new team today: the stint starts now, with a change if he played elsewhere.
    t["stint_change"] = np.where(continuing, t["stint_change"], has_history.astype(int))
    # Role brought from the previous team; for a debut today, his latest role (old team).
    t["prior_role_min"] = np.where(continuing, t["prior_role_min"], t["min_avg_l10"])
    t["prior_role_usage"] = np.where(continuing, t["prior_role_usage"], t["usage_l10"])
    try:
        n_low = load_strategy(get_settings().strategy_config).new_team_low_conf_games
    except FileNotFoundError:
        n_low = 10
    t["new_team"] = ((t["stint_change"] == 1) & (t["stint_games"] < n_low)).astype(int)

    t = t.copy()  # de-fragment
    # --- teammates and opponents: absences, departures (all team-games, for history too) ---
    tg_cols = ["game_id", "team_id", "game_date", "season"]
    own_tg = t[tg_cols]
    opp_tg = t[["game_id", "opp_team_id", "game_date", "season"]].rename(
        columns={"opp_team_id": "team_id"}
    )
    tl = inp.team_logs
    hist_tg = tl.loc[tl["season_type"] != "preseason", tg_cols]
    all_tg = pd.concat([own_tg, opp_tg, hist_tg]).drop_duplicates(["game_id", "team_id"])
    absences, rotation = team_absences(
        inp.logs, inp.team_logs, players, all_tg, statuses, current_team
    )
    abs_cols = [
        "absent_n",
        "absent_min_sum",
        "absent_usage_sum",
        "absent_starters",
        *[f"absent_min_{k}" for k in "GFCU"],
    ]
    t = t.merge(
        absences[["game_id", "team_id", *abs_cols, "departed_min_sum", "departed_usage_sum"]],
        on=["game_id", "team_id"],
        how="left",
    )
    opp_abs = absences[["game_id", "team_id", *abs_cols]].rename(
        columns={"team_id": "opp_team_id", **{c: f"opp_{c}" for c in abs_cols}}
    )
    t = t.merge(opp_abs, on=["game_id", "opp_team_id"], how="left")
    zero_cols = [
        *abs_cols,
        *[f"opp_{c}" for c in abs_cols],
        "departed_min_sum",
        "departed_usage_sum",
    ]
    t[zero_cols] = t[zero_cols].fillna(0.0)
    pos_arr = t["pos"].to_numpy()
    for prefix, out_col in (
        ("absent_min_", "same_pos_absent_min"),
        ("opp_absent_min_", "opp_same_pos_absent_min"),
    ):
        cols = {k: t[f"{prefix}{k}"].to_numpy() for k in "GFCU"}
        t[out_col] = np.select([pos_arr == k for k in "GFCU"], [cols[k] for k in "GFCU"], 0.0)

    t = t.copy()
    # --- arrivals and returns, summed over teammates in the same game (excluding self) ---
    w_arr = np.where(
        t["stint_change"].to_numpy() == 1,
        np.clip((ARRIVED_FADE_GAMES - t["stint_games"].to_numpy()) / ARRIVED_FADE_GAMES, 0, 1),
        0.0,
    )
    returning_today = continuing & (t["games_missed"].to_numpy() >= RETURN_MIN_MISSED)
    since = _team_games_between(inp.games, t["team_id"], t["last_return_date"], t["game_date"])
    fade = np.where(
        (since >= 1) & (since < RETURN_FADE_GAMES),
        (RETURN_FADE_GAMES - since) / RETURN_FADE_GAMES,
        0.0,
    )
    w_ret_recent = np.where(continuing & ~returning_today, fade, 0.0)
    contrib = {
        "arrived_min_sum": w_arr * t["prior_role_min"].fillna(0).to_numpy(),
        "arrived_usage_sum": w_arr * t["prior_role_usage"].fillna(0).to_numpy(),
        "returning_min_sum": np.where(returning_today, t["min_avg_l10"].fillna(0), 0.0)
        + w_ret_recent * t["return_role_min"].fillna(0).to_numpy(),
        "returning_usage_sum": np.where(returning_today, t["usage_l10"].fillna(0), 0.0)
        + w_ret_recent * t["return_role_usage"].fillna(0).to_numpy(),
    }
    team_key = [t["game_id"], t["team_id"]]
    for col, c in contrib.items():
        c = pd.Series(c, index=t.index)
        t[col] = c.groupby(team_key).transform("sum") - c

    t = t.copy()
    # --- with vs without today's most important absent teammate ---
    wo = with_without(inp.logs, rotation, t)
    t = t.merge(wo, on=["game_id", "player_id"], how="left")
    wo_cols = [
        "wo_games",
        "wo_min_delta",
        "wo_usage_delta",
        *[f"wo_{s_}_pm_delta" for s_ in ("pts", "reb", "ast", "fg3m")],
    ]
    t[wo_cols] = t[wo_cols].astype(float).fillna(0.0)

    # opponent and own team
    ts = team_state(inp.team_logs)
    opp = ts.rename(
        columns={
            "team_id": "opp_team_id",
            "def_rtg_l15": "opp_def_rtg_l15",
            "def_rtg_std": "opp_def_rtg_std_raw",
            "pace_l15": "opp_pace_l15",
            "dreb_pct_l15": "opp_dreb_pct_l15",
            "state_season": "opp_state_season",
            "off_rtg_l20": "opp_off_rtg_l20",
            "def_rtg_l20": "opp_def_rtg_l20",
            "pace_l20": "opp_pace_l20",
            "net_l20": "opp_net_l20",
        }
    )
    t = _asof(t, opp, by="opp_team_id")
    t["opp_def_rtg_std"] = np.where(
        t["opp_state_season"] == t["season"], t["opp_def_rtg_std_raw"], np.nan
    )
    own_t = ts[["team_id", "game_date", "off_rtg_l20", "def_rtg_l20", "pace_l20", "net_l20"]]
    own_t = own_t.rename(
        columns={
            "off_rtg_l20": "team_off_rtg_l20",
            "def_rtg_l20": "team_def_rtg_l20",
            "pace_l20": "team_pace_l20",
            "net_l20": "team_net_l20",
        }
    )
    t = _asof(t, own_t, by="team_id")
    t["def_key"] = t["opp_team_id"].astype("int64").astype(str) + ":" + t["pos"]
    t = _asof(t, defense_by_position_state(inp.logs, inp.team_logs, players), by="def_key")

    t = t.copy()  # de-fragment after many column additions
    # rating-based margin and total, from the player's team's perspective
    pace = (t["team_pace_l20"] + t["opp_pace_l20"]) / 2
    team_pts = pace / 100 * (t["team_off_rtg_l20"] + t["opp_def_rtg_l20"]) / 2
    opp_pts = pace / 100 * (t["opp_off_rtg_l20"] + t["team_def_rtg_l20"]) / 2
    hca = np.where(t["is_home"].astype(bool), HOME_COURT_PTS, -HOME_COURT_PTS)
    t["rating_team_margin"] = team_pts - opp_pts + hca
    t["rating_total"] = team_pts + opp_pts
    t["rating_team_pts"] = t["rating_total"] / 2 + t["rating_team_margin"] / 2

    # market lines (not v1 features; see module docstring)
    ml = market_lines(inp.game_lines, as_of)
    t = t.merge(ml, on="game_id", how="left")
    home = t["is_home"].astype(bool)
    t["market_team_margin"] = np.where(home, -t["market_spread_home"], t["market_spread_home"])

    t = t.copy()
    # schedule
    sched = schedule_features(inp.games)[
        ["game_id", "team_id", "rest_days", "b2b", "games_last_7", "road_trip_len", "opp_rest_days"]
    ]
    t = t.merge(sched, on=["game_id", "team_id"], how="left")

    # standings
    st = standings(inp.games, inp.teams, t)
    if len(st):
        t = t.merge(st, on=["team_id", "season", "game_date"], how="left")
    else:
        for c in (
            "win_pct",
            "conf_rank",
            "games_back_6",
            "games_back_10",
            "games_remaining",
            "clinched_top6",
            "eliminated",
        ):
            t[c] = np.nan
    t["late_season"] = (t["games_remaining"] <= LATE_SEASON_GAMES_LEFT).astype(int)
    # Playoffs and play-in: rotations tighten and stars play more. The regular-season rest and
    # tanking signals mean the opposite here, so they are switched off.
    t["is_postseason"] = t["season_type"].isin(["playoffs", "playin"]).astype(int)
    post = t["is_postseason"] == 1
    t.loc[post, ["late_season", "clinched_top6", "eliminated"]] = 0
    t.loc[post, "games_remaining"] = np.nan
    t["is_home"] = t["is_home"].astype(int)

    # Safety net: every model input must be numeric, whatever the input frames looked like.
    for c in FEATURE_COLUMNS:
        if t[c].dtype == object:
            t[c] = pd.to_numeric(t[c], errors="raise")
    keep = [
        "game_id",
        "player_id",
        "team_id",
        "opp_team_id",
        "game_date",
        "season",
        "season_type",
        "pos",
        "injury_status",
        "market_team_margin",
        "market_total",
    ]
    keep = [c for c in keep if c in t.columns]
    ys = [c for c in TARGET_COLUMNS if c in t.columns]
    return (
        t[keep + FEATURE_COLUMNS + ys]
        .sort_values(["game_date", "game_id", "player_id"])
        .reset_index(drop=True)
    )


# --- entry points ------------------------------------------------------------


def training_frame(inp: FeatureInputs, seasons: list[str] | None = None) -> pd.DataFrame:
    """One row per player who played in a finished non-preseason game, with targets."""
    inp = _prepare(inp)
    p = _played(inp.logs)
    if seasons:
        p = p[p["season"].isin(seasons)]
    p = p.merge(
        inp.team_logs[["game_id", "team_id", "opp_team_id", "is_home"]],
        on=["game_id", "team_id"],
        how="left",
    )
    targets = p[
        [
            "game_id",
            "player_id",
            "team_id",
            "opp_team_id",
            "is_home",
            "game_date",
            "season",
            "season_type",
        ]
    ].copy()
    targets["y_minutes"] = p["minutes"].astype(float)
    for s in ("pts", "reb", "ast", "fg3m"):
        targets[f"y_{s}"] = p[s].astype(float)
    targets["y_pra"] = targets["y_pts"] + targets["y_reb"] + targets["y_ast"]
    scores = inp.games[["game_id", "home_score", "away_score"]]
    targets = targets.merge(scores, on="game_id", how="left")
    home_margin = (targets["home_score"] - targets["away_score"]).astype(float)
    targets["y_team_margin"] = np.where(targets["is_home"].astype(bool), home_margin, -home_margin)
    targets["y_game_total"] = (targets["home_score"] + targets["away_score"]).astype(float)
    targets = targets.drop(columns=["home_score", "away_score"])
    return assemble(inp, targets)


def latest_statuses(injuries: pd.DataFrame | None, as_of: datetime) -> pd.DataFrame:
    """Statuses from the most recent report at or before `as_of` that covers each game.

    Each report is a full list: a player missing from the latest report is no longer
    injured, even if an earlier report had him Out. Taking each player's last row instead
    would leave him Out forever.
    """
    cols = ["game_id", "team_id", "player_id", "status"]
    if injuries is None or not len(injuries):
        return pd.DataFrame(columns=cols)
    inj = injuries[pd.to_datetime(injuries["report_ts"], utc=True) <= pd.Timestamp(as_of)]
    inj = inj.dropna(subset=["game_id"])
    if not len(inj):
        return pd.DataFrame(columns=cols)
    latest = inj.groupby("game_id")["report_ts"].transform("max")
    inj = inj[inj["report_ts"] == latest].dropna(subset=["player_id"])
    return inj.drop_duplicates(["game_id", "player_id"], keep="last")[cols]


def established_players(
    logs: pd.DataFrame, team_games: pd.DataFrame, current_team: dict[int, int] | None = None
) -> pd.DataFrame:
    """Players a slate must include even if they're outside the recent rotation: someone back
    from a long absence, a starter who sat out preseason, or a new arrival by trade.

    A player belongs to the team of his latest game, or to today's roster team when
    `current_team` (live only) says so. Established for that team means:
      - >=5 games averaging >=15 minutes for it this season, or
      - a new arrival (latest game elsewhere): >=5 games averaging >=12 minutes this season
        for any team, or
      - before he has played for it this season: >=20 games averaging >=15 minutes last season.
    """
    played = _played(logs, include_preseason=True).sort_values("game_date")
    rows = []
    for tg in team_games.itertuples(index=False):
        before = played[played["game_date"] < tg.game_date]
        latest = before.drop_duplicates("player_id", keep="last")
        last_team = dict(zip(latest["player_id"], latest["team_id"], strict=True))
        team_now = {p: (current_team or {}).get(p, t) for p, t in last_team.items()}
        mine = [p for p, t in team_now.items() if t == tg.team_id]
        arrivals = {p for p in mine if last_team[p] != tg.team_id}
        reg = before[(before["season_type"] != "preseason") & before["player_id"].isin(mine)]
        season = reg[reg["season"] == tg.season]
        cur = (
            season[season["team_id"] == tg.team_id]
            .groupby("player_id")["minutes"]
            .agg(["count", "mean"])
        )
        ok = set(cur[(cur["count"] >= 5) & (cur["mean"] >= 15)].index)
        anywhere = (
            season[season["player_id"].isin(arrivals)]
            .groupby("player_id")["minutes"]
            .agg(["count", "mean"])
        )
        ok |= set(anywhere[(anywhere["count"] >= 5) & (anywhere["mean"] >= 12)].index)
        last = (
            reg[reg["season"] == prev_season(tg.season)]
            .groupby("player_id")["minutes"]
            .agg(["count", "mean"])
        )
        ok |= {
            p
            for p in last[(last["count"] >= 20) & (last["mean"] >= 15)].index
            if p not in cur.index
        }
        rows += [{"game_id": tg.game_id, "team_id": tg.team_id, "player_id": p} for p in ok]
    return pd.DataFrame(rows, columns=["game_id", "team_id", "player_id"])


def roster_map(players: pd.DataFrame) -> dict[int, int]:
    """Today's roster {player_id: team_id} from players.current_team_id (live slates only)."""
    if "current_team_id" not in players:
        return {}
    known = players.dropna(subset=["current_team_id"])
    return dict(
        zip(known["player_id"].astype(int), known["current_team_id"].astype(int), strict=True)
    )


def slate_frame(
    inp: FeatureInputs,
    game_date: date,
    as_of: datetime,
    extra_players: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Live rows for games on `game_date`: the recent rotation, established players (including
    new arrivals by trade, placed by today's roster), anyone the injury report lists as able to
    play, and `extra_players` [(player_id, team_id)] such as players with a DraftKings line.
    Players listed Out or Doubtful are excluded; players traded away are departures, not
    candidates."""
    inp = _prepare(inp)
    day = inp.games[inp.games["game_date"] == pd.Timestamp(game_date)]
    team_games = pd.concat(
        [
            day.assign(team_id=day["home_team_id"], opp_team_id=day["away_team_id"], is_home=True),
            day.assign(team_id=day["away_team_id"], opp_team_id=day["home_team_id"], is_home=False),
        ]
    )[["game_id", "team_id", "opp_team_id", "is_home", "game_date", "season", "season_type"]]
    if not len(team_games):
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    statuses = latest_statuses(inp.injuries, as_of)
    roster = roster_map(inp.players)  # today's rosters: places traded players on new teams
    _, rotation = team_absences(inp.logs, inp.team_logs, inp.players, team_games, statuses, roster)
    keys = ["game_id", "team_id", "player_id"]
    parts = [
        rotation[keys] if len(rotation) else pd.DataFrame(columns=keys),
        established_players(inp.logs, team_games, roster),
        statuses[~statuses["status"].isin(ABSENT_STATUSES)][keys],
    ]
    if extra_players:
        extra = pd.DataFrame(extra_players, columns=["player_id", "team_id"]).merge(
            team_games[["game_id", "team_id"]], on="team_id"
        )
        parts.append(extra[keys])
    cand = pd.concat(parts).drop_duplicates(keys)
    # Empty parts make concat fall back to object columns; joins need real integer ids.
    cand = cand.astype({"game_id": str, "team_id": "int64", "player_id": "int64"})
    out = statuses[statuses["status"].isin(ABSENT_STATUSES)][keys].assign(_out=True)
    cand = cand.merge(out, on=keys, how="left")
    cand = cand[cand["_out"].isna()].drop(columns="_out")
    targets = cand.merge(team_games, on=["game_id", "team_id"])
    targets = targets.merge(
        statuses[["game_id", "player_id", "status"]].rename(columns={"status": "injury_status"}),
        on=["game_id", "player_id"],
        how="left",
    )
    targets["player_id"] = targets["player_id"].astype("int64")
    return assemble(inp, targets, statuses, as_of, roster)


# --- loading from Postgres ---------------------------------------------------


def load_inputs(session, seasons: list[str] | None = None) -> FeatureInputs:
    conn = session.connection()
    where = "WHERE g.season = ANY(:seasons)" if seasons else ""
    params = {"seasons": seasons} if seasons else {}

    def q(sql: str, p: dict | None = None) -> pd.DataFrame:
        df = pd.read_sql_query(text(sql), conn, params=p if p is not None else params)
        for c in df.columns:  # Postgres NUMERIC arrives as Decimal objects
            non_null = df[c].dropna()
            if df[c].dtype == object and len(non_null) and isinstance(non_null.iloc[0], Decimal):
                df[c] = df[c].astype(float)
        return df

    logs = q(f"""SELECT l.*, g.game_date, g.season, g.season_type FROM player_game_logs l
                 JOIN games g USING (game_id) {where}""")
    team_logs = q(f"""SELECT t.*, g.game_date, g.season, g.season_type FROM team_game_logs t
                      JOIN games g USING (game_id) {where}""")
    games = q(f"SELECT * FROM games g {where}")
    players = q("SELECT player_id, position, current_team_id FROM players", {})
    teams = q("SELECT team_id, abbreviation FROM teams", {})
    injuries = q("SELECT report_ts, game_id, team_id, player_id, status FROM injury_snapshots", {})
    lines = q("SELECT snapshot_ts, game_id, book, market, point FROM game_line_snapshots", {})
    return FeatureInputs(logs, team_logs, games, players, teams, injuries, lines)


def build_training_set(session, seasons: list[str]) -> pd.DataFrame:
    # Load one extra earlier season so the first requested season has priors.
    first = sorted(seasons)[0]
    return training_frame(load_inputs(session, [prev_season(first), *seasons]), seasons)


def build_features(
    session,
    game_date: date,
    as_of_ts: datetime,
    extra_players: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    from nbaprops.pipeline.stats import season_for_date

    season = season_for_date(game_date)
    return slate_frame(
        load_inputs(session, [prev_season(season), season]), game_date, as_of_ts, extra_players
    )


# --- Questionable players: two scenarios for teammates -------------------------


def questionable_scenarios(
    inp: FeatureInputs,
    game_date: date,
    as_of: datetime,
    extra_players: list[tuple[int, int]] | None = None,
    p_play: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Slate features as reported, plus an alternative where each team's most important
    Questionable player (most recent minutes) is Out.

    A Questionable player's own props void if he sits, so his own projection uses the
    as-reported frame (he plays). His teammates' props don't void, and their minutes and usage
    depend on whether he plays, so they are simulated under both and mixed by `p_play`.
    Returns (base, alternative or None, keys[game_id, team_id, player_id, p_play])."""
    if p_play is None:
        try:
            p_play = load_strategy(get_settings().strategy_config).p_play_questionable
        except FileNotFoundError:
            p_play = 0.5
    base = slate_frame(inp, game_date, as_of, extra_players)
    empty = pd.DataFrame(columns=["game_id", "team_id", "player_id", "p_play"])
    if not len(base) or "injury_status" not in base:
        return base, None, empty
    q = base[base["injury_status"] == "Questionable"]
    if not len(q):
        return base, None, empty
    q = q.assign(_importance=q["min_avg_l10"].fillna(q["std_min_avg"]).fillna(0))
    keys = (
        q.sort_values("_importance", ascending=False)
        .drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "player_id"]]
        .assign(p_play=p_play)
        .reset_index(drop=True)
    )
    # Alternative report: the latest report for each affected game, with the key players Out.
    statuses = latest_statuses(inp.injuries, as_of)
    alt_rows = statuses.copy()
    marks = set(zip(keys["game_id"], keys["player_id"], strict=True))
    hit = [(g, p) in marks for g, p in zip(alt_rows["game_id"], alt_rows["player_id"], strict=True)]
    alt_rows.loc[hit, "status"] = "Out"
    alt_rows = alt_rows[alt_rows["game_id"].isin(keys["game_id"])].assign(
        report_ts=pd.Timestamp(as_of)
    )
    alt_inp = replace(inp, injuries=pd.concat([inp.injuries, alt_rows], ignore_index=True))
    alt = slate_frame(alt_inp, game_date, as_of, extra_players)
    return base, alt, keys


def estimate_play_rates(
    injuries: pd.DataFrame, logs: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """How often players listed with each status actually played, using each player's last
    listing before tip. Feed the Questionable rate into `p_play_questionable`."""
    inj = injuries.dropna(subset=["game_id", "player_id"]).merge(
        games[["game_id", "tip_time"]], on="game_id", how="inner"
    )
    inj = inj[
        pd.to_datetime(inj["report_ts"], utc=True) < pd.to_datetime(inj["tip_time"], utc=True)
    ]
    last = inj.sort_values("report_ts").drop_duplicates(["game_id", "player_id"], keep="last")
    finished = set(games.loc[games["status"] == "final", "game_id"])
    last = last[last["game_id"].isin(finished)]
    played = logs[(logs["minutes"].fillna(0) > 0) & (~logs["dnp"].fillna(False).astype(bool))]
    played_keys = set(zip(played["game_id"], played["player_id"], strict=True))
    last = last.assign(
        played=[
            (g, p) in played_keys for g, p in zip(last["game_id"], last["player_id"], strict=True)
        ]
    )
    return last.groupby("status")["played"].agg(listings="count", play_rate="mean").reset_index()


def build_feature_scenarios(
    session, game_date: date, as_of_ts: datetime, extra_players: list[tuple[int, int]] | None = None
):
    """Database version of `questionable_scenarios` (loads inputs once)."""
    from nbaprops.pipeline.stats import season_for_date

    season = season_for_date(game_date)
    inp = load_inputs(session, [prev_season(season), season])
    return questionable_scenarios(inp, game_date, as_of_ts, extra_players)


# Named groups of inputs, for with/without comparisons in the Stage 6 backtest.
FEATURE_GROUPS = {
    # trades, roster changes, opponent absences, with/without splits (ninth review)
    "roster": [
        c
        for c in FEATURE_COLUMNS
        if c.startswith(("stint_", "departed_", "arrived_", "returning_", "wo_", "opp_absent_"))
        or c == "opp_same_pos_absent_min"
    ],
}
