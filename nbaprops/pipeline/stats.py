"""Stage 1: box scores from the NBA stats site (via nba_api), historical and daily.

Requests per season (about 1,300 total, roughly 30-40 minutes with polite pacing):
  LeagueGameLog (teams)   1 per season type  -> games + team box scores
  PlayerGameLogs          1 per season type  -> every player-game, fractional minutes
  PlayerIndex             1                  -> positions
  BoxScoreTraditionalV3   1 per game         -> starters + did-not-play reasons
Possessions, pace, ratings and usage are computed here (Basketball-Reference formulas)
instead of making a second per-game request.

Every response is cached under data/cache/nba/, so an interrupted backfill resumes where it
stopped. Preseason games are loaded (preseason minutes feed new-team priors) but models must
exclude season_type = 'preseason' from training targets.

The NBA stats site often blocks cloud IPs: run the backfill from your own machine.
"""

import gzip
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from nbaprops.db.base import session_scope
from nbaprops.db.models import Game, Player, PlayerGameLog, Team, TeamGameLog
from nbaprops.pipeline.reference import ensure_player
from nbaprops.pipeline.schedule import season_from_game_id

log = logging.getLogger(__name__)

SEASON_TYPES = {
    "Pre Season": "preseason",
    "Regular Season": "regular",
    "PlayIn": "playin",
    "Playoffs": "playoffs",
}
REGULATION_MINUTES = 240
OT_MINUTES = 25  # five players x five minutes

Rows = list[dict[str, Any]]
Fetch = Callable[..., dict[str, Rows]]


# --- Fetching (cache + retry + pacing) ---------------------------------------


def _endpoint(name: str):
    from nba_api.stats import endpoints

    return {
        "leaguegamelog": endpoints.LeagueGameLog,
        "playergamelogs": endpoints.PlayerGameLogs,
        "boxscoretraditionalv3": endpoints.BoxScoreTraditionalV3,
        "playerindex": endpoints.PlayerIndex,
    }[name]


def _normalized(endpoint_obj) -> dict:
    """{data set name: list of row dicts} for both response formats. nba_api's
    get_normalized_dict() only reads the older "resultSets" format and returns {} for the
    V3 endpoints (BoxScoreTraditionalV3), which its own per-endpoint parser handles."""
    data = endpoint_obj.get_normalized_dict()
    if data:
        return data
    sets = endpoint_obj.nba_response.get_data_sets(endpoint_obj.endpoint)
    return {
        name: [dict(zip(ds["headers"], row, strict=True)) for row in ds["data"]]
        for name, ds in sets.items()
    }


class StatsFetcher:
    """Calls nba_api with retries and pauses, caching each response as gzipped JSON."""

    def __init__(
        self,
        cache_dir: Path = Path("data/cache/nba"),
        pause: float = 1.0,
        retries: int = 4,
        timeout: int = 45,
        refresh: bool = False,
    ):
        """refresh=True re-downloads even when cached (picks up the NBA's stat corrections)."""
        self.refresh = refresh
        self.cache_dir = cache_dir
        self.pause = pause
        self.retries = retries
        self.timeout = timeout
        self.requests_made = 0

    def __call__(self, endpoint: str, cache_key: str, cache: bool = True, **params) -> dict:
        path = self.cache_dir / endpoint / f"{cache_key}.json.gz"
        if cache and not self.refresh and path.exists():
            with gzip.open(path, "rt") as f:
                cached = json.load(f)
            if cached:  # an empty cached response is a failed fetch: get it again
                return cached
        cls = _endpoint(endpoint)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                data = _normalized(cls(**params, timeout=self.timeout))
                if not data:
                    raise ValueError("empty response")
                break
            except Exception as exc:  # timeouts, connection resets, bad JSON from the site
                last_error = exc
                wait = self.pause * (2 ** (attempt + 1)) + random.uniform(0, 1)
                log.warning("%s %s failed (%s); retrying in %.0fs", endpoint, cache_key, exc, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"{endpoint} {cache_key} failed after {self.retries} tries") from (
                last_error
            )
        self.requests_made += 1
        time.sleep(self.pause + random.uniform(0, self.pause / 2))
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as f:
                json.dump(data, f)
        return data


# --- Small parsers -----------------------------------------------------------


class BadMinutes(ValueError):
    """A minutes value the site sent garbled (seen: "12:0-57"). Not the same as zero."""


def parse_minutes(value: Any) -> float | None:
    """Accepts 34.57, "34:12", "PT34M12.00S", "" / None (did not play). Raises BadMinutes on
    anything else: callers decide, because "unreadable" must never be read as "didn't play"."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    try:
        if s.startswith("PT"):
            mins, _, secs = s[2:].rstrip("S").partition("M")
            return float(mins or 0) + float(secs or 0) / 60
        if ":" in s:
            mins, secs = s.split(":", 1)
            return int(mins) + int(float(secs)) / 60
        return float(s)
    except ValueError as exc:
        raise BadMinutes(f"unreadable minutes {value!r}") from exc


def minutes_or_unknown(value: Any, where: str) -> float | None:
    """Game logs: an unreadable value becomes unknown (empty), with a warning."""
    try:
        return parse_minutes(value)
    except BadMinutes as exc:
        log.warning("%s: %s; recording minutes as unknown", where, exc)
        return None


def parse_game_date(value: str) -> date:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(value[:19] if "T" in value else value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized game date {value!r}")


def season_for_date(d: date) -> str:
    """NBA season containing this date: September onward belongs to the new season."""
    start = d.year if d.month >= 9 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def _int(v: Any) -> int | None:
    return None if v is None or v == "" else int(v)


# --- Transforms (pure) -------------------------------------------------------


def team_logs_to_rows(team_rows: Rows, season_type: str) -> tuple[Rows, Rows, list[str]]:
    """LeagueGameLog (teams) -> (games, team_game_logs, issues)."""
    by_game: dict[str, Rows] = defaultdict(list)
    for r in team_rows:
        by_game[r["GAME_ID"]].append(r)

    games, logs, issues = [], [], []
    for game_id, pair in by_game.items():
        if len(pair) != 2:
            issues.append(f"{game_id}: {len(pair)} team rows (expected 2)")
            continue
        home = next((r for r in pair if " vs. " in r["MATCHUP"]), None)
        away = next((r for r in pair if " @ " in r["MATCHUP"]), None)
        if home is None or away is None:
            issues.append(f"{game_id}: cannot tell home from away ({pair[0]['MATCHUP']})")
            continue
        minutes = float(home["MIN"] or REGULATION_MINUTES)
        games.append(
            {
                "game_id": game_id,
                "season": season_from_game_id(game_id),
                "season_type": season_type,
                "game_date": parse_game_date(home["GAME_DATE"]),
                "home_team_id": home["TEAM_ID"],
                "away_team_id": away["TEAM_ID"],
                "home_score": _int(home["PTS"]),
                "away_score": _int(away["PTS"]),
                "overtimes": max(0, round((minutes - REGULATION_MINUTES) / OT_MINUTES)),
                "status": "final",
            }
        )
        for me, opp, is_home in ((home, away, True), (away, home, False)):
            logs.append(
                {
                    "game_id": game_id,
                    "team_id": me["TEAM_ID"],
                    "opp_team_id": opp["TEAM_ID"],
                    "is_home": is_home,
                    "pts": _int(me["PTS"]),
                    "minutes": float(me["MIN"]) if me["MIN"] not in (None, "") else None,
                    **{
                        col.lower(): _int(me[col])
                        for col in (
                            "FGM",
                            "FGA",
                            "FG3M",
                            "FG3A",
                            "FTM",
                            "FTA",
                            "OREB",
                            "DREB",
                            "REB",
                            "AST",
                            "STL",
                            "BLK",
                            "TOV",
                            "PF",
                        )
                    },
                }
            )
    return games, logs, issues


def player_logs_to_rows(player_rows: Rows) -> tuple[Rows, dict[int, str]]:
    """PlayerGameLogs -> (player_game_logs rows, {player_id: name} seen)."""
    rows, names = [], {}
    for r in player_rows:
        names[r["PLAYER_ID"]] = r["PLAYER_NAME"]
        rows.append(
            {
                "game_id": r["GAME_ID"],
                "player_id": r["PLAYER_ID"],
                "team_id": r["TEAM_ID"],
                "dnp": False,
                "started": None,  # unknown until the box score is loaded
                "minutes": minutes_or_unknown(r["MIN"], f"game {r['GAME_ID']}"),
                "pts": _int(r["PTS"]),
                "reb": _int(r["REB"]),
                "oreb": _int(r["OREB"]),
                "dreb": _int(r["DREB"]),
                "ast": _int(r["AST"]),
                "stl": _int(r["STL"]),
                "blk": _int(r["BLK"]),
                "tov": _int(r["TOV"]),
                "pf": _int(r["PF"]),
                "fgm": _int(r["FGM"]),
                "fga": _int(r["FGA"]),
                "fg3m": _int(r["FG3M"]),
                "fg3a": _int(r["FG3A"]),
                "ftm": _int(r["FTM"]),
                "fta": _int(r["FTA"]),
                "plus_minus": _int(r["PLUS_MINUS"]),
            }
        )
    return rows, names


def boxscore_to_updates(box_rows: Rows) -> tuple[dict[int, bool], Rows, dict[int, str]]:
    """BoxScoreTraditionalV3 PlayerStats -> (started by player, did-not-play rows, names).

    Starters carry a position ("F", "C", "G"); bench players have an empty one. Players
    who didn't play have no minutes and a comment such as "DNP - Coach's Decision".
    """
    started, dnp_rows, names = {}, [], {}
    for r in box_rows:
        pid = r["personId"]
        names[pid] = f"{r.get('firstName', '')} {r.get('familyName', '')}".strip()
        try:
            mins = parse_minutes(r.get("minutes"))
        except BadMinutes as exc:
            # Something is there, so he played; his minutes come from the game logs anyway.
            log.warning("game %s, player %s: %s; treating as played", r["gameId"], pid, exc)
            started[pid] = bool((r.get("position") or "").strip())
            continue
        if not mins:
            dnp_rows.append(
                {
                    "game_id": r["gameId"],
                    "player_id": pid,
                    "team_id": r["teamId"],
                    "dnp": True,
                    "dnp_reason": (r.get("comment") or "").strip() or None,
                    "started": False,
                    "minutes": 0,
                }
            )
        else:
            started[pid] = bool((r.get("position") or "").strip())
    return started, dnp_rows, names


# --- Advanced stats: formulas (pure) and the SQL that applies them ------------


def possessions(t: dict, o: dict) -> float:
    """Basketball-Reference team possessions: average of both teams' estimates."""

    def est(a: dict, b: dict) -> float:
        orb_share = a["oreb"] / (a["oreb"] + b["dreb"]) if (a["oreb"] + b["dreb"]) else 0
        return a["fga"] + 0.4 * a["fta"] - 1.07 * orb_share * (a["fga"] - a["fgm"]) + a["tov"]

    return 0.5 * (est(t, o) + est(o, t))


def usage_rate(p: dict, t: dict) -> float | None:
    """Share of team plays used while on the floor (0.25 = 25%)."""
    plays_team = t["fga"] + 0.44 * t["fta"] + t["tov"]
    if not p["minutes"] or not plays_team:
        return None
    return ((p["fga"] + 0.44 * p["fta"] + p["tov"]) * (t["minutes"] / 5)) / (
        p["minutes"] * plays_team
    )


_POSS_SQL = """
UPDATE team_game_logs t SET possessions = 0.5 * (
    (t.fga + 0.4 * t.fta
       - 1.07 * (t.oreb::numeric / NULLIF(t.oreb + o.dreb, 0)) * (t.fga - t.fgm) + t.tov)
  + (o.fga + 0.4 * o.fta
       - 1.07 * (o.oreb::numeric / NULLIF(o.oreb + t.dreb, 0)) * (o.fga - o.fgm) + o.tov))
FROM team_game_logs o
WHERE o.game_id = t.game_id AND o.team_id = t.opp_team_id AND t.game_id = ANY(:ids)
"""
_RATINGS_SQL = """
UPDATE team_game_logs t SET
    pace = 48 * t.possessions / NULLIF(t.minutes / 5, 0),
    off_rtg = 100 * t.pts / NULLIF(t.possessions, 0),
    def_rtg = 100 * o.pts / NULLIF(t.possessions, 0)
FROM team_game_logs o
WHERE o.game_id = t.game_id AND o.team_id = t.opp_team_id AND t.game_id = ANY(:ids)
"""
_USAGE_SQL = """
UPDATE player_game_logs p SET usage_pct = LEAST(9.9999,
    (p.fga + 0.44 * p.fta + p.tov) * (t.minutes / 5)
      / (p.minutes * NULLIF(t.fga + 0.44 * t.fta + t.tov, 0)))
FROM team_game_logs t
WHERE p.game_id = t.game_id AND p.team_id = t.team_id AND p.minutes > 0
  AND p.game_id = ANY(:ids)
"""


def refresh_advanced(session: Session, game_ids: list[str]) -> None:
    for sql in (_POSS_SQL, _RATINGS_SQL, _USAGE_SQL):
        session.execute(text(sql), {"ids": game_ids})


# --- Validation (pure over loaded rows) --------------------------------------


def validate(team_logs: Rows, player_logs: Rows, tolerance_min: float = 1.5) -> list[str]:
    """Plan Stage 1 checks: player points sum to team points; player minutes to team minutes."""
    issues = []
    pts: dict[tuple, int] = defaultdict(int)
    mins: dict[tuple, float] = defaultdict(float)
    for p in player_logs:
        key = (p["game_id"], p["team_id"])
        pts[key] += p["pts"] or 0
        mins[key] += p["minutes"] or 0
    for t in team_logs:
        key = (t["game_id"], t["team_id"])
        if key not in pts:
            issues.append(f"{key[0]} team {key[1]}: no player rows")
            continue
        if t["pts"] is not None and pts[key] != t["pts"]:
            issues.append(f"{key[0]} team {key[1]}: players {pts[key]} pts vs team {t['pts']}")
        if t["minutes"] and abs(mins[key] - t["minutes"]) > tolerance_min:
            issues.append(
                f"{key[0]} team {key[1]}: players {mins[key]:.1f} min vs team {t['minutes']:.0f}"
            )
    return issues


# --- Database writes ---------------------------------------------------------


def _upsert(session: Session, model, rows: Rows, keys: list[str], update_cols: list[str]):
    for i in range(0, len(rows), 2000):  # stay under Postgres' bind-parameter limit
        chunk = rows[i : i + 2000]
        stmt = insert(model).values(chunk)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=keys, set_={c: stmt.excluded[c] for c in update_cols}
            )
        )


def load_logs(
    session: Session, games: Rows, team_logs: Rows, player_rows: Rows, names: dict[int, str]
) -> None:
    """Parents before children: games, players, then team and player logs."""
    if games:
        # Keep any tip time the schedule loader set; fill in results.
        _upsert(
            session,
            Game,
            games,
            ["game_id"],
            ["home_score", "away_score", "overtimes", "status", "season_type"],
        )
    for pid, name in names.items():
        ensure_player(session, pid, name)
    if team_logs:
        _upsert(
            session,
            TeamGameLog,
            team_logs,
            ["game_id", "team_id"],
            [c for c in team_logs[0] if c not in ("game_id", "team_id")],
        )
    known = {g["game_id"] for g in games}
    rows = [r for r in player_rows if r["game_id"] in known]
    if rows:
        # Box-score fields (started, dnp_reason) are owned by apply_boxscore: never overwritten.
        _upsert(
            session,
            PlayerGameLog,
            rows,
            ["game_id", "player_id"],
            [c for c in rows[0] if c not in ("game_id", "player_id", "started")],
        )


def apply_boxscore(session: Session, game_id: str, box_rows: Rows) -> None:
    started, dnp_rows, names = boxscore_to_updates(box_rows)
    for pid, name in names.items():
        ensure_player(session, pid, name)
    for pid, is_starter in started.items():
        session.execute(
            update(PlayerGameLog)
            .where(PlayerGameLog.game_id == game_id, PlayerGameLog.player_id == pid)
            .values(started=is_starter)
        )
    if dnp_rows:
        stmt = insert(PlayerGameLog).values(dnp_rows)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["game_id", "player_id"],
                set_={"dnp_reason": stmt.excluded.dnp_reason},
                where=PlayerGameLog.dnp.is_(True),  # never mark a player who played as DNP
            )
        )
    session.execute(update(Game).where(Game.game_id == game_id).values(boxscore_loaded=True))


def update_players_from_index(session: Session, index_rows: Rows, set_team: bool) -> int:
    """Positions from the NBA player index; with set_team, also each player's current team.

    set_team is only for the current season: an older season's index describes old rosters.
    A TEAM_ID of 0 (free agent) clears the current team."""
    now = datetime.now(UTC)
    n = 0
    for r in index_rows:
        # Rookies and new signings join the table here, before their first game: otherwise
        # the injury report and DraftKings can't be matched to them until they've played.
        first, last = r.get("PLAYER_FIRST_NAME") or "", r.get("PLAYER_LAST_NAME") or ""
        ensure_player(session, int(r["PERSON_ID"]), f"{first} {last}".strip())
        values = {}
        if r.get("POSITION"):
            values["position"] = r["POSITION"][:8]
        if set_team:
            team = r.get("TEAM_ID")
            values["current_team_id"] = int(team) if team else None
            values["roster_updated"] = now
        if values:
            n += (
                session.execute(
                    update(Player).where(Player.player_id == r["PERSON_ID"]).values(**values)
                ).rowcount
                or 0
            )
    return n


# --- Orchestration -----------------------------------------------------------


def nba_only(games: Rows, team_logs: Rows, player_rows: Rows, known: set[int]):
    """Drop games involving teams that aren't NBA teams (preseason games against touring
    international clubs, e.g. team 50009), with all their team and player rows: they aren't
    modeled, and their teams aren't in `teams`."""
    keep = {
        g["game_id"] for g in games if g["home_team_id"] in known and g["away_team_id"] in known
    }
    dropped = len(games) - len(keep)
    if dropped:
        log.info("skipped %d games against non-NBA teams", dropped)
    return (
        [g for g in games if g["game_id"] in keep],
        [r for r in team_logs if r["game_id"] in keep],
        [r for r in player_rows if r["game_id"] in keep and r.get("team_id", 0) in known],
    )


def names_in(player_rows: Rows, names: dict[int, str]) -> dict[int, str]:
    """Only the players who still have rows: a touring club's players are dropped with their
    games, and must not be registered as players either."""
    kept = {r["player_id"] for r in player_rows}
    return {pid: n for pid, n in names.items() if pid in kept}


def load_period(
    fetch: Fetch,
    season: str,
    season_type_name: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[str], list[str]]:
    """Load one season type (optionally a date range). Returns (game_ids, issues)."""
    season_type = SEASON_TYPES[season_type_name]
    span = f"_{date_from:%Y%m%d}_{date_to:%Y%m%d}" if date_from and date_to else ""
    key = f"{season}_{season_type}{span}"
    dates = {}
    if date_from and date_to:
        dates = {
            "date_from_nullable": f"{date_from:%m/%d/%Y}",
            "date_to_nullable": f"{date_to:%m/%d/%Y}",
        }
    cache = not dates  # daily pulls are always fresh: same-day data can still change
    teams = fetch(
        "leaguegamelog",
        f"{key}_T",
        cache=cache,
        season=season,
        season_type_all_star=season_type_name,
        player_or_team_abbreviation="T",
        **dates,
    )["LeagueGameLog"]
    if not teams:
        return [], []
    players = fetch(
        "playergamelogs",
        key,
        cache=cache,
        season_nullable=season,
        season_type_nullable=season_type_name,
        **dates,
    )["PlayerGameLogs"]

    games, team_logs, issues = team_logs_to_rows(teams, season_type)
    player_rows, names = player_logs_to_rows(players)
    with session_scope() as s:
        known = set(s.scalars(select(Team.team_id)).all())
    games, team_logs, player_rows = nba_only(games, team_logs, player_rows, known)
    names = names_in(player_rows, names)
    issues += validate(team_logs, player_rows)
    game_ids = [g["game_id"] for g in games]
    with session_scope() as s:
        load_logs(s, games, team_logs, player_rows, names)
        refresh_advanced(s, game_ids)
    log.info(
        "%s %s: %d games, %d player rows, %d issues",
        season,
        season_type,
        len(games),
        len(player_rows),
        len(issues),
    )
    return game_ids, issues


MAX_FAILURES_IN_A_ROW = 5
COOL_DOWN_SEC = 120


class SiteBlocked(RuntimeError):
    """The stats site refused MAX_FAILURES_IN_A_ROW requests in a row."""


@dataclass
class BoxscoreRun:
    loaded: int = 0
    failed: list[str] = field(default_factory=list)  # refused or timed out: worth retrying


def load_boxscores(fetch: Fetch, game_ids: list[str]) -> BoxscoreRun:
    """One request per game not yet loaded; commits per game so a crash loses nothing."""
    with session_scope() as s:
        pending = s.scalars(
            select(Game.game_id).where(Game.game_id.in_(game_ids), Game.boxscore_loaded.is_(False))
        ).all()
    run, in_a_row = BoxscoreRun(), 0
    for i, game_id in enumerate(sorted(pending), 1):
        try:
            rows = fetch("boxscoretraditionalv3", game_id, game_id=game_id)["PlayerStats"]
        except RuntimeError as exc:
            # The stats site throttles long runs: skip this game (it stays unloaded, so the
            # next run retries it), let the site cool down, and carry on.
            run.failed.append(game_id)
            in_a_row += 1
            if in_a_row >= MAX_FAILURES_IN_A_ROW:
                log.error(
                    "%d box scores failed in a row; the stats site is refusing requests",
                    in_a_row,
                )
                raise SiteBlocked(f"{in_a_row} box scores failed in a row") from exc
            log.warning("skipping %s for now (%s); pausing %ds", game_id, exc, COOL_DOWN_SEC)
            time.sleep(COOL_DOWN_SEC)
            continue
        in_a_row = 0
        if not rows:
            # leave boxscore_loaded false so the next run retries it
            log.warning("box score for %s came back empty; will retry next run", game_id)
            continue
        with session_scope() as s:
            apply_boxscore(s, game_id, rows)
        run.loaded += 1
        if i % 100 == 0:
            log.info("box scores: %d / %d", i, len(pending))
    if run.failed:
        log.warning(
            "%d box scores not loaded yet (run the backfill again to retry): %s",
            len(run.failed),
            ", ".join(run.failed[:20]),
        )
    return run


def backfill_season(fetch: Fetch, season: str, boxscores: bool = True) -> list[str]:
    all_ids, issues = [], []
    for name in SEASON_TYPES:
        ids, problems = load_period(fetch, season, name)
        all_ids += ids
        issues += problems
    index = fetch("playerindex", season, season=season)["PlayerIndex"]
    with session_scope() as s:
        # Current team only from the current season's index; older ones describe old rosters.
        update_players_from_index(s, index, set_team=(season == season_for_date(date.today())))
    if boxscores:
        load_boxscores(fetch, all_ids)
    for msg in issues[:50]:
        log.warning("validation: %s", msg)
    if len(issues) > 50:
        log.warning("validation: %d more issues not shown", len(issues) - 50)
    return issues


RETRY_DAYS = 30  # the daily job also retries box scores still missing from this many days back


def catch_up_start(season: str, day: date) -> date:
    """The earliest date in the last RETRY_DAYS with a game but no player logs: a morning
    without `stats-day` (or a blocked one) is filled in by the next run, instead of leaving
    those games without stats (and their picks unsettled) for good."""
    with session_scope() as s:
        first = s.scalar(
            select(func.min(Game.game_date)).where(
                Game.season == season,
                Game.game_date >= day - timedelta(days=RETRY_DAYS),
                Game.game_date <= day,
                Game.season_type != "cup_final",
                # game logs, not box scores: a box score alone only adds the players who
                # did NOT play, and that wrongly passed as "loaded" (Oct 22 picks voided)
                ~select(TeamGameLog.game_id).where(TeamGameLog.game_id == Game.game_id).exists(),
            )
        )
    return min(first, day) if first else day


def pull_day(fetch: Fetch, day: date) -> list[str]:
    """Daily job: yesterday's games (all season types), their box scores, any box scores still
    missing from the last RETRY_DAYS, and today's rosters (positions and current teams)."""
    season = season_for_date(day)
    start = catch_up_start(season, day)
    if start < day:
        log.info("catching up game logs from %s (days missed since)", start)
    ids, issues = [], []
    for name in SEASON_TYPES:  # one request per season type covers the whole range
        got, problems = load_period(fetch, season, name, start, day)
        ids += got
        issues += problems
    with session_scope() as s:
        pending = s.scalars(
            select(Game.game_id).where(
                Game.season == season,
                Game.status == "final",
                Game.boxscore_loaded.is_(False),
                Game.game_date >= day - timedelta(days=RETRY_DAYS),
                Game.season_type != "cup_final",  # no box-score logs for the Cup final
            )
        ).all()
    load_boxscores(fetch, sorted(set(ids) | set(pending)))
    # Rosters every day (trades happen any day): positions for debuts, and each player's current
    # team so live slates place a traded player on his new team before he plays for it.
    index = fetch("playerindex", f"{season}_{day:%Y%m%d}", cache=False, season=season)
    with session_scope() as s:
        update_players_from_index(s, index["PlayerIndex"], set_team=True)
    return issues


def yesterday_et(now: datetime) -> date:
    from zoneinfo import ZoneInfo

    return (now.astimezone(ZoneInfo("America/New_York")) - timedelta(days=1)).date()


BLOCK_WAIT_SEC = 30 * 60  # how long the stats site's blocks have lasted
MAX_BLOCK_WAITS = 8  # about four hours of waiting before giving up
CATCH_UP_ROUNDS = 3


def _boxscores_loaded(seasons: list[str]) -> int:
    with session_scope() as s:
        return (
            s.scalar(
                select(func.count())
                .select_from(Game)
                .where(Game.season.in_(seasons), Game.boxscore_loaded.is_(True))
            )
            or 0
        )


def backfill_resilient(
    fetch: Fetch, seasons: list[str], boxscores: bool = True, sleep=time.sleep
) -> dict[str, list[str]]:
    """Backfill unattended. The stats site blocks for a while every few hundred requests:
    when it does, wait the block out and resume the same season (what's loaded is cached or
    marked loaded, so a resumed season only does what's left). Gives up only after
    MAX_BLOCK_WAITS waits in a row with no progress between them. A last pass retries box
    scores skipped along the way. Returns issues per season."""
    results: dict[str, list[str]] = {}
    waits = 0
    progress = (0, _boxscores_loaded(seasons))

    def wait_out_block(reason: str) -> None:
        nonlocal waits, progress
        now = (len(results), _boxscores_loaded(seasons))
        waits = 1 if now != progress else waits + 1  # progress since the last wait resets it
        progress = now
        if waits > MAX_BLOCK_WAITS:
            raise RuntimeError(
                f"The stats site kept refusing requests ({reason}). Everything loaded so far "
                "is saved: run the backfill again later to continue."
            )
        log.warning(
            "stats site is refusing requests (%s): waiting %d minutes, then resuming "
            "(wait %d of %d without progress)",
            reason,
            BLOCK_WAIT_SEC // 60,
            waits,
            MAX_BLOCK_WAITS,
        )
        sleep(BLOCK_WAIT_SEC)

    queue = list(seasons)
    while queue:
        season = queue[0]
        try:
            results[season] = backfill_season(fetch, season, boxscores=boxscores)
        except RuntimeError as exc:
            wait_out_block(str(exc))
            continue
        log.info(
            "%s done: %d validation issues, %d requests so far",
            season,
            len(results[season]),
            getattr(fetch, "requests_made", 0),
        )
        queue.pop(0)
    if not boxscores:
        return results
    rounds = 0
    while rounds < CATCH_UP_ROUNDS:
        with session_scope() as s:
            missing = s.scalars(
                select(Game.game_id).where(
                    Game.season.in_(seasons),
                    Game.status == "final",
                    Game.boxscore_loaded.is_(False),
                )
            ).all()
        if not missing:
            break
        log.info("catch-up: %d box scores still missing", len(missing))
        try:
            run = load_boxscores(fetch, list(missing))
        except SiteBlocked as exc:
            wait_out_block(str(exc))  # a block doesn't use up a round
            continue
        rounds += 1
        if not run.failed:
            break  # what's left came back empty from the site: the daily job keeps retrying
    return results
