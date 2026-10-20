"""Load the NBA schedule (preseason through playoffs) into `games`.

Source: the NBA's public CDN schedule file, which needs no API key and isn't the rate-limited
stats site. Field names below follow that file's structure; the parser is defensive, and
`parse_schedule` is unit-tested against a fixture of the same shape. Verify on first live run.
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from nbaprops.db.models import Game

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
ET = ZoneInfo("America/New_York")

# Third-to-first digits of an NBA game id encode the game type.
_SEASON_TYPE_BY_PREFIX = {
    "001": "preseason",
    "002": "regular",
    "004": "playoffs",
    "005": "playin",
    "006": "cup_final",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}


def season_from_game_id(game_id: str) -> str:
    """ "0022600001" -> "2026-27"."""
    start = 2000 + int(game_id[3:5])
    return f"{start}-{str(start + 1)[2:]}"


def parse_schedule(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the CDN schedule JSON into `games` row dicts. Skips All-Star and unknown types."""
    rows = []
    for day in payload.get("leagueSchedule", {}).get("gameDates", []):
        for g in day.get("games", []):
            game_id = g.get("gameId", "")
            season_type = _SEASON_TYPE_BY_PREFIX.get(game_id[:3])
            if not season_type:
                continue
            home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
            if not home.get("teamId") or not away.get("teamId"):
                continue  # TBD playoff matchups
            tip_utc = datetime.fromisoformat(g["gameDateTimeUTC"].replace("Z", "+00:00"))
            status = {1: "scheduled", 2: "live", 3: "final"}.get(g.get("gameStatus"), "scheduled")
            rows.append(
                {
                    "game_id": game_id,
                    "season": season_from_game_id(game_id),
                    "season_type": season_type,
                    "game_date": tip_utc.astimezone(ET).date(),
                    "tip_time": tip_utc,
                    "home_team_id": home["teamId"],
                    "away_team_id": away["teamId"],
                    "status": status,
                }
            )
    return rows


def fetch_schedule(timeout: float = 30.0) -> dict[str, Any]:
    resp = requests.get(SCHEDULE_URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def upsert_games(session: Session, rows: list[dict[str, Any]]) -> int:
    """Insert new games; update tip time, status, and date on existing ones (reschedules).
    Teams must already be seeded (see reference.seed_teams)."""
    # Skip games against teams that aren't in `teams` (preseason exhibitions vs
    # international clubs, e.g. team 50015): not modeled, and they'd break the foreign key.
    import logging as _logging

    from sqlalchemy import text as _text

    _known = set(session.execute(_text("SELECT team_id FROM teams")).scalars())
    _skip = [r for r in rows if r["home_team_id"] not in _known or r["away_team_id"] not in _known]
    if _skip:
        _logging.getLogger(__name__).info(
            "schedule: skipped %d games with non-NBA teams", len(_skip)
        )
    rows = [r for r in rows if r not in _skip]
    if not rows:
        return 0
    stmt = insert(Game).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Game.game_id],
            set_={
                "tip_time": stmt.excluded.tip_time,
                "game_date": stmt.excluded.game_date,
                "status": stmt.excluded.status,
            },
        )
    )
    return len(rows)
