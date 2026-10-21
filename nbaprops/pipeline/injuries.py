"""Official NBA injury report (Stage 2).

Reports are PDFs at
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHAM.pdf
labelled e.g. "Injury Report: 12/19/25 07:30 AM". The NBA has also listed 15-minute reports,
so `candidate_urls` tries quarter-hour file names before the hourly one.

Extracted text has one row per line; the date, game time, matchup and team columns are only
printed on the first row they apply to, so the parser carries them forward. Long reasons wrap
onto a following line, which is appended to the previous row.
"""

import io
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from nba_api.stats.static import teams as static_teams
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from nbaprops.db.models import Game, InjurySnapshot
from nbaprops.pipeline.names import last_first_to_first_last
from nbaprops.pipeline.reference import queue_unmatched, resolve_player_by_name

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
STATUSES = ("Out", "Questionable", "Doubtful", "Probable", "Available")
_REASON_STARTS = (
    "Injury/Illness",
    "G League",
    "Not With Team",
    "Personal Reasons",
    "Rest",
    "League Suspension",
    "Health and Safety",
    "Concussion Protocol",
)
_TEAM_NAMES = sorted((t["full_name"] for t in static_teams.get_teams()), key=len, reverse=True)
# The report's own spellings, where they differ from the reference data.
_TEAM_ALIASES = {"LA Clippers": "Los Angeles Clippers"}
# Every way a team's cell can start a line -> its full name. The 2026-27 report's text
# extracts with the spaces squashed out of team names ("Philadelphia76ers"), so each name is
# also matched without spaces. Longest first, so "Los Angeles Clippers" wins over shorter forms.
_TEAM_FORMS = sorted(
    {
        form: full
        for name, full in [(n, n) for n in _TEAM_NAMES] + list(_TEAM_ALIASES.items())
        for form in (name, name.replace(" ", ""))
    }.items(),
    key=lambda kv: len(kv[0]),
    reverse=True,
)
_TEAM_BY_NAME = {t["full_name"]: t["id"] for t in static_teams.get_teams()}
_TEAM_BY_ABBR = {t["abbreviation"]: t["id"] for t in static_teams.get_teams()}

_HEADER = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2}\s*[AP]M)")
_DATE = re.compile(r"^(\d{2}/\d{2}/\d{4})\s+")
_TIME = re.compile(r"^(\d{2}:\d{2})\s*\(ET\)\s+")
_MATCHUP = re.compile(r"^([A-Z]{2,3})@([A-Z]{2,3})\s+")
_PLAYER = re.compile(
    r"^(?P<name>[^\W\d_][\w'.\- ]*?,\s*[\w'.\- ]+?)\s+"
    r"(?P<status>" + "|".join(STATUSES) + r")\b\s*(?P<reason>.*)$"
)


@dataclass
class InjuryRow:
    report_ts: datetime
    game_date: date | None
    away_abbr: str | None
    home_abbr: str | None
    team_name: str | None
    player_name: str  # "Last, First" as printed
    status: str
    reason: str


def candidate_urls(now: datetime, hours_back: int = 6) -> list[str]:
    """Most recent first: quarter-hour names, then the hourly name, for each hour back."""
    local = now.astimezone(ET)
    urls = []
    for h in range(hours_back + 1):
        t = local - timedelta(hours=h)
        day = t.strftime("%Y-%m-%d")
        hh = t.strftime("%I")
        ampm = t.strftime("%p")
        minutes = [m for m in (45, 30, 15) if h > 0 or m <= local.minute]
        urls += [f"{BASE}{day}_{hh}_{m:02d}{ampm}.pdf" for m in minutes]
        urls.append(f"{BASE}{day}_{hh}{ampm}.pdf")
    return urls


def parse_report_text(text: str) -> list[InjuryRow]:
    rows: list[InjuryRow] = []
    report_ts = None
    game_date = None
    away = home = team = None
    orphans = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEADER.search(line)
        if m:
            report_ts = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%m/%d/%y %I:%M %p")
            report_ts = report_ts.replace(tzinfo=ET)
            continue
        if line.startswith(("Page ", "Game Date")):
            continue
        if m := _DATE.match(line):
            game_date = datetime.strptime(m.group(1), "%m/%d/%Y").date()
            line = line[m.end() :]
        if m := _TIME.match(line):
            line = line[m.end() :]
        if m := _MATCHUP.match(line):
            away, home = m.group(1), m.group(2)
            line = line[m.end() :]
        for form, full in _TEAM_FORMS:
            if line.startswith(form):
                team = full
                line = line[len(form) :].strip()
                break
        if not line or line.startswith("NOT YET SUBMITTED"):
            continue
        if m := _PLAYER.match(line):
            if report_ts is None:
                continue
            rows.append(
                InjuryRow(
                    report_ts=report_ts,
                    game_date=game_date,
                    away_abbr=away,
                    home_abbr=home,
                    team_name=team,
                    player_name=m.group("name").strip(),
                    status=m.group("status"),
                    reason=m.group("reason").strip(),
                )
            )
        elif line.startswith(_REASON_STARTS):
            if rows and not rows[-1].reason:
                rows[-1].reason = line  # the whole reason wrapped onto the next line
            else:
                orphans += 1  # a reason whose player-name cell didn't extract; can't attribute it
        elif rows:
            rows[-1].reason = f"{rows[-1].reason} {line}".strip()
    if orphans:
        log.info("injury report: %d reason lines without a player name were skipped", orphans)
    return rows


def fetch_latest_report(now: datetime, session: requests.Session | None = None) -> tuple[str, str]:
    """Return (url, extracted text) of the most recent report available."""
    import pdfplumber

    http = session or requests.Session()
    http.headers.setdefault("User-Agent", "Mozilla/5.0")
    for url in candidate_urls(now):
        resp = http.get(url, timeout=20)
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            continue
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return url, text
    raise FileNotFoundError("no injury report found in the last few hours")


def store_report(session: Session, rows: list[InjuryRow], snapshot_ts: datetime) -> int:
    """Insert rows (idempotent per report). Returns rows written."""
    written = 0
    for r in rows:
        team_id = _TEAM_BY_NAME.get(r.team_name or "")
        game_id = None
        if r.game_date and r.home_abbr in _TEAM_BY_ABBR and r.away_abbr in _TEAM_BY_ABBR:
            game_id = session.scalar(
                select(Game.game_id).where(
                    Game.game_date == r.game_date,
                    Game.home_team_id == _TEAM_BY_ABBR[r.home_abbr],
                    Game.away_team_id == _TEAM_BY_ABBR[r.away_abbr],
                )
            )
        player_id = resolve_player_by_name(session, last_first_to_first_last(r.player_name))
        if player_id is None:
            queue_unmatched(
                session, "injury_report", last_first_to_first_last(r.player_name), snapshot_ts
            )
        res = session.execute(
            insert(InjurySnapshot)
            .values(
                snapshot_ts=snapshot_ts,
                report_ts=r.report_ts,
                game_id=game_id,
                team_id=team_id,
                player_id=player_id,
                player_name_raw=r.player_name,
                status=r.status,
                reason=r.reason,
            )
            .on_conflict_do_nothing(constraint="uq_injury_report_row")
            .returning(InjurySnapshot.id)
        )
        written += len(res.fetchall())  # rowcount is unreliable for ON CONFLICT inserts
    return written
