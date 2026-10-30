"""Confirmed starters before tip-off, and the picks they put in doubt.

The injury report says who is out, not who moves into the starting five. A bench player
starting tonight (Herbert Jones on Oct 29, starting for Dejounte Murray) plays well beyond
the role his projection assumed, and the gap shows up as a large, false edge.

Each cycle reads the NBA's live box score (cdn.nba.com, reachable from cloud runners) for
games tipping within the hour. Once it lists five starters per team, they are stored in
lineup_snapshots and compared with each player's last 10 games:
  - "now starting": starts tonight, started fewer than half of his last 10
  - "not starting": active but on the bench, started at least half of his last 10
Open picks on those players get one alert each and a `lineup` tag in their context (the
review can measure them later); paper picks stay as flagged. Pricing holds back new picks
on them, like the sanity cap.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
WINDOW = timedelta(minutes=60)  # check games tipping within the hour
MIN_GAMES = 3  # history needed before a start or a benching counts as a change
# the schedule loader's headers: the NBA's CDN turns away bare clients
from nbaprops.pipeline.schedule import _HEADERS as HEADERS  # noqa: E402


def _yes(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def parse_lineup(payload: dict) -> dict[int, dict[int, bool]] | None:
    """{team_id: {player_id: starter}} for ACTIVE players, or None until both teams list
    exactly five starters (before that the box score has no lineup worth trusting)."""
    game = (payload or {}).get("game") or {}
    out: dict[int, dict[int, bool]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side) or {}
        tid = team.get("teamId")
        players = team.get("players") or []
        active = {
            int(p["personId"]): _yes(p.get("starter"))
            for p in players
            if "personId" in p and str(p.get("status", "ACTIVE")).upper() == "ACTIVE"
        }
        if tid is None or sum(active.values()) != 5:
            return None
        out[int(tid)] = active
    return out


def fetch_boxscore(game_id: str, timeout: float = 10.0) -> dict | None:
    """The live box score, or None when it isn't published yet (403/404) or unreachable."""
    try:
        resp = requests.get(BOXSCORE_URL.format(game_id), headers=HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        log.info("lineups: %s unreachable (%s)", game_id, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def store_lineup(session, game_id: str, now: datetime, lineup: dict) -> int:
    rows = [
        {"g": game_id, "t": tid, "p": pid, "s": starter, "ts": now}
        for tid, players in lineup.items()
        for pid, starter in players.items()
    ]
    for r in rows:
        session.execute(
            text("""INSERT INTO lineup_snapshots (snapshot_ts, game_id, team_id, player_id,
                        starter, confirmed)
                    SELECT :ts, :g, :t, :p, :s, true
                    WHERE EXISTS (SELECT 1 FROM players WHERE player_id = :p)"""),
            r,
        )
    return len(rows)


def role_changes(session, day: date) -> list[dict]:
    """Players whose confirmed role tonight differs from their last 10 games."""
    rows = session.execute(
        text("""
        WITH latest AS (
            SELECT DISTINCT ON (l.game_id, l.player_id) l.game_id, l.player_id, l.starter
            FROM lineup_snapshots l JOIN games g USING (game_id)
            WHERE g.game_date = :d AND l.confirmed
            ORDER BY l.game_id, l.player_id, l.snapshot_ts DESC)
        SELECT lt.game_id, lt.player_id, lt.starter, h.n, h.starts
        FROM latest lt CROSS JOIN LATERAL (
            SELECT count(*) AS n, coalesce(sum(CASE WHEN x.started THEN 1 ELSE 0 END), 0) AS starts
            FROM (SELECT pgl.started FROM player_game_logs pgl JOIN games g2 USING (game_id)
                  WHERE pgl.player_id = lt.player_id AND g2.game_date < :d
                    AND pgl.started IS NOT NULL
                  ORDER BY g2.game_date DESC LIMIT 10) x) h
        WHERE h.n >= :k"""),
        {"d": day, "k": MIN_GAMES},
    ).all()
    out = []
    for gid, pid, starter, n, starts in rows:
        usual = starts / n >= 0.5
        if starter and not usual:
            out.append(
                {
                    "game_id": gid,
                    "player_id": int(pid),
                    "change": "now starting",
                    "starts": int(starts),
                    "games": int(n),
                }
            )
        elif not starter and usual:
            out.append(
                {
                    "game_id": gid,
                    "player_id": int(pid),
                    "change": "not starting",
                    "starts": int(starts),
                    "games": int(n),
                }
            )
    return out


def changed_players(session, day: date) -> set[tuple[str, int]]:
    return {(c["game_id"], c["player_id"]) for c in role_changes(session, day)}


def check(session, now: datetime, notifier, fetch=fetch_boxscore) -> list[dict]:
    """One cycle's lineup step: load starters for games tipping within the hour, then alert
    (once) and tag open picks on players whose role changed. Returns the changes."""
    from nbaprops.ops import alerts

    day = now.astimezone(ET).date()
    due = (
        session.execute(
            text("""SELECT g.game_id FROM games g
                WHERE g.tip_time BETWEEN :now AND :hi AND g.status <> 'final'
                  AND NOT EXISTS (SELECT 1 FROM lineup_snapshots l
                                  WHERE l.game_id = g.game_id AND l.confirmed)"""),
            {"now": now - timedelta(minutes=5), "hi": now + WINDOW},
        )
        .scalars()
        .all()
    )
    for gid in due:
        lineup = parse_lineup(fetch(gid) or {})
        if lineup:
            store_lineup(session, gid, now, lineup)
            log.info("lineups: starters confirmed for %s", gid)
    changes = role_changes(session, day)
    for c in changes:
        picks = session.execute(
            text("""SELECT p.pick_id, pl.full_name, p.market, p.side, p.line, p.stake
                    FROM picks p LEFT JOIN players pl USING (player_id)
                    WHERE p.game_id = :g AND p.player_id = :p AND p.result IS NULL
                      AND p.mode = 'paper'"""),
            {"g": c["game_id"], "p": c["player_id"]},
        ).all()
        if not picks:
            continue
        session.execute(
            text("""UPDATE picks SET context = coalesce(context, '{}'::jsonb)
                        || jsonb_build_object('lineup', CAST(:t AS text))
                    WHERE pick_id = ANY(:ids)"""),
            {"t": c["change"], "ids": [r.pick_id for r in picks]},
        )
        staked = [r for r in picks if float(r.stake or 0) > 0]
        if staked:
            name = staked[0].full_name or f"player {c['player_id']}"
            bets = ", ".join(f"{r.market} {r.side} {float(r.line)}" for r in staked)
            alerts.once(
                session,
                f"lineup:{c['game_id']}:{c['player_id']}",
                now,
                notifier,
                f"Lineup change: {name} is {c['change']} tonight (started {c['starts']} of "
                f"his last {c['games']}). Open pick: {bets}. Its projection assumed his "
                "usual role; reconsider before betting real money.",
            )
    return changes
