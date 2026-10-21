"""Reference data: teams and players, seeded from nba_api's bundled static lists (no network)."""

import difflib
from datetime import datetime

from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from nbaprops.db.models import (
    InjurySnapshot,
    OddsSnapshot,
    Player,
    PlayerAlias,
    Team,
    UnmatchedName,
)
from nbaprops.pipeline.names import compact_forms, normalize_name

# Spellings some books use that differ from the NBA's official team names.
TEAM_NAME_ALIASES = {
    "la clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
}


def seed_teams(session: Session) -> int:
    rows = [
        {"team_id": t["id"], "abbreviation": t["abbreviation"], "name": t["full_name"]}
        for t in static_teams.get_teams()
    ]
    stmt = insert(Team).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Team.team_id],
            set_={"abbreviation": stmt.excluded.abbreviation, "name": stmt.excluded.name},
        )
    )
    return len(rows)


def seed_players(session: Session) -> int:
    """Upsert every player nba_api knows about, plus a normalized-name alias for each
    active player. Names shared by two active players get no alias (ambiguous)."""
    roster = static_players.get_players()
    rows = [
        {"player_id": p["id"], "full_name": p["full_name"], "is_active": p["is_active"]}
        for p in roster
    ]
    stmt = insert(Player).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Player.player_id],
            set_={"full_name": stmt.excluded.full_name, "is_active": stmt.excluded.is_active},
        )
    )

    by_key: dict[str, list[int]] = {}
    for p in roster:
        if p["is_active"]:
            by_key.setdefault(normalize_name(p["full_name"]), []).append(p["id"])
    aliases = [
        {"alias_normalized": k, "player_id": ids[0], "source": "nba_static"}
        for k, ids in by_key.items()
        if len(ids) == 1
    ]
    if aliases:
        session.execute(insert(PlayerAlias).values(aliases).on_conflict_do_nothing())
    return len(rows)


def ensure_player(session: Session, player_id: int, name: str) -> None:
    """Insert a placeholder player row for ids the static list doesn't know yet (rookies), and
    a name alias so the injury report can be matched to him (an existing alias, possibly a
    namesake's, is never overwritten; a nameless row gets none)."""
    real = (name or "").strip()
    session.execute(
        insert(Player)
        # a nameless row gets a placeholder (filled in later from box scores or the roster
        # refresh) rather than stopping a long backfill
        .values(player_id=player_id, full_name=real or f"Player {player_id}", is_active=True)
        .on_conflict_do_nothing()
    )
    if real:  # after the player row: the alias refers to it
        session.execute(
            insert(PlayerAlias)
            .values(alias_normalized=normalize_name(real), player_id=player_id, source="roster")
            .on_conflict_do_nothing()
        )


def team_id_lookup(session: Session) -> dict[str, int]:
    """Lowercased full team name -> team_id, including known alias spellings."""
    lookup = {name.lower(): tid for tid, name in session.execute(select(Team.team_id, Team.name))}
    for alias, official in TEAM_NAME_ALIASES.items():
        if official.lower() in lookup:
            lookup[alias] = lookup[official.lower()]
    return lookup


def resolve_player_by_name(session: Session, name: str) -> int | None:
    key = normalize_name(name)
    hit = session.scalar(select(PlayerAlias.player_id).where(PlayerAlias.alias_normalized == key))
    if hit is not None:
        return hit
    # Squashed spaces ("Dereck LivelyII"): accept only a single, unambiguous player.
    for form in compact_forms(key):
        ids = set(
            session.scalars(
                select(PlayerAlias.player_id).where(
                    func.replace(PlayerAlias.alias_normalized, " ", "") == form
                )
            ).all()
        )
        if len(ids) == 1:
            return ids.pop()
    return None


# --- review queue for names that matched nobody (plan, Stage 2) ---------------


def queue_unmatched(session: Session, source: str, name: str, seen: datetime) -> None:
    stmt = insert(UnmatchedName).values(
        source=source,
        name_normalized=normalize_name(name),
        name_raw=name,
        first_seen=seen,
        last_seen=seen,
        seen_count=1,
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[UnmatchedName.source, UnmatchedName.name_normalized],
            set_={"last_seen": stmt.excluded.last_seen, "seen_count": UnmatchedName.seen_count + 1},
        )
    )


def suggest_players(session: Session, name: str, n: int = 3) -> list[tuple[int, str]]:
    """Closest active players by normalized name (suggestions only, never auto-applied)."""
    active = session.execute(
        select(Player.player_id, Player.full_name).where(Player.is_active.is_(True))
    ).all()
    keys = {normalize_name(full): (pid, full) for pid, full in active}
    return [keys[k] for k in difflib.get_close_matches(normalize_name(name), keys, n=n, cutoff=0.6)]


def add_alias(session: Session, name: str, player_id: int, source: str = "manual") -> int:
    """Map a spelling to a player, clear it from the queue, and fill in past rows.
    Returns the number of past odds and injury rows updated."""
    key = normalize_name(name)
    session.execute(
        insert(PlayerAlias)
        .values(
            alias_normalized=key,
            player_id=player_id,
            source=source,
        )
        .on_conflict_do_update(
            index_elements=[PlayerAlias.alias_normalized],
            set_={"player_id": player_id, "source": source},
        )
    )
    session.execute(delete(UnmatchedName).where(UnmatchedName.name_normalized == key))
    fixed = 0
    for model in (OddsSnapshot, InjurySnapshot):
        rows = session.execute(
            select(model.id, model.player_name_raw).where(model.player_id.is_(None))
        ).all()
        ids = [i for i, raw in rows if normalize_name(_as_first_last(raw)) == key]
        if ids:
            fixed += (
                session.execute(
                    update(model).where(model.id.in_(ids)).values(player_id=player_id)
                ).rowcount
                or 0
            )
    return fixed


def _as_first_last(raw: str) -> str:
    """Injury reports print "Last, First"; odds print "First Last"."""
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return raw
