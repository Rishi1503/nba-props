"""Write odds to Postgres: match events to games, resolve players, log changes only.

Change-only rule (plan, Stage 2): a prop row is written when its line or prices differ from
the latest stored row for the same key, or when the snapshot is a closing snapshot. Keys:
  main line:        (book, player, market, "main")             -> compares line + prices
  alternate/milest: (book, player, market, line_type, line)    -> compares prices
so a main line moving 26.5 -> 27.5 is one change on one key, while each alt rung is its own key.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from nbaprops.db.models import Game, GameLineSnapshot, OddsPoll, OddsSnapshot, ProviderEvent
from nbaprops.pipeline.names import normalize_name
from nbaprops.pipeline.odds import EventInfo, GameLineQuote, PropQuote
from nbaprops.pipeline.reference import (
    ensure_player,
    queue_unmatched,
    resolve_player_by_name,
    team_id_lookup,
)

MATCH_WINDOW = timedelta(hours=12)


@dataclass
class StoreResult:
    props_seen: int = 0
    props_matched: int = 0
    props_written: int = 0
    props_withdrawn: int = 0
    lines_written: int = 0
    unmatched_players: set[str] = field(default_factory=set)
    pulled_players: set[str] = field(default_factory=set)

    @property
    def match_rate(self) -> float:
        """Share of prop quotes matched to an NBA player (plan target: 98%)."""
        return self.props_matched / self.props_seen if self.props_seen else 1.0


# --- Event -> game -----------------------------------------------------------


def match_event(session: Session, provider: str, event: EventInfo) -> str | None:
    """Return the NBA game_id for a provider event, caching the match. None if no game fits."""
    cached = session.get(ProviderEvent, (provider, event.provider_event_id))
    if cached:
        return cached.game_id

    teams = team_id_lookup(session)
    home = teams.get(event.home_team.lower())
    away = teams.get(event.away_team.lower())
    if home is None or away is None:
        return None
    candidates = session.scalars(
        select(Game).where(Game.home_team_id == home, Game.away_team_id == away)
    ).all()
    best = None
    for g in candidates:
        if g.tip_time is None:
            continue
        gap = abs(g.tip_time - event.commence_time)
        if gap <= MATCH_WINDOW and (best is None or gap < abs(best.tip_time - event.commence_time)):
            best = g
    if best is None:
        return None
    session.add(
        ProviderEvent(
            provider=provider, provider_event_id=event.provider_event_id, game_id=best.game_id
        )
    )
    session.flush()
    return best.game_id


# --- Props -------------------------------------------------------------------


def _resolve_player(session: Session, q: PropQuote) -> int | None:
    """The quote's player id. Each player is ensured or looked up once per session (one odds
    snapshot): the same ~300 players repeat across ~5,500 props, and two database round
    trips per prop made a full snapshot take ~20 minutes, past GitHub's time limit."""
    if q.nba_player_id is not None:
        ensured = session.info.setdefault("ensured_players", set())
        key = (q.nba_player_id, (q.player_name_raw or "").strip())
        if key not in ensured:
            ensure_player(session, q.nba_player_id, q.player_name_raw)
            ensured.add(key)
        return q.nba_player_id
    names = session.info.setdefault("resolved_names", {})
    if q.player_name_raw not in names:
        names[q.player_name_raw] = resolve_player_by_name(session, q.player_name_raw)
    return names[q.player_name_raw]


def _prop_key(book: str, pkey: str, market: str, line_type: str, line: float) -> tuple:
    return (book, pkey, market, line_type, None if line_type == "main" else round(line, 1))


def _latest_props(session: Session, game_id: str) -> dict[tuple, tuple]:
    """Current state per prop key: (line, over, under, player_id, name_raw), or None once the
    prop has been withdrawn (pulled, suspended, or an alternate line taken down)."""
    rows = session.execute(
        select(
            OddsSnapshot.book,
            OddsSnapshot.player_id,
            OddsSnapshot.player_name_raw,
            OddsSnapshot.market,
            OddsSnapshot.line_type,
            OddsSnapshot.line,
            OddsSnapshot.over_price,
            OddsSnapshot.under_price,
            OddsSnapshot.is_withdrawn,
        )
        .where(OddsSnapshot.game_id == game_id)
        .order_by(OddsSnapshot.snapshot_ts, OddsSnapshot.id)
    ).all()
    latest = {}
    for book, pid, name, market, line_type, line, over, under, gone in rows:
        pkey = str(pid) if pid is not None else normalize_name(name)
        key = _prop_key(book, pkey, market, line_type, float(line))
        latest[key] = None if gone else (float(line), over, under, pid, name)
    return latest


def store_props(
    session: Session,
    provider: str,
    game_id: str,
    quotes: list[PropQuote],
    snapshot_ts: datetime,
    is_close: bool = False,
    result: StoreResult | None = None,
) -> StoreResult:
    result = result or StoreResult()
    latest = _latest_props(session, game_id)
    seen: set[tuple] = set()
    books: set[str] = set()
    for q in quotes:
        result.props_seen += 1
        pid = _resolve_player(session, q)
        if pid is None:
            result.unmatched_players.add(q.player_name_raw)
            queue_unmatched(session, "propline", q.player_name_raw, snapshot_ts)
        else:
            result.props_matched += 1
        pkey = str(pid) if pid is not None else normalize_name(q.player_name_raw)
        key = _prop_key(q.book, pkey, q.market, q.line_type, q.line)
        seen.add(key)
        books.add(q.book)
        current = (q.line, q.over_price, q.under_price)
        prior = latest.get(key)
        # Change-only, including closing ticks: the board at any moment (the close included)
        # is the latest row per prop plus withdrawals, so rewriting unchanged quotes on every
        # closing tick would only multiply storage (about 0.6 GB a season, beyond free tiers).
        if prior is not None and prior[:3] == current:
            continue
        session.add(
            OddsSnapshot(
                snapshot_ts=snapshot_ts,
                provider=provider,
                book=q.book,
                game_id=game_id,
                player_id=pid,
                player_name_raw=q.player_name_raw,
                market=q.market,
                line_type=q.line_type,
                line=Decimal(str(q.line)),
                over_price=q.over_price,
                under_price=q.under_price,
                is_close=is_close,
            )
        )
        latest[key] = (*current, pid, q.player_name_raw)
        result.props_written += 1
    # Withdrawals: a prop this book quoted before but not in this snapshot is off the board.
    # Only for books present in the snapshot, so one book missing from a response (an API
    # glitch) isn't read as that book pulling every prop.
    for key, state in latest.items():
        if state is None or key in seen or key[0] not in books:
            continue
        line, _, _, pid, name = state
        session.add(
            OddsSnapshot(
                snapshot_ts=snapshot_ts,
                provider=provider,
                book=key[0],
                game_id=game_id,
                player_id=pid,
                player_name_raw=name,
                market=key[2],
                line_type=key[3],
                line=Decimal(str(line)),
                over_price=None,
                under_price=None,
                is_close=is_close,
                is_withdrawn=True,
            )
        )
        result.props_withdrawn += 1
    session.flush()
    return result


def record_poll(
    session: Session,
    provider: str,
    game_id: str,
    quotes: list[PropQuote],
    snapshot_ts: datetime,
    is_close: bool = False,
) -> None:
    """Note that this game was polled, and which books answered, even if nothing changed."""
    session.add(
        OddsPoll(
            game_id=game_id,
            snapshot_ts=snapshot_ts,
            provider=provider,
            books=sorted({q.book for q in quotes}),
            n_props=len(quotes),
            is_close=is_close,
        )
    )


# --- Game lines --------------------------------------------------------------


def store_game_lines(
    session: Session,
    provider: str,
    game_id: str,
    quotes: list[GameLineQuote],
    snapshot_ts: datetime,
    is_close: bool = False,
    result: StoreResult | None = None,
) -> StoreResult:
    result = result or StoreResult()
    rows = session.execute(
        select(
            GameLineSnapshot.book,
            GameLineSnapshot.market,
            GameLineSnapshot.point,
            GameLineSnapshot.home_price,
            GameLineSnapshot.away_price,
        )
        .where(GameLineSnapshot.game_id == game_id)
        .order_by(GameLineSnapshot.snapshot_ts, GameLineSnapshot.id)
    ).all()
    latest = {(b, m): (float(p) if p is not None else None, h, a) for b, m, p, h, a in rows}
    for q in quotes:
        current = (q.point, q.home_price, q.away_price)
        if not is_close and latest.get((q.book, q.market)) == current:
            continue
        session.add(
            GameLineSnapshot(
                snapshot_ts=snapshot_ts,
                provider=provider,
                book=q.book,
                game_id=game_id,
                market=q.market,
                point=Decimal(str(q.point)) if q.point is not None else None,
                home_price=q.home_price,
                away_price=q.away_price,
                is_close=is_close,
            )
        )
        latest[(q.book, q.market)] = current
        result.lines_written += 1
    session.flush()
    return result


def has_close(session: Session, game_id: str) -> bool:
    """Whether a closing snapshot was taken. Read from the poll record: closing ticks only
    write changed quotes, so an unchanged close leaves no odds rows."""
    return (
        session.scalar(
            select(OddsPoll.id)
            .where(OddsPoll.game_id == game_id, OddsPoll.is_close.is_(True))
            .limit(1)
        )
        is not None
    )
