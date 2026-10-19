"""End-to-end Stage 2 tests against real Postgres: seed -> schedule -> snapshots -> injuries."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

from nbaprops.db.base import get_engine, session_scope
from nbaprops.db.models import (
    Base,
    GameLineSnapshot,
    InjurySnapshot,
    OddsSnapshot,
    Player,
    ProviderEvent,
)
from nbaprops.pipeline import injuries, reference, run, schedule
from nbaprops.pipeline.odds import EventInfo, parse_events

pytestmark = pytest.mark.db
FIX = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)  # 10:00 AM ET, opening day


class FakeProvider:
    """Serves the fixture; `move` lets a test change prices between snapshots."""

    name = "propline"

    def __init__(self):
        self.payload = json.loads((FIX / "propline_event_odds.json").read_text())
        self.calls = 0
        self.last_quota = None

    def get_events(self) -> list[EventInfo]:
        p = self.payload
        return parse_events([{k: p[k] for k in ("id", "home_team", "away_team", "commence_time")}])

    def get_event_odds(self, event_id, closing=False):
        self.calls += 1
        return copy.deepcopy(self.payload)

    def move(self, book, market, desc, side, price=None, point=None):
        for b in self.payload["bookmakers"]:
            if b["key"] != book:
                continue
            for m in b["markets"]:
                if m["key"] != market or m.get("line_type", "main") != "main":
                    continue
                for o in m["outcomes"]:
                    if o.get("description") == desc and (side is None or o["name"] == side):
                        if price is not None:
                            o["price"] = price
                        if point is not None:
                            o["point"] = point


@pytest.fixture(scope="module")
def engine():
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("Postgres not reachable; start it with `make db-up`")
    return eng


@pytest.fixture
def db(engine, tmp_path, monkeypatch):
    tables = ", ".join(t.name for t in reversed(Base.metadata.sorted_tables))
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    with session_scope() as s:
        reference.seed_teams(s)
        reference.seed_players(s)
        schedule.upsert_games(
            s, schedule.parse_schedule(json.loads((FIX / "schedule.json").read_text()))
        )
    monkeypatch.setattr(run, "RAW_DIR", tmp_path / "raw")
    return tmp_path


def _count(model, *where):
    with session_scope() as s:
        return s.scalar(select(func.count()).select_from(model).where(*where))


def test_seed_counts(db):
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(Player)) > 5000


def test_started_games_are_skipped(db):
    provider = FakeProvider()
    tip = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)
    assert run.odds_snapshot(provider, tip).props_seen == 0
    assert provider.calls == 0  # no request spent on a game already under way


def test_first_snapshot_writes_everything_and_archives(db):
    provider = FakeProvider()
    result = run.odds_snapshot(provider, T0)
    # DK: Cade main/alt/milestone, Brown, Duren, Rookie = 6; FanDuel Cade main = 1
    assert result.props_written == 7
    assert result.lines_written == 5  # DK spread + total, Pinnacle moneyline + spread + total
    assert result.unmatched_players == {"Rookie Unknownson"}
    assert _count(ProviderEvent) == 1
    assert list((db / "raw" / "propline").rglob("*.json.gz"))
    with session_scope() as s:
        ms = s.scalars(select(OddsSnapshot).where(OddsSnapshot.line_type == "milestone")).one()
        assert float(ms.line) == 19.5 and ms.player_id == 1630595 and ms.game_id == "0022600001"


def test_unchanged_snapshot_writes_nothing(db):
    provider = FakeProvider()
    run.odds_snapshot(provider, T0)
    again = run.odds_snapshot(provider, T0 + timedelta(hours=3))
    assert again.props_seen == 7  # the game was really re-read, not skipped
    assert again.props_written == 0 and again.lines_written == 0
    assert _count(OddsSnapshot) == 7


def test_only_changed_props_are_written(db):
    provider = FakeProvider()
    run.odds_snapshot(provider, T0)
    provider.move("draftkings", "player_points", "Cade Cunningham", "Over", price=-130)
    provider.move("draftkings", "player_points", "Cade Cunningham", "Under", price=100)
    later = run.odds_snapshot(provider, T0 + timedelta(hours=3))
    assert later.props_written == 1
    with session_scope() as s:
        rows = s.scalars(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.player_id == 1630595,
                OddsSnapshot.book == "draftkings",
                OddsSnapshot.line_type == "main",
            )
            .order_by(OddsSnapshot.snapshot_ts)
        ).all()
        assert [(r.over_price, r.under_price) for r in rows] == [(-115, -105), (-130, 100)]


def test_main_line_move_is_one_change_on_one_key(db):
    provider = FakeProvider()
    run.odds_snapshot(provider, T0)
    provider.move("draftkings", "player_points", "Jaylen Brown (BOS)", None, point=25.5)
    later = run.odds_snapshot(provider, T0 + timedelta(hours=3))
    assert later.props_written == 1
    assert _count(OddsSnapshot, OddsSnapshot.player_id == 1627759) == 2


def test_an_unchanged_close_writes_nothing_but_is_recorded(db):
    """Closing ticks are change-only (storage): the board at tip is the latest row per prop,
    and the closing poll records that the close was taken."""
    provider = FakeProvider()
    run.odds_snapshot(provider, T0)
    before = _count(OddsSnapshot, OddsSnapshot.id.isnot(None))
    close_ts = T0 + timedelta(hours=4, minutes=45)
    run.odds_snapshot(provider, close_ts, {"0022600001"}, is_close=True)
    assert _count(OddsSnapshot, OddsSnapshot.id.isnot(None)) == before
    with session_scope() as s:
        assert run.odds_store.has_close(s, "0022600001")


def test_tick_takes_a_close_on_each_tick_before_tip(db, monkeypatch):
    provider = FakeProvider()
    tip = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)
    monkeypatch.setattr(run, "injury_snapshot", lambda now: 0)  # no network in tests
    assert run.tick(provider, tip - timedelta(minutes=30)).close_game_ids == ["0022600001"]
    provider.move("draftkings", "player_points", "Cade Cunningham", "Over", price=-140)
    assert run.tick(provider, tip - timedelta(minutes=15)).close_game_ids == ["0022600001"]
    # change-only closes: the first tick is the game's first snapshot (all 7 quotes are new);
    # the second writes only the one moved quote
    assert _count(OddsSnapshot, OddsSnapshot.is_close.is_(True)) == 7 + 1
    with session_scope() as s:
        assert run.odds_store.has_close(s, "0022600001")


def test_game_lines_keep_pinnacle_spread_and_total(db):
    run.odds_snapshot(FakeProvider(), T0)
    with session_scope() as s:
        pin = {
            r.market: r
            for r in s.scalars(select(GameLineSnapshot).where(GameLineSnapshot.book == "pinnacle"))
        }
        assert float(pin["spreads"].point) == -3.0
        assert float(pin["totals"].point) == 223.5
        assert pin["h2h"].home_price == -135


def test_injury_rows_resolve_players_and_are_idempotent(db):
    rows = injuries.parse_report_text((FIX / "injury_report_2025-12-19.txt").read_text())
    with session_scope() as s:
        first = injuries.store_report(s, rows, T0)
    with session_scope() as s:
        second = injuries.store_report(s, rows, T0 + timedelta(minutes=15))
    assert first == 17 and second == 0
    with session_scope() as s:
        tatum = s.scalars(
            select(InjurySnapshot).where(InjurySnapshot.player_name_raw == "Tatum, Jayson")
        ).one()
        assert tatum.player_id == 1628369 and tatum.status == "Out"
        harper = s.scalars(
            select(InjurySnapshot).where(InjurySnapshot.player_name_raw == "Harper Jr., Ron")
        ).one()
        assert harper.player_id == 1631199  # suffix handled
