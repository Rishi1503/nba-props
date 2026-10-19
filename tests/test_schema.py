"""Integration test against a real Postgres with migrations applied (alembic upgrade head)."""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

from nbaprops.db.base import get_engine, session_scope
from nbaprops.db.models import Base, Game, OddsSnapshot, Player, PlayerGameLog, Team

pytestmark = pytest.mark.db


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
def clean(engine):
    tables = ", ".join(t.name for t in reversed(Base.metadata.sorted_tables))
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    yield


def test_all_tables_exist(engine):
    existing = set(inspect(engine).get_table_names())
    expected = {t.name for t in Base.metadata.sorted_tables}
    assert expected <= existing, f"missing tables: {expected - existing}; run `make migrate`"


def _seed(s):
    """Insert parents before children: models declare no ORM relationships, so the
    session does not reorder inserts by foreign key. Pipelines upsert in this order too."""
    s.add_all(
        [
            Team(team_id=1610612738, abbreviation="BOS", name="Boston Celtics"),
            Team(team_id=1610612765, abbreviation="DET", name="Detroit Pistons"),
            Player(player_id=1628369, full_name="Jayson Tatum", position="F"),
        ]
    )
    s.flush()
    s.add(
        Game(
            game_id="0022600001",
            season="2026-27",
            season_type="regular",
            game_date=date(2026, 10, 20),
            home_team_id=1610612765,
            away_team_id=1610612738,
        )
    )
    s.flush()


def test_insert_and_read_box_score(clean):
    with session_scope() as s:
        _seed(s)
        s.add(
            PlayerGameLog(
                game_id="0022600001",
                player_id=1628369,
                team_id=1610612738,
                started=True,
                minutes=36.5,
                pts=28,
                reb=9,
                ast=6,
                fg3m=3,
            )
        )
    with session_scope() as s:
        row = s.get(PlayerGameLog, ("0022600001", 1628369))
        assert row.pts == 28 and row.started


def test_market_check_constraint_rejects_unknown_market(clean):
    with session_scope() as s:
        _seed(s)
    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(
                OddsSnapshot(
                    snapshot_ts=datetime.now(UTC),
                    provider="propline",
                    book="draftkings",
                    game_id="0022600001",
                    player_id=1628369,
                    player_name_raw="Jayson Tatum",
                    market="stl",
                    line=1.5,
                    over_price=-120,
                )
            )


def test_odds_row_needs_a_price(clean):
    with session_scope() as s:
        _seed(s)
    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(
                OddsSnapshot(
                    snapshot_ts=datetime.now(UTC),
                    provider="propline",
                    book="draftkings",
                    game_id="0022600001",
                    player_id=1628369,
                    player_name_raw="Jayson Tatum",
                    market="pts",
                    line=26.5,
                )
            )
