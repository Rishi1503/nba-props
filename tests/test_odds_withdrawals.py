"""Prop withdrawals (migration 0005): the change-only log records props leaving the board,
and the grading view takes one close per main line."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import OddsSnapshot, PlayerGameLog
from nbaprops.pipeline import odds_store
from nbaprops.pipeline.odds import PropQuote
from tests.test_revalidation import _reset

GAME = "0022600001"
CADE = 1630595
T0 = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)


@pytest.fixture
def db():
    _reset(with_schedule=True)


def q(book, line_type, line, over=-110, under=-110, market="pts"):
    return PropQuote(book, CADE, "Cade Cunningham", market, line_type, line, over, under)


def store(quotes, ts, close=False):
    with session_scope() as s:
        return odds_store.store_props(s, "propline", GAME, quotes, ts, is_close=close)


def rows():
    with session_scope() as s:
        return [
            (r.book, r.line_type, float(r.line), r.over_price, r.is_withdrawn, r.is_close)
            for r in s.scalars(select(OddsSnapshot).order_by(OddsSnapshot.id))
        ]


@pytest.mark.db
def test_a_prop_leaving_the_board_is_recorded(db):
    store(
        [
            q("draftkings", "main", 26.5),
            q("draftkings", "alternate", 29.5, 150, -180),
            q("fanduel", "main", 26.5),
        ],
        T0,
    )
    r = store(
        [q("draftkings", "main", 26.5), q("fanduel", "main", 26.5)], T0 + timedelta(minutes=15)
    )
    assert r.props_withdrawn == 1 and r.props_written == 0  # unchanged quotes: no rows
    assert rows()[-1] == ("draftkings", "alternate", 29.5, None, True, False)


@pytest.mark.db
def test_a_book_missing_from_a_snapshot_is_not_a_withdrawal(db):
    store([q("draftkings", "main", 26.5), q("fanduel", "main", 26.5)], T0)
    r = store([q("fanduel", "main", 26.5)], T0 + timedelta(minutes=15))  # DraftKings absent
    assert r.props_withdrawn == 0


@pytest.mark.db
def test_a_returning_prop_is_written_again_and_withdrawn_once(db):
    store([q("draftkings", "main", 26.5), q("draftkings", "alternate", 29.5)], T0)
    store([q("draftkings", "main", 26.5)], T0 + timedelta(minutes=15))
    r = store([q("draftkings", "main", 26.5)], T0 + timedelta(minutes=30))
    assert r.props_withdrawn == 0  # already off the board: not withdrawn twice
    r = store(
        [q("draftkings", "main", 26.5), q("draftkings", "alternate", 29.5)],
        T0 + timedelta(minutes=45),
    )
    assert r.props_written == 1 and rows()[-1][4] is False


@pytest.mark.db
def test_a_main_line_move_updates_the_prop_without_a_withdrawal(db):
    store([q("draftkings", "main", 26.5)], T0)
    r = store([q("draftkings", "main", 27.5)], T0 + timedelta(minutes=15))
    assert r.props_written == 1 and r.props_withdrawn == 0


@pytest.mark.db
def test_grading_view_takes_one_close_per_main_line_and_drops_withdrawn_props(db):
    with session_scope() as s:
        tip = s.execute(text(f"SELECT tip_time FROM games WHERE game_id = '{GAME}'")).scalar()
    # closing window: the main line moves 26.5 -> 27.5; the alternate is pulled before tip
    store(
        [q("draftkings", "main", 26.5), q("draftkings", "alternate", 29.5, 150, -180)],
        tip - timedelta(minutes=30),
        close=True,
    )
    store([q("draftkings", "main", 27.5, -120, -100)], tip - timedelta(minutes=10), close=True)
    with session_scope() as s:
        s.execute(text(f"UPDATE games SET status = 'final' WHERE game_id = '{GAME}'"))
        s.add(
            PlayerGameLog(
                game_id=GAME,
                player_id=CADE,
                team_id=1610612765,
                minutes=36,
                pts=27,
                reb=5,
                ast=9,
                fg3m=3,
            )
        )
    with session_scope() as s:
        graded = s.execute(
            text(
                "SELECT line_type, line, over_price, result FROM graded_props "
                f"WHERE book = 'draftkings' AND player_id = {CADE} AND market = 'pts'"
            )
        ).all()
    assert [(g.line_type, float(g.line), g.over_price, g.result) for g in graded] == [
        ("main", 27.5, -120, "under")
    ]  # one close: the final main line; no pulled alternate
