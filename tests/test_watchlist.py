"""Picks are recorded only near tip-off; earlier finds are a watchlist (record_window_min)."""

from datetime import timedelta

import pytest
from sqlalchemy import text

from nbaprops.db.base import session_scope
from nbaprops.pricing import slate
from tests.test_slate import CFG, DAY, G1, G3, NOW, TIP1, seed


def _cfg():
    # the shipped 2-hour window; odds age is tested elsewhere (the fixture's polls are old)
    return CFG.model_copy(update={"record_window_min": 120, "max_odds_age_min": 10**6})


@pytest.mark.db
def test_early_runs_make_a_watchlist_not_picks():
    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, _cfg())  # 8 hours before the first tip-off
    assert res.watchlist > 0 and len(res.picks) == 0
    assert any("watchlist" in m for m in res.skipped)
    with session_scope() as s:
        assert s.execute(text("SELECT count(*) FROM picks")).scalar() == 0
        logged = s.execute(text("SELECT count(*) FROM candidate_log WHERE flag")).scalar()
    assert logged >= res.watchlist  # the watchlist lives in the candidate log


@pytest.mark.db
def test_the_pre_tip_pass_records_the_near_games_picks_only():
    seed()
    with session_scope() as s:
        slate.price_slate(s, DAY, NOW, _cfg())  # the morning watchlist blocks nothing
        res = slate.price_slate(s, DAY, TIP1 - timedelta(minutes=60), _cfg())
    assert len(res.picks) > 0
    assert set(res.picks["game_id"]) == {G1}  # G3 tips hours later: still the watchlist
    with session_scope() as s:
        games = set(s.execute(text("SELECT DISTINCT game_id FROM picks")).scalars())
    assert games == {G1} and G3 not in games


def test_the_shipped_strategy_records_within_two_hours():
    from nbaprops.config import load_strategy
    from tests.conftest import SHIPPED_STRATEGY

    assert load_strategy(SHIPPED_STRATEGY).record_window_min == 120
