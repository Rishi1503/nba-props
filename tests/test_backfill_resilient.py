"""An unattended backfill against a stats site that blocks every few hundred requests:
blocks are waited out and the season resumes; skipped box scores are caught up at the end;
a site that never comes back ends the run with a message to run it again later."""

import pytest
from sqlalchemy import select

from nbaprops.db.base import session_scope
from nbaprops.db.models import Game
from nbaprops.pipeline import stats
from tests.test_revalidation import stats_db  # noqa: F401  (fixture)
from tests.test_stats import FakeFetch


def _loaded() -> dict[str, bool]:
    with session_scope() as s:
        return dict(s.execute(select(Game.game_id, Game.boxscore_loaded)).all())


@pytest.mark.db
def test_a_block_is_waited_out_and_the_season_resumes(stats_db, monkeypatch):  # noqa: F811
    monkeypatch.setattr(stats.time, "sleep", lambda s: None)
    fetch, blocked, waits = FakeFetch(), [2], []

    def site(endpoint, key, cache=True, **params):
        if endpoint == "leaguegamelog" and blocked[0]:
            blocked[0] -= 1
            raise RuntimeError(f"{endpoint} {key} failed after 4 tries")
        return fetch(endpoint, key, cache=cache, **params)

    out = stats.backfill_resilient(site, ["2025-26"], sleep=waits.append)
    assert waits == [stats.BLOCK_WAIT_SEC] * 2  # two blocks, each waited out
    assert "2025-26" in out
    assert _loaded() == {"0022500001": True, "0022500002": False}  # game 2's box is empty


@pytest.mark.db
def test_box_scores_skipped_mid_season_are_caught_up(stats_db, monkeypatch):  # noqa: F811
    cool_downs = []
    monkeypatch.setattr(stats.time, "sleep", cool_downs.append)
    fetch, failures, waits = FakeFetch(), [1], []

    def site(endpoint, key, cache=True, **params):
        if endpoint == "boxscoretraditionalv3" and key == "0022500001" and failures[0]:
            failures[0] -= 1
            raise RuntimeError(f"{endpoint} {key} failed after 4 tries")
        return fetch(endpoint, key, cache=cache, **params)

    stats.backfill_resilient(site, ["2025-26"], sleep=waits.append)
    assert _loaded()["0022500001"] is True  # skipped in the season pass, loaded in catch-up
    assert cool_downs == [stats.COOL_DOWN_SEC]  # the one skip paused briefly
    assert waits == []  # no long wait: game 2 came back empty, which isn't a block
    asked = [k for e, k in fetch.calls if e == "boxscoretraditionalv3"]
    assert asked.count("0022500002") == 2  # season pass + one catch-up try, then left


@pytest.mark.db
def test_a_site_that_never_comes_back_ends_the_run(stats_db, monkeypatch):  # noqa: F811
    monkeypatch.setattr(stats, "MAX_BLOCK_WAITS", 2)
    waits = []

    def refused(endpoint, key, cache=True, **params):
        raise RuntimeError(f"{endpoint} {key} failed after 4 tries")

    with pytest.raises(RuntimeError, match="run the backfill again later"):
        stats.backfill_resilient(refused, ["2025-26"], sleep=waits.append)
    assert waits == [stats.BLOCK_WAIT_SEC] * 2


def test_preseason_games_against_non_nba_teams_are_skipped():
    """2022-23's preseason had games against touring clubs (team 50009): they and all their
    rows are dropped before loading, instead of breaking the games table's team rule."""
    bos, nyk, club = 1610612738, 1610612752, 50009
    games = [
        {"game_id": "g1", "home_team_id": bos, "away_team_id": nyk},
        {"game_id": "g2", "home_team_id": bos, "away_team_id": club},
    ]
    team_logs = [
        {"game_id": "g1", "team_id": bos},
        {"game_id": "g1", "team_id": nyk},
        {"game_id": "g2", "team_id": bos},
        {"game_id": "g2", "team_id": club},
    ]
    players = [
        {"game_id": "g1", "player_id": 1},
        {"game_id": "g2", "player_id": 1},
        {"game_id": "g2", "player_id": 2},
    ]
    g, t, p = stats.nba_only(games, team_logs, players, known={bos, nyk})
    assert [x["game_id"] for x in g] == ["g1"]
    assert {x["game_id"] for x in t} == {"g1"} and len(t) == 2
    assert [x["game_id"] for x in p] == ["g1"]
