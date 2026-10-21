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
        {"game_id": "g1", "player_id": 1, "team_id": bos},
        {"game_id": "g1", "player_id": 9, "team_id": club},  # a club guest in g1
        {"game_id": "g2", "player_id": 1, "team_id": bos},
        {"game_id": "g2", "player_id": 2, "team_id": club},
    ]
    g, t, p = stats.nba_only(games, team_logs, players, known={bos, nyk})
    assert [x["game_id"] for x in g] == ["g1"]
    assert {x["game_id"] for x in t} == {"g1"} and len(t) == 2
    assert [x["game_id"] for x in p] == ["g1"]
    assert [x["player_id"] for x in p] == [1]  # the club guest is dropped too
    # ... and only players who still have rows get registered as players
    names = {1: "Jayson Tatum", 2: None, 9: "Club Guest"}
    assert stats.names_in(p, names) == {1: "Jayson Tatum"}


@pytest.mark.db
def test_a_player_without_a_name_gets_a_placeholder(stats_db):  # noqa: F811
    from nbaprops.db.models import Player
    from nbaprops.pipeline.reference import ensure_player

    with session_scope() as s:
        ensure_player(s, 196294096, None)
        ensure_player(s, 196294097, "  ")
    with session_scope() as s:
        assert s.get(Player, 196294096).full_name == "Player 196294096"
        assert s.get(Player, 196294097).full_name == "Player 196294097"


@pytest.mark.db
def test_a_block_during_box_scores_resumes_the_same_season(stats_db, monkeypatch):  # noqa: F811
    """2022-23 stopped ~700 box scores short when the site blocked: the season must be
    resumed after the wait, not left to the end-of-run catch-up."""
    monkeypatch.setattr(stats.time, "sleep", lambda s: None)
    monkeypatch.setattr(stats, "MAX_FAILURES_IN_A_ROW", 1)
    fetch, blocked, waits = FakeFetch(), [1], []

    def site(endpoint, key, cache=True, **params):
        if endpoint == "boxscoretraditionalv3" and blocked[0]:
            blocked[0] -= 1
            raise RuntimeError(f"{endpoint} {key} failed after 4 tries")
        return fetch(endpoint, key, cache=cache, **params)

    stats.backfill_resilient(site, ["2025-26"], sleep=waits.append)
    assert waits == [stats.BLOCK_WAIT_SEC]
    assert _loaded()["0022500001"] is True


@pytest.mark.db
def test_waits_with_progress_between_them_dont_count_toward_the_limit(
    stats_db,  # noqa: F811
    monkeypatch,
):
    """A healthy multi-season run blocks every ~700 requests: only waits in a row with
    nothing loaded between them count toward giving up."""
    monkeypatch.setattr(stats.time, "sleep", lambda s: None)
    monkeypatch.setattr(stats, "MAX_FAILURES_IN_A_ROW", 1)
    monkeypatch.setattr(stats, "MAX_BLOCK_WAITS", 1)
    fetch, waits = FakeFetch(), []
    fail = {"leaguegamelog": 1, "0022500002": 1}

    def site(endpoint, key, cache=True, **params):
        k = endpoint if endpoint == "leaguegamelog" else key
        if fail.get(k):
            fail[k] -= 1
            raise RuntimeError(f"{endpoint} {key} failed after 4 tries")
        return fetch(endpoint, key, cache=cache, **params)

    stats.backfill_resilient(site, ["2025-26"], sleep=waits.append)  # doesn't give up
    assert waits == [stats.BLOCK_WAIT_SEC] * 2  # game 1 loaded between the two blocks
    assert _loaded()["0022500001"] is True


def test_garbled_minutes_are_never_read_as_zero():
    with pytest.raises(stats.BadMinutes):
        stats.parse_minutes("12:0-57")
    assert stats.parse_minutes("PT34M12.00S") == pytest.approx(34.2)
    row = {
        "personId": 7,
        "firstName": "A",
        "familyName": "B",
        "gameId": "g",
        "teamId": 1,
        "position": "",
        "comment": "",
        "minutes": "12:0-57",
    }
    started, dnp, _ = stats.boxscore_to_updates([row])
    assert started == {7: False} and dnp == []  # played (off the bench), not DNP
    assert stats.minutes_or_unknown("12:0-57", "game g") is None  # game logs: unknown


@pytest.mark.db
def test_the_roster_refresh_adds_players_not_yet_in_the_table(stats_db):  # noqa: F811
    """Rookies and new signings come in from the player index before their first game, so the
    injury report and DraftKings can be matched to them."""
    from nbaprops.db.models import Player

    was = 1610612764
    with session_scope() as s:
        stats.update_players_from_index(
            s,
            [
                {
                    "PERSON_ID": 1642900,
                    "PLAYER_FIRST_NAME": "AJ",
                    "PLAYER_LAST_NAME": "Dybantsa",
                    "TEAM_ID": was,
                    "POSITION": "F",
                },
            ],
            set_team=True,
        )
    with session_scope() as s:
        p = s.get(Player, 1642900)
        assert (p.full_name, p.current_team_id, p.position) == ("AJ Dybantsa", was, "F")
