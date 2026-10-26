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


def test_compact_forms_for_squashed_names():
    from nbaprops.pipeline.names import compact_forms

    assert compact_forms("dereck livelyii") == ["derecklivelyii", "derecklively"]
    assert compact_forms("yanickonan niederhauser") == ["yanickonanniederhauser"]


@pytest.mark.db
def test_squashed_report_names_match_their_players(stats_db):  # noqa: F811
    """The 2026-27 report squashes spaces inside names ("LivelyII,Dereck",
    "Niederhauser,YanicKonan"). Invented names, so the seeded real roster can't interfere
    (against it, "LivelyII,Dereck" resolves to the real Dereck Lively II, 1641726)."""
    from nbaprops.db.models import Player
    from nbaprops.pipeline.names import last_first_to_first_last
    from nbaprops.pipeline.reference import ensure_player, resolve_player_by_name

    with session_scope() as s:
        for pid, name in (
            (9001, "Zedd Quillon II"),
            (9002, "Orvo Thrask III"),
            (9003, "Yannick Konaz Velderhaus"),
            (9004, "Marqus Zmithe"),
            (9005, "Mar Qus Zmithe"),
        ):
            ensure_player(s, pid, name)
        s.add(Player(player_id=9006, full_name="Bennick Simzor", is_active=False))  # no alias
    with session_scope() as s:

        def resolve(raw):
            return resolve_player_by_name(s, last_first_to_first_last(raw))

        assert resolve("QuillonII,Zedd") == 9001
        assert resolve("ThraskIII,Orvo") == 9002
        assert resolve("Velderhaus,YannickKonaz") == 9003
        assert resolve("Usmithe,Marq") is None  # "marqusmithe" fits two players: no guess
        assert resolve("Simzor,Bennick") is None  # in the table, but never given an alias
        stats.update_players_from_index(
            s,
            [
                {
                    "PERSON_ID": 9006,
                    "PLAYER_FIRST_NAME": "Bennick",
                    "PLAYER_LAST_NAME": "Simzor",
                    "TEAM_ID": 1610612758,
                    "POSITION": "G",
                },
                {
                    "PERSON_ID": 9007,
                    "PLAYER_FIRST_NAME": "Ajax",
                    "PLAYER_LAST_NAME": "Dybanz",
                    "TEAM_ID": 1610612764,
                    "POSITION": "F",
                },
            ],
            set_team=True,
        )
        assert resolve("Simzor,Bennick") == 9006  # the roster refresh gave him his alias
        assert resolve("Dybanz,Ajax") == 9007  # a rookie, matchable before his first game


@pytest.mark.db
def test_stats_day_catches_up_missed_days(stats_db):  # noqa: F811
    """A morning without `stats-day` (or a blocked one) used to leave that day's games with no
    stats for good: the next run loads from the earliest day still missing."""
    from datetime import date

    from nbaprops.db.models import PlayerGameLog, TeamGameLog

    def game(gid, d):
        return Game(
            game_id=gid,
            season="2025-26",
            season_type="regular",
            game_date=d,
            tip_time=None,
            home_team_id=1610612738,
            away_team_id=1610612752,
            status="scheduled",
        )

    day = date(2026, 1, 10)
    with session_scope() as s:
        s.add_all(
            [
                game("x1", date(2026, 1, 7)),
                game("x2", date(2026, 1, 8)),
                game("x3", date(2026, 1, 9)),
            ]
        )
        s.flush()
        s.add(TeamGameLog(game_id="x2", team_id=1610612738))  # x2: game logs loaded
        # x1: only a box score's did-not-play row, still missing its game logs
        s.add(
            PlayerGameLog(
                game_id="x1",
                player_id=1628369,
                team_id=1610612738,
                minutes=0,
                dnp=True,
                dnp_reason="DND - Injury/Illness",
            )
        )
    assert stats.catch_up_start("2025-26", day) == date(2026, 1, 7)  # x1 and x3 have no stats
    with session_scope() as s:
        for gid in ("x1", "x3"):
            s.add(TeamGameLog(game_id=gid, team_id=1610612738))
    assert stats.catch_up_start("2025-26", day) == day  # nothing missing: just yesterday
