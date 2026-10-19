from datetime import date

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

from nbaprops.db.base import get_engine, session_scope
from nbaprops.db.models import Base, Game, Player, PlayerGameLog, TeamGameLog
from nbaprops.pipeline import reference, stats
from tests import stats_fixtures as fx

# --- pure --------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(34.5, 34.5), ("34:30", 34.5), ("PT34M30.00S", 34.5), ("", None), (None, None), ("0:00", 0)],
)
def test_parse_minutes(value, expected):
    assert stats.parse_minutes(value) == expected


@pytest.mark.parametrize("value", ["2025-10-21", "2025-10-21T00:00:00", "Oct 21, 2025"])
def test_parse_game_date(value):
    assert stats.parse_game_date(value) == date(2025, 10, 21)


def test_season_for_date():
    assert stats.season_for_date(date(2026, 10, 20)) == "2026-27"
    assert stats.season_for_date(date(2027, 4, 11)) == "2026-27"
    assert stats.season_for_date(date(2026, 9, 30)) == "2026-27"  # preseason camp


def test_team_rows_to_games():
    games, logs, issues = stats.team_logs_to_rows(fx.season_payload()["teams"], "regular")
    by_id = {g["game_id"]: g for g in games}
    assert set(by_id) == {"0022500001", "0022500002"}
    assert issues == ["0022500003: 1 team rows (expected 2)"]
    g1 = by_id["0022500001"]
    assert (g1["home_team_id"], g1["away_team_id"], g1["overtimes"]) == (fx.DET, fx.BOS, 0)
    assert by_id["0022500002"]["overtimes"] == 1
    assert g1["season"] == "2025-26" and g1["game_date"] == date(2025, 10, 21)
    det = next(t for t in logs if t["team_id"] == fx.DET)
    assert det["is_home"] and det["opp_team_id"] == fx.BOS and det["fga"] > 0


def test_validation_passes_on_consistent_data_and_catches_errors():
    payload = fx.season_payload()
    _, logs, _ = stats.team_logs_to_rows(payload["teams"], "regular")
    players, _ = stats.player_logs_to_rows(payload["players"])
    assert stats.validate(logs, players) == []
    players[0]["pts"] += 2
    issues = stats.validate(logs, players)
    assert len(issues) == 1 and "pts" in issues[0]


def test_possessions_known_value():
    t = {"fga": 88, "fta": 20, "oreb": 10, "dreb": 34, "fgm": 40, "tov": 12}
    o = {"fga": 85, "fta": 24, "oreb": 8, "dreb": 36, "fgm": 38, "tov": 14}
    assert stats.possessions(t, o) == pytest.approx(97.928, abs=1e-3)
    assert stats.possessions(o, t) == pytest.approx(stats.possessions(t, o))


def test_usage_rate_known_value():
    team = {"fga": 88, "fta": 20, "tov": 12, "minutes": 240}
    player = {"fga": 20, "fta": 6, "tov": 3, "minutes": 36}
    expected = (20 + 0.44 * 6 + 3) * 48 / (36 * (88 + 0.44 * 20 + 12))
    assert stats.usage_rate(player, team) == pytest.approx(expected)
    assert stats.usage_rate({**player, "minutes": 0}, team) is None


def test_boxscore_starters_and_dnp():
    started, dnp, names = stats.boxscore_to_updates(fx.boxscore_rows())
    assert sum(started.values()) == 10 and len(started) == 16
    assert {r["dnp_reason"] for r in dnp} == {"DNP - Injury/Illness", "DNP - Coach's Decision"}
    assert all(r["dnp"] and not r["started"] for r in dnp)


# --- database ----------------------------------------------------------------


class FakeFetch:
    """Serves fixtures by endpoint, honoring season type and date filters like the real site."""

    def __init__(self):
        self.payload = fx.season_payload()
        self.calls: list[tuple] = []

    def _in_range(self, rows, params):
        lo, hi = params.get("date_from_nullable"), params.get("date_to_nullable")
        if not lo:
            return rows
        lo_d = date(int(lo[6:]), int(lo[:2]), int(lo[3:5]))
        hi_d = date(int(hi[6:]), int(hi[:2]), int(hi[3:5]))
        return [r for r in rows if lo_d <= stats.parse_game_date(r["GAME_DATE"]) <= hi_d]

    def __call__(self, endpoint, cache_key, cache=True, **params):
        self.calls.append((endpoint, cache_key))
        if endpoint == "leaguegamelog":
            regular = params["season_type_all_star"] == "Regular Season"
            rows = self._in_range(self.payload["teams"], params) if regular else []
            return {"LeagueGameLog": rows}
        if endpoint == "playergamelogs":
            regular = params["season_type_nullable"] == "Regular Season"
            rows = self._in_range(self.payload["players"], params) if regular else []
            return {"PlayerGameLogs": rows}
        if endpoint == "playerindex":
            return {"PlayerIndex": [{"PERSON_ID": 1630595, "POSITION": "G"}]}
        if endpoint == "boxscoretraditionalv3":
            gid = params["game_id"]
            rows = fx.boxscore_rows(gid) if gid == "0022500001" else []
            return {"PlayerStats": rows}
        raise AssertionError(endpoint)


@pytest.fixture
def db():
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("Postgres not reachable")
    tables = ", ".join(t.name for t in reversed(Base.metadata.sorted_tables))
    with eng.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    with session_scope() as s:
        reference.seed_teams(s)
        reference.seed_players(s)


def _scalar(q):
    with session_scope() as s:
        return s.scalar(q)


@pytest.mark.db
def test_backfill_loads_games_logs_and_reports_issues(db):
    fetch = FakeFetch()
    issues = stats.backfill_season(fetch, "2025-26")
    assert any("0022500003" in i for i in issues)
    assert _scalar(select(func.count()).select_from(Game)) == 2
    assert _scalar(select(Game.overtimes).where(Game.game_id == "0022500002")) == 1
    assert _scalar(select(Game.home_score).where(Game.game_id == "0022500001")) > 0
    # 32 players who played + 2 DNP rows from the box score
    assert _scalar(select(func.count()).select_from(PlayerGameLog)) == 34
    assert _scalar(select(Player.position).where(Player.player_id == 1630595)) == "G"
    assert _scalar(select(Player.full_name).where(Player.player_id == 1642999)) == (
        "Rookie Unlisted"
    )  # unknown id inserted as a placeholder


@pytest.mark.db
def test_sql_possessions_match_python_formula(db):
    stats.backfill_season(FakeFetch(), "2025-26", boxscores=False)
    with session_scope() as s:
        logs = {
            t.team_id: t
            for t in s.scalars(select(TeamGameLog).where(TeamGameLog.game_id == "0022500001"))
        }
        d, b = logs[fx.DET], logs[fx.BOS]
        as_dict = lambda t: {k: getattr(t, k) for k in ("fga", "fta", "oreb", "dreb", "fgm", "tov")}  # noqa: E731
        expected = stats.possessions(as_dict(d), as_dict(b))
        assert float(d.possessions) == pytest.approx(expected, abs=0.01)
        assert float(d.possessions) == pytest.approx(float(b.possessions), abs=0.01)
        assert float(d.pace) == pytest.approx(48 * expected / 48, abs=0.01)  # 240 min -> /48
        assert float(d.off_rtg) == pytest.approx(100 * d.pts / expected, abs=0.01)
        assert float(d.def_rtg) == pytest.approx(100 * b.pts / expected, abs=0.01)


@pytest.mark.db
def test_usage_invariant_minutes_weighted_average_is_twenty_percent(db):
    stats.backfill_season(FakeFetch(), "2025-26", boxscores=False)
    with session_scope() as s:
        for game_id in ("0022500001", "0022500002"):  # regulation and overtime
            rows = s.execute(
                select(PlayerGameLog.team_id, PlayerGameLog.minutes, PlayerGameLog.usage_pct).where(
                    PlayerGameLog.game_id == game_id, PlayerGameLog.minutes > 0
                )
            ).all()
            by_team: dict[int, list] = {}
            for team, mins, usg in rows:
                by_team.setdefault(team, []).append((float(mins), float(usg)))
            assert len(by_team) == 2 and all(len(v) == 8 for v in by_team.values())
            for team_rows in by_team.values():
                total = sum(m for m, _ in team_rows)
                weighted = sum(m * u for m, u in team_rows) / total
                assert weighted == pytest.approx(0.20, abs=0.002)


@pytest.mark.db
def test_boxscore_sets_starters_dnp_and_is_not_refetched(db):
    fetch = FakeFetch()
    stats.backfill_season(fetch, "2025-26")
    with session_scope() as s:
        starters = s.scalar(
            select(func.count())
            .select_from(PlayerGameLog)
            .where(PlayerGameLog.game_id == "0022500001", PlayerGameLog.started.is_(True))
        )
        assert starters == 10
        dnp = s.scalars(select(PlayerGameLog).where(PlayerGameLog.dnp.is_(True))).all()
        assert {r.dnp_reason for r in dnp} == {"DNP - Injury/Illness", "DNP - Coach's Decision"}
        assert s.scalar(select(Game.boxscore_loaded).where(Game.game_id == "0022500001"))
    box_calls = sum(1 for e, _ in fetch.calls if e == "boxscoretraditionalv3")
    stats.backfill_season(fetch, "2025-26")
    # The loaded game is not fetched again; only game 2 (empty box score) is retried.
    assert sum(1 for e, _ in fetch.calls if e == "boxscoretraditionalv3") == box_calls + 1
    assert _scalar(select(func.count()).select_from(PlayerGameLog)) == 34  # idempotent


@pytest.mark.db
def test_box_score_never_marks_a_player_who_played_as_dnp(db):
    stats.backfill_season(FakeFetch(), "2025-26", boxscores=False)
    rows = fx.boxscore_rows()
    rows[0] = {**rows[0], "minutes": ""}  # box says Cade had no minutes; logs say he played
    with session_scope() as s:
        stats.apply_boxscore(s, "0022500001", rows)
    assert _scalar(select(PlayerGameLog.dnp).where(PlayerGameLog.player_id == 1630595)) is False


@pytest.mark.db
def test_pull_day_loads_only_that_day(db):
    fetch = FakeFetch()
    stats.pull_day(fetch, date(2025, 10, 22))
    assert _scalar(select(func.count()).select_from(Game)) == 1
    assert _scalar(select(Game.game_id)) == "0022500002"
    keys = [k for e, k in fetch.calls if e == "leaguegamelog"]
    assert "2025-26_regular_20251022_20251022_T" in keys
