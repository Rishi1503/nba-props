"""One test (or more) per gap found in the Stage 0-3 revalidation."""

import gzip
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import func, select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import (
    Game,
    OddsSnapshot,
    PlayerGameLog,
    UnmatchedName,
)
from nbaprops.features import build as fb
from nbaprops.pipeline import reference, run, stats
from tests import feature_fixtures as ff
from tests import stats_fixtures as sfx
from tests.test_pipeline_store import FIX, T0, FakeProvider
from tests.test_stats import FakeFetch


def _reset(with_schedule: bool) -> None:
    from sqlalchemy.exc import OperationalError

    from nbaprops.db.base import get_engine
    from nbaprops.db.models import Base
    from nbaprops.pipeline import schedule

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
        if with_schedule:
            payload = json.loads((FIX / "schedule.json").read_text())
            schedule.upsert_games(s, schedule.parse_schedule(payload))


@pytest.fixture
def stats_db():
    _reset(with_schedule=False)


@pytest.fixture
def store_db(tmp_path, monkeypatch):
    _reset(with_schedule=True)
    monkeypatch.setattr(run, "RAW_DIR", tmp_path / "raw")


# --- Stage 1 -----------------------------------------------------------------


@pytest.mark.db
def test_started_is_unknown_until_box_score_and_survives_reloads(stats_db):
    stats.backfill_season(FakeFetch(), "2025-26", boxscores=False)
    with session_scope() as s:
        assert s.scalar(select(func.count()).where(PlayerGameLog.started.is_not(None))) == 0
    stats.load_boxscores(FakeFetch(), ["0022500001"])
    stats.backfill_season(FakeFetch(), "2025-26", boxscores=False)  # logs reloaded
    with session_scope() as s:
        starters = s.scalar(
            select(func.count()).where(
                PlayerGameLog.game_id == "0022500001", PlayerGameLog.started.is_(True)
            )
        )
    assert starters == 10  # the reload did not wipe the box-score starters


@pytest.mark.db
def test_empty_box_score_is_retried_not_marked_loaded(stats_db):
    fetch = FakeFetch()
    stats.backfill_season(fetch, "2025-26")  # game 2's box score is empty in the fixture
    with session_scope() as s:
        loaded = dict(s.execute(select(Game.game_id, Game.boxscore_loaded)).all())
    assert loaded == {"0022500001": True, "0022500002": False}
    before = sum(1 for e, _ in fetch.calls if e == "boxscoretraditionalv3")
    stats.load_boxscores(fetch, ["0022500001", "0022500002"])
    after = sum(1 for e, _ in fetch.calls if e == "boxscoretraditionalv3")
    assert after - before == 1  # only the empty one is asked for again


def test_daily_pulls_are_never_cached():
    calls = []

    def fetch(endpoint, key, cache=True, **params):
        calls.append((endpoint, cache))
        return {"LeagueGameLog": []}

    stats.load_period(fetch, "2025-26", "Regular Season", date(2025, 10, 22), date(2025, 10, 22))
    assert calls == [("leaguegamelog", False)]


def test_refresh_bypasses_cache(tmp_path, monkeypatch):
    class Endpoint:
        n = 0

        def __init__(self, **kw):
            Endpoint.n += 1

        def get_normalized_dict(self):
            return {"X": [{"n": Endpoint.n}]}

    monkeypatch.setattr(stats, "_endpoint", lambda name: Endpoint)
    monkeypatch.setattr(stats.time, "sleep", lambda s: None)
    stats.StatsFetcher(cache_dir=tmp_path)("x", "k")
    assert stats.StatsFetcher(cache_dir=tmp_path)("x", "k") == {"X": [{"n": 1}]}  # cached
    assert stats.StatsFetcher(cache_dir=tmp_path, refresh=True)("x", "k") == {"X": [{"n": 2}]}
    with gzip.open(tmp_path / "x" / "k.json.gz", "rt") as f:
        assert json.load(f) == {"X": [{"n": 2}]}  # refreshed copy replaces the old one


# --- Stage 2: review queue, match rate, grading -------------------------------


@pytest.mark.db
def test_unmatched_names_are_queued_and_alias_add_backfills(store_db):
    provider = FakeProvider()
    r1 = run.odds_snapshot(provider, T0)
    assert r1.match_rate == pytest.approx(6 / 7)
    run.odds_snapshot(provider, T0 + timedelta(hours=1))
    with session_scope() as s:
        q = s.scalars(select(UnmatchedName)).one()
        assert q.name_raw == "Rookie Unknownson" and q.seen_count == 2
        reference.ensure_player(s, 1642000, "Rookie Unknownson")
    with session_scope() as s:
        fixed = reference.add_alias(s, "Rookie Unknownson", 1642000)
    assert fixed == 1
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(UnmatchedName)) == 0
        assert s.scalar(select(func.count()).where(OddsSnapshot.player_id.is_(None))) == 0
    r3 = run.odds_snapshot(FakeProvider(), T0 + timedelta(hours=2))
    assert r3.match_rate == 1.0


@pytest.mark.db
def test_suggestions_offer_close_spellings(store_db):
    with session_scope() as s:
        hints = reference.suggest_players(s, "Nicolas Claxton")
    assert any("Claxton" in name for _, name in hints)


@pytest.mark.db
def test_graded_props_view_uses_latest_close_and_grades(store_db):
    provider = FakeProvider()
    tip = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)
    run.odds_snapshot(provider, tip - timedelta(minutes=30), {"0022600001"}, is_close=True)
    provider.move("draftkings", "player_points", "Cade Cunningham", "Over", price=-150)
    run.odds_snapshot(provider, tip - timedelta(minutes=10), {"0022600001"}, is_close=True)
    with session_scope() as s:
        s.execute(text("UPDATE games SET status = 'final' WHERE game_id = '0022600001'"))
        s.add_all(
            [
                PlayerGameLog(
                    game_id="0022600001",
                    player_id=1630595,
                    team_id=1610612765,
                    minutes=36,
                    pts=31,
                    reb=5,
                    ast=9,
                    fg3m=3,
                ),  # Cade: 31 pts
                PlayerGameLog(
                    game_id="0022600001",
                    player_id=1627759,
                    team_id=1610612738,
                    minutes=34,
                    pts=24,
                    reb=6,
                    ast=4,
                    fg3m=2,
                ),  # Brown: 24 pts
                PlayerGameLog(
                    game_id="0022600001", player_id=1631105, team_id=1610612765, dnp=True, minutes=0
                ),  # Duren: did not play
            ]
        )
    with session_scope() as s:
        rows = {
            (r.player_id, r.line_type, float(r.line)): r
            for r in s.execute(text("SELECT * FROM graded_props WHERE book = 'draftkings'"))
        }
    cade = rows[(1630595, "main", 26.5)]
    assert cade.result == "over" and cade.actual == 31 and cade.over_price == -150  # latest
    assert rows[(1630595, "milestone", 19.5)].result == "over"
    assert rows[(1630595, "alternate", 29.5)].result == "over"
    assert rows[(1627759, "main", 24.5)].result == "under"
    assert rows[(1631105, "main", 10.5)].result == "void"


# --- Stage 3: injury reports, slate coverage, seeding signals ----------------


def _report(ts, rows):
    return [
        {"report_ts": ts, "game_id": "G1", "team_id": 1, "player_id": p, "status": st}
        for p, st in rows
    ]


def test_player_cleared_in_a_later_report_is_no_longer_out():
    t1 = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
    t2 = t1 + timedelta(hours=2)
    inj = pd.DataFrame(_report(t1, [(10, "Out"), (11, "Out")]) + _report(t2, [(11, "Out")]))
    now = fb.latest_statuses(inj, t2 + timedelta(minutes=1))
    assert list(now["player_id"]) == [11]  # player 10 was cleared
    earlier = fb.latest_statuses(inj, t1 + timedelta(minutes=1))
    assert sorted(earlier["player_id"]) == [10, 11]


def test_slate_includes_star_back_from_long_absence():
    inp = ff.build()
    out = (
        (inp.logs.player_id == ff.STAR)
        & (inp.logs.season == "2025-26")
        & (inp.logs.season_type == "regular")
        & (inp.logs.game_date >= date(2025, 11, 1))
    )
    inp.logs.loc[out, ["dnp", "minutes", "pts"]] = [True, 0, 0]
    day = ff.future_date()
    slate = fb.slate_frame(inp, day, datetime(day.year, day.month, day.day, 18, tzinfo=UTC))
    assert ff.STAR in set(slate.player_id)  # established, though outside the recent rotation
    mates = slate[(slate.team_id == ff.A) & (slate.player_id != ff.STAR)]
    assert (mates["absent_n"] == 0).all()  # his absence is already in teammates' recent form


def test_slate_accepts_players_with_lines():
    inp = ff.build()
    day = ff.future_date()
    slate = fb.slate_frame(
        inp,
        day,
        datetime(day.year, day.month, day.day, 18, tzinfo=UTC),
        extra_players=[(123456, ff.D)],
    )
    row = slate[slate.player_id == 123456]
    assert len(row) == 1 and row.iloc[0]["team_id"] == ff.D


def test_clinched_and_eliminated_flags():
    east = [
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DET",
        "IND",
        "MIA",
        "MIL",
        "NYK",
        "ORL",
        "PHI",
        "TOR",
        "WAS",
    ]
    teams = pd.DataFrame(
        [{"team_id": i + 1, "abbreviation": a} for i, a in enumerate(east)]
        + [{"team_id": 99, "abbreviation": "LAL"}]
    )
    games, gid = [], 0
    for i in range(15):
        wins = 70 - 5 * i  # 70, 65, ..., 0 after 78 games: 4 left each
        for k in range(78):
            gid += 1
            games.append(
                {
                    "game_id": f"g{gid}",
                    "season": "2025-26",
                    "season_type": "regular",
                    "game_date": pd.Timestamp(date(2025, 10, 22) + timedelta(days=k)),
                    "home_team_id": i + 1,
                    "away_team_id": 99,
                    "home_score": 110 if k < wins else 90,
                    "away_score": 100,
                    "status": "final",
                }
            )
    target = pd.DataFrame({"season": ["2025-26"], "game_date": [pd.Timestamp(date(2026, 4, 1))]})
    st = fb.standings(pd.DataFrame(games), teams, target).set_index("team_id")
    east_ids = list(range(1, 16))
    st = st.loc[east_ids].sort_values("conf_rank")
    assert list(st["clinched_top6"]) == [1] * 6 + [0] * 9  # 6th has 45 wins > 7th's max 44
    assert list(st["eliminated"]) == [0] * 10 + [1] * 5  # 11th: 20 + 4 < 10th's 25
    assert (st["games_remaining"] == 4).all()


def test_features_include_seeding_signals():
    df = fb.training_frame(ff.build())
    assert {"clinched_top6", "eliminated"} <= set(df.columns)
    assert df["clinched_top6"].isin([0, 1]).all()


def test_box_fixture_sanity():
    assert len(sfx.boxscore_rows()) == 18
