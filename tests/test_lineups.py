"""Confirmed starters before tip-off: the picks a lineup change puts in doubt."""

from datetime import timedelta

import pytest
from sqlalchemy import text

from nbaprops.db.base import session_scope
from nbaprops.ops import alerts
from nbaprops.pipeline import lineups
from nbaprops.pricing import slate
from tests.test_slate import BROWN, CADE, CFG, DAY, G1, NOW, TIP1, seed


def _team(pid, starter, n=0, status="ACTIVE"):
    return {"personId": pid, "starter": "1" if starter else "0", "status": status, "order": n}


def _box(home_id, home, away_id, away):
    return {
        "game": {
            "gameId": G1,
            "homeTeam": {"teamId": home_id, "players": home},
            "awayTeam": {"teamId": away_id, "players": away},
        }
    }


def test_a_lineup_counts_only_once_both_teams_list_five_starters():
    five = [_team(900 + i, True) for i in range(5)]
    box = _box(
        1,
        five + [_team(10, False), _team(11, False, status="INACTIVE")],
        2,
        [_team(800 + i, True) for i in range(5)],
    )
    lineup = lineups.parse_lineup(box)
    assert set(lineup) == {1, 2} and sum(lineup[1].values()) == 5
    assert 10 in lineup[1] and 11 not in lineup[1]  # inactive players aren't in the lineup
    early = _box(1, five[:4], 2, [_team(800 + i, True) for i in range(5)])
    assert lineups.parse_lineup(early) is None  # not published yet: do nothing
    assert lineups.parse_lineup({}) is None
    assert lineups.parse_lineup(
        _box(1, [{**p, "starter": True} for p in five], 2, [_team(800 + i, True) for i in range(5)])
    )  # true, too


def _history():
    """Ten earlier games: Cade started every one, Brown only two."""
    with session_scope() as s:
        home, away = s.execute(
            text("SELECT home_team_id, away_team_id FROM games WHERE game_id = :g"), {"g": G1}
        ).one()
        teams = {CADE: home, BROWN: away}
        for i in range(10):
            gid = f"00226099{i:02d}"
            s.execute(
                text("""INSERT INTO games (game_id, season, season_type, game_date, tip_time,
                                  home_team_id, away_team_id, status, overtimes)
                              SELECT :g, season, season_type, :d, tip_time - :back, home_team_id,
                                  away_team_id, 'final', 0 FROM games WHERE game_id = :g1"""),
                {
                    "g": gid,
                    "d": DAY - timedelta(days=12 - i),
                    "back": timedelta(days=12 - i),
                    "g1": G1,
                },
            )
            for pid, started in ((CADE, True), (BROWN, i < 2)):
                s.execute(
                    text("""INSERT INTO player_game_logs (game_id, player_id, team_id, dnp, started)
                                  VALUES (:g, :p, :t, false, :s)"""),
                    {"g": gid, "p": pid, "t": teams[pid], "s": started},
                )
    return teams


@pytest.mark.db
def test_lineup_changes_alert_once_tag_picks_and_hold_back_new_ones():
    seed()
    teams = _history()
    with session_scope() as s:
        slate.price_slate(s, DAY, NOW, CFG)
        s.execute(
            text("UPDATE picks SET stake = 5 WHERE player_id IN (:a, :b)"), {"a": CADE, "b": BROWN}
        )
        row = s.execute(text("SELECT * FROM picks WHERE player_id = :c LIMIT 1"), {"c": CADE})
        d = dict(row.mappings().first())  # a second open pick, on Brown, copied from Cade's
        d.pop("pick_id")
        d.update(player_id=BROWN, side="under", context=None)
        s.execute(
            text(
                f"INSERT INTO picks ({', '.join(d)}) VALUES ({', '.join(':' + k for k in d)}) "
                "ON CONFLICT DO NOTHING"
            ),
            d,
        )
        before = s.execute(
            text("SELECT count(*) FROM picks WHERE player_id IN (:a, :b)"), {"a": CADE, "b": BROWN}
        ).scalar()
    assert before >= 1
    box = _box(
        teams[CADE],
        [_team(CADE, False)] + [_team(950 + i, True) for i in range(5)],
        teams[BROWN],
        [_team(BROWN, True)] + [_team(960 + i, True) for i in range(4)],
    )
    rec = alerts.RecordingNotifier()
    at = TIP1 - timedelta(minutes=40)
    with session_scope() as s:
        changes = lineups.check(s, at, rec, fetch=lambda gid: box)
    got = {(c["player_id"], c["change"]) for c in changes}
    assert got == {(CADE, "not starting"), (BROWN, "now starting")}
    with session_scope() as s:
        tags = dict(
            s.execute(
                text("""SELECT player_id, context->>'lineup' FROM picks
                                      WHERE player_id IN (:a, :b)"""),
                {"a": CADE, "b": BROWN},
            ).all()
        )
    assert set(tags.values()) <= {"not starting", "now starting"} and tags
    assert len(rec.sent) == len(tags) and all("Lineup change" in m for m in rec.sent)
    with session_scope() as s:  # a later cycle: no repeat alerts, no refetch needed
        lineups.check(s, at + timedelta(minutes=30), rec, fetch=lambda gid: None)
    assert len(rec.sent) == len(tags)
    with session_scope() as s:  # pricing now holds back new picks on both players
        res = slate.price_slate(s, DAY, at, CFG, write=False)
    if len(res.picks):
        assert not res.picks["player_id"].astype(int).isin([CADE, BROWN]).any()


@pytest.mark.db
def test_nothing_happens_before_the_lineup_is_published():
    seed()
    _history()
    rec = alerts.RecordingNotifier()
    with session_scope() as s:
        assert lineups.check(s, TIP1 - timedelta(minutes=40), rec, fetch=lambda gid: None) == []
        assert s.execute(text("SELECT count(*) FROM lineup_snapshots")).scalar() == 0
    assert rec.sent == []
