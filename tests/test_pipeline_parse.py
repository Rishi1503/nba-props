"""Pure parsing and scheduling tests (no database, no network)."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from nbaprops.pipeline import injuries, schedule
from nbaprops.pipeline.names import (
    last_first_to_first_last,
    normalize_name,
    parse_provider_player_id,
)
from nbaprops.pipeline.odds import parse_game_lines, parse_props, parse_quota
from nbaprops.pipeline.run import plan_tick

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def event_odds():
    return json.loads((FIX / "propline_event_odds.json").read_text())


# --- names -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, key",
    [
        ("Luka Dončić", "luka doncic"),
        ("Jaylen Brown (BOS)", "jaylen brown"),
        ("Ron Harper Jr.", "ron harper"),
        ("Gary Payton II", "gary payton"),
        ("Shai Gilgeous-Alexander", "shai gilgeous alexander"),
        ("De'Aaron Fox", "deaaron fox"),
    ],
)
def test_normalize_name(raw, key):
    assert normalize_name(raw) == key


def test_last_first_conversion():
    assert last_first_to_first_last("Harper Jr., Ron") == "Ron Harper Jr."
    assert normalize_name(last_first_to_first_last("Harper Jr., Ron")) == "ron harper"


def test_provider_player_id():
    assert parse_provider_player_id("nba:1630595") == 1630595
    assert parse_provider_player_id("mlb:592450") is None
    assert parse_provider_player_id(None) is None


# --- odds parsing ------------------------------------------------------------


def _find(quotes, **kw):
    return [q for q in quotes if all(getattr(q, k) == v for k, v in kw.items())]


def test_props_pair_over_and_under(event_odds):
    quotes = parse_props(event_odds)
    (cade,) = _find(
        quotes, book="draftkings", nba_player_id=1630595, market="pts", line_type="main"
    )
    assert (cade.line, cade.over_price, cade.under_price) == (26.5, -115, -105)


def test_team_suffix_stripped(event_odds):
    (jb,) = _find(parse_props(event_odds), book="draftkings", nba_player_id=1627759)
    assert jb.player_name_raw == "Jaylen Brown"


def test_alternate_line_kept_separately(event_odds):
    (alt,) = _find(parse_props(event_odds), line_type="alternate")
    assert (alt.line, alt.over_price, alt.under_price) == (29.5, 160, -210)


def test_milestone_normalized_to_half_point(event_odds):
    (ms,) = _find(parse_props(event_odds), line_type="milestone")
    assert ms.line == 19.5  # "20+" == over 19.5
    assert ms.over_price == -400 and ms.under_price is None


def test_player_without_nba_id_kept_by_name(event_odds):
    (rk,) = _find(parse_props(event_odds), player_name_raw="Rookie Unknownson")
    assert rk.nba_player_id is None and rk.market == "reb"


def test_unsupported_markets_and_periods_skipped(event_odds):
    quotes = parse_props(event_odds)
    assert not [q for q in quotes if q.line == 0.5]  # player_steals not tracked
    assert not _find(quotes, book="fanduel", line=7.5)  # first-quarter market skipped
    assert len(_find(quotes, book="fanduel")) == 1


def test_game_lines(event_odds):
    lines = parse_game_lines(event_odds)
    (pin_spread,) = _find(lines, book="pinnacle", market="spreads")
    assert (pin_spread.point, pin_spread.home_price, pin_spread.away_price) == (-3.0, -104, -106)
    (pin_total,) = _find(lines, book="pinnacle", market="totals")
    assert (pin_total.point, pin_total.home_price, pin_total.away_price) == (223.5, -103, -107)
    (ml,) = _find(lines, book="pinnacle", market="h2h")
    assert ml.point is None and ml.home_price == -135
    # DraftKings team total and Pinnacle alternate spread are excluded
    assert len(_find(lines, book="draftkings", market="totals")) == 1
    assert len(_find(lines, book="pinnacle", market="spreads")) == 1


def test_quota_headers():
    q = parse_quota({"X-Daily-Limit": "1000", "X-Daily-Used": "42", "X-Daily-Remaining": "958"})
    assert q.remaining == 958
    assert parse_quota({}) is None


# --- schedule ----------------------------------------------------------------


def test_schedule_parse():
    rows = schedule.parse_schedule(json.loads((FIX / "schedule.json").read_text()))
    ids = {r["game_id"]: r for r in rows}
    assert set(ids) == {"0012600005", "0022600001", "0022600003"}  # All-Star and TBD skipped
    assert ids["0012600005"]["season_type"] == "preseason"
    assert ids["0012600005"]["status"] == "final"
    assert ids["0022600001"]["season"] == "2026-27"
    # 01:30 UTC on Oct 21 is 9:30 PM ET on Oct 20
    assert ids["0022600003"]["game_date"] == date(2026, 10, 20)


# --- injury report -----------------------------------------------------------


@pytest.fixture(scope="module")
def report_rows():
    return injuries.parse_report_text((FIX / "injury_report_2025-12-19.txt").read_text())


def _row(rows, name):
    (r,) = [r for r in rows if r.player_name == name]
    return r


def test_report_carries_forward_context(report_rows):
    hauser = _row(report_rows, "Hauser, Sam")
    assert hauser.team_name == "Boston Celtics"
    assert (hauser.away_abbr, hauser.home_abbr) == ("MIA", "BOS")
    assert hauser.game_date == date(2025, 12, 19)
    assert hauser.status == "Probable"
    assert hauser.report_ts.hour == 7 and hauser.report_ts.minute == 30


def test_report_suffix_names_and_teams(report_rows):
    harper = _row(report_rows, "Harper Jr., Ron")
    assert harper.status == "Out" and harper.reason == "G League - Two-Way"
    embiid = _row(report_rows, "Embiid, Joel")
    assert embiid.team_name == "Philadelphia 76ers" and embiid.home_abbr == "NYK"


def test_report_wrapped_reason_joined(report_rows):
    mccain = _row(report_rows, "McCain, Jared")
    assert mccain.status == "Available"
    assert mccain.reason == "Injury/Illness - Right Thumb; Surgery Recovery - Splint"


def test_report_new_game_and_orphan_line(report_rows):
    collins = _row(report_rows, "Collins, Zach")
    assert (collins.away_abbr, collins.home_abbr, collins.team_name) == (
        "CHI",
        "CLE",
        "Chicago Bulls",
    )
    # The orphaned "Left Hamstring" reason is not glued onto Essengue's reason
    assert _row(report_rows, "Essengue, Noa").reason == "Injury/Illness - Left Shoulder; Surgery"
    assert _row(report_rows, "Mitchell, Donovan").status == "Questionable"
    assert len(report_rows) == 17


def test_report_url_candidates_newest_first():
    now = datetime(2026, 10, 20, 17, 40, tzinfo=UTC)  # 1:40 PM ET
    urls = injuries.candidate_urls(now, hours_back=1)
    assert urls[0].endswith("Injury-Report_2026-10-20_01_30PM.pdf")
    assert "Injury-Report_2026-10-20_01PM.pdf" in urls[2]
    assert urls[-1].endswith("Injury-Report_2026-10-20_12PM.pdf")


# --- tick scheduling ---------------------------------------------------------


def test_tick_full_snapshot_at_10am_et_in_october():
    now = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)  # EDT: 10:00 AM
    assert plan_tick(now, []).full_snapshot


def test_tick_survives_daylight_saving_change():
    # After Nov 1, 2026 New York is UTC-5: 10:00 AM ET is 15:00 UTC, not 14:00.
    assert not plan_tick(datetime(2026, 11, 2, 14, 0, tzinfo=UTC), []).full_snapshot
    assert plan_tick(datetime(2026, 11, 2, 15, 0, tzinfo=UTC), []).full_snapshot


def test_tick_idle_off_schedule():
    plan = plan_tick(datetime(2026, 10, 20, 15, 15, tzinfo=UTC), [])  # 11:15 AM ET
    assert plan.idle and not plan.injuries


def test_tick_closes_every_tick_in_the_last_35_minutes():
    tip = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)
    games = [("0022600001", tip), ("0022600002", tip + timedelta(hours=1))]
    # Ticks at 30 and 15 minutes before tip both take a close (a late GitHub run still gets one)
    assert plan_tick(tip - timedelta(minutes=30), games).close_game_ids == ["0022600001"]
    assert plan_tick(tip - timedelta(minutes=15), games).close_game_ids == ["0022600001"]
    assert plan_tick(tip - timedelta(minutes=45), games).close_game_ids == []
    # Never "close" a game that has started
    assert plan_tick(tip + timedelta(minutes=1), games).close_game_ids == []
    assert plan_tick(tip - timedelta(minutes=15), games).injuries


def test_pulled_markets_are_skipped_and_reported(event_odds):
    from nbaprops.pipeline.odds import pulled_players

    payload = json.loads(json.dumps(event_odds))
    dk = payload["bookmakers"][0]
    dk["markets"].append(
        {
            "key": "player_assists",
            "line_type": "main",
            "team": None,
            "period": None,
            "suspended_at": "2026-10-20T18:41:00Z",
            "outcomes": [
                {
                    "name": "Over",
                    "description": "Cade Cunningham",
                    "point": 8.5,
                    "price": -120,
                    "player_id": "nba:1630595",
                },
                {
                    "name": "Under",
                    "description": "Cade Cunningham",
                    "point": 8.5,
                    "price": -110,
                    "player_id": "nba:1630595",
                },
            ],
        }
    )
    assert not [q for q in parse_props(payload) if q.market == "ast"]
    assert pulled_players(payload) == {("draftkings", 1630595, "Cade Cunningham")}
    assert pulled_players(event_odds) == set()


def test_the_2026_report_format_with_squashed_team_names():
    """The 2026-27 report's text squashes the spaces out of team names (and after the comma in
    player names). Every row must still get its team, and no team name may stick to a player."""
    text = "\n".join(
        [
            "Injury Report: 10/21/26 05:00 PM",
            "Page 1 of 3",
            "Game Date Game Time Matchup Team Player Name Current Status Reason",
            "10/21/2026 07:00 (ET) PHI@MIN Philadelphia76ers Barlow,Dominick Out G League-Two-Way",
            "Embiid,Joel Questionable Injury/Illness-LeftKnee;Soreness",
            "MinnesotaTimberwolves DiVincenzo,Donte Out Injury/Illness-RightToe;Sprain",
            "10/21/2026 10:30 (ET) LAC@SAC LAClippers Harden,James Probable Rest",
            "LA Clippers Leonard,Kawhi Out Injury/Illness-RightKnee;Soreness",
        ]
    )
    rows = injuries.parse_report_text(text)
    got = [(r.team_name, r.player_name, r.status) for r in rows]
    assert got == [
        ("Philadelphia 76ers", "Barlow,Dominick", "Out"),
        ("Philadelphia 76ers", "Embiid,Joel", "Questionable"),
        ("Minnesota Timberwolves", "DiVincenzo,Donte", "Out"),
        ("Los Angeles Clippers", "Harden,James", "Probable"),
        ("Los Angeles Clippers", "Leonard,Kawhi", "Out"),
    ]
