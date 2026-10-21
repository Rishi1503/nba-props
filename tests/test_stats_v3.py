"""V3 box scores through nba_api's real parser (the backfill crashed on these: nba_api's
get_normalized_dict() returns {} for V3 responses), and the fetcher's handling of empty
responses and empty cache files."""

import gzip
import json

import pytest
from nba_api.stats.endpoints._parsers import get_parser_for_endpoint

from nbaprops.pipeline import stats


def _player(pid, first, last, pos, mins, comment=""):
    return {
        "personId": pid,
        "firstName": first,
        "familyName": last,
        "nameI": "",
        "playerSlug": "",
        "position": pos,
        "comment": comment,
        "jerseyNum": "",
        "statistics": {"minutes": mins, "points": 10},
    }


def _team(tid, players):
    return {
        "teamId": tid,
        "teamCity": "",
        "teamName": "",
        "teamTricode": "",
        "teamSlug": "",
        "players": players,
        "statistics": {"minutes": "240:00"},
        "starters": {},
        "bench": {},
    }


RAW_V3 = {
    "boxScoreTraditional": {
        "gameId": "0022100001",
        "homeTeam": _team(
            1610612749,
            [
                _player(203507, "Giannis", "Antetokounmpo", "F", "PT32M10.00S"),
                _player(201572, "Brook", "Lopez", "", "PT18M00.00S"),
            ],
        ),
        "awayTeam": _team(
            1610612751, [_player(202681, "Kyrie", "Irving", "", "", "DNP - Coach's Decision")]
        ),
    }
}


class _Response:
    def __init__(self, raw):
        self.raw = raw

    def get_data_sets(self, endpoint):
        return get_parser_for_endpoint(endpoint, self.raw).get_data_sets()


def _fake_endpoint(raw, legacy=None, calls=None):
    class FakeV3:
        endpoint = "boxscoretraditionalv3"

        def __init__(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            self.nba_response = _Response(raw)

        def get_normalized_dict(self):
            return legacy or {}  # what nba_api returns for V3 responses: nothing

    return FakeV3


def test_a_v3_box_score_reads_starters_bench_and_dnp(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "_endpoint", lambda name: _fake_endpoint(RAW_V3))
    fetch = stats.StatsFetcher(cache_dir=tmp_path, pause=0)
    rows = fetch("boxscoretraditionalv3", "0022100001", game_id="0022100001")["PlayerStats"]
    started, dnp, names = stats.boxscore_to_updates(rows)
    assert started == {203507: True, 201572: False}  # a position means he started
    assert [(r["player_id"], r["dnp_reason"]) for r in dnp] == [(202681, "DNP - Coach's Decision")]
    assert names[203507] == "Giannis Antetokounmpo"
    cached = tmp_path / "boxscoretraditionalv3" / "0022100001.json.gz"
    with gzip.open(cached, "rt") as f:
        assert len(json.load(f)["PlayerStats"]) == 3  # the parsed rows are what's cached


def test_older_format_responses_pass_through(tmp_path, monkeypatch):
    legacy = {"LeagueGameLog": [{"GAME_ID": "1"}]}
    monkeypatch.setattr(stats, "_endpoint", lambda name: _fake_endpoint({}, legacy=legacy))
    fetch = stats.StatsFetcher(cache_dir=tmp_path, pause=0)
    assert fetch("leaguegamelog", "x") == legacy


def test_an_empty_cache_file_is_fetched_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stats, "_endpoint", lambda name: _fake_endpoint(RAW_V3, calls=calls))
    path = tmp_path / "boxscoretraditionalv3" / "0022100001.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump({}, f)  # what the crashed backfill left behind
    fetch = stats.StatsFetcher(cache_dir=tmp_path, pause=0)
    assert len(fetch("boxscoretraditionalv3", "0022100001")["PlayerStats"]) == 3
    assert len(calls) == 1


def test_an_empty_response_is_retried_and_never_cached(tmp_path, monkeypatch):
    empty = {
        "boxScoreTraditional": {"gameId": "x", "homeTeam": _team(1, []), "awayTeam": _team(2, [])}
    }
    calls = []

    class Empty(_fake_endpoint(empty, calls=calls)):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.nba_response = type("R", (), {"get_data_sets": lambda self, e: {}})()

    monkeypatch.setattr(stats, "_endpoint", lambda name: Empty)
    monkeypatch.setattr(stats.time, "sleep", lambda s: None)
    fetch = stats.StatsFetcher(cache_dir=tmp_path, pause=0, retries=3)
    with pytest.raises(RuntimeError, match="failed after 3 tries"):
        fetch("boxscoretraditionalv3", "x")
    assert len(calls) == 3
    assert not (tmp_path / "boxscoretraditionalv3" / "x.json.gz").exists()
