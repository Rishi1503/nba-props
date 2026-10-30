"""A full odds snapshot checks each player once, not once per prop (it took ~20 minutes)."""

from types import SimpleNamespace

from nbaprops.pipeline import odds_store
from nbaprops.pipeline.odds import PropQuote


def _quotes(players, per_player):
    return [
        PropQuote(book, pid, f"Player {pid}", market, "main", 10.5, -110, -110)
        for pid in players
        for book in ("draftkings", "fanduel")
        for market in ("pts", "reb", "ast", "fg3m", "pra")[:per_player]
    ]


def test_each_player_is_ensured_once_per_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(odds_store, "ensure_player", lambda s, pid, name: calls.append(pid))
    session = SimpleNamespace(info={})
    quotes = _quotes(range(50), 5) * 10  # 5,000 props, 50 players
    assert [odds_store._resolve_player(session, q) for q in quotes] == [
        q.nba_player_id for q in quotes
    ]
    assert sorted(calls) == list(range(50))
    odds_store._resolve_player(SimpleNamespace(info={}), quotes[0])  # a new snapshot checks again
    assert calls.count(0) == 2


def test_names_without_ids_are_looked_up_once(monkeypatch):
    lookups = []

    def by_name(s, name):
        lookups.append(name)
        return {"Jalen Brunson": 1628973}.get(name)

    monkeypatch.setattr(odds_store, "resolve_player_by_name", by_name)
    session = SimpleNamespace(info={})
    for name in ["Jalen Brunson", "Nobody Known"] * 20:
        q = PropQuote("draftkings", None, name, "pts", "main", 20.5, -110, -110)
        expected = 1628973 if name == "Jalen Brunson" else None
        assert odds_store._resolve_player(session, q) == expected
    assert sorted(lookups) == ["Jalen Brunson", "Nobody Known"]
