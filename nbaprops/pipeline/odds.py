"""Odds provider adapter (Stage 2). One interface, one class per provider.

PropLine response shape (from its SDK, v0.63):
  GET /v1/sports/basketball_nba/events -> [{id, home_team, away_team, commence_time}, ...]
  GET /v1/sports/basketball_nba/events/{id}/odds?markets=...&bookmakers=...
    -> {id, home_team, away_team, commence_time,
        bookmakers: [{key, title, markets: [{key, line_type, team, outcomes: [
            {name, description, point, price, player_id, side, last_change_at}]}]}]}
  Quota headers on every response: X-Daily-Limit / -Used / -Remaining / -Reset.

Pulled markets: a market with `suspended_at` set has been taken off the board (often a late
scratch). Its last prices are stale, so `parse_props` skips it and `pulled_players` reports it.

Coverage note: PropLine carries NBA player props for DraftKings, FanDuel, Bovada and
Unibet, but Pinnacle only for NBA game lines. The prop reference price is therefore a
consensus of the non-DraftKings books; Pinnacle feeds spreads and totals.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import requests

from nbaprops.pipeline.names import normalize_name, parse_provider_player_id, strip_team_suffix

SPORT = "basketball_nba"
BOOKS = ["draftkings", "fanduel", "bovada", "unibet", "pinnacle"]
PROP_MARKETS = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra",
}
GAME_MARKETS = ["h2h", "spreads", "totals"]


@dataclass(frozen=True)
class EventInfo:
    provider_event_id: str
    home_team: str
    away_team: str
    commence_time: datetime


@dataclass(frozen=True)
class PropQuote:
    book: str
    nba_player_id: int | None
    player_name_raw: str
    market: str  # pts / reb / ast / fg3m / pra
    line_type: str  # main / alternate / milestone
    line: float  # milestone "20+" normalized to 19.5
    over_price: int | None
    under_price: int | None


@dataclass(frozen=True)
class GameLineQuote:
    book: str
    market: str  # h2h / spreads / totals
    point: float | None  # home handicap for spreads, total for totals
    home_price: int | None  # totals: over
    away_price: int | None  # totals: under


@dataclass(frozen=True)
class Quota:
    limit: int
    used: int
    remaining: int


class OddsProvider(Protocol):
    name: str

    def get_events(self) -> list[EventInfo]: ...

    def get_event_odds(self, event_id: str, closing: bool = False) -> dict[str, Any]: ...


# --- Parsing (pure) ----------------------------------------------------------


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_events(payload: list[dict[str, Any]]) -> list[EventInfo]:
    return [
        EventInfo(
            provider_event_id=str(e["id"]),
            home_team=e["home_team"],
            away_team=e["away_team"],
            commence_time=parse_time(e["commence_time"]),
        )
        for e in payload
        if not e.get("is_outright")
    ]


_MILESTONE_NAME = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+")


def _milestone_point(outcome: dict[str, Any]) -> float | None:
    """DraftKings (and FanDuel, Unibet) send milestone rungs with an empty point and the
    threshold in the name ("20+ Points", "3+ Assists"): N+ is the point N."""
    m = _MILESTONE_NAME.match(outcome.get("name") or "")
    return float(m.group(1)) if m else None


def _normalize_line(point: float, line_type: str) -> float:
    """A milestone "N+" (point N, integer) is the same bet as over N - 0.5."""
    if line_type == "milestone" and float(point).is_integer():
        return float(point) - 0.5
    return float(point)


def parse_props(payload: dict[str, Any]) -> list[PropQuote]:
    """Pair Over/Under outcomes into one quote per (book, player, market, line_type, line)."""
    sides: dict[tuple, dict[str, Any]] = defaultdict(dict)
    meta: dict[tuple, tuple[int | None, str]] = {}

    for book in payload.get("bookmakers", []):
        book_key = book["key"]
        for market in book.get("markets", []):
            ours = PROP_MARKETS.get(market.get("key"))
            if ours is None or market.get("period") not in (None, "full", "game"):
                continue
            if market.get("suspended_at"):
                continue  # off the board: last prices are stale
            line_type = market.get("line_type") or "main"
            for o in market.get("outcomes", []):
                point = o.get("point")
                if point is None and line_type == "milestone":
                    point = _milestone_point(o)
                if point is None or o.get("price") is None:
                    continue
                name = strip_team_suffix(o.get("description") or "")
                if not name:
                    continue
                pid = parse_provider_player_id(o.get("player_id"))
                player_key = pid if pid is not None else normalize_name(name)
                line = _normalize_line(point, line_type)
                key = (book_key, player_key, ours, line_type, line)
                side = (o.get("name") or "").lower()
                if side not in ("over", "under"):
                    side = "over"  # milestone rungs are listed as a single "N+" selection
                sides[key][side] = int(o["price"])
                meta[key] = (pid, name)

    quotes = []
    for key, prices in sides.items():
        book_key, _, market, line_type, line = key
        pid, name = meta[key]
        quotes.append(
            PropQuote(
                book=book_key,
                nba_player_id=pid,
                player_name_raw=name,
                market=market,
                line_type=line_type,
                line=line,
                over_price=prices.get("over"),
                under_price=prices.get("under"),
            )
        )
    return quotes


def pulled_players(payload: dict[str, Any]) -> set[tuple[str, int | None, str]]:
    """(book, nba_player_id, name) for every player whose prop market a book has pulled."""
    pulled = set()
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") not in PROP_MARKETS or not market.get("suspended_at"):
                continue
            for o in market.get("outcomes", []):
                name = strip_team_suffix(o.get("description") or "")
                if name:
                    pulled.add((book["key"], parse_provider_player_id(o.get("player_id")), name))
    return pulled


def parse_game_lines(payload: dict[str, Any]) -> list[GameLineQuote]:
    """Main full-game moneyline, spread and total per book (team totals and alts skipped)."""
    quotes = []
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            if key not in GAME_MARKETS or market.get("team") is not None:
                continue
            if (market.get("line_type") or "main") != "main":
                continue
            if market.get("period") not in (None, "full", "game") or market.get("suspended_at"):
                continue
            home = away = None
            point = None
            for o in market.get("outcomes", []):
                side = o.get("side")
                name = (o.get("name") or "").lower()
                if key == "totals":
                    point = o.get("point")
                    if name == "over":
                        home = o.get("price")
                    elif name == "under":
                        away = o.get("price")
                elif side == "home":
                    home = o.get("price")
                    if key == "spreads":
                        point = o.get("point")
                elif side == "away":
                    away = o.get("price")
            if home is None and away is None:
                continue
            quotes.append(
                GameLineQuote(
                    book=book["key"],
                    market=key,
                    point=float(point) if point is not None else None,
                    home_price=int(home) if home is not None else None,
                    away_price=int(away) if away is not None else None,
                )
            )
    return quotes


def parse_quota(headers: Any) -> Quota | None:
    try:
        return Quota(
            limit=int(headers["X-Daily-Limit"]),
            used=int(headers["X-Daily-Used"]),
            remaining=int(headers["X-Daily-Remaining"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


# --- HTTP client -------------------------------------------------------------


class QuotaExhausted(RuntimeError):
    pass


class PropLineProvider:
    name = "propline"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.prop-line.com/v1",
        quota_reserve: int = 100,
        timeout: float = 20.0,
        session: requests.Session | None = None,
        retries: int = 2,
        backoff: tuple[float, ...] = (2.0, 5.0),
        sleep=None,
    ):
        if not api_key:
            raise ValueError("PROPLINE_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.quota_reserve = quota_reserve
        self.timeout = timeout
        self.http = session or requests.Session()
        self.http.headers["X-API-Key"] = api_key
        self.last_quota: Quota | None = None
        self.retries = retries
        self.backoff = backoff
        self._sleep = sleep or __import__("time").sleep

    def _get(
        self, path: str, params: dict[str, str] | None = None, use_reserve: bool = False
    ) -> Any:
        """use_reserve=True (closing snapshots only) may dip into the reserved requests."""
        q = self.last_quota
        if q and (q.remaining <= 0 or (not use_reserve and q.remaining <= self.quota_reserve)):
            raise QuotaExhausted(
                f"{q.remaining} requests left today; {self.quota_reserve} are reserved "
                "for closing snapshots"
            )
        # Retry the transient failures (a 20 s read timeout failed the Oct 27 6 PM pricing run):
        # timeouts, dropped connections, rate limits and server errors.
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                resp = self.http.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                if last:
                    raise
                self._sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
                continue
            if getattr(resp, "status_code", 200) in (429, 500, 502, 503, 504) and not last:
                self._sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
                continue
            break
        self.last_quota = parse_quota(resp.headers) or self.last_quota
        resp.raise_for_status()
        return resp.json()

    def get_events(self) -> list[EventInfo]:
        return parse_events(self._get(f"/sports/{SPORT}/events"))

    def get_event_odds(self, event_id: str, closing: bool = False) -> dict[str, Any]:
        """One request returns every book and market for the event."""
        return self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            params={
                "markets": ",".join([*PROP_MARKETS, *GAME_MARKETS]),
                "bookmakers": ",".join(BOOKS),
            },
            use_reserve=closing,
        )
