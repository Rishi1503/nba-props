"""Pipeline command line. Run `python -m nbaprops.pipeline.run --help`.

Stage 1 (run from your own machine: the NBA stats site blocks many cloud servers)
  backfill    load full seasons of box scores (default 2021-22 through 2025-26)
  stats-day   daily job: yesterday's games, missing box scores, new players' positions
Stage 2
  seed        teams + players from nba_api's bundled lists (daily, for trades and rookies)
  schedule    load the NBA schedule into `games` (daily)
  snapshot    full odds snapshot now (add --close for a closing snapshot)
  injuries    fetch and store the latest official injury report
  tick        what the scheduler runs every 15 minutes; decides what to do from the time in
              New York, so it is correct on both sides of daylight saving time
  aliases     list names that matched no player, with suggested matches
  alias-add   map a name to a player id: alias-add "Nic Claxton" 1629651
  play-rates  how often players listed Questionable (etc.) actually played, from logged data
Stage 3
  features    write the training feature set to a Parquet file
"""

import argparse
import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from nbaprops.db.base import session_scope
from nbaprops.db.models import Game
from nbaprops.pipeline import injuries, odds_store, reference, schedule, stats
from nbaprops.pipeline.odds import (
    OddsProvider,
    PropLineProvider,
    parse_game_lines,
    parse_props,
    pulled_players,
)
from nbaprops.settings import get_settings

log = logging.getLogger("nbaprops.pipeline")
ET = ZoneInfo("America/New_York")

SNAPSHOT_HOURS_ET = (10, 15, 18)  # openers, midday, evening (plan, Stage 2)
TICK_MINUTES = 15
# A closing snapshot is taken on every tick in the last 35 minutes before tip; grading and CLV
# use the latest one. GitHub's scheduled runs can start 5-15 minutes late, so a single narrow
# window could be missed entirely.
CLOSE_WINDOW = timedelta(minutes=35)
LOOKAHEAD = timedelta(hours=36)  # only spend requests on games tipping within this window
RAW_DIR = Path("data/raw")


# --- Scheduling decision (pure) ----------------------------------------------


@dataclass
class TickPlan:
    full_snapshot: bool = False
    close_game_ids: list[str] = field(default_factory=list)

    @property
    def injuries(self) -> bool:
        return self.full_snapshot or bool(self.close_game_ids)

    @property
    def idle(self) -> bool:
        return not (self.full_snapshot or self.close_game_ids)


def plan_tick(now: datetime, upcoming: list[tuple[str, datetime]]) -> TickPlan:
    """`upcoming`: (game_id, tip_time) for scheduled games. Times are timezone-aware."""
    local = now.astimezone(ET)
    plan = TickPlan(full_snapshot=local.hour in SNAPSHOT_HOURS_ET and local.minute < TICK_MINUTES)
    for game_id, tip in upcoming:
        if timedelta(0) < tip - now <= CLOSE_WINDOW:
            plan.close_game_ids.append(game_id)
    return plan


# --- Actions -----------------------------------------------------------------


def _archive(provider: str, event_id: str, payload: dict, ts: datetime) -> None:
    """Keep every raw response so anything can be re-parsed later."""
    folder = RAW_DIR / provider / ts.astimezone(ET).strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ts.strftime('%H%M%S')}_{event_id}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def odds_snapshot(
    provider: OddsProvider,
    now: datetime,
    only_game_ids: set[str] | None = None,
    is_close: bool = False,
) -> odds_store.StoreResult:
    result = odds_store.StoreResult()
    events = provider.get_events()
    with session_scope() as s:
        for ev in events:
            if ev.commence_time - now > LOOKAHEAD or ev.commence_time <= now:
                continue
            game_id = odds_store.match_event(s, provider.name, ev)
            if game_id is None:
                log.warning(
                    "no game for event %s %s @ %s", ev.provider_event_id, ev.away_team, ev.home_team
                )
                continue
            if only_game_ids is not None and game_id not in only_game_ids:
                continue
            payload = provider.get_event_odds(ev.provider_event_id, closing=is_close)
            _archive(provider.name, ev.provider_event_id, payload, now)
            for book, _, name in pulled_players(payload):
                result.pulled_players.add(f"{name} ({book})")
            quotes = parse_props(payload)
            odds_store.store_props(s, provider.name, game_id, quotes, now, is_close, result)
            odds_store.record_poll(s, provider.name, game_id, quotes, now, is_close)
            odds_store.store_game_lines(
                s, provider.name, game_id, parse_game_lines(payload), now, is_close, result
            )
    log.info(
        "odds%s: %d props seen, %d prop rows and %d game-line rows written",
        " (close)" if is_close else "",
        result.props_seen,
        result.props_written,
        result.lines_written,
    )
    log.info("player match rate: %.1f%% (plan target 98%%)", 100 * result.match_rate)
    if result.unmatched_players:
        log.warning(
            "unmatched players (see `aliases`): %s", ", ".join(sorted(result.unmatched_players))
        )
    if result.pulled_players:
        log.warning(
            "props pulled, possible late scratch: %s", ", ".join(sorted(result.pulled_players))
        )
    quota = getattr(provider, "last_quota", None)
    if quota:
        log.info("quota: %d of %d requests left today", quota.remaining, quota.limit)
    return result


def injury_snapshot(now: datetime) -> int:
    url, text = injuries.fetch_latest_report(now)
    rows = injuries.parse_report_text(text)
    with session_scope() as s:
        written = injuries.store_report(s, rows, now)
    log.info("injuries: %s -> %d rows parsed, %d new", url.rsplit("/", 1)[-1], len(rows), written)
    return written


def tick(provider: OddsProvider, now: datetime) -> TickPlan:
    with session_scope() as s:
        upcoming = s.execute(
            select(Game.game_id, Game.tip_time).where(
                Game.tip_time > now, Game.tip_time <= now + LOOKAHEAD, Game.status == "scheduled"
            )
        ).all()
    plan = plan_tick(now, [(g, t) for g, t in upcoming])
    if plan.idle:
        log.info("tick: nothing scheduled at %s ET", now.astimezone(ET).strftime("%H:%M"))
        return plan
    # Closing snapshots first: they are the ones that must not be missed.
    if plan.close_game_ids:
        odds_snapshot(provider, now, set(plan.close_game_ids), is_close=True)
    if plan.full_snapshot:
        odds_snapshot(provider, now)
    if plan.injuries:
        try:
            injury_snapshot(now)
        except Exception as exc:  # injuries must never block odds logging
            log.error("injury report failed: %s", exc)
    return plan


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nbaprops.pipeline.run")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    sub.add_parser("schedule")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--close", action="store_true")
    sub.add_parser("injuries")
    sub.add_parser("tick")
    bf = sub.add_parser("backfill")
    # 2021-22 is a warm-up season: it supplies last-season priors for 2022-23 (plan, Stage 1).
    bf.add_argument(
        "--seasons", nargs="+", default=["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    )
    bf.add_argument(
        "--refresh",
        action="store_true",
        help="re-download cached responses (picks up stat corrections)",
    )
    bf.add_argument("--no-boxscores", action="store_true", help="skip per-game box scores")
    bf.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    day = sub.add_parser("stats-day")
    day.add_argument("--date", help="YYYY-MM-DD (default: yesterday in New York)")
    sub.add_parser("aliases")
    sub.add_parser("play-rates")
    al = sub.add_parser("alias-add")
    al.add_argument("name")
    al.add_argument("player_id", type=int)
    feat = sub.add_parser("features")
    feat.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26"])
    feat.add_argument("--out", default="data/features/train.parquet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = datetime.now(UTC)

    if args.cmd == "seed":
        with session_scope() as s:
            log.info("seeded %d teams", reference.seed_teams(s))
            log.info("seeded %d players", reference.seed_players(s))
    elif args.cmd == "schedule":
        rows = schedule.parse_schedule(schedule.fetch_schedule())
        with session_scope() as s:
            log.info("schedule: %d games upserted", schedule.upsert_games(s, rows))
    elif args.cmd == "injuries":
        injury_snapshot(now)
    elif args.cmd == "backfill":
        fetch = stats.StatsFetcher(pause=args.pause, refresh=args.refresh)
        with session_scope() as s:
            reference.seed_teams(s)
        for season in args.seasons:
            issues = stats.backfill_season(fetch, season, boxscores=not args.no_boxscores)
            log.info(
                "%s done: %d validation issues, %d requests so far",
                season,
                len(issues),
                fetch.requests_made,
            )
    elif args.cmd == "aliases":
        from nbaprops.db.models import UnmatchedName

        with session_scope() as s:
            queue = s.scalars(select(UnmatchedName).order_by(UnmatchedName.seen_count.desc()))
            for u in queue:
                hints = ", ".join(
                    f"{n} ({pid})" for pid, n in reference.suggest_players(s, u.name_raw)
                )
                print(
                    f"{u.name_raw!r:32} {u.source:14} seen {u.seen_count:>4}x  "
                    f"suggest: {hints or 'none'}"
                )
    elif args.cmd == "play-rates":
        import pandas as pd
        from sqlalchemy import text as sql

        from nbaprops.features.build import estimate_play_rates

        with session_scope() as s:
            conn = s.connection()
            inj = pd.read_sql_query(sql("SELECT * FROM injury_snapshots"), conn)
            logs = pd.read_sql_query(
                sql("SELECT game_id, player_id, minutes, dnp FROM player_game_logs"), conn
            )
            games = pd.read_sql_query(sql("SELECT game_id, tip_time, status FROM games"), conn)
        print(estimate_play_rates(inj, logs, games).to_string(index=False))
        print("Set p_play_questionable in config/strategy.yaml to the Questionable rate.")
    elif args.cmd == "alias-add":
        with session_scope() as s:
            fixed = reference.add_alias(s, args.name, args.player_id)
        log.info("alias added; %d past rows now matched", fixed)
    elif args.cmd == "features":
        from nbaprops.features.build import build_training_set  # needs the model extra

        with session_scope() as s:
            df = build_training_set(s, args.seasons)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        log.info("features: %d rows x %d columns -> %s", len(df), df.shape[1], args.out)
    elif args.cmd == "stats-day":
        day_ = date.fromisoformat(args.date) if args.date else stats.yesterday_et(now)
        issues = stats.pull_day(stats.StatsFetcher(), day_)
        log.info("stats for %s: %d validation issues", day_, len(issues))
    else:
        provider = PropLineProvider(get_settings().propline_api_key)
        if args.cmd == "snapshot":
            odds_snapshot(provider, now, is_close=args.close)
        else:
            tick(provider, now)


if __name__ == "__main__":
    main()
