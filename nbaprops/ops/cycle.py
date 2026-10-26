"""The production cycle (plan, Stage 8), run every 15 minutes by the scheduler.

Decides what to do from the time in New York (correct across daylight saving), in order:

  1. odds and injuries (closing lines first): never delayed or blocked by anything below
  2. settle whatever has become final (idempotent; Stage 8's 8 AM job, run every cycle)
  3. at 10:00, 3:00 PM and 6:00 PM: project the whole slate with the champion (challengers
     in shadow), price it, and alert the new picks
  4. late changes, on any cycle: a game whose injury statuses changed since its projection is
     re-projected and re-priced; DraftKings pulling a player's props (with no injury news)
     re-prices that game and alerts a possible late scratch
  5. health checks, each alerting once: stale odds on a game day, last night's box scores
     missing, no champion model

Each step is isolated: a failure alerts with this run's log link and the cycle moves on.
Steps are injectable (`Deps`) so the orchestration can be tested without a provider or model.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from nbaprops.ops import alerts

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
PRICING_HOURS_ET = (10, 15, 18)  # the daily pricing runs (plan, Stage 8)
TICK_MINUTES = 15
BOXSCORE_DEADLINE_ET = 9  # plan: alert if last night's games are missing by 9:00 AM
SHADOW_SIMS = 2000  # simulations per game for shadow challengers


@dataclass
class Deps:
    """What a cycle calls. Defaults are the real implementations (see `real_deps`)."""

    tick: Callable  # (now) -> TickPlan-like with .full_snapshot
    settle: Callable  # (session, today, now) -> SettleResult-like with .picks/.summary()
    project: Callable  # (session, today, now, game_ids | None) -> int rows projected
    price: Callable  # (session, today, now) -> SlateResult-like with .picks/.skipped
    changed_games: Callable  # (session, today, now) -> {game_id: what changed}
    pulled_games: Callable  # (session, now) -> {game_id: [player names]} pulled this cycle
    health: Callable  # (session, today, now) -> list[(key, message)]
    ramp: Callable | None = None  # (session) -> (real stake multiplier, note)
    # (now) -> bool: a pricing time has passed today with no pricing run since. Without it,
    # only a cycle starting in the first minutes of a pricing hour prices (is_pricing_run).
    pricing_due: Callable | None = None
    full_snapshot: Callable | None = None  # (now) -> None: every game's odds, before pricing


@dataclass
class CycleReport:
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    new_picks: int = 0


def is_pricing_run(now: datetime) -> bool:
    local = now.astimezone(ET)
    return local.hour in PRICING_HOURS_ET and local.minute < TICK_MINUTES


def pricing_due_in(session, now: datetime) -> bool:
    """A pricing time (10 AM, 3 PM, 6 PM New York) has passed today and no cycle has priced
    since. GitHub starts scheduled jobs late, often 15-60 minutes at the top of the hour: a
    fixed "first 15 minutes" window skipped whole days of picks (Oct 23-25, 2026). A failed
    pricing step isn't recorded as done, so the next cycle tries again."""
    local = now.astimezone(ET)
    slots = [datetime.combine(local.date(), time(h), ET) for h in PRICING_HOURS_ET]
    passed = [t for t in slots if t <= local]
    if not passed:
        return False
    done = session.execute(
        text("""SELECT 1 FROM ops_runs WHERE kind = 'cycle' AND started_ts >= :since
                AND actions @> '["price"]'::jsonb LIMIT 1"""),
        {"since": passed[-1]},
    ).first()
    return done is None


def run_cycle(session_factory, now: datetime, deps: Deps, notifier: alerts.Notifier) -> CycleReport:
    rep = CycleReport()
    today = now.astimezone(ET).date()

    def step(name: str, fn):
        try:
            out = fn()
            rep.actions.append(name)
            return out
        except Exception as exc:  # isolate: one failure must not stop the rest
            log.exception("%s failed", name)
            rep.errors.append(f"{name}: {exc}")
            notifier.send(
                alerts.with_link(
                    f"Job step '{name}' failed: {exc}\n{traceback.format_exc(limit=3)[-600:]}"
                )
            )
            return None

    due = deps.pricing_due(now) if deps.pricing_due else is_pricing_run(now)
    plan = step("odds", lambda: deps.tick(now))
    if due and deps.full_snapshot and not getattr(plan, "full_snapshot", False):
        step("snapshot", lambda: deps.full_snapshot(now))  # fresh odds for a late pricing run

    def _settle():
        with session_factory() as s:
            r = deps.settle(s, today, now)
        if r.picks or r.parlays:
            notifier.send(r.summary())
        return r

    step("settle", _settle)

    def _price_and_alert(label: str):
        with session_factory() as s:
            res = deps.price(s, today, now)
            new = _new_picks(s, now)
            factor, note = deps.ramp(s) if deps.ramp else (1.0, "")
            for msg in res.skipped:
                alerts.once(s, f"skip:{today}:{msg}", now, notifier, f"Pricing note: {msg}")
        rep.new_picks += len(new)
        if len(new):
            notifier.send(format_picks(new, label, factor, note))
        return res

    if due:
        step("project", lambda: _with(session_factory, lambda s: deps.project(s, today, now, None)))
        step("price", lambda: _price_and_alert(f"{now.astimezone(ET):%-I:%M %p} picks"))
    else:
        changed = (
            step(
                "changes",
                lambda: _with(session_factory, lambda s: deps.changed_games(s, today, now)),
            )
            or {}
        )
        pulled = (
            step("pulls", lambda: _with(session_factory, lambda s: deps.pulled_games(s, now))) or {}
        )
        if changed:
            notifier.send(
                "Injury news: "
                + "; ".join(f"{g}: {m}" for g, m in changed.items())
                + ". Re-projecting those games."
            )
            step(
                "reproject",
                lambda: _with(
                    session_factory, lambda s: deps.project(s, today, now, list(changed))
                ),
            )
        if pulled:
            notifier.send(
                "DraftKings pulled props (possible late scratch): "
                + "; ".join(f"{g}: {', '.join(names)}" for g, names in pulled.items())
            )
        if changed or pulled:
            step("reprice", lambda: _price_and_alert("Late-change picks"))

    def _health():
        with session_factory() as s:
            for key, msg in deps.health(s, today, now):
                alerts.once(s, key, now, notifier, msg)

    step("health", _health)
    _record(session_factory, "cycle", now, rep)
    return rep


def _record(session_factory, kind: str, started: datetime, rep: CycleReport) -> None:
    """Note this run in ops_runs (the go-live check reads it). Never fails the cycle."""
    from datetime import UTC

    from nbaprops.db.models import OpsRun

    try:
        with session_factory() as s:
            s.add(
                OpsRun(
                    kind=kind,
                    started_ts=started,
                    finished_ts=datetime.now(UTC),
                    ok=not rep.errors,
                    actions=rep.actions,
                    errors=rep.errors,
                )
            )
    except Exception:  # pragma: no cover - recording must never break production
        log.exception("could not record the run")


def _with(session_factory, fn):
    with session_factory() as s:
        return fn(s)


def _new_picks(session, now: datetime) -> pd.DataFrame:
    return pd.read_sql_query(
        text("""SELECT p.*, pl.full_name FROM picks p LEFT JOIN players pl USING (player_id)
                 WHERE p.flagged_ts = :t ORDER BY p.stake DESC, p.ev DESC"""),
        session.connection(),
        params={"t": now},
    )


def format_picks(
    picks: pd.DataFrame, label: str, real_factor: float = 1.0, ramp_note: str = ""
) -> str:
    """A phone-sized message: staked picks first (with the real-money amount after the
    go-live ramp), tracked flags summarized."""
    staked = picks[picks["stake"].astype(float) > 0]
    tracked = len(picks) - len(staked)
    lines = [
        f"{label}: {len(staked)} to bet" + (f", {tracked} tracked (not staked)" if tracked else "")
    ]
    for r in staked.itertuples(index=False):
        bet = f" (bet {float(r.stake) * real_factor:.2f})" if real_factor != 1.0 else ""
        who = getattr(r, "full_name", None) or r.player_id
        lines.append(
            f"- {who} {r.market} {r.side} {float(r.line):g} at "
            f"{int(r.dk_price):+d}: fair {float(r.fair_prob):.1%}, "
            f"EV {float(r.ev):+.1%}, stake {float(r.stake):.2f}{bet}"
            + (" (high-conf)" if r.tier == "high_conf" else "")
        )
    if ramp_note and len(staked):
        lines.append(ramp_note)
    return "\n".join(lines)


# --- real implementations ---------------------------------------------------------------


def changed_games(session, today: date, now: datetime) -> dict:
    """Games today whose injury statuses changed after the champion's projections."""
    from nbaprops.ops import bundles
    from nbaprops.pricing import slate

    run_id = bundles.champion(session)
    games = slate.slate_games(session, today, now)
    if run_id is None or games.empty:
        return {}
    grids = slate.load_grids(session, run_id, games["game_id"].tolist(), now)
    return slate.injury_changes(session, grids, now)


def pulled_games(session, now: datetime) -> dict:
    """DraftKings props withdrawn in this cycle's snapshot, by game."""
    rows = session.execute(
        text("""
        SELECT o.game_id, o.player_name_raw FROM odds_snapshots o
        WHERE o.book = 'draftkings' AND o.is_withdrawn AND o.snapshot_ts = :t
          AND o.line_type = 'main'"""),
        {"t": now},
    ).all()
    out: dict = {}
    for g, name in rows:
        out.setdefault(g, [])
        if name not in out[g]:
            out[g].append(name)
    return out


def health(
    session, today: date, now: datetime, max_odds_age_min: int = 120, db_size_limit_mb: int = 500
) -> list:
    """(dedupe key, message) for each current problem."""
    from nbaprops.ops import bundles

    out = []
    local = now.astimezone(ET)
    if bundles.champion(session) is None:
        out.append(
            (
                f"no-champion:{today}",
                "No champion model: import a production bundle "
                "(python -m nbaprops.ops.run import-bundle PATH --champion).",
            )
        )
    if local.hour >= BOXSCORE_DEADLINE_ET:
        missing = (
            session.execute(
                text("""
            SELECT g.game_id FROM games g WHERE g.game_date = :d AND g.status = 'final'
              AND NOT EXISTS (SELECT 1 FROM player_game_logs l WHERE l.game_id = g.game_id)"""),
                {"d": today - timedelta(days=1)},
            )
            .scalars()
            .all()
        )
        if missing:
            out.append(
                (
                    f"boxscores:{today - timedelta(days=1)}",
                    f"Last night's box scores are missing for {len(missing)} games: run "
                    "`make stats-day` on your machine.",
                )
            )
    if local.hour >= 10:
        stale = session.execute(
            text("""
            SELECT g.game_id, MAX(p.snapshot_ts) AS last FROM games g
            LEFT JOIN odds_polls p ON p.game_id = g.game_id AND p.books ? 'draftkings'
            WHERE g.game_date = :d AND g.status = 'scheduled' AND g.tip_time > :now
            GROUP BY g.game_id"""),
            {"d": today, "now": now},
        ).all()
        cutoff = now - timedelta(minutes=max_odds_age_min)
        for gid, last in stale:
            if last is None or last < cutoff:
                age = (
                    "never"
                    if last is None
                    else f"{int((now - last).total_seconds() // 60)} min ago"
                )
                out.append(
                    (
                        f"stale:{today}:{gid}",
                        f"DraftKings odds for {gid} last polled {age}: check the collector.",
                    )
                )
    stuck = (
        session.execute(
            text("""
        SELECT game_id FROM games WHERE status <> 'final'
          AND tip_time BETWEEN :a AND :b"""),
            {"a": now - timedelta(days=3), "b": now - timedelta(hours=6)},
        )
        .scalars()
        .all()
    )
    for gid in stuck:
        out.append(
            (
                f"stuck:{gid}",
                f"Game {gid} still isn't final 6 hours after tip "
                "(postponed or suspended?). Its picks are voided 36 hours after tip.",
            )
        )
    size_mb = session.execute(text("SELECT pg_database_size(current_database())")).scalar() / 1e6
    if size_mb >= 0.8 * db_size_limit_mb:
        out.append(
            (
                f"dbsize:{today:%Y-%m}:{int(size_mb // 25)}",
                f"Database is {size_mb:.0f} MB of a {db_size_limit_mb} MB limit: "
                "prune old raw odds or move to a larger plan before it fills.",
            )
        )
    return out


def real_deps(provider, cfg) -> Deps:
    from nbaprops.db.base import session_scope
    from nbaprops.ops import bundles
    from nbaprops.ops.settle import settle
    from nbaprops.pipeline.run import odds_snapshot, tick
    from nbaprops.pricing import slate

    def project(session, today, now, game_ids):
        from nbaprops.models.project import project_slate

        run_id = bundles.champion(session)
        if run_id is None:
            return 0
        extra = dk_players(session, today, now)
        if game_ids is None:  # a pricing run: only games that haven't tipped
            game_ids = slate.slate_games(session, today, now)["game_id"].tolist()
            if not game_ids:
                return 0
        n = 0
        for rid in [run_id, *bundles.stored_challengers(session)]:  # challengers: shadow only
            # Shadow challengers are only compared (log loss), not priced: fewer simulations
            # are plenty and save scheduled-job minutes.
            sims = cfg.n_sims if rid == run_id else min(cfg.n_sims, SHADOW_SIMS)
            out = project_slate(
                session, bundles.load(session, rid), today, now, sims, extra, game_ids
            )
            n += len(out)
        return n

    return Deps(
        tick=lambda now: tick(provider, now),
        full_snapshot=lambda now: odds_snapshot(provider, now),
        pricing_due=lambda now: _with(session_scope, lambda s: pricing_due_in(s, now)),
        settle=settle,
        project=project,
        price=lambda s, today, now: slate.price_slate(s, today, now, cfg),
        changed_games=changed_games,
        pulled_games=pulled_games,
        health=lambda s, today, now: health(
            s, today, now, cfg.max_odds_age_min, cfg.db_size_limit_mb
        ),
        ramp=lambda s: real_ramp(s, cfg),
    )


def real_ramp(session, cfg) -> tuple[float, str]:
    from nbaprops.ops.place import ramp

    return ramp(session, cfg.ramp_bets, cfg.ramp_factor)


def dk_players(session, today: date, now: datetime) -> list[tuple[int, int]]:
    """(player_id, team_id) for everyone DraftKings has a line on today, so players outside the
    usual rotation still get projected (and can be priced). Team from today's roster."""
    rows = session.execute(
        text("""
        SELECT DISTINCT o.player_id, p.current_team_id FROM odds_snapshots o
        JOIN games g USING (game_id) JOIN players p ON p.player_id = o.player_id
        WHERE g.game_date = :d AND o.book = 'draftkings' AND o.snapshot_ts <= :now
          AND p.current_team_id IN (g.home_team_id, g.away_team_id)"""),
        {"d": today, "now": now},
    ).all()
    return [(int(p), int(t)) for p, t in rows]
