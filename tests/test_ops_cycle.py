"""Stage 8: the 15-minute production cycle."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text

from nbaprops.db.base import session_scope
from nbaprops.db.models import OddsSnapshot, Pick
from nbaprops.ops import alerts, cycle
from tests.test_slate import BROWN, CADE, DAY, G1, G3, POLL, seed

PRICING = datetime(2026, 10, 20, 14, 0, tzinfo=UTC)  # 10:00 AM New York (EDT)
QUIET = datetime(2026, 10, 20, 14, 30, tzinfo=UTC)


@dataclass
class Fake:
    calls: list = field(default_factory=list)
    changed: dict = field(default_factory=dict)
    pulled: dict = field(default_factory=dict)
    fail: set = field(default_factory=set)
    skipped: list = field(default_factory=list)

    def deps(self) -> cycle.Deps:
        def rec(name, value=None):
            def f(*args):
                self.calls.append((name, args[-1] if name == "project" else None))
                if name in self.fail:
                    raise RuntimeError(f"{name} broke")
                return value(*args) if callable(value) else value

            return f

        class Settled:
            picks = parlays = 0

        class Priced:
            skipped = self.skipped

        def price(session, today, now):
            session.add(
                Pick(
                    run_id="r1",
                    game_id=G1,
                    player_id=CADE,
                    market="pts",
                    line=Decimal("26.5"),
                    side="over",
                    tier="high_conf",
                    mode="paper",
                    flagged_ts=now,
                    dk_price=100,
                    fair_prob=Decimal("0.56"),
                    ev=Decimal("0.12"),
                    stake=Decimal("20"),
                )
            )
            session.flush()
            return Priced()

        return cycle.Deps(
            tick=rec("odds"),
            settle=rec("settle", Settled()),
            project=rec("project", 10),
            price=rec("price", price),
            changed_games=rec("changes", lambda *a: self.changed),
            pulled_games=rec("pulls", lambda *a: self.pulled),
            health=rec("health", []),
        )


def names(fake):
    return [c[0] for c in fake.calls]


@pytest.mark.db
def test_a_pricing_run_projects_the_slate_prices_and_alerts():
    seed()
    fake, rec = (
        Fake(skipped=["0022600003: DraftKings odds older than 120 min"]),
        alerts.RecordingNotifier(),
    )
    rep = cycle.run_cycle(session_scope, PRICING, fake.deps(), rec)
    assert names(fake) == ["odds", "settle", "project", "price", "health"]
    assert fake.calls[2] == ("project", None)  # the whole slate
    assert rep.new_picks == 1 and not rep.errors
    assert any("10:00 AM picks: 1 to bet" in m and "EV +12.0%" in m for m in rec.sent)
    assert any("Cade Cunningham pts over 26.5" in m for m in rec.sent)  # names, not ids
    assert any("Pricing note" in m for m in rec.sent)
    # the same skip note at 3 PM is not alerted again
    fake.calls.clear()
    rec.sent.clear()
    cycle.run_cycle(session_scope, PRICING + timedelta(hours=5), fake.deps(), rec)
    assert not any("Pricing note" in m for m in rec.sent)


@pytest.mark.db
def test_injury_news_reprojects_only_that_game():
    seed()
    fake, rec = Fake(changed={G1: "1627759: Questionable -> Out"}), alerts.RecordingNotifier()
    cycle.run_cycle(session_scope, QUIET, fake.deps(), rec)
    assert names(fake) == ["odds", "settle", "changes", "pulls", "project", "price", "health"]
    assert ("project", [G1]) in fake.calls
    assert any(m.startswith("Injury news") and "Questionable -> Out" in m for m in rec.sent)


@pytest.mark.db
def test_pulled_props_reprice_without_reprojecting():
    seed()
    fake, rec = Fake(pulled={G1: ["Jaylen Brown"]}), alerts.RecordingNotifier()
    cycle.run_cycle(session_scope, QUIET, fake.deps(), rec)
    assert "project" not in names(fake) and "price" in names(fake)
    assert any("possible late scratch" in m and "Jaylen Brown" in m for m in rec.sent)


@pytest.mark.db
def test_a_quiet_tick_only_collects_settles_and_checks():
    seed()
    fake = Fake()
    cycle.run_cycle(session_scope, QUIET, fake.deps(), alerts.RecordingNotifier())
    assert names(fake) == ["odds", "settle", "changes", "pulls", "health"]


@pytest.mark.db
def test_a_failing_step_alerts_and_the_rest_still_runs():
    seed()
    fake, rec = Fake(fail={"odds"}), alerts.RecordingNotifier()
    rep = cycle.run_cycle(session_scope, PRICING, fake.deps(), rec)
    assert rep.errors == ["odds: odds broke"]
    assert names(fake) == ["odds", "settle", "project", "price", "health"]
    assert any("Job step 'odds' failed" in m for m in rec.sent)


def test_pricing_runs_follow_new_york_time_across_daylight_saving():
    assert cycle.is_pricing_run(PRICING)  # 10:00 EDT
    nov = datetime(2026, 11, 10, 15, 0, tzinfo=UTC)  # 10:00 EST
    assert cycle.is_pricing_run(nov)
    assert not cycle.is_pricing_run(nov - timedelta(hours=1))  # 9:00 EST
    assert cycle.is_pricing_run(datetime(2026, 11, 10, 23, 14, tzinfo=UTC))  # 6:14 PM
    assert not cycle.is_pricing_run(datetime(2026, 11, 10, 23, 15, tzinfo=UTC))


# --- the real health checks and helpers ------------------------------------------------


def health_at(now: datetime) -> list:
    with session_scope() as s:
        return cycle.health(s, now.astimezone(cycle.ET).date(), now)


@pytest.mark.db
def test_health_flags_stale_odds_missing_box_scores_and_no_champion():
    seed()
    afternoon = POLL + timedelta(hours=7)  # 1:00 PM New York; DraftKings last polled at 6 AM
    msgs = dict(health_at(afternoon))
    assert {f"stale:{DAY}:{G1}", f"stale:{DAY}:{G3}"} <= set(msgs)
    assert "no-champion" not in " ".join(msgs)  # seed's r1 is the champion
    with session_scope() as s:  # a final game yesterday with no box scores
        s.execute(
            text("UPDATE games SET game_date = :d, status = 'final' WHERE game_id = '0012600005'"),
            {"d": DAY - timedelta(days=1)},
        )
        s.execute(text("UPDATE model_runs SET status = 'retired'"))
    msgs = dict(health_at(afternoon))
    assert f"boxscores:{DAY - timedelta(days=1)}" in msgs and f"no-champion:{DAY}" in msgs
    early = datetime(2026, 10, 20, 12, 30, tzinfo=UTC)  # 8:30 AM: before the deadline
    assert not any(k.startswith("boxscores") for k in dict(health_at(early)))


@pytest.mark.db
def test_pulled_games_and_lined_players():
    seed()
    now = POLL + timedelta(hours=2)
    with session_scope() as s:
        s.add(
            OddsSnapshot(
                snapshot_ts=now,
                provider="propline",
                book="draftkings",
                game_id=G1,
                player_id=BROWN,
                player_name_raw="Jaylen Brown",
                market="pts",
                line_type="main",
                line=Decimal("24.5"),
                over_price=None,
                under_price=None,
                is_withdrawn=True,
            )
        )
        s.execute(
            text("UPDATE players SET current_team_id = 1610612765 WHERE player_id = :p"),
            {"p": CADE},
        )
    with session_scope() as s:
        assert cycle.pulled_games(s, now) == {G1: ["Jaylen Brown"]}
        assert cycle.pulled_games(s, now + timedelta(minutes=15)) == {}  # only this cycle's
        assert cycle.dk_players(s, DAY, now) == [(CADE, 1610612765)]


@pytest.mark.db
def test_real_wiring_collects_settles_projects_and_prices(monkeypatch, tmp_path):
    """real_deps end to end on the seeded slate: the real collector (fixture provider), real
    settlement and pricing. Only the model projection is stubbed (it needs seasons of box
    scores); it must be called for the champion and every shadow challenger."""
    from nbaprops.config import load_strategy
    from nbaprops.models import project as pj
    from nbaprops.ops import bundles
    from nbaprops.pipeline import run as pipeline_run
    from nbaprops.settings import get_settings
    from tests.test_pipeline_store import FakeProvider

    seed()
    monkeypatch.setattr(pipeline_run, "RAW_DIR", tmp_path / "raw")
    projected = []
    monkeypatch.setattr(bundles, "load", lambda s, rid: rid)
    monkeypatch.setattr(bundles, "stored_challengers", lambda s: ["shadow-1"])
    monkeypatch.setattr(
        pj, "project_slate", lambda s, b, d, now, n, extra, ids: projected.append((b, ids, n)) or []
    )
    cfg = load_strategy(get_settings().strategy_config)
    rec = alerts.RecordingNotifier()
    now = POLL + timedelta(hours=4)  # 10:00 AM New York: a pricing run
    rep = cycle.run_cycle(session_scope, now, cycle.real_deps(FakeProvider(), cfg), rec)
    assert not rep.errors, rep.errors
    assert rep.actions == ["odds", "settle", "project", "price", "health"]
    # champion, then shadow; a pricing run projects only games that haven't tipped
    assert [b for b, _, _ in projected] == ["r1", "shadow-1"]
    assert all(sorted(ids) == sorted([G1, G3]) for _, ids, _ in projected)
    assert [n for _, _, n in projected] == [cfg.n_sims, cycle.SHADOW_SIMS]  # shadow: fewer
    with session_scope() as s:
        polled = s.execute(
            text("SELECT count(*) FROM odds_polls WHERE snapshot_ts = :t"), {"t": now}
        ).scalar()
        picks = s.execute(
            text("SELECT count(*) FROM picks WHERE flagged_ts = :t"), {"t": now}
        ).scalar()
    assert polled >= 1  # the collector ran and recorded its poll
    assert picks == rep.new_picks and picks > 0
    assert any("picks:" in m for m in rec.sent)


@pytest.mark.db
def test_health_warns_before_the_database_fills():
    seed()
    with session_scope() as s:
        size = s.execute(text("SELECT pg_database_size(current_database())")).scalar() / 1e6
        tiny = cycle.health(s, DAY, POLL + timedelta(hours=1), 120, int(size))  # at 100%
        roomy = cycle.health(s, DAY, POLL + timedelta(hours=1), 120, int(size * 10))
    assert any(k.startswith("dbsize:") for k, _ in tiny)
    assert not any(k.startswith("dbsize:") for k, _ in roomy)


@pytest.mark.db
def test_health_flags_a_game_stuck_after_tip():
    from tests.test_slate import TIP1

    seed()  # G1 tipped at 19:00 and is still 'scheduled'
    msgs = dict(health_at(TIP1 + timedelta(hours=7)))
    assert f"stuck:{G1}" in msgs and "postponed" in msgs[f"stuck:{G1}"]
    assert f"stuck:{G1}" not in dict(health_at(TIP1 + timedelta(hours=2)))
