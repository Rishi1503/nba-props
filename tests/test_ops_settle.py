"""Stage 8: settlement, model storage in Postgres, weekly retrain, alert delivery."""

from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import BankrollEntry, ModelRun, Parlay, ParlayLeg, Pick, PlayerGameLog
from nbaprops.ops import alerts, bundles
from nbaprops.ops.settle import settle
from tests.test_slate import BROWN, CADE, DAY, G1, G3, NOW, WEMBY, seed

DUREN = 1631105
NEXT_DAY = DAY + timedelta(days=1)
MORNING = NOW + timedelta(hours=20)


def pick(player, game, side, line, price, stake, mode="paper", placed=None, market="pts"):
    return Pick(
        run_id="r1",
        game_id=game,
        player_id=player,
        market=market,
        line=Decimal(str(line)),
        side=side,
        tier="high_conf",
        mode=mode,
        flagged_ts=NOW,
        dk_price=price,
        price_placed=placed,
        fair_prob=Decimal("0.6"),
        ev=Decimal("0.05"),
        stake=Decimal(str(stake)),
    )


def settle_setup():
    seed()
    with session_scope() as s:
        s.execute(text(f"UPDATE games SET status = 'final' WHERE game_id = '{G1}'"))
        s.add_all(
            [
                PlayerGameLog(
                    game_id=G1,
                    player_id=CADE,
                    team_id=1610612765,
                    minutes=36,
                    pts=31,
                    reb=5,
                    ast=9,
                    fg3m=3,
                ),
                PlayerGameLog(
                    game_id=G1,
                    player_id=BROWN,
                    team_id=1610612738,
                    minutes=34,
                    pts=24,
                    reb=6,
                    ast=4,
                    fg3m=2,
                ),
            ]
        )
        picks = [
            pick(CADE, G1, "over", 26.5, 100, 20),  # 31: win, paper
            pick(BROWN, G1, "under", 24.5, -110, 10, mode="real", placed=-105),  # 24: win
            pick(DUREN, G1, "over", 10.5, -110, 5),  # no box-score line: void
            pick(WEMBY, G3, "over", 23.5, 100, 20),  # game not final: waits
        ]
        s.add_all(picks)
        s.flush()
        par = Parlay(
            created_ts=NOW,
            slate_date=DAY,
            tier="satellite",
            mode="paper",
            joint_prob=Decimal("0.4"),
            dk_price=264,
            ev=Decimal("0.05"),
            stake=Decimal("5"),
        )
        s.add(par)
        s.flush()
        s.add_all(
            [
                ParlayLeg(parlay_id=par.parlay_id, pick_id=picks[0].pick_id),
                ParlayLeg(parlay_id=par.parlay_id, pick_id=picks[2].pick_id),
            ]
        )


@pytest.mark.db
def test_settlement_grades_prices_clv_parlays_and_the_ledger():
    settle_setup()
    with session_scope() as s:
        res = settle(s, NEXT_DAY, MORNING)
    assert res.picks == 3 and res.record == {"win": 2, "loss": 0, "push": 0, "void": 1}
    with session_scope() as s:
        by = {p.player_id: p for p in s.scalars(select(Pick))}
        cade, brown, duren, wemby = by[CADE], by[BROWN], by[DUREN], by[WEMBY]
        assert (cade.result, float(cade.pnl), cade.closing_price) == ("win", 20.0, 100)
        # consensus close at 26.5 is -110/-110 (0.5); taken at +100 (implied 0.5): CLV 0
        assert float(cade.clv) == pytest.approx(0.0, abs=1e-4)
        # real bet: P&L at the placed -105, CLV against the price taken
        assert brown.result == "win" and float(brown.pnl) == pytest.approx(10 * 100 / 105, abs=0.01)
        assert float(brown.clv) == pytest.approx(0.5 - 105 / 205, abs=1e-4)
        assert duren.result == "void" and float(duren.pnl) == 0
        assert wemby.result is None  # G3 isn't final
        par = s.scalars(select(Parlay)).one()
        assert par.result == "win" and float(par.pnl) == pytest.approx(5.0)  # void leg drops out
        ledger = s.get(BankrollEntry, NEXT_DAY)
        assert float(ledger.bankroll) == pytest.approx(1000 + 9.52, abs=0.01)  # real P&L only
    assert res.pnl["paper"] == pytest.approx(25.0) and res.pnl["real"] == pytest.approx(9.52, 0.01)
    with session_scope() as s:
        again = settle(s, NEXT_DAY, MORNING + timedelta(minutes=15))
    assert again.picks == 0 and again.parlays == 0  # idempotent
    with session_scope() as s:
        assert float(s.get(BankrollEntry, NEXT_DAY).bankroll) == pytest.approx(1009.52, 0.01)


@pytest.mark.db
def test_a_parlay_waits_for_every_leg():
    settle_setup()
    with session_scope() as s:
        legs = s.scalars(select(ParlayLeg)).all()
        wemby = s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one()
        s.delete(legs[1])
        s.add(ParlayLeg(parlay_id=legs[0].parlay_id, pick_id=wemby.pick_id))
    with session_scope() as s:
        settle(s, NEXT_DAY, MORNING)
        assert s.scalars(select(Parlay)).one().result is None  # G3 leg still open


# --- bundles in Postgres --------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_bundle():
    from nbaprops.features import build as fb
    from nbaprops.models import project as pj
    from tests import feature_fixtures as ff

    df = fb.training_frame(ff.build())
    b = pj.fit_bundle(
        df[df.season == "2024-25"], df[df.season == "2025-26"].head(1500), df, "b1", n_sims=200
    )
    return b, df


@pytest.mark.db
def test_bundles_round_trip_and_champion_promotion(small_bundle):
    from nbaprops.models import project as pj
    from tests.test_revalidation import _reset

    b, df = small_bundle
    _reset(with_schedule=False)
    with session_scope() as s:
        bundles.save(s, b, champion=True)
    with session_scope() as s:
        again = bundles.load(s, "b1")
        assert bundles.champion(s) == "b1"
    day = df[df.game_date == df.game_date.max()]
    when = pd.Timestamp("2026-04-01", tz="UTC").to_pydatetime()
    pd.testing.assert_frame_equal(
        pj.project_frame(day, b, 500, df, when), pj.project_frame(day, again, 500, df, when)
    )
    second = pj.Bundle("b2", b.minutes, b.rates, b.scenario, b.calibrator, b.meta)
    with session_scope() as s:
        bundles.save(s, second)
        assert bundles.stored_challengers(s) == ["b2"] and bundles.champion(s) == "b1"
        bundles.promote(s, "b2")
    with session_scope() as s:
        status = dict(s.execute(select(ModelRun.run_id, ModelRun.status)).all())
    assert status == {"b1": "retired", "b2": "champion"}
    with session_scope() as s, pytest.raises(FileNotFoundError):
        bundles.load(s, "missing")


@pytest.mark.db
def test_weekly_retrain_stores_a_challenger_with_the_champions_features(small_bundle, monkeypatch):
    from datetime import date

    from nbaprops.ops import retrain as rt
    from tests.test_revalidation import _reset

    b, df = small_bundle
    _reset(with_schedule=False)
    with session_scope() as s:
        bundles.save(s, b, champion=True)
    monkeypatch.setattr(rt, "build_training_set", lambda s, seasons: df)
    with session_scope() as s:
        run_id = rt.retrain(s, date(2026, 4, 6), first_train_season="2024-25")
    assert run_id == "wk-20260406"
    with session_scope() as s:
        new = bundles.load(s, run_id)
        assert bundles.champion(s) == "b1" and bundles.stored_challengers(s) == [run_id]
    assert new.minutes.features == b.minutes.features and new.rates.features == b.rates.features
    assert new.calibrator.curves == b.calibrator.curves and new.meta["parent"] == "b1"
    assert new.minutes.trained_through < "2026-04-06"


# --- alerts ---------------------------------------------------------------------------


def test_long_messages_are_split_under_the_service_limit():
    text_ = "\n".join(f"line {i} " + "x" * 80 for i in range(100))
    parts = alerts._chunks(text_, 1900)
    assert all(len(p) <= 1900 for p in parts) and "\n".join(parts) == text_
    assert alerts._chunks("y" * 5000, 1900)[0] == "y" * 1900


def test_notifier_selection_and_delivery(monkeypatch):
    sent = []

    class Resp:
        status_code = 204

    monkeypatch.setattr(
        alerts.requests, "post", lambda url, json, timeout: sent.append((url, json)) or Resp()
    )
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    n = alerts.from_env()
    assert isinstance(n, alerts.DiscordNotifier) and n.send("hello")
    assert sent == [("https://discord.test/hook", {"content": "hello"})]
    monkeypatch.delenv("DISCORD_WEBHOOK_URL")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    assert isinstance(alerts.from_env(), alerts.TelegramNotifier)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert isinstance(alerts.from_env(), alerts.LogNotifier)


def test_a_failed_alert_never_raises(monkeypatch):
    def boom(*a, **k):
        raise alerts.requests.ConnectionError("down")

    monkeypatch.setattr(alerts.requests, "post", boom)
    assert alerts.DiscordNotifier("https://discord.test/hook").send("x") is False


@pytest.mark.db
def test_the_same_problem_alerts_once():
    from tests.test_revalidation import _reset

    _reset(with_schedule=False)
    rec = alerts.RecordingNotifier()
    with session_scope() as s:
        assert alerts.once(s, "stale:2026-10-20:g1", NOW, rec, "stale")
        assert not alerts.once(s, "stale:2026-10-20:g1", NOW + timedelta(minutes=15), rec, "stale")
        assert alerts.once(s, "stale:2026-10-21:g1", NOW, rec, "stale again")
    assert rec.sent == ["stale", "stale again"]


@pytest.mark.db
def test_clv_is_read_across_when_the_market_moves_past_the_line():
    """Bet Cade over 26.5 at +100; by the close every book has moved to 27.5. Nobody quotes
    26.5 any more, but this is exactly the pick the market agreed with: its CLV comes from
    the 27.5 consensus read across with Cade's projection (mean 31)."""
    from nbaprops.db.models import OddsSnapshot
    from nbaprops.pricing import engine
    from tests.test_slate import TIP1, grid

    settle_setup()
    late = TIP1 - timedelta(minutes=10)
    with session_scope() as s:
        for book, o, u in (
            ("draftkings", -110, -110),
            ("fanduel", -110, -110),
            ("bovada", -110, -110),
        ):
            s.add(
                OddsSnapshot(
                    snapshot_ts=late,
                    provider="propline",
                    book=book,
                    game_id=G1,
                    player_id=CADE,
                    player_name_raw="Cade",
                    market="pts",
                    line_type="main",
                    line=Decimal("27.5"),
                    over_price=o,
                    under_price=u,
                )
            )
    with session_scope() as s:
        settle(s, NEXT_DAY, MORNING)
        cade = s.scalars(select(Pick).where(Pick.player_id == CADE)).one()
        expect = engine.shifted_tail(grid(31), 26.5, 27.5, 0.5) - 0.5  # took +100
        assert cade.clv is not None and float(cade.clv) == pytest.approx(expect, abs=1e-4)
        assert float(cade.clv) > 0  # the market moved his way


@pytest.mark.db
def _a_later_game_with_stats(days_after: int = 2) -> None:
    """Stats loading is working: a game days after G3 has its box score."""
    from datetime import UTC, datetime

    from nbaprops.db.models import Game

    d = DAY + timedelta(days=days_after)
    with session_scope() as s:
        s.add(
            Game(
                game_id="0022600099",
                season="2026-27",
                season_type="regular",
                game_date=d,
                tip_time=datetime(d.year, d.month, d.day, 23, 0, tzinfo=UTC),
                home_team_id=1610612765,
                away_team_id=1610612738,
                status="final",
            )
        )
        s.flush()
        s.add(
            PlayerGameLog(
                game_id="0022600099",
                player_id=CADE,
                team_id=1610612765,
                minutes=30,
                pts=20,
                reb=4,
                ast=6,
                fg3m=2,
            )
        )


def test_picks_on_a_game_that_never_finished_are_voided():
    """G3 never becomes final. It's only treated as unplayed once later games' stats are in:
    without them (a missed morning of `stats-day`), its pick waits instead of being voided."""
    from tests.test_slate import TIP1

    settle_setup()  # G3 (Wembanyama's game) never became final
    with session_scope() as s:
        early = settle(s, NEXT_DAY, TIP1 + timedelta(hours=20))
        assert s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one().result is None
    with session_scope() as s:
        # G3 tips 6.5 hours after G1, so 44 hours after G1's tip is 37.5 after G3's
        waiting = settle(s, NEXT_DAY + timedelta(days=1), TIP1 + timedelta(hours=44))
        assert s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one().result is None
    _a_later_game_with_stats()
    with session_scope() as s:
        later = settle(s, NEXT_DAY + timedelta(days=2), TIP1 + timedelta(hours=68))
        wemby = s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one()
        assert wemby.result == "void" and float(wemby.pnl) == 0
    assert early.record["void"] == 1  # Duren (did not play in a loaded box score)
    assert waiting.record["void"] == 0 and later.record["void"] == 1  # then Wembanyama


def test_voids_made_only_for_missing_stats_are_reopened():
    """The bug: two days without `stats-day` and the old 36-hour rule voided every pick on
    normal games. The repair reopens exactly those picks and their parlays, and never a real
    did-not-play void (that one needs the game's box score)."""
    from nbaprops.ops.settle import unvoid_missing_stats
    from tests.test_slate import TIP1

    settle_setup()
    with session_scope() as s:
        settle(s, NEXT_DAY, TIP1 + timedelta(hours=20))  # Duren: a real void (G1 is loaded)
        wemby = s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one()
        wemby.result, wemby.pnl = "void", Decimal("0")  # what the old rule did
        par = Parlay(
            created_ts=NOW,
            slate_date=DAY,
            tier="satellite",
            mode="paper",
            same_game=False,
            joint_prob=Decimal("0.3"),
            dk_price=250,
            ev=Decimal("0.05"),
            stake=Decimal("5"),
            result="void",
            pnl=Decimal("0"),
        )
        s.add(par)
        s.flush()
        s.add(ParlayLeg(parlay_id=par.parlay_id, pick_id=wemby.pick_id))
        par_id = par.parlay_id
    with session_scope() as s:
        assert unvoid_missing_stats(s) == (1, 1)
    with session_scope() as s:
        assert s.scalars(select(Pick).where(Pick.player_id == WEMBY)).one().result is None
        assert s.get(Parlay, par_id).result is None
        voids = s.scalars(select(Pick.result).where(Pick.result == "void")).all()
        assert voids == ["void"]  # Duren's real void stays
