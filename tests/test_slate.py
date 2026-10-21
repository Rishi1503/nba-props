"""price_slate against Postgres: picks, parlays, reruns, freshness, guards."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from scipy.stats import poisson
from sqlalchemy import func, select, text

from nbaprops.config import load_strategy
from nbaprops.db.base import session_scope
from nbaprops.db.models import (
    BankrollEntry,
    ModelRun,
    OddsPoll,
    OddsSnapshot,
    Parlay,
    ParlayLeg,
    Pick,
    Projection,
)
from nbaprops.pricing import slate
from nbaprops.settings import get_settings
from tests.test_revalidation import _reset

DAY = date(2026, 10, 20)
G1, G3 = "0022600001", "0022600003"  # DET-BOS 19:00 UTC, SAS-OKC 01:30 UTC next day
CADE, BROWN, WEMBY, SGA = 1630595, 1627759, 1641705, 1628983
TIP1 = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)
POLL = TIP1 - timedelta(hours=9)  # 10:00 UTC
NOW = TIP1 - timedelta(hours=8)
CFG = load_strategy(get_settings().strategy_config)


def grid(mean):
    return {f"{k + 0.5:.1f}": round(float(poisson.sf(k, mean)), 4) for k in range(0, 60)}


def seed(polls=True, bankroll=True, fd_fresh=True):
    _reset(with_schedule=True)
    with session_scope() as s:
        s.add(
            ModelRun(
                run_id="r1",
                git_sha="test",
                trained_through=date(2026, 6, 20),
                config={},
                status="champion",
            )
        )
        s.flush()
        for gid, pid, mean in ((G1, CADE, 31), (G1, BROWN, 24.6), (G3, WEMBY, 27), (G3, SGA, 33)):
            s.add(
                Projection(
                    run_id="r1",
                    game_id=gid,
                    player_id=pid,
                    market="pts",
                    as_of_ts=NOW - timedelta(minutes=30),
                    mean=Decimal(str(mean)),
                    sd=Decimal("6"),
                    p_over=grid(mean),
                    low_confidence=False,
                )
            )
        quotes = [  # (game, player, line, DK over/under, consensus over/under)
            (G1, CADE, 26.5, 100, -120, -110, -110),
            (G1, BROWN, 24.5, -110, -110, -110, -110),
            (G3, WEMBY, 23.5, 100, -120, -110, -110),
            (G3, SGA, 29.5, 100, -120, -110, -110),
        ]
        for gid, pid, line, dko, dku, co, cu in quotes:
            for book, o, u in (("draftkings", dko, dku), ("fanduel", co, cu), ("bovada", co, cu)):
                s.add(
                    OddsSnapshot(
                        snapshot_ts=POLL,
                        provider="propline",
                        book=book,
                        game_id=gid,
                        player_id=pid,
                        player_name_raw=str(pid),
                        market="pts",
                        line_type="main",
                        line=Decimal(str(line)),
                        over_price=o,
                        under_price=u,
                    )
                )
        s.add(
            OddsSnapshot(
                snapshot_ts=POLL,
                provider="propline",
                book="draftkings",
                game_id=G1,
                player_id=CADE,
                player_name_raw="Cade",
                market="pts",
                line_type="milestone",
                line=Decimal("29.5"),
                over_price=150,
                under_price=None,
            )
        )
        if polls:
            for gid in (G1, G3):
                books = ["bovada", "draftkings", "fanduel"]
                s.add(
                    OddsPoll(
                        game_id=gid, snapshot_ts=POLL, provider="propline", books=books, n_props=10
                    )
                )
                if not fd_fresh:  # a later poll where FanDuel didn't answer
                    s.add(
                        OddsPoll(
                            game_id=gid,
                            snapshot_ts=NOW - timedelta(minutes=5),
                            provider="propline",
                            books=["bovada", "draftkings"],
                            n_props=8,
                        )
                    )
        if bankroll:
            s.add(BankrollEntry(entry_date=DAY - timedelta(days=1), bankroll=Decimal("1000")))


def count(model):
    with session_scope() as s:
        return s.scalar(select(func.count()).select_from(model))


@pytest.mark.db
def test_prices_the_slate_and_writes_picks_and_a_satellite():
    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG)
    assert res.run_id == "r1" and res.bankroll == 1000 and not res.skipped
    by_player = res.picks.set_index("player_id")
    assert {CADE, WEMBY, SGA} <= set(by_player.index) and BROWN not in by_player.index
    assert (res.picks["stake"] <= 20.0 + 1e-9).all()  # 2% of 1000
    assert res.picks.groupby("game_id")["stake"].sum().max() <= 40 + 1e-9
    with session_scope() as s:
        rows = s.scalars(select(Pick)).all()
        assert len(rows) == len(res.picks)
        cade = next(r for r in rows if r.player_id == CADE)
        assert cade.mode == "paper" and cade.run_id == "r1" and cade.p_consensus is not None
        assert cade.flagged_ts == NOW and cade.line_type in ("main", "milestone")
        sats = s.scalars(select(Parlay)).all()
        assert sats and all(p.tier == "satellite" and p.slate_date == DAY for p in sats)
        legs = s.execute(
            select(ParlayLeg.parlay_id, Pick.game_id).join(Pick, Pick.pick_id == ParlayLeg.pick_id)
        ).all()
        for pid in {p.parlay_id for p in sats}:
            assert len({g for q, g in legs if q == pid}) == 2  # different games


@pytest.mark.db
def test_reruns_keep_the_first_flag_and_respect_earlier_exposure():
    seed()
    with session_scope() as s:
        first = slate.price_slate(s, DAY, NOW, CFG)
    n_picks, n_parlays = count(Pick), count(Parlay)
    with session_scope() as s:
        again = slate.price_slate(s, DAY, NOW + timedelta(minutes=30), CFG)
    assert again.picks.empty  # every flagged player already has today's pick
    assert count(Pick) == n_picks and count(Parlay) == n_parlays
    with session_scope() as s:
        flagged = {r.player_id: r.flagged_ts for r in s.scalars(select(Pick))}
    assert set(flagged.values()) == {NOW} and len(first.picks) == n_picks


@pytest.mark.db
def test_a_stale_board_is_not_priced():
    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, POLL + timedelta(hours=3), CFG)  # > 120 min old
    assert res.picks.empty
    stale = [m for m in res.skipped if "older than 120 min" in m]
    assert sorted(m.split(":")[0] for m in stale) == [G1, G3]


@pytest.mark.db
def test_a_silent_consensus_book_drops_out_of_the_consensus():
    seed(fd_fresh=False)
    with session_scope() as s:  # FanDuel last answered 125 min ago (limit 120); DK 70 min
        res = slate.price_slate(s, DAY, POLL + timedelta(minutes=125), CFG, write=False)
    # FanDuel is dropped: one consensus book left, so picks are model-only
    assert len(res.picks) and not res.picks["has_consensus"].any()
    assert (res.picks["ev"] >= CFG.ev_min_model_only - 1e-12).all()


@pytest.mark.db
def test_started_games_and_missing_bankroll():
    seed()
    with session_scope() as s:
        later = slate.price_slate(s, DAY, TIP1 + timedelta(minutes=1), CFG, write=False)
    assert G1 not in set(later.picks.get("game_id", []))  # tipped: never priced
    seed(bankroll=False)
    with session_scope() as s, pytest.raises(slate.NoBankroll):
        slate.price_slate(s, DAY, NOW, CFG)


@pytest.mark.db
def test_a_boost_needs_three_different_games():
    """Two games on this slate: a boost parlay needs 3-4 legs from different games, so none
    is built (a successful boost is covered in test_parlays)."""
    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, boosts=[(0.5, 10.0)], write=False)
    assert not [p for p in res.parlays if p.tier == "boost"]


@pytest.mark.db
def test_no_projections_means_nothing_is_priced():
    seed()
    with session_scope() as s:
        s.execute(text("DELETE FROM projections"))
        res = slate.price_slate(s, DAY, NOW, CFG)
    assert res.run_id is None and "no projections" in res.skipped[0]


@pytest.mark.db
def test_manual_milestone_uses_the_same_milestone_logic():
    """Cade 30+ (line 29.5) typed in at +150. The consensus main line is 26.5 at -110/-110
    (a coin flip), so the model distribution (mean 31) is shifted to match it there."""
    seed()
    with session_scope() as s:
        typed = slate.price_manual_milestone(s, G1, CADE, "pts", 29.5, 150, NOW, CFG)
    expect = slate.engine.shifted_tail(grid(31), 29.5, 26.5, 0.5)
    assert typed["has_consensus"]
    assert typed["p_fair"] == pytest.approx(expect, abs=1e-4)
    assert typed["ev"] == pytest.approx(expect * 1.5 - (1 - expect), abs=1e-4)  # d = 2.5
    assert (typed["stake"] > 0) == typed["flag"]


@pytest.mark.db
def test_command_renders_the_board(capsys, monkeypatch):
    seed()
    from nbaprops.pricing import run

    real = slate.price_slate
    monkeypatch.setattr(
        slate,
        "price_slate",
        lambda s, d, now, cfg, mode, boosts, write: real(s, d, NOW, cfg, mode, boosts, write),
    )
    res = run.main(["--date", str(DAY), "--dry-run"])
    out = capsys.readouterr().out
    assert "bankroll 1,000.00" in out and "satellite:" in out and len(res.picks) >= 3
    assert count(Pick) == 0  # dry run wrote nothing


@pytest.mark.db
def test_stakes_on_games_that_already_tipped_still_count_toward_the_day():
    """A 6 PM run must count the stakes placed on an earlier game that has since tipped."""
    seed()
    with session_scope() as s:
        s.add(
            Pick(
                run_id="r1",
                game_id=G1,
                player_id=BROWN,
                market="pts",
                line=Decimal("24.5"),
                side="over",
                tier="standard",
                mode="paper",
                flagged_ts=NOW,
                dk_price=-110,
                fair_prob=Decimal("0.55"),
                ev=Decimal("0.05"),
                stake=Decimal("95"),
            )
        )
    late = TIP1 + timedelta(minutes=30)  # G1 has tipped; only G3 is left to price
    with session_scope() as s:
        s.add(
            OddsPoll(
                game_id=G3,
                snapshot_ts=late - timedelta(minutes=5),
                provider="propline",
                books=["bovada", "draftkings", "fanduel"],
                n_props=8,
            )
        )
    with session_scope() as s:
        res = slate.price_slate(s, DAY, late, CFG, write=False)
    assert res.picks["stake"].sum() + sum(p.stake for p in res.parlays) <= 5.0 + 1e-9


@pytest.mark.db
def test_a_game_with_injury_news_after_its_projections_is_not_priced():
    from nbaprops.db.models import InjurySnapshot

    seed()
    with session_scope() as s:
        for ts, status in (
            (NOW - timedelta(hours=2), "Questionable"),
            (NOW - timedelta(minutes=10), "Out"),
        ):  # after the 10:30 projection
            s.add(
                InjurySnapshot(
                    snapshot_ts=ts,
                    report_ts=ts,
                    game_id=G1,
                    team_id=1610612738,
                    player_id=BROWN,
                    player_name_raw="Brown, Jaylen",
                    status=status,
                    reason="Injury",
                )
            )
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, write=False)
    assert G1 not in set(res.picks["game_id"]) and G3 in set(res.picks["game_id"])
    note = next(m for m in res.skipped if m.startswith(G1))
    assert f"Jaylen Brown ({BROWN}): Questionable -> Out" in note


@pytest.mark.db
def test_tracked_standard_flags_are_written_with_zero_stake():
    seed()
    assert CFG.stake_standard_flags is False  # the approved strategy
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG)
    with session_scope() as s:
        rows = [(r.tier, float(r.stake)) for r in s.scalars(select(Pick))]
    assert len(rows) == len(res.picks)
    assert all(stake == 0 for tier, stake in rows if tier == "standard")
    assert all(stake > 0 for tier, stake in rows if tier == "high_conf")


@pytest.mark.db
def test_a_boosted_single_is_written_as_a_one_leg_boost():
    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, boosts=[(0.5, 25.0, "single")])
    boosted = [p for p in res.parlays if p.tier == "boost"]
    assert len(boosted) == 1 and len(boosted[0].legs) == 1 and boosted[0].stake <= 25.0
    with session_scope() as s:
        par = s.scalars(select(Parlay).where(Parlay.tier == "boost")).one()
        legs = s.scalars(
            select(Pick)
            .join(ParlayLeg, ParlayLeg.pick_id == Pick.pick_id)
            .where(ParlayLeg.parlay_id == par.parlay_id)
        ).all()
        assert float(par.boost_pct) == 0.5 and len(legs) == 1
        assert legs[0].stake == 0 and legs[0].tier in ("boost_leg", "standard")
        per_player = s.execute(
            select(Pick.game_id, Pick.player_id, func.count()).group_by(
                Pick.game_id, Pick.player_id
            )
        ).all()
        assert all(n == 1 for *_, n in per_player)  # still one pick per player per game


@pytest.mark.db
def test_the_slate_reports_props_blocked_only_by_low_confidence():
    seed()
    with session_scope() as s:
        s.execute(text(f"UPDATE projections SET low_confidence = true WHERE player_id = {SGA}"))
        res = slate.price_slate(s, DAY, NOW, CFG, write=False)
    assert SGA not in set(res.picks["player_id"]) and res.blocked_low_confidence >= 1


@pytest.mark.db
def test_questionable_teammate_scenarios_flow_through_pricing():
    seed()
    # Wembanyama over 23.5: about 10% if Brown plays (mean 18), near-certain if he sits
    fragile = {"player_id": BROWN, "p_play": 0.5, "in": grid(18), "out": grid(37)}
    sturdy = {"player_id": BROWN, "p_play": 0.5, "in": grid(30), "out": grid(32)}
    with session_scope() as s:
        for pid, sc in ((CADE, sturdy), (WEMBY, fragile)):
            s.execute(
                select(Projection).where(Projection.player_id == pid)
            ).scalar_one().scenarios = sc
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, write=False)
    by = res.picks.set_index("player_id")
    assert CADE in by.index and by.loc[CADE, "ev_if_plays"] >= 0  # +EV either way
    assert WEMBY not in by.index and res.blocked_questionable >= 1  # only if Brown sits


@pytest.mark.db
def test_the_printed_board_shows_player_names(capsys):
    from nbaprops.pricing import run as prun

    seed()
    with session_scope() as s:
        res = slate.price_slate(s, DAY, NOW, CFG, write=False)
    board = prun.render(res, {CADE: "Cade Cunningham"})
    assert "Cade Cunningham" in board and f"{CADE} pts" not in board
