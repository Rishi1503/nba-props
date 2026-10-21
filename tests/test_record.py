"""The Record page: daily results with parlay legs, success rates by period, and the
"tail every pick at a flat stake" calculator. Expected values are worked by hand.

The Stage 9 synthetic season: 350 settled picks over 35 days from Monday Nov 2, all at -150,
winning exactly 3 in 5. At $10 a win pays 10 * 100/150 = $6.67, so every 5 picks net
3 * 6.67 - 2 * 10 = $0: picks alone break even exactly.
"""

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import Game, Parlay, ParlayLeg, Pick
from tests.test_api import client  # noqa: F401  (fixture)
from tests.test_slate import NOW
from tests.test_stage9 import START, history

D = 1 + 100 / 150  # decimal odds at -150


def seed_record():
    """The season, plus on day one: a two-leg parlay that cashed (both legs won), one that
    lost (one leg lost), and a tracked (unstaked) pick that won at +100."""
    history()
    with session_scope() as s:
        day0 = s.scalars(select(Game).where(Game.game_date == START)).one()
        picks = s.scalars(
            select(Pick).where(Pick.game_id == day0.game_id).order_by(Pick.player_id)
        ).all()
        assert [p.result for p in picks[:5]] == ["win", "win", "win", "loss", "loss"]
        for legs, result in (((0, 1), "win"), ((0, 3), "loss")):
            pnl = 5 * (D * D - 1) if result == "win" else -5
            par = Parlay(
                created_ts=NOW,
                slate_date=START,
                tier="satellite",
                mode="paper",
                same_game=False,
                joint_prob=Decimal("0.36"),
                dk_price=178,
                ev=Decimal("0.05"),
                stake=Decimal("5"),
                result=result,
                pnl=Decimal(f"{pnl:.2f}"),
            )
            s.add(par)
            s.flush()
            for i in legs:
                s.add(ParlayLeg(parlay_id=par.parlay_id, pick_id=picks[i].pick_id))
        taken = [p.player_id for p in picks]
        other = s.execute(
            text(
                "SELECT player_id FROM players WHERE NOT (player_id = ANY(:t)) "
                "ORDER BY player_id LIMIT 1"
            ),
            {"t": taken},
        ).scalar()
        s.add(
            Pick(
                run_id="r1",
                game_id=day0.game_id,
                player_id=other,
                market="ast",
                line=Decimal("4.5"),
                side="over",
                tier="standard",
                mode="paper",
                flagged_ts=NOW,
                dk_price=100,
                fair_prob=Decimal("0.55"),
                ev=Decimal("0.1"),
                stake=Decimal("0"),
                result="win",
                pnl=Decimal("0"),
            )
        )


@pytest.mark.db
def test_tail_every_pick_at_a_flat_stake(client):  # noqa: F811
    seed_record()
    picks_only = client.get("/tail", params={"stake": 10, "include_parlays": "false"}).json()
    assert picks_only["bets"] == 350 and picks_only["pending"] == 0
    assert picks_only["profit"] == pytest.approx(0.0, abs=0.01)  # 3 wins pay for 2 losses
    assert picks_only["wagered"] == 3500 and picks_only["record"]["win"] == 210
    both = client.get("/tail", params={"stake": 10}).json()
    expect = 10 * (D * D - 1) - 10  # the cashed parlay, minus the lost one
    assert both["profit"] == pytest.approx(expect, abs=0.01) and both["parlays"] == 2
    assert both["wagered"] == 3520 and both["roi"] == pytest.approx(expect / 3520, abs=1e-4)
    assert both["daily"][0]["date"] == str(START)
    assert both["daily"][-1]["cumulative"] == pytest.approx(both["profit"], abs=0.01)
    tracked = client.get("/tail", params={"stake": 10, "include_tracked": "true"}).json()
    assert tracked["profit"] == pytest.approx(expect + 10, abs=0.01)  # +100 win at $10
    big = client.get("/tail", params={"stake": 25, "include_parlays": "false"}).json()
    assert big["wagered"] == 25 * 350  # scales with the stake
    assert client.get("/tail", params={"stake": 0}).status_code == 422


@pytest.mark.db
def test_success_rates_by_week_month_and_year(client):  # noqa: F811
    seed_record()
    weeks = client.get("/success", params={"period": "week"}).json()
    assert len(weeks) == 5 and weeks[-1]["period"] == str(START)  # newest first
    assert all(w["picks"]["hit_rate"] == pytest.approx(0.6) for w in weeks)
    first = weeks[-1]
    assert first["picks"]["won"] == 42 and first["picks"]["lost"] == 28
    assert first["parlays"] == {"won": 1, "lost": 1, "hit_rate": 0.5}
    assert first["legs"] == {"won": 3, "lost": 1, "hit_rate": 0.75}
    months = {
        m["period"]: m["picks"] for m in client.get("/success", params={"period": "month"}).json()
    }
    assert months["2026-11"]["won"] == 174 and months["2026-12"]["won"] == 36
    years = client.get("/success", params={"period": "year"}).json()
    assert years[0]["period"] == "2026" and years[0]["picks"]["won"] == 210
    with_tracked = client.get(
        "/success", params={"period": "year", "include_tracked": "true"}
    ).json()
    assert with_tracked[0]["picks"]["won"] == 211
    assert client.get("/success", params={"period": "day"}).status_code == 422


@pytest.mark.db
def test_daily_record_shows_every_leg(client):  # noqa: F811
    seed_record()
    days = client.get(
        "/record", params={"start": str(START), "end": str(START + timedelta(days=40))}
    ).json()
    assert len(days) == 35 and days[0]["date"] > days[-1]["date"]  # newest first
    first = days[-1]
    assert first["date"] == str(START)
    assert first["record"] == {"win": 6, "loss": 4, "push": 0, "void": 0}  # staked only
    assert len(first["picks"]) == 11  # 10 bets + the tracked pick
    won, lost = sorted(first["parlays"], key=lambda p: p["result"], reverse=True)
    assert won["result"] == "win" and won["legs_hit"] == 2
    assert [leg["result"] for leg in lost["legs"]] == ["win", "loss"]
    assert all(leg["player"] for p in first["parlays"] for leg in p["legs"])
