"""Stage 10 API: login, every read endpoint, every write action."""

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from nbaprops.api import auth
from nbaprops.api.main import create_app
from nbaprops.db.base import session_scope
from nbaprops.db.models import BacktestReport, InjurySnapshot, OddsSnapshot, Parlay, Pick
from nbaprops.pricing import slate
from tests.test_slate import BROWN, CADE, CFG, DAY, G1, G3, NOW, WEMBY, seed

PASSWORD = "correct horse battery staple"


def fake_draws(session, game_id, now, cfg):
    rng = np.random.default_rng(0)
    base = rng.normal(28, 6, 20000)
    return {
        (G1, CADE): {"pts": np.round(base)},
        (G1, BROWN): {"pts": np.round(base * 0.8 + rng.normal(0, 3, 20000))},
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", auth.hash_password(PASSWORD))
    monkeypatch.setenv("DASHBOARD_SECRET", "s" * 48)
    app = create_app(
        session_factory=session_scope,
        cfg_loader=lambda: CFG,
        game_draws=fake_draws,
        clock=lambda: NOW,
    )
    c = TestClient(app)
    token = c.post("/login", json={"password": PASSWORD}).json()["token"]
    c.headers["Authorization"] = f"Bearer {token}"
    return c


def priced():
    seed()
    with session_scope() as s:
        slate.price_slate(s, DAY, NOW, CFG)


# --- auth -----------------------------------------------------------------------------


def test_passwords_and_tokens():
    stored = auth.hash_password(PASSWORD)
    assert auth.verify_password(PASSWORD, stored) and not auth.verify_password("nope", stored)
    assert not auth.verify_password(PASSWORD, "garbage")
    token, exp = auth.make_token("k", now=1000.0, days=1)
    assert auth.check_token(token, "k", now=1000.0 + 3600)
    assert not auth.check_token(token, "k", now=exp + 1)  # expired
    assert not auth.check_token(token, "other", now=1000.0)  # wrong key
    payload, sig = token.split(".")
    assert not auth.check_token(payload + "." + sig[:-2] + "AA", "k", now=1000.0)  # tampered


@pytest.mark.db
def test_login_guards(monkeypatch):
    seed()
    app = create_app(session_factory=session_scope, cfg_loader=lambda: CFG, clock=lambda: NOW)
    c = TestClient(app)
    assert c.post("/login", json={"password": PASSWORD}).status_code == 503  # not configured
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", auth.hash_password(PASSWORD))
    monkeypatch.setenv("DASHBOARD_SECRET", "s" * 48)
    assert c.get(f"/picks/{DAY}").status_code == 401  # no token
    assert c.get(f"/picks/{DAY}", headers={"Authorization": "Bearer junk"}).status_code == 401
    for _ in range(auth.MAX_FAILURES):
        assert c.post("/login", json={"password": "wrong"}).status_code == 401
    assert c.post("/login", json={"password": PASSWORD}).status_code == 429  # locked out
    assert c.get("/health").json()["champion"] == "r1"  # health stays open


# --- reads ------------------------------------------------------------------------------


@pytest.mark.db
def test_slate_with_injuries(client):
    seed()
    with session_scope() as s:
        s.add(
            InjurySnapshot(
                snapshot_ts=NOW - timedelta(hours=1),
                report_ts=NOW - timedelta(hours=1),
                game_id=G1,
                team_id=1610612738,
                player_id=BROWN,
                player_name_raw="Brown, Jaylen",
                status="Questionable",
                reason="Knee",
            )
        )
    body = client.get(f"/slate/{DAY}").json()
    games = {g["game_id"]: g for g in body["games"]}
    assert set(games) == {G1, G3}
    assert (games[G1]["home"], games[G1]["away"]) == ("DET", "BOS")
    assert games[G1]["injuries"] == [
        {"player_id": BROWN, "name": "Jaylen Brown", "status": "Questionable"}
    ]


@pytest.mark.db
def test_picks_with_names_and_movement_badges(client):
    priced()
    later = NOW + timedelta(minutes=30)
    with session_scope() as s:  # DraftKings moves Cade's line after the flag
        s.add(
            OddsSnapshot(
                snapshot_ts=later,
                provider="propline",
                book="draftkings",
                game_id=G1,
                player_id=CADE,
                player_name_raw="Cade",
                market="pts",
                line_type="main",
                line=Decimal("27.5"),
                over_price=-115,
                under_price=-105,
            )
        )
    app = create_app(session_factory=session_scope, cfg_loader=lambda: CFG, clock=lambda: later)
    c = TestClient(app, headers=dict(client.headers))
    picks = {p["player_id"]: p for p in c.get(f"/picks/{DAY}").json()}
    cade = picks[CADE]
    assert cade["player"] == "Cade Cunningham" and cade["moved"]
    assert cade["line_now"] == 27.5 and cade["price_now"] == -115
    assert all(not p["moved"] for pid, p in picks.items() if pid != CADE)


@pytest.mark.db
def test_parlays_and_player_detail(client):
    priced()
    pars = client.get(f"/parlays/{DAY}").json()
    assert pars and all(len(p["legs"]) == 2 and p["payout_decimal"] > 1 for p in pars)
    assert all(leg["player"] for p in pars for leg in p["legs"])
    detail = client.get(f"/players/{CADE}/projection", params={"day": str(DAY)}).json()
    hist = detail["markets"]["pts"]["histogram"]
    assert detail["name"] == "Cade Cunningham"
    assert sum(h["p"] for h in hist) == pytest.approx(1.0, abs=0.01)
    assert max(hist, key=lambda h: h["p"])["value"] in range(28, 33)  # mean 31
    assert len(detail["line_history"]) >= 1


@pytest.mark.db
def test_performance_review_and_export(client):
    from tests.test_stage9 import TODAY, history

    history()
    perf = client.get("/performance", params={"start": "2026-11-01", "end": str(TODAY)}).json()
    assert perf["summary"]["bets"] == 350 and perf["record"]["win"] == 210
    assert set(perf["by_market"]) == {"pts", "reb"} and perf["calibration"]
    reb = client.get(
        "/performance", params={"start": "2026-11-01", "end": str(TODAY), "market": "reb"}
    ).json()
    assert reb["summary"]["bets"] == 175
    csv_text = client.get("/export/2026", params={"mode": "paper"}).text
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("date,kind,player") and len(lines) == 1 + 350
    rv = client.get("/review").json()
    assert "review" in rv and "golive" in rv


@pytest.mark.db
def test_backtest_reports(client):
    seed()
    with session_scope() as s:
        s.add(
            BacktestReport(
                run_id="bt-1",
                created_ts=NOW,
                season="2025-26",
                holdout="clean",
                report={"layer_a": {"targets": {"passes": True}}},
            )
        )
    assert [b["run_id"] for b in client.get("/backtests").json()] == ["bt-1"]
    assert client.get("/backtests/bt-1").json()["layer_a"]["targets"]["passes"] is True
    assert client.get("/backtests/nope").status_code == 404


# --- writes ---------------------------------------------------------------------------------


@pytest.mark.db
def test_marking_picks_and_parlays_placed(client):
    priced()
    with session_scope() as s:
        pick_id = s.scalars(select(Pick.pick_id).where(Pick.player_id == CADE)).one()
        parlay_id = s.scalars(select(Parlay.parlay_id)).first()
    r = client.post(f"/picks/{pick_id}/placed", json={"price": 105, "stake": 15})
    assert r.status_code == 200
    assert (
        client.post(f"/picks/{pick_id}/placed", json={"price": 105, "stake": 15}).status_code == 409
    )
    assert client.post("/picks/999999/placed", json={"price": 105, "stake": 15}).status_code == 404
    picks = {p["player_id"]: p for p in client.get(f"/picks/{DAY}").json()}
    assert picks[CADE]["real_price"] == 105 and picks[CADE]["real_stake"] == 15
    r = client.post(f"/parlays/{parlay_id}/placed", json={"price": 250, "stake": 5})
    assert r.status_code == 200
    with session_scope() as s:
        real = s.get(Parlay, r.json()["real_parlay_id"])
        assert real.mode == "real" and real.price_placed == 250
        n = s.execute(
            text("SELECT count(*) FROM parlay_legs WHERE parlay_id = :p"), {"p": real.parlay_id}
        ).scalar()
        assert n == 2


@pytest.mark.db
def test_boost_and_manual_prices(client):
    seed()
    r = client.post("/boosts", json={"pct": 0.5, "max_stake": 25, "kind": "single"}).json()
    assert len(r["boosted"]) == 1 and r["boosted"][0]["stake"] <= 25
    m = client.post(
        "/prices/manual/milestone",
        json={"game_id": G1, "player_id": CADE, "market": "pts", "line": 29.5, "price": 150},
    ).json()
    assert {"p_fair", "ev", "flag", "stake"} <= set(m)


@pytest.mark.db
def test_same_game_parlay_pricing(client):
    priced()
    with session_scope() as s:
        ids = dict(s.execute(select(Pick.player_id, Pick.pick_id)).all())
    assert BROWN not in ids  # his 24.5 at -110 isn't +EV on this slate: add a second leg
    with session_scope() as s:
        p = Pick(
            run_id="r1",
            game_id=G1,
            player_id=BROWN,
            market="pts",
            line=Decimal("18.5"),
            side="over",
            tier="standard",
            mode="paper",
            flagged_ts=NOW,
            dk_price=-110,
            fair_prob=Decimal("0.6"),
            ev=Decimal("0.05"),
            stake=Decimal("0"),
        )
        s.add(p)
        s.flush()
        ids[BROWN] = p.pick_id

    r = client.post("/prices/manual/sgp", json={"pick_ids": [ids[CADE], ids[BROWN]], "price": 200})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["joint_prob"] < 1 and "haircut" in body["notes"][0]
    other = client.post(
        "/prices/manual/sgp", json={"pick_ids": [ids[CADE], ids[WEMBY]], "price": 200}
    )
    assert other.status_code == 422  # different games


@pytest.mark.db
def test_market_switch_and_settings(client):
    seed()
    assert client.post("/markets/reb", json={"disabled": True, "reason": "test"}).status_code == 200
    s = client.get("/settings").json()
    assert s["strategy"]["ev_min"] == CFG.ev_min
    assert [m["market"] for m in s["markets"] if m["disabled"]] == ["reb"]
    assert client.post("/markets/nope", json={"disabled": True}).status_code == 404


@pytest.mark.db
def test_cors_allows_only_the_dashboard(monkeypatch):
    seed()
    monkeypatch.setenv("DASHBOARD_ORIGINS", "https://props.example.com")
    c = TestClient(
        create_app(session_factory=session_scope, cfg_loader=lambda: CFG, clock=lambda: NOW)
    )
    ok = c.options(
        "/picks/2026-10-20",
        headers={"Origin": "https://props.example.com", "Access-Control-Request-Method": "GET"},
    )
    bad = c.options(
        "/picks/2026-10-20",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://props.example.com"
    assert "access-control-allow-origin" not in bad.headers


@pytest.mark.db
def test_bet_tracker_running_totals(client):
    from tests.test_stage9 import TODAY, history

    history(per_day=2, days=3)  # 6 settled paper picks: W W W L L W, 10 staked each
    out = client.get(
        "/bets", params={"start": "2026-11-01", "end": str(TODAY), "mode": "paper"}
    ).json()
    assert len(out["bets"]) == 6 and out["totals"]["open"] == 0
    assert out["totals"]["record"] == {"win": 4, "loss": 2, "push": 0, "void": 0}
    assert out["bets"][-1]["running_pnl"] == pytest.approx(4 * 6.67 - 2 * 10, abs=0.01)
    assert client.get("/bets").json()["bets"] == []  # no real bets yet


@pytest.mark.db
def test_every_migration_downgrades_and_upgrades_with_data_present():
    """CI's migration check runs on an empty database; this one runs on a populated one
    (demo data: unknown starts, withdrawals, parlays, projections with scenarios...).
    Migration 0003's downgrade failed here before it handled unknown starts."""
    import os
    import subprocess
    from pathlib import Path

    from nbaprops.api import demo
    from tests.test_revalidation import _reset

    _reset(with_schedule=False)
    demo.seed()
    with session_scope() as s:
        s.execute(
            text(
                "UPDATE player_game_logs SET started = NULL WHERE player_id IN "
                "(SELECT player_id FROM player_game_logs LIMIT 3)"
            )
        )
    root = Path(__file__).resolve().parents[1]
    for step in (["downgrade", "base"], ["upgrade", "head"]):
        r = subprocess.run(
            ["alembic", *step], cwd=root, env=dict(os.environ), capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-800:]
    _reset(with_schedule=False)  # leave a clean database for the next test
