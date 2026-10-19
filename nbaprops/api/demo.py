"""Demo data, to click through the dashboard before real data exists:
`python -m nbaprops.api.demo` (needs an empty, migrated database).

Fills: teams and players; a five-week settled paper season (CLV, results); today's three
games with projections and odds from DraftKings and two consensus books, priced by the real
engine (so the board shows genuine picks and satellites); a Questionable listing; a stored
backtest report; a bankroll. Refuses to run on a database that already has real picks.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
from scipy.stats import poisson
from sqlalchemy import select, text

from nbaprops.db.base import session_scope
from nbaprops.db.models import (
    BacktestReport,
    BankrollEntry,
    Game,
    InjurySnapshot,
    ModelRun,
    OddsPoll,
    OddsSnapshot,
    Pick,
    PlayerGameLog,
    Projection,
)
from nbaprops.ops.cycle import ET
from nbaprops.pipeline import reference

TEAMS = [("BOS", "NYK"), ("DEN", "LAL"), ("OKC", "SAS")]
MARKETS_MEAN = {"pts": 1.0, "reb": 0.35, "ast": 0.3}


def _grid(mean: float) -> dict:
    hi = int(mean * 2.5) + 10
    return {f"{k + 0.5:.1f}": round(float(poisson.sf(k, mean)), 4) for k in range(hi)}


def _american(p: float) -> int:
    d = 1 / p
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def seed(now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    today = now.astimezone(ET).date()
    rng = np.random.default_rng(7)
    with session_scope() as s:
        if s.scalar(select(Pick.pick_id).where(Pick.mode == "real").limit(1)):
            raise SystemExit("This database has real bets: the demo only runs on an empty one.")
        reference.seed_teams(s)
        reference.seed_players(s)
        abbr = dict(s.execute(text("SELECT abbreviation, team_id FROM teams")).all())
        players = (
            s.execute(
                text("SELECT player_id FROM players WHERE is_active ORDER BY player_id LIMIT 60")
            )
            .scalars()
            .all()
        )
        s.add(
            ModelRun(
                run_id="demo",
                git_sha="demo",
                trained_through=today - timedelta(days=1),
                config={},
                status="champion",
            )
        )
        s.add(
            BankrollEntry(
                entry_date=today - timedelta(days=40),
                bankroll=Decimal("1000"),
                note="demo bankroll",
            )
        )
        s.flush()
        # five weeks of settled paper picks
        start = today - timedelta(days=36)
        for d in range(35):
            day = start + timedelta(days=d)
            gid = f"9{day:%m%d}0001"[:10]
            s.add(
                Game(
                    game_id=gid,
                    season="2026-27",
                    season_type="regular",
                    game_date=day,
                    tip_time=datetime(day.year, day.month, day.day, 23, tzinfo=UTC),
                    home_team_id=abbr["BOS"],
                    away_team_id=abbr["NYK"],
                    status="final",
                )
            )
            s.flush()
            for i, pid in enumerate(players[:8]):
                fair = float(rng.uniform(0.55, 0.75))
                win = rng.random() < fair
                price = -int(rng.integers(110, 190))
                dec = 1 + 100 / abs(price)
                stake = 10.0 if fair >= 0.7 else 0.0
                s.add(
                    Pick(
                        run_id="demo",
                        game_id=gid,
                        player_id=pid,
                        market=["pts", "reb", "ast"][i % 3],
                        line=Decimal("12.5"),
                        side="over",
                        tier="high_conf" if fair >= 0.7 else "standard",
                        mode="paper",
                        flagged_ts=datetime(day.year, day.month, day.day, 15, tzinfo=UTC),
                        dk_price=price,
                        fair_prob=Decimal(f"{fair:.4f}"),
                        p_model=Decimal(f"{fair:.4f}"),
                        p_consensus=Decimal("0.6"),
                        ev=Decimal(f"{fair * dec - 1:.4f}"),
                        stake=Decimal(f"{stake:.2f}"),
                        result="win" if win else "loss",
                        pnl=Decimal(f"{stake * (dec - 1) if win else -stake:.2f}"),
                        clv=Decimal(f"{rng.normal(0.012, 0.03):.4f}"),
                    )
                )
        # today's slate
        poll = now - timedelta(minutes=20)
        for g, (home, away) in enumerate(TEAMS):
            gid = f"8{today:%m%d}000{g}"[:10]
            tip = (
                datetime.combine(today, datetime.min.time(), ET)
                .replace(hour=19 + g)
                .astimezone(UTC)
            )
            s.add(
                Game(
                    game_id=gid,
                    season="2026-27",
                    season_type="regular",
                    game_date=today,
                    tip_time=tip,
                    home_team_id=abbr[home],
                    away_team_id=abbr[away],
                    status="scheduled",
                )
            )
            s.flush()
            s.add(
                OddsPoll(
                    game_id=gid,
                    snapshot_ts=poll,
                    provider="demo",
                    books=["bovada", "draftkings", "fanduel"],
                    n_props=30,
                )
            )
            for j, pid in enumerate(players[10 + 4 * g : 14 + 4 * g]):
                base = float(rng.uniform(14, 30))
                for market, scale in MARKETS_MEAN.items():
                    mean = base * scale
                    s.add(
                        Projection(
                            run_id="demo",
                            game_id=gid,
                            player_id=pid,
                            market=market,
                            as_of_ts=poll,
                            mean=Decimal(f"{mean:.2f}"),
                            sd=Decimal(f"{np.sqrt(mean) * 1.2:.2f}"),
                            p_over=_grid(mean),
                            low_confidence=False,
                        )
                    )
                    # lines from half a sd above the mean to a full sd below it: the low ones
                    # are the heavy-favorite overs that make high-confidence picks
                    sd = np.sqrt(mean) * 1.2
                    line = np.floor(mean - rng.uniform(-0.5, 1.0) * sd) + 0.5
                    p_model = float(poisson.sf(int(line), mean))
                    p_mkt = float(np.clip(p_model + rng.normal(-0.05, 0.02), 0.15, 0.9))
                    for book, shade in (
                        ("draftkings", rng.normal(-0.05, 0.02)),
                        ("fanduel", 0),
                        ("bovada", 0),
                    ):
                        p = float(np.clip(p_mkt + shade, 0.15, 0.85))
                        s.add(
                            OddsSnapshot(
                                snapshot_ts=poll,
                                provider="demo",
                                book=book,
                                game_id=gid,
                                player_id=pid,
                                player_name_raw=str(pid),
                                market=market,
                                line_type="main",
                                line=Decimal(str(line)),
                                over_price=_american(min(p * 1.045, 0.95)),
                                under_price=_american(min((1 - p) * 1.045, 0.95)),
                            )
                        )
                if g == 0 and j == 0:
                    s.add(
                        InjurySnapshot(
                            snapshot_ts=poll,
                            report_ts=poll,
                            game_id=gid,
                            team_id=abbr[home],
                            player_id=pid,
                            player_name_raw="demo",
                            status="Questionable",
                            reason="Ankle; soreness",
                        )
                    )
            # last games for the player page
            for k in range(1, 11):
                past = f"7{(today - timedelta(days=k)):%m%d}{g}{k % 10}"[:10]
                if s.get(Game, past) is None:
                    s.add(
                        Game(
                            game_id=past,
                            season="2026-27",
                            season_type="regular",
                            game_date=today - timedelta(days=k),
                            status="final",
                            tip_time=tip - timedelta(days=k),
                            home_team_id=abbr[home],
                            away_team_id=abbr[away],
                        )
                    )
                    s.flush()
                for pid in players[10 + 4 * g : 14 + 4 * g]:
                    s.add(
                        PlayerGameLog(
                            game_id=past,
                            player_id=pid,
                            team_id=abbr[home],
                            minutes=Decimal(f"{rng.uniform(24, 38):.1f}"),
                            pts=int(rng.poisson(20)),
                            reb=int(rng.poisson(7)),
                            ast=int(rng.poisson(5)),
                            fg3m=int(rng.poisson(2)),
                        )
                    )
        s.add(
            BacktestReport(
                run_id="bt-demo",
                created_ts=now,
                season="2025-26",
                holdout="clean (demo)",
                report=_demo_report(),
            )
        )
    with session_scope() as s:
        from nbaprops.config import load_strategy
        from nbaprops.pricing import slate
        from nbaprops.settings import get_settings

        res = slate.price_slate(s, today, now, load_strategy(get_settings().strategy_config))
    return {
        "date": str(today),
        "picks_today": int(len(res.picks)),
        "parlays_today": len(res.parlays),
        "notes": res.skipped,
    }


def _demo_report() -> dict:
    markets = {}
    for m, imp in (("pts", 0.021), ("reb", 0.034), ("ast", 0.041), ("fg3m", 0.06), ("pra", 0.02)):
        markets[m] = {
            "accuracy": {"mae_improvement": imp},
            "probabilities": {"log_loss_calibrated": 0.66 - imp, "ece_calibrated": 0.014},
            "pit": {"verdict": "flat: the distribution's shape is right"},
        }
    return {
        "run_id": "bt-demo",
        "holdout": "clean (demo)",
        "layer_a": {"season": "2025-26", "markets": markets, "targets": {"passes": True}},
    }


if __name__ == "__main__":
    print(seed())
