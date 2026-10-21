"""The dashboard API (plan, Stage 10). Run: `uvicorn nbaprops.api.main:app`.

Every route except /health and /login needs `Authorization: Bearer <token>` from /login.
CORS allows the origins in DASHBOARD_ORIGINS (comma-separated: the website's address).
/health is the post-deploy smoke test: the newest odds poll's age and the champion model.
"""

import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from nbaprops.api import auth, views

ET = ZoneInfo("America/New_York")


class Login(BaseModel):
    password: str


class Placed(BaseModel):
    price: int = Field(description="American odds you got, e.g. -110")
    stake: float = Field(gt=0)


class Boost(BaseModel):
    pct: float = Field(gt=0, le=2, description="profit boost, e.g. 0.30 for 30%")
    max_stake: float = Field(gt=0)
    kind: str = Field("parlay", pattern="^(parlay|single)$")


class ManualMilestone(BaseModel):
    game_id: str
    player_id: int
    market: str = Field(pattern="^(pts|reb|ast|fg3m|pra)$")
    line: float = Field(description='"20+" is 19.5')
    price: int


class ManualSGP(BaseModel):
    pick_ids: list[int] = Field(min_length=2)
    price: int = Field(description="DraftKings' same-game parlay price")
    boost_pct: float = 0.0


class MarketSwitch(BaseModel):
    disabled: bool
    reason: str | None = None


def create_app(
    session_factory=None,
    cfg_loader: Callable | None = None,
    game_draws: Callable | None = None,
    clock: Callable | None = None,
) -> FastAPI:
    """Injectable for tests: the session factory, config, the same-game simulation, and
    the clock."""
    from nbaprops.db.base import session_scope

    session_factory = session_factory or session_scope
    clock = clock or (lambda: datetime.now(UTC))

    def load_cfg():
        if cfg_loader:
            return cfg_loader()
        from nbaprops.config import load_strategy
        from nbaprops.settings import get_settings

        return load_strategy(get_settings().strategy_config)

    app = FastAPI(title="NBA props dashboard API", version="1.0")
    origins = [o.strip() for o in os.environ.get("DASHBOARD_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    throttle = auth.Throttle()

    def session():
        with session_factory() as s:
            yield s

    DB = Annotated[object, Depends(session)]

    def require(authorization: str = Header(default="")):
        _, secret = auth.settings_from_env()
        token = authorization.removeprefix("Bearer ").strip()
        if not secret or not token or not auth.check_token(token, secret):
            raise HTTPException(401, "log in again")

    def today() -> date:
        return clock().astimezone(ET).date()

    # --- open --------------------------------------------------------------------------

    @app.get("/health")
    def health(s: DB):
        from nbaprops.ops import bundles

        last = s.execute(text("SELECT max(snapshot_ts) FROM odds_polls")).scalar()
        age = None if last is None else int((clock() - last).total_seconds() // 60)
        return {"ok": True, "champion": bundles.champion(s), "odds_poll_age_min": age}

    @app.post("/login")
    def login(body: Login, request: Request):
        who = request.client.host if request.client else "?"
        if throttle.locked(who):
            raise HTTPException(429, "too many attempts: wait 15 minutes")
        stored, secret = auth.settings_from_env()
        if not stored or not secret:
            raise HTTPException(503, "login isn't configured (DASHBOARD_PASSWORD_HASH/SECRET)")
        if not auth.verify_password(body.password, stored):
            throttle.fail(who)
            raise HTTPException(401, "wrong password")
        throttle.clear(who)
        token, exp = auth.make_token(secret)
        return {"token": token, "expires": datetime.fromtimestamp(exp, UTC).isoformat()}

    # --- reads -----------------------------------------------------------------------

    guard = [Depends(require)]

    @app.get("/slate/{day}", dependencies=guard)
    def get_slate(s: DB, day: date):
        return views.slate(s, day, clock())

    @app.get("/picks/{day}", dependencies=guard)
    def get_picks(s: DB, day: date, mode: str = "paper"):
        return views.picks(s, day, clock(), mode)

    @app.get("/parlays/{day}", dependencies=guard)
    def get_parlays(s: DB, day: date, mode: str = "paper"):
        return views.parlays(s, day, mode)

    @app.get("/players/{player_id}/projection", dependencies=guard)
    def get_projection(s: DB, player_id: int, day: date | None = None):
        return views.jsonable(views.player_projection(s, player_id, day or today()))

    @app.get("/performance", dependencies=guard)
    def get_performance(
        s: DB,
        start: date | None = None,
        end: date | None = None,
        mode: str = "paper",
        market: str | None = None,
        tier: str | None = None,
    ):
        end = end or today()
        start = start or views.rv.season_start(end)
        return views.jsonable(views.performance(s, start, end, mode, market, tier))

    @app.get("/review", dependencies=guard)
    def get_review(s: DB, mode: str = "paper"):
        from nbaprops.ops import golive

        return views.jsonable(
            {
                "review": views.rv.weekly_review(s, today(), mode),
                "golive": golive.checklist(s, today()),
            }
        )

    @app.get("/backtests", dependencies=guard)
    def list_backtests(s: DB):
        return views.backtests(s)

    @app.get("/backtests/{run_id}", dependencies=guard)
    def get_backtest(s: DB, run_id: str):
        rep = views.backtest(s, run_id)
        if rep is None:
            raise HTTPException(404, "no such backtest")
        return rep

    @app.get("/export/{year}", dependencies=guard, response_class=PlainTextResponse)
    def export(s: DB, year: int, mode: str = "real"):
        return PlainTextResponse(
            views.export_csv(s, year, mode),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="bets-{year}-{mode}.csv"'},
        )

    @app.get("/bets", dependencies=guard)
    def get_bets(s: DB, start: date | None = None, end: date | None = None, mode: str = "real"):
        end = end or today()
        return views.bets(s, start or views.rv.season_start(end), end, mode)

    @app.get("/record", dependencies=guard)
    def get_record(s: DB, start: date | None = None, end: date | None = None, mode: str = "paper"):
        end = end or today()
        return views.record(s, start or end - timedelta(days=60), end, mode)

    @app.get("/success", dependencies=guard)
    def get_success(
        s: DB, mode: str = "paper", period: str = "week", include_tracked: bool = False
    ):
        if period not in ("week", "month", "year"):
            raise HTTPException(422, "period must be week, month or year")
        return views.success(s, mode, period, include_tracked)

    @app.get("/tail", dependencies=guard)
    def get_tail(
        s: DB,
        stake: float = 10.0,
        mode: str = "paper",
        start: date | None = None,
        include_tracked: bool = False,
        include_parlays: bool = True,
    ):
        if not 0 < stake <= 100000:
            raise HTTPException(422, "stake must be between 0 and 100,000")
        return views.tail(s, stake, mode, start, include_tracked, include_parlays)

    @app.get("/bankroll", dependencies=guard)
    def get_bankroll(s: DB):
        return views.bankroll(s)

    @app.get("/settings", dependencies=guard)
    def get_settings_(s: DB):
        return views.settings(s, load_cfg())

    # --- writes ------------------------------------------------------------------------

    @app.post("/picks/{pick_id}/placed", dependencies=guard)
    def place_pick(s: DB, pick_id: int, body: Placed):
        from nbaprops.ops.place import AlreadyPlaced, place

        try:
            real = place(s, pick_id, body.price, body.stake, clock())
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except AlreadyPlaced as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"real_pick_id": real.pick_id}

    @app.post("/parlays/{parlay_id}/placed", dependencies=guard)
    def place_parlay(s: DB, parlay_id: int, body: Placed):
        from nbaprops.db.models import Parlay, ParlayLeg

        par = s.get(Parlay, parlay_id)
        if par is None:
            raise HTTPException(404, "no such parlay")
        real = Parlay(
            **{
                c: getattr(par, c)
                for c in (
                    "slate_date",
                    "tier",
                    "same_game",
                    "joint_prob",
                    "dk_price",
                    "boost_pct",
                    "ev",
                )
            },
            created_ts=clock(),
            mode="real",
            price_placed=body.price,
            stake=body.stake,
        )
        s.add(real)
        s.flush()
        for leg in s.query(ParlayLeg).filter(ParlayLeg.parlay_id == parlay_id).all():
            s.add(ParlayLeg(parlay_id=real.parlay_id, pick_id=leg.pick_id))
        return {"real_parlay_id": real.parlay_id}

    @app.post("/boosts", dependencies=guard)
    def boost(s: DB, body: Boost):
        from nbaprops.pricing import slate

        res = slate.price_slate(
            s, today(), clock(), load_cfg(), boosts=[(body.pct, body.max_stake, body.kind)]
        )
        boosted = [p for p in res.parlays if p.tier == "boost"]
        return {
            "skipped": res.skipped,
            "boosted": [
                {
                    "legs": [
                        {
                            "player_id": lg.player_id,
                            "market": lg.market,
                            "line": lg.line,
                            "side": lg.side,
                        }
                        for lg in p.legs
                    ],
                    "price": p.american,
                    "ev": round(p.ev, 4),
                    "stake": p.stake,
                    "p_all_win": round(p.p_all_win, 4),
                }
                for p in boosted
            ],
        }

    @app.post("/prices/manual/milestone", dependencies=guard)
    def manual_milestone(s: DB, body: ManualMilestone):
        from nbaprops.pricing.slate import price_manual_milestone

        return price_manual_milestone(
            s, body.game_id, body.player_id, body.market, body.line, body.price, clock(), load_cfg()
        )

    @app.post("/prices/manual/sgp", dependencies=guard)
    def manual_sgp(s: DB, body: ManualSGP):
        return sgp_price(s, body, clock(), load_cfg(), game_draws or simulate_game_draws)

    @app.post("/markets/{market}", dependencies=guard)
    def switch_market(s: DB, market: str, body: MarketSwitch):
        from nbaprops.db.models import MarketControl

        if market not in ("pts", "reb", "ast", "fg3m", "pra"):
            raise HTTPException(404, "no such market")
        s.merge(
            MarketControl(
                market=market, disabled=body.disabled, reason=body.reason, changed_ts=clock()
            )
        )
        return {"market": market, "disabled": body.disabled}

    return app


def sgp_price(session, body: ManualSGP, now: datetime, cfg, game_draws: Callable) -> dict:
    """A same-game parlay of flagged picks at DraftKings' quoted price: the joint hit rate
    comes from simulating that game (dependence preserved), with the 3-point haircut."""
    from nbaprops.db.models import Pick
    from nbaprops.pricing import engine, parlays
    from nbaprops.pricing.odds_math import to_decimal
    from nbaprops.pricing.slate import bankroll_on

    picks = [session.get(Pick, i) for i in body.pick_ids]
    if any(p is None for p in picks):
        raise HTTPException(404, "unknown pick")
    if len({p.game_id for p in picks}) != 1:
        raise HTTPException(422, "a same-game parlay's legs must share one game")
    draws = game_draws(session, picks[0].game_id, now, cfg)
    legs, raw = [], {}
    for i, p in enumerate(picks):
        line = float(p.line)
        d = draws.get((p.game_id, p.player_id), {}).get(p.market)
        if d is None:
            raise HTTPException(422, f"no simulation for pick {p.pick_id}")
        hit = (d > line) if p.side == "over" else (d < line)
        raw[(p.game_id, p.player_id, p.market, line, p.side)] = float(hit.mean())
        legs.append(
            parlays.Leg(
                i,
                p.game_id,
                p.player_id,
                p.market,
                line,
                p.side,
                to_decimal(p.dk_price),
                float(p.fair_prob),
            )
        )
    try:
        par = parlays.same_game_parlay(
            legs,
            body.price,
            draws,
            raw,
            engine.Rules.from_config(cfg),
            bankroll_on(session, now.astimezone(ET).date()),
            body.boost_pct,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "joint_prob": round(par.p_all_win, 4),
        "ev": round(par.ev, 4),
        "stake": round(par.stake, 2),
        "flag": par.ev > 0,
        "notes": par.notes,
    }


def simulate_game_draws(session, game_id: str, now: datetime, cfg) -> dict:
    """Simulate one game with the champion model: {(game_id, player_id): {market: draws}}."""
    from nbaprops.features.build import build_feature_scenarios
    from nbaprops.models.project import _draws_by_player
    from nbaprops.ops import bundles
    from nbaprops.ops.cycle import dk_players

    run_id = bundles.champion(session)
    if run_id is None:
        raise HTTPException(503, "no champion model")
    day = session.execute(
        text("SELECT game_date FROM games WHERE game_id = :g"), {"g": game_id}
    ).scalar()
    frame, _, _ = build_feature_scenarios(session, day, now, dk_players(session, day, now))
    frame = frame[frame["game_id"] == game_id]
    draws, _ = _draws_by_player(frame, bundles.load(session, run_id), cfg.n_sims, None)
    return draws


app = create_app()
