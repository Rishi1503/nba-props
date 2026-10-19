"""Daily pricing against Postgres (plan, Stage 7 `price_slate`).

Reads the active model run's projections and the odds board, prices every DraftKings prop
the model covers (two-way lines and milestones), sizes picks, builds parlays from them, and
writes picks and parlays. Runs at 10:00 AM, 3:00 PM and 6:00 PM New York and after late
news, so it is careful about repeats:

  - games that have tipped are never priced;
  - a game whose DraftKings odds haven't been polled within max_odds_age_min is skipped,
    and consensus books that stopped answering are dropped from the consensus;
  - picks already made today count against the caps, and their players get no new pick;
  - writes are idempotent: one pick per player per game per mode (first flag wins).

The daily cap covers singles and parlays together (plan: daily scaling).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from nbaprops.db.models import Parlay as ParlayRow
from nbaprops.db.models import ParlayLeg, Pick
from nbaprops.models.simulate import line_probs
from nbaprops.pricing import engine, market, parlays

MIN_STAKE = 0.50  # dollars: smaller stakes after scaling aren't worth placing


class NoBankroll(RuntimeError):
    pass


@dataclass
class SlateResult:
    game_date: date
    as_of: datetime
    run_id: str | None
    bankroll: float = 0.0
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    parlays: list = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    candidates: int = 0
    # props that cleared every flag rule except low confidence (minutes uncertainty, including
    # teammates of an unresolved Questionable player): the volume that rule costs
    blocked_low_confidence: int = 0
    # props blocked only by the Questionable-teammate rule (questionable_teammates)
    blocked_questionable: int = 0


# --- loaders -------------------------------------------------------------------------


def _q(session, sql: str, **params) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), session.connection(), params=params)


def slate_games(session, game_date: date, as_of: datetime) -> pd.DataFrame:
    """Games that day that haven't tipped yet."""
    g = _q(
        session, "SELECT game_id, game_date, tip_time FROM games WHERE game_date = :d", d=game_date
    )
    tip = pd.to_datetime(g["tip_time"], utc=True)
    return g[tip.isna() | (tip > pd.Timestamp(as_of))].reset_index(drop=True)


def active_run(session, game_ids: list[str]) -> str | None:
    """The champion if it has projections for these games, else the newest run that does."""
    if not game_ids:
        return None
    rows = _q(
        session,
        """
        SELECT r.run_id, r.status, r.created_ts FROM model_runs r
        WHERE EXISTS (SELECT 1 FROM projections p
                      WHERE p.run_id = r.run_id AND p.game_id = ANY(:g))
        ORDER BY (r.status = 'champion') DESC, r.created_ts DESC LIMIT 1""",
        g=game_ids,
    )
    return None if rows.empty else rows["run_id"].iloc[0]


def load_grids(session, run_id: str, game_ids: list[str], as_of: datetime) -> pd.DataFrame:
    """Latest projection per (game, player, market) made at or before `as_of`."""
    g = _q(
        session,
        """
        SELECT DISTINCT ON (game_id, player_id, market)
               game_id, player_id, market, p_over, low_confidence, scenarios, as_of_ts
        FROM projections WHERE run_id = :r AND game_id = ANY(:g) AND as_of_ts <= :t
        ORDER BY game_id, player_id, market, as_of_ts DESC""",
        r=run_id,
        g=game_ids,
        t=as_of,
    )
    g["player_id"] = g["player_id"].astype("int64")
    return g


def load_odds(session, game_ids: list[str], as_of: datetime) -> pd.DataFrame:
    o = _q(
        session,
        """SELECT * FROM odds_snapshots
                       WHERE game_id = ANY(:g) AND snapshot_ts <= :t AND player_id IS NOT NULL""",
        g=game_ids,
        t=as_of,
    )
    o["line"] = o["line"].astype(float)
    o["player_id"] = o["player_id"].astype("int64")
    return o


def last_polls(session, game_ids: list[str], as_of: datetime) -> pd.DataFrame:
    """Latest poll time per (game, book) at or before `as_of`."""
    p = _q(
        session,
        """SELECT game_id, snapshot_ts, books FROM odds_polls
                       WHERE game_id = ANY(:g) AND snapshot_ts <= :t""",
        g=game_ids,
        t=as_of,
    )
    if p.empty:
        return pd.DataFrame(columns=["game_id", "book", "last_poll"])
    p = p.explode("books").rename(columns={"books": "book"})
    return p.groupby(["game_id", "book"])["snapshot_ts"].max().rename("last_poll").reset_index()


def injury_changes(session, grids: pd.DataFrame, as_of: datetime) -> dict[str, str]:
    """{game_id: what changed} for games whose injury statuses changed after their oldest
    projection was made. Teammates' minutes and usage depend on who plays, so those games
    must be re-projected before they are priced (Stage 8 reruns them automatically)."""
    from nbaprops.features.build import latest_statuses

    if grids.empty:
        return {}
    ids = grids["game_id"].unique().tolist()
    inj = _q(
        session,
        """SELECT report_ts, game_id, team_id, player_id, status
                         FROM injury_snapshots WHERE game_id = ANY(:g)""",
        g=ids,
    )
    if inj.empty:
        return {}
    ids_ = [int(x) for x in inj["player_id"].dropna().unique()]
    names = dict(
        session.execute(
            text("SELECT player_id, full_name FROM players WHERE player_id = ANY(:p)"), {"p": ids_}
        ).all()
    )
    made = grids.groupby("game_id")["as_of_ts"].min()
    now = latest_statuses(inj, as_of)
    out = {}
    for gid, ts in made.items():
        before = latest_statuses(inj, pd.Timestamp(ts).to_pydatetime())
        a = dict(
            zip(
                before.loc[before["game_id"] == gid, "player_id"],
                before.loc[before["game_id"] == gid, "status"],
                strict=True,
            )
        )
        b = dict(
            zip(
                now.loc[now["game_id"] == gid, "player_id"],
                now.loc[now["game_id"] == gid, "status"],
                strict=True,
            )
        )
        diff = sorted(
            (int(p), a.get(p, "Available"), b.get(p, "Available"))
            for p in set(a) | set(b)
            if a.get(p) != b.get(p)
        )
        if diff:
            out[gid] = ", ".join(f"{names.get(p, p)} ({p}): {x} -> {y}" for p, x, y in diff)
    return out


def bankroll_on(session, day: date) -> float:
    r = _q(
        session,
        """SELECT bankroll FROM bankroll_ledger WHERE entry_date <= :d
                       ORDER BY entry_date DESC LIMIT 1""",
        d=day,
    )
    if r.empty:
        raise NoBankroll("No bankroll recorded: add a bankroll_ledger row first (plan, Stage 9).")
    return float(r["bankroll"].iloc[0])


def committed_today(session, game_date: date, mode: str) -> tuple[pd.DataFrame, float, int]:
    """(singles already made on this slate date, parlay stakes, satellites). Every game that
    day counts, including ones that have already tipped: a 6 PM run must still see the stakes
    on the 3 PM game, or the day's exposure could pass the daily cap."""
    singles = _q(
        session,
        """SELECT p.game_id, p.player_id, p.stake FROM picks p JOIN games g USING (game_id)
           WHERE g.game_date = :d AND p.mode = :m""",
        d=game_date,
        m=mode,
    )
    singles["stake"] = singles["stake"].astype(float)
    singles["game_date"] = game_date
    par = _q(
        session,
        """SELECT tier, stake FROM parlays WHERE slate_date = :d AND mode = :m""",
        d=game_date,
        m=mode,
    )
    return singles, float(par["stake"].astype(float).sum()), int((par["tier"] == "satellite").sum())


# --- pricing ---------------------------------------------------------------------------


def model_at_lines(grids: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    """Model over/under/push at every DraftKings two-way line, from the stored grids."""
    dk = board[(board["book"] == market.TARGET_BOOK) & (board["line_type"] != "milestone")]
    dk = dk.dropna(subset=["over_price", "under_price"])
    gl = grids.set_index(["game_id", "player_id", "market"])
    rows = []
    for q in dk.drop_duplicates(engine.KEY).itertuples(index=False):
        k = (q.game_id, int(q.player_id), q.market)
        if k not in gl.index:
            continue
        over, under, push = line_probs(gl.loc[k, "p_over"], q.line)
        rows.append(
            {
                "game_id": q.game_id,
                "player_id": int(q.player_id),
                "market": q.market,
                "line": q.line,
                "p_over": over,
                "p_under": under,
                "p_push": push,
                "low_confidence": bool(gl.loc[k, "low_confidence"]),
            }
        )
        sc = gl.loc[k, "scenarios"] if "scenarios" in gl.columns else None
        if isinstance(sc, dict):  # a key Questionable teammate: both outcomes at this line
            for tag in ("in", "out"):
                o, u, ps = line_probs(sc[tag], q.line)
                rows[-1].update({f"p_over_{tag}": o, f"p_under_{tag}": u, f"p_push_{tag}": ps})
    return pd.DataFrame(rows)


def fresh_board(
    odds: pd.DataFrame, polls: pd.DataFrame, as_of: datetime, max_age_min: int
) -> tuple[pd.DataFrame, list[str]]:
    """The board at `as_of`, minus books whose last poll for a game is older than the limit.
    Returns (board, games skipped because DraftKings itself is stale)."""
    b = market.board(odds, as_of)
    if b.empty:
        return b, []
    cutoff = pd.Timestamp(as_of) - timedelta(minutes=max_age_min)
    b = b.merge(polls, on=["game_id", "book"], how="left")
    fresh = pd.to_datetime(b["last_poll"], utc=True) >= cutoff
    stale_dk = sorted(set(b.loc[(b["book"] == market.TARGET_BOOK) & ~fresh, "game_id"]))
    return b[fresh].drop(columns="last_poll"), stale_dk


def price_slate(
    session,
    game_date: date,
    as_of: datetime,
    cfg,
    mode: str = "paper",
    boosts: list[tuple[float, float]] | None = None,
    write: bool = True,
) -> SlateResult:
    """Price one day's slate. `boosts`: (profit boost, max stake) pairs typed in from the
    DraftKings app. `cfg`: the validated strategy config."""
    rules = engine.Rules.from_config(cfg)
    res = SlateResult(game_date, as_of, None)
    games = slate_games(session, game_date, as_of)
    if games.empty:
        res.skipped.append("no games left to price today")
        return res
    ids = games["game_id"].tolist()
    res.run_id = active_run(session, ids)
    if res.run_id is None:
        res.skipped.append("no projections for today's games (run the slate projection first)")
        return res
    res.bankroll = bankroll_on(session, game_date)
    grids = apply_live_calibration(
        load_grids(session, res.run_id, ids, as_of), live_curves(session)
    )
    board, stale = fresh_board(
        load_odds(session, ids, as_of), last_polls(session, ids, as_of), as_of, cfg.max_odds_age_min
    )
    res.skipped += [f"{g}: DraftKings odds older than {cfg.max_odds_age_min} min" for g in stale]
    board = board[~board["game_id"].isin(stale)]
    changed = injury_changes(session, grids, as_of)
    res.skipped += [
        f"{g}: injury news after the projections ({m}); re-project first"
        for g, m in changed.items()
    ]
    board = board[~board["game_id"].isin(changed)]
    grids = grids[~grids["game_id"].isin(changed)]
    gd = games.set_index("game_id")["game_date"]
    grids = grids.assign(game_date=grids["game_id"].map(gd))
    model = model_at_lines(grids, board)
    if len(model):
        model = model.assign(game_date=model["game_id"].map(gd))
    cands = engine.build_candidates(model, grids, board, rules)
    off = disabled_markets(session)
    if off and len(cands):
        res.skipped.append(f"markets switched off: {', '.join(sorted(off))}")
        cands = cands[~cands["market"].isin(off)].reset_index(drop=True)
    if write and len(cands):
        keep = cands["flag"].astype(bool) | (
            cands["line_type"].isin(["main", "milestone"])
            & (cands["ev"] >= cfg.candidate_log_min_ev)
        )
        log_candidates(session, res, cands[keep], model)
    res.candidates = int(len(cands))
    if len(cands):
        thr = np.where(cands["has_consensus"].astype(bool), rules.ev_min, rules.ev_min_model_only)
        res.blocked_low_confidence = int(
            (
                (cands["ev"].to_numpy() >= thr)
                & cands["agree"].astype(bool).to_numpy()
                & cands["low_confidence"].fillna(False).astype(bool).to_numpy()
            ).sum()
        )
        res.blocked_questionable = int(
            (
                (cands["ev"].to_numpy() >= thr)
                & cands["agree"].astype(bool).to_numpy()
                & ~cands["low_confidence"].fillna(False).astype(bool).to_numpy()
                & ~engine.questionable_ok(cands, rules)
            ).sum()
        )
    if cands.empty:
        res.skipped.append("no DraftKings props matched the projections")
        return res
    singles_done, parlays_today, sats_today = committed_today(session, game_date, mode)
    picks = engine.select_and_size(cands, res.bankroll, rules, committed=singles_done, daily=False)
    # parlays from today's new flagged singles
    legs = parlays.legs_from_picks(picks) if len(picks) else []
    built = parlays.satellites(
        legs, rules, res.bankroll, max(0, rules.max_satellites_per_slate - sats_today)
    )
    used = {lg.pick_ref for p in built for lg in p.legs}
    # Boosted singles: never on a player picked in an earlier run or staked in this one; a
    # player with a tracked (zero-stake) pick in this run is eligible only on that same prop,
    # whose row then backs the boost (one pick per player per game).
    taken = {
        (g, int(p)) for g, p in zip(singles_done["game_id"], singles_done["player_id"], strict=True)
    }
    tracked = {}
    for i, r in enumerate(picks.itertuples(index=False)) if len(picks) else []:
        key = (r.game_id, int(r.player_id))
        if r.stake > 0:
            taken.add(key)
        else:
            tracked[key] = (i, (r.market, float(r.line), r.side))
    for boost in boosts or []:
        pct, max_stake, kind = (*boost, "parlay")[:3]
        if kind == "single":
            ok = [
                not ((g, int(p)) in tracked and (m, float(ln), sd) != tracked[(g, int(p))][1])
                for g, p, m, ln, sd in zip(
                    cands["game_id"],
                    cands["player_id"],
                    cands["market"],
                    cands["line"],
                    cands["side"],
                    strict=True,
                )
            ]
            pool = cands[ok].reset_index(drop=True)
            hit = parlays.boost_single(pool, pct, max_stake, rules, res.bankroll, taken)
            if hit is None:
                continue
            row, par = hit
            leg = par.legs[0]
            key = (leg.game_id, leg.player_id)
            if key in tracked:
                ref = tracked.pop(key)[0]  # the existing tracked row backs the boost
            else:  # a zero-stake pick row of its own tier
                extra = row.to_frame().T.assign(
                    stake=0.0, tracked_only=True, tier_override="boost_leg"
                )
                picks = pd.concat([picks, extra], ignore_index=True)
                ref = len(picks) - 1
            par.legs = (
                parlays.Leg(
                    ref,
                    leg.game_id,
                    leg.player_id,
                    leg.market,
                    leg.line,
                    leg.side,
                    leg.decimal,
                    leg.p_win,
                    leg.p_push,
                ),
            )
            taken.add(key)
            built.append(par)
            continue
        b = parlays.boost_parlay(legs, pct, max_stake, rules, res.bankroll, used)
        if b is not None:
            built.append(b)
            used |= {lg.pick_ref for lg in b.legs}
    # one daily cap across singles and parlays, counting what's already committed
    already = float(singles_done["stake"].sum()) + parlays_today
    new_total = float(picks["stake"].sum() if len(picks) else 0) + sum(p.stake for p in built)
    room = max(rules.max_daily_pct * res.bankroll - already, 0.0)
    scale = min(1.0, room / new_total) if new_total > 0 else 1.0
    if len(picks):
        picks["stake"] = (picks["stake"] * scale).round(2)
    for p in built:
        p.stake = round(p.stake * scale, 2)
    keep = (
        ((picks["stake"] >= MIN_STAKE) | picks["tracked_only"])
        if len(picks)
        else pd.Series(dtype=bool)
    )
    dropped = set(np.flatnonzero(~keep.to_numpy())) if len(picks) else set()
    built = [
        p for p in built if p.stake >= MIN_STAKE and not {lg.pick_ref for lg in p.legs} & dropped
    ]
    res.picks = picks[keep].reset_index(drop=True) if len(picks) else picks
    res.parlays = built
    if write:
        _write(session, res, picks, keep, mode)
    return res


# --- Stage 9: market controls, live calibration, the candidate log -----------------------


def disabled_markets(session) -> set[str]:
    return set(session.execute(text("SELECT market FROM market_controls WHERE disabled")).scalars())


def live_curves(session) -> dict:
    return {
        m: c for m, c in session.execute(text("SELECT market, curve FROM live_calibration")).all()
    }


def _recal(grid: dict, curve: dict) -> dict:
    keys = list(grid)
    v = np.interp(np.array([grid[k] for k in keys], float), curve["x"], curve["y"])
    return {k: round(float(p), 4) for k, p in zip(keys, v, strict=True)}


def apply_live_calibration(grids: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Apply the live (second-layer) calibration to each stored grid. For a Questionable
    teammate, each scenario is recalibrated and the blend recomputed from them, so it stays
    exactly p_play * in + (1 - p_play) * out."""
    if not curves or grids.empty:
        return grids
    out = grids.copy()
    for i, r in out.iterrows():
        curve = curves.get(r["market"])
        if curve is None:
            continue
        sc = r.get("scenarios")
        if isinstance(sc, dict):
            sc = {**sc, "in": _recal(sc["in"], curve), "out": _recal(sc["out"], curve)}
            p = sc["p_play"]
            out.at[i, "scenarios"] = sc
            out.at[i, "p_over"] = {
                k: round(p * sc["in"][k] + (1 - p) * sc["out"][k], 4) for k in sc["in"]
            }
        else:
            out.at[i, "p_over"] = _recal(r["p_over"], curve)
    return out


def log_candidates(session, res: "SlateResult", cands: pd.DataFrame, model: pd.DataFrame) -> None:
    """Write every priced prop (flagged or not) with the inputs `engine.decide` uses."""
    from nbaprops.db.models import CandidateLog

    probs = model.set_index(engine.KEY)[["p_over", "p_under"]] if len(model) else pd.DataFrame()
    rows = []
    for r in cands.to_dict("records"):
        milestone = r.get("line_type") == "milestone"
        if milestone:
            inputs = {
                k: r.get(k)
                for k in (
                    "p_model",
                    "p_fair",
                    "price",
                    "has_consensus",
                    "low_confidence",
                    "ev_if_plays",
                    "ev_if_sits",
                )
            }
        else:
            key = (r["game_id"], r["player_id"], r["market"], r["line"])
            p = probs.loc[key] if len(probs) and key in probs.index else {}
            inputs = {
                k: r.get(k) for k in engine.DECISION_COLUMNS + engine.SCENARIO_COLUMNS if k in r
            }
            inputs.update({"p_over": p.get("p_over"), "p_under": p.get("p_under")})
        inputs = {
            k: (
                None
                if v is None or (isinstance(v, float) and np.isnan(v))
                else (
                    bool(v)
                    if isinstance(v, (bool, np.bool_))
                    else float(v)
                    if isinstance(v, (int, float, np.number))
                    else v
                )
            )
            for k, v in inputs.items()
        }
        rows.append(
            CandidateLog(
                logged_ts=res.as_of,
                run_id=res.run_id,
                game_id=r["game_id"],
                player_id=int(r["player_id"]),
                market=r["market"],
                line=float(r["line"]),
                line_type=r.get("line_type", "main"),
                kind="milestone" if milestone else "two_way",
                inputs=inputs,
                side=r["side"],
                ev=float(r["ev"]),
                p_fair=float(r["p_fair"]),
                has_consensus=bool(r["has_consensus"]),
                flag=bool(r["flag"]),
            )
        )
    session.add_all(rows)
    session.flush()


def _dec(x, places=6):
    return (
        None
        if x is None or (isinstance(x, float) and np.isnan(x))
        else Decimal(str(round(float(x), places)))
    )


def _tier(r) -> str:
    if getattr(r, "tier_override", None) == "boost_leg":
        return "boost_leg"
    return "high_conf" if bool(r.high_confidence) else "standard"


def _unboosted_american(p) -> int:
    """DraftKings' own price: the legs' product (the boost is stored separately)."""
    from nbaprops.pricing.odds_math import to_american

    if p.same_game or not p.boost_pct:
        return p.american
    return int(round(to_american(float(np.prod([lg.decimal for lg in p.legs])))))


def _write(session, res: SlateResult, picks: pd.DataFrame, keep: pd.Series, mode: str) -> None:
    """Insert picks (first flag wins), then parlays whose legs are all recorded."""
    ids: dict[int, int] = {}
    for i, r in enumerate(picks.itertuples(index=False)):
        if not keep.iloc[i]:
            continue
        stmt = (
            insert(Pick)
            .values(
                run_id=res.run_id,
                game_id=r.game_id,
                player_id=int(r.player_id),
                market=r.market,
                line=_dec(r.line, 1),
                line_type=r.line_type,
                side=r.side,
                tier=_tier(r),
                mode=mode,
                flagged_ts=res.as_of,
                dk_price=int(r.price),
                fair_prob=_dec(r.p_fair),
                p_model=_dec(r.p_model),
                p_consensus=_dec(r.p_cons),
                kelly=_dec(r.kelly, 4),
                ev=_dec(r.ev, 4),
                stake=_dec(r.stake, 2),
            )
            .on_conflict_do_nothing(constraint="uq_picks_game_player_mode")
            .returning(Pick.pick_id)
        )
        new = session.execute(stmt).scalar()
        if new is not None:
            ids[i] = new
    for p in res.parlays:
        if not all(lg.pick_ref in ids for lg in p.legs):
            continue  # a leg lost the race to an earlier run's pick: skip the parlay
        row = ParlayRow(
            created_ts=res.as_of,
            slate_date=res.game_date,
            tier=p.tier,
            mode=mode,
            same_game=p.same_game,
            joint_prob=_dec(p.p_all_win),
            dk_price=_unboosted_american(p),
            boost_pct=_dec(p.boost_pct, 4),
            ev=_dec(p.ev, 4),
            stake=_dec(p.stake, 2),
        )
        session.add(row)
        session.flush()
        for lg in p.legs:
            session.add(ParlayLeg(parlay_id=row.parlay_id, pick_id=ids[lg.pick_ref]))
    session.flush()


def price_manual_milestone(
    session,
    game_id: str,
    player_id: int,
    stat: str,
    line: float,
    dk_price: int,
    as_of: datetime,
    cfg,
) -> dict:
    """EV and stake for a milestone typed in from the DraftKings app (missing from the feed):
    "20+ points at +150" is line 19.5, price 150. Same logic as feed milestones."""
    rules = engine.Rules.from_config(cfg)
    run_id = active_run(session, [game_id])
    if run_id is None:
        return {"error": "no projection for this game"}
    grids = load_grids(session, run_id, [game_id], as_of)
    board, _ = fresh_board(
        load_odds(session, [game_id], as_of),
        last_polls(session, [game_id], as_of),
        as_of,
        cfg.max_odds_age_min,
    )
    typed = pd.DataFrame(
        [
            {
                "id": -1,
                "snapshot_ts": as_of,
                "book": market.TARGET_BOOK,
                "game_id": game_id,
                "player_id": int(player_id),
                "market": stat,
                "line_type": "milestone",
                "line": float(line),
                "over_price": int(dk_price),
                "under_price": None,
                "is_withdrawn": False,
            }
        ]
    )
    board = pd.concat(
        [
            board[~((board["book"] == market.TARGET_BOOK) & (board["line_type"] == "milestone"))],
            typed,
        ],
        ignore_index=True,
    )
    c = engine.milestone_candidates(grids, board, rules)
    c = c[(c["player_id"] == int(player_id)) & (c["market"] == stat)] if len(c) else c
    if c.empty:
        return {"error": "no projection for this player and stat"}
    r = c.iloc[0]
    bankroll = bankroll_on(session, as_of.date())
    return {
        "p_fair": round(float(r["p_fair"]), 4),
        "ev": round(float(r["ev"]), 4),
        "flag": bool(r["flag"]),
        "has_consensus": bool(r["has_consensus"]),
        "stake": round(float(r["kelly"]) * bankroll, 2) if r["flag"] else 0.0,
        "note": "stake before today's game and daily caps",
    }
