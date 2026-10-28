"""Database schema. Mirrors the Stage 0 table in the implementation plan.

Conventions:
- NBA IDs are used as-is: player_id and team_id are integers, game_id is the NBA's
  10-character string (e.g. "0022500001").
- Prices are stored as American odds (integers). Probabilities and money as Numeric.
- Closed sets (market, side, tier, mode, result, status) are VARCHAR with CHECK
  constraints, which are easier to extend in migrations than native Postgres enums.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from nbaprops.db.base import Base

MARKETS = ("pts", "reb", "ast", "fg3m", "pra")
SIDES = ("over", "under")
# boost_leg: a single that only exists to back a boosted bet (kept out of flag statistics)
PICK_TIERS = ("standard", "high_conf", "boost_leg")
PARLAY_TIERS = ("satellite", "boost", "alt_combo")
MODES = ("paper", "real")
RESULTS = ("win", "loss", "push", "void")
RUN_STATUSES = ("challenger", "champion", "retired")
LINE_TYPES = ("main", "alternate", "milestone")
GAME_MARKETS = ("h2h", "spreads", "totals")
SEASON_TYPES = ("preseason", "regular", "playoffs", "playin", "cup_final")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _in_or_null(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IS NULL OR {_in(col, values)}"


Money = Numeric(12, 2)
Prob = Numeric(7, 6)


# --- Reference data ----------------------------------------------------------


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    abbreviation: Mapped[str] = mapped_column(String(3), unique=True)
    name: Mapped[str] = mapped_column(Text)


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    full_name: Mapped[str] = mapped_column(Text)
    position: Mapped[str | None] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Current roster from the NBA's player index, refreshed daily. Live slates use it to place
    # a traded player on his new team before he has played for it. Never used for history:
    # it describes today's rosters, not past ones.
    current_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"))
    roster_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PlayerAlias(Base):
    """Maps sportsbook / injury-report spellings to NBA player IDs."""

    __tablename__ = "player_aliases"

    alias_normalized: Mapped[str] = mapped_column(Text, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    source: Mapped[str] = mapped_column(String(32))  # e.g. "propline", "injury_report", "manual"
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UnmatchedName(Base):
    """Names a source used that matched no player: the manual review queue (plan, Stage 2).
    Resolve with `python -m nbaprops.pipeline.run alias-add "Name" PLAYER_ID`."""

    __tablename__ = "unmatched_names"

    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    name_normalized: Mapped[str] = mapped_column(Text, primary_key=True)
    name_raw: Mapped[str] = mapped_column(Text)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    seen_count: Mapped[int] = mapped_column(Integer, default=1)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        CheckConstraint(_in("season_type", SEASON_TYPES)),
        Index("ix_games_date", "game_date"),
    )

    game_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    season: Mapped[str] = mapped_column(String(7))  # "2025-26"
    season_type: Mapped[str] = mapped_column(String(16))
    game_date: Mapped[date] = mapped_column(Date)
    tip_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    overtimes: Mapped[int] = mapped_column(Integer, default=0)
    # scheduled / final / postponed
    status: Mapped[str] = mapped_column(String(16), default="scheduled")
    # Set once the per-game box score (starters, DNP reasons) has been loaded.
    boxscore_loaded: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )


# --- Box scores --------------------------------------------------------------


class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"
    __table_args__ = (Index("ix_pgl_player", "player_id"),)

    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    # NULL until the box score is loaded: unknown is not the same as "came off the bench".
    started: Mapped[bool | None] = mapped_column(Boolean)
    dnp: Mapped[bool] = mapped_column(Boolean, default=False)
    dnp_reason: Mapped[str | None] = mapped_column(Text)
    minutes: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    pts: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    oreb: Mapped[int | None] = mapped_column(Integer)
    dreb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)
    fgm: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    fg3a: Mapped[int | None] = mapped_column(Integer)
    ftm: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    plus_minus: Mapped[int | None] = mapped_column(Integer)
    usage_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))


class TeamGameLog(Base):
    __tablename__ = "team_game_logs"

    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), primary_key=True)
    opp_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"))
    is_home: Mapped[bool | None] = mapped_column(Boolean)
    pts: Mapped[int | None] = mapped_column(Integer)
    minutes: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    fgm: Mapped[int | None] = mapped_column(Integer)
    fga: Mapped[int | None] = mapped_column(Integer)
    fg3m: Mapped[int | None] = mapped_column(Integer)
    fg3a: Mapped[int | None] = mapped_column(Integer)
    ftm: Mapped[int | None] = mapped_column(Integer)
    fta: Mapped[int | None] = mapped_column(Integer)
    oreb: Mapped[int | None] = mapped_column(Integer)
    dreb: Mapped[int | None] = mapped_column(Integer)
    reb: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    stl: Mapped[int | None] = mapped_column(Integer)
    blk: Mapped[int | None] = mapped_column(Integer)
    tov: Mapped[int | None] = mapped_column(Integer)
    pf: Mapped[int | None] = mapped_column(Integer)
    possessions: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    pace: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    off_rtg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    def_rtg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))


# --- Snapshots (append-only) -------------------------------------------------


class InjurySnapshot(Base):
    __tablename__ = "injury_snapshots"
    __table_args__ = (
        UniqueConstraint("report_ts", "player_name_raw", "team_id", name="uq_injury_report_row"),
        Index("ix_injury_player_ts", "player_id", "report_ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    report_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # time printed on report
    game_id: Mapped[str | None] = mapped_column(ForeignKey("games.game_id"))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"))
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.player_id"))
    player_name_raw: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))  # Out/Doubtful/Questionable/Probable/Available
    reason: Mapped[str | None] = mapped_column(Text)


class LineupSnapshot(Base):
    __tablename__ = "lineup_snapshots"
    __table_args__ = (Index("ix_lineup_game_ts", "game_id", "snapshot_ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"))
    starter: Mapped[bool] = mapped_column(Boolean)
    confirmed: Mapped[bool] = mapped_column(Boolean)


class ProviderEvent(Base):
    """Maps an odds provider's event id to an NBA game_id (matched once, then cached)."""

    __tablename__ = "provider_events"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    provider_event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), index=True)
    matched_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OddsSnapshot(Base):
    """Change-only player-prop log: a row is written when a line or price changes,
    plus a full set of closing rows (is_close) just before tip."""

    __tablename__ = "odds_snapshots"
    __table_args__ = (
        CheckConstraint(_in("market", MARKETS)),
        CheckConstraint(_in("line_type", LINE_TYPES)),
        CheckConstraint(
            "over_price IS NOT NULL OR under_price IS NOT NULL OR is_withdrawn",
            name="odds_snapshots_price_check",
        ),
        Index("ix_odds_prop_ts", "game_id", "player_id", "market", "book", "snapshot_ts"),
        Index("ix_odds_close", "game_id", postgresql_where="is_close"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str] = mapped_column(String(32))
    book: Mapped[str] = mapped_column(String(32))  # "draftkings", "fanduel", ...
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.player_id"))
    player_name_raw: Mapped[str] = mapped_column(Text)
    market: Mapped[str] = mapped_column(String(8))
    line_type: Mapped[str] = mapped_column(String(10), default="main")
    # Milestone "20+" is stored as line 19.5 so every row reads as over/under a half-point.
    line: Mapped[Decimal] = mapped_column(Numeric(6, 1))
    over_price: Mapped[int | None] = mapped_column(Integer)
    under_price: Mapped[int | None] = mapped_column(Integer)
    is_close: Mapped[bool] = mapped_column(Boolean, default=False)
    # The prop left the board (pulled, suspended, or an alternate line removed): prices are
    # NULL. Without these rows a replay would treat the last price as still available.
    is_withdrawn: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class OddsPoll(Base):
    """One row per game per snapshot pass, written even when no price changed. The odds log
    is change-only, so without this "unchanged" and "the collector stopped" look the same:
    freshness checks and pricing read the last poll per book from here."""

    __tablename__ = "odds_polls"
    __table_args__ = (Index("ix_odds_polls_game_ts", "game_id", "snapshot_ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str] = mapped_column(String(32))
    books: Mapped[list] = mapped_column(JSONB)  # books with at least one prop quote
    n_props: Mapped[int] = mapped_column(Integer)
    is_close: Mapped[bool] = mapped_column(Boolean, default=False)


class GameLineSnapshot(Base):
    """Change-only log of game lines (moneyline, spread, total) per book.
    Pinnacle game lines are the sharp input for simulated pace and margin."""

    __tablename__ = "game_line_snapshots"
    __table_args__ = (
        CheckConstraint(_in("market", GAME_MARKETS)),
        Index("ix_gameline_game_ts", "game_id", "book", "market", "snapshot_ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str] = mapped_column(String(32))
    book: Mapped[str] = mapped_column(String(32))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    market: Mapped[str] = mapped_column(String(8))
    # spreads: home handicap; totals: the total; h2h: null
    point: Mapped[Decimal | None] = mapped_column(Numeric(6, 1))
    # spreads/h2h: home and away prices; totals: over (home_price) and under (away_price)
    home_price: Mapped[int | None] = mapped_column(Integer)
    away_price: Mapped[int | None] = mapped_column(Integer)
    is_close: Mapped[bool] = mapped_column(Boolean, default=False)


# --- Model output ------------------------------------------------------------


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (CheckConstraint(_in("status", RUN_STATUSES)),)

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    git_sha: Mapped[str] = mapped_column(String(40))
    trained_through: Mapped[date] = mapped_column(Date)
    config: Mapped[dict] = mapped_column(JSONB)  # copy of strategy.yaml at run time
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="challenger")


class Projection(Base):
    __tablename__ = "projections"
    __table_args__ = (CheckConstraint(_in("market", MARKETS)),)

    run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id"), primary_key=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), primary_key=True)
    market: Mapped[str] = mapped_column(String(8), primary_key=True)
    as_of_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    mean: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    sd: Mapped[Decimal] = mapped_column(Numeric(6, 2))
    p_over: Mapped[dict] = mapped_column(JSONB)  # {"24.5": 0.5412, ...}
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    # Teammates of a key Questionable player: {"player_id", "p_play", "in": grid, "out": grid},
    # the grids if he plays / sits, at the same lines as p_over (their p_play blend). Pricing
    # uses them to require a bet to be +EV whichever way the injury news goes.
    scenarios: Mapped[dict | None] = mapped_column(JSONB)
    # who is missing from the player's team (features.context.role_context)
    context: Mapped[dict | None] = mapped_column(JSONB)


# --- Betting -----------------------------------------------------------------


class Pick(Base):
    __tablename__ = "picks"
    __table_args__ = (
        CheckConstraint(_in("market", MARKETS)),
        CheckConstraint(_in("side", SIDES)),
        CheckConstraint(_in("tier", PICK_TIERS)),
        CheckConstraint(_in("mode", MODES)),
        CheckConstraint(_in_or_null("result", RESULTS)),
        CheckConstraint(_in("line_type", LINE_TYPES)),
        Index("ix_picks_game", "game_id"),
        # One pick per player per game per mode: pricing reruns (10 AM, 3 PM, 6 PM, late
        # news) keep the first flag, which is when the bet would be placed.
        UniqueConstraint("game_id", "player_id", "mode", name="uq_picks_game_player_mode"),
    )

    pick_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id"))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"))
    market: Mapped[str] = mapped_column(String(8))
    line: Mapped[Decimal] = mapped_column(Numeric(6, 1))
    line_type: Mapped[str] = mapped_column(String(10), default="main", server_default="main")
    side: Mapped[str] = mapped_column(String(5))
    tier: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(8), default="paper")
    flagged_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dk_price: Mapped[int] = mapped_column(Integer)
    price_placed: Mapped[int | None] = mapped_column(Integer)
    fair_prob: Mapped[Decimal] = mapped_column(Prob)
    p_model: Mapped[Decimal | None] = mapped_column(Prob)  # recalibrated model probability
    p_consensus: Mapped[Decimal | None] = mapped_column(Prob)  # None = model-only pick
    # quarter-Kelly fraction after the per-bet cap, before game/day scaling
    kelly: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    ev: Mapped[Decimal] = mapped_column(Numeric(7, 4))
    stake: Mapped[Decimal] = mapped_column(Money)
    closing_price: Mapped[int | None] = mapped_column(Integer)
    clv: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
    result: Mapped[str | None] = mapped_column(String(8))
    pnl: Mapped[Decimal | None] = mapped_column(Money)
    context: Mapped[dict | None] = mapped_column(JSONB)  # the projection's, at flag time


class Parlay(Base):
    __tablename__ = "parlays"
    __table_args__ = (
        CheckConstraint(_in("tier", PARLAY_TIERS)),
        CheckConstraint(_in("mode", MODES)),
        CheckConstraint(_in_or_null("result", RESULTS)),
    )

    parlay_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    slate_date: Mapped[date | None] = mapped_column(Date)
    tier: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(8), default="paper")
    same_game: Mapped[bool] = mapped_column(Boolean, default=False)
    joint_prob: Mapped[Decimal] = mapped_column(Prob)
    dk_price: Mapped[int] = mapped_column(Integer)
    boost_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ev: Mapped[Decimal] = mapped_column(Numeric(7, 4))
    stake: Mapped[Decimal] = mapped_column(Money)
    price_placed: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[str | None] = mapped_column(String(8))
    pnl: Mapped[Decimal | None] = mapped_column(Money)


class ParlayLeg(Base):
    __tablename__ = "parlay_legs"

    parlay_id: Mapped[int] = mapped_column(ForeignKey("parlays.parlay_id"), primary_key=True)
    pick_id: Mapped[int] = mapped_column(ForeignKey("picks.pick_id"), primary_key=True)


class ManualPrice(Base):
    """Prices and boosts typed in from the DraftKings app (SGPs, missing milestones, boosts)."""

    __tablename__ = "manual_prices"
    __table_args__ = (CheckConstraint(_in("target_type", ("prop", "parlay", "boost"))),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    entered_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    target_type: Mapped[str] = mapped_column(String(8))
    # free-form key, e.g. "203999:pts:19.5:over"
    target_ref: Mapped[str | None] = mapped_column(Text)
    dk_price: Mapped[int | None] = mapped_column(Integer)
    boost_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    max_stake: Mapped[Decimal | None] = mapped_column(Money)


class ModelFile(Base):
    """A model bundle's files (gzip-compressed JSON), so scheduled jobs on fresh runners can
    load the active model, and every run_id stays reproducible (plan, Stage 11)."""

    __tablename__ = "model_files"

    run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.run_id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    content: Mapped[bytes] = mapped_column(LargeBinary)


class AlertLog(Base):
    """One row per alert key: repeats of the same problem (a stale board, a missing box score)
    alert once, not every 15 minutes."""

    __tablename__ = "alert_log"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    first_sent: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int] = mapped_column(Integer, default=1)


class OpsRun(Base):
    """One row per production cycle (and per manual fix you record): the go-live check's
    "14 straight days with no manual fixes" is read from here."""

    __tablename__ = "ops_runs"
    __table_args__ = (Index("ix_ops_runs_started", "started_ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # cycle | retrain | manual_fix
    started_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ok: Mapped[bool] = mapped_column(Boolean)
    actions: Mapped[list] = mapped_column(JSONB, default=list)
    errors: Mapped[list] = mapped_column(JSONB, default=list)
    note: Mapped[str | None] = mapped_column(Text)


class CandidateLog(Base):
    """Every DraftKings prop each pricing run priced, flagged or not, with the inputs the
    engine's decision uses. Paper-data tuning replays decisions from these (a threshold or
    weight change can only be judged on props the current settings rejected, too)."""

    __tablename__ = "candidate_log"
    __table_args__ = (
        Index("ix_candidate_log_game", "game_id"),
        Index("ix_candidate_log_ts", "logged_ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    logged_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    run_id: Mapped[str] = mapped_column(String(40))
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    player_id: Mapped[int] = mapped_column(Integer)
    market: Mapped[str] = mapped_column(String(8))
    line: Mapped[float] = mapped_column(Float)
    line_type: Mapped[str] = mapped_column(String(10))
    kind: Mapped[str] = mapped_column(String(10))  # two_way | milestone
    inputs: Mapped[dict] = mapped_column(JSONB)  # engine.DECISION_COLUMNS (+ scenarios)
    side: Mapped[str] = mapped_column(String(5))
    ev: Mapped[float] = mapped_column(Float)
    p_fair: Mapped[float] = mapped_column(Float)
    has_consensus: Mapped[bool] = mapped_column(Boolean)
    flag: Mapped[bool] = mapped_column(Boolean)


class MarketControl(Base):
    """Markets switched off (plan, Stage 9: negative CLV after 100+ picks). Pricing skips
    them until re-enabled."""

    __tablename__ = "market_controls"
    __table_args__ = (CheckConstraint(_in("market", MARKETS)),)

    market: Mapped[str] = mapped_column(String(8), primary_key=True)
    disabled: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(Text)
    changed_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LiveCalibration(Base):
    """A second calibration layer fit on live results (monthly), applied to the champion's
    probabilities at pricing time. Kept only when it beats no adjustment on later data."""

    __tablename__ = "live_calibration"
    __table_args__ = (CheckConstraint(_in("market", MARKETS)),)

    market: Mapped[str] = mapped_column(String(8), primary_key=True)
    curve: Mapped[dict] = mapped_column(JSONB)  # {"x": [...], "y": [...]}
    n: Mapped[int] = mapped_column(Integer)
    fitted_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BacktestReport(Base):
    """Stage 6 reports, stored for the website's Backtests page (files on a laptop aren't
    reachable from the hosted API)."""

    __tablename__ = "backtest_reports"

    run_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    created_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    season: Mapped[str] = mapped_column(String(7))
    holdout: Mapped[str] = mapped_column(Text)
    report: Mapped[dict] = mapped_column(JSONB)


class BankrollEntry(Base):
    __tablename__ = "bankroll_ledger"

    entry_date: Mapped[date] = mapped_column(Date, primary_key=True)
    deposits: Mapped[Decimal] = mapped_column(Money, default=0)
    withdrawals: Mapped[Decimal] = mapped_column(Money, default=0)
    settled_pnl: Mapped[Decimal] = mapped_column(Money, default=0)
    bankroll: Mapped[Decimal] = mapped_column(Money)
    note: Mapped[str | None] = mapped_column(Text)
