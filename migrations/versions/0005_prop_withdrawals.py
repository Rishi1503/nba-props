"""Prop withdrawals and a corrected grading view.

odds_snapshots gains is_withdrawn: a row with NULL prices written when a book stops quoting a
prop it quoted before (pulled, suspended, or an alternate line removed). Without it, a replay
of the change-only log treats the last price as still available.

graded_props now takes each prop's close as its last closing-or-withdrawn row before tip,
keyed WITHOUT the line for main lines (so a main line that moves inside the closing window
has one close, not one per line it passed through), and drops props withdrawn at the close.

Revision ID: 0005
Revises: 0004
"""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels = None
depends_on = None

GRADED_PROPS_VIEW = """
CREATE VIEW graded_props AS
WITH last_rows AS (
    SELECT DISTINCT ON (o.game_id, o.book, o.player_id, o.market, o.line_type,
                        CASE WHEN o.line_type = 'main' THEN NULL ELSE o.line END)
           o.id, o.game_id, o.book, o.player_id, o.market, o.line_type, o.line,
           o.over_price, o.under_price, o.snapshot_ts, o.is_withdrawn
    FROM odds_snapshots o
    JOIN games g ON g.game_id = o.game_id
    WHERE (o.is_close OR o.is_withdrawn) AND o.player_id IS NOT NULL
      AND (g.tip_time IS NULL OR o.snapshot_ts < g.tip_time)
    ORDER BY o.game_id, o.book, o.player_id, o.market, o.line_type,
             CASE WHEN o.line_type = 'main' THEN NULL ELSE o.line END,
             o.snapshot_ts DESC, o.id DESC
),
closes AS (
    SELECT * FROM last_rows WHERE NOT is_withdrawn
),
actuals AS (
    SELECT c.*,
           CASE c.market
               WHEN 'pts' THEN l.pts WHEN 'reb' THEN l.reb WHEN 'ast' THEN l.ast
               WHEN 'fg3m' THEN l.fg3m WHEN 'pra' THEN l.pts + l.reb + l.ast
           END AS actual,
           (l.player_id IS NULL OR l.dnp OR COALESCE(l.minutes, 0) = 0) AS did_not_play
    FROM closes c
    JOIN games g ON g.game_id = c.game_id AND g.status = 'final'
    LEFT JOIN player_game_logs l ON l.game_id = c.game_id AND l.player_id = c.player_id
)
SELECT id AS odds_snapshot_id, game_id, book, player_id, market, line_type, line,
       over_price, under_price, snapshot_ts AS close_ts, actual,
       CASE WHEN did_not_play THEN 'void'
            WHEN actual > line THEN 'over'
            WHEN actual < line THEN 'under'
            ELSE 'push' END AS result
FROM actuals
"""


def _old_view() -> str:
    path = next(Path(__file__).parent.glob("0003_*.py"))
    spec = importlib.util.spec_from_file_location("m0003", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRADED_PROPS_VIEW


def upgrade() -> None:
    op.add_column(
        "odds_snapshots",
        sa.Column("is_withdrawn", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.drop_constraint("odds_snapshots_check", "odds_snapshots", type_="check")
    op.create_check_constraint(
        "odds_snapshots_price_check",
        "odds_snapshots",
        "over_price IS NOT NULL OR under_price IS NOT NULL OR is_withdrawn",
    )
    op.execute("DROP VIEW IF EXISTS graded_props")
    op.execute(GRADED_PROPS_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS graded_props")
    op.execute("DELETE FROM odds_snapshots WHERE is_withdrawn")  # NULL prices: old rule forbids
    op.drop_constraint("odds_snapshots_price_check", "odds_snapshots", type_="check")
    op.create_check_constraint(
        "odds_snapshots_check",
        "odds_snapshots",
        "over_price IS NOT NULL OR under_price IS NOT NULL",
    )
    op.drop_column("odds_snapshots", "is_withdrawn")
    op.execute(_old_view())
