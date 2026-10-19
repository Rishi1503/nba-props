"""graded_props from the change-only log: a prop's close is its last row before tip.

Closing ticks now write only changed quotes (an unchanged quote is still on the board), so
the close can't require a row written on a closing tick. Main lines keep one close per
(book, player, market) whatever the line; withdrawn props are dropped.

Revision ID: 0010
Revises: 0009
"""

import importlib.util
from pathlib import Path

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
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
    WHERE o.player_id IS NOT NULL
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


def _view_from(prefix: str) -> str:
    path = next(Path(__file__).parent.glob(f"{prefix}_*.py"))
    spec = importlib.util.spec_from_file_location(f"m{prefix}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRADED_PROPS_VIEW


def upgrade() -> None:
    op.execute("DROP VIEW IF EXISTS graded_props")
    op.execute(GRADED_PROPS_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS graded_props")
    op.execute(_view_from("0005"))
