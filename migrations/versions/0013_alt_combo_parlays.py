"""alt_combo parlay tier

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-27 21:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("parlays_tier_check", "parlays", type_="check")
    op.create_check_constraint(
        "parlays_tier_check", "parlays", "tier IN ('satellite', 'boost', 'alt_combo')"
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM parlay_legs WHERE parlay_id IN "
        "(SELECT parlay_id FROM parlays WHERE tier = 'alt_combo')"
    )
    op.execute("DELETE FROM parlays WHERE tier = 'alt_combo'")
    op.drop_constraint("parlays_tier_check", "parlays", type_="check")
    op.create_check_constraint("parlays_tier_check", "parlays", "tier IN ('satellite', 'boost')")
