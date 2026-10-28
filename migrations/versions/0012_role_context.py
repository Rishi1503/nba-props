"""role-change context on projections and picks

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-27 20:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("projections", "picks"):
        op.add_column(table, sa.Column("context", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    for table in ("projections", "picks"):
        op.drop_column(table, "context")
