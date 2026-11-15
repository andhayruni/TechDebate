"""research

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-15 03:42:18.254706
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=False),
        sa.Column("results", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["users.id"], name=op.f("fk_research_agent_id_users")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_research_post_id_messages")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_research")),
        sa.UniqueConstraint("agent_id", "post_id", name=op.f("uq_research_agent_id_post_id")),
    )


def downgrade() -> None:
    op.drop_table("research")
