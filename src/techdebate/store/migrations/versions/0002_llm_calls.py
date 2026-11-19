"""llm calls

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-15 03:42:18.254706
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=False),
        sa.Column("prompt_id", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Double[float](), nullable=False),
        sa.Column("duration_s", sa.Double[float](), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["users.id"], name=op.f("fk_llm_calls_agent_id_users")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_llm_calls_post_id_messages")),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_llm_calls")),
        sa.UniqueConstraint("id", name=op.f("uq_llm_calls_id")),
    )
    op.create_index(op.f("ix_llm_calls_post_id"), "llm_calls", ["post_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_llm_calls_post_id"), table_name="llm_calls")
    op.drop_table("llm_calls")
