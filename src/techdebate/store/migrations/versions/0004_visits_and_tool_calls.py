"""visits and tool calls

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-21 20:10:05.537432
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "visits",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("at_research", sa.Boolean(), nullable=False),
        sa.Column("comment_ids", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["users.id"], name=op.f("fk_visits_agent_id_users")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_visits_post_id_messages")),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_visits")),
        sa.UniqueConstraint("id", name=op.f("uq_visits_id")),
    )
    op.create_index(op.f("ix_visits_post_id"), "visits", ["post_id"])

    op.create_table(
        "tool_calls",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("visit_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=False),
        sa.Column("tool", sa.String(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("evidence", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_s", sa.Double[float](), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["users.id"], name=op.f("fk_tool_calls_agent_id_users")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_tool_calls_post_id_messages")),
        sa.ForeignKeyConstraint(["visit_id"], ["visits.id"], name=op.f("fk_tool_calls_visit_id_visits")),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_tool_calls")),
        sa.UniqueConstraint("id", name=op.f("uq_tool_calls_id")),
    )
    op.create_index(op.f("ix_tool_calls_post_id"), "tool_calls", ["post_id"])

    # SQLite cannot add a foreign key to a table, so batch mode rebuilds these two with their rows. The rows stored
    # before this migration belong to no visit, and have no output or error.
    with op.batch_alter_table("llm_calls") as batch_op:
        batch_op.add_column(sa.Column("visit_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("output", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("error", sa.Text(), nullable=True))
        batch_op.create_foreign_key(batch_op.f("fk_llm_calls_visit_id_visits"), "visits", ["visit_id"], ["id"])

    with op.batch_alter_table("notifications") as batch_op:
        batch_op.add_column(sa.Column("visit_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(batch_op.f("fk_notifications_visit_id_visits"), "visits", ["visit_id"], ["id"])


def downgrade() -> None:
    with op.batch_alter_table("notifications") as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_notifications_visit_id_visits"), type_="foreignkey")
        batch_op.drop_column("visit_id")

    with op.batch_alter_table("llm_calls") as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_llm_calls_visit_id_visits"), type_="foreignkey")
        batch_op.drop_column("error")
        batch_op.drop_column("output")
        batch_op.drop_column("visit_id")

    op.drop_index(op.f("ix_tool_calls_post_id"), table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index(op.f("ix_visits_post_id"), table_name="visits")
    op.drop_table("visits")
