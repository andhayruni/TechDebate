"""users, messages and notifications

Revision ID: 0001
Revises:
Create Date: 2026-11-11 18:12:58.700373
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("username", name=op.f("uq_users_username")),
    )
    op.create_table(
        "messages",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=True),
        sa.Column("parent_id", sa.String(), nullable=True),
        sa.Column("author_id", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["author_id"], ["users.id"], name=op.f("fk_messages_author_id_users")),
        sa.ForeignKeyConstraint(["parent_id"], ["messages.id"], name=op.f("fk_messages_parent_id_messages")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_messages_post_id_messages")),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_messages")),
        sa.UniqueConstraint("id", name=op.f("uq_messages_id")),
    )
    op.create_index(op.f("ix_messages_post_id"), "messages", ["post_id"])
    op.create_table(
        "notifications",
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("post_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], name=op.f("fk_notifications_message_id_messages")),
        sa.ForeignKeyConstraint(["post_id"], ["messages.id"], name=op.f("fk_notifications_post_id_messages")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_notifications_user_id_users")),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_notifications")),
        sa.UniqueConstraint("id", name=op.f("uq_notifications_id")),
    )
    op.create_index(op.f("ix_notifications_user_id_closed_at"), "notifications", ["user_id", "closed_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_notifications_user_id_closed_at"), table_name="notifications")
    op.drop_table("notifications")
    op.drop_index(op.f("ix_messages_post_id"), table_name="messages")
    op.drop_table("messages")
    op.drop_table("users")
