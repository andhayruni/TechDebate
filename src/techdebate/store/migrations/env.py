"""Alembic's entry point. techdebate runs it on the store's own connection whenever it opens the database. The alembic
command (alembic.ini at the repository root) runs it on db_path from the settings, to write a new migration."""

from typing import Literal

from alembic import context
from alembic.autogenerate.api import AutogenContext
from sqlalchemy import Connection, create_engine

from techdebate.config import get_settings
from techdebate.store.tables import Base, UTCDateTime


def _render_item(type_: str, obj: object, autogen_context: AutogenContext) -> str | Literal[False]:
    # The column type only changes what Python reads and writes; the database sees a plain datetime.
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime()"
    return False


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        render_as_batch=True,  # SQLite changes most things by rebuilding the table
        render_item=_render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


_connection: Connection | None = context.config.attributes.get("connection")
if _connection is not None:
    _run(_connection)
else:
    with create_engine(f"sqlite:///{get_settings().db_path}").connect() as connection:
        _run(connection)
