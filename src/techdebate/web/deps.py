"""What pages get from the app: the session it opened when it started, and the settings it was built with."""

from fastapi import Request

from techdebate.config import Settings
from techdebate.session import Session


def get_session(request: Request) -> Session:
    """The session the app opened when it started."""
    session: Session = request.state.session
    return session


def get_app_settings(request: Request) -> Settings:
    """The settings the app was built with."""
    settings: Settings = request.app.state.settings
    return settings
