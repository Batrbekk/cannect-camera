"""Simple X-Station-Token authentication for the edge API.

The backend sends a shared secret in the ``X-Station-Token`` header on
every pull request.  If no token is configured (empty string), auth is
skipped -- this allows local development without secrets.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from src.config.settings import settings


def verify_token(x_station_token: str | None = Header(None)) -> None:
    """Verify the shared secret token from backend requests.

    Raises
    ------
    HTTPException (401)
        When a token is configured but the request header does not match.
    """
    if not settings.station_token:
        return  # no token configured = dev mode, skip auth
    if x_station_token != settings.station_token:
        raise HTTPException(status_code=401, detail="Invalid station token")
