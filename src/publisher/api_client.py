"""Async HTTP client for publishing analytics payloads to the CANNECT.AI API.

Uses ``httpx`` with TLS 1.3, Bearer token authentication, and retry logic
with exponential backoff.
"""

from __future__ import annotations

import logging
import ssl

import httpx

from src.aggregation.models import AnalyticsPayload

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES: int = 3
_BACKOFF_FACTORS: tuple[float, ...] = (1.0, 2.0, 4.0)
_REQUEST_TIMEOUT_SEC: float = 10.0


def _build_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that enforces TLS 1.3 minimum."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_default_certs()
    return ctx


class ApiClient:
    """Async HTTP client for the CANNECT.AI analytics ingest endpoint.

    Parameters
    ----------
    api_url:
        Full URL of the analytics events endpoint
        (e.g. ``https://cannect.ai/api/analytics/events``).
    api_key:
        Bearer token used for authentication.
    """

    def __init__(self, api_url: str, api_key: str) -> None:
        self._api_url = api_url
        self._api_key = api_key

        ssl_ctx = _build_ssl_context()

        self._client = httpx.AsyncClient(
            verify=ssl_ctx,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_SEC),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, payload: AnalyticsPayload) -> bool:
        """POST an analytics payload to the API with retry.

        Makes up to :data:`_MAX_RETRIES` attempts with exponential backoff
        (1 s, 2 s, 4 s).

        Returns
        -------
        bool
            ``True`` if the API responded with a 2xx status code,
            ``False`` otherwise (all retries exhausted or non-retryable error).
        """
        body = payload.model_dump_json(by_alias=True)

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(self._api_url, content=body)

                if response.is_success:
                    logger.info(
                        "Payload sent successfully (%d) for station=%s camera=%s",
                        response.status_code,
                        payload.stationId,
                        payload.cameraId,
                    )
                    return True

                # Server error (5xx) — retryable
                if response.status_code >= 500:
                    logger.warning(
                        "Server error %d on attempt %d/%d — retrying",
                        response.status_code,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                else:
                    # Client error (4xx) — not retryable
                    logger.error(
                        "Client error %d sending payload — not retrying: %s",
                        response.status_code,
                        response.text[:500],
                    )
                    return False

            except httpx.TimeoutException:
                logger.warning(
                    "Timeout on attempt %d/%d sending payload",
                    attempt + 1,
                    _MAX_RETRIES,
                )
            except httpx.TransportError as exc:
                logger.warning(
                    "Transport error on attempt %d/%d: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Unexpected error on attempt %d/%d sending payload",
                    attempt + 1,
                    _MAX_RETRIES,
                )

            # Exponential backoff before next attempt
            if attempt < _MAX_RETRIES - 1:
                import asyncio

                backoff = _BACKOFF_FACTORS[attempt]
                logger.debug("Backing off %.1fs before retry", backoff)
                await asyncio.sleep(backoff)

        logger.error(
            "All %d attempts exhausted — payload for station=%s camera=%s dropped",
            _MAX_RETRIES,
            payload.stationId,
            payload.cameraId,
        )
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()
        logger.info("ApiClient closed")
