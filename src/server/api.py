"""FastAPI application for the CANNECT.AI CV-analytics edge module.

This is a **pull-based** API: the backend periodically calls these
endpoints to retrieve aggregated metrics.  The Mini PC never pushes
data on its own (except as an optional fallback).

Shared state (CounterStore, AdTracker, CameraManager) is injected once
at startup via :func:`setup_api` and then accessed through module-level
references.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, Query

from src.aggregation.models import (
    AttentionMetrics,
    CameraHealthInfo,
    CameraMetrics,
    CampaignEntry,
    CampaignMetricsResponse,
    CurrentAdInfo,
    CurrentAdPayload,
    DemographicMetrics,
    GlobalMetrics,
    HealthResponse,
    MetricsResponse,
    TrafficMetrics,
)
from src.config.settings import settings
from src.server.auth import verify_token

if TYPE_CHECKING:
    from src.aggregation.ad_tracker import AdTracker
    from src.aggregation.counters import CounterStore
    from src.capture.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared state -- populated by setup_api()
# ---------------------------------------------------------------------------

_counter_store: CounterStore | None = None
_ad_tracker: AdTracker | None = None
_camera_manager: CameraManager | None = None


def setup_api(
    counter_store: CounterStore,
    ad_tracker: AdTracker,
    camera_manager: CameraManager | None = None,
) -> None:
    """Inject shared state before the server starts accepting requests.

    Must be called from ``main.py`` (or equivalent) **before**
    :func:`create_app` or at least before the first request is served.
    """
    global _counter_store, _ad_tracker, _camera_manager  # noqa: PLW0603
    _counter_store = counter_store
    _ad_tracker = ad_tracker
    _camera_manager = camera_manager
    logger.info(
        "API state injected: counter_store=%s, ad_tracker=%s, camera_manager=%s",
        type(counter_store).__name__,
        type(ad_tracker).__name__,
        type(camera_manager).__name__ if camera_manager else "None",
    )


# ---------------------------------------------------------------------------
# psutil helpers (optional dependency)
# ---------------------------------------------------------------------------

def _get_cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0)
    except Exception:
        return 0.0


def _get_ram_usage_mb() -> int:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return int(mem.used / (1024 * 1024))
    except Exception:
        return 0


def _get_disk_free_mb() -> int:
    try:
        import psutil
        disk = psutil.disk_usage("/")
        return int(disk.free / (1024 * 1024))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and return the FastAPI application with all edge endpoints."""

    app = FastAPI(
        title="CANNECT CV-Analytics Edge API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # -------------------------------------------------------------------
    # GET / -- service info
    # -------------------------------------------------------------------

    @app.get("/", dependencies=[Depends(verify_token)])
    async def root() -> dict:
        return {
            "service": "cannect-cv-analytics",
            "stationId": settings.station_id,
            "version": "0.1.0",
        }

    # -------------------------------------------------------------------
    # GET /metrics -- global aggregated metrics
    # -------------------------------------------------------------------

    @app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(verify_token)])
    async def get_metrics(
        since: str | None = Query(None, description="ISO-8601 lower-bound timestamp"),
        reset: bool = Query(False, description="Reset counters after reading"),
    ) -> MetricsResponse:
        assert _counter_store is not None, "CounterStore not initialised"
        assert _ad_tracker is not None, "AdTracker not initialised"

        raw = _counter_store.get_global_metrics()

        traffic_data = raw.get("traffic", {})
        attention_data = raw.get("attention", {})
        demographics_data = raw.get("demographics", {})
        by_camera_raw = raw.get("by_camera", {})

        traffic = TrafficMetrics(**traffic_data) if traffic_data else TrafficMetrics()
        attention = AttentionMetrics(**attention_data) if attention_data else AttentionMetrics()
        demographics = DemographicMetrics(**demographics_data) if demographics_data else DemographicMetrics()

        by_camera: dict[str, CameraMetrics] = {}
        for cam_id, cam_data in by_camera_raw.items():
            by_camera[cam_id] = CameraMetrics(**cam_data) if cam_data else CameraMetrics()

        current_ad: CurrentAdInfo = _ad_tracker.get_current_ad()

        now = datetime.now(timezone.utc).isoformat()

        response = MetricsResponse(
            stationId=settings.station_id,
            timestamp=now,
            sinceTimestamp=since,
            currentAd=current_ad,
            **{"global": GlobalMetrics(
                traffic=traffic,
                attention=attention,
                demographics=demographics,
                by_camera=by_camera,
            )},
        )

        if reset:
            _counter_store.reset()
            logger.info("Counters reset after /metrics pull (reset=true)")

        return response

    # -------------------------------------------------------------------
    # GET /metrics/by-campaign -- per-campaign breakdown
    # -------------------------------------------------------------------

    @app.get(
        "/metrics/by-campaign",
        response_model=CampaignMetricsResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_campaign_metrics() -> CampaignMetricsResponse:
        assert _counter_store is not None, "CounterStore not initialised"

        raw_campaigns = _counter_store.get_campaign_metrics()
        campaigns = [CampaignEntry(**entry) for entry in raw_campaigns]

        return CampaignMetricsResponse(
            stationId=settings.station_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            campaigns=campaigns,
        )

    # -------------------------------------------------------------------
    # GET /health -- system and camera health
    # -------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, dependencies=[Depends(verify_token)])
    async def get_health() -> HealthResponse:
        cameras: dict[str, CameraHealthInfo] = {}

        if _camera_manager is not None:
            try:
                raw_health = _camera_manager.get_health()
                for cam_id, info in raw_health.items():
                    fps = info.get("fps", 0.0)
                    is_alive = info.get("is_alive", False)
                    last_frame_time = info.get("last_frame_time")

                    last_frame_at: str | None = None
                    if last_frame_time:
                        try:
                            last_frame_at = datetime.fromtimestamp(
                                last_frame_time, tz=timezone.utc
                            ).isoformat()
                        except (OSError, ValueError):
                            last_frame_at = None

                    cameras[cam_id] = CameraHealthInfo(
                        status="online" if is_alive else "offline",
                        fps=round(fps, 1),
                        last_frame_at=last_frame_at,
                    )
            except Exception:
                logger.exception("Failed to read camera health")

        return HealthResponse(
            stationId=settings.station_id,
            cpu_usage=_get_cpu_percent(),
            ram_usage=_get_ram_usage_mb(),
            disk_free_mb=_get_disk_free_mb(),
            cameras=cameras,
        )

    # -------------------------------------------------------------------
    # POST /current-ad -- playback events from local video player
    # -------------------------------------------------------------------

    @app.post("/current-ad", dependencies=[Depends(verify_token)])
    async def post_current_ad(payload: CurrentAdPayload) -> dict:
        assert _ad_tracker is not None, "AdTracker not initialised"

        _ad_tracker.on_playback_event(
            event=payload.event,
            campaign_id=payload.campaignId,
            video_id=payload.videoId,
            started_at=payload.startedAt,
            expected_duration=payload.expectedDuration,
        )

        logger.debug(
            "Received playback event: %s campaign=%s video=%s",
            payload.event,
            payload.campaignId,
            payload.videoId,
        )

        return {"ok": True}

    return app
