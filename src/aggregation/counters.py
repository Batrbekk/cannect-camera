"""Per-campaign in-memory counters for the CANNECT.AI CV analytics pipeline.

The CV pipeline feeds detection/attention/demographic data into CounterStore,
which segregates all metrics by the currently-playing campaignId.  The backend
pulls aggregated snapshots via the edge HTTP API.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

from src.aggregation.models import (
    AgeGroup,
    AttentionMetrics,
    CampaignEntry,
    CampaignMetrics,
    CameraMetrics,
    CurrentAdInfo,
    DemographicMetrics,
    Direction,
    Gender,
    GlobalMetrics,
    TrafficMetrics,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Sentinel key used when no ad campaign is active.
_IDLE_KEY = "__idle__"


class CounterStore:
    """Thread-safe, per-campaign in-memory metric counters.

    The CV pipeline calls ``add_*`` methods as detections happen.  The edge
    HTTP handler calls ``get_*`` / ``to_dict`` to serve pull requests from the
    backend, and ``reset`` when the backend confirms receipt.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Current ad context (set by AdTracker)
        self._current_campaign_id: str | None = None
        self._current_video_id: str | None = None

        # Global aggregated metrics
        self._global_traffic = TrafficMetrics()
        self._global_attention = AttentionMetrics()
        self._global_demographics = DemographicMetrics()

        # Per-campaign breakdown: campaignId -> CampaignEntry
        self._by_campaign: dict[str, CampaignEntry] = {}

        # Per-camera breakdown: camera_id -> CameraMetrics
        self._by_camera: dict[str, CameraMetrics] = {}

        # Dwell-time samples for average calculation
        self._dwell_times: list[float] = []

        # Playback tracking: campaignId -> {count, total_seconds}
        self._playbacks: dict[str, dict[str, int | float]] = {}

        logger.debug("CounterStore initialised")

    # ------------------------------------------------------------------
    # Campaign context
    # ------------------------------------------------------------------

    def set_current_campaign(self, campaign_id: str | None, video_id: str | None) -> None:
        """Set the currently-playing ad campaign (called by AdTracker)."""
        with self._lock:
            self._current_campaign_id = campaign_id
            self._current_video_id = video_id
            logger.debug(
                "Current campaign set: campaign=%s video=%s",
                campaign_id,
                video_id,
            )

    def get_current_campaign(self) -> tuple[str | None, str | None]:
        """Return ``(campaignId, videoId)`` for the currently-playing ad."""
        with self._lock:
            return self._current_campaign_id, self._current_video_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _campaign_key(self) -> str:
        """Return the dict key for the active campaign (``_IDLE_KEY`` if none)."""
        return self._current_campaign_id or _IDLE_KEY

    def _ensure_campaign_entry(self, key: str) -> CampaignEntry:
        """Return or create the CampaignEntry for *key*."""
        if key not in self._by_campaign:
            self._by_campaign[key] = CampaignEntry(
                campaignId=None if key == _IDLE_KEY else key,
                videoId=self._current_video_id,
            )
        return self._by_campaign[key]

    def _ensure_camera_entry(self, camera_id: str) -> CameraMetrics:
        """Return or create the CameraMetrics for *camera_id*."""
        if camera_id not in self._by_camera:
            self._by_camera[camera_id] = CameraMetrics()
        return self._by_camera[camera_id]

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_person(self, camera_id: str, direction: Direction | None) -> None:
        """Record a person detection with optional direction for the current campaign."""
        with self._lock:
            # Global traffic
            self._global_traffic.people_total += 1
            if direction is not None:
                if direction == Direction.TO_SCREEN:
                    self._global_traffic.toScreen += 1
                elif direction == Direction.FROM_SCREEN:
                    self._global_traffic.fromScreen += 1
                elif direction == Direction.LEFT:
                    self._global_traffic.left += 1
                elif direction == Direction.RIGHT:
                    self._global_traffic.right += 1

            # Per-campaign
            key = self._campaign_key()
            entry = self._ensure_campaign_entry(key)
            entry.metrics.impressions += 1

            # Per-camera
            cam = self._ensure_camera_entry(camera_id)
            cam.people += 1

    def add_attention(
        self,
        camera_id: str,
        track_id: int,
        dwell_time: float,
        is_over_5s: bool,
        is_looking: bool,
    ) -> None:
        """Record an attention observation for a tracked person."""
        with self._lock:
            if is_looking:
                # Global attention
                self._global_attention.total_looking += 1

                # Per-camera
                cam = self._ensure_camera_entry(camera_id)
                cam.looking += 1

                # Per-campaign
                key = self._campaign_key()
                entry = self._ensure_campaign_entry(key)
                entry.metrics.total_looking += 1

            if is_over_5s:
                self._global_attention.attention_over_5s += 1

                cam = self._ensure_camera_entry(camera_id)
                cam.attention_over_5s += 1

                key = self._campaign_key()
                entry = self._ensure_campaign_entry(key)
                entry.metrics.deep_views_over_5s += 1

            # Record dwell time for averages
            if dwell_time > 0:
                self._dwell_times.append(dwell_time)
                self._global_attention.avg_dwell_time = (
                    sum(self._dwell_times) / len(self._dwell_times)
                )

                key = self._campaign_key()
                entry = self._ensure_campaign_entry(key)
                # Recompute campaign avg_dwell_time is costly with list per campaign,
                # so we use a running average stored in the campaign metrics.
                n = entry.metrics.total_looking or 1
                prev_avg = entry.metrics.avg_dwell_time
                entry.metrics.avg_dwell_time = round(
                    prev_avg + (dwell_time - prev_avg) / n, 2
                )

    def add_demographic(self, gender: Gender, age_group: AgeGroup) -> None:
        """Record a demographic classification."""
        with self._lock:
            self._global_demographics.analyzed_count += 1
            self._global_demographics.gender[gender.value] = (
                self._global_demographics.gender.get(gender.value, 0) + 1
            )
            self._global_demographics.age_groups[age_group.value] = (
                self._global_demographics.age_groups.get(age_group.value, 0) + 1
            )

            # Per-campaign
            key = self._campaign_key()
            entry = self._ensure_campaign_entry(key)
            entry.metrics.gender[gender.value] = (
                entry.metrics.gender.get(gender.value, 0) + 1
            )
            entry.metrics.age_groups[age_group.value] = (
                entry.metrics.age_groups.get(age_group.value, 0) + 1
            )

    def add_unique_viewer(self, count: int) -> None:
        """Set the current unique viewer count."""
        with self._lock:
            self._global_attention.unique_viewers = count

            # Distribute to current campaign
            key = self._campaign_key()
            entry = self._ensure_campaign_entry(key)
            entry.metrics.unique_viewers = count

    def add_playback(
        self, campaign_id: str, video_id: str, duration_seconds: float
    ) -> None:
        """Record a completed playback event for a campaign."""
        with self._lock:
            # Track in playback dict
            if campaign_id not in self._playbacks:
                self._playbacks[campaign_id] = {"count": 0, "total_seconds": 0.0}
            self._playbacks[campaign_id]["count"] += 1  # type: ignore[operator]
            self._playbacks[campaign_id]["total_seconds"] += duration_seconds  # type: ignore[operator]

            # Update campaign entry
            entry = self._ensure_campaign_entry(campaign_id)
            entry.campaignId = campaign_id
            entry.videoId = video_id
            entry.totalPlaybacks += 1
            entry.totalPlaybackSeconds += duration_seconds

            logger.debug(
                "Playback recorded: campaign=%s video=%s duration=%.1fs",
                campaign_id,
                video_id,
                duration_seconds,
            )

    # ------------------------------------------------------------------
    # Read accessors (called by edge HTTP handlers)
    # ------------------------------------------------------------------

    def get_global_metrics(self) -> dict:
        """Return global aggregated metrics across all campaigns."""
        with self._lock:
            return GlobalMetrics(
                traffic=self._global_traffic.model_copy(),
                attention=self._global_attention.model_copy(),
                demographics=self._global_demographics.model_copy(),
                by_camera={
                    cid: cm.model_copy() for cid, cm in self._by_camera.items()
                },
            ).model_dump()

    def get_campaign_metrics(self) -> list[dict]:
        """Return a per-campaign breakdown of metrics."""
        with self._lock:
            result: list[dict] = []
            for key, entry in self._by_campaign.items():
                result.append(entry.model_copy(deep=True).model_dump())
            return result

    def get_camera_metrics(self) -> dict[str, dict]:
        """Return per-camera metric breakdown."""
        with self._lock:
            return {
                cid: cm.model_copy().model_dump()
                for cid, cm in self._by_camera.items()
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all counters (called after backend pulls with ``reset=true``)."""
        with self._lock:
            logger.info(
                "Resetting counters (people=%d, campaigns=%d)",
                self._global_traffic.people_total,
                len(self._by_campaign),
            )
            self._global_traffic = TrafficMetrics()
            self._global_attention = AttentionMetrics()
            self._global_demographics = DemographicMetrics()
            self._by_campaign.clear()
            self._by_camera.clear()
            self._dwell_times.clear()
            self._playbacks.clear()

    # ------------------------------------------------------------------
    # Serialisation (for SQLite persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a fully serialisable snapshot of all counter state."""
        with self._lock:
            return {
                "current_campaign_id": self._current_campaign_id,
                "current_video_id": self._current_video_id,
                "global_traffic": self._global_traffic.model_dump(),
                "global_attention": self._global_attention.model_dump(),
                "global_demographics": self._global_demographics.model_dump(),
                "by_campaign": {
                    k: v.model_dump() for k, v in self._by_campaign.items()
                },
                "by_camera": {
                    k: v.model_dump() for k, v in self._by_camera.items()
                },
                "dwell_times": list(self._dwell_times),
                "playbacks": {k: dict(v) for k, v in self._playbacks.items()},
            }

    def from_dict(self, data: dict) -> None:
        """Restore counter state from a previously serialised snapshot."""
        with self._lock:
            self._current_campaign_id = data.get("current_campaign_id")
            self._current_video_id = data.get("current_video_id")

            self._global_traffic = TrafficMetrics(**data.get("global_traffic", {}))
            self._global_attention = AttentionMetrics(**data.get("global_attention", {}))
            self._global_demographics = DemographicMetrics(
                **data.get("global_demographics", {})
            )

            self._by_campaign = {
                k: CampaignEntry(**v)
                for k, v in data.get("by_campaign", {}).items()
            }
            self._by_camera = {
                k: CameraMetrics(**v)
                for k, v in data.get("by_camera", {}).items()
            }
            self._dwell_times = list(data.get("dwell_times", []))
            self._playbacks = {
                k: dict(v) for k, v in data.get("playbacks", {}).items()
            }

            logger.info(
                "Counters restored from snapshot: people=%d, campaigns=%d",
                self._global_traffic.people_total,
                len(self._by_campaign),
            )
