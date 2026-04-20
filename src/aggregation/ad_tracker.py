"""Tracks which ad is currently playing on the station's screen.

Receives playback lifecycle events from the local video player and
updates the CounterStore so that CV detections are attributed to the
correct campaign.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from src.aggregation.counters import CounterStore
from src.aggregation.models import CurrentAdInfo, CurrentAdPayload

logger = logging.getLogger(__name__)

# Maximum number of playback records to keep in memory.
_MAX_HISTORY = 100


class AdTracker:
    """Manage playback lifecycle and campaign attribution.

    The local video player posts events (``playback_started``,
    ``playback_ended``, ``playback_changed``) to the edge HTTP server,
    which delegates to this class.  AdTracker in turn drives
    :meth:`CounterStore.set_current_campaign` and
    :meth:`CounterStore.add_playback`.

    Parameters
    ----------
    counter_store:
        The shared CounterStore instance that accumulates CV metrics.
    """

    def __init__(self, counter_store: CounterStore) -> None:
        self._counter_store = counter_store
        self._lock = Lock()

        # Current playback state
        self._current_campaign_id: str | None = None
        self._current_video_id: str | None = None
        self._playback_started_at: str | None = None
        self._playback_start_mono: float | None = None
        self._expected_duration: int | None = None

        # Recent playback history (most recent last)
        self._history: deque[dict] = deque(maxlen=_MAX_HISTORY)

        logger.debug("AdTracker initialised")

    # ------------------------------------------------------------------
    # Playback event handler
    # ------------------------------------------------------------------

    def on_playback_event(
        self,
        event: str,
        campaign_id: str | None,
        video_id: str | None,
        started_at: str | None = None,
        expected_duration: int | None = None,
    ) -> None:
        """Handle a playback lifecycle event from the video player.

        Parameters
        ----------
        event:
            One of ``playback_started``, ``playback_ended``,
            ``playback_changed``.
        campaign_id:
            The campaign associated with the video, or ``None``.
        video_id:
            Identifier of the specific video asset.
        started_at:
            ISO-8601 timestamp when playback began (provided by the player).
        expected_duration:
            Expected length of the video in seconds.
        """
        with self._lock:
            if event == "playback_started":
                self._handle_start(campaign_id, video_id, started_at, expected_duration)
            elif event == "playback_ended":
                self._handle_end()
            elif event == "playback_changed":
                self._handle_changed(campaign_id, video_id, started_at, expected_duration)
            else:
                logger.warning("Unknown playback event: %s", event)

    # ------------------------------------------------------------------
    # Internal handlers (called under lock)
    # ------------------------------------------------------------------

    def _handle_start(
        self,
        campaign_id: str | None,
        video_id: str | None,
        started_at: str | None,
        expected_duration: int | None,
    ) -> None:
        """Begin tracking a new playback."""
        self._current_campaign_id = campaign_id
        self._current_video_id = video_id
        self._playback_started_at = started_at or datetime.now(tz=timezone.utc).isoformat()
        self._playback_start_mono = time.monotonic()
        self._expected_duration = expected_duration

        # Propagate to CounterStore
        self._counter_store.set_current_campaign(campaign_id, video_id)

        logger.info(
            "Playback started: campaign=%s video=%s expected_duration=%s",
            campaign_id,
            video_id,
            expected_duration,
        )

    def _handle_end(self) -> None:
        """Finalise the current playback and record its duration."""
        if self._playback_start_mono is not None:
            duration = time.monotonic() - self._playback_start_mono
        elif self._expected_duration is not None:
            duration = float(self._expected_duration)
        else:
            duration = 0.0

        # Record playback in counters if there was an active campaign
        if self._current_campaign_id is not None and duration > 0:
            self._counter_store.add_playback(
                campaign_id=self._current_campaign_id,
                video_id=self._current_video_id or "",
                duration_seconds=round(duration, 2),
            )

        # Append to history
        self._history.append({
            "campaignId": self._current_campaign_id,
            "videoId": self._current_video_id,
            "startedAt": self._playback_started_at,
            "endedAt": datetime.now(tz=timezone.utc).isoformat(),
            "durationSeconds": round(duration, 2),
        })

        logger.info(
            "Playback ended: campaign=%s video=%s duration=%.1fs",
            self._current_campaign_id,
            self._current_video_id,
            duration,
        )

        # Clear current state
        self._current_campaign_id = None
        self._current_video_id = None
        self._playback_started_at = None
        self._playback_start_mono = None
        self._expected_duration = None

        # Propagate to CounterStore
        self._counter_store.set_current_campaign(None, None)

    def _handle_changed(
        self,
        campaign_id: str | None,
        video_id: str | None,
        started_at: str | None,
        expected_duration: int | None,
    ) -> None:
        """End the previous playback and immediately start a new one."""
        self._handle_end()
        self._handle_start(campaign_id, video_id, started_at, expected_duration)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def get_current_ad(self) -> CurrentAdInfo:
        """Return information about the currently-playing ad."""
        with self._lock:
            return CurrentAdInfo(
                campaignId=self._current_campaign_id,
                videoId=self._current_video_id,
                playbackStartedAt=self._playback_started_at,
            )

    def get_playback_history(self) -> list[dict]:
        """Return recent playback records (up to the last 100)."""
        with self._lock:
            return list(self._history)
