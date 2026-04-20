"""Attention (dwell-time) tracking per person.

Tracks how long each person directs their gaze toward the screen and
provides aggregate statistics used to build the *attention* event payload.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.aggregation.models import GazeStatus
from src.config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class _TrackState:
    """Internal bookkeeping for a single tracked person."""

    # Accumulated dwell time (seconds).
    total_dwell: float = 0.0

    # True while the person is considered to be actively attending.
    is_attending: bool = False

    # Timestamp when the current attention session started.
    attend_start: float | None = None

    # Counter: consecutive frames with *direct* gaze.
    direct_streak: int = 0

    # Counter: consecutive frames with deviation > threshold.
    deviation_streak: int = 0

    # Total frames while gaze was "direct".
    direct_frames: int = 0

    # Total frames observed (all gaze statuses).
    total_frames: int = 0

    # Whether this person ever triggered attention (looking).
    has_looked: bool = False


class AttentionTracker:
    """Track per-person attention duration and compute aggregate scores.

    Attention is started when the person maintains a ``direct`` gaze for
    at least ``settings.attention_start_frames`` consecutive frames and
    stopped when gaze deviates beyond the partial range for
    ``settings.attention_stop_frames`` consecutive frames.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id: int) -> _TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackState()
        return self._tracks[track_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, track_id: int, gaze_status: GazeStatus, timestamp: float) -> None:
        """Update attention state for *track_id* at *timestamp*.

        Parameters
        ----------
        track_id:
            Unique tracker identifier.
        gaze_status:
            Current gaze classification (direct / partial / glance).
        timestamp:
            Current epoch time (seconds).
        """
        state = self._get_or_create(track_id)
        state.total_frames += 1

        is_looking = gaze_status in (GazeStatus.DIRECT, GazeStatus.PARTIAL)
        is_deviated = gaze_status == GazeStatus.GLANCE

        if is_looking:
            state.direct_frames += 1
            state.direct_streak += 1
            state.deviation_streak = 0
        else:
            state.deviation_streak += 1
            state.direct_streak = 0

        # --- Start attention (direct OR partial = looking at screen) ---
        if not state.is_attending and state.direct_streak >= 1:
            state.is_attending = True
            state.has_looked = True
            state.attend_start = timestamp
            logger.debug("Track %d: attention started at %.2f", track_id, timestamp)

        # --- Stop attention (must look away for 5+ frames) ---
        if (
            state.is_attending
            and state.deviation_streak >= settings.attention_stop_frames
        ):
            if state.attend_start is not None:
                elapsed = timestamp - state.attend_start
                state.total_dwell += max(0.0, elapsed)
            state.is_attending = False
            state.attend_start = None
            logger.debug("Track %d: attention stopped at %.2f", track_id, timestamp)

    def get_dwell_time(self, track_id: int) -> float:
        """Return total accumulated dwell time in seconds for *track_id*.

        If the person is currently attending, the ongoing session is
        included in the total.
        """
        state = self._tracks.get(track_id)
        if state is None:
            return 0.0

        total = state.total_dwell
        if state.is_attending and state.attend_start is not None:
            total += max(0.0, time.time() - state.attend_start)
        return total

    def is_attention_over_threshold(self, track_id: int) -> bool:
        """Return ``True`` when dwell time >= ``settings.attention_dwell_threshold_sec``."""
        return self.get_dwell_time(track_id) >= settings.attention_dwell_threshold_sec

    def get_stats(self) -> dict:
        """Compute aggregate attention statistics across all tracks.

        Returns
        -------
        dict
            ``{totalLooking, attentionOver5s, averageDwellTime, attentionScore}``
        """
        total_people = len(self._tracks)
        total_looking = 0
        attention_over_5s = 0
        dwell_sum = 0.0
        direct_count = 0

        for track_id, state in self._tracks.items():
            if state.has_looked:
                total_looking += 1
            dwell = self.get_dwell_time(track_id)
            dwell_sum += dwell
            if dwell >= settings.attention_dwell_threshold_sec:
                attention_over_5s += 1
            direct_count += state.direct_frames

        avg_dwell = dwell_sum / max(total_looking, 1)

        # Attention score (0-100).
        dwell_weight = min(1.0, avg_dwell / 10.0)
        frequency_weight = min(1.0, total_looking / max(total_people, 1))
        total_direct_frames = sum(s.direct_frames for s in self._tracks.values())
        total_all_frames = sum(s.total_frames for s in self._tracks.values())
        direct_ratio = total_direct_frames / max(total_all_frames, 1)

        attention_score = min(
            100,
            int(dwell_weight * 40 + frequency_weight * 30 + direct_ratio * 30),
        )

        return {
            "totalLooking": total_looking,
            "attentionOver5s": attention_over_5s,
            "averageDwellTime": round(avg_dwell, 2),
            "attentionScore": attention_score,
        }

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Remove data for tracks that are no longer active.

        Parameters
        ----------
        active_track_ids:
            Set of track IDs still being tracked.  Any track **not** in
            this set will have its attention state discarded.
        """
        stale = [tid for tid in self._tracks if tid not in active_track_ids]
        for tid in stale:
            # Finalise ongoing session before removal.
            state = self._tracks[tid]
            if state.is_attending and state.attend_start is not None:
                state.total_dwell += max(0.0, time.time() - state.attend_start)
                state.is_attending = False
                state.attend_start = None
            del self._tracks[tid]

        if stale:
            logger.debug("AttentionTracker cleanup: removed %d stale tracks.", len(stale))
