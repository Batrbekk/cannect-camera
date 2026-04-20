"""ByteTrack multi-object tracker for the CANNECT.AI CV analytics pipeline.

Implements the ByteTrack algorithm (Zhang et al., 2022) which associates
*every* detection box -- including low-confidence ones -- to maximize
tracking recall while keeping precision high via two-stage matching.

Reference: https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import logging
import time
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from src.aggregation.models import DetectionResult, Direction
from src.config.settings import settings

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(IntEnum):
    """Lifecycle state of a single track."""
    New = auto()
    Tracked = auto()
    Lost = auto()
    Removed = auto()


# ---------------------------------------------------------------------------
# Kalman-filter helpers
# ---------------------------------------------------------------------------

def _make_kalman_filter() -> KalmanFilter:
    """Create a constant-velocity Kalman filter for bounding-box tracking.

    State vector (8-d):
        [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
    Measurement vector (4-d):
        [x_center, y_center, aspect_ratio, height]
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # Transition matrix (constant velocity model)
    kf.F = np.eye(8, dtype=np.float64)
    kf.F[:4, 4:] = np.eye(4, dtype=np.float64)

    # Measurement matrix
    kf.H = np.eye(4, 8, dtype=np.float64)

    # Measurement noise
    kf.R[2:, 2:] *= 10.0

    # Initial covariance
    kf.P[4:, 4:] *= 1000.0  # high uncertainty on velocities
    kf.P *= 10.0

    # Process noise
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    return kf


def _bbox_to_z(bbox: tuple[int, int, int, int]) -> NDArray[np.float64]:
    """Convert (x1, y1, x2, y2) bbox to Kalman measurement [cx, cy, a, h]."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float64)


def _x_to_bbox(state: NDArray[np.float64]) -> tuple[int, int, int, int]:
    """Convert Kalman state [cx, cy, a, h, ...] back to (x1, y1, x2, y2)."""
    cx, cy, a, h = state[:4].flatten()
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


# ---------------------------------------------------------------------------
# IoU computation & matching
# ---------------------------------------------------------------------------

def _iou_batch(
    bboxes_a: NDArray[np.float64],
    bboxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise IoU between two sets of (x1, y1, x2, y2) boxes.

    Parameters
    ----------
    bboxes_a : (N, 4) array
    bboxes_b : (M, 4) array

    Returns
    -------
    iou_matrix : (N, M) array of IoU values.
    """
    n = bboxes_a.shape[0]
    m = bboxes_b.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m), dtype=np.float64)

    # Broadcast intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0])  # (N, M)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1])
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2])
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _linear_assignment(
    cost_matrix: NDArray[np.float64],
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the assignment problem with a cost threshold.

    Uses the Hungarian algorithm via ``scipy.optimize.linear_sum_assignment``.

    Returns
    -------
    matches : list of (row_idx, col_idx) pairs
    unmatched_rows : row indices with no valid match
    unmatched_cols : col indices with no valid match
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append((r, c))
        unmatched_rows.discard(r)
        unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

class Track:
    """Single object track with Kalman-filter state estimation."""

    _next_id: int = 1  # class-level auto-increment counter

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        score: float,
        frame_rate: int,
    ) -> None:
        self.track_id: int = Track._next_id
        Track._next_id += 1

        self._kf: KalmanFilter = _make_kalman_filter()
        self._kf.x[:4] = _bbox_to_z(bbox).reshape(4, 1)

        self.score: float = score
        self.state: TrackState = TrackState.New
        self.frame_count: int = 1
        self._frame_rate: int = frame_rate

        now = time.time()
        self.first_seen: float = now
        self.last_seen: float = now

        # Direction tracking -- store (cx, cy) per frame for trajectory.
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._center_history: list[tuple[float, float]] = [(cx, cy)]

        self._frames_since_update: int = 0

        logger.debug("Created track %d  bbox=%s  score=%.3f", self.track_id, bbox, score)

    # -- properties ----------------------------------------------------------

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Current estimated bounding box from the Kalman state."""
        return _x_to_bbox(self._kf.x)

    @property
    def direction(self) -> Direction | None:
        """Return the computed movement direction, or None if undetermined."""
        return self.get_direction()

    @property
    def is_confirmed(self) -> bool:
        """A track is considered confirmed after ``min_track_frames`` frames."""
        return self.frame_count >= settings.min_track_frames

    # -- Kalman wrappers -----------------------------------------------------

    def predict(self) -> None:
        """Advance the Kalman filter one time-step (no measurement)."""
        self._kf.predict()
        self._frames_since_update += 1

    def update(self, bbox: tuple[int, int, int, int], score: float) -> None:
        """Correct the Kalman state with a matched detection."""
        z = _bbox_to_z(bbox).reshape(4, 1)
        self._kf.update(z)
        self.score = score
        self.frame_count += 1
        self.last_seen = time.time()
        self._frames_since_update = 0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._center_history.append((cx, cy))

        # Keep memory bounded -- last 120 positions is plenty.
        if len(self._center_history) > 120:
            self._center_history = self._center_history[-120:]

    # -- direction -----------------------------------------------------------

    def get_direction(self) -> Direction | None:
        """Calculate movement direction from the center trajectory.

        Uses the displacement between the first and last recorded center
        positions.  The primary axis (horizontal vs vertical) determines
        the direction.  For top-mounted cameras, decreasing-y means the
        person is walking *toward* the screen/camera.

        Returns ``None`` when there are fewer than ``settings.min_track_frames``
        observations.
        """
        if len(self._center_history) < settings.min_track_frames:
            return None

        first_cx, first_cy = self._center_history[0]
        last_cx, last_cy = self._center_history[-1]

        dx = last_cx - first_cx
        dy = last_cy - first_cy

        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return None

        if abs(dy) >= abs(dx):
            # Primarily vertical movement.
            # Decreasing y (moving up in image coords) -> toward camera/screen.
            return Direction.TO_SCREEN if dy < 0 else Direction.FROM_SCREEN
        else:
            return Direction.RIGHT if dx > 0 else Direction.LEFT

    # -- helpers -------------------------------------------------------------

    def mark_tracked(self) -> None:
        self.state = TrackState.Tracked

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the auto-increment counter (useful between sessions)."""
        cls._next_id = 1

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, state={self.state.name}, "
            f"frames={self.frame_count}, bbox={self.bbox})"
        )


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """ByteTrack multi-object tracker.

    Parameters
    ----------
    track_thresh : float
        Confidence threshold that splits detections into high / low groups.
    track_buffer : int
        Number of frames a lost track is kept before removal.
    match_thresh : float
        IoU threshold (as a *cost* ceiling: ``1 - IoU``) for the first
        association round.
    frame_rate : int
        Expected processing frame-rate -- used for internal bookkeeping.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 10,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self._tracked_tracks: list[Track] = []
        self._lost_tracks: list[Track] = []
        self._removed_tracks: list[Track] = []
        self._frame_id: int = 0

        logger.info(
            "ByteTracker initialised  thresh=%.2f  buffer=%d  match=%.2f  fps=%d",
            track_thresh,
            track_buffer,
            match_thresh,
            frame_rate,
        )

    # -- public API ----------------------------------------------------------

    def update(
        self,
        detections: list[DetectionResult],
        frame_shape: tuple[int, ...],
    ) -> list[Track]:
        """Run one tracking cycle and return active tracks.

        Parameters
        ----------
        detections : list[DetectionResult]
            Raw detector output for the current frame.
        frame_shape : tuple
            Shape of the current frame ``(height, width, ...)``.

        Returns
        -------
        list[Track]
            Tracks with state ``Tracked`` (confirmed and currently visible).
        """
        self._frame_id += 1

        # -----------------------------------------------------------------
        # 0. Unpack detections into numpy arrays
        # -----------------------------------------------------------------
        if detections:
            bboxes = np.array([d.bbox for d in detections], dtype=np.float64)
            scores = np.array([d.confidence for d in detections], dtype=np.float64)
        else:
            bboxes = np.empty((0, 4), dtype=np.float64)
            scores = np.empty(0, dtype=np.float64)

        # Filter detections by minimum bbox height.
        keep = (bboxes[:, 3] - bboxes[:, 1]) >= settings.min_person_bbox_height if len(bboxes) else np.array([], dtype=bool)
        bboxes = bboxes[keep]
        scores = scores[keep]

        # -----------------------------------------------------------------
        # 1. Split detections into high-score and low-score groups
        # -----------------------------------------------------------------
        high_mask = scores >= self.track_thresh
        low_mask = ~high_mask

        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]

        logger.debug(
            "Frame %d: %d detections (%d high, %d low)",
            self._frame_id,
            len(bboxes),
            len(high_bboxes),
            len(low_bboxes),
        )

        # -----------------------------------------------------------------
        # 2. Predict new locations for all existing tracks
        # -----------------------------------------------------------------
        all_tracks = self._tracked_tracks + self._lost_tracks
        for t in all_tracks:
            t.predict()

        # Only consider currently-tracked (not lost) tracks for 1st round.
        tracked_pool = [t for t in self._tracked_tracks if t.state == TrackState.Tracked]
        # New tracks that haven't been confirmed yet still participate.
        new_pool = [t for t in self._tracked_tracks if t.state == TrackState.New]
        tracked_pool += new_pool

        # -----------------------------------------------------------------
        # 3. First association -- high-score detections vs tracked tracks
        # -----------------------------------------------------------------
        matched_track_indices_1: list[int] = []
        matched_det_indices_1: list[int] = []
        unmatched_tracks_1: list[int]
        unmatched_dets_1: list[int]

        if len(tracked_pool) > 0 and len(high_bboxes) > 0:
            track_bboxes = np.array([t.bbox for t in tracked_pool], dtype=np.float64)
            iou_matrix = _iou_batch(track_bboxes, high_bboxes)
            cost_matrix = 1.0 - iou_matrix
            matches, unmatched_tracks_1, unmatched_dets_1 = _linear_assignment(
                cost_matrix, thresh=1.0 - self.match_thresh,
            )
            for ti, di in matches:
                matched_track_indices_1.append(ti)
                matched_det_indices_1.append(di)
        else:
            unmatched_tracks_1 = list(range(len(tracked_pool)))
            unmatched_dets_1 = list(range(len(high_bboxes)))

        # Apply first-round matches.
        for ti, di in zip(matched_track_indices_1, matched_det_indices_1):
            track = tracked_pool[ti]
            bb = tuple(int(v) for v in high_bboxes[di])
            track.update(bb, float(high_scores[di]))  # type: ignore[arg-type]
            track.mark_tracked()

        # -----------------------------------------------------------------
        # 4. Second association -- low-score detections vs remaining tracks
        # -----------------------------------------------------------------
        remaining_tracks = [tracked_pool[i] for i in unmatched_tracks_1]

        matched_track_indices_2: list[int] = []
        matched_det_indices_2: list[int] = []
        unmatched_tracks_2: list[int]

        if len(remaining_tracks) > 0 and len(low_bboxes) > 0:
            track_bboxes = np.array([t.bbox for t in remaining_tracks], dtype=np.float64)
            iou_matrix = _iou_batch(track_bboxes, low_bboxes)
            cost_matrix = 1.0 - iou_matrix
            # Use a relaxed threshold for low-score matches (0.5 IoU).
            matches, unmatched_tracks_2, _ = _linear_assignment(
                cost_matrix, thresh=0.5,
            )
            for ti, di in matches:
                matched_track_indices_2.append(ti)
                matched_det_indices_2.append(di)
        else:
            unmatched_tracks_2 = list(range(len(remaining_tracks)))

        for ti, di in zip(matched_track_indices_2, matched_det_indices_2):
            track = remaining_tracks[ti]
            bb = tuple(int(v) for v in low_bboxes[di])
            track.update(bb, float(low_scores[di]))  # type: ignore[arg-type]
            track.mark_tracked()

        # -----------------------------------------------------------------
        # 5. Handle unmatched tracks -> lost or removed
        # -----------------------------------------------------------------
        still_unmatched = [remaining_tracks[i] for i in unmatched_tracks_2]
        for track in still_unmatched:
            if track.state != TrackState.Lost:
                track.mark_lost()

        # Also try to match lost tracks with unmatched high-score detections.
        unmatched_high_bboxes = high_bboxes[unmatched_dets_1] if unmatched_dets_1 else np.empty((0, 4), dtype=np.float64)
        unmatched_high_scores = high_scores[unmatched_dets_1] if unmatched_dets_1 else np.empty(0, dtype=np.float64)

        recovered_det_indices: list[int] = []
        if len(self._lost_tracks) > 0 and len(unmatched_high_bboxes) > 0:
            lost_bboxes = np.array([t.bbox for t in self._lost_tracks], dtype=np.float64)
            iou_matrix = _iou_batch(lost_bboxes, unmatched_high_bboxes)
            cost_matrix = 1.0 - iou_matrix
            matches, _, unmatched_det_remaining = _linear_assignment(
                cost_matrix, thresh=1.0 - self.match_thresh,
            )
            for ti, di in matches:
                track = self._lost_tracks[ti]
                bb = tuple(int(v) for v in unmatched_high_bboxes[di])
                track.update(bb, float(unmatched_high_scores[di]))  # type: ignore[arg-type]
                track.mark_tracked()
                recovered_det_indices.append(di)

            # Update the set of truly-unmatched detections.
            truly_unmatched_dets = [unmatched_high_bboxes[i] for i in unmatched_det_remaining]
            truly_unmatched_scores = [float(unmatched_high_scores[i]) for i in unmatched_det_remaining]
        else:
            truly_unmatched_dets = list(unmatched_high_bboxes)
            truly_unmatched_scores = [float(s) for s in unmatched_high_scores]

        # -----------------------------------------------------------------
        # 6. Create new tracks from unmatched high-score detections
        # -----------------------------------------------------------------
        for bbox_arr, sc in zip(truly_unmatched_dets, truly_unmatched_scores):
            bb = tuple(int(v) for v in bbox_arr)
            new_track = Track(bbox=bb, score=sc, frame_rate=self.frame_rate)  # type: ignore[arg-type]
            self._tracked_tracks.append(new_track)
            logger.debug("New track %d from unmatched detection", new_track.track_id)

        # -----------------------------------------------------------------
        # 7. Update track pools
        # -----------------------------------------------------------------
        new_tracked: list[Track] = []
        new_lost: list[Track] = []

        for track in self._tracked_tracks:
            if track.state == TrackState.Tracked or track.state == TrackState.New:
                new_tracked.append(track)
            elif track.state == TrackState.Lost:
                new_lost.append(track)

        # Re-check lost tracks for timeout.
        for track in self._lost_tracks:
            if track.state == TrackState.Tracked:
                # Recovered in step 5 above.
                new_tracked.append(track)
            elif track._frames_since_update > self.track_buffer:
                track.mark_removed()
                self._removed_tracks.append(track)
                logger.debug(
                    "Removed track %d after %d lost frames",
                    track.track_id,
                    track._frames_since_update,
                )
            else:
                new_lost.append(track)

        # Also move still_unmatched tracks that were previously tracked.
        # (They were already marked Lost above and will be in new_lost via
        #  the tracked_tracks loop.)

        self._tracked_tracks = new_tracked
        self._lost_tracks = new_lost

        # Bound the removed list to prevent unbounded memory growth.
        if len(self._removed_tracks) > 1000:
            self._removed_tracks = self._removed_tracks[-500:]

        # -----------------------------------------------------------------
        # 8. Return confirmed, currently-visible tracks
        # -----------------------------------------------------------------
        output = [t for t in self._tracked_tracks if t.is_confirmed and t.state == TrackState.Tracked]

        logger.debug(
            "Frame %d result: %d active tracks, %d lost, %d removed total",
            self._frame_id,
            len(output),
            len(self._lost_tracks),
            len(self._removed_tracks),
        )

        return output

    # -- housekeeping --------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracks and reset the frame counter."""
        self._tracked_tracks.clear()
        self._lost_tracks.clear()
        self._removed_tracks.clear()
        self._frame_id = 0
        Track.reset_id_counter()
        logger.info("ByteTracker reset")

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def tracked_track_count(self) -> int:
        return len(self._tracked_tracks)

    @property
    def lost_track_count(self) -> int:
        return len(self._lost_tracks)
