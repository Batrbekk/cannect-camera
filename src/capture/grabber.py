"""Frame grabber for RTSP and USB camera capture.

Runs in a dedicated thread to provide non-blocking frame acquisition with
automatic reconnection on stream failure.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Reconnection constants
_INITIAL_BACKOFF_SEC: float = 1.0
_MAX_BACKOFF_SEC: float = 30.0
_BACKOFF_MULTIPLIER: float = 2.0


class FrameGrabber:
    """Captures frames from an RTSP stream or USB camera via OpenCV.

    The grabber runs a background thread that reads frames at the camera's
    native FPS and makes only the most recent frame available to consumers
    via :meth:`get_frame`. A frame-skip mechanism ensures consumers operate
    at the configured *processing* FPS rather than the full capture rate.

    Parameters
    ----------
    source:
        RTSP URL (``rtsp://...``) or integer device index for a USB camera.
    camera_id:
        Human-readable identifier used in log messages and health reports.
    capture_fps:
        Native capture rate of the camera (frames per second).
    processing_fps:
        Down-sampled rate at which frames are made available for processing.
    """

    def __init__(
        self,
        source: str | int,
        camera_id: str,
        capture_fps: int = 30,
        processing_fps: int = 10,
    ) -> None:
        self._source = source
        self._camera_id = camera_id
        self._capture_fps = max(capture_fps, 1)
        self._processing_fps = max(processing_fps, 1)

        # Frame skip: keep every Nth frame so the consumer sees ~processing_fps.
        self._frame_skip: int = max(self._capture_fps // self._processing_fps, 1)

        # Thread-safe frame buffer (latest frame only).
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_ready = False

        # Health / monitoring counters.
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._fps_counter: int = 0
        self._fps_timer: float = 0.0

        # Thread lifecycle.
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[%s] Capture thread already running.", self._camera_id)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"grabber-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Capture thread started for source=%s", self._camera_id, self._source)

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[%s] Capture thread did not exit cleanly.", self._camera_id)
            self._thread = None
        self._release_capture()
        logger.info("[%s] Capture thread stopped.", self._camera_id)

    def is_alive(self) -> bool:
        """Return ``True`` if the capture thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest captured frame.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            ``(True, frame)`` when a fresh frame is available, or
            ``(False, None)`` when no frame has been captured yet or the
            stream is down.
        """
        with self._lock:
            if self._frame is not None and self._frame_ready:
                # Hand over a copy so the consumer can mutate freely.
                frame = self._frame.copy()
                self._frame_ready = False
                return True, frame
            return False, None

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total number of frames delivered to the buffer."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Measured processing FPS (frames written to buffer per second)."""
        return self._fps

    @property
    def last_frame_time(self) -> float:
        """Epoch timestamp of the last frame written to the buffer."""
        return self._last_frame_time

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def get_health(self) -> dict:
        """Return a snapshot of health metrics."""
        return {
            "camera_id": self._camera_id,
            "is_alive": self.is_alive(),
            "fps": round(self._fps, 2),
            "frame_count": self._frame_count,
            "last_frame_time": self._last_frame_time,
        }

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _open_capture(self) -> bool:
        """Open the video capture source. Returns ``True`` on success."""
        self._release_capture()
        try:
            if isinstance(self._source, int):
                self._cap = cv2.VideoCapture(self._source)
            else:
                # Use FFMPEG backend for RTSP; set a short timeout so we
                # don't block forever on an unreachable camera.
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._cap is not None and self._cap.isOpened():
                logger.info("[%s] Opened video source: %s", self._camera_id, self._source)
                return True

            logger.error("[%s] Failed to open video source: %s", self._camera_id, self._source)
            self._release_capture()
            return False
        except Exception:
            logger.exception("[%s] Exception opening video source: %s", self._camera_id, self._source)
            self._release_capture()
            return False

    def _release_capture(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("[%s] Error releasing capture.", self._camera_id)
            finally:
                self._cap = None

    def _reconnect(self) -> bool:
        """Try to reconnect with exponential backoff.

        Returns ``True`` once a connection is established, or ``False`` if
        the stop event is set during the wait.
        """
        backoff = _INITIAL_BACKOFF_SEC
        while not self._stop_event.is_set():
            logger.info(
                "[%s] Attempting reconnect in %.1fs ...",
                self._camera_id,
                backoff,
            )
            if self._stop_event.wait(timeout=backoff):
                # Stop was requested during the wait.
                return False

            if self._open_capture():
                return True

            backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_SEC)

        return False

    def _capture_loop(self) -> None:
        """Main loop executed by the background thread."""
        if not self._open_capture():
            if not self._reconnect():
                logger.error("[%s] Capture loop exiting — could not connect.", self._camera_id)
                return

        grab_index: int = 0
        self._fps_timer = time.monotonic()
        self._fps_counter = 0

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    break
                grab_index = 0
                continue

            ret, frame = self._cap.read()

            if not ret or frame is None:
                logger.warning("[%s] Frame read failed — initiating reconnect.", self._camera_id)
                if not self._reconnect():
                    break
                grab_index = 0
                continue

            grab_index += 1

            # Down-sample: only publish every Nth frame.
            if grab_index % self._frame_skip != 0:
                continue

            now = time.monotonic()

            with self._lock:
                self._frame = frame
                self._frame_ready = True

            # Update health counters (no lock needed — single writer).
            self._frame_count += 1
            self._last_frame_time = time.time()
            self._fps_counter += 1

            # Recalculate measured FPS every second.
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = now

        self._release_capture()
        logger.info("[%s] Capture loop exited.", self._camera_id)
