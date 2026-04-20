"""Camera manager that orchestrates multiple FrameGrabber instances.

Provides a single entry-point for the rest of the CV pipeline to obtain
frames from all configured cameras and to inspect their health status.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.capture.grabber import FrameGrabber
from src.config.settings import settings

logger = logging.getLogger(__name__)


class CameraManager:
    """Creates and manages :class:`FrameGrabber` instances for every camera
    defined in the application settings.

    Camera IDs are assigned as ``camera_1`` through ``camera_{N}`` where
    *N* equals ``settings.camera.camera_count``.
    """

    def __init__(self) -> None:
        self._grabbers: dict[str, FrameGrabber] = {}
        self._build_grabbers()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_grabbers(self) -> None:
        """Instantiate a :class:`FrameGrabber` for each configured camera."""
        count: int = settings.camera.camera_count

        for idx in range(1, count + 1):
            camera_id = f"camera_{idx}"
            source = settings.camera.get_url(idx)

            grabber = FrameGrabber(
                source=source,
                camera_id=camera_id,
                capture_fps=settings.capture_fps,
                processing_fps=settings.processing_fps,
            )
            self._grabbers[camera_id] = grabber
            logger.info(
                "Registered %s with source=%s (capture=%d fps, processing=%d fps)",
                camera_id,
                source,
                settings.capture_fps,
                settings.processing_fps,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Start capture threads for every camera."""
        logger.info("Starting all %d camera grabbers ...", len(self._grabbers))
        for camera_id, grabber in self._grabbers.items():
            try:
                grabber.start()
            except Exception:
                logger.exception("Failed to start grabber for %s", camera_id)

    def stop_all(self) -> None:
        """Stop capture threads for every camera."""
        logger.info("Stopping all camera grabbers ...")
        for camera_id, grabber in self._grabbers.items():
            try:
                grabber.stop()
            except Exception:
                logger.exception("Error stopping grabber for %s", camera_id)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frames(self) -> dict[str, np.ndarray]:
        """Return the latest frame from each *active* camera.

        Only cameras that have a fresh frame available are included in
        the returned dictionary.  Cameras that are disconnected or have
        not yet produced a frame are silently omitted.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of ``camera_id`` to the most recent BGR frame.
        """
        frames: dict[str, np.ndarray] = {}

        for camera_id, grabber in self._grabbers.items():
            ok, frame = grabber.get_frame()
            if ok and frame is not None:
                frames[camera_id] = frame

        return frames

    # ------------------------------------------------------------------
    # Health / monitoring
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, dict]:
        """Return per-camera health status.

        Returns
        -------
        dict[str, dict]
            Mapping of ``camera_id`` to a dict containing:

            * ``fps`` -- measured processing FPS.
            * ``frame_count`` -- total frames captured.
            * ``is_alive`` -- whether the capture thread is running.
            * ``last_frame_time`` -- epoch timestamp of the last frame.
        """
        health: dict[str, dict] = {}

        for camera_id, grabber in self._grabbers.items():
            health[camera_id] = grabber.get_health()

        return health

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        """Return an ordered list of camera identifiers."""
        return list(self._grabbers.keys())

    def __len__(self) -> int:
        return len(self._grabbers)

    def __contains__(self, camera_id: str) -> bool:
        return camera_id in self._grabbers

    def __getitem__(self, camera_id: str) -> FrameGrabber:
        return self._grabbers[camera_id]
