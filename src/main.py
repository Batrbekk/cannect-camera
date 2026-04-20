"""CANNECT.AI CV Analytics Pipeline — pull-based architecture.

Runs the CV pipeline loop continuously, feeds metrics into CounterStore,
and serves an edge HTTP API (FastAPI) for the backend to pull from.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import date
from typing import Any

import numpy as np
import uvicorn

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful-degradation imports.  Each module is optional — the pipeline
# simply skips the corresponding analysis step when it is unavailable.
# ---------------------------------------------------------------------------

try:
    from src.capture.camera_manager import CameraManager
except Exception:
    CameraManager = None  # type: ignore[assignment,misc]
    logger.warning("CameraManager unavailable")

try:
    from src.detection.person_detector import PersonDetector
except Exception:
    PersonDetector = None  # type: ignore[assignment,misc]
    logger.warning("PersonDetector unavailable")

try:
    from src.detection.face_detector import FaceDetector
except Exception:
    FaceDetector = None  # type: ignore[assignment,misc]
    logger.warning("FaceDetector unavailable — face analysis will be skipped")

try:
    from src.tracking.bytetrack import ByteTracker
except Exception:
    ByteTracker = None  # type: ignore[assignment,misc]
    logger.warning("ByteTracker unavailable — tracking will be skipped")

try:
    from src.analysis.head_pose import HeadPoseEstimator
except Exception:
    HeadPoseEstimator = None  # type: ignore[assignment,misc]
    logger.warning("HeadPoseEstimator unavailable")

try:
    from src.analysis.gender_age import GenderAgeEstimator
except Exception:
    GenderAgeEstimator = None  # type: ignore[assignment,misc]
    logger.warning("GenderAgeEstimator unavailable")

try:
    from src.analysis.face_embedding import FaceEmbedder, FaceStore
except Exception:
    FaceEmbedder = None  # type: ignore[assignment,misc]
    FaceStore = None  # type: ignore[assignment,misc]
    logger.warning("Face embedding unavailable")

try:
    from src.analysis.attention import AttentionTracker
except Exception:
    AttentionTracker = None  # type: ignore[assignment,misc]
    logger.warning("AttentionTracker unavailable")

try:
    from src.analysis.emotion import EmotionClassifier
except Exception:
    EmotionClassifier = None  # type: ignore[assignment,misc]
    logger.warning("EmotionClassifier unavailable")

try:
    from src.aggregation.counters import CounterStore
except Exception:
    CounterStore = None  # type: ignore[assignment,misc]
    logger.warning("CounterStore unavailable")

try:
    from src.aggregation.ad_tracker import AdTracker
except Exception:
    AdTracker = None  # type: ignore[assignment,misc]
    logger.warning("AdTracker unavailable")

try:
    from src.aggregation.persistence import CounterPersistence
except Exception:
    CounterPersistence = None  # type: ignore[assignment,misc]
    logger.warning("CounterPersistence unavailable")

try:
    from src.server import setup_api, create_app
except Exception:
    setup_api = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    logger.warning("Edge server (setup_api / create_app) unavailable")

from src.aggregation.models import (
    AgeGroup,
    Direction,
    GazeStatus,
    Gender,
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CVAnalyticsPipeline:
    """Pull-based CV analytics pipeline for CANNECT.AI.

    Processes frames from up to 5 cameras, runs person/face detection,
    tracking and analysis, and feeds all metrics into a CounterStore that
    the backend pulls from via the edge HTTP API.
    """

    def __init__(self) -> None:
        self._configure_logging()
        logger.info("Initialising CVAnalyticsPipeline (pull-based) ...")

        self.start_time: float = time.monotonic()
        self._running: bool = False
        self._uvicorn_server: uvicorn.Server | None = None
        self._current_date: date = date.today()

        # -- Capture ---------------------------------------------------
        self.camera_manager: Any = None
        if CameraManager is not None:
            try:
                self.camera_manager = CameraManager()
            except Exception:
                logger.exception("Failed to initialise CameraManager")

        # -- Detection -------------------------------------------------
        self.person_detector: Any = None
        if PersonDetector is not None:
            try:
                model = f"{settings.models_dir}/yolov8n.onnx"
                self.person_detector = PersonDetector(model_path=model)
            except Exception:
                logger.exception("Failed to initialise PersonDetector")

        self.face_detector: Any = None
        if FaceDetector is not None:
            try:
                model = f"{settings.models_dir}/scrfd_500m.onnx"
                self.face_detector = FaceDetector(model_path=model)
            except Exception:
                logger.exception("Failed to initialise FaceDetector")

        # -- Tracking (one ByteTracker per camera) ---------------------
        self.trackers: dict[str, Any] = {}
        if ByteTracker is not None and self.camera_manager is not None:
            for cam_id in self.camera_manager.camera_ids:
                try:
                    self.trackers[cam_id] = ByteTracker()
                except Exception:
                    logger.exception("Failed to create ByteTracker for %s", cam_id)

        # -- Analysis --------------------------------------------------
        self.head_pose: Any = None
        if HeadPoseEstimator is not None:
            try:
                self.head_pose = HeadPoseEstimator()
            except Exception:
                logger.exception("Failed to initialise HeadPoseEstimator")

        self.gender_age: Any = None
        if GenderAgeEstimator is not None:
            try:
                model = f"{settings.models_dir}/genderage.onnx"
                self.gender_age = GenderAgeEstimator(model_path=model)
            except Exception:
                logger.exception("Failed to initialise GenderAgeEstimator")

        self.face_embedder: Any = None
        self.face_store: Any = None
        if FaceEmbedder is not None and FaceStore is not None:
            try:
                model = f"{settings.models_dir}/mobilefacenet.onnx"
                self.face_embedder = FaceEmbedder(model_path=model)
                self.face_store = FaceStore(
                    match_threshold=settings.face_match_threshold,
                    ttl_sec=settings.unique_face_ttl_sec,
                )
            except Exception:
                logger.exception("Failed to initialise face embedding")

        self.attention_tracker: Any = None
        if AttentionTracker is not None:
            try:
                self.attention_tracker = AttentionTracker()
            except Exception:
                logger.exception("Failed to initialise AttentionTracker")

        self.emotion_classifier: Any = None
        if EmotionClassifier is not None:
            try:
                model = f"{settings.models_dir}/hsemotion.onnx"
                self.emotion_classifier = EmotionClassifier(model_path=model)
            except Exception:
                logger.exception("Failed to initialise EmotionClassifier")

        # -- Aggregation -----------------------------------------------
        self.counter_store: Any = None
        if CounterStore is not None:
            try:
                self.counter_store = CounterStore()
            except Exception:
                logger.exception("Failed to initialise CounterStore")

        self.ad_tracker: Any = None
        if AdTracker is not None:
            try:
                self.ad_tracker = AdTracker()
            except Exception:
                logger.exception("Failed to initialise AdTracker")

        self.counter_persistence: Any = None
        if CounterPersistence is not None:
            try:
                self.counter_persistence = CounterPersistence(
                    db_path=settings.counter_persistence_path,
                )
            except Exception:
                logger.exception("Failed to initialise CounterPersistence")

        # -- Edge HTTP API (FastAPI) -----------------------------------
        self.app: Any = None
        if setup_api is not None and create_app is not None:
            try:
                setup_api(self.counter_store, self.ad_tracker, self.camera_manager)
                self.app = create_app()
            except Exception:
                logger.exception("Failed to setup FastAPI edge server")

        logger.info(
            "Pipeline initialised: cameras=%s, person=%s, face=%s, trackers=%d, "
            "headpose=%s, gender_age=%s, embedding=%s, attention=%s, emotion=%s, "
            "counter_store=%s, ad_tracker=%s",
            self.camera_manager is not None,
            self.person_detector is not None,
            self.face_detector is not None,
            len(self.trackers),
            self.head_pose is not None,
            self.gender_age is not None,
            self.face_embedder is not None,
            self.attention_tracker is not None,
            self.emotion_classifier is not None,
            self.counter_store is not None,
            self.ad_tracker is not None,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start cameras, launch the edge API server, and run the CV loop."""
        self._running = True
        self.start_time = time.monotonic()

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.shutdown()))
            except NotImplementedError:
                # Signal handlers are not supported on Windows for SIGTERM
                pass

        # 1. Start cameras
        if self.camera_manager is not None:
            self.camera_manager.start_all()
            logger.info("All cameras started")

        # 2. Start FastAPI server in a background thread
        if self.app is not None:
            self._start_uvicorn_background()
            logger.info(
                "Edge API server started on %s:%d",
                settings.server_host,
                settings.server_port,
            )

        # 3. Load persisted counters from SQLite if available
        if self.counter_persistence is not None and self.counter_store is not None:
            try:
                self.counter_persistence.load_into(self.counter_store)
                logger.info("Loaded persisted counters from SQLite")
            except Exception:
                logger.exception("Failed to load persisted counters")

        # 4. Launch background maintenance tasks
        persist_task = asyncio.create_task(self._persist_loop(), name="persist")
        cleanup_task = asyncio.create_task(self._cleanup_loop(), name="cleanup")
        day_reset_task = asyncio.create_task(self._day_reset_loop(), name="day-reset")

        # 5. Main processing loop
        frame_interval = 1.0 / settings.processing_fps
        logger.info("Entering main loop at %d fps", settings.processing_fps)

        try:
            while self._running:
                cycle_start = time.monotonic()

                if self.camera_manager is not None:
                    frames = await asyncio.to_thread(self.camera_manager.get_frames)

                    for camera_id, frame in frames.items():
                        try:
                            await self._process_camera_frame(camera_id, frame)
                        except Exception:
                            logger.exception("Error processing frame for %s", camera_id)

                # Pace the loop
                elapsed = time.monotonic() - cycle_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        finally:
            await self.shutdown()

            for task in (persist_task, cleanup_task, day_reset_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    async def _process_camera_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Run detection + analysis on a single camera frame and feed
        results into CounterStore."""

        if self.person_detector is None:
            return

        # 1. Person detection
        detections = await asyncio.to_thread(self.person_detector.detect, frame)
        if not detections:
            return

        # 2. Update tracker
        tracker = self.trackers.get(camera_id)
        tracked_persons: list[Any] = []
        if tracker is not None:
            tracked_persons = await asyncio.to_thread(tracker.update, detections, frame)
        else:
            tracked_persons = detections

        for person in tracked_persons:
            track_id = getattr(person, "track_id", None)
            frame_count = getattr(person, "frame_count", settings.min_track_frames)

            if frame_count < settings.min_track_frames:
                continue

            # Extract person crop
            bbox = getattr(person, "bbox", None)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            person_crop = frame[
                max(0, int(y1)):min(frame.shape[0], int(y2)),
                max(0, int(x1)):min(frame.shape[1], int(x2)),
            ]
            if person_crop.size == 0:
                continue

            # -- Feed traffic into CounterStore --
            direction = getattr(person, "direction", None)
            if self.counter_store is not None and direction is not None:
                self.counter_store.add_person(camera_id, direction)

            # 3. Face detection on person crop
            face_result = None
            if self.face_detector is not None:
                try:
                    face_results = await asyncio.to_thread(
                        self.face_detector.detect, person_crop
                    )
                    if face_results:
                        face_result = face_results[0]
                except Exception:
                    logger.debug("Face detection failed for track %s", track_id)

            if face_result is None:
                continue

            face_bbox = face_result.bbox
            fx1, fy1, fx2, fy2 = face_bbox
            face_crop = person_crop[
                max(0, int(fy1)):min(person_crop.shape[0], int(fy2)),
                max(0, int(fx1)):min(person_crop.shape[1], int(fx2)),
            ]
            if face_crop.size == 0:
                continue

            # 4. Head pose -> classify gaze -> update attention
            gaze_status: GazeStatus | None = None
            is_looking = False
            dwell_time = 0.0
            is_over_5s = False

            if self.head_pose is not None:
                try:
                    yaw, pitch, roll = await asyncio.to_thread(
                        self.head_pose.estimate, face_crop
                    )
                    gaze_status = self._classify_gaze(yaw, pitch)
                    is_looking = gaze_status in (GazeStatus.DIRECT, GazeStatus.PARTIAL)
                except Exception:
                    logger.debug("Head pose estimation failed for track %s", track_id)

            if self.attention_tracker is not None and gaze_status is not None and track_id is not None:
                try:
                    dwell_time, is_over_5s = self.attention_tracker.update(
                        track_id=track_id,
                        gaze_status=gaze_status,
                    )
                except Exception:
                    logger.debug("Attention tracking failed for track %s", track_id)

            # Feed attention into CounterStore
            if self.counter_store is not None and track_id is not None:
                self.counter_store.add_attention(
                    camera_id, track_id, dwell_time, is_over_5s, is_looking
                )

            # 5. Gender/age estimation (only on frontal faces)
            is_frontal = gaze_status in (GazeStatus.DIRECT, GazeStatus.PARTIAL) if gaze_status else False
            if self.gender_age is not None and is_frontal:
                try:
                    gender, gender_conf, age, age_conf = await asyncio.to_thread(
                        self.gender_age.estimate, face_crop
                    )
                    if (
                        gender_conf >= settings.gender_confidence
                        and age_conf >= settings.age_confidence
                        and self.counter_store is not None
                    ):
                        age_group = AgeGroup.from_age(age)
                        self.counter_store.add_demographic(gender, age_group)
                except Exception:
                    logger.debug("Gender/age estimation failed for track %s", track_id)

            # 6. Face embedding for unique viewer counting
            if self.face_embedder is not None and self.face_store is not None:
                try:
                    embedding = await asyncio.to_thread(
                        self.face_embedder.embed, face_crop
                    )
                    self.face_store.match_or_add(embedding, camera_id=camera_id)
                    if self.counter_store is not None:
                        self.counter_store.add_unique_viewer(
                            self.face_store.get_unique_count()
                        )
                except Exception:
                    logger.debug("Face embedding failed for track %s", track_id)

            # 7. Emotion classification (optional, only when looking)
            if (
                self.emotion_classifier is not None
                and gaze_status in (GazeStatus.DIRECT, GazeStatus.PARTIAL)
            ):
                try:
                    await asyncio.to_thread(
                        self.emotion_classifier.classify, face_crop
                    )
                except Exception:
                    logger.debug("Emotion classification failed for track %s", track_id)

    # ------------------------------------------------------------------
    # Gaze classification helper
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_gaze(yaw: float, pitch: float) -> GazeStatus:
        """Map head-pose angles to a GazeStatus category."""
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        if abs_yaw <= settings.direct_yaw_max and abs_pitch <= settings.direct_pitch_max:
            return GazeStatus.DIRECT
        if abs_yaw <= settings.partial_yaw_max:
            return GazeStatus.PARTIAL
        return GazeStatus.GLANCE

    # ------------------------------------------------------------------
    # Background task: persist counters every 5 minutes
    # ------------------------------------------------------------------

    async def _persist_loop(self) -> None:
        """Persist CounterStore state to SQLite every 5 minutes."""
        interval = 300  # 5 minutes
        while self._running:
            await asyncio.sleep(interval)
            if self.counter_persistence is not None and self.counter_store is not None:
                try:
                    self.counter_persistence.save_from(self.counter_store)
                    logger.info("Persisted counters to SQLite")
                except Exception:
                    logger.exception("Error persisting counters")

    # ------------------------------------------------------------------
    # Background task: cleanup expired face store entries every 5 min
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Purge expired entries from the face store every 5 minutes."""
        interval = 300  # 5 minutes
        while self._running:
            await asyncio.sleep(interval)
            if self.face_store is not None:
                try:
                    removed = self.face_store.cleanup_expired()
                    if removed > 0:
                        logger.info("Cleaned up %d expired face entries", removed)
                except Exception:
                    logger.exception("Error during face store cleanup")

    # ------------------------------------------------------------------
    # Background task: auto-reset daily counters at midnight
    # ------------------------------------------------------------------

    async def _day_reset_loop(self) -> None:
        """Check once per minute if the date has changed; if so, reset
        daily counters (when ``counter_reset_on_day_change`` is enabled)."""
        while self._running:
            await asyncio.sleep(60)

            if not settings.counter_reset_on_day_change:
                continue

            today = date.today()
            if today != self._current_date:
                logger.info(
                    "Day changed from %s to %s — resetting daily counters",
                    self._current_date.isoformat(),
                    today.isoformat(),
                )
                self._current_date = today

                # Persist before reset so we don't lose yesterday's data
                if self.counter_persistence is not None and self.counter_store is not None:
                    try:
                        self.counter_persistence.save_from(self.counter_store)
                    except Exception:
                        logger.exception("Error persisting counters before day reset")

                if self.counter_store is not None:
                    try:
                        self.counter_store.reset()
                        logger.info("Daily counters reset")
                    except Exception:
                        logger.exception("Error resetting daily counters")

    # ------------------------------------------------------------------
    # Uvicorn management
    # ------------------------------------------------------------------

    def _start_uvicorn_background(self) -> None:
        """Launch the FastAPI edge server in a daemon thread."""
        config = uvicorn.Config(
            self.app,
            host=settings.server_host,
            port=settings.server_port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)

        thread = threading.Thread(
            target=self._uvicorn_server.run,
            name="uvicorn-edge-api",
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully stop all components."""
        if not self._running:
            return

        logger.info("Shutting down CVAnalyticsPipeline ...")
        self._running = False

        # Stop cameras
        if self.camera_manager is not None:
            try:
                self.camera_manager.stop_all()
            except Exception:
                logger.exception("Error stopping cameras")

        # Persist final counter state
        if self.counter_persistence is not None and self.counter_store is not None:
            try:
                self.counter_persistence.save_from(self.counter_store)
                logger.info("Final counter state persisted to SQLite")
            except Exception:
                logger.exception("Error persisting final counters")

        # Stop uvicorn
        if self._uvicorn_server is not None:
            try:
                self._uvicorn_server.should_exit = True
            except Exception:
                logger.exception("Error stopping uvicorn")

        logger.info("Pipeline shutdown complete")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_logging() -> None:
        """Set up structured logging for the application."""
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stderr,
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    pipeline = CVAnalyticsPipeline()
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        pass
