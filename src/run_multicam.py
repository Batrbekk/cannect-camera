"""
CANNECT.AI -- Multi-camera runner with single preview window.

Runs CV pipeline on ALL available cameras simultaneously:
- All cameras: person detection + tracking + head pose + attention → CounterStore
- Only camera_1 (first working): OpenCV preview window with overlays
- FastAPI edge server running on :8080 for backend pull

Usage:
    python -m src.run_multicam [--preview-cam 0] [--width 640] [--height 480]
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Imports with graceful degradation
# ---------------------------------------------------------------------------

_modules: dict[str, bool] = {}

try:
    from src.detection.person_detector import PersonDetector
    _modules["person_detector"] = True
except Exception:
    PersonDetector = None  # type: ignore
    _modules["person_detector"] = False

try:
    from src.detection.face_detector import FaceDetector
    _modules["face_detector"] = True
except Exception:
    FaceDetector = None  # type: ignore
    _modules["face_detector"] = False

try:
    from src.tracking.bytetrack import ByteTracker
    _modules["tracker"] = True
except Exception:
    ByteTracker = None  # type: ignore
    _modules["tracker"] = False

try:
    from src.analysis.head_pose import HeadPoseEstimator, classify_gaze
    _modules["head_pose"] = True
except Exception:
    HeadPoseEstimator = None  # type: ignore
    _modules["head_pose"] = False

try:
    from src.analysis.attention import AttentionTracker
    _modules["attention"] = True
except Exception:
    AttentionTracker = None  # type: ignore
    _modules["attention"] = False

try:
    from src.aggregation.models import GazeStatus, Direction, Gender, AgeGroup
    from src.aggregation.counters import CounterStore
    from src.aggregation.ad_tracker import AdTracker
    _modules["aggregation"] = True
except Exception:
    CounterStore = None  # type: ignore
    AdTracker = None  # type: ignore
    GazeStatus = None  # type: ignore
    Gender = None  # type: ignore
    AgeGroup = None  # type: ignore
    _modules["aggregation"] = False

try:
    from src.detection.face_detector import FaceDetector
    _modules["face_detector"] = True
except Exception:
    FaceDetector = None  # type: ignore
    _modules["face_detector"] = False

try:
    from src.analysis.gender_age import GenderAgeEstimator
    _modules["gender_age"] = True
except Exception:
    GenderAgeEstimator = None  # type: ignore
    _modules["gender_age"] = False

try:
    from src.analysis.face_embedding import FaceEmbedder, FaceStore
    _modules["face_embed"] = True
except Exception:
    FaceEmbedder = None  # type: ignore
    FaceStore = None  # type: ignore
    _modules["face_embed"] = False

try:
    from src.server.api import create_app, setup_api
    _modules["server"] = True
except Exception:
    create_app = None  # type: ignore
    setup_api = None  # type: ignore
    _modules["server"] = False

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multicam")

# Colours
COL_GREEN = (0, 255, 0)
COL_CYAN = (255, 255, 0)
COL_RED = (0, 0, 255)
COL_YELLOW = (0, 255, 255)
COL_WHITE = (255, 255, 255)
COL_BLACK = (0, 0, 0)
COL_PANEL_BG = (30, 30, 30)
COL_MAGENTA = (255, 0, 255)

GAZE_COLOURS = {"direct": COL_GREEN, "partial": COL_YELLOW, "glance": COL_RED}

TEST_CAMPAIGNS = [
    {"campaignId": "camp_nike_summer_2026", "videoId": "vid_nike_001"},
    {"campaignId": "camp_coca_cola_q2", "videoId": "vid_coke_002"},
    {"campaignId": "camp_samsung_galaxy", "videoId": "vid_samsung_003"},
]


# ---------------------------------------------------------------------------
# Camera worker — runs in its own thread, no GUI
# ---------------------------------------------------------------------------

class CameraWorker:
    """Processes one camera: detect → track → head pose → attention → CounterStore."""

    def __init__(
        self,
        camera_id: str,
        cv_index: int,
        person_detector,
        head_pose_estimator,
        counter_store,
        face_detector=None,
        gender_age=None,
        face_store=None,
        width: int = 640,
        height: int = 480,
    ):
        self.camera_id = camera_id
        self.cv_index = cv_index
        self.person_detector = person_detector
        self.head_pose = head_pose_estimator
        self.counter_store = counter_store
        self.face_detector = face_detector
        self.gender_age = gender_age
        self.face_store = face_store  # shared across all cameras for cross-cam re-ID
        self.width = width
        self.height = height

        self.tracker = ByteTracker(frame_rate=10) if ByteTracker else None
        self.attention_tracker = AttentionTracker() if AttentionTracker else None

        self._confirmed_tracks: set[int] = set()
        self._pending_person_count: dict[int, object] = {}  # tid -> direction, waiting for face dedup
        self._counted_face_ids: set[int] = set()  # face IDs already counted as people
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        # Shared state for preview
        self._latest_frame: np.ndarray | None = None
        self._latest_display: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.is_alive: bool = False

    def start(self) -> bool:
        """Open camera and start processing thread. Returns False if can't open."""
        self._cap = cv2.VideoCapture(self.cv_index)
        if not self._cap.isOpened():
            log.warning("[%s] Cannot open camera index %d", self.camera_id, self.cv_index)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Test read
        ret, frame = self._cap.read()
        if not ret or frame is None:
            log.warning("[%s] Camera %d opened but can't read frame", self.camera_id, self.cv_index)
            self._cap.release()
            return False

        log.info("[%s] Camera index %d opened: %dx%d", self.camera_id, self.cv_index, frame.shape[1], frame.shape[0])
        self.is_alive = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.camera_id}")
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        self.is_alive = False

    def get_display_frame(self) -> np.ndarray | None:
        """Get the latest frame with overlays drawn (for preview camera only)."""
        with self._frame_lock:
            return self._latest_display.copy() if self._latest_display is not None else None

    def _loop(self):
        """Main processing loop (runs in thread)."""
        frame_interval = 1.0 / 10  # 10 fps processing
        fps_counter = 0
        fps_start = time.monotonic()

        while self._running:
            t0 = time.monotonic()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            display = frame.copy()
            self._process(frame, display)

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_display = display

            self.frame_count += 1
            fps_counter += 1
            if time.monotonic() - fps_start >= 1.0:
                self.fps = fps_counter / (time.monotonic() - fps_start)
                fps_counter = 0
                fps_start = time.monotonic()

            # Throttle to ~10fps processing
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process(self, frame: np.ndarray, display: np.ndarray):
        """Run CV pipeline on frame."""
        now = time.time()
        detections = []
        tracks = []

        # Person detection
        if self.person_detector:
            try:
                detections = self.person_detector.detect(frame)
            except Exception:
                pass

        # Tracking
        if self.tracker and detections:
            try:
                tracks = self.tracker.update(detections, frame.shape[:2])
            except Exception:
                pass

        active_ids: set[int] = set()

        for track in tracks:
            tid = track.track_id
            bbox = track.bbox
            active_ids.add(tid)

            direction = track.direction if hasattr(track, "direction") else None
            dir_str = direction.value if direction and hasattr(direction, "value") else None

            # Draw person box
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), COL_GREEN, 2)
            label = f"ID:{tid}"
            if dir_str:
                label += f" {dir_str}"
            cv2.putText(display, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GREEN, 1, cv2.LINE_AA)

            # Count new tracks — deduplicated via face embeddings
            if tid not in self._confirmed_tracks:
                self._confirmed_tracks.add(tid)
                # Don't add_person here — wait for face dedup below
                self._pending_person_count[tid] = direction

            # --- Face detection + analysis on person crop ---
            face_bbox_abs = None
            face_landmarks_abs = None
            yaw, pitch = 0.0, 0.0

            # Try SCRFD face detector first
            if self.face_detector:
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.05)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(frame.shape[1], x2 + pad_x)
                cy2 = min(frame.shape[0], y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    try:
                        face_dets = self.face_detector.detect(crop)
                        if face_dets:
                            best = max(face_dets, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                            fx1, fy1, fx2, fy2 = best.bbox
                            face_bbox_abs = (cx1+fx1, cy1+fy1, cx1+fx2, cy1+fy2)
                            if best.landmarks:
                                face_landmarks_abs = [(cx1+lx, cy1+ly) for lx, ly in best.landmarks]
                            # Draw face box
                            bx1, by1, bx2, by2 = [int(v) for v in face_bbox_abs]
                            cv2.rectangle(display, (bx1, by1), (bx2, by2), COL_CYAN, 2)
                    except Exception:
                        pass

            # Head pose — use face bbox if available, else upper 60%
            if self.head_pose:
                if face_bbox_abs:
                    hp_bbox = face_bbox_abs
                else:
                    head_h = int((y2 - y1) * 0.6)
                    hp_bbox = (x1, y1, x2, y1 + head_h)

                try:
                    result = self.head_pose.estimate(frame, hp_bbox)
                    if result is not None:
                        yaw, pitch, roll = result
                        gaze = classify_gaze(yaw, pitch) if GazeStatus else None
                        gaze_str = gaze.value if gaze else "glance"

                        # Draw head pose arrow
                        bx1, by1, bx2, by2 = [int(v) for v in hp_bbox]
                        cx_a = (bx1 + bx2) // 2
                        cy_a = (by1 + by2) // 2
                        length = max(25, (bx2 - bx1) // 3)
                        dx = int(length * np.sin(np.radians(yaw)))
                        dy = int(-length * np.sin(np.radians(pitch)))
                        colour = GAZE_COLOURS.get(gaze_str, COL_RED)
                        cv2.arrowedLine(display, (cx_a, cy_a), (cx_a + dx, cy_a + dy), colour, 2, tipLength=0.3)
                        cv2.putText(display, gaze_str, (bx1, by2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

                        # Update attention — only when face IS detected
                        if self.attention_tracker and gaze:
                            self.attention_tracker.update(tid, gaze, now)
                            dwell = self.attention_tracker.get_dwell_time(tid)
                            over5s = self.attention_tracker.is_attention_over_threshold(tid)

                            if dwell > 0:
                                col = COL_GREEN if over5s else (0, 165, 255)
                                cv2.putText(display, f"ATT:{dwell:.1f}s{'!' if over5s else ''}", (x1, y1 - 22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

                            if self.counter_store:
                                self.counter_store.add_attention(
                                    self.camera_id, tid, dwell, over5s,
                                    is_looking=(gaze_str in ("direct", "partial")),
                                )
                except Exception:
                    pass

            # --- Gender / Age (needs face detector + aligned face) ---
            if self.gender_age and self.face_detector and face_landmarks_abs:
                try:
                    aligned = self.face_detector.align_face(frame, face_landmarks_abs)
                    gender, g_conf, age, a_conf = self.gender_age.predict(aligned, yaw, pitch)
                    g_str = gender.value if hasattr(gender, "value") else str(gender)
                    # Draw demographics
                    label = f"{g_str[0].upper()}/{age}" if age else g_str[0].upper()
                    cv2.putText(display, label, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1)
                    # Feed into CounterStore
                    if self.counter_store and Gender and AgeGroup:
                        g_enum = Gender(g_str) if g_str in ("male", "female") else Gender.UNKNOWN
                        a_enum = AgeGroup.from_age(age) if age else AgeGroup.ADULT
                        self.counter_store.add_demographic(g_enum, a_enum)
                except Exception:
                    pass

            # --- Face embedding: dedup person count + cross-camera unique viewers ---
            if self.face_store and self.face_detector and face_landmarks_abs:
                try:
                    aligned = self.face_detector.align_face(frame, face_landmarks_abs)
                    if hasattr(self, '_face_embedder') and self._face_embedder:
                        emb = self._face_embedder.extract(aligned)
                        matched_id, sim = self.face_store.match(emb)
                        is_new_person = False
                        if matched_id is None:
                            face_id = self.face_store.add(emb, now)
                            is_new_person = True
                            matched_id = face_id

                        # Count person only if this face_id hasn't been counted yet
                        if tid in self._pending_person_count:
                            if matched_id not in self._counted_face_ids:
                                self._counted_face_ids.add(matched_id)
                                direction = self._pending_person_count.pop(tid, None)
                                if self.counter_store:
                                    self.counter_store.add_person(self.camera_id, direction)
                            else:
                                # Same person already counted from another camera
                                self._pending_person_count.pop(tid, None)
                except Exception:
                    pass
            elif tid in self._pending_person_count:
                # No face detection available — count without dedup (fallback)
                if not self.face_store:
                    direction = self._pending_person_count.pop(tid, None)
                    if self.counter_store:
                        self.counter_store.add_person(self.camera_id, direction)

        if not tracks:
            for det in detections:
                bx1, by1, bx2, by2 = [int(v) for v in det.bbox]
                cv2.rectangle(display, (bx1, by1), (bx2, by2), COL_GREEN, 1)

        # Cleanup attention
        if self.attention_tracker:
            try:
                self.attention_tracker.cleanup(active_ids)
            except Exception:
                pass

        # Flush old pending person counts (tracks seen >30 frames without face match)
        stale_pending = [tid for tid in self._pending_person_count if tid not in active_ids]
        for tid in stale_pending:
            direction = self._pending_person_count.pop(tid, None)
            if self.counter_store:
                self.counter_store.add_person(self.camera_id, direction)

        # Update unique viewer count from shared face store
        if self.face_store and self.counter_store:
            try:
                self.counter_store.add_unique_viewer(self.face_store.get_unique_count())
                self.face_store.cleanup()  # remove expired embeddings
            except Exception:
                pass

        # Info overlay
        info = f"{self.camera_id} | FPS:{self.fps:.0f} | P:{len(tracks)}"
        cv2.putText(display, info, (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Multi-camera runner
# ---------------------------------------------------------------------------

class MultiCamRunner:
    """Runs all cameras, shows preview for one."""

    def __init__(self, preview_cam_index: int = 0, width: int = 640, height: int = 480):
        self.preview_cam_index = preview_cam_index
        self.width = width
        self.height = height
        self.models_dir = Path(__file__).parent.parent / "models"

        # Shared modules
        self.person_detector = None
        self.head_pose = None
        self.face_detector = None
        self.gender_age = None
        self.face_embedder = None
        self.face_store = None  # shared across ALL cameras for cross-cam re-ID
        self.counter_store = CounterStore() if CounterStore else None
        self.ad_tracker = AdTracker(self.counter_store) if (AdTracker and self.counter_store) else None

        self._campaign_index = -1
        self.workers: list[CameraWorker] = []
        self.preview_worker: CameraWorker | None = None

        self._init_models()

    def _init_models(self):
        """Load shared ML models (one instance shared across all cameras)."""
        if PersonDetector:
            p = self.models_dir / "yolov8n.onnx"
            if p.exists():
                try:
                    self.person_detector = PersonDetector(str(p))
                    log.info("PersonDetector loaded")
                except Exception as e:
                    log.warning("PersonDetector failed: %s", e)

        if FaceDetector:
            p = self.models_dir / "scrfd_500m.onnx"
            if p.exists():
                try:
                    self.face_detector = FaceDetector(str(p))
                    log.info("FaceDetector (SCRFD) loaded")
                except Exception as e:
                    log.warning("FaceDetector failed: %s", e)

        if HeadPoseEstimator:
            try:
                self.head_pose = HeadPoseEstimator()
                log.info("HeadPoseEstimator loaded")
            except Exception as e:
                log.warning("HeadPoseEstimator failed: %s", e)

        if GenderAgeEstimator:
            p = self.models_dir / "genderage.onnx"
            if p.exists():
                try:
                    self.gender_age = GenderAgeEstimator(str(p))
                    log.info("GenderAgeEstimator loaded")
                except Exception as e:
                    log.warning("GenderAgeEstimator failed: %s", e)

        if FaceEmbedder and FaceStore:
            p = self.models_dir / "mobilefacenet.onnx"
            if p.exists():
                try:
                    self.face_embedder = FaceEmbedder(str(p))
                    self.face_store = FaceStore()  # ONE store for ALL cameras = cross-cam dedup
                    log.info("FaceEmbedder + FaceStore loaded (cross-camera re-ID)")
                except Exception as e:
                    log.warning("FaceEmbedder failed: %s", e)

    def _scan_cameras(self) -> list[tuple[int, int, int]]:
        """Scan for working camera indices. Returns [(cv_index, w, h), ...]."""
        working = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working.append((idx, frame.shape[1], frame.shape[0]))
                cap.release()
        return working

    def run(self):
        """Main entry: scan cameras, start workers, show preview."""
        # Start FastAPI
        if self.counter_store and self.ad_tracker and setup_api and create_app:
            try:
                import uvicorn
                setup_api(self.counter_store, self.ad_tracker)
                app = create_app()
                cfg = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
                server = uvicorn.Server(cfg)
                threading.Thread(target=server.run, daemon=True, name="fastapi").start()
                log.info("FastAPI edge server on http://0.0.0.0:8080")
            except Exception as e:
                log.warning("FastAPI start failed: %s", e)

        # Scan cameras
        log.info("Scanning cameras...")
        cams = self._scan_cameras()
        log.info("Found %d working cameras: %s", len(cams), [c[0] for c in cams])

        if not cams:
            log.error("No cameras found!")
            sys.exit(1)

        # Create workers: camera_1 through camera_N
        for i, (cv_idx, w, h) in enumerate(cams[:5]):  # max 5 cameras
            cam_id = f"camera_{i + 1}"
            worker = CameraWorker(
                camera_id=cam_id,
                cv_index=cv_idx,
                person_detector=self.person_detector,
                head_pose_estimator=self.head_pose,
                counter_store=self.counter_store,
                face_detector=self.face_detector,
                gender_age=self.gender_age,
                face_store=self.face_store,
                width=self.width,
                height=self.height,
            )
            # Give worker access to face embedder
            worker._face_embedder = self.face_embedder
            if worker.start():
                self.workers.append(worker)
                # First successful worker = preview camera
                if self.preview_worker is None and i == self.preview_cam_index:
                    self.preview_worker = worker

        if not self.preview_worker and self.workers:
            self.preview_worker = self.workers[0]

        log.info("Started %d camera workers. Preview: %s",
                 len(self.workers),
                 self.preview_worker.camera_id if self.preview_worker else "NONE")

        if not self.preview_worker:
            log.error("No preview camera available")
            sys.exit(1)

        # OpenCV window for preview camera only
        window_name = "CANNECT.AI - Multi-Camera (Preview)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width + 320, self.height)

        # Start background push to backend
        self._last_push = time.monotonic()
        self._push_interval = 60  # seconds
        self._backend_url = settings.api_url  # http://192.168.0.106:3000/api/analytics/events
        threading.Thread(target=self._push_loop, daemon=True, name="push-backend").start()
        log.info("Backend push enabled -> %s (every %ds)", self._backend_url, self._push_interval)

        log.info("Running. Keys: q=quit, r=reset, c=campaign")

        try:
            while True:
                display = self.preview_worker.get_display_frame()
                if display is None:
                    time.sleep(0.03)
                    continue

                # Draw side panel
                panel = self._draw_panel(display.shape[0])
                combined = np.hstack([display, panel])

                cv2.imshow(window_name, combined)

                key = cv2.waitKey(33) & 0xFF  # ~30fps display
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    if self.counter_store:
                        self.counter_store.reset()
                    log.info("Metrics reset")
                elif key == ord("c"):
                    self._cycle_campaign()
        except KeyboardInterrupt:
            pass
        finally:
            for w in self.workers:
                w.stop()
            cv2.destroyAllWindows()
            log.info("Stopped")

    def _push_loop(self):
        """Push metrics to backend every 60 seconds."""
        import httpx
        while True:
            time.sleep(self._push_interval)
            if not self.counter_store:
                continue
            try:
                gm = self.counter_store.get_global_metrics()
                traffic = gm.get("traffic", {})
                attention = gm.get("attention", {})
                demographics = gm.get("demographics", {})
                campaign_id = None
                if self.ad_tracker:
                    ad = self.ad_tracker.get_current_ad()
                    campaign_id = ad.campaignId

                payload = {
                    "stationId": settings.station_id,
                    "cameraId": "camera_1",
                    "campaignId": campaign_id,
                    "events": [
                        {
                            "type": "traffic",
                            "data": {"people": {"total": traffic.get("people_total", 0)}},
                        },
                        {
                            "type": "attention",
                            "data": {
                                "peopleAttention": {
                                    "totalLooking": attention.get("total_looking", 0),
                                    "attentionOver5s": attention.get("attention_over_5s", 0),
                                    "averageDwellTime": round(attention.get("avg_dwell_time", 0.0), 1),
                                    "uniqueViewers": attention.get("unique_viewers", 0),
                                }
                            },
                        },
                        {
                            "type": "demographic",
                            "data": {
                                "gender": demographics.get("gender", {}),
                                "ageGroups": demographics.get("age_groups", {}),
                            },
                        },
                    ],
                }

                r = httpx.post(self._backend_url, json=payload, timeout=10)
                log.info("Push -> %s: %d (people=%d looking=%d)",
                         self._backend_url, r.status_code,
                         traffic.get("people_total", 0),
                         attention.get("total_looking", 0))
            except Exception as e:
                log.warning("Push failed: %s", e)

    def _cycle_campaign(self):
        if not self.ad_tracker:
            return
        self._campaign_index += 1
        if self._campaign_index >= len(TEST_CAMPAIGNS):
            self._campaign_index = -1
            self.ad_tracker.on_playback_event("playback_ended", None, None)
            log.info("Campaign -> IDLE")
        else:
            c = TEST_CAMPAIGNS[self._campaign_index]
            self.ad_tracker.on_playback_event(
                "playback_started", c["campaignId"], c["videoId"],
                datetime.now(tz=timezone.utc).isoformat(), 30,
            )
            log.info("Campaign -> %s", c["campaignId"])

    def _draw_panel(self, height: int, panel_w: int = 320) -> np.ndarray:
        """Draw metrics + camera status panel."""
        panel = np.full((height, panel_w, 3), COL_PANEL_BG, dtype=np.uint8)
        y = 28
        lh = 22

        def put(text, val="", colour=COL_WHITE, header=False):
            nonlocal y
            fs = 0.55 if header else 0.42
            th = 2 if header else 1
            c = COL_CYAN if header else colour
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
            if val:
                (vw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.putText(panel, val, (panel_w - 10 - vw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
            y += lh

        put("CANNECT.AI Multi-Cam", header=True)
        put("  PULL-BASED MODE", colour=COL_MAGENTA)
        y += 2

        # Campaign
        if self.ad_tracker:
            ad = self.ad_tracker.get_current_ad()
            put("CAMPAIGN", header=True)
            put("  ID:", (ad.campaignId or "(idle)")[:22], colour=COL_MAGENTA)
        y += 2

        # Global metrics from CounterStore
        if self.counter_store:
            try:
                gm = self.counter_store.get_global_metrics()
                t = gm.get("traffic", {})
                a = gm.get("attention", {})
                d = gm.get("demographics", {})

                put("TRAFFIC", header=True)
                put("  People:", str(t.get("people_total", 0)))
                put("  ->Screen:", str(t.get("toScreen", 0)))
                put("  <-Screen:", str(t.get("fromScreen", 0)))
                y += 2

                put("ATTENTION", header=True)
                put("  Looking:", str(a.get("total_looking", 0)))
                put("  >5s:", str(a.get("attention_over_5s", 0)), colour=COL_GREEN)
                put("  Avg dwell:", f"{a.get('avg_dwell_time', 0.0):.1f}s")
                put("  Unique:", str(a.get("unique_viewers", 0)))
                y += 2

                put("DEMOGRAPHICS", header=True)
                g = d.get("gender", {})
                put("  M/F/U:", f"{g.get('male',0)}/{g.get('female',0)}/{g.get('unknown',0)}")
                ag = d.get("age_groups", {})
                put("  Age:", f"C:{ag.get('child',0)} T:{ag.get('teen',0)} Y:{ag.get('young',0)} A:{ag.get('adult',0)} S:{ag.get('senior',0)}")
                y += 2
            except Exception:
                pass

        # Camera status
        put("CAMERAS", header=True)
        for w in self.workers:
            status = "LIVE" if w.is_alive else "OFF"
            preview = " [PREVIEW]" if w is self.preview_worker else ""
            colour = COL_GREEN if w.is_alive else COL_RED
            put(f"  {w.camera_id}:", f"{status} FPS:{w.fps:.0f} F:{w.frame_count}{preview}", colour=colour)

        # Bottom
        y = height - 35
        cv2.line(panel, (8, y - 3), (panel_w - 8, y - 3), (80, 80, 80), 1)
        cv2.putText(panel, "Edge: http://0.0.0.0:8080", (8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)
        cv2.putText(panel, "q=quit r=reset c=campaign", (8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

        return panel


def main():
    parser = argparse.ArgumentParser(description="CANNECT.AI Multi-Camera Runner")
    parser.add_argument("--preview-cam", type=int, default=0, help="Which camera index (0-based in detected list) to show preview for (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    runner = MultiCamRunner(preview_cam_index=args.preview_cam, width=args.width, height=args.height)
    runner.run()


if __name__ == "__main__":
    main()
