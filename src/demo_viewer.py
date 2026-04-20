"""
CANNECT.AI -- Demo Viewer with OpenCV preview window (pull-based architecture).

Runs the full CV pipeline on a local webcam with real-time visualisation:
- Person detection bounding boxes (green)
- Face detection bounding boxes (cyan)
- Head pose arrows (red/yellow/green based on gaze status)
- Attention timer overlay per person
- Real-time metrics panel on the right (5-camera layout, per-campaign)
- Embedded FastAPI edge server (pull-based: backend curls /metrics)
- Campaign cycling via 'c' key for testing

Usage:
    python -m src.demo_viewer [--camera 1] [--width 1280] [--height 720]
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
# Project imports -- graceful degradation if models not downloaded yet
# ---------------------------------------------------------------------------

_HAS_PERSON_DETECTOR = False
_HAS_FACE_DETECTOR = False
_HAS_HEAD_POSE = False
_HAS_GENDER_AGE = False
_HAS_FACE_EMBED = False
_HAS_EMOTION = False
_HAS_TRACKER = False

try:
    from src.detection.person_detector import PersonDetector
    _HAS_PERSON_DETECTOR = True
except Exception:
    PersonDetector = None  # type: ignore[assignment,misc]

try:
    from src.detection.face_detector import FaceDetector
    _HAS_FACE_DETECTOR = True
except Exception:
    FaceDetector = None  # type: ignore[assignment,misc]

try:
    from src.tracking.bytetrack import ByteTracker
    _HAS_TRACKER = True
except Exception:
    ByteTracker = None  # type: ignore[assignment,misc]

try:
    from src.analysis.head_pose import HeadPoseEstimator, classify_gaze
    _HAS_HEAD_POSE = True
except Exception:
    HeadPoseEstimator = None  # type: ignore[assignment,misc]

try:
    from src.analysis.gender_age import GenderAgeEstimator
    _HAS_GENDER_AGE = True
except Exception:
    GenderAgeEstimator = None  # type: ignore[assignment,misc]

try:
    from src.analysis.face_embedding import FaceEmbedder, FaceStore
    _HAS_FACE_EMBED = True
except Exception:
    FaceEmbedder = None  # type: ignore[assignment,misc]
    FaceStore = None  # type: ignore[assignment,misc]

try:
    from src.analysis.emotion import EmotionClassifier
    _HAS_EMOTION = True
except Exception:
    EmotionClassifier = None  # type: ignore[assignment,misc]

try:
    from src.analysis.attention import AttentionTracker
except Exception:
    AttentionTracker = None  # type: ignore[assignment,misc]

try:
    from src.aggregation.models import GazeStatus, Direction, Gender, AgeGroup
except Exception:
    GazeStatus = None  # type: ignore[assignment,misc]
    Direction = None  # type: ignore[assignment,misc]
    Gender = None  # type: ignore[assignment,misc]
    AgeGroup = None  # type: ignore[assignment,misc]

# Pull-based architecture imports
try:
    from src.aggregation.counters import CounterStore
except Exception:
    CounterStore = None  # type: ignore[assignment,misc]

try:
    from src.aggregation.ad_tracker import AdTracker
except Exception:
    AdTracker = None  # type: ignore[assignment,misc]

try:
    from src.server.api import create_app, setup_api
except Exception:
    create_app = None  # type: ignore[assignment,misc]
    setup_api = None  # type: ignore[assignment,misc]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("demo_viewer")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COL_GREEN = (0, 255, 0)
COL_CYAN = (255, 255, 0)
COL_RED = (0, 0, 255)
COL_YELLOW = (0, 255, 255)
COL_WHITE = (255, 255, 255)
COL_BLACK = (0, 0, 0)
COL_DARK = (40, 40, 40)
COL_ORANGE = (0, 165, 255)
COL_PANEL_BG = (30, 30, 30)
COL_MAGENTA = (255, 0, 255)

GAZE_COLOURS = {
    "direct": COL_GREEN,
    "partial": COL_YELLOW,
    "glance": COL_RED,
}

# Test campaigns for 'c' key cycling
TEST_CAMPAIGNS = [
    {"campaignId": "camp_nike_summer_2026", "videoId": "vid_nike_001"},
    {"campaignId": "camp_coca_cola_q2", "videoId": "vid_coke_002"},
    {"campaignId": "camp_samsung_galaxy", "videoId": "vid_samsung_003"},
]

# 5 camera IDs matching the production architecture
CAMERA_IDS = ["camera_1", "camera_2", "camera_3", "camera_4", "camera_5"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_person_box(frame: np.ndarray, bbox: tuple, track_id: int | None = None,
                    direction: str | None = None) -> None:
    """Draw person bounding box with optional track ID and direction."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_GREEN, 2)
    label_parts: list[str] = []
    if track_id is not None:
        label_parts.append(f"ID:{track_id}")
    if direction:
        label_parts.append(direction)
    if label_parts:
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COL_GREEN, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_BLACK, 1, cv2.LINE_AA)


def draw_face_box(frame: np.ndarray, bbox: tuple, landmarks: list | None = None) -> None:
    """Draw face bounding box and landmarks."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_CYAN, 2)
    if landmarks:
        for lx, ly in landmarks:
            cv2.circle(frame, (int(lx), int(ly)), 2, COL_CYAN, -1)


def draw_head_pose(frame: np.ndarray, bbox: tuple, yaw: float, pitch: float,
                   gaze_status: str) -> None:
    """Draw head pose arrow from face center."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(30, (x2 - x1) // 2)

    # Arrow direction based on yaw/pitch
    dx = int(length * np.sin(np.radians(yaw)))
    dy = int(-length * np.sin(np.radians(pitch)))
    colour = GAZE_COLOURS.get(gaze_status, COL_RED)
    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), colour, 2, tipLength=0.3)

    # Gaze label
    cv2.putText(frame, gaze_status, (x1, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_attention_timer(frame: np.ndarray, bbox: tuple, dwell_time: float,
                         is_over_threshold: bool) -> None:
    """Draw attention timer above the person box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    colour = COL_GREEN if is_over_threshold else COL_ORANGE
    label = f"ATT: {dwell_time:.1f}s"
    if is_over_threshold:
        label += " !"
    cv2.putText(frame, label, (x1, y1 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def draw_demographics(frame: np.ndarray, bbox: tuple, gender: str | None,
                      age: int | None) -> None:
    """Draw gender/age label below face box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    parts: list[str] = []
    if gender:
        parts.append(gender[0].upper())  # M / F / U
    if age is not None:
        parts.append(str(age))
    if parts:
        label = "/".join(parts)
        cv2.putText(frame, label, (x1, y2 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)


def draw_metrics_panel(
    frame: np.ndarray,
    metrics: dict,
    campaign_info: dict,
    camera_metrics: dict[str, dict],
    panel_w: int = 340,
) -> np.ndarray:
    """Draw a dark metrics panel on the right side and return the combined frame.

    Updated for pull-based architecture with per-campaign attribution and 5 cameras.
    """
    h, w = frame.shape[:2]
    panel = np.full((h, panel_w, 3), COL_PANEL_BG, dtype=np.uint8)

    y = 30
    line_h = 24

    def put(text: str, val: str = "", colour=COL_WHITE, header: bool = False):
        nonlocal y
        font_scale = 0.6 if header else 0.45
        thickness = 2 if header else 1
        col = COL_CYAN if header else colour
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, col, thickness, cv2.LINE_AA)
        if val:
            # Right-align value
            (vw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(panel, val, (panel_w - 12 - vw, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        y += line_h

    put("CANNECT.AI Analytics", header=True)
    put("  [PULL-BASED MODE]", colour=COL_MAGENTA)
    y += 3
    cv2.line(panel, (8, y - 12), (panel_w - 8, y - 12), (80, 80, 80), 1)

    # FPS
    put("FPS:", f"{metrics.get('fps', 0):.1f}")
    y += 3

    # Campaign
    put("CAMPAIGN", header=True)
    cid = campaign_info.get("campaignId") or "(idle)"
    vid = campaign_info.get("videoId") or "-"
    put("  ID:", cid[:24], colour=COL_MAGENTA)
    put("  Video:", vid[:20])
    y += 3

    # Traffic
    put("TRAFFIC", header=True)
    put("  People total:", str(metrics.get("people_total", 0)))
    put("  -> Screen:", str(metrics.get("to_screen", 0)))
    put("  <- Screen:", str(metrics.get("from_screen", 0)))
    put("  Left:", str(metrics.get("dir_left", 0)))
    put("  Right:", str(metrics.get("dir_right", 0)))
    y += 3

    # Attention
    put("ATTENTION", header=True)
    put("  Looking now:", str(metrics.get("looking_now", 0)))
    put("  Attention >5s:", str(metrics.get("attention_over_5s", 0)), colour=COL_GREEN)
    put("  Avg dwell:", f"{metrics.get('avg_dwell', 0.0):.1f}s")
    put("  Score:", f"{metrics.get('attention_score', 0)}/100")
    y += 3

    # Demographics
    put("DEMOGRAPHICS", header=True)
    put("  Male:", str(metrics.get("male", 0)))
    put("  Female:", str(metrics.get("female", 0)))
    put("  Unknown:", str(metrics.get("gender_unknown", 0)))
    y += 3

    # Cameras (5-camera layout)
    put("CAMERAS (5)", header=True)
    for cam_id in CAMERA_IDS:
        cam = camera_metrics.get(cam_id, {})
        people = cam.get("people", 0)
        looking = cam.get("looking", 0)
        att5 = cam.get("attention_over_5s", 0)
        status = "LIVE" if cam_id == "camera_1" else "---"
        put(f"  {cam_id}:", f"{status}  P:{people} L:{looking} A5:{att5}")
    y += 3

    # Unique viewers
    put("VIEWERS", header=True)
    put("  Unique:", str(metrics.get("unique_viewers", 0)))
    put("  Repeat:", str(metrics.get("repeat_viewers", 0)))

    # Server status at bottom
    y = h - 40
    cv2.line(panel, (8, y - 5), (panel_w - 8, y - 5), (80, 80, 80), 1)
    cv2.putText(panel, "Edge API: http://0.0.0.0:8080", (10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GREEN, 1, cv2.LINE_AA)
    cv2.putText(panel, "Keys: q=quit  r=reset  c=campaign", (10, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    # Combine
    return np.hstack([frame, panel])


# ---------------------------------------------------------------------------
# FastAPI server thread
# ---------------------------------------------------------------------------

def _start_fastapi_server(counter_store, ad_tracker) -> None:
    """Start the FastAPI edge server in a daemon thread.

    This lets you curl http://localhost:8080/metrics while the demo runs.
    """
    if create_app is None or setup_api is None:
        log.warning("FastAPI server modules not available -- skipping edge API")
        return

    import uvicorn

    setup_api(counter_store=counter_store, ad_tracker=ad_tracker)
    app = create_app()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="fastapi-edge")
    thread.start()
    log.info("FastAPI edge server started on http://0.0.0.0:8080  (curl /metrics)")


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class DemoViewer:
    """OpenCV-based demo viewer with full pipeline visualisation.

    Updated for pull-based architecture:
    - Uses CounterStore for metric accumulation (per-campaign attribution)
    - Uses AdTracker for current-ad display
    - Starts the FastAPI edge server alongside the viewer
    - Supports 5 cameras in the metrics panel
    - 'c' key cycles through test campaigns
    """

    def __init__(self, camera_index: int = 1, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.models_dir = Path(__file__).parent.parent / "models"

        # Camera ID for this demo instance (camera_1 on the webcam)
        self.camera_id = "camera_1"

        # --- Pull-based stores ---
        self.counter_store = CounterStore() if CounterStore is not None else None
        self.ad_tracker = None
        if AdTracker is not None and self.counter_store is not None:
            self.ad_tracker = AdTracker(counter_store=self.counter_store)

        # Campaign cycling state
        self._campaign_index: int = -1  # -1 = idle (no campaign)

        # Display-only metrics (populated from CounterStore each frame)
        self._display_metrics: dict = {
            "fps": 0.0,
            "people_total": 0,
            "to_screen": 0,
            "from_screen": 0,
            "dir_left": 0,
            "dir_right": 0,
            "looking_now": 0,
            "attention_over_5s": 0,
            "avg_dwell": 0.0,
            "attention_score": 0,
            "male": 0,
            "female": 0,
            "gender_unknown": 0,
            "child": 0,
            "teen": 0,
            "young": 0,
            "adult": 0,
            "senior": 0,
            "unique_viewers": 0,
            "repeat_viewers": 0,
        }

        # FPS calculation
        self._frame_times: list[float] = []
        self._confirmed_tracks: set[int] = set()

        # Init pipeline modules
        self._init_modules()

    def _init_modules(self) -> None:
        """Initialise available pipeline modules."""
        self.person_detector = None
        self.face_detector = None
        self.tracker = None
        self.head_pose = None
        self.gender_age = None
        self.face_embedder = None
        self.face_store = None
        self.emotion_cls = None
        self.attention_tracker = None

        # Person detector
        if _HAS_PERSON_DETECTOR:
            model_path = self.models_dir / "yolov8n.onnx"
            if model_path.exists():
                try:
                    self.person_detector = PersonDetector(str(model_path))
                    log.info("PersonDetector loaded")
                except Exception as e:
                    log.warning("PersonDetector failed: %s", e)

        # Face detector
        if _HAS_FACE_DETECTOR:
            model_path = self.models_dir / "scrfd_500m.onnx"
            if model_path.exists():
                try:
                    self.face_detector = FaceDetector(str(model_path))
                    log.info("FaceDetector loaded")
                except Exception as e:
                    log.warning("FaceDetector failed: %s", e)

        # Tracker
        if _HAS_TRACKER:
            try:
                self.tracker = ByteTracker(frame_rate=10)
                log.info("ByteTracker initialised")
            except Exception as e:
                log.warning("ByteTracker failed: %s", e)

        # Head pose
        if _HAS_HEAD_POSE:
            try:
                self.head_pose = HeadPoseEstimator()
                log.info("HeadPoseEstimator loaded")
            except Exception as e:
                log.warning("HeadPoseEstimator failed: %s", e)

        # Gender/Age
        if _HAS_GENDER_AGE:
            model_path = self.models_dir / "genderage.onnx"
            if model_path.exists():
                try:
                    self.gender_age = GenderAgeEstimator(str(model_path))
                    log.info("GenderAgeEstimator loaded")
                except Exception as e:
                    log.warning("GenderAgeEstimator failed: %s", e)

        # Face embedder
        if _HAS_FACE_EMBED and FaceEmbedder and FaceStore:
            model_path = self.models_dir / "mobilefacenet.onnx"
            if model_path.exists():
                try:
                    self.face_embedder = FaceEmbedder(str(model_path))
                    self.face_store = FaceStore()
                    log.info("FaceEmbedder + FaceStore loaded")
                except Exception as e:
                    log.warning("FaceEmbedder failed: %s", e)

        # Emotion
        if _HAS_EMOTION:
            model_path = self.models_dir / "hsemotion.onnx"
            if model_path.exists():
                try:
                    self.emotion_cls = EmotionClassifier(str(model_path))
                    log.info("EmotionClassifier loaded")
                except Exception as e:
                    log.warning("EmotionClassifier failed: %s", e)

        # Attention tracker (no model needed)
        if AttentionTracker is not None:
            try:
                self.attention_tracker = AttentionTracker()
                log.info("AttentionTracker initialised")
            except Exception as e:
                log.warning("AttentionTracker failed: %s", e)

        # Summary
        modules = [
            ("PersonDetector", self.person_detector),
            ("FaceDetector", self.face_detector),
            ("ByteTracker", self.tracker),
            ("HeadPose", self.head_pose),
            ("GenderAge", self.gender_age),
            ("FaceEmbed", self.face_embedder),
            ("Emotion", self.emotion_cls),
            ("Attention", self.attention_tracker),
        ]
        active = [name for name, mod in modules if mod is not None]
        missing = [name for name, mod in modules if mod is None]
        log.info("Active modules: %s", ", ".join(active) if active else "NONE")
        if missing:
            log.info("Unavailable (models not found): %s", ", ".join(missing))

    def run(self) -> None:
        """Main demo loop -- open webcam, process, display, serve metrics."""

        # --- Start the FastAPI edge server alongside the viewer ---
        if self.counter_store is not None and self.ad_tracker is not None:
            _start_fastapi_server(self.counter_store, self.ad_tracker)
        else:
            log.warning(
                "CounterStore/AdTracker not available -- edge API will not start"
            )

        log.info("Opening camera %d at %dx%d", self.camera_index, self.width, self.height)

        # Try to open camera -- auto-detect working index and backend
        cap = self._open_camera()
        if cap is None:
            log.error("Cannot open any camera")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Allow camera to warm up
        time.sleep(0.3)

        window_name = "CANNECT.AI - CV Analytics Demo (Pull-Based)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width + 340, self.height)

        log.info(
            "Demo started. Press 'q'/ESC to quit, 'r' to reset metrics, "
            "'c' to cycle campaign."
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame grab failed, retrying...")
                    time.sleep(0.1)
                    continue

                t0 = time.monotonic()

                # --- Process frame ---
                display = frame.copy()
                self._process_frame(frame, display)

                # --- FPS ---
                elapsed = time.monotonic() - t0
                self._frame_times.append(elapsed)
                if len(self._frame_times) > 30:
                    self._frame_times = self._frame_times[-30:]
                avg_time = sum(self._frame_times) / len(self._frame_times)
                self._display_metrics["fps"] = 1.0 / max(avg_time, 0.001)

                # --- Sync display metrics from CounterStore ---
                self._sync_display_metrics()

                # --- Gather campaign info for panel ---
                campaign_info = self._get_campaign_display()

                # --- Gather per-camera metrics for panel ---
                camera_metrics = self._get_camera_display()

                # --- Draw metrics panel ---
                combined = draw_metrics_panel(
                    display, self._display_metrics, campaign_info, camera_metrics
                )

                # --- Show ---
                cv2.imshow(window_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                elif key == ord("r"):  # reset metrics
                    self._reset_metrics()
                    log.info("Metrics reset")
                elif key == ord("c"):  # cycle campaign
                    self._cycle_campaign()

        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            log.info("Demo stopped")

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Try multiple camera indices and backends to find a working one.

        On Windows, index 1 typically works (index 0 often fails to read).
        """
        indices = [self.camera_index]
        # Also try nearby indices
        for i in range(5):
            if i != self.camera_index:
                indices.append(i)

        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    log.info("Camera %d opened OK (%dx%d)", idx, frame.shape[1], frame.shape[0])
                    return cap
                cap.release()

        return None

    # ------------------------------------------------------------------
    # Campaign cycling
    # ------------------------------------------------------------------

    def _cycle_campaign(self) -> None:
        """Cycle through test campaigns (or back to idle)."""
        if self.ad_tracker is None:
            log.warning("AdTracker not available -- cannot cycle campaigns")
            return

        self._campaign_index += 1
        if self._campaign_index >= len(TEST_CAMPAIGNS):
            # Back to idle
            self._campaign_index = -1
            self.ad_tracker.on_playback_event(
                event="playback_ended",
                campaign_id=None,
                video_id=None,
            )
            log.info("Campaign set to IDLE (no active campaign)")
        else:
            camp = TEST_CAMPAIGNS[self._campaign_index]
            self.ad_tracker.on_playback_event(
                event="playback_changed" if self._campaign_index > 0 else "playback_started",
                campaign_id=camp["campaignId"],
                video_id=camp["videoId"],
                started_at=datetime.now(tz=timezone.utc).isoformat(),
                expected_duration=30,
            )
            log.info("Campaign changed -> %s", camp["campaignId"])

    # ------------------------------------------------------------------
    # Display metric sync
    # ------------------------------------------------------------------

    def _sync_display_metrics(self) -> None:
        """Pull latest values from CounterStore into display dict."""
        if self.counter_store is None:
            return

        try:
            raw = self.counter_store.get_global_metrics()
            traffic = raw.get("traffic", {})
            attention = raw.get("attention", {})
            demographics = raw.get("demographics", {})

            self._display_metrics["people_total"] = traffic.get("people_total", 0)
            self._display_metrics["to_screen"] = traffic.get("toScreen", 0)
            self._display_metrics["from_screen"] = traffic.get("fromScreen", 0)
            self._display_metrics["dir_left"] = traffic.get("left", 0)
            self._display_metrics["dir_right"] = traffic.get("right", 0)

            self._display_metrics["attention_over_5s"] = attention.get("attention_over_5s", 0)
            self._display_metrics["avg_dwell"] = attention.get("avg_dwell_time", 0.0)
            self._display_metrics["unique_viewers"] = attention.get("unique_viewers", 0)

            gender = demographics.get("gender", {})
            self._display_metrics["male"] = gender.get("male", 0)
            self._display_metrics["female"] = gender.get("female", 0)
            self._display_metrics["gender_unknown"] = gender.get("unknown", 0)

            age_groups = demographics.get("age_groups", {})
            self._display_metrics["child"] = age_groups.get("child", 0)
            self._display_metrics["teen"] = age_groups.get("teen", 0)
            self._display_metrics["young"] = age_groups.get("young", 0)
            self._display_metrics["adult"] = age_groups.get("adult", 0)
            self._display_metrics["senior"] = age_groups.get("senior", 0)
        except Exception:
            pass

    def _get_campaign_display(self) -> dict:
        """Return current campaign info for the panel."""
        if self.ad_tracker is not None:
            try:
                info = self.ad_tracker.get_current_ad()
                return {
                    "campaignId": info.campaignId,
                    "videoId": info.videoId,
                }
            except Exception:
                pass
        return {"campaignId": None, "videoId": None}

    def _get_camera_display(self) -> dict[str, dict]:
        """Return per-camera metrics for the panel."""
        if self.counter_store is not None:
            try:
                return self.counter_store.get_camera_metrics()
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, display: np.ndarray) -> None:
        """Run pipeline on a single frame and draw overlays on display."""
        now = time.time()
        person_detections = []
        tracks = []

        # --- Person detection ---
        if self.person_detector is not None:
            try:
                person_detections = self.person_detector.detect(frame)
            except Exception as e:
                log.debug("Person detection error: %s", e)

        # --- Tracking ---
        if self.tracker is not None and person_detections:
            try:
                tracks = self.tracker.update(person_detections, frame.shape[:2])
            except Exception as e:
                log.debug("Tracker error: %s", e)

        # Draw person boxes and process each track
        looking_now = 0
        active_track_ids: set[int] = set()

        if tracks:
            for track in tracks:
                track_id = track.track_id
                bbox = track.bbox
                active_track_ids.add(track_id)

                direction_str = None
                direction_enum = None
                if hasattr(track, "direction") and track.direction is not None:
                    direction_str = track.direction.value if hasattr(track.direction, "value") else str(track.direction)
                    direction_enum = track.direction if isinstance(track.direction, Direction) else None

                # Count new confirmed tracks -- feed into CounterStore
                if track_id not in self._confirmed_tracks:
                    self._confirmed_tracks.add(track_id)
                    if self.counter_store is not None:
                        self.counter_store.add_person(self.camera_id, direction_enum)

                draw_person_box(display, bbox, track_id, direction_str)

                # --- Face analysis on person crop ---
                self._process_face(frame, display, bbox, track_id, now)

                # --- Attention timer overlay (reads from attention_tracker fed in _process_face) ---
                if self.attention_tracker is not None:
                    try:
                        dwell = self.attention_tracker.get_dwell_time(track_id)
                        over_thresh = self.attention_tracker.is_attention_over_threshold(track_id)
                        if dwell > 0:
                            draw_attention_timer(display, bbox, dwell, over_thresh)
                            looking_now += 1
                    except Exception:
                        pass
        elif person_detections:
            # No tracker -- just draw raw detections
            for det in person_detections:
                draw_person_box(display, det.bbox)

        # --- Update attention metrics for display ---
        self._display_metrics["looking_now"] = looking_now
        if self.attention_tracker is not None:
            try:
                stats = self.attention_tracker.get_stats()
                self._display_metrics["attention_over_5s"] = stats.get("attentionOver5s", 0)
                self._display_metrics["avg_dwell"] = stats.get("averageDwellTime", 0.0)
                self._display_metrics["attention_score"] = stats.get("attentionScore", 0)
                self.attention_tracker.cleanup(active_track_ids)
            except Exception:
                pass

        # --- Unique viewers ---
        if self.face_store is not None:
            try:
                count = self.face_store.get_unique_count()
                self._display_metrics["unique_viewers"] = count
                if self.counter_store is not None:
                    self.counter_store.add_unique_viewer(count)
            except Exception:
                pass

        # --- Info overlay on frame (updated for pull-based) ---
        info = f"PULL-BASED | Persons: {len(tracks or person_detections)} | "
        info += f"Cam: {self.camera_id} | "
        info += f"Modules: {sum(1 for m in [self.person_detector, self.face_detector, self.tracker, self.head_pose] if m is not None)}/8"
        cv2.putText(display, info, (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    def _process_face(self, frame: np.ndarray, display: np.ndarray,
                      person_bbox: tuple, track_id: int, now: float) -> None:
        """Run face analysis on a person crop.

        Two paths:
        1. If SCRFD face detector is available: detect face -> head pose -> gender/age
        2. Fallback: use HeadPoseEstimator directly on person upper-body crop
           (MediaPipe FaceLandmarker has built-in face detection)
        """
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame.shape[:2]

        # Expand crop slightly
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.05)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return

        # --- Face detection (if SCRFD available) ---
        abs_face_bbox = None
        abs_landmarks = None

        if self.face_detector is not None:
            try:
                face_dets = self.face_detector.detect(crop)
                if face_dets:
                    best_face = max(face_dets, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    fx1, fy1, fx2, fy2 = best_face.bbox
                    abs_face_bbox = (cx1 + fx1, cy1 + fy1, cx1 + fx2, cy1 + fy2)
                    if best_face.landmarks:
                        abs_landmarks = [(cx1 + lx, cy1 + ly) for lx, ly in best_face.landmarks]
                    draw_face_box(display, abs_face_bbox, abs_landmarks)
            except Exception:
                pass

        # --- Head pose (works even without SCRFD via MediaPipe fallback) ---
        yaw, pitch = 0.0, 0.0
        gaze_str = "glance"
        head_pose_ok = False

        if self.head_pose is not None:
            # Use face bbox if available, otherwise use upper 60% of person crop (head area)
            if abs_face_bbox is not None:
                hp_bbox = abs_face_bbox
            else:
                # Estimate head region: upper 60% of person bbox
                head_h = int((y2 - y1) * 0.6)
                hp_bbox = (x1, y1, x2, y1 + head_h)

            try:
                yaw, pitch, roll = self.head_pose.estimate(frame, hp_bbox)
                head_pose_ok = True

                # Draw face region if SCRFD didn't find one
                if abs_face_bbox is None:
                    abs_face_bbox = hp_bbox
                    bx1, by1, bx2, by2 = [int(v) for v in hp_bbox]
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), COL_CYAN, 1)

                if GazeStatus is not None:
                    gaze = classify_gaze(yaw, pitch)
                    gaze_str = gaze.value if hasattr(gaze, "value") else str(gaze)

                draw_head_pose(display, abs_face_bbox, yaw, pitch, gaze_str)

                # Update attention tracker
                if self.attention_tracker is not None and GazeStatus is not None:
                    self.attention_tracker.update(track_id, classify_gaze(yaw, pitch), now)

                    # Feed attention into CounterStore
                    dwell = self.attention_tracker.get_dwell_time(track_id)
                    over_5s = self.attention_tracker.is_attention_over_threshold(track_id)
                    if self.counter_store is not None:
                        self.counter_store.add_attention(
                            camera_id=self.camera_id,
                            track_id=track_id,
                            dwell_time=dwell,
                            is_over_5s=over_5s,
                            is_looking=(gaze_str in ("direct", "partial")),
                        )
            except Exception:
                pass

        # --- Gender / Age (requires face detector with landmarks for alignment) ---
        gender_str = None
        age_val = None
        if self.gender_age is not None and abs_landmarks is not None and self.face_detector is not None:
            try:
                aligned = self.face_detector.align_face(frame, abs_landmarks)
                gender, g_conf, age, a_conf = self.gender_age.predict(aligned, yaw, pitch)
                gender_str = gender.value if hasattr(gender, "value") else str(gender)
                age_val = age

                if self.counter_store is not None and Gender is not None and AgeGroup is not None:
                    try:
                        g_enum = Gender(gender_str) if gender_str in ("male", "female") else Gender.UNKNOWN
                        a_enum = AgeGroup.from_age(age) if age is not None else AgeGroup.ADULT
                        self.counter_store.add_demographic(g_enum, a_enum)
                    except Exception:
                        pass
            except Exception:
                pass
        draw_demographics(display, abs_face_bbox, gender_str, age_val)

        # --- Face embedding ---
        if self.face_embedder is not None and self.face_store is not None:
            try:
                if abs_landmarks or best_face.landmarks:
                    lm = abs_landmarks or best_face.landmarks
                    aligned = self.face_detector.align_face(frame, lm)
                    emb = self.face_embedder.extract(aligned)
                    matched_id, sim = self.face_store.match(emb)
                    if matched_id is None:
                        self.face_store.add(emb, now)
            except Exception:
                pass

    def _reset_metrics(self) -> None:
        """Reset all accumulated metrics."""
        # Reset CounterStore
        if self.counter_store is not None:
            self.counter_store.reset()

        # Reset display metrics
        for key in self._display_metrics:
            if key != "fps":
                self._display_metrics[key] = 0 if isinstance(self._display_metrics[key], int) else 0.0

        self._confirmed_tracks.clear()

        if self.attention_tracker is not None:
            try:
                self.attention_tracker.cleanup(set())
            except Exception:
                pass

        log.info("All metrics reset (CounterStore + display)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CANNECT.AI CV Analytics Demo Viewer (Pull-Based)")
    parser.add_argument("--camera", type=int, default=1,
                        help="Camera device index (default: 1, Windows webcam)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    args = parser.parse_args()

    viewer = DemoViewer(camera_index=args.camera, width=args.width, height=args.height)
    viewer.run()


if __name__ == "__main__":
    main()
