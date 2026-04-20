"""Head-pose estimation via MediaPipe Face Landmarker (tasks API) + solvePnP.

Estimates yaw, pitch, and roll from 478 facial landmarks projected onto a
canonical 3-D head model using OpenCV's Perspective-n-Point solver.

Compatible with MediaPipe >= 0.10.14 (new ``tasks`` API).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.aggregation.models import GazeStatus
from src.config.settings import settings

logger = logging.getLogger(__name__)

# ======================================================================
# Reference 3-D model points (canonical head, in mm)
# ======================================================================
_LANDMARK_INDICES: list[int] = [1, 33, 263, 61, 291, 199]

_MODEL_POINTS_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],         # Nose tip
        [-43.3, 32.7, -26.0],    # Right eye outer corner
        [43.3, 32.7, -26.0],     # Left eye outer corner
        [-28.9, -28.9, -24.1],   # Right mouth corner
        [28.9, -28.9, -24.1],    # Left mouth corner
        [0.0, -63.6, -12.5],     # Chin centre
    ],
    dtype=np.float64,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_model(models_dir: str | Path) -> Path:
    """Download face_landmarker.task if it doesn't exist yet."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "face_landmarker.task"
    if not model_path.exists():
        logger.info("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
        size_mb = model_path.stat().st_size / 1024 / 1024
        logger.info("Downloaded face_landmarker.task (%.1f MB)", size_mb)
    return model_path


# ======================================================================
# HeadPoseEstimator
# ======================================================================

class HeadPoseEstimator:
    """Estimate head orientation (yaw, pitch, roll) using MediaPipe Face Landmarker.

    Uses the new ``mediapipe.tasks.vision.FaceLandmarker`` API (>= 0.10.14).
    Automatically downloads the face_landmarker.task model on first use.
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        import mediapipe as mp

        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"

        model_path = _ensure_model(models_dir)

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        logger.info("HeadPoseEstimator initialised (MediaPipe Face Landmarker tasks API).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int],
    ) -> tuple[float, float, float] | None:
        """Estimate head pose for the face inside *face_bbox*.

        Parameters
        ----------
        frame:
            Full BGR frame (uint8).
        face_bbox:
            ``(x1, y1, x2, y2)`` bounding box of the detected face.

        Returns
        -------
        tuple[float, float, float] | None
            ``(yaw, pitch, roll)`` in degrees.  Returns ``None`` if
            no face / landmarks detected in the crop.
        """
        x1, y1, x2, y2 = face_bbox
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_frame, int(x2))
        y2 = min(h_frame, int(y2))

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # MediaPipe expects RGB.
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_crop,
        )

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None  # no face detected in crop

        landmarks = result.face_landmarks[0]
        crop_h, crop_w = face_crop.shape[:2]

        # Build the 2-D image points from landmark positions.
        image_points = np.array(
            [
                [landmarks[idx].x * crop_w,
                 landmarks[idx].y * crop_h]
                for idx in _LANDMARK_INDICES
            ],
            dtype=np.float64,
        )

        # Approximate camera intrinsic matrix (pinhole model).
        focal_length = float(crop_w)
        centre = (crop_w / 2.0, crop_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, centre[0]],
                [0.0, focal_length, centre[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector -> rotation matrix -> Euler angles.
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.vstack((pose_mat, [0, 0, 0, 1]))[:3]
        )

        pitch: float = float(euler_angles[0, 0])
        yaw: float = float(euler_angles[1, 0])
        roll: float = float(euler_angles[2, 0])

        # Clamp to [-180, 180].
        yaw = ((yaw + 180) % 360) - 180
        pitch = ((pitch + 180) % 360) - 180
        roll = ((roll + 180) % 360) - 180

        return yaw, pitch, roll


# ======================================================================
# Gaze classification helper
# ======================================================================

def classify_gaze(yaw: float, pitch: float) -> GazeStatus:
    """Classify gaze into direct / partial / glance from head angles.

    * **direct** -- |yaw| <= 15 deg **and** |pitch| <= 10 deg
    * **partial** -- |yaw| <= 30 deg (but not direct)
    * **glance** -- everything else
    """
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw <= settings.direct_yaw_max and abs_pitch <= settings.direct_pitch_max:
        return GazeStatus.DIRECT

    if abs_yaw <= settings.partial_yaw_max:
        return GazeStatus.PARTIAL

    return GazeStatus.GLANCE
