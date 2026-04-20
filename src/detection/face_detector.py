"""SCRFD face detector using ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.aggregation.models import FaceDetectionResult

logger = logging.getLogger(__name__)

# SCRFD standard input size
_INPUT_SIZE = 640

# Detection threshold
_CONFIDENCE_THRESHOLD = 0.5
_NMS_IOU_THRESHOLD = 0.4

# Feature-map strides produced by SCRFD backbone
_FEAT_STRIDES = (8, 16, 32)

# Number of anchors per location for each stride level
_NUM_ANCHORS = 2

# Alignment target landmarks (ArcFace 112x112 standard reference)
_ALIGNED_SIZE = 112
_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _available_providers() -> list[str]:
    """Return ONNX Runtime execution providers in priority order."""
    available = ort.get_available_providers()
    providers: list[str] = []

    if "HailoExecutionProvider" in available:
        providers.append("HailoExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def _distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Convert anchor centre + distance offsets to ``(x1, y1, x2, y2)``.

    Parameters
    ----------
    points:
        Anchor centres, shape ``(N, 2)``.
    distance:
        Regression deltas ``(left, top, right, bottom)``, shape ``(N, 4)``.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Convert anchor centre + distance offsets to landmark coordinates.

    Parameters
    ----------
    points:
        Anchor centres, shape ``(N, 2)``.
    distance:
        Landmark offsets, shape ``(N, 10)`` — 5 keypoints x 2.

    Returns
    -------
    np.ndarray
        Landmark coordinates, shape ``(N, 5, 2)``.
    """
    num_points = distance.shape[1] // 2
    result = np.empty((distance.shape[0], num_points, 2), dtype=distance.dtype)
    for i in range(num_points):
        result[:, i, 0] = points[:, 0] + distance[:, 2 * i]
        result[:, i, 1] = points[:, 1] + distance[:, 2 * i + 1]
    return result


class FaceDetector:
    """SCRFD face detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path:
        Filesystem path to the SCRFD ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_file}")

        providers = _available_providers()
        logger.info(
            "Loading face detection model from %s (providers=%s)",
            model_file,
            providers,
        )

        self._session = ort.InferenceSession(str(model_file), providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name

        # Determine whether the model produces landmark outputs.
        # SCRFD w/ keypoints has 9 outputs (3 strides x 3 heads: score, bbox, kps),
        # without keypoints it has 6 outputs.
        output_names = [o.name for o in self._session.get_outputs()]
        self._has_landmarks = len(output_names) == 9

        meta = self._session.get_inputs()[0]
        logger.info(
            "Model loaded — input %s shape=%s dtype=%s, landmarks=%s",
            meta.name,
            meta.shape,
            meta.type,
            self._has_landmarks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[FaceDetectionResult]:
        """Run face detection on a BGR frame.

        Parameters
        ----------
        frame:
            Input image in BGR (HWC, uint8).

        Returns
        -------
        list[FaceDetectionResult]
            Detected faces with bounding boxes, confidence, and optional
            5-point landmarks.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to FaceDetector.detect")
            return []

        orig_h, orig_w = frame.shape[:2]

        blob, scale, pad_w, pad_h = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        detections = self._postprocess(outputs, scale, pad_w, pad_h, orig_w, orig_h)

        logger.debug(
            "Face detection: %d faces on %dx%d frame",
            len(detections),
            orig_w,
            orig_h,
        )
        return detections

    @staticmethod
    def align_face(
        frame: np.ndarray,
        landmarks: list[tuple[float, float]],
    ) -> np.ndarray:
        """Align and crop a face to 112x112 using a similarity transform.

        This produces an aligned face image suitable for downstream
        embedding extraction (e.g. ArcFace).

        Parameters
        ----------
        frame:
            Full BGR image from which the face was detected.
        landmarks:
            Five facial keypoints ``[(x, y), ...]`` — left eye, right eye,
            nose tip, left mouth corner, right mouth corner.

        Returns
        -------
        np.ndarray
            Aligned face crop, 112x112, BGR, uint8.
        """
        src_pts = np.array(landmarks, dtype=np.float32)
        dst_pts = _REF_LANDMARKS.copy()

        # Estimate affine (partial) — uses the first 2 point pairs for
        # the rigid part and all 5 for least-squares refinement.
        transform_matrix = _estimate_similarity_transform(src_pts, dst_pts)

        aligned = cv2.warpAffine(
            frame,
            transform_matrix,
            (_ALIGNED_SIZE, _ALIGNED_SIZE),
            borderValue=(0, 0, 0),
        )
        return aligned

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, int, int]:
        """Letterbox-resize, normalise, transpose.

        Returns ``(blob, scale, pad_w, pad_h)`` where *scale* and *pad*
        values are needed to map detections back to the original frame.
        """
        orig_h, orig_w = frame.shape[:2]

        # Compute uniform scale to fit within _INPUT_SIZE
        scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding (bottom-right)
        pad_w = _INPUT_SIZE - new_w
        pad_h = _INPUT_SIZE - new_h
        padded = cv2.copyMakeBorder(
            resized,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # BGR -> RGB, normalise to [0, 1], HWC -> CHW, add batch dim
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        return blob, scale, pad_w, pad_h

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> list[FaceDetectionResult]:
        """Decode SCRFD outputs, apply NMS, and rescale to original coords.

        SCRFD outputs are grouped per stride level.  For a model *with*
        landmarks the layout is::

            stride 8  -> score_8, bbox_8, kps_8
            stride 16 -> score_16, bbox_16, kps_16
            stride 32 -> score_32, bbox_32, kps_32

        Without landmarks, kps outputs are absent.
        """
        all_scores: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_kps: list[np.ndarray | None] = []

        num_strides = len(_FEAT_STRIDES)

        # Detect output layout:
        # Layout A (interleaved):  score_8, bbox_8, kps_8, score_16, ...
        # Layout B (grouped):     score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, ...
        # Detect by checking if outputs are grouped by type (all scores first)
        # Heuristic: if output[0] and output[1] have different last dims → interleaved
        if len(outputs) >= 6 and outputs[0].shape[-1] == outputs[1].shape[-1]:
            # Layout B: grouped (scores together, boxes together, kps together)
            score_outputs = outputs[0:num_strides]
            bbox_outputs = outputs[num_strides:num_strides * 2]
            kps_outputs = outputs[num_strides * 2:num_strides * 3] if self._has_landmarks else [None] * num_strides
        else:
            # Layout A: interleaved
            outputs_per_stride = 3 if self._has_landmarks else 2
            score_outputs = [outputs[i * outputs_per_stride] for i in range(num_strides)]
            bbox_outputs = [outputs[i * outputs_per_stride + 1] for i in range(num_strides)]
            kps_outputs = [outputs[i * outputs_per_stride + 2] for i in range(num_strides)] if self._has_landmarks else [None] * num_strides

        for idx, stride in enumerate(_FEAT_STRIDES):
            score_out = score_outputs[idx]
            bbox_out = bbox_outputs[idx]
            kps_out = kps_outputs[idx]

            # Build anchor grid for this stride
            feat_h = _INPUT_SIZE // stride
            feat_w = _INPUT_SIZE // stride
            anchor_centres = _make_anchor_centres(feat_w, feat_h, stride, _NUM_ANCHORS)

            # Scores — flatten across anchors
            scores = score_out.reshape(-1)

            # Ensure anchor grid matches score count
            if len(scores) != len(anchor_centres):
                # Recalculate num_anchors from actual data
                actual_anchors = len(scores) // (feat_h * feat_w)
                if actual_anchors > 0:
                    anchor_centres = _make_anchor_centres(feat_w, feat_h, stride, actual_anchors)

            # Filter by confidence early for efficiency
            keep = scores >= _CONFIDENCE_THRESHOLD
            if not np.any(keep):
                continue

            scores = scores[keep]
            anchor_centres_k = anchor_centres[keep]

            # Decode boxes: offsets are in stride-pixel units
            bbox_deltas = (bbox_out.reshape(-1, 4) * stride)[keep]
            boxes = _distance2bbox(anchor_centres_k, bbox_deltas)

            all_scores.append(scores)
            all_boxes.append(boxes)

            if kps_out is not None:
                kps_deltas = (kps_out.reshape(-1, 10) * stride)[keep]
                kps = _distance2kps(anchor_centres_k, kps_deltas)
                all_kps.append(kps)
            else:
                all_kps.append(None)

        if not all_scores:
            return []

        scores_cat = np.concatenate(all_scores)
        boxes_cat = np.concatenate(all_boxes)

        has_kps = all(k is not None for k in all_kps) and len(all_kps) > 0
        kps_cat: np.ndarray | None = None
        if has_kps:
            kps_cat = np.concatenate([k for k in all_kps if k is not None])

        # NMS
        boxes_xywh = np.empty_like(boxes_cat)
        boxes_xywh[:, 0] = boxes_cat[:, 0]
        boxes_xywh[:, 1] = boxes_cat[:, 1]
        boxes_xywh[:, 2] = boxes_cat[:, 2] - boxes_cat[:, 0]
        boxes_xywh[:, 3] = boxes_cat[:, 3] - boxes_cat[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores_cat.tolist(),
            _CONFIDENCE_THRESHOLD,
            _NMS_IOU_THRESHOLD,
        )

        if len(indices) == 0:
            return []

        indices = np.asarray(indices).flatten()

        results: list[FaceDetectionResult] = []
        for i in indices:
            # Map from padded-model space back to original image
            x1, y1, x2, y2 = boxes_cat[i]
            x1 = np.clip(x1 / scale, 0, orig_w)
            y1 = np.clip(y1 / scale, 0, orig_h)
            x2 = np.clip(x2 / scale, 0, orig_w)
            y2 = np.clip(y2 / scale, 0, orig_h)

            landmarks: list[tuple[float, float]] | None = None
            if kps_cat is not None:
                raw_kps = kps_cat[i]  # (5, 2)
                landmarks = [
                    (
                        float(np.clip(raw_kps[j, 0] / scale, 0, orig_w)),
                        float(np.clip(raw_kps[j, 1] / scale, 0, orig_h)),
                    )
                    for j in range(raw_kps.shape[0])
                ]

            results.append(
                FaceDetectionResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(scores_cat[i]),
                    landmarks=landmarks,
                )
            )

        return results


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _make_anchor_centres(
    feat_w: int,
    feat_h: int,
    stride: int,
    num_anchors: int,
) -> np.ndarray:
    """Generate anchor centre coordinates for a single feature-map level.

    Returns
    -------
    np.ndarray
        Shape ``(feat_h * feat_w * num_anchors, 2)`` with ``(x, y)``
        coordinates in input-pixel space.
    """
    shift_x = np.arange(feat_w) * stride
    shift_y = np.arange(feat_h) * stride
    grid_x, grid_y = np.meshgrid(shift_x, shift_y)
    centres = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)

    # Repeat for each anchor at the same location
    centres = np.tile(centres, (1, num_anchors)).reshape(-1, 2)
    return centres


def _estimate_similarity_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Estimate a 2x3 similarity (rigid + uniform scale) transform.

    Uses a least-squares fit over all point correspondences to compute
    the transform matrix ``M`` such that ``dst ~ M @ [src; 1]``.

    Parameters
    ----------
    src:
        Source landmarks, shape ``(N, 2)``.
    dst:
        Destination landmarks, shape ``(N, 2)``.

    Returns
    -------
    np.ndarray
        2x3 affine matrix suitable for ``cv2.warpAffine``.
    """
    num = src.shape[0]

    # Build linear system  [x, -y, 1, 0] [a]   [u]
    #                      [y,  x, 0, 1] [b] = [v]
    #                                     [tx]
    #                                     [ty]
    A = np.zeros((2 * num, 4), dtype=np.float64)
    b = np.zeros((2 * num,), dtype=np.float64)

    for i in range(num):
        sx, sy = src[i]
        dx, dy = dst[i]
        A[2 * i] = [sx, -sy, 1, 0]
        A[2 * i + 1] = [sy, sx, 0, 1]
        b[2 * i] = dx
        b[2 * i + 1] = dy

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, neg_b, tx, ty = params

    M = np.array(
        [[a, -neg_b, tx], [neg_b, a, ty]],
        dtype=np.float64,
    )
    return M
