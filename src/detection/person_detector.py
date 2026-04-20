"""YOLOv8n person detector using ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.aggregation.models import DetectionResult
from src.config.settings import settings

logger = logging.getLogger(__name__)

# YOLOv8 input dimensions
_INPUT_SIZE = 640

# COCO class id for person
_PERSON_CLASS_ID = 0

# NMS / confidence thresholds
_CONFIDENCE_THRESHOLD = 0.25
_NMS_IOU_THRESHOLD = 0.45


def _available_providers() -> list[str]:
    """Return ONNX Runtime execution providers in priority order.

    Prefers the Hailo accelerator when available, falling back to CPU.
    """
    available = ort.get_available_providers()
    providers: list[str] = []

    # Hailo NPU provider (edge deployment)
    if "HailoExecutionProvider" in available:
        providers.append("HailoExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


class PersonDetector:
    """YOLOv8-nano person detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path:
        Filesystem path to the YOLOv8n ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_file}")

        providers = _available_providers()
        logger.info(
            "Loading person detection model from %s (providers=%s)",
            model_file,
            providers,
        )

        self._session = ort.InferenceSession(str(model_file), providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name

        meta = self._session.get_inputs()[0]
        logger.info(
            "Model loaded — input %s shape=%s dtype=%s",
            meta.name,
            meta.shape,
            meta.type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run person detection on a BGR frame.

        Parameters
        ----------
        frame:
            Input image in BGR (HWC, uint8) as returned by ``cv2.imread``.

        Returns
        -------
        list[DetectionResult]
            Filtered person detections with absolute pixel coordinates.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to PersonDetector.detect")
            return []

        orig_h, orig_w = frame.shape[:2]

        blob = self._preprocess(frame)
        raw_output = self._session.run(None, {self._input_name: blob})[0]
        detections = self._postprocess(raw_output, orig_w, orig_h)

        logger.debug(
            "Person detection: %d detections on %dx%d frame",
            len(detections),
            orig_w,
            orig_h,
        )
        return detections

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Resize, normalise, transpose to CHW and add batch dimension.

        Returns a ``float32`` tensor of shape ``(1, 3, 640, 640)``.
        """
        img = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, uint8 -> float32 [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension -> (1, 3, 640, 640)
        return np.expand_dims(img, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> list[DetectionResult]:
        """Decode YOLOv8 output, apply NMS and filter for persons.

        YOLOv8 output shape: ``(1, 84, 8400)`` where 84 = 4 bbox + 80 classes.
        Each of the 8400 anchors stores ``[cx, cy, w, h, class_scores...]``.
        """
        # Squeeze batch dim -> (84, 8400) then transpose -> (8400, 84)
        predictions = np.squeeze(raw_output, axis=0).T

        # Split bounding boxes and class scores
        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Filter by person class
        person_scores = class_scores[:, _PERSON_CLASS_ID]
        mask = person_scores >= _CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        person_scores = person_scores[mask]

        # Convert centre-format to corner-format (x1, y1, x2, y2) in model space
        boxes_xyxy = np.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

        # Scale to original image dimensions
        scale_x = orig_w / _INPUT_SIZE
        scale_y = orig_h / _INPUT_SIZE
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # OpenCV NMS expects (x, y, w, h) as *integers*
        boxes_xywh = np.empty_like(boxes_xyxy)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            person_scores.tolist(),
            _CONFIDENCE_THRESHOLD,
            _NMS_IOU_THRESHOLD,
        )

        if len(indices) == 0:
            return []

        # Flatten in case OpenCV returns a nested array
        indices = np.asarray(indices).flatten()

        min_height = settings.min_person_bbox_height

        results: list[DetectionResult] = []
        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            bbox_height = y2 - y1

            if bbox_height < min_height:
                continue

            results.append(
                DetectionResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(person_scores[idx]),
                    class_id=_PERSON_CLASS_ID,
                )
            )

        return results
