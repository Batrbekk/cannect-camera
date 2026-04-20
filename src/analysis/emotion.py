"""Emotion classification from aligned face crops.

Uses the HSEmotion MobileNet ONNX model to classify facial expressions
into one of five categories: joy, interest, surprise, neutral, or
negative.  Inference is gated on attention — the classifier should only
be invoked when the person is actively looking at the screen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort

from src.aggregation.models import EmotionType
from src.config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# HSEmotion input dimensions.
_INPUT_SIZE: int = 224

# Mapping from HSEmotion model output indices to our reduced emotion set.
# The HSEmotion model outputs 8 classes:
#   0=anger, 1=contempt, 2=disgust, 3=fear, 4=happiness, 5=neutral,
#   6=sadness, 7=surprise
# We map them to 5 coarser categories.
_EMOTION_MAP: dict[int, EmotionType] = {
    0: EmotionType.NEGATIVE,    # anger
    1: EmotionType.NEGATIVE,    # contempt
    2: EmotionType.NEGATIVE,    # disgust
    3: EmotionType.NEGATIVE,    # fear
    4: EmotionType.JOY,         # happiness
    5: EmotionType.NEUTRAL,     # neutral
    6: EmotionType.NEGATIVE,    # sadness
    7: EmotionType.SURPRISE,    # surprise
}

# Confidence threshold below which we fall back to neutral.
_CONFIDENCE_THRESHOLD: float = 0.5


class EmotionClassifier:
    """Classify facial emotion from an aligned face crop.

    Parameters
    ----------
    model_path:
        Path to the HSEmotion MobileNet ONNX model.
    """

    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name
        logger.info("EmotionClassifier loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """Resize and normalise *aligned_face* for HSEmotion inference.

        Steps:
        1. Resize to 224x224.
        2. Convert to float32 and scale to [0, 1].
        3. Normalise with ImageNet mean/std.
        4. Transpose HWC -> CHW.
        5. Add batch dimension -> (1, 3, 224, 224).
        """
        img = cv2.resize(aligned_face, (_INPUT_SIZE, _INPUT_SIZE))
        img = img.astype(np.float32) / 255.0

        # ImageNet normalisation.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, aligned_face: np.ndarray) -> tuple[EmotionType, float]:
        """Classify the dominant emotion in *aligned_face*.

        This method should only be called when the person is actively
        attending (looking at the screen).  The caller is responsible for
        gating on attention state.

        Parameters
        ----------
        aligned_face:
            BGR 112x112 aligned face crop (uint8).

        Returns
        -------
        tuple[EmotionType, float]
            ``(emotion, confidence)``.  If confidence is below
            ``_CONFIDENCE_THRESHOLD`` (0.5), emotion falls back to
            :pyattr:`EmotionType.NEUTRAL`.
        """
        blob = self._preprocess(aligned_face)
        raw_output = self._session.run(
            [self._output_name], {self._input_name: blob}
        )[0]
        logits = raw_output.flatten()

        probabilities = self._softmax(logits)

        # Aggregate probabilities for our 5 coarser categories.
        aggregated: dict[EmotionType, float] = {e: 0.0 for e in EmotionType}
        for idx, prob in enumerate(probabilities):
            if idx in _EMOTION_MAP:
                aggregated[_EMOTION_MAP[idx]] += float(prob)

        # Special handling: "interest" is modelled as moderate happiness +
        # surprise that doesn't dominate either category.
        joy_score = aggregated[EmotionType.JOY]
        surprise_score = aggregated[EmotionType.SURPRISE]
        if 0.15 <= joy_score <= 0.45 and 0.10 <= surprise_score <= 0.40:
            interest_score = (joy_score + surprise_score) * 0.6
            aggregated[EmotionType.INTEREST] = interest_score

        # Pick the top emotion.
        best_emotion = max(aggregated, key=lambda e: aggregated[e])
        best_conf = aggregated[best_emotion]

        if best_conf < _CONFIDENCE_THRESHOLD:
            return EmotionType.NEUTRAL, best_conf

        return best_emotion, round(best_conf, 3)
