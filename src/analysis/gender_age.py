"""Gender and age estimation from aligned face crops.

Uses the InsightFace *genderage* ONNX model to predict binary gender
and continuous age from a 112x112 aligned face image.  Predictions are
gated by a frontality check (yaw/pitch thresholds) so that only
reasonably frontal faces produce demographic estimates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from src.aggregation.models import AgeGroup, Gender
from src.config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# InsightFace genderage model input size.
_INPUT_SIZE: int = 96


class GenderAgeEstimator:
    """Predict gender and age from an aligned face crop.

    Parameters
    ----------
    model_path:
        Path to the InsightFace *genderage* ONNX model.
    """

    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name
        logger.info("GenderAgeEstimator loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """Resize and normalise *aligned_face* for inference.

        Accepts a 112x112 BGR uint8 image, resizes to the model's expected
        96x96 input, normalises to [0, 1], and converts to NCHW float32.
        """
        import cv2

        img = cv2.resize(aligned_face, (_INPUT_SIZE, _INPUT_SIZE))
        img = (img.astype(np.float32) - 127.5) / 127.5  # InsightFace normalisation
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0)  # add batch dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_frontal(yaw: float, pitch: float) -> bool:
        """Return ``True`` if the face is frontal enough for estimation.

        Uses thresholds of |yaw| <= 30 and |pitch| <= 20 degrees.
        """
        return abs(yaw) <= 30.0 and abs(pitch) <= 20.0

    def predict(
        self,
        aligned_face: np.ndarray,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> tuple[Gender, float, int, float]:
        """Estimate gender and age.

        Parameters
        ----------
        aligned_face:
            BGR 112x112 aligned face crop (uint8).
        yaw:
            Horizontal head rotation in degrees (from head-pose estimator).
        pitch:
            Vertical head rotation in degrees.

        Returns
        -------
        tuple[Gender, float, int, float]
            ``(gender, gender_confidence, age, age_confidence)``

            * *gender* – :pyclass:`Gender.MALE`, ``FEMALE``, or ``UNKNOWN``
              if the confidence is below ``settings.gender_confidence``.
            * *gender_confidence* – raw sigmoid output in [0, 1].
            * *age* – integer age estimate.
            * *age_confidence* – heuristic confidence for the age prediction.
        """
        # Gate on frontality.
        if not self.is_frontal(yaw, pitch):
            logger.debug(
                "Face not frontal (yaw=%.1f, pitch=%.1f); skipping gender/age.",
                yaw,
                pitch,
            )
            return Gender.UNKNOWN, 0.0, 0, 0.0

        blob = self._preprocess(aligned_face)
        output = self._session.run([self._output_name], {self._input_name: blob})[0]
        output = output.flatten()

        # InsightFace genderage output layout (fc1, shape [1,3]):
        #   output[0:2] = gender logits [male_logit, female_logit] (argmax → 0=male, 1=female)
        #   output[2]   = age / 100 (multiply by 100 to get age in years)
        male_logit: float = float(output[0])
        female_logit: float = float(output[1])
        age_raw: float = float(output[2]) * 100.0

        # Softmax for gender confidence
        exp_m = float(np.exp(male_logit - max(male_logit, female_logit)))
        exp_f = float(np.exp(female_logit - max(male_logit, female_logit)))
        total = exp_m + exp_f
        male_prob = exp_m / total
        female_prob = exp_f / total

        # Map to Gender enum
        gender_conf: float
        if male_prob >= settings.gender_confidence:
            gender = Gender.MALE
            gender_conf = male_prob
        elif female_prob >= settings.gender_confidence:
            gender = Gender.FEMALE
            gender_conf = female_prob
        else:
            gender = Gender.UNKNOWN
            gender_conf = max(male_prob, female_prob)

        # Age: clamp to a reasonable range
        age: int = max(0, min(100, int(round(age_raw))))

        # Heuristic age confidence: higher when face is more frontal.
        yaw_factor = max(0.0, 1.0 - abs(yaw) / 90.0)
        pitch_factor = max(0.0, 1.0 - abs(pitch) / 90.0)
        age_conf: float = round(yaw_factor * pitch_factor, 3)

        return gender, gender_conf, age, age_conf
