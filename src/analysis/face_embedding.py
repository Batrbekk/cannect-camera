"""Face embedding extraction and in-memory face store.

Provides :class:`FaceEmbedder` for producing 512-dimensional face embeddings
from aligned face crops using MobileFaceNet, and :class:`FaceStore` for
de-duplicating identities via cosine similarity with a configurable TTL.

PRIVACY: All embeddings are held exclusively in RAM and are **never**
serialised to disk.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from src.config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# MobileFaceNet input dimensions.
_INPUT_SIZE: int = 112


class FaceEmbedder:
    """Extract 512-dim normalised embeddings from aligned face images.

    Parameters
    ----------
    model_path:
        Path to the MobileFaceNet ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name
        logger.info("FaceEmbedder loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(aligned_face: np.ndarray) -> np.ndarray:
        """Normalise and reshape an aligned 112x112 BGR face for inference.

        Steps:
        1. Convert to float32 and scale to [0, 1].
        2. Normalise with ImageNet-style mean/std.
        3. Transpose HWC -> CHW.
        4. Add batch dimension -> (1, 3, 112, 112).
        """
        img = aligned_face.astype(np.float32) / 255.0
        # Standard normalisation used by MobileFaceNet training.
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension.
        return np.expand_dims(img, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return a 512-dim L2-normalised embedding.

        Parameters
        ----------
        aligned_face:
            BGR image of shape ``(112, 112, 3)`` (uint8).

        Returns
        -------
        np.ndarray
            Float32 vector of length 512, L2-normalised.
        """
        blob = self._preprocess(aligned_face)
        output = self._session.run([self._output_name], {self._input_name: blob})[0]
        embedding = output.flatten().astype(np.float32)
        # L2 normalise so cosine similarity == dot product.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


# ======================================================================
# FaceStore — in-memory identity store with TTL
# ======================================================================

class FaceStore:
    """In-memory face identity store backed by cosine similarity.

    Each embedding is associated with a unique integer face ID and a
    timestamp.  Entries older than *ttl* seconds are evicted by
    :meth:`cleanup`.

    PRIVACY: embeddings are stored **only** in RAM and are never written
    to disk.
    """

    def __init__(self, ttl: int | None = None) -> None:
        self._ttl: int = ttl if ttl is not None else settings.unique_face_ttl_sec
        self._next_id: int = 0
        # {face_id: (embedding, timestamp)}
        self._store: dict[int, tuple[np.ndarray, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, timestamp: float) -> int:
        """Insert *embedding* and return a unique face ID."""
        face_id = self._next_id
        self._next_id += 1
        self._store[face_id] = (embedding.copy(), timestamp)
        return face_id

    def match(self, embedding: np.ndarray) -> tuple[int | None, float]:
        """Find the closest match for *embedding*.

        Returns
        -------
        tuple[int | None, float]
            ``(face_id, similarity)`` if the best cosine similarity
            exceeds :pyattr:`settings.face_match_threshold`, otherwise
            ``(None, best_similarity)``.  If the store is empty the
            similarity is ``0.0``.
        """
        if not self._store:
            return None, 0.0

        best_id: int | None = None
        best_sim: float = -1.0

        for face_id, (stored_emb, _ts) in self._store.items():
            # Both vectors are L2-normalised so dot == cosine sim.
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = face_id

        threshold: float = settings.face_match_threshold
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    def get_unique_count(self) -> int:
        """Return the number of unique face identities currently stored."""
        return len(self._store)

    def cleanup(self) -> None:
        """Remove entries whose timestamp is older than the TTL."""
        now = time.time()
        expired = [
            fid
            for fid, (_emb, ts) in self._store.items()
            if (now - ts) > self._ttl
        ]
        for fid in expired:
            del self._store[fid]
        if expired:
            logger.debug("FaceStore cleanup: removed %d expired entries.", len(expired))
