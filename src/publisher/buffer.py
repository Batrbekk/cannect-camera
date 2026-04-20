"""Offline payload buffer for the CANNECT.AI analytics pipeline.

When the network is unavailable, unsent :class:`AnalyticsPayload` objects are
persisted to disk as individual JSON files.  Once connectivity is restored the
buffer is drained in FIFO order.

Files are named with an ISO-8601 timestamp so lexicographic sorting equals
chronological ordering.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.aggregation.models import AnalyticsPayload

logger = logging.getLogger(__name__)

# At 60-second aggregation windows, 1 440 files cover 24 hours.
_MAX_BUFFERED_FILES: int = 1440


class OfflineBuffer:
    """Disk-backed FIFO buffer for analytics payloads.

    Parameters
    ----------
    buffer_dir:
        Directory where buffered JSON files are stored.  Created
        automatically if it does not exist.
    """

    def __init__(self, buffer_dir: str = "buffer") -> None:
        self._buffer_dir = Path(buffer_dir)
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Offline buffer directory: %s", self._buffer_dir.resolve())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, payload: AnalyticsPayload) -> None:
        """Serialize *payload* to a JSON file in the buffer directory.

        If the buffer already contains :data:`_MAX_BUFFERED_FILES` entries
        the oldest file is evicted (FIFO) before saving the new one.
        """
        self._evict_if_full()

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        filename = f"payload_{ts}.json"
        filepath = self._buffer_dir / filename

        try:
            data = payload.model_dump_json(by_alias=True)
            filepath.write_text(data, encoding="utf-8")
            logger.info("Buffered payload to %s", filename)
        except Exception:
            logger.exception("Failed to save payload to buffer file %s", filepath)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[AnalyticsPayload]:
        """Load every buffered payload, sorted oldest-first.

        Returns
        -------
        list[AnalyticsPayload]
            Deserialized payloads ready to be re-sent. Files that fail to
            parse are logged and skipped.
        """
        files = self._sorted_files()
        payloads: list[AnalyticsPayload] = []

        for filepath in files:
            try:
                raw = filepath.read_text(encoding="utf-8")
                payload = AnalyticsPayload.model_validate_json(raw)
                payloads.append(payload)
            except Exception:
                logger.exception("Failed to load buffered file %s — skipping", filepath.name)

        logger.debug("Loaded %d buffered payloads", len(payloads))
        return payloads

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def remove(self, filename: str) -> None:
        """Delete a successfully-sent payload file by *filename*.

        Parameters
        ----------
        filename:
            The basename of the file within the buffer directory.
        """
        filepath = self._buffer_dir / filename
        try:
            filepath.unlink(missing_ok=True)
            logger.debug("Removed buffered file %s", filename)
        except Exception:
            logger.exception("Error removing buffer file %s", filename)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of buffered payload files."""
        return len(self._sorted_files())

    def _sorted_files(self) -> list[Path]:
        """Return all ``payload_*.json`` files sorted lexicographically."""
        files = sorted(self._buffer_dir.glob("payload_*.json"))
        return files

    def _evict_if_full(self) -> None:
        """Remove the oldest file(s) if the buffer is at capacity."""
        files = self._sorted_files()

        while len(files) >= _MAX_BUFFERED_FILES:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                logger.warning(
                    "Buffer full (%d files) — evicted oldest: %s",
                    _MAX_BUFFERED_FILES,
                    oldest.name,
                )
            except Exception:
                logger.exception("Failed to evict buffer file %s", oldest.name)
