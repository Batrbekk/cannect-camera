"""SQLite write-through cache for CounterStore snapshots and raw events.

Provides crash-recovery for in-memory counters and an offline event queue
so that data is never lost when the backend is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from src.aggregation.counters import CounterStore

logger = logging.getLogger(__name__)

_SCHEMA_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS counter_snapshots (
    date TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_SCHEMA_RAW_EVENTS = """
CREATE TABLE IF NOT EXISTS raw_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    camera_id TEXT,
    campaign_id TEXT,
    event_type TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sent INTEGER DEFAULT 0
);
"""


class CounterPersistence:
    """Persist counter snapshots and raw events to a local SQLite database.

    The database is created automatically on first use.  All public methods
    are thread-safe (guarded by an internal lock).

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.  Parent directories
        are created if they do not exist.
    """

    def __init__(self, db_path: str = "data/counters.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()

        # Ensure the parent directory exists.
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)

        self._init_db()
        logger.info("CounterPersistence ready: %s", self._db_path)

    # ------------------------------------------------------------------
    # Database bootstrap
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new per-call connection (SQLite is not thread-safe by default)."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(_SCHEMA_SNAPSHOTS)
                conn.execute(_SCHEMA_RAW_EVENTS)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Counter snapshot persistence
    # ------------------------------------------------------------------

    def save(self, counters: CounterStore) -> None:
        """Serialise the current counter state and upsert into SQLite.

        The snapshot is keyed by today's date (``YYYY-MM-DD``), so at most
        one row per day is stored.
        """
        date_key = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        data_json = json.dumps(counters.to_dict(), default=str)
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO counter_snapshots (date, data, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        data = excluded.data,
                        updated_at = excluded.updated_at
                    """,
                    (date_key, data_json, now_iso),
                )
                conn.commit()
                logger.debug("Counter snapshot saved for %s", date_key)
            finally:
                conn.close()

    def load(self) -> dict | None:
        """Load the latest counter snapshot for today.

        Returns
        -------
        dict or None
            The deserialised snapshot dictionary, or ``None`` if no
            snapshot exists for today.
        """
        date_key = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT data FROM counter_snapshots WHERE date = ?",
                    (date_key,),
                ).fetchone()
                if row is None:
                    return None
                data: dict = json.loads(row["data"])
                logger.debug("Counter snapshot loaded for %s", date_key)
                return data
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Raw event queue (offline replay)
    # ------------------------------------------------------------------

    def save_raw_event(
        self,
        station_id: str,
        camera_id: str,
        campaign_id: str | None,
        event_type: str,
        data: dict,
    ) -> None:
        """Store an individual analytics event for later replay.

        Parameters
        ----------
        station_id:
            Station identifier.
        camera_id:
            Camera that produced the event.
        campaign_id:
            Campaign active at the time of the event (may be ``None``).
        event_type:
            One of ``traffic``, ``attention``, ``demographic``, etc.
        data:
            Arbitrary event payload.
        """
        data_json = json.dumps(data, default=str)
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO raw_events
                        (station_id, camera_id, campaign_id, event_type, data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (station_id, camera_id, campaign_id, event_type, data_json, now_iso),
                )
                conn.commit()
            finally:
                conn.close()

    def get_pending_events(self, limit: int = 1000) -> list[dict]:
        """Retrieve unsent events (``sent = 0``), oldest first.

        Parameters
        ----------
        limit:
            Maximum number of events to return.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``station_id``, ``camera_id``,
            ``campaign_id``, ``event_type``, ``data`` (parsed), and
            ``created_at``.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """
                    SELECT id, station_id, camera_id, campaign_id,
                           event_type, data, created_at
                    FROM raw_events
                    WHERE sent = 0
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

                results: list[dict] = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "station_id": row["station_id"],
                        "camera_id": row["camera_id"],
                        "campaign_id": row["campaign_id"],
                        "event_type": row["event_type"],
                        "data": json.loads(row["data"]),
                        "created_at": row["created_at"],
                    })
                return results
            finally:
                conn.close()

    def mark_sent(self, event_ids: list[int]) -> None:
        """Mark events as successfully sent to the backend.

        Parameters
        ----------
        event_ids:
            List of ``raw_events.id`` values to flag as sent.
        """
        if not event_ids:
            return

        with self._lock:
            conn = self._get_connection()
            try:
                placeholders = ",".join("?" for _ in event_ids)
                conn.execute(
                    f"UPDATE raw_events SET sent = 1 WHERE id IN ({placeholders})",
                    event_ids,
                )
                conn.commit()
                logger.debug("Marked %d events as sent", len(event_ids))
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup(self, max_age_days: int = 7) -> None:
        """Delete data older than *max_age_days*.

        Removes old counter snapshots and sent raw events beyond the
        retention window.
        """
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)
        ).strftime("%Y-%m-%d")
        cutoff_iso = (
            datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)
        ).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                # Old snapshots
                cur = conn.execute(
                    "DELETE FROM counter_snapshots WHERE date < ?", (cutoff,)
                )
                snapshots_deleted = cur.rowcount

                # Sent events older than retention
                cur = conn.execute(
                    "DELETE FROM raw_events WHERE sent = 1 AND created_at < ?",
                    (cutoff_iso,),
                )
                events_deleted = cur.rowcount

                conn.commit()
                if snapshots_deleted or events_deleted:
                    logger.info(
                        "Cleanup: removed %d snapshots, %d sent events (older than %s)",
                        snapshots_deleted,
                        events_deleted,
                        cutoff,
                    )
            finally:
                conn.close()
