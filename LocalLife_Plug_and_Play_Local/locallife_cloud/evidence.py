"""Opt-in benchmark evidence recording.

A thesis benchmark wants frames to point at, but a RealSense at 30 FPS with
aligned depth fills a laptop disk in minutes, and a writer on the inference
thread turns a latency measurement into a measurement of the writer. So:

* off by default, and obvious when it is on;
* sampled, not every frame;
* bounded by both duration and megabytes, whichever runs out first;
* written from a background thread with a bounded queue that drops rather than
  blocks -- dropping evidence is acceptable, stalling the pipeline being
  measured is not;
* finalised on shutdown so a half-written session is still readable;
* linked to the CSV by `event_id`, so a row and its frames find each other.

Depth is referenced, not duplicated: a full raw depth stream is the thing that
actually fills the disk, and the finalised events already carry the derived
measurements.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

METADATA_NAME = "evidence.jsonl"


@dataclass
class RecordingSettings:
    """Bounded by construction; every limit has a default that fits a laptop."""

    enabled: bool = False
    # 12 FPS is enough to see a bag land and to read an overlay, and is an
    # eighth of the raw data of a 30 FPS capture.
    sample_fps: float = 12.0
    max_duration_seconds: float = 300.0
    max_megabytes: float = 750.0
    # Frames either side of a finalised event, which is the part anyone
    # actually reviews.
    frames_around_event: int = 8
    # Never on by default: evidence may show a room, and uploading it is a
    # separate decision from recording it.
    upload_to_cloud: bool = False

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.max_duration_seconds <= 0 or self.max_megabytes <= 0:
            raise ValueError("recording limits must be positive")


@dataclass
class RecordingStatus:
    active: bool = False
    session_directory: str | None = None
    frames_written: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    stopped_reason: str | None = None
    started_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "session_directory": self.session_directory,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "megabytes_written": round(self.bytes_written / 1_000_000, 2),
            "stopped_reason": self.stopped_reason,
            "elapsed_seconds": (
                None if self.started_at is None else round(time.time() - self.started_at, 1)
            ),
        }


class EvidenceRecorder:
    """Sampled, bounded, non-blocking evidence writer.

    `writer` is injected: the real one encodes an image, the tests count calls.
    Keeping encoding out of this class is what lets the limits and the
    drop-rather-than-block behaviour be tested without OpenCV.
    """

    def __init__(
        self,
        root: Path,
        settings: RecordingSettings | None = None,
        *,
        writer: Callable[[Path, Any], int] | None = None,
        clock: Callable[[], float] = time.time,
        queue_size: int = 32,
    ) -> None:
        self.root = Path(root).resolve()
        self.settings = settings or RecordingSettings()
        self.settings.validate()
        self.status = RecordingStatus()
        self._writer = writer or self._default_writer
        self._clock = clock
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_sample_at = 0.0
        self._session: Path | None = None
        self._metadata = None

    # --------------------------------------------------------------- control
    def start(self, session_id: str) -> RecordingStatus:
        if not self.settings.enabled:
            self.status.stopped_reason = "recording is switched off"
            return self.status
        with self._lock:
            if self.status.active:
                return self.status
            self._session = self.root / "evidence" / session_id
            self._session.mkdir(parents=True, exist_ok=True)
            self._metadata = (self._session / METADATA_NAME).open("a", encoding="utf-8")
            self.status = RecordingStatus(
                active=True, session_directory=str(self._session),
                started_at=self._clock(),
            )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._drain, name="evidence-writer", daemon=True,
            )
            self._thread.start()
        return self.status

    def stop(self, reason: str = "stopped by the operator") -> RecordingStatus:
        with self._lock:
            if not self.status.active:
                return self.status
            self.status.active = False
            self.status.stopped_reason = reason
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            if self._metadata is not None:
                # Flush and close so a session killed mid-run is still readable.
                self._metadata.flush()
                self._metadata.close()
                self._metadata = None
        return self.status

    # --------------------------------------------------------------- capture
    def offer(
        self,
        frame: Any,
        *,
        frame_id: int,
        processing_mode: str,
        event_id: str | None = None,
        kind: str = "overlay",
    ) -> bool:
        """Offer a frame. Returns whether it was queued.

        Never blocks and never raises: an inference loop must not wait on a
        disk, and a full queue means the disk is already behind.
        """
        if not self.status.active:
            return False
        now = self._clock()
        if self._exceeded_limits(now):
            return False
        # An event frame always goes in; ordinary frames are sampled.
        if event_id is None:
            interval = 1.0 / self.settings.sample_fps
            if now - self._last_sample_at < interval:
                return False
        self._last_sample_at = now
        record = {
            "frame_id": frame_id, "kind": kind, "event_id": event_id,
            "processing_mode": processing_mode, "timestamp": now,
        }
        try:
            self._queue.put_nowait((record, frame))
        except queue.Full:
            with self._lock:
                self.status.frames_dropped += 1
            return False
        return True

    def _exceeded_limits(self, now: float) -> bool:
        # `is None`, not `or`: a started_at of 0.0 is falsy, so `or now` would
        # reset the window on every frame and the duration limit would never
        # fire. Only a test clock starts at zero, but the bug is real.
        started = now if self.status.started_at is None else self.status.started_at
        if now - started >= self.settings.max_duration_seconds:
            self.stop("reached the configured maximum duration")
            return True
        if self.status.bytes_written >= self.settings.max_megabytes * 1_000_000:
            self.stop("reached the configured maximum size")
            return True
        return False

    # ---------------------------------------------------------------- writing
    @staticmethod
    def _default_writer(path: Path, frame: Any) -> int:  # pragma: no cover - needs cv2
        import cv2

        cv2.imwrite(str(path), frame)
        return path.stat().st_size if path.is_file() else 0

    def _drain(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                record, frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                assert self._session is not None
                name = f"{record['frame_id']:08d}_{record['kind']}.png"
                written = self._writer(self._session / name, frame)
                record["file"] = name
                record["bytes"] = written
                with self._lock:
                    self.status.frames_written += 1
                    self.status.bytes_written += int(written or 0)
                    if self._metadata is not None:
                        self._metadata.write(json.dumps(record) + "\n")
                        self._metadata.flush()
            except Exception as exc:  # noqa: BLE001 - evidence must never crash a run
                LOGGER.warning("Could not write evidence frame: %s", exc)

    # ---------------------------------------------------------------- reading
    def records(self) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        location = self._session / METADATA_NAME
        if not location.is_file():
            return []
        entries = []
        for line in location.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def disk_usage(self) -> dict[str, Any]:
        """Where the evidence is and how much of the disk it has taken."""
        if self._session is None or not self._session.is_dir():
            return {"path": None, "megabytes": 0.0, "files": 0}
        files = [item for item in self._session.rglob("*") if item.is_file()]
        return {
            "path": str(self._session),
            "megabytes": round(sum(item.stat().st_size for item in files) / 1_000_000, 2),
            "files": len(files),
        }
