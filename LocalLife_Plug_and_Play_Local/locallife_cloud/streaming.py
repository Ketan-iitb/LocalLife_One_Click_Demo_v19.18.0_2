"""Latest-frame, fair dual-camera inference scheduling."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .comparison import CAMERA_IDS, DualCameraCoordinator


LOGGER = logging.getLogger(__name__)


class LatestFrameProcessor:
    """Analyze the newest frame per camera while older queued frames are dropped."""

    def __init__(self, manager: DualCameraCoordinator) -> None:
        self.manager = manager
        self.condition = threading.Condition()
        self.pending: dict[str, dict[str, Any]] = {}
        self.statistics: dict[str, dict[str, Any]] = {
            camera_id: {
                "accepted": 0, "processed": 0, "dropped": 0, "busy": False,
                "last_error": None,
            }
            for camera_id in CAMERA_IDS
        }
        self._next_camera = 0
        self._stopping = False
        self.thread = threading.Thread(
            target=self._run, name="latest-frame-inference", daemon=True
        )
        self.thread.start()

    def submit(self, camera_id: str, packet: dict[str, Any]) -> dict[str, Any]:
        with self.condition:
            stats = self.statistics[camera_id]
            stats["accepted"] += 1
            if camera_id in self.pending:
                stats["dropped"] += 1
            self.pending[camera_id] = packet
            self.condition.notify()
            return self.snapshot(camera_id)

    def snapshot(self, camera_id: str | None = None) -> dict[str, Any]:
        with self.condition:
            if camera_id is not None:
                stats = dict(self.statistics[camera_id])
                stats["queued"] = camera_id in self.pending
                return stats
            return {
                key: {**value, "queued": key in self.pending}
                for key, value in self.statistics.items()
            }

    def _take_next(self) -> list[tuple[str, dict[str, Any]]] | None:
        with self.condition:
            while not self.pending and not self._stopping:
                self.condition.wait(timeout=1.0)
            if self._stopping:
                return None
            batch: list[tuple[str, dict[str, Any]]] = []
            for offset in range(len(CAMERA_IDS)):
                index = (self._next_camera + offset) % len(CAMERA_IDS)
                camera_id = CAMERA_IDS[index]
                if camera_id in self.pending:
                    packet = self.pending.pop(camera_id)
                    self.statistics[camera_id]["busy"] = True
                    batch.append((camera_id, packet))
            if batch:
                first_index = CAMERA_IDS.index(batch[0][0])
                self._next_camera = (first_index + 1) % len(CAMERA_IDS)
            return batch

    def _run(self) -> None:
        while True:
            work = self._take_next()
            if work is None:
                return
            packets = dict(work)
            try:
                self.manager.process_packets(packets)
                with self.condition:
                    for camera_id in packets:
                        self.statistics[camera_id]["processed"] += 1
                        self.statistics[camera_id]["last_error"] = None
            except Exception as exc:  # pragma: no cover - retains environment-specific failures
                LOGGER.exception("Background inference failed for %s", ", ".join(packets))
                with self.condition:
                    for camera_id in packets:
                        self.statistics[camera_id]["last_error"] = str(exc)
            finally:
                with self.condition:
                    for camera_id in packets:
                        self.statistics[camera_id]["busy"] = False
                    self.condition.notify_all()

    def stop(self) -> None:
        with self.condition:
            self._stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=3.0)
