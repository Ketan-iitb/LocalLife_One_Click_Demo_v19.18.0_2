"""Persist experiment records locally and synchronize them to Cloud Storage."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class ResultStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_jsonl(self, name: str, record: dict[str, Any]) -> Path:
        destination = self.directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, destination.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False, default=str) + "\n")
        return destination

    def append_csv(self, name: str, row: dict[str, Any], columns: list[str]) -> Path:
        """Append one row, writing the header the first time the file is created.

        Spreadsheet-readable companion to `append_jsonl`, so an experiment run
        can be opened directly in Excel without a conversion step.
        """
        destination = self.directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            new_file = not destination.is_file() or destination.stat().st_size == 0
            with destination.open("a", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({key: row.get(key) for key in columns})
        return destination

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        destination = self.directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with self._lock:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
            temporary.replace(destination)
        return destination

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        destination = self.directory / name
        if not destination.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self._lock, destination.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipped invalid %s record at line %s", name, line_number)
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


class BucketSync:
    def __init__(self, local_directory: Path, bucket_uri: str, interval_seconds: int = 180) -> None:
        self.local_directory = local_directory
        self.bucket_uri = bucket_uri.rstrip("/") + "/cloud-v15"
        self.interval_seconds = max(30, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sync_once(self) -> bool:
        if not shutil.which("gcloud"):
            LOGGER.warning("gcloud is unavailable; results remain in %s", self.local_directory)
            return False
        command = ["gcloud", "storage", "rsync", "-r", str(self.local_directory), self.bucket_uri]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=150)
        except (OSError, subprocess.TimeoutExpired) as exc:
            LOGGER.warning("Cloud Storage synchronization failed: %s", exc)
            return False
        if result.returncode:
            LOGGER.warning("Cloud Storage synchronization failed: %s", result.stderr.strip())
            return False
        LOGGER.info("Synchronized experiment artifacts to %s", self.bucket_uri)
        return True

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sync_once()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="bucket-sync", daemon=True)
            self._thread.start()

    def stop(self, *, final_sync: bool = True) -> None:
        self._stop_event.set()
        if final_sync:
            self.sync_once()
