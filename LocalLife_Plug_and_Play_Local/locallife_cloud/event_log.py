"""Canonical, idempotent persistence of finalised measurement events.

Why this module exists
----------------------
`measurements.csv` came back blank from real runs, and fixing the download
button would not have helped: the download route already reads exactly the file
the pipeline writes. The defect was upstream, in *what* was written and *when*.

The previous writer ran on every processed frame and appended a row the first
time a track happened to carry a volume -- a transient per-frame estimate, not
the finalised event. Two consequences, both seen in the field:

* With no empty-bin baseline, `realsense_volume_l` stays None on every frame, so
  the condition never fired and the file was never created at all. The operator
  downloaded a header and concluded the export was broken.
* Its idempotency key was the tracker's `track_id`, which restarts at 1 on every
  process restart and on cloud reconnection, so the same physical object could
  be written twice, or a new object silently suppressed as a duplicate.

Here the unit of record is the *canonical accepted deposit* -- the same event
the ledger and the dashboard count -- keyed by a durable `event_id` that
survives restart, refresh, retry and reconnection. A rejected deposit is kept
too, with its reason, so a run's withheld measurements are auditable rather than
invisible.

Local and cloud modes write through this same class, with the same schema, on
the laptop side, so losing the cloud VM never loses the research record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MEASUREMENT_CSV_NAME = "measurements.csv"

# Excel opens a UTF-8 file as the local 8-bit codepage unless it sees a BOM,
# which turns any non-ASCII label into mojibake. `utf-8-sig` writes the BOM;
# Python's own csv reader strips it transparently, so the round trip is clean.
CSV_ENCODING = "utf-8-sig"

# `status` is part of the schema rather than a separate file so one download
# answers "what was accepted" and "what was withheld, and why".
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"


@dataclass
class PersistResult:
    """What happened to one `record()` call.

    `duplicate` is a success: the event was already durable, which is exactly
    what should happen when a frame repeats, a browser refreshes, or the cloud
    link drops and reconnects mid-deposit.
    """

    event_id: str
    written: bool
    duplicate: bool = False
    error: str | None = None
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _Failure:
    event_id: str
    reason: str
    row: dict[str, Any] = field(default_factory=dict)


def resolve_event_id(
    session_id: str, camera_id: str, track_id: int | None, *, suffix: str = "",
) -> str:
    """A durable id for one finalised object.

    Deliberately derived, not random: the same physical deposit must resolve to
    the same id if the pipeline retries it, so a retry cannot append a second
    row. `track_id` alone is not enough (it restarts at 1 every run), hence the
    session and camera in the digest.
    """
    key = f"{session_id}|{camera_id}|{track_id}|{suffix}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class MeasurementEventLog:
    """One row per finalised event, written once, to a path callers can see.

    Thread-safe: the pipeline writes from its processing thread while Flask
    serves the download from another.
    """

    COLUMNS = [
        # Identity and provenance -- first, so a spreadsheet opens on the keys.
        # `event_id` is the permanent measurement id from stable_identity.py,
        # never the detector's track number, which is renumbered mid-object.
        "session_id", "event_id", "track_id", "timestamp", "processing_mode",
        "camera_source", "camera_id", "calibration_id", "operating_mode",
        "diagnostic", "status", "reason",
        # Classification
        "object_type", "label", "accepted_class",
        "colour", "color", "colour_confidence", "color_confidence",
        "material", "material_confidence", "sorting_result", "sorting_status",
        "bag_count",
        # Dimensions and volume
        "length_mm", "width_mm", "height_mm",
        "estimated_litres", "volume_l", "added_volume_l", "displaced_volume_l",
        "volume_before_l", "volume_after_l", "volume_uncertainty_l",
        "volume_confidence", "overall_confidence",
        # Method and quality
        "dimension_confidence", "dimension_method", "depth_coverage_percent",
        "measurement_method", "measurement_quality", "volume_rejection_reason",
        "calibration_valid", "stable_frames", "detector_track_ids",
        "model_version", "pipeline_version",
        # Optional ground truth, filled in by the operator for benchmark runs.
        "ground_truth_litres", "absolute_error_litres", "percentage_error",
    ]

    # British and American spellings of the same field are both written, with
    # the same value. The spec names `colour`/`sorting_result`, the pipeline and
    # every existing export use `color`/`sorting_status`, and silently dropping
    # either would break one of them.
    ALIASES = {
        "colour": "color",
        "colour_confidence": "color_confidence",
        "sorting_result": "sorting_status",
        "object_type": "label",
        "estimated_litres": "volume_l",
    }

    def __init__(
        self,
        directory: Path,
        *,
        name: str = MEASUREMENT_CSV_NAME,
        session_id: str | None = None,
    ) -> None:
        # Resolved once, from the configured application data directory -- never
        # from the current working directory, which differs between the one-click
        # launcher, a developer shell and the service under Windows.
        self.directory = Path(directory).resolve()
        self.path = self.directory / name
        self._lock = threading.Lock()
        self._session_id = session_id
        self._failures: list[_Failure] = []
        self._persisted = 0
        self._last_error: str | None = None
        self._last_write_at: float | None = None
        self._seen: set[str] = set()
        self._recover_existing_event_ids()

    def assert_writable(self) -> None:
        """Fail loudly at startup if the CSV cannot be written.

        An unwritable results directory used to surface hours later as an empty
        download. The operator should learn about it before the demonstration,
        not after it.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            probe = self.directory / ".locallife_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Measurement CSV persistence cannot start: {self.path} is not "
                f"writable ({exc}). Fix the results directory before running; "
                "the system will not record measurements without it."
            ) from exc

    def startup_report(self, *, diagnostic: bool = False) -> list[str]:
        """The three lines the operator must see before a run begins."""
        return [
            f"Measurement mode: {'diagnostic' if diagnostic else 'normal'}",
            "History ledger: enabled",
            f"CSV persistence: enabled ({self.path})",
        ]

    # ------------------------------------------------------------- identity
    @property
    def session_id(self) -> str:
        """The operator session these events belong to.

        Read from the same `operator_session.json` the dashboard uses, so a row
        written by the pipeline and a session shown in the UI agree. Cached
        after the first successful read.
        """
        if self._session_id:
            return self._session_id
        location = self.directory / "operator_session.json"
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("session_id"):
                self._session_id = str(payload["session_id"])
                return self._session_id
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return "unknown-session"

    def _recover_existing_event_ids(self) -> None:
        """Rebuild the idempotency set from what is already on disk.

        Without this, restarting the service would re-accept an event it had
        already written. Earlier sessions are never rewritten or truncated --
        only read.
        """
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding=CSV_ENCODING, newline="") as source:
                for row in csv.DictReader(source):
                    event_id = (row.get("event_id") or "").strip()
                    if event_id:
                        self._seen.add(event_id)
                        self._persisted += 1
        except (OSError, csv.Error) as exc:
            # A damaged file must not take the pipeline down; it does mean
            # duplicates become possible, so say so loudly.
            LOGGER.warning("Could not read existing %s: %s", self.path, exc)
            self._last_error = f"Could not read existing CSV: {exc}"

    # -------------------------------------------------------------- writing
    def record(self, row: dict[str, Any]) -> PersistResult:
        """Persist one finalised event. Idempotent on `event_id`.

        Returns rather than raises: a storage failure must reach the operator's
        screen as a retryable condition, not kill the measurement thread.
        """
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            return PersistResult(event_id="", written=False, error="missing event_id")
        with self._lock:
            if event_id in self._seen:
                return PersistResult(event_id=event_id, written=False, duplicate=True, path=self.path)
            payload = dict(row)
            payload.setdefault("session_id", self.session_id)
            payload.setdefault("status", STATUS_ACCEPTED)
            payload = self._with_aliases(payload)
            payload = self._with_ground_truth_error(payload)
            try:
                self._append(payload)
            except OSError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                # One queued copy per event: a repeated attempt for the same
                # immutable event must not grow the retry queue.
                if all(item.event_id != event_id for item in self._failures):
                    self._failures.append(_Failure(event_id=event_id, reason=reason, row=payload))
                self._last_error = reason
                LOGGER.error("Failed to persist event %s to %s: %s", event_id, self.path, reason)
                return PersistResult(event_id=event_id, written=False, error=reason, path=self.path)
            self._seen.add(event_id)
            self._persisted += 1
            self._last_write_at = time.time()
            LOGGER.info("Persisted event %s to %s", event_id, self.path)
            return PersistResult(event_id=event_id, written=True, path=self.path)

    def _append(self, row: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.is_file() or self.path.stat().st_size == 0
        # newline="" is required by the csv module on Windows; without it every
        # row gains a blank line and Excel shows alternating empty rows.
        with self.path.open("a", encoding=CSV_ENCODING, newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=self.COLUMNS, extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,
            )
            if new_file:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in self.COLUMNS})
            # Durability: a crash or a closed laptop lid between the deposit and
            # the next flush would otherwise lose the row that the dashboard has
            # already shown as complete.
            output.flush()
            os.fsync(output.fileno())

    @classmethod
    def _with_aliases(cls, row: dict[str, Any]) -> dict[str, Any]:
        """Fill each spelling from whichever one the caller supplied."""
        for alias, canonical in cls.ALIASES.items():
            if row.get(alias) in (None, "") and row.get(canonical) not in (None, ""):
                row[alias] = row[canonical]
            elif row.get(canonical) in (None, "") and row.get(alias) not in (None, ""):
                row[canonical] = row[alias]
        return row

    @staticmethod
    def _with_ground_truth_error(row: dict[str, Any]) -> dict[str, Any]:
        """Fill in the error columns when a ground truth was supplied."""
        truth = row.get("ground_truth_litres")
        estimate = row.get("volume_l") if row.get("volume_l") is not None else row.get("estimated_litres")
        if truth in (None, "") or estimate in (None, ""):
            return row
        try:
            truth_value = float(truth)
            estimate_value = float(estimate)
        except (TypeError, ValueError):
            return row
        row["absolute_error_litres"] = round(abs(estimate_value - truth_value), 6)
        if truth_value:
            row["percentage_error"] = round(
                abs(estimate_value - truth_value) / abs(truth_value) * 100.0, 4
            )
        return row

    def set_ground_truth(self, event_id: str, litres: float) -> bool:
        """Attach a ground truth to an already-written row, rewriting in place.

        Benchmark runs let the operator enter the true volume after the object
        is finalised, so the row has to be updated rather than appended.
        """
        if not self.path.is_file():
            return False
        with self._lock:
            try:
                with self.path.open("r", encoding=CSV_ENCODING, newline="") as source:
                    rows = list(csv.DictReader(source))
            except (OSError, csv.Error):
                return False
            found = False
            for row in rows:
                if row.get("event_id") == event_id:
                    row["ground_truth_litres"] = litres
                    self._with_ground_truth_error(row)
                    found = True
            if not found:
                return False
            # Write through a temporary file and replace atomically: a crash
            # mid-rewrite must not leave a half-written record of the session.
            temporary = self.path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding=CSV_ENCODING, newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=self.COLUMNS, extrasaction="ignore",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in self.COLUMNS})
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path)
            return True

    # -------------------------------------------------------------- reading
    def csv_text(self) -> str:
        """Exactly what the download route serves, header-only when empty."""
        if self.path.is_file():
            try:
                return self.path.read_text(encoding=CSV_ENCODING)
            except OSError as exc:
                LOGGER.error("Could not read %s: %s", self.path, exc)
        import io

        output = io.StringIO()
        csv.DictWriter(output, fieldnames=self.COLUMNS).writeheader()
        return output.getvalue()

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding=CSV_ENCODING, newline="") as source:
                return list(csv.DictReader(source))
        except (OSError, csv.Error):
            return []

    def recent(self, limit: int = 30) -> list[dict[str, Any]]:
        """The newest finalised events, for the dashboard's history table.

        Purely a reader. The research page used to show only the waste-plant
        ledger, which geometry-validation mode deliberately leaves empty, so a
        run that WAS recording measurements displayed "Waste ledger disabled in
        validation mode" and looked broken.
        """
        rows = self.rows()[-max(1, limit):]
        return [
            {
                "event_id": row.get("event_id"),
                "timestamp": row.get("timestamp"),
                "object_type": row.get("object_type") or row.get("label"),
                "colour": row.get("colour") or row.get("color"),
                "material": row.get("material"),
                "length_mm": row.get("length_mm"),
                "width_mm": row.get("width_mm"),
                "height_mm": row.get("height_mm"),
                "litres": row.get("estimated_litres") or row.get("volume_l"),
                "status": row.get("status"),
                "reason": row.get("reason"),
                # What the operator asked to see per row.
                "history_saved": True,
                "csv_saved": True,
            }
            for row in reversed(rows)
        ]

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        """What the operator page shows: saved, failed, and where the file is."""
        return {
            "csv_path": str(self.path),
            "session_id": self.session_id,
            "events_persisted": self._persisted,
            # So the page can say when the canonical file last grew. A CSV
            # already downloaded is a snapshot; it never updates in Excel, and
            # the operator needs to see that the live file is still moving.
            "last_write_at": self._last_write_at,
            "persistence_failures": len(self._failures),
            "last_error": self._last_error,
            "failed_event_ids": [item.event_id for item in self._failures],
            # Drives the "CSV saved" / "CSV persistence failed" badge.
            "healthy": not self._failures,
            "empty_note": (
                "No completed measurements recorded" if self._persisted == 0 else None
            ),
        }

    def retry_failed(self) -> dict[str, Any]:
        """Re-attempt every row that failed to persist. Operator-triggered."""
        with self._lock:
            pending, self._failures = self._failures, []
        recovered, still_failing = 0, []
        for failure in pending:
            result = self.record(failure.row)
            if result.ok:
                recovered += 1
            else:
                still_failing.append(failure)
        with self._lock:
            self._failures.extend(still_failing)
            if not self._failures:
                self._last_error = None
        return {"recovered": recovered, "still_failing": len(still_failing)}
