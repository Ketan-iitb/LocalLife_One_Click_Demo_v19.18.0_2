"""Paired RealSense / Logitech measurement events for the thesis comparison.

One physical object placed in the bin is one *comparison event*. Each camera
finalises its own measurement independently (its own tracker, stability rule
and measurement id); this log only groups the two finalised measurements under
one shared `comparison_event_id` and writes one long-format row per camera:

* both cameras are kept separate -- nothing is fused, averaged or substituted;
* a camera that produced no finalised result within the pairing window gets an
  explicit `missing` row, never a copy of the other camera's numbers;
* `measurement_id` (the camera's own permanent event id) is the idempotency
  key, so repeated frames, refreshes, retries and restarts add nothing;
* ground truth is attached per comparison event, after the fact, and is never
  visible to either camera's inference.

The CSV on disk is the canonical store: the download route serves this same
file, and a failed write is queued for retry instead of stopping measurement.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

LOGGER = logging.getLogger(__name__)

CSV_NAME = "comparison_measurements.csv"
GROUND_TRUTH_NAME = "comparison_ground_truth.json"
CSV_ENCODING = "utf-8-sig"
CAMERAS = ("realsense", "logitech")
STATUS_MISSING = "missing"

COLUMNS = [
    "session_id", "comparison_event_id", "measurement_id", "camera_source", "timestamp",
    "processing_mode", "calibration_id", "calibration_method",
    "object_type", "geometry_method", "geometry_confidence",
    "length_mm", "width_mm", "height_mm",
    "cylinder_diameter_mm", "cylinder_height_mm", "cylinder_fit_residual",
    "bounding_box_volume_litres", "mesh_or_shape_volume_litres", "selected_volume_litres",
    "volume_meaning",
    "ground_truth_volume_litres", "ground_truth_method", "dataset_split",
    "absolute_error_litres", "percentage_error",
    "colour", "colour_confidence", "material", "material_confidence", "sorting_result",
    "overall_confidence", "processing_time_ms", "status", "reason",
    "model_version", "pipeline_version",
    # Flat names the operator spreadsheet uses; same values as above.
    "camera", "sorting", "diameter_mm", "radius_mm", "volume_liters",
    "confidence", "fit_confidence", "rejection_reason",
]

GROUND_TRUTH_FIELDS = (
    "actual_object", "actual_material", "actual_colour",
    "actual_length_mm", "actual_width_mm", "actual_height_mm",
    "reference_volume_litres", "ground_truth_method", "dataset_split",
)
GROUND_TRUTH_METHODS = (
    "cuboid_dimensions", "cylinder_dimensions", "manufacturer_capacity",
    "displacement_reference_container", "bounding_box_reference",
)


def _object_family(label: Any) -> str:
    text = str(label or "").lower()
    for family in ("bag", "box", "carton", "can", "bottle"):
        if family in text:
            return family
    return text.strip() or "unknown"


def _compatible(left: str, right: str) -> bool:
    return "unknown" in (left, right) or left == right


@dataclass
class _OpenEvent:
    comparison_event_id: str
    opened_at: float
    object_family: str
    cameras: dict[str, str] = field(default_factory=dict)  # camera -> measurement_id


def comparison_row(
    measurement: dict[str, Any], comparison_event_id: str, session_id: str,
) -> dict[str, Any]:
    """Map one camera's canonical measurement row onto the long-format schema."""
    shape = measurement.get("shape_geometry") or {}
    selected = shape.get("selected_volume_litres")
    if selected is None:
        selected = measurement.get("estimated_litres", measurement.get("volume_l"))
    shape_volume = shape.get("cylinder_volume_litres")
    if shape_volume is None:
        shape_volume = shape.get("mesh_volume_litres")
    return {
        "session_id": session_id,
        "comparison_event_id": comparison_event_id,
        "measurement_id": measurement.get("event_id"),
        "camera_source": measurement.get("camera_source") or measurement.get("camera_id"),
        "timestamp": measurement.get("timestamp"),
        "processing_mode": measurement.get("processing_mode"),
        "calibration_id": measurement.get("calibration_id"),
        "calibration_method": measurement.get("calibration_method"),
        "object_type": measurement.get("object_type") or measurement.get("label"),
        "geometry_method": shape.get("geometry_method") or measurement.get("geometry_method"),
        "geometry_confidence": shape.get("geometry_confidence"),
        "length_mm": _first(shape.get("length_mm"), measurement.get("length_mm")),
        "width_mm": _first(shape.get("width_mm"), measurement.get("width_mm")),
        "height_mm": _first(shape.get("height_mm"), measurement.get("height_mm")),
        "cylinder_diameter_mm": shape.get("cylinder_diameter_mm"),
        "cylinder_height_mm": shape.get("cylinder_height_mm"),
        "cylinder_fit_residual": shape.get("cylinder_fit_residual"),
        "bounding_box_volume_litres": shape.get("bounding_box_volume_litres"),
        "mesh_or_shape_volume_litres": shape_volume,
        "selected_volume_litres": selected,
        "volume_meaning": shape.get("volume_meaning"),
        "colour": measurement.get("colour") or measurement.get("color"),
        "colour_confidence": measurement.get("colour_confidence", measurement.get("color_confidence")),
        "material": measurement.get("material"),
        "material_confidence": measurement.get("material_confidence"),
        "sorting_result": measurement.get("sorting_result") or measurement.get("sorting_status"),
        "overall_confidence": measurement.get("overall_confidence", measurement.get("dimension_confidence")),
        "processing_time_ms": measurement.get("processing_time_ms"),
        "status": measurement.get("status"),
        "reason": measurement.get("reason") or measurement.get("volume_rejection_reason"),
        "model_version": measurement.get("model_version"),
        "pipeline_version": measurement.get("pipeline_version"),
        "radius_mm": shape.get("radius_mm"),
        "fit_confidence": shape.get("fit_confidence"),
    }


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _with_aliases(row: dict[str, Any]) -> dict[str, Any]:
    row["camera"] = row.get("camera_source")
    row["sorting"] = row.get("sorting_result")
    row["diameter_mm"] = row.get("cylinder_diameter_mm")
    row["volume_liters"] = row.get("selected_volume_litres")
    row["confidence"] = row.get("overall_confidence")
    row["rejection_reason"] = row.get("reason")
    return row


def _with_errors(row: dict[str, Any]) -> dict[str, Any]:
    truth, estimate = row.get("ground_truth_volume_litres"), row.get("selected_volume_litres")
    row["absolute_error_litres"] = row["percentage_error"] = None
    try:
        truth_value, estimate_value = float(truth), float(estimate)
    except (TypeError, ValueError):
        return row
    row["absolute_error_litres"] = round(abs(estimate_value - truth_value), 6)
    if truth_value:
        row["percentage_error"] = round(abs(estimate_value - truth_value) / abs(truth_value) * 100.0, 4)
    return row


class PairedComparisonLog:
    """Groups finalised per-camera measurements into comparison events."""

    def __init__(self, directory: Path, *, window_seconds: float = 20.0,
                 session_id: str | None = None,
                 expected_cameras: tuple[str, ...] = CAMERAS) -> None:
        self.directory = Path(directory).resolve()
        self.path = self.directory / CSV_NAME
        self.ground_truth_path = self.directory / GROUND_TRUTH_NAME
        self.window_seconds = float(window_seconds)
        # Research mode: both cameras (paired) or one camera alone. Only an
        # expected camera can be reported missing.
        self.expected_cameras = tuple(expected_cameras)
        self.session_id = session_id or "unknown-session"
        self._lock = threading.RLock()
        self._open: list[_OpenEvent] = []
        self._seen: set[str] = set()
        self._failures: list[dict[str, Any]] = []
        self._last_write_at: float | None = None
        self._last_error: str | None = None
        self._ground_truth: dict[str, dict[str, Any]] = self._load_ground_truth()
        for row in self._rotate_outdated_header():
            if row.get("measurement_id"):
                self._seen.add(str(row["measurement_id"]))
        for row in self.rows():
            if row.get("measurement_id"):
                self._seen.add(str(row["measurement_id"]))
        LOGGER.info("Comparison CSV: %s (%d rows)", self.path, len(self._seen))

    def _rotate_outdated_header(self) -> list[dict[str, Any]]:
        """Keep an older-schema file intact beside a fresh one.

        Appending rows under a different header would misalign every column
        in Excel. The old file is renamed, never rewritten, so previous
        sessions stay readable; its ids still count for idempotency.
        """
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding=CSV_ENCODING, newline="") as source:
                header = next(csv.reader(source), None)
        except OSError:
            return []
        if header is None or header == COLUMNS:
            return []
        rows = self.rows()
        archived = self.path.with_name(f"{self.path.stem}.before-{int(time.time())}{self.path.suffix}")
        self.path.replace(archived)
        LOGGER.warning("Comparison CSV schema changed; previous file kept as %s", archived)
        return rows

    # ------------------------------------------------------------ pairing
    def record_measurement(self, measurement: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        """Assign a finalised camera measurement to a comparison event and persist it."""
        measurement_id = str(measurement.get("event_id") or "").strip()
        camera = str(measurement.get("camera_source") or measurement.get("camera_id") or "")
        if not measurement_id or camera not in self.expected_cameras:
            return {"written": False, "error": "measurement_id and camera_source are required"}
        now = time.time() if now is None else float(now)
        with self._lock:
            self.flush_expired(now=now)
            if measurement_id in self._seen:
                return {"written": False, "duplicate": True, "measurement_id": measurement_id}
            family = _object_family(measurement.get("object_type") or measurement.get("label"))
            candidates = [
                event for event in self._open
                if camera not in event.cameras and _compatible(event.object_family, family)
                and now - event.opened_at <= self.window_seconds
            ]
            if candidates:
                event = min(candidates, key=lambda item: now - item.opened_at)
            else:
                event = _OpenEvent(f"cmp-{uuid4().hex[:12]}", now, family)
                self._open.append(event)
            event.cameras[camera] = measurement_id
            if event.object_family == "unknown":
                event.object_family = family
            row = comparison_row(measurement, event.comparison_event_id, self.session_id)
            written = self._write(row)
            if len(event.cameras) == len(self.expected_cameras):
                self._open.remove(event)
            return {"written": written, "comparison_event_id": event.comparison_event_id,
                    "measurement_id": measurement_id}

    def flush_expired(self, *, now: float | None = None) -> int:
        """Close events whose window passed, writing one `missing` row per absent camera."""
        now = time.time() if now is None else float(now)
        written = 0
        with self._lock:
            for event in [item for item in self._open if now - item.opened_at > self.window_seconds]:
                self._open.remove(event)
                for camera in self.expected_cameras:
                    if camera in event.cameras:
                        continue
                    written += int(self._write({
                        "session_id": self.session_id,
                        "comparison_event_id": event.comparison_event_id,
                        "measurement_id": f"{event.comparison_event_id}-{camera}-missing",
                        "camera_source": camera,
                        "timestamp": event.opened_at + self.window_seconds,
                        "status": STATUS_MISSING,
                        "reason": "no_finalised_measurement_within_pairing_window",
                    }))
        return written

    # ------------------------------------------------------------ writing
    def _write(self, row: dict[str, Any]) -> bool:
        measurement_id = str(row["measurement_id"])
        if measurement_id in self._seen:
            return False
        row = _with_aliases(_with_errors({**row, **self._truth_columns(row.get("comparison_event_id"))}))
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.is_file() or self.path.stat().st_size == 0
            with self.path.open("a", encoding=CSV_ENCODING, newline="") as output:
                writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({key: row.get(key) for key in COLUMNS})
                output.flush()
                os.fsync(output.fileno())
        except OSError as exc:
            # Never raised into the measurement thread: queued and retried.
            self._last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Comparison CSV write failed at %s", self.path)
            if all(item["measurement_id"] != measurement_id for item in self._failures):
                self._failures.append(row)
            LOGGER.error("Comparison row %s not persisted: %s", measurement_id, self._last_error)
            return False
        self._seen.add(measurement_id)
        self._last_write_at = time.time()
        LOGGER.info("Comparison CSV row %s written to %s (%d rows)", measurement_id, self.path, len(self._seen))
        return True

    def retry_failed(self) -> dict[str, Any]:
        with self._lock:
            pending, self._failures = self._failures, []
            recovered = sum(1 for row in pending if self._write(row))
            if not self._failures:
                self._last_error = None
            return {"recovered": recovered, "still_failing": len(self._failures)}

    # ------------------------------------------------------- ground truth
    def _load_ground_truth(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self.ground_truth_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def _truth_columns(self, comparison_event_id: Any) -> dict[str, Any]:
        truth = self._ground_truth.get(str(comparison_event_id)) or {}
        if not truth:
            return {}
        return {
            "ground_truth_volume_litres": truth.get("reference_volume_litres"),
            "ground_truth_method": truth.get("ground_truth_method"),
            "dataset_split": truth.get("dataset_split", "evaluation"),
        }

    def set_ground_truth(self, comparison_event_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Attach operator-entered ground truth to every row of one comparison event."""
        method = values.get("ground_truth_method")
        if method not in GROUND_TRUTH_METHODS:
            raise ValueError(f"ground_truth_method must be one of {', '.join(GROUND_TRUTH_METHODS)}")
        split = values.get("dataset_split", "evaluation")
        if split not in ("evaluation", "calibration"):
            raise ValueError("dataset_split must be evaluation or calibration")
        truth = {key: values.get(key) for key in GROUND_TRUTH_FIELDS if values.get(key) is not None}
        truth["dataset_split"] = split
        if method == "manufacturer_capacity":
            # A nominal capacity is not the container's external occupied volume.
            truth["note"] = "manufacturer nominal capacity, not external occupied volume"
        with self._lock:
            rows = self.rows()
            if not any(row.get("comparison_event_id") == comparison_event_id for row in rows):
                raise KeyError(comparison_event_id)
            self._ground_truth[comparison_event_id] = truth
            temporary = self.ground_truth_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._ground_truth, indent=2), encoding="utf-8")
            temporary.replace(self.ground_truth_path)
            for row in rows:
                if row.get("comparison_event_id") == comparison_event_id:
                    row.update(self._truth_columns(comparison_event_id))
                    _with_errors(row)
            self._rewrite(rows)
        return {"comparison_event_id": comparison_event_id, "ground_truth": truth}

    def _rewrite(self, rows: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding=CSV_ENCODING, newline="") as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in COLUMNS})
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)

    # ------------------------------------------------------------ reading
    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding=CSV_ENCODING, newline="") as source:
                return list(csv.DictReader(source))
        except (OSError, csv.Error):
            return []

    def csv_text(self) -> str:
        """The latest snapshot of the canonical file, header-only when empty."""
        with self._lock:
            self.flush_expired()
            if self.path.is_file():
                return self.path.read_text(encoding=CSV_ENCODING)
        output = io.StringIO()
        csv.DictWriter(output, fieldnames=COLUMNS).writeheader()
        return output.getvalue()

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Side-by-side view for the dashboard: newest comparison events first."""
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            event = grouped.setdefault(row["comparison_event_id"], {
                "comparison_event_id": row["comparison_event_id"],
                "ground_truth": self._ground_truth.get(row["comparison_event_id"]),
                "cameras": {},
            })
            event["cameras"][row.get("camera_source") or "unknown"] = row
        events = list(grouped.values())[-max(1, limit):]
        for event in events:
            realsense, logitech = event["cameras"].get("realsense"), event["cameras"].get("logitech")
            event["differences"] = self._differences(realsense, logitech)
        return list(reversed(events))

    @staticmethod
    def _differences(realsense: dict[str, Any] | None, logitech: dict[str, Any] | None) -> dict[str, Any]:
        def _number(row: dict[str, Any] | None, key: str) -> float | None:
            try:
                return None if row is None or row.get(key) in (None, "") else float(row[key])
            except (TypeError, ValueError):
                return None

        output: dict[str, Any] = {}
        for key in ("length_mm", "width_mm", "height_mm", "selected_volume_litres", "processing_time_ms"):
            left, right = _number(realsense, key), _number(logitech, key)
            output[key] = None if left is None or right is None else round(right - left, 4)
        return output

    def status(self) -> dict[str, Any]:
        rows = self.rows()
        return {
            "csv_path": str(self.path),
            "recorded_measurements": sum(1 for row in rows if row.get("status") != STATUS_MISSING),
            "realsense_rows": sum(1 for row in rows if row.get("camera_source") == "realsense"),
            "logitech_rows": sum(1 for row in rows if row.get("camera_source") == "logitech"),
            "missing_rows": sum(1 for row in rows if row.get("status") == STATUS_MISSING),
            "comparison_events": len({row.get("comparison_event_id") for row in rows}),
            "open_events": len(self._open),
            "last_csv_write": self._last_write_at,
            "persistence_status": "healthy" if not self._failures else "retry_pending",
            "pending_retries": len(self._failures),
            "last_error": self._last_error,
            "snapshot_note": "A downloaded CSV is a snapshot; download again for the newest rows.",
        }
