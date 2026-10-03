"""Per-attempt experiment log: every measurement attempt, including every failure.

This is not the object ledger (one row per tracked object, updated as it
settles) and not the frame log. One row here is one deliberate trial: the
operator places a reference object, states what it really is, and the system
records what each camera produced within a time limit -- a measurement, or the
reason there was none (detection failure, missing depth, missing calibration,
unstable or rejected geometry, a stale/predicted reading, a timeout, an
ambiguous scene).

Evaluation is computed over ALL attempts of a run: accuracy figures are only
over valid measurements, and are always reported next to the failure rate, so
accepted-reading accuracy is never presented as overall performance.
Calibration objects are kept apart from validation objects, and a run freezes
the configuration it was started with (a changed configuration is flagged).
Template-derived volumes are excluded from geometric-accuracy figures.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from . import __version__

ATTEMPT_FILE = "attempts.jsonl"
RUN_FILE = "runs.json"
ROLES = ("calibration", "validation")
OUTCOMES = (
    "success", "detection_failure", "missing_depth", "missing_calibration", "unstable_measurement",
    "rejected_geometry", "stale_or_predicted", "timeout", "ambiguous_multiple_objects", "no_volume",
)
FIELDS = [
    "attempt_id", "run_id", "role", "camera", "outcome", "reason", "recorded_at",
    "object_id", "object_type", "reference_length_mm", "reference_width_mm", "reference_height_mm",
    "reference_volume_l", "reference_volume_definition", "reference_source",
    "actual_colour", "actual_material", "expected_count",
    "detected_label", "resolved_label", "detector_score", "track_id", "observation_status", "origin_source",
    "colour", "colour_share", "material", "material_model_score", "material_label_agreement", "material_samples",
    "reported_volume_l", "raw_geometric_volume_l", "calibration_factor", "volume_relationship",
    "measured_length_mm", "measured_width_mm", "measured_height_mm",
    "uncertainty_l", "uncertainty_method", "depth_coverage_percent", "measurement_method", "measurement_quality",
    "template_volume_used", "template_id", "snapped_to_class",
    "capture_timestamp", "processed_at", "frame_id", "age_s", "objects_in_view", "counted_objects",
    "evidence_files", "code_version", "config_hash", "config_changed_during_run", "notes",
]
_SECRET_WORDS = ("token", "password", "secret", "key", "credential")


def code_version() -> str:
    """The deployed source commit (telemetry.code_commit), else the package version."""
    from .telemetry import code_commit

    return code_commit() or __version__


def config_snapshot(config: Any) -> dict[str, Any]:
    """Measurement-relevant configuration, secrets removed, JSON-safe."""
    raw = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(vars(config))
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if any(word in key.lower() for word in _SECRET_WORDS):
            continue
        clean[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    return clean


def config_hash(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()[:16]


def classify_outcome(detection: Any, volume: float | None) -> tuple[str, str]:
    """Outcome category and its reason for one camera's chosen detection."""
    if detection is None:
        return "detection_failure", "no fresh, semantically confirmed object in view"
    if getattr(detection, "observation_status", "fresh") != "fresh":
        return "stale_or_predicted", f"reading is {detection.observation_status}, not a fresh measurement"
    if volume is not None:
        return "success", ""
    reason = str(getattr(detection, "volume_rejection_reason", None)
                 or getattr(detection, "measurement_quality", None) or "no volume reported")
    text = reason.lower()
    if "depth" in text or "coverage" in text or "sparse" in text:
        return "missing_depth", reason
    if any(word in text for word in ("calib", "baseline", "reference", "scale", "distance", "camera-moved")):
        return "missing_calibration", reason
    if "stabiliz" in text or "unstable" in text:
        return "unstable_measurement", reason
    if "reject" in text or "implausible" in text or "background" in text or "withheld" in text:
        return "rejected_geometry", reason
    return "no_volume", reason


class ExperimentLog:
    def __init__(self, root: Path, *, evidence: bool = False, max_evidence_files: int = 200,
                 version: str | None = None) -> None:
        self.root = Path(root) / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_enabled = bool(evidence)
        self.max_evidence_files = max(0, int(max_evidence_files))
        self.version = version or code_version()
        self.lock = threading.RLock()
        self.active_run: dict[str, Any] | None = None
        runs = self._read_runs()
        self.active_run = next((run for run in runs if run.get("stopped_at") is None), None)

    # ------------------------------------------------------------------ runs
    def _read_runs(self) -> list[dict[str, Any]]:
        try:
            return json.loads((self.root / RUN_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _write_runs(self, runs: list[dict[str, Any]]) -> None:
        temporary = self.root / (RUN_FILE + ".tmp")
        temporary.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        temporary.replace(self.root / RUN_FILE)

    def start_run(self, name: str, config: Any, notes: str = "") -> dict[str, Any]:
        with self.lock:
            if self.active_run is not None:
                raise ValueError(f"Run {self.active_run['run_id']} is still active; stop it first")
            snapshot = config_snapshot(config)
            run = {"run_id": uuid4().hex[:10], "name": name.strip()[:80] or "evaluation", "notes": notes[:500],
                   "started_at": time.time(), "stopped_at": None, "code_version": self.version,
                   "config_hash": config_hash(snapshot), "config": snapshot}
            self._write_runs(self._read_runs() + [run])
            self.active_run = run
            return run

    def stop_run(self) -> dict[str, Any]:
        with self.lock:
            if self.active_run is None:
                raise ValueError("No evaluation run is active")
            runs = self._read_runs()
            for run in runs:
                if run["run_id"] == self.active_run["run_id"]:
                    run["stopped_at"] = time.time()
                    stopped = run
            self._write_runs(runs)
            self.active_run = None
            return stopped

    def runs(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in run.items() if k != "config"} for run in self._read_runs()]

    # -------------------------------------------------------------- attempts
    def record(self, attempt: dict[str, Any], config: Any) -> dict[str, Any]:
        with self.lock:
            current = config_hash(config_snapshot(config))
            run = self.active_run
            row = {field: attempt.get(field) for field in FIELDS}
            row.update(
                attempt_id=attempt.get("attempt_id") or uuid4().hex[:12],
                run_id=None if run is None else run["run_id"],
                recorded_at=attempt.get("recorded_at") or time.time(),
                code_version=self.version, config_hash=current,
                config_changed_during_run=bool(run is not None and run["config_hash"] != current),
            )
            if row["role"] not in ROLES:
                raise ValueError("role must be 'calibration' or 'validation'")
            if row["outcome"] not in OUTCOMES:
                raise ValueError(f"unknown outcome {row['outcome']!r}")
            with (self.root / ATTEMPT_FILE).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=_json_default) + "\n")
            return row

    def attempts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for line in (self.root / ATTEMPT_FILE).read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if run_id is None or row.get("run_id") == run_id:
                    rows.append(row)
        except OSError:
            pass
        return rows

    def csv(self, run_id: str | None = None) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in self.attempts(run_id):
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
        return output.getvalue()

    # -------------------------------------------------------------- evidence
    def save_evidence(self, attempt_id: str, camera: str, frame: np.ndarray | None,
                      depth_m: np.ndarray | None) -> list[str]:
        """Optional RGB (JPEG) + depth (16-bit PNG, millimetres), oldest pruned."""
        if not self.evidence_enabled or self.max_evidence_files <= 0:
            return []
        try:
            import cv2
        except ImportError:
            return []
        folder = self.root / "evidence"
        folder.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        if frame is not None:
            path = folder / f"{attempt_id}_{camera}_rgb.jpg"
            if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                saved.append(path.name)
        if depth_m is not None:
            millimetres = np.nan_to_num(np.asarray(depth_m, dtype=np.float64) * 1000.0, nan=0.0)
            path = folder / f"{attempt_id}_{camera}_depth_mm.png"
            if cv2.imwrite(str(path), np.clip(millimetres, 0, 65535).astype(np.uint16)):
                saved.append(path.name)
        files = sorted(folder.iterdir(), key=lambda item: item.stat().st_mtime)
        for stale in files[: max(0, len(files) - self.max_evidence_files)]:
            stale.unlink(missing_ok=True)
        return saved


def evaluate(rows: Iterable[dict[str, Any]], *, extra_calibration_objects: Iterable[str] = ()) -> dict[str, Any]:
    """Run-level evaluation over ALL attempts, per camera, calibration kept apart."""
    rows = list(rows)
    calibration_objects = {str(r.get("object_id") or "").lower() for r in rows if r.get("role") == "calibration"}
    calibration_objects |= {str(name).lower() for name in extra_calibration_objects if name}
    calibration_objects.discard("")
    validation = [r for r in rows if r.get("role") == "validation"]
    contaminated = sorted({str(r.get("object_id") or "").lower() for r in validation}
                          & calibration_objects)
    usable = [r for r in validation if str(r.get("object_id") or "").lower() not in calibration_objects]
    cameras: dict[str, Any] = {}
    for camera in sorted({r.get("camera") for r in usable if r.get("camera")}):
        mine = [r for r in usable if r.get("camera") == camera]
        outcomes = Counter(r.get("outcome") for r in mine)
        valid = [r for r in mine if r.get("outcome") == "success"]
        geometric = [r for r in valid if not r.get("template_volume_used") and not r.get("snapped_to_class")
                     and _number(r.get("reference_volume_l")) and _number(r.get("reported_volume_l")) is not None]
        errors = [float(r["reported_volume_l"]) - float(r["reference_volume_l"]) for r in geometric]
        refs = [float(r["reference_volume_l"]) for r in geometric]
        colour = [r for r in valid if r.get("actual_colour")]
        material = [r for r in valid if r.get("actual_material")]
        counted = [r for r in mine if r.get("expected_count") is not None and r.get("objects_in_view") is not None]
        cameras[camera] = {
            "attempts": len(mine),
            "valid_measurements": len(valid),
            "failure_rate": None if not mine else round(1 - len(valid) / len(mine), 4),
            "timeout_rate": None if not mine else round(outcomes.get("timeout", 0) / len(mine), 4),
            "outcomes": dict(outcomes),
            "volume": _error_metrics(errors, refs),
            "volume_basis": "valid measurements only; template-derived and class-snapped values excluded",
            "excluded_template_or_snapped": len(valid) - len(geometric),
            "colour_correct": _share(colour, lambda r: str(r.get("colour", "")).lower() == str(r["actual_colour"]).lower()),
            "material_correct": _share(material, lambda r: str(r.get("material", "")).lower() == str(r["actual_material"]).lower()),
            "count_errors": {
                "n": len(counted),
                "mean_absolute": None if not counted else round(statistics_mean(
                    abs(int(r["objects_in_view"]) - int(r["expected_count"])) for r in counted), 4),
            },
        }
    config_hashes = {r.get("config_hash") for r in validation}
    return {
        "total_attempts": len(rows),
        "validation_attempts": len(validation),
        "calibration_attempts": len(rows) - len(validation),
        "excluded_validation_attempts_on_calibration_objects": len(validation) - len(usable),
        "objects_used_for_both_calibration_and_validation": contaminated,
        "configuration_consistent": len(config_hashes) <= 1 and not any(
            r.get("config_changed_during_run") for r in validation),
        "cameras": cameras,
        "note": "Per-camera results are independent; nothing here is fused. Accuracy needs reference "
                "volumes measured independently of the cameras.",
    }


def _error_metrics(errors: list[float], references: list[float]) -> dict[str, Any]:
    if not errors:
        return {"n": 0, "mae_l": None, "rmse_l": None, "bias_l": None, "mape_percent": None}
    array = np.asarray(errors, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    return {
        "n": int(array.size),
        "mae_l": round(float(np.mean(np.abs(array))), 6),
        "rmse_l": round(float(np.sqrt(np.mean(array * array))), 6),
        "bias_l": round(float(np.mean(array)), 6),
        "mape_percent": round(float(np.mean(np.abs(array) / refs) * 100.0), 4),
    }


def _share(rows: list[dict[str, Any]], correct) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "share": None}
    hits = sum(1 for row in rows if correct(row))
    return {"n": len(rows), "correct": hits, "share": round(hits / len(rows), 4)}


def statistics_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def collect_attempt(station: Any, camera: str, *, requested_at: float, timeout_s: float,
                    clock=time.time, sleep=time.sleep) -> dict[str, Any]:
    """Wait (bounded) for a frame processed after `requested_at`; describe what it produced.

    Polls until a fresh measurement exists or the time limit passes. The last
    outcome seen is what gets recorded, so an object that never stops
    stabilising is logged as unstable, and no new frame at all as a timeout.
    """
    deadline = requested_at + max(0.5, float(timeout_s))
    result: dict[str, Any] = {"camera": camera, "outcome": "timeout",
                              "reason": f"no frame processed within {timeout_s:.1f} s"}
    while True:
        processed_at = getattr(station, "last_frame_processed_at", None) or 0.0
        analysis = getattr(station, "latest_analysis", None)
        if analysis is not None and processed_at > requested_at:
            result = describe_analysis(analysis, camera, processed_at)
            result["_frame"] = getattr(station, "latest_processed_frame", None)
            result["_depth"] = (getattr(station, "latest_depth", None) if camera == "realsense"
                                else getattr(station, "latest_monocular_depth", None))
            if result["outcome"] == "success":
                return result
        if clock() >= deadline:
            return result
        sleep(0.1)


def describe_analysis(analysis: Any, camera: str, processed_at: float) -> dict[str, Any]:
    from .geometry import is_phantom_detection

    fresh = [item for item in analysis.detections
             if not is_phantom_detection(item) and item.accepted_class is not None]
    current = [item for item in fresh if getattr(item, "observation_status", "fresh") == "fresh"]
    base = {"camera": camera, "objects_in_view": len(current), "capture_timestamp": float(analysis.timestamp),
            "processed_at": float(processed_at)}
    if len(current) > 1:
        return {**base, "outcome": "ambiguous_multiple_objects",
                "reason": f"{len(current)} objects in view; a trial needs exactly one"}
    chosen = current[0] if current else (fresh[0] if fresh else None)
    volume = None if chosen is None else (
        chosen.realsense_volume_l if camera == "realsense" else chosen.monocular_volume_l)
    outcome, reason = classify_outcome(chosen, volume)
    if chosen is None:
        return {**base, "outcome": outcome, "reason": reason}
    shape = getattr(chosen, "to_dict", lambda: {})()
    return {
        **base, "outcome": outcome, "reason": reason,
        "detected_label": chosen.label, "resolved_label": chosen.resolved_label,
        "detector_score": round(float(chosen.confidence), 4), "track_id": chosen.track_id,
        "observation_status": chosen.observation_status, "origin_source": chosen.origin_source or chosen.source,
        "colour": chosen.color, "colour_share": round(float(chosen.color_confidence), 4),
        "material": chosen.material, "material_model_score": chosen.material_model_score,
        "material_label_agreement": chosen.material_label_agreement, "material_samples": chosen.material_samples,
        "reported_volume_l": volume, "raw_geometric_volume_l": chosen.volume_raw_geometric_l,
        "calibration_factor": chosen.volume_calibration_factor, "volume_relationship": chosen.volume_relationship,
        "measured_length_mm": chosen.footprint_length_mm, "measured_width_mm": chosen.footprint_width_mm,
        "measured_height_mm": chosen.physical_height_mm,
        "uncertainty_l": chosen.volume_uncertainty_l, "uncertainty_method": chosen.uncertainty_method,
        "depth_coverage_percent": chosen.depth_coverage_percent,
        "measurement_method": chosen.measurement_method, "measurement_quality": chosen.measurement_quality,
        "template_volume_used": bool(chosen.box_template_volume_used), "template_id": chosen.box_template_id,
        "snapped_to_class": False,
        "frame_id": chosen.frame_id,
        "age_s": (shape.get("provenance") or {}).get("age_s"),
    }
