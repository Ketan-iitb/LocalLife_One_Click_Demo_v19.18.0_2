"""Independent camera stations and reproducible paired thesis measurements."""

from __future__ import annotations

import json
import logging
import math
import shutil
import threading
import time
from collections import Counter
from dataclasses import replace
from typing import Any
from uuid import uuid4

import numpy as np

from . import __version__

from .accuracy import fuse_pair_volume
from .config import AppConfig
from .geometry import fixed_bin_mask, is_phantom_source
from .inference import MetricDepthEstimator, create_segmenter
from .ledger import waste_object_type
from .paired_events import PairedComparisonLog
from .pipeline import VisionPipeline, filter_waste_detections
from .storage import ResultStore


LOGGER = logging.getLogger(__name__)

CAMERA_IDS = ("realsense", "logitech")
REFERENCE_FILE = "reference_trials.jsonl"


def infer_camera_id(source: str, requested: str | None = None) -> str:
    if requested:
        normalized = requested.strip().lower()
        if normalized not in CAMERA_IDS:
            raise ValueError("Camera ID must be realsense or logitech")
        return normalized
    normalized = source.strip().lower()
    if any(label in normalized for label in ("logitech", "c920", "webcam", "video:")):
        return "logitech"
    return "realsense"


def _event_time(record: dict[str, Any]) -> float:
    return float(record.get("deposited_at") or record.get("observed_at") or 0.0)


def match_deposits(
    realsense_records: list[dict[str, Any]],
    logitech_records: list[dict[str, Any]],
    *,
    tolerance_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    """One-to-one type-aware temporal matching; matching is not ground truth."""
    hardware = sorted(
        (record for record in realsense_records if record.get("status") == "deposited"),
        key=_event_time,
    )
    monocular = [record for record in logitech_records if record.get("status") == "deposited"]
    used: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for left in hardware:
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for right in monocular:
            right_id = str(right.get("entry_id", ""))
            if right_id in used or right.get("object_type") != left.get("object_type"):
                continue
            gap = abs(_event_time(left) - _event_time(right))
            if gap > tolerance_seconds:
                continue
            color_penalty = 0.5 if left.get("color") != right.get("color") else 0.0
            candidates.append((gap + color_penalty, gap, right))
        if not candidates:
            continue
        _, gap, right = min(candidates, key=lambda item: item[0])
        used.add(str(right["entry_id"]))
        left_liters, right_liters = left.get("volume_l"), right.get("volume_l")
        difference = (
            None if left_liters is None or right_liters is None
            else float(right_liters) - float(left_liters)
        )
        pairs.append({
            "pair_id": f"{left['entry_id']}__{right['entry_id']}",
            "realsense_entry_id": left["entry_id"],
            "logitech_entry_id": right["entry_id"],
            "object_type": left["object_type"],
            "realsense_color": left.get("color"),
            "logitech_color": right.get("color"),
            "color_agreement": left.get("color") == right.get("color"),
            "realsense_volume_l": left_liters,
            "logitech_volume_l": right_liters,
            "realsense_measurement_method": left.get("measurement_method"),
            "logitech_measurement_method": right.get("measurement_method"),
            "realsense_calibration_mode": left.get("calibration_mode"),
            "logitech_calibration_mode": right.get("calibration_mode"),
            "difference_l": None if difference is None else round(difference, 6),
            "absolute_difference_l": None if difference is None else round(abs(difference), 6),
            "difference_percent": (
                None if difference is None or not left_liters
                else round(difference / float(left_liters) * 100.0, 4)
            ),
            "timestamp_gap_seconds": round(gap, 4),
            "matched_at": max(_event_time(left), _event_time(right)),
            "match_basis": "object-type-and-timestamp",
        })
        # Operational fusion (ported from Accuracy Deployment v3.0): one
        # weighted figure per matched physical drop, plus the colour the
        # operator page shows. Raw per-camera values above are unchanged --
        # fusion is an operational reading, never ground truth.
        pair = pairs[-1]
        fused = fuse_pair_volume(left, right)
        pair["fused_volume_l"] = fused["volume_l"]
        pair["fusion_reliability"] = fused["reliability"]
        pair["realsense_weight"] = fused["realsense_weight"]
        pair["logitech_weight"] = fused["logitech_weight"]
        if pair["color_agreement"]:
            pair["operational_color"] = pair["realsense_color"]
        else:
            left_confidence = float(left.get("color_confidence") or 0.0)
            right_confidence = float(right.get("color_confidence") or 0.0)
            pair["operational_color"] = (
                pair["realsense_color"] if left_confidence >= right_confidence
                else pair["logitech_color"]
            )
    return pairs


def _error_metrics(errors: list[float], references: list[float]) -> dict[str, Any]:
    if not errors:
        return {"samples": 0, "mae_l": None, "rmse_l": None, "bias_l": None, "mape_percent": None}
    values = np.asarray(errors, dtype=np.float64)
    truth = np.asarray(references, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mae_l": round(float(np.mean(np.abs(values))), 6),
        "rmse_l": round(float(np.sqrt(np.mean(values * values))), 6),
        "bias_l": round(float(np.mean(values)), 6),
        "mape_percent": round(float(np.mean(np.abs(values / truth)) * 100), 4),
    }


class DualCameraCoordinator:
    """Two isolated ledgers and baselines sharing only cloud inference models."""

    def __init__(
        self,
        config: AppConfig,
        *,
        detector: Any | None = None,
        depth_estimator: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.inference_lock = threading.RLock()
        shared_detector = detector if detector is not None else create_segmenter(config)
        shared_depth = depth_estimator
        if shared_depth is None and config.enable_monocular_depth:
            shared_depth = MetricDepthEstimator(config, getattr(shared_detector, "device", None))
        hardware = replace(config, results_dir=config.results_dir / "realsense", enable_monocular_depth=False)
        webcam = replace(config, results_dir=config.results_dir / "logitech",
                         roi=config.logitech_roi if config.logitech_roi is not None else config.roi)
        self.pipelines = {
            "realsense": VisionPipeline(
                hardware, detector=shared_detector, camera_id="realsense", inference_lock=self.inference_lock
            ),
            "logitech": VisionPipeline(
                webcam, detector=shared_detector, depth_estimator=shared_depth,
                camera_id="logitech", inference_lock=self.inference_lock,
            ),
        }
        self._recent_semantic_presence: dict[str, float] = {}
        self._recent_semantic_box_presence: dict[str, float] = {}
        self.store = ResultStore(config.results_dir / "comparison")
        expected = {
            "paired": ("realsense", "logitech"),
            "realsense_only": ("realsense",),
            "logitech_only": ("logitech",),
        }[config.research_mode]
        # Canonical long-format comparison CSV: one row per finalised camera
        # measurement, grouped by physical object (paired_events.py).
        self.paired_log = PairedComparisonLog(
            config.results_dir / "comparison",
            window_seconds=config.comparison_pair_window_s,
            expected_cameras=expected,
        )
        self.attach_paired_listeners()
        # Each station knows the other's geometry only so it can refuse to
        # measure with it (coordinates.frame_consistency).
        self.pipelines["logitech"].peer_intrinsics = self.pipelines["realsense"].latest_intrinsics
        # Recipe pipeline (pointcloud_volume.py / recipe_*.py): a separate,
        # additive result -- see config.recipe_enabled's own comment for why
        # it defaults off and why it is loaded/cached lazily rather than
        # eagerly like the shared YOLOE detector above.
        self._recipe_detector: Any | None = None
        self._recipe_material_classifier: Any | None = None
        self._recipe_fallback_material_classifier: Any | None = None
        self._recipe_config: Any | None = None
        self._recipe_lock = threading.Lock()
        self._recipe_cache: dict[str, Any] | None = None
        self._recipe_cache_at: float = 0.0
        self._recipe_import_error: str | None = None

    def set_waste_ledger(self, enabled: bool) -> dict[str, Any]:
        """Turn the waste ledger (the recorded measurement history) on or off.

        `operating_mode` is deliberately NOT touched. An earlier version of this
        switch flipped the mode to "waste", which also switched the classifier
        to strict waste mode -- where only plastic bags, paper bags and
        cardboard boxes are accepted -- so turning history on stopped the system
        detecting the very object being measured. Recording history and
        restricting what counts as waste are separate decisions.

        Each station holds its own `replace()`d copy of the config, so every one
        has to be set -- changing only the coordinator's copy would leave the
        pipelines running in the old mode.
        """
        self.config.ledger_enabled = enabled
        for pipeline in self.pipelines.values():
            pipeline.config.ledger_enabled = enabled
        return {
            "waste_ledger_enabled": enabled,
            "operating_mode": self.config.operating_mode,
        }

    def attach_paired_listeners(self) -> None:
        """Route each expected camera's persisted rows into the paired log.

        Called again whenever a station is swapped in after construction.
        """
        for camera_id in self.paired_log.expected_cameras:
            self.pipelines[camera_id].measurement_listener = self._record_paired_measurement

    def paired_comparison(self, limit: int = 20) -> dict[str, Any]:
        return {
            "research_mode": self.config.research_mode,
            "events": self.paired_log.events(limit),
            "status": self.paired_log.status(),
            "logitech_calibration": self.camera("logitech").logitech_calibration_status(),
            "logitech_stages": self.camera("logitech").stage_report(),
        }

    def _record_paired_measurement(self, row: dict[str, Any]) -> None:
        self.paired_log.session_id = self.pipelines["realsense"].event_log.session_id
        self.paired_log.record_measurement(row)

    def camera(self, camera_id: str) -> VisionPipeline:
        if camera_id not in self.pipelines:
            raise ValueError("Camera ID must be realsense or logitech")
        return self.pipelines[camera_id]

    def warmup(self) -> dict[str, Any]:
        hardware = self.camera("realsense")
        webcam = self.camera("logitech")
        with self.inference_lock:
            if hasattr(hardware.detector, "load"):
                hardware.detector.load()
            if webcam.depth_estimator is not None and hasattr(webcam.depth_estimator, "load"):
                try:
                    webcam.depth_estimator.load()
                except Exception as exc:  # noqa: BLE001
                    # RealSense must keep running; Logitech reports why it cannot measure.
                    LOGGER.warning("Depth Anything V2 (%s) failed to load: %s", webcam.config.depth_model, exc)
                    webcam.depth_load_error = f"{type(exc).__name__}: {exc}"
                    webcam.depth_estimator = None
        LOGGER.info(
            "Logitech inference: detector %s on %s, %d prompts, image size %s, confidence %.2f; "
            "Depth Anything V2 %s",
            webcam.config.detector_model, getattr(webcam.detector, "device", "?"), len(webcam.config.prompts),
            webcam.config.image_size, webcam.config.logitech_detector_confidence,
            webcam.config.depth_model if webcam.depth_estimator is not None else f"OFF ({webcam.depth_load_error})",
        )
        return {"shared_detector": hardware.config.detector_model,
                "logitech_depth_model": webcam.config.depth_model if webcam.depth_estimator else None,
                "runtime": getattr(hardware.detector, "runtime", {}), "cameras": list(CAMERA_IDS)}

    def process_packets(self, packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Batch the shared detector across the newest camera frames.

        Both cameras use the same large YOLOE model.  Running two independent
        one-frame calls wastes GPU launch time and makes the web result lag.
        This path takes at most one newest packet per camera, performs one
        detector batch, runs monocular depth only for Logitech, then assembles
        each station's independent measurement and ledger state.
        """
        camera_ids = [camera_id for camera_id in CAMERA_IDS if camera_id in packets]
        if not camera_ids:
            return {}
        packets = dict(packets)
        for camera_id in camera_ids:
            frame, intrinsics = self.camera(camera_id).prepare_input(
                packets[camera_id]["frame"], packets[camera_id].get("intrinsics"),
            )
            packets[camera_id] = {**packets[camera_id], "frame": frame, "intrinsics": intrinsics}
        frames = [packets[camera_id]["frame"] for camera_id in camera_ids]
        started = time.perf_counter()
        hardware = self.camera("realsense")
        webcam = self.camera("logitech")
        with self.inference_lock:
            if hasattr(hardware.detector, "detect_camera_batch"):
                detections_batch = hardware.detector.detect_camera_batch(dict(zip(camera_ids, frames, strict=True)))
            else:
                detections_batch = hardware.detector.detect_batch(frames)
            predicted_by_camera: dict[str, np.ndarray | None] = {
                camera_id: None for camera_id in camera_ids
            }
            if "logitech" in packets and webcam.depth_estimator is not None:
                predicted_by_camera["logitech"] = webcam.predict_depth(packets["logitech"]["frame"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(camera_ids)
        detections_by_camera = dict(zip(camera_ids, detections_batch, strict=True))
        semantic_presence: dict[str, bool] = {}
        # Per-camera "this camera's own neural detector confirmed a BOX-family
        # object this frame". Logitech is this rig's designated appearance and
        # object-type camera (it has no metric depth and never contributes
        # geometry), so RealSense uses this purely to decide *what kind of
        # object* it is measuring -- which is what opens the table-relative
        # cuboid geometry path in `pipeline._assemble`. Every millimetre of
        # the resulting measurement still comes exclusively from RealSense
        # depth; only the object-type call is shared.
        semantic_box_presence: dict[str, bool] = {}
        for camera_id in camera_ids:
            station = self.camera(camera_id)
            frame = packets[camera_id]["frame"]
            region = fixed_bin_mask(frame.shape, station.config.roi, station.config.bin_polygon)
            confirmed = filter_waste_detections(
                detections_by_camera[camera_id], frame.shape, region, station.config
            )
            semantic_presence[camera_id] = bool(confirmed)
            semantic_box_presence[camera_id] = any(
                waste_object_type(item.label) == "box" for item in confirmed
            )
            if semantic_presence[camera_id]:
                self._recent_semantic_presence[camera_id] = time.monotonic()
            if semantic_box_presence[camera_id]:
                self._recent_semantic_box_presence[camera_id] = time.monotonic()
        results: dict[str, Any] = {}
        for camera_id in camera_ids:
            detections = detections_by_camera[camera_id]
            packet = packets[camera_id]
            peer_id = "logitech" if camera_id == "realsense" else "realsense"
            peer_seen_at = self._recent_semantic_presence.get(peer_id)
            peer_present = semantic_presence.get(peer_id, False) or bool(
                peer_seen_at is not None and time.monotonic() - peer_seen_at <= 2.5
            )
            # Same short grace window as `peer_present` above: the two cameras
            # are inferred in one shared batch, but either detector can miss a
            # frame, and an object type does not change between frames the way
            # a detection can flicker.
            peer_box_seen_at = self._recent_semantic_box_presence.get(peer_id)
            peer_box = semantic_box_presence.get(peer_id, False) or bool(
                peer_box_seen_at is not None and time.monotonic() - peer_box_seen_at <= 2.5
            )
            if camera_id == "logitech":
                self.pipelines["logitech"].peer_intrinsics = self.pipelines["realsense"].latest_intrinsics
            try:
                results[camera_id] = self.camera(camera_id).process_precomputed(
                    packet["frame"],
                    detections=detections,
                    predicted_depth=predicted_by_camera[camera_id],
                    depth_m=packet.get("depth_m"),
                    intrinsics=packet.get("intrinsics"),
                    source=str(packet.get("source", "live")),
                    timestamp=packet.get("timestamp"),
                    inference_ms=elapsed_ms,
                    persist=bool(packet.get("persist", True)),
                    peer_bag_present=peer_present,
                    peer_box_present=peer_box,
                )
            except Exception:  # noqa: BLE001
                # One camera's failure must never take the other's frame with
                # it, and must never clear its tracks.
                LOGGER.exception("%s frame processing failed", camera_id)
                results[camera_id] = None
        return {key: value for key, value in results.items() if value is not None}

    def automatic_empty_setup(self, logitech_distance_m: float | None = None) -> dict[str, Any]:
        """Capture both reusable empty-scene profiles in one coordinated action."""
        hardware = self.camera("realsense")
        webcam = self.camera("logitech")
        if hardware.latest_frame is None or webcam.latest_frame is None:
            raise ValueError("Wait until both live camera images appear before automatic setup")
        if hardware.latest_depth is None or hardware.latest_intrinsics is None:
            raise ValueError("RealSense depth and factory intrinsics must be live before automatic setup")
        if webcam.latest_intrinsics is None:
            raise ValueError("Logitech lens field of view is missing from the live bridge")
        if logitech_distance_m is not None:
            if not np.isfinite(logitech_distance_m) or logitech_distance_m <= 0:
                raise ValueError("Logitech distance must be a finite positive number of meters")
            webcam.config.logitech_reference_distance_m = float(logitech_distance_m)
        baselines = {
            "realsense": hardware.set_baseline(),
            "logitech": webcam.set_baseline(),
        }
        if logitech_distance_m is not None:
            webcam.store.save_json(
                "calibration/reference_distance.json",
                {"distance_m": float(logitech_distance_m), "captured_at": time.time()},
            )
        return {
            "baselines": baselines,
            "logitech_distance_m": logitech_distance_m,
            "provisional_logitech": logitech_distance_m is None,
            "message": (
                "Both saved profiles are ready. Logitech uses the measured distance."
                if logitech_distance_m is not None
                else "Both saved profiles are ready. Logitech liters are provisional until a distance is entered."
            ),
        }

    @property
    def frames_processed(self) -> int:
        return sum(station.frames_processed for station in self.pipelines.values())

    def pairs(self) -> list[dict[str, Any]]:
        return match_deposits(
            self.camera("realsense").ledger.all_records(),
            self.camera("logitech").ledger.all_records(),
            tolerance_seconds=self.config.comparison_match_seconds,
        )

    def record_reference(
        self,
        known_liters: float,
        *,
        pair_id: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        if not math.isfinite(known_liters) or known_liters <= 0:
            raise ValueError("Reference volume must be a finite positive number of liters")
        pairs = self.pairs()
        if not pairs:
            raise ValueError("Deposit and match one object in both cameras before recording ground truth")
        pair = next((item for item in pairs if item["pair_id"] == pair_id), None) if pair_id else pairs[-1]
        if pair is None:
            raise ValueError("The requested comparison pair does not exist")
        if any(item.get("pair_id") == pair["pair_id"] for item in self.reference_trials()):
            raise ValueError("A physical reference has already been recorded for this object pair")
        record = {
            "trial_id": uuid4().hex[:12],
            "recorded_at": time.time(),
            "known_volume_l": float(known_liters),
            "notes": notes.strip(),
            **pair,
        }
        self.store.append_jsonl(REFERENCE_FILE, record)
        return record

    def reference_trials(self) -> list[dict[str, Any]]:
        return self.store.read_jsonl(REFERENCE_FILE)

    # ------------------------------------------------------- benchmark mode
    def benchmark_session(self, mode: str | None = None) -> "BenchmarkSession":
        """The collector for one processing mode, created on first use.

        Kept per mode rather than per run so a local session and a cloud
        session over the same recorded input can be compared directly.
        """
        from .benchmark import BenchmarkSession

        resolved = mode or self.config.processing_mode
        if not hasattr(self, "_benchmark_sessions"):
            self._benchmark_sessions: dict[str, BenchmarkSession] = {}
        if resolved not in self._benchmark_sessions:
            self._benchmark_sessions[resolved] = BenchmarkSession(
                self.operator_session()["session_id"], resolved,
                recorded_input_id=self.config.benchmark_input_id or None,
            )
        return self._benchmark_sessions[resolved]

    def benchmark_summary(self) -> dict[str, Any]:
        from .benchmark import compare

        sessions = getattr(self, "_benchmark_sessions", {})
        return {
            "enabled": self.config.benchmark_mode,
            "recorded_input_id": self.config.benchmark_input_id or None,
            **compare(sessions.get("local"), sessions.get("cloud")),
        }

    def benchmark_csv(self) -> str:
        """Every sampled frame from every mode, in one export."""
        import csv
        import io

        from .benchmark import BENCHMARK_COLUMNS

        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=BENCHMARK_COLUMNS, extrasaction="ignore",
        )
        writer.writeheader()
        for session in getattr(self, "_benchmark_sessions", {}).values():
            for row in session.rows():
                writer.writerow(row)
        return output.getvalue()

    def operator_session(self) -> dict[str, Any]:
        """The current operator session, created on first use and kept on disk."""
        location = self.config.results_dir / "operator_session.json"
        if location.is_file():
            try:
                payload = json.loads(location.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("session_id"):
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        payload = {"session_id": uuid4().hex[:12], "started_at": time.time()}
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def update_color_map(self, mapping: dict[str, str]) -> dict[str, Any]:
        """Set the operational colour-to-waste-stream labels.

        These are operator labels only: the colour each camera actually
        measured stays in the raw record untouched.
        """
        cleaned = {
            str(key).strip().lower(): str(value).strip()
            for key, value in mapping.items()
            if str(key).strip() and str(value).strip()
        }
        self.config.color_waste_streams = cleaned
        for station in self.pipelines.values():
            station.config.color_waste_streams = dict(cleaned)
            station.ledger.color_streams = dict(cleaned)
        payload = {
            "mapping": cleaned,
            "updated_at": time.time(),
            "note": "Mapping changes affect future operational interpretation; "
                    "raw detected color remains stored.",
        }
        self.store.save_json("color_map.json", payload)
        return payload

    def start_new_operator_session(self) -> dict[str, Any]:
        """Archive the finished session and start a clean one.

        Nothing is deleted: each camera's ledger and the reference trials are
        copied into results_dir/sessions/<stamp>-<id>/ before the live lists are
        cleared, so restarting a session never destroys earlier measurements.
        """
        root = self.config.results_dir
        previous = self.operator_session()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        archive = root / "sessions" / f"{stamp}-{previous['session_id']}"
        archive.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for camera_id in CAMERA_IDS:
            camera_results = self.camera(camera_id).config.results_dir
            for name, saved_as in (
                ("waste_plant_ledger.jsonl", f"{camera_id}_history.jsonl"),
                ("measurements.csv", f"{camera_id}_measurements.csv"),
            ):
                source = camera_results / name
                if source.is_file():
                    shutil.copy2(source, archive / saved_as)
                    copied.append(saved_as)
        references = self.store.directory / REFERENCE_FILE
        if references.is_file():
            shutil.copy2(references, archive / REFERENCE_FILE)
            copied.append(REFERENCE_FILE)
        (archive / "session.json").write_text(
            json.dumps({**previous, "ended_at": time.time(), "files": copied}, indent=2),
            encoding="utf-8",
        )
        resets = {camera_id: self.camera(camera_id).clear_history() for camera_id in CAMERA_IDS}
        if references.is_file():
            references.unlink()
        current = {
            "session_id": uuid4().hex[:12],
            "started_at": time.time(),
            "previous_session_archive": str(archive),
        }
        (root / "operator_session.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
        return {**current, "reset": resets}

    def comparison(self) -> dict[str, Any]:
        pairs = self.pairs()
        differences = [float(pair["difference_l"]) for pair in pairs if pair["difference_l"] is not None]
        trials = self.reference_trials()
        reference_values: list[float] = []
        hardware_errors: list[float] = []
        webcam_errors: list[float] = []
        for trial in trials:
            known = float(trial["known_volume_l"])
            if trial.get("realsense_volume_l") is None or trial.get("logitech_volume_l") is None:
                continue
            reference_values.append(known)
            hardware_errors.append(float(trial["realsense_volume_l"]) - known)
            webcam_errors.append(float(trial["logitech_volume_l"]) - known)
        hardware_total = self.camera("realsense").ledger.summary()["deposited_count"]
        webcam_total = self.camera("logitech").ledger.summary()["deposited_count"]
        operational_colors: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            color = str(pair.get("operational_color") or "unknown")
            item = operational_colors.setdefault(color, {
                "color": color, "count": 0, "volume_l": 0.0,
                "waste_stream": self.config.color_waste_streams.get(color.lower()),
            })
            item["count"] += 1
            item["volume_l"] += float(pair.get("fused_volume_l") or 0.0)
        operational = {
            "deposited_count": len(pairs),
            "cumulative_volume_l": round(
                sum(float(pair.get("fused_volume_l") or 0.0) for pair in pairs), 6
            ),
            "colors": sorted(
                ({**item, "volume_l": round(float(item["volume_l"]), 6)}
                 for item in operational_colors.values()),
                key=lambda item: (-item["count"], item["color"]),
            ),
            "color_map": dict(self.config.color_waste_streams),
            "note": "One matched physical drop is counted once; fusion is operational, not ground truth.",
        }
        return {
            "operational": operational,
            "paired_count": len(pairs),
            "realsense_unmatched": hardware_total - len(pairs),
            "logitech_unmatched": webcam_total - len(pairs),
            "mean_difference_l": None if not differences else round(float(np.mean(differences)), 6),
            "mean_absolute_difference_l": None if not differences else round(float(np.mean(np.abs(differences))), 6),
            "rmse_between_cameras_l": None if not differences else round(float(np.sqrt(np.mean(np.square(differences)))), 6),
            "color_agreement_percent": None if not pairs else round(
                sum(bool(pair["color_agreement"]) for pair in pairs) / len(pairs) * 100, 2
            ),
            "match_window_seconds": self.config.comparison_match_seconds,
            "accuracy_requires_independent_reference": True,
            "reference_trials": len(reference_values),
            "realsense_accuracy": _error_metrics(hardware_errors, reference_values),
            "logitech_accuracy": _error_metrics(webcam_errors, reference_values),
            "pairs": list(reversed(pairs[-self.config.history_limit :])),
        }

    def fused_result(self) -> dict[str, Any]:
        """Combine both cameras' current confirmed object into one reported result.

        The fixed installation holds exactly one physical bag or box in the
        measurement bin at a time, so RealSense's and Logitech's own current
        confirmed detections (when both have one) are observations of the
        *same* object. Appearance observations may be combined at scene level,
        but geometry is never blended: RealSense's factory-calibrated stereo
        depth is the sole physical-dimension and final-volume authority.
        Logitech's monocular estimate is retained only as a diagnostic
        comparison. Phantom (unconfirmed depth-silhouette) detections are
        excluded entirely.
        """
        candidates: dict[str, dict[str, Any]] = {}
        for camera_id in CAMERA_IDS:
            analysis = self.camera(camera_id).latest_analysis
            if analysis is None:
                continue
            confirmed = [
                item for item in analysis.detections
                if item.tracking_status in {"confirmed", "predicted"}
                and not is_phantom_source(item.source)
                and item.accepted_class is not None
            ]
            if not confirmed:
                continue
            best = max(confirmed, key=lambda item: item.area_pixels)
            volume = best.realsense_volume_l if camera_id == "realsense" else best.monocular_volume_l
            candidates[camera_id] = {
                "volume_l": volume,
                "uncertainty_l": best.volume_uncertainty_l,
                "color": best.color if best.color not in {"", "unknown"} else None,
                "material": best.material if best.material not in {"", "unknown"} else None,
                "material_confidence": float(best.material_confidence or 0.0),
                "label": best.label,
            }

        if not candidates:
            return {
                "available": False,
                "sources": [],
                "volume_l": None,
                "volume_source": None,
                "color": "unknown",
                "material": "unknown",
                "per_camera": {},
                "agreement_l": None,
                "agreement_percent": None,
                "message": (
                    "Waiting for a confirmed test object on either camera"
                    if self.config.operating_mode == "geometry_validation"
                    else "Waiting for a confirmed bag or box on either camera"
                ),
            }

        volumes = {
            camera_id: item["volume_l"] for camera_id, item in candidates.items()
            if item["volume_l"] is not None
        }
        fused_volume: float | None = None
        agreement_l: float | None = None
        agreement_percent: float | None = None
        if volumes:
            # Build-playbook invariant: RealSense is the sole authority for
            # geometry and liters.  Logitech may contribute colour/material
            # and an optional comparison reading, but must never move the
            # number presented as the fused/final volume.
            fused_volume = volumes.get("realsense")
            if len(volumes) == 2:
                difference = abs(volumes["realsense"] - volumes["logitech"])
                reference = max(volumes.values()) or 1.0
                agreement_l = round(difference, 3)
                agreement_percent = round(max(0.0, 1.0 - difference / reference) * 100.0, 1)

        colors = [item["color"] for item in candidates.values() if item["color"]]
        fused_color = Counter(colors).most_common(1)[0][0] if colors else "unknown"
        materials = [
            (item["material"], item["material_confidence"]) for item in candidates.values() if item["material"]
        ]
        fused_material = max(materials, key=lambda item: item[1])[0] if materials else "unknown"

        sources = sorted(candidates.keys())
        if len(sources) == 2:
            message = (
                "RealSense supplies final volume; both cameras contribute visual classification"
                if agreement_percent is None or agreement_percent >= 70.0
                else f"Both cameras see an object, but their volumes disagree by {agreement_l:.1f} L "
                f"({100 - agreement_percent:.0f}%); final volume remains RealSense-only"
            )
        else:
            message = (
                "Only Logitech currently has a confirmed object; waiting for RealSense geometry"
                if sources[0] == "logitech"
                else "Only realsense currently has a confirmed object; showing its measured volume"
            )

        return {
            "available": True,
            "sources": sources,
            "volume_l": None if fused_volume is None else round(fused_volume, 3),
            "volume_source": "realsense" if fused_volume is not None else None,
            "color": fused_color,
            "material": fused_material,
            "per_camera": {
                camera_id: {
                    "volume_l": item["volume_l"],
                    "color": item["color"] or "unknown",
                    "material": item["material"] or "unknown",
                    "label": item["label"],
                }
                for camera_id, item in candidates.items()
            },
            "agreement_l": agreement_l,
            "agreement_percent": agreement_percent,
            "message": message,
        }

    def recipe_result(self) -> dict[str, Any]:
        """Recipe Steps 2-6, run against whatever frames the two stations
        already hold -- entirely separate from, and never blocking or
        altering, the tracked/ledgered measurement above.

        Disabled by default (see config.recipe_enabled). When enabled, the
        result is cached and only recomputed at most once per
        config.recipe_refresh_seconds, since it runs its own YOLO
        segmentation and CLIP classification pass -- real inference work,
        not free to repeat on every ~550ms dashboard poll. Every failure
        mode here (missing optional dependency, no frame yet, an unexpected
        exception from the recipe modules themselves) degrades to an
        "available": False result rather than raising, so a problem in this
        additive feature can never take down /api/state for the dashboard's
        primary measurement.

        Runs the v3 blueprint's pipeline (`recipe_config.yaml`'s tuning
        knobs, the plane-fit box volume, the LAB color classifier, the
        CLIP(+fallback) material cascade) -- this dashboard-wired path only
        ever has a single RealSense view per poll, so box volume here is
        always the v3 §5.3 single-view fallback (flagged
        `single_view_volume`, confidence dampened); the two-view accuracy
        path is reachable through `recipe_pipeline.process_object()`
        directly by a caller that captures a second angle.
        """
        if not self.config.recipe_enabled:
            return {"available": False, "reason": "disabled"}

        now = time.time()
        if self._recipe_cache is not None and now - self._recipe_cache_at < self.config.recipe_refresh_seconds:
            return self._recipe_cache

        if not self._recipe_lock.acquire(blocking=False):
            # A refresh is already running on another thread (e.g. a second
            # concurrent /api/state poll); serve the last known result
            # rather than blocking this request on a multi-second YOLO/CLIP
            # pass.
            return self._recipe_cache or {"available": False, "reason": "computing"}

        try:
            if self._recipe_import_error is not None:
                return {"available": False, "reason": "unavailable", "detail": self._recipe_import_error}

            try:
                from .recipe_config import RecipeConfig
                from .recipe_detect import DEFAULT_RECIPE_DETECTOR_MODEL, RecipeDetector
                from .recipe_material import RecipeMaterialClassifier, RecipeMaterialFallbackClassifier
                from .recipe_pipeline import process_object
            except ImportError as exc:
                self._recipe_import_error = str(exc)
                LOGGER.warning("Recipe pipeline enabled but its dependencies are missing (%s)", exc)
                return {"available": False, "reason": "unavailable", "detail": str(exc)}

            hardware = self.camera("realsense")
            webcam = self.camera("logitech")
            if hardware.latest_frame is None or hardware.latest_depth is None or hardware.latest_intrinsics is None:
                result = {"available": False, "reason": "waiting_for_realsense_frame"}
                self._recipe_cache = result
                self._recipe_cache_at = now
                return result

            if self._recipe_config is None:
                self._recipe_config = RecipeConfig.from_yaml()
            if self._recipe_detector is None:
                self._recipe_detector = RecipeDetector(model_name=DEFAULT_RECIPE_DETECTOR_MODEL)
            if self._recipe_material_classifier is None:
                self._recipe_material_classifier = RecipeMaterialClassifier(self.config)
            if self._recipe_fallback_material_classifier is None:
                self._recipe_fallback_material_classifier = RecipeMaterialFallbackClassifier(
                    checkpoint_path=self._recipe_config.material.fallback_checkpoint_path,
                    backbone=self._recipe_config.material.fallback_model,
                )

            try:
                payload = process_object(
                    hardware.latest_frame,
                    hardware.latest_depth,
                    hardware.latest_intrinsics,
                    webcam.latest_frame,
                    detector=self._recipe_detector,
                    material_classifier=self._recipe_material_classifier,
                    fallback_material_classifier=self._recipe_fallback_material_classifier,
                    config=self._recipe_config,
                    realsense_baseline_depth_m=getattr(hardware, "reference_realsense", None),
                )
                result = {"available": True, **payload}
            except Exception as exc:  # pragma: no cover - defensive, keeps /api/state alive
                LOGGER.warning("Recipe pipeline failed on this frame (%s)", exc)
                result = {"available": False, "reason": "error", "detail": str(exc)}

            self._recipe_cache = result
            self._recipe_cache_at = now
            return result
        finally:
            self._recipe_lock.release()

    def state(self) -> dict[str, Any]:
        states = {camera_id: station.state() for camera_id, station in self.pipelines.items()}
        # Preserve the previous single-camera API for existing RealSense bridges.
        cloud = {
            "enabled": bool(self.config.cloud_enabled),
            "processing_mode": "cloud" if self.config.cloud_enabled else "local",
            "project": self.config.cloud_project or None,
            "vm_name": self.config.cloud_vm_name or None,
            "zone": self.config.cloud_zone or None,
            # Only the launcher knows whether the VM is actually up; the
            # backend reports "unknown" rather than inventing a status.
            "vm_status": self.config.cloud_vm_status or (
                "unknown" if self.config.cloud_enabled else "not used"
            ),
            "pi_host": self.config.pi_host or None,
        }
        return {**states["realsense"], "mode": "independent-dual-camera-comparison",
                "cloud": cloud, "operator_session": self.operator_session(),
                "operating_mode": self.config.operating_mode,
                "build_version": __version__,
                "frames_processed": self.frames_processed, "cameras": states, "comparison": self.comparison(),
                "fused": self.fused_result(), "recipe_result": self.recipe_result()}
