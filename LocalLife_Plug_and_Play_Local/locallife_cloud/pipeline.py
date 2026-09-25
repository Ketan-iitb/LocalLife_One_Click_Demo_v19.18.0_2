"""End-to-end detection, tracking, calibration, and volume measurement."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from .config import (
    AppConfig,
    GEOMETRY_VALIDATION_REJECT_LABELS,
    GEOMETRY_VALIDATION_REJECT_WORDS,
)
from .geometry import (
    classify_color,
    combined_mask,
    detect_scene_objects,
    fixed_bin_mask,
    fuse_scene_detections,
    intersection_over_union,
    is_phantom_source,
    roi_mask,
)
from .inference import MetricDepthEstimator, create_segmenter
from .material import MaterialClassifier
from .ledger import WastePlantLedger, waste_object_type
from .logitech import DETECTOR_ONLY_SOURCE, bound_logitech_detections, stabilize_background_depth
from .heightmap_volume import (
    HeightMapSettings,
    HeightMapVolume,
    incremental_deposit,
    integrate_height_map,
)
from uuid import uuid4

from .sorting_rules import classify_sorting, mis_sort_family
from . import __version__
from .stable_identity import Observation, StabilitySettings, StableObjectRegistry, observations_from
from .footprint import DimensionSmoother
from .logitech_footprint import FootprintResult, measure_footprint
from .logitech_autocal import BaselineLearner, camera_height_from_plane
from .logitech_geometry import (
    CHANGE_RECOVERED_SOURCE,
    GeometryStabiliser,
    minimum_object_pixels,
    static_background_reason,
    unclaimed_foreground_islands,
)
from .diagnostics import HardwareDiagnostics
from .event_log import MeasurementEventLog, STATUS_ACCEPTED, STATUS_REJECTED, resolve_event_id
from .storage import ResultStore
from .tracking import ObjectTracker
from .box_templates import load_box_templates, match_box_template
from .logitech_calibration import RELATIVE_ONLY_MESSAGE, LogitechCalibrationStore, LogitechLens, depth_output_kind
from .coordinates import clip_to_region, frame_consistency, restore_mask
from .logitech_factor import LogitechVolumeFactors, geometry_group
from .logitech_volume import axis_aligned_plane, fit_plane_alignment, metric_object_volume, stable_volume
from .vocabulary import object_type as canonical_object_type
from .deposit_state import DepositStateMachine, FrameObservation
from .logitech_calibration import METRIC_OUTPUT, depth_output_kind
from .measurement_mask import (BACKGROUND_REASONS, DETECTOR_MASK, FOREGROUND_COMPONENT,
                               NO_NEW_DEPOSIT, clean, deposit_component, looks_like_background,
                               rgb_change)
from .logitech_metric import (CALIBRATION_SET, EVALUATION_SET, RECOMMENDED_SAMPLES, CameraSetup,
                              HeightSample, LogitechMetricStore, fit_height_calibration,
                              integrate_volume_l, robust_height_cm, stable_statistics,
                              zone_signature)
from .measurement_zone import MeasurementZone, MeasurementZoneStore
from .readiness import MeasurementReadiness
from .shape_geometry import (CYLINDER, CYLINDER_REJECTIONS, UNCERTAIN, GeometryLock,
                             ObjectSignature, ShapeGeometry, measure_shape)
from .types import BoxVolumeMeasurement, CameraIntrinsics, DepthCalibration, Detection, FrameAnalysis
from .volume import (
    ReferencePlane,
    aggregate_box_measurements,
    calibrate_monocular_depth,
    estimate_box_volume_cuboid,
    estimate_object_dimensions,
    estimate_volume,
    fit_reference_plane,
    fit_support_plane_from_background,
    object_plane_points,
    reference_plane_is_usable,
    recover_elevated_object_mask,
    synthesize_plane_depth,
)

LOGGER = logging.getLogger(__name__)

# Pixels trimmed from a Logitech object mask before shape geometry: Depth
# Anything V2 smooths depth across object boundaries, so the rim is background.
LOGITECH_MASK_ERODE_PX = 2


# Compound labels that contain "bag" as a sub-word but name furniture or an
# accessory, not a waste container. These matter specifically when the
# neural detector falls back to its broad, uncurated "prompt-free"
# vocabulary (e.g. Ultralytics' CLIP text-prompt dependency failed to
# install) instead of our curated bag/box prompts: that fallback model's
# label set includes classes like "bean bag chair" that would otherwise pass
# a naive "contains the word bag" check and get treated as a real bag.
NON_WASTE_BAG_QUALIFIERS = {
    "chair", "sofa", "couch", "seat", "cushion", "pillow", "bed", "ottoman", "stool",
    "backpack", "rucksack", "laptop", "briefcase", "duffel", "sports", "handbag", "purse",
    "laundry", "hamper", "basket", "storage",
}

# A phantom/silhouette detection (see geometry.fuse_scene_detections /
# geometry.PHANTOM_DETECTION_SOURCES) is a changed-depth region with no
# neural label ever confirming it is really a waste bag or box -- it exists
# purely so a real, already-confirmed object keeps being tracked and measured
# through a brief detector dropout. It must never be written to the durable
# waste-plant ledger: a background object that happens to register as
# "changed" (a blanket fold, a doorway edge, a camera pan) has exactly the
# same zero-confidence signature, and depositing one records fictional waste
# with a fictional volume, as happened with the 87-118 L phantom "deposits"
# reported live against a real ~37 L object.
# Materials that contradict a rigid-box label.
SOFT_MATERIALS = ("fabric", "textile", "cloth", "foam", "plastic film")


def classification_conflict(detection: Detection) -> str | None:
    """A box-family label with a soft material is not a trustworthy cuboid."""
    label = _normalized_label(detection.label)
    if not set(label.split()) & {"box", "boxes", "carton", "cartons", "parcel", "parcels", "package"}:
        return None
    material = (detection.material or "").lower()
    if any(word in material for word in SOFT_MATERIALS) and detection.material_confidence >= 0.4:
        return f"detector says {detection.label}; material says {detection.material}"
    return None


def _is_phantom_detection(detection: Detection) -> bool:
    return is_phantom_source(detection.source)


def is_bag_detection(label: str) -> bool:
    """The fixed installation recognizes bags and sacks, never generic objects."""
    normalized = " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())
    words = set(normalized.split())
    if not words & {"bag", "bags", "sack", "sacks"}:
        return False
    return not (words & NON_WASTE_BAG_QUALIFIERS)


def is_supported_waste_detection(label: str) -> bool:
    return accepted_object_class(label) is not None


NEGATIVE_WASTE_LABELS = {
    "pillow", "cushion", "blanket", "bedding", "chair", "office chair", "chair wheel",
    "furniture", "furniture leg", "bottle", "plastic bottle", "lotion bottle",
    "soda can", "aluminium drink can", "aluminum drink can", "tin can", "drink can",
    "backpack", "rucksack", "laptop bag", "briefcase", "duffel bag", "sports bag",
    "handbag", "purse", "shoe", "sneaker", "sandal", "slipper", "boot", "clothing",
    "person", "hand", "foot",
    "laundry basket", "laundry hamper", "fabric storage basket", "curtain", "drape",
    "chair cover", "floor mat", "rug", "power cable", "power adapter", "power strip",
    "charger", "door",
}


def _normalized_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def accepted_object_class(label: str, operating_mode: str = "waste") -> str | None:
    """Map detector vocabulary to the active mode's accepted object family.

    Waste mode has only three families. Broad labels such as a bare ``bag``
    are deliberately not accepted there: the
    real-hardware result set shows backpacks, laptop cases and bedding being
    forced into a waste-bag class.  A bag must carry a plastic/waste or paper
    qualifier, and cardboard must be a box/carton/parcel rather than an
    arbitrary flat sheet. Geometry-validation mode instead maps recognized
    non-background household items to ``measurement_object``.
    """
    normalized = _normalized_label(label)
    if operating_mode == "geometry_validation":
        words = set(normalized.replace("(", " ").replace(")", " ").split())
        if (
            not normalized
            or normalized in GEOMETRY_VALIDATION_REJECT_LABELS
            or words & GEOMETRY_VALIDATION_REJECT_WORDS
        ):
            return None
        return "measurement_object"
    if normalized in NEGATIVE_WASTE_LABELS:
        return None
    words = set(normalized.replace("(", " ").replace(")", " ").split())
    if words & {"box", "boxes", "carton", "cartons", "parcel", "parcels"}:
        return "cardboard_box"
    if not words & {"bag", "bags", "sack", "sacks"}:
        return None
    if words & NON_WASTE_BAG_QUALIFIERS:
        return None
    if words & {"paper", "kraft"}:
        return "paper_bag"
    if words & {
        "plastic", "polythene", "polyethylene", "garbage", "trash", "waste",
        "refuse", "rubbish", "bin",
    }:
        return "plastic_bag"
    return None


def reject_prompt_conflicts(detections: list[Detection]) -> list[Detection]:
    """Reject a waste label when a similarly confident lookalike owns the same pixels."""
    negatives = [
        item for item in detections
        if _normalized_label(item.label) in NEGATIVE_WASTE_LABELS
    ]
    accepted: list[Detection] = []
    for detection in detections:
        if not is_supported_waste_detection(detection.label):
            continue
        conflict = False
        for negative in negatives:
            ax1, ay1, ax2, ay2 = detection.box
            bx1, by1, bx2, by2 = negative.box
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0, min(ay2, by2) - max(ay1, by1)
            )
            smaller_box = max(1, min(
                max(0, ax2 - ax1) * max(0, ay2 - ay1),
                max(0, bx2 - bx1) * max(0, by2 - by1),
            ))
            containment = intersection / smaller_box
            mask_overlap = 0.0
            if (
                detection.mask is not None and negative.mask is not None
                and detection.mask.shape == negative.mask.shape
            ):
                smaller_mask = max(1, min(
                    int(np.count_nonzero(detection.mask)), int(np.count_nonzero(negative.mask)),
                ))
                mask_overlap = int(np.count_nonzero(detection.mask & negative.mask)) / smaller_mask
            overlaps = (
                intersection_over_union(detection.box, negative.box) >= 0.25
                or containment >= 0.55
                or mask_overlap >= 0.45
            )
            if overlaps and negative.confidence >= detection.confidence * 0.60:
                conflict = True
                break
        if not conflict:
            accepted.append(detection)
    return accepted


def filter_waste_detections(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
    region: np.ndarray,
    config: AppConfig,
    *,
    min_pixels: int | None = None,
    rejections: Counter | None = None,
    confidence: float | None = None,
) -> list[Detection]:
    """Apply installation-aware quality gates before tracking or measuring.

    Open-vocabulary models deliberately have high recall. They also return tiny
    shelf objects and scene-sized furniture masks for generic prompts. These
    deterministic gates keep only a plausible waste-container footprint inside
    the measurement region, then collapse nested prompt duplicates.
    """
    frame_height, frame_width = frame_shape[:2]
    region_area = max(1, int(np.count_nonzero(region)))
    minimum_area = max(
        int(region_area * config.min_detection_area_fraction),
        min(config.min_component_pixels if min_pixels is None else min_pixels, max(1, int(region_area * 0.10))),
    )
    threshold = config.detector_confidence if confidence is None else confidence
    rejections = Counter() if rejections is None else rejections
    retained: list[Detection] = []
    candidates = (
        detections
        if config.operating_mode == "geometry_validation"
        else reject_prompt_conflicts(detections)
    )
    for detection in candidates:
        detection.accepted_class = accepted_object_class(
            detection.label, config.operating_mode,
        )
        if detection.accepted_class is None:
            rejections["class_not_accepted"] += 1
            continue
        if config.operating_mode == "waste" and config.bag_only and not is_bag_detection(detection.label):
            rejections["class_not_accepted"] += 1
            continue
        if config.operating_mode == "waste" and not config.bag_only and not is_supported_waste_detection(detection.label):
            rejections["class_not_accepted"] += 1
            continue
        if detection.source.startswith("yolo") and detection.confidence < threshold:
            rejections["below_confidence"] += 1
            continue
        x1, y1, x2, y2 = detection.box
        width, height = max(0, x2 - x1), max(0, y2 - y1)
        if width < frame_width * config.min_detection_side_fraction:
            rejections["box_too_narrow"] += 1
            continue
        if height < frame_height * config.min_detection_side_fraction:
            rejections["box_too_short"] += 1
            continue
        center_x = min(frame_width - 1, max(0, (x1 + x2) // 2))
        center_y = min(frame_height - 1, max(0, (y1 + y2) // 2))
        if not region[center_y, center_x]:
            rejections["centre_outside_roi"] += 1
            continue
        mask = combined_mask([detection], (frame_height, frame_width)) & region
        area = int(np.count_nonzero(mask))
        if area < minimum_area:
            rejections["mask_area_too_small"] += 1
            continue
        if area / region_area > config.max_detection_area_fraction:
            rejections["mask_area_too_large"] += 1
            continue
        retained.append(detection)

    return deduplicate_overlapping_detections(retained, (frame_height, frame_width))


def deduplicate_overlapping_detections(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
    *,
    iou_threshold: float = 0.45,
    nested_threshold: float = 0.65,
) -> list[Detection]:
    """Collapse duplicate/overlapping boxes for the same physical object.

    Open-vocabulary detection plus depth-scene fusion can each independently
    propose a box for the same container (a prompt match and its own fused
    silhouette, or two overlapping scene fragments that survived merging).
    Keep only the highest-confidence, largest box per cluster of same-type
    overlapping detections so one bag is never tracked/counted twice.
    """
    frame_height, frame_width = frame_shape[:2]
    unique: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: (-item.confidence, -item.area_pixels)):
        duplicate = False
        candidate_mask = combined_mask([candidate], (frame_height, frame_width))
        candidate_area = max(1, int(np.count_nonzero(candidate_mask)))
        for existing in unique:
            if (
                existing.accepted_class != "measurement_object"
                and candidate.accepted_class != "measurement_object"
                and waste_object_type(existing.label) != waste_object_type(candidate.label)
            ):
                continue
            existing_mask = combined_mask([existing], (frame_height, frame_width))
            smaller = min(candidate_area, max(1, int(np.count_nonzero(existing_mask))))
            nested = int(np.count_nonzero(candidate_mask & existing_mask)) / smaller
            ax1, ay1, ax2, ay2 = candidate.box
            bx1, by1, bx2, by2 = existing.box
            box_intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0, min(ay2, by2) - max(ay1, by1)
            )
            smaller_box = max(1, min(
                max(0, ax2 - ax1) * max(0, ay2 - ay1),
                max(0, bx2 - bx1) * max(0, by2 - by1),
            ))
            box_nested = box_intersection / smaller_box
            if (
                intersection_over_union(candidate.box, existing.box) >= iou_threshold
                or nested >= nested_threshold
                or box_nested >= nested_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return sorted(unique, key=lambda item: item.area_pixels, reverse=True)


def summarize_depth_signal(depth_m: np.ndarray | None) -> dict[str, Any]:
    """Expose whether a real, usable depth signal reaches the cloud service."""
    if depth_m is None or depth_m.ndim != 2:
        return {
            "available": False,
            "valid_pixels": 0,
            "total_pixels": 0,
            "valid_percent": 0.0,
            "min_m": None,
            "median_m": None,
            "max_m": None,
        }
    valid = np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 20.0)
    values = depth_m[valid]
    return {
        "available": True,
        "valid_pixels": int(values.size),
        "total_pixels": int(depth_m.size),
        "valid_percent": round(float(values.size / max(1, depth_m.size) * 100.0), 2),
        "min_m": None if not values.size else round(float(np.min(values)), 4),
        "median_m": None if not values.size else round(float(np.median(values)), 4),
        "max_m": None if not values.size else round(float(np.max(values)), 4),
    }



def _object_signature(detection: Detection, camera_id: str) -> ObjectSignature:
    """A coarse identity for the detection this frame.

    Track ids restart at 1 after a reset and are handed out again once a track
    expires, so the shape frozen for one object could be returned for the next
    one holding that id -- which is how a bottle's 89 x 89 x 181 mm reappeared
    on a backpack, a carton and on background. The signature travels with the
    frozen shape, and a different object starts its own measurement.
    """
    x1, y1, x2, y2 = (float(value) for value in detection.box)
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    area = float(np.count_nonzero(detection.mask)) if detection.mask is not None else width * height
    family = (detection.accepted_class or (detection.label or "").split()[-1:] or [""])[0]
    return ObjectSignature(
        camera=camera_id, label=str(family).lower(),
        centre_x=(x1 + x2) / 2.0, centre_y=(y1 + y2) / 2.0,
        area=max(area, 1.0), aspect=min(width, height) / max(width, height),
    )


def _tracked_component(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Keep the island the tracked detection sits on.

    Two objects that touch, or an object joined to the floor by a shadow,
    arrive as one mask; integrating all of it reports their sum as this
    object's volume. Only the component under the detection's own box is
    measured -- the detector's mask itself is untouched.
    """
    import cv2

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    x1, y1, x2, y2 = (int(value) for value in box)
    centre_y, centre_x = (y1 + y2) // 2, (x1 + x2) // 2
    chosen = 0
    if 0 <= centre_y < labels.shape[0] and 0 <= centre_x < labels.shape[1]:
        chosen = int(labels[centre_y, centre_x])
    if chosen == 0:
        # The centre fell in a hole (a handle, a bottle's neck): take the
        # island holding most of the box instead.
        inside = labels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        values, counts = np.unique(inside[inside > 0], return_counts=True)
        if values.size == 0:
            return mask
        chosen = int(values[int(counts.argmax())])
    return labels == chosen


def _spans_the_background(mask: np.ndarray, region: np.ndarray | None) -> bool:
    """Is this mask the room rather than an object placed in it?"""
    if int(np.count_nonzero(mask)) < 500:
        return False
    return looks_like_background(mask, region) is not None


def _zone_limits_m(zone: Any) -> tuple[float, float] | None:
    """The calibrated mat's own size: nothing standing on it can be larger."""
    width = getattr(zone, "width_m", None)
    depth = getattr(zone, "depth_m", None)
    if width and depth:
        return (float(width), float(depth))
    return None


class VisionPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        detector: Any | None = None,
        depth_estimator: Any | None = None,
        material_classifier: Any | None = None,
        camera_id: str = "realsense",
        inference_lock: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.camera_id = camera_id
        self.inference_lock = inference_lock or threading.RLock()
        self.detector = detector or create_segmenter(config)
        self.depth_estimator = (
            depth_estimator
            if depth_estimator is not None
            else MetricDepthEstimator(config, getattr(self.detector, "device", None))
            if config.enable_monocular_depth
            else None
        )
        self.material_classifier = material_classifier or (
            MaterialClassifier(config, getattr(self.detector, "device", None))
            if config.enable_material_classification
            else None
        )
        self.tracker = ObjectTracker(
            confirmation_frames=config.tracker_confirm_frames,
            max_missing_frames=config.tracker_max_missing_frames,
            minimum_iou=config.tracker_minimum_iou,
            maximum_center_distance=config.tracker_max_center_distance,
            phantom_max_missing_frames=config.tracker_phantom_max_missing_frames,
        )
        # Loaded once per pipeline instance, not per frame: box_templates.yaml
        # rarely changes, and re-reading a YAML file on every processed
        # frame would be a needless (if small) per-frame cost. Only matters
        # for RealSense (`_measure_box_cuboid` below is only ever called for
        # camera_id == "realsense"), but loading it unconditionally is
        # cheap and keeps this simple.
        self.box_templates = load_box_templates()
        self.store = ResultStore(config.results_dir)
        self.ledger = WastePlantLedger(
            self.store,
            color_streams=config.color_waste_streams,
            history_limit=config.history_limit,
            camera_id=camera_id,
        )
        self.quarantined_history_entries = (
            self.ledger.quarantine_implausible(config.logitech_max_item_volume_l)
            if camera_id == "logitech" else 0
        )
        if self.quarantined_history_entries:
            LOGGER.warning(
                "Excluded %s implausible historic Logitech deposit(s) from active totals",
                self.quarantined_history_entries,
            )
        self.lock = threading.RLock()
        self.baseline_rgb: np.ndarray | None = None
        self.baseline_realsense: np.ndarray | None = None
        self.baseline_monocular: np.ndarray | None = None
        self.reference_rgb: np.ndarray | None = None
        self.reference_realsense: np.ndarray | None = None
        self.reference_monocular: np.ndarray | None = None
        self.calibration: DepthCalibration | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_processed_frame: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_monocular_depth: np.ndarray | None = None
        self.latest_intrinsics: CameraIntrinsics | None = None
        self.latest_intrinsics_origin = "factory-calibrated" if camera_id == "realsense" else "not-received"
        self.latest_analysis: FrameAnalysis | None = None
        self.latest_analysis_timestamp = 0.0
        self.frames_processed = 0
        self.frames_received = 0
        self.latest_frame_timestamp = 0.0
        self.last_frame_received_at: float | None = None
        self.last_frame_processed_at: float | None = None
        self.baseline_frame_count = 0
        self.baseline_noise_m = 0.0
        self.baseline_noise_map: np.ndarray | None = None
        self.reference_plane: ReferencePlane | None = None
        # Where the support plane / reference surface actually came from this
        # frame: "captured-baseline" (an empty-scene capture, preferred),
        # "live-frame-background" (fitted from the floor around the object --
        # see the round-24 block in `_assemble`), or "none".
        self.support_plane_source = "none"
        self.calibration_mode = "not-calibrated"
        self.last_background_stabilization: dict[str, Any] = {
            "applied": False, "scale": 1.0, "offset_m": 0.0, "anchor_pixels": 0,
        }
        self._occupied_logitech_mask: np.ndarray | None = None
        self.committed_bags = self.ledger.summary()["deposited_bags"]
        self._recent_depth_frames: deque[np.ndarray] = deque(maxlen=config.baseline_window_frames)
        self._recent_monocular_frames: deque[np.ndarray] = deque(maxlen=config.baseline_window_frames)
        self._volume_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(config.volume_window_frames, config.settle_frames))
        )
        # Playbook sections 5 and 10: the bin is not emptied between deposits,
        # so a bag's own contribution is the change in total occupied bin
        # volume across its arrival. `_previous_bin_total_l` is the last frame's
        # total (the state before whatever appears next), captured per track the
        # moment that track is created and compared against the total at the
        # moment it is deposited.
        self._previous_bin_total_l: float | None = None
        self._bin_total_before_track: dict[int, float] = {}
        # Height grid of the scene as it stood when the last deposit was
        # accepted. Every later deposit is measured against this, so already
        # committed material can never be counted again.
        self._committed_scene: HeightMapVolume | None = None
        # Canonical event persistence. Rows are written when a measurement is
        # *finalised*, keyed by a durable event id -- not per frame, and not
        # keyed by track_id, which restarts at 1 on every run. See event_log.py
        # for why the per-frame writer this replaces produced blank exports.
        self.event_log = MeasurementEventLog(config.results_dir)
        # Track ids already finalised into the event log this run, so a settled
        # object is considered once and never re-offered.
        self._csv_logged: set[int] = set()
        # Volume history length at which a non-ledger mode considers a track
        # settled; mirrors the ledger's own settle rule so both modes finalise
        # on the same evidence.
        self._last_persist_result: dict[str, Any] | None = None
        self._dimension_smoother = DimensionSmoother(
            window=config.dimension_smoothing_frames,
        )
        # Permanent measurement identity. The detector renumbers a stationary
        # object (a pillow went 10 -> 52, a can 54 -> 63), so its track id is
        # only an association hint here and never the event id.
        # detector track id -> permanent event id, for the frames after the
        # detector renumbers an object that is already committed.
        self._permanent_event_ids: dict[int, int] = {}
        self.identities = StableObjectRegistry(
            StabilitySettings(
                window_frames=config.stability_window_frames,
                min_valid_stable_frames=config.min_valid_stable_frames,
                max_centroid_shift_px=config.max_centroid_shift_px,
                min_mask_iou=config.min_mask_iou,
                max_depth_change_mm=config.max_depth_change_mm,
                max_volume_variation_percent=config.max_volume_variation_percent,
                finalisation_hold_seconds=config.finalisation_hold_seconds,
            )
        )
        # Support plane recorded at calibration, and whether the live camera
        # still matches it. An old plane must never be applied to a new pose.
        self._calibration_id: str | None = None
        self._calibration_valid = True
        # Table-relative box cuboid: per-track accepted-frame history feeding
        # `aggregate_box_measurements()` (Revised Dual-Camera Volume
        # Estimation recipe, section 13 -- median L/W/H across accepted
        # frames, never a per-frame sum). Separate from `_volume_history`
        # (which smooths the already-multiplied scalar liters figure)
        # because this keeps the three dimensions independently aggregated
        # before the one final multiplication, and separate from
        # `_box_frames_considered` (below) which counts every attempted
        # frame, including ones `estimate_box_volume_cuboid()` itself
        # rejected, for the diagnostic "frames_considered" vs
        # "frames_accepted" distinction the PDF's result contract asks for.
        self._box_measurement_history: dict[int, deque[BoxVolumeMeasurement]] = defaultdict(
            lambda: deque(maxlen=config.box_aggregation_window_frames)
        )
        self._box_frames_considered: dict[int, int] = defaultdict(int)
        # Shape method voting and freeze per track (shape_geometry.py).
        self._geometry_lock = GeometryLock(required_frames=max(3, config.settle_frames))
        self.diagnostics = HardwareDiagnostics(
            config.results_dir / "hardware_diagnostics", camera_id,
            enabled=config.hardware_diagnostic, interval_s=config.hardware_diagnostic_interval_s,
        )
        self._frame_context: dict[str, Any] = {}
        # Per-camera production-path counters (frames -> detections -> masks ->
        # tracks -> DA-V2 -> finalised), shown on the research dashboard.
        self.stage_counters: Counter = Counter()
        self.last_stage_rejections: dict[str, int] = {}
        # Frozen empirical correction for Logitech volumes (logitech_factor.py).
        self.volume_factors = (
            LogitechVolumeFactors(
                config.results_dir / "calibration" / "logitech_volume_factors.json",
                camera_setup=f"{camera_id}:{config.depth_model}",
            )
            if camera_id == "logitech" else None
        )
        # Live Logitech volumes per track, and the trimmed-median stable value.
        self._logitech_volume_samples: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=9))
        self._logitech_volume_spread: dict[int, float] = {}
        self.last_logitech_volume_diagnostics: dict[str, Any] = {}
        # Why Depth Anything V2 is unavailable, when it is (shown on the dashboard).
        self.depth_load_error: str | None = None
        # Why a stored calibration was refused (camera moved, ROI or resolution changed).
        self.calibration_rejected_reason: str | None = None
        # The other camera's intrinsics, so this one can refuse to use them.
        self.peer_intrinsics: CameraIntrinsics | None = None
        self._depth_cache: tuple[float, np.ndarray] | None = None
        # Support plane assumed from the ROI's own median depth while no
        # empty-scene calibration exists (Logitech only).
        self._uncalibrated_plane = None
        # Frames each track has waited for a stable volume (Logitech timeout).
        self._measurement_frames: dict[int, int] = {}
        # The spread statistics behind each track's current answer.
        self._stability: dict[int, dict[str, Any]] = {}
        # Consecutive frames a track was refused for showing no change.
        self._deposit_refusals: dict[int, int] = {}
        # The last readiness chain logged, so the line appears on a change.
        self._last_chain: tuple[Any, ...] | None = None
        # What each live track looked like last frame, so a reused id cannot
        # inherit the previous object's measurements (shape_geometry.py).
        self._track_signatures: dict[int, ObjectSignature] = {}
        # The mat this camera measures on. Each camera sees it from its own
        # place, so the zone is stored per camera and never shared.
        self.zones = MeasurementZoneStore(config.results_dir / "calibration")
        self.measurement_zone: MeasurementZone | None = self.zones.get(camera_id)
        self.zone_source = "saved-zone" if self.measurement_zone is not None else "configured-roi"
        # Where this camera is in the deposit it is watching (deposit_state.py).
        self.deposit_state = DepositStateMachine(settle_frames=max(2, config.settle_frames))
        # The last frame's measurement-mask decision, for the diagnostics bundle.
        self.last_measurement_mask: dict[str, Any] = {}
        # Set when a relative-depth checkpoint could not be scaled to metres.
        self.relative_depth_reason: str | None = None
        # The Logitech's own metric layer: the installation it was calibrated
        # in, the ruler-measured samples and the fitted height mapping.
        self.metric_store = LogitechMetricStore(config.results_dir / "calibration")
        self.height_calibration = self.metric_store.calibration if camera_id == "logitech" else None
        self.height_calibration_reason: str | None = None
        # What the last measured frame saw, so a calibration sample can be
        # captured from the object standing in the zone right now.
        self.last_metric_context: dict[str, Any] = {}
        # Why the last Logitech footprint measured or refused, for diagnostics.
        self.last_footprint_result: FootprintResult | None = None
        # The fixed camera's floor plane, fitted once per resolution.
        self._logitech_plane_cache: tuple[tuple[int, int], Any] | None = None
        self._logitech_plane_reason: str | None = None
        self._logitech_plane_source = "none"
        # Rolling median of each Logitech track's dimensions and volume.
        self._logitech_geometry = GeometryStabiliser(
            window=9, minimum_frames=3, tolerance=0.25,
        )
        # Where the camera height came from, and the empty scene it learned by
        # itself -- both shown in the diagnostics so a derived value is never
        # mistaken for a measured one.
        self.logitech_distance_source = (
            "operator_measured" if config.logitech_reference_distance_m else "none"
        )
        self.logitech_derived_distance_m: float | None = None
        self._baseline_learner = BaselineLearner(depth_noise_m=config.depth_noise_m)
        self.auto_baseline_state: dict[str, Any] = {}
        # Frames each confirmed track has waited for finalisation (non-waste modes).
        self._unfinalised_frames: dict[int, int] = {}
        # Detector / foreground / final / rejected masks of the latest Logitech
        # frame, for the diagnostic overlay.
        self.logitech_mask_debug: dict[str, Any] = {}
        # Called with each persisted measurement row (DualCameraCoordinator
        # pairs the two cameras through it).
        self.measurement_listener: Callable[[dict[str, Any]], None] | None = None
        # Logitech metric-depth calibration samples and fit, kept apart from
        # evaluation data (logitech_calibration.py).
        self.logitech_calibration = (
            LogitechCalibrationStore(config.results_dir / "calibration" / "logitech_depth.json")
            if camera_id == "logitech" else None
        )
        # Checkerboard lens profile, applied to every Logitech frame before
        # detection, depth inference and geometry.
        self.logitech_lens = (
            LogitechLens(
                config.results_dir / "calibration" / "logitech_lens.json",
                config.results_dir.parent / "dual_camera_calibration.json",
            )
            if camera_id == "logitech" else None
        )
        self._color_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=max(3, config.volume_window_frames))
        )
        self._material_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=max(3, config.volume_window_frames))
        )
        self._material_frame_counts: dict[int, int] = defaultdict(int)
        # Live "seen" totals are intentionally independent from the durable
        # ledger.  A detected bag should appear immediately even while its
        # baseline/depth measurement is still being validated.
        self._session_seen_tracks: dict[int, dict[str, Any]] = {}
        self.saved_profile_loaded = False
        self.baseline_restore_state = "disabled" if not config.restore_saved_baseline else "not-found"
        self.saved_baseline_changed_fraction: float | None = None
        self._saved_baseline_matching_frames = 0
        self._automatic_baseline_stable_frames = 0
        self._automatic_baseline_previous: np.ndarray | None = None
        self._automatic_baseline_status = "disabled" if not config.automatic_baseline else "waiting-for-empty-stable-scene"
        self._load_volume_calibration()
        self._load_saved_baseline()

    def warmup(self) -> dict[str, Any]:
        # Model downloads only happen here, on first use, and only need network
        # access once (results are cached to disk). Any failure here (offline
        # first run, a blocked download, a corrupted cache) must not prevent
        # the one-click launcher from starting a working dashboard: fall back
        # to the dependency-free local detector/measurement instead of crashing.
        if hasattr(self.detector, "load"):
            try:
                self.detector.load()
            except Exception as exc:
                LOGGER.warning(
                    "%s: could not load detector model %s (%s). Falling back to the "
                    "local OpenCV background detector for this run. Check the internet "
                    "connection, then restart the one-click launcher to retry the "
                    "AI model download.",
                    self.camera_id, self.config.detector_model, exc,
                )
                from .inference import AdaptiveForegroundSegmenter

                self.detector = AdaptiveForegroundSegmenter(self.config)
        if self.depth_estimator is not None and hasattr(self.depth_estimator, "load"):
            try:
                self.depth_estimator.load()
            except Exception as exc:
                LOGGER.warning(
                    "%s: could not load depth model %s (%s). Depth-based volume "
                    "stays unavailable for this run.",
                    self.camera_id, self.config.depth_model, exc,
                )
                self.depth_estimator = None
                self.depth_load_error = f"{type(exc).__name__}: {exc}"
        if self.material_classifier is not None:
            try:
                self.material_classifier.load()
            except Exception as exc:  # pragma: no cover - MaterialClassifier.load() already guards this
                LOGGER.warning("%s: could not load material classifier (%s)", self.camera_id, exc)
                self.material_classifier.enabled = False
        return {
            "detector": self.config.detector_model,
            "depth_model": self.config.depth_model if self.depth_estimator else None,
            "material_model": self.config.material_model if (
                self.material_classifier is not None and self.material_classifier.enabled
            ) else None,
            "runtime": getattr(self.detector, "runtime", {}),
            "prompts": list(self.config.prompts),
            "negative_prompts": list(self.config.negative_prompts),
        }

    def set_baseline(
        self,
        frame: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            image = self.latest_frame if frame is None else frame
            reference = self.latest_depth if depth_m is None else depth_m
            if image is None:
                raise ValueError("Capture an empty-scene frame before setting the baseline")

            self.baseline_rgb = image.copy()
            self.baseline_noise_map = None
            self.baseline_noise_m = 0.0
            depth_frames = [
                item for item in self._recent_depth_frames
                if reference is not None and item.shape == reference.shape
            ] if frame is None and depth_m is None else []
            if reference is not None and depth_frames:
                valid_stack = np.stack([
                    np.where(np.isfinite(item) & (item > 0.10), item, np.nan)
                    for item in depth_frames
                ])
                with np.errstate(invalid="ignore"):
                    median_reference = np.nanmedian(valid_stack, axis=0)
                self.baseline_realsense = np.where(
                    np.isfinite(median_reference), median_reference, reference
                ).astype(np.float32)
                residuals = np.abs(valid_stack - self.baseline_realsense[None, :, :])
                with np.errstate(invalid="ignore"):
                    robust_noise = 1.4826 * np.nanmedian(residuals, axis=0)
                self.baseline_noise_map = np.where(np.isfinite(robust_noise), robust_noise, 0).astype(np.float32)
                region = fixed_bin_mask(image.shape, self.config.roi, self.config.bin_polygon)
                samples = self.baseline_noise_map[region]
                self.baseline_noise_m = float(np.median(samples)) if samples.size else 0.0
                self.baseline_frame_count = len(depth_frames)
            else:
                self.baseline_realsense = None if reference is None else reference.copy()
                self.baseline_frame_count = 0 if reference is None else 1
            reference = self.baseline_realsense
            self.reference_rgb = self.baseline_rgb.copy()
            self.reference_realsense = None if reference is None else reference.copy()
            self.latest_intrinsics = intrinsics or self.latest_intrinsics
            self.baseline_monocular = None
            self.calibration = None
            self.calibration_mode = "not-calibrated"

            if self.depth_estimator is not None:
                with self.inference_lock:
                    predicted = self.depth_estimator.estimate_batch([image])[0]
                if self.camera_id == "logitech":
                    fitted = self._fitted_logitech_calibration(predicted.shape)
                    if fitted is not None:
                        # Scale and shift fitted from several tape-measured
                        # reference distances (calibration data only).
                        self.calibration = fitted
                        self.calibration_mode = "independent-measured-distance"
                    elif self.config.logitech_reference_distance_m > 0:
                        region = fixed_bin_mask(image.shape, self.config.roi, self.config.bin_polygon)
                        values = predicted[region & np.isfinite(predicted) & (predicted > 0.10)]
                        if not values.size:
                            raise ValueError("The Logitech depth model returned no valid baseline pixels")
                        scale = self.config.logitech_reference_distance_m / float(np.median(values))
                        self.calibration = DepthCalibration(
                            scale=scale, offset_m=0.0, rmse_m=0.0, sample_pixels=int(values.size),
                            method="reference-distance-scale",
                            calibration_id=uuid4().hex[:12],
                            calibrated_at=time.time(),
                            reference_distance_m=float(self.config.logitech_reference_distance_m),
                            sample_count=1,
                            resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
                        )
                        self.calibration_mode = "independent-measured-distance"
                    else:
                        self.calibration = DepthCalibration(
                            scale=1.0, offset_m=0.0, rmse_m=0.0, sample_pixels=int(predicted.size),
                            method="model-metric-unverified",
                        )
                        self.calibration_mode = "model-metric-unverified"
                elif reference is not None:
                    self.calibration = calibrate_monocular_depth(
                        predicted,
                        reference,
                        mask=roi_mask(predicted.shape, self.config.roi),
                    )
                    if self.calibration is not None:
                        self.calibration_mode = "aligned-realsense-reference"
                self.baseline_monocular = (
                    self.calibration.apply(predicted) if self.calibration is not None else predicted
                )
                if self.camera_id == "logitech" and frame is None and self._recent_monocular_frames:
                    compatible = [
                        self.calibration.apply(item) if self.calibration is not None else item
                        for item in self._recent_monocular_frames if item.shape == predicted.shape
                    ]
                    if compatible:
                        stack = np.stack(compatible)
                        self.baseline_monocular = np.median(stack, axis=0).astype(np.float32)
                        self.baseline_frame_count = len(compatible)
            self.reference_monocular = (
                None if self.baseline_monocular is None else self.baseline_monocular.copy()
            )
            floor = self.baseline_monocular if self.camera_id == "logitech" else self.baseline_realsense
            region = fixed_bin_mask(image.shape, self.config.roi, self.config.bin_polygon)
            fitted_plane = fit_reference_plane(floor, self.latest_intrinsics, mask=region)
            # Never publish dimensions from a baseline that does not contain
            # one coherent support surface. The depth/reference arrays remain
            # saved, but unsafe plane-relative L/W/H stays unavailable.
            self.reference_plane = fitted_plane if reference_plane_is_usable(fitted_plane) else None
            # A calibration is identified by the pose it was fitted at; every
            # measurement records which one produced it.
            self._calibration_id = uuid4().hex[:12]
            self._calibration_valid = True
            self._committed_scene = None
            self._occupied_logitech_mask = (
                np.zeros(image.shape[:2], dtype=bool) if self.camera_id == "logitech" else None
            )
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            self._geometry_lock.clear()
            self._track_signatures.clear()
            self._color_history.clear()
            self._material_history.clear()
            self._material_frame_counts.clear()
            self.committed_bags = self.ledger.summary()["deposited_bags"]

            baseline_directory = self.config.results_dir / "baselines"
            baseline_directory.mkdir(parents=True, exist_ok=True)
            for stale_name in (
                "realsense_depth_m.npy",
                "monocular_depth_m.npy",
                "noise_m.npy",
                "reference_realsense_depth_m.npy",
                "reference_monocular_depth_m.npy",
                "occupied_logitech_mask.npy",
            ):
                (baseline_directory / stale_name).unlink(missing_ok=True)
            np.save(baseline_directory / "rgb.npy", self.baseline_rgb)
            if self.baseline_realsense is not None:
                np.save(baseline_directory / "realsense_depth_m.npy", self.baseline_realsense)
            if self.baseline_monocular is not None:
                np.save(baseline_directory / "monocular_depth_m.npy", self.baseline_monocular)
            if self.baseline_noise_map is not None:
                np.save(baseline_directory / "noise_m.npy", self.baseline_noise_map)
            self._save_working_reference()

            summary = {
                "captured_at": time.time(),
                "frame_shape": list(image.shape),
                "has_realsense_depth": reference is not None,
                "has_monocular_depth": self.baseline_monocular is not None,
                "realsense_depth_signal": summarize_depth_signal(reference),
                "intrinsics": None if self.latest_intrinsics is None else self.latest_intrinsics.to_dict(),
                "calibration": None if self.calibration is None else self.calibration.to_dict(),
                "baseline_frame_count": self.baseline_frame_count,
                "fixed_bin_polygon": [list(vertex) for vertex in self.config.bin_polygon],
                "camera_id": self.camera_id,
                "baseline_noise_m": round(self.baseline_noise_m, 6),
                "reference_plane": None if self.reference_plane is None else self.reference_plane.to_dict(),
                "calibration_mode": self.calibration_mode,
                "volume_calibration_factor": self.config.volume_calibration_factor,
                "measurement_ready": self._logitech_measurement_ready() if self.camera_id == "logitech" else True,
            }
            self.store.save_json("baselines/metadata.json", summary)
            self.saved_profile_loaded = True
            self.baseline_restore_state = "captured-current-scene"
            self._automatic_baseline_status = "ready"
            self._automatic_baseline_stable_frames = self.config.automatic_baseline_frames
            self.saved_baseline_changed_fraction = 0.0
            self._saved_baseline_matching_frames = self.config.saved_baseline_validation_frames
            LOGGER.info("Captured empty-scene baseline: %s", summary)
            return summary

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        source: str = "live",
        timestamp: float | None = None,
        persist: bool = True,
    ) -> FrameAnalysis:
        return self.process_batch(
            [frame],
            depths=[depth_m],
            intrinsics=[intrinsics],
            sources=[source],
            timestamps=[time.time() if timestamp is None else timestamp],
            persist=persist,
        )[0]

    def update_preview(
        self,
        frame: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        timestamp: float | None = None,
        intrinsics_origin: str = "",
    ) -> None:
        """Publish the newest camera frame without waiting for GPU inference."""
        received_timestamp = time.time() if timestamp is None else float(timestamp)
        with self.lock:
            if received_timestamp < self.latest_frame_timestamp:
                return
            self.latest_frame = frame.copy()
            self.latest_depth = None if depth_m is None else depth_m.copy()
            if intrinsics is not None:
                self.latest_intrinsics = intrinsics
            if intrinsics_origin:
                self.latest_intrinsics_origin = intrinsics_origin
            self.latest_frame_timestamp = received_timestamp
            self.last_frame_received_at = time.time()
            self.frames_received += 1

    def process_batch(
        self,
        frames: list[np.ndarray],
        *,
        depths: list[np.ndarray | None] | None = None,
        intrinsics: list[CameraIntrinsics | None] | None = None,
        sources: list[str] | None = None,
        timestamps: list[float] | None = None,
        persist: bool = True,
    ) -> list[FrameAnalysis]:
        if not frames:
            return []
        total = len(frames)
        depths = depths if depths is not None else [None] * total
        intrinsics = intrinsics if intrinsics is not None else [None] * total
        sources = sources if sources is not None else ["batch"] * total
        timestamps = timestamps if timestamps is not None else [time.time()] * total
        if any(len(values) != total for values in (depths, intrinsics, sources, timestamps)):
            raise ValueError("Frame, depth, intrinsics, source, and timestamp batches must have equal lengths")

        prepared = [self.prepare_input(frame, camera) for frame, camera in zip(frames, intrinsics)]
        frames = [item[0] for item in prepared]
        intrinsics = [item[1] for item in prepared]
        started = time.perf_counter()
        with self.inference_lock:
            detections_batch = self.detector.detect_batch(frames)
            predictions = (
                self.depth_estimator.estimate_batch(frames)
                if self.depth_estimator is not None
                else [None] * total
            )
        elapsed_per_frame_ms = (time.perf_counter() - started) * 1000.0 / total
        with self.lock:
            results: list[FrameAnalysis] = []
            for frame, depth, camera, source, timestamp, detections, prediction in zip(
                frames, depths, intrinsics, sources, timestamps, detections_batch, predictions, strict=True
            ):
                result = self._assemble(
                    frame,
                    depth,
                    camera,
                    source,
                    timestamp,
                    detections,
                    prediction,
                    elapsed_per_frame_ms,
                    persist,
                )
                results.append(result)
            return results

    def process_precomputed(
        self,
        frame: np.ndarray,
        *,
        detections: list[Detection],
        predicted_depth: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        source: str = "live",
        timestamp: float | None = None,
        inference_ms: float = 0.0,
        persist: bool = True,
        peer_bag_present: bool = False,
        peer_box_present: bool = False,
    ) -> FrameAnalysis:
        """Assemble a result from inference shared across camera stations."""
        with self.lock:
            return self._assemble(
                frame,
                depth_m,
                intrinsics,
                source,
                time.time() if timestamp is None else float(timestamp),
                detections,
                predicted_depth,
                inference_ms,
                persist,
                peer_bag_present,
                peer_box_present,
            )

    def _apply_box_cuboid(self, detection: Detection, cuboid: BoxVolumeMeasurement, warnings: list[str]) -> None:
        """Write one `BoxVolumeMeasurement` (single-frame or track-aggregated
        -- see `aggregate_box_measurements`) onto `detection`, including
        template matching and the same implausibility ceiling every other
        measurement method respects. Factored out of the box-family
        measurement block in `_assemble` so it can be called a second time,
        with an aggregated result, once this frame's detection has a known
        track (see the post-tracking loop below `self.tracker.update()`)."""
        detection.box_length_mm = round(cuboid.length_mm, 2)
        detection.box_width_mm = round(cuboid.width_mm, 2)
        detection.box_height_mm = round(cuboid.height_mm, 2)
        detection.box_volume_confidence = round(cuboid.volume_confidence, 4)
        detection.box_volume_flags = cuboid.flags
        detection.box_frames_considered = cuboid.frames_considered
        detection.box_frames_accepted = cuboid.frames_accepted
        detection.box_dimension_std_mm = cuboid.dimension_std_mm
        # Keep the new general dimension contract aligned with the legacy box
        # fields so API/dashboard consumers have one place to read L/W/H.
        detection.footprint_length_mm = round(cuboid.length_mm, 2)
        detection.footprint_width_mm = round(cuboid.width_mm, 2)
        detection.physical_height_mm = round(cuboid.height_mm, 2)
        detection.dimension_confidence = round(cuboid.volume_confidence, 4)
        detection.dimension_flags = cuboid.flags
        detection.dimension_method = "realsense_table_relative_cuboid"
        template_match = match_box_template(
            cuboid.length_mm, cuboid.width_mm, cuboid.height_mm, self.box_templates,
        )
        # A reported liters figure this round must still respect the same
        # implausibility ceiling as every other measurement method -- the
        # cuboid math is far more principled than the per-pixel sum it
        # replaces here, but it is still a single-view estimate from real,
        # noisy depth, and a badly leaking mask can still push it past
        # what's physically possible for this bin.
        if template_match is not None:
            detection.box_template_id = template_match.template.id
            detection.box_template_nominal_volume_liters = (
                template_match.template.nominal_volume_liters
            )
            if template_match.template.nominal_volume_liters <= self.config.realsense_max_item_volume_l:
                detection.realsense_volume_l = round(
                    template_match.template.nominal_volume_liters, 6
                )
                detection.measurement_method = "table-relative-cuboid-template"
                detection.measurement_quality = "template-matched"
                detection.height_above_baseline_cm = round(cuboid.height_mm / 10.0, 1)
        elif cuboid.volume_liters <= self.config.realsense_max_item_volume_l:
            detection.realsense_volume_l = round(cuboid.volume_liters, 6)
            detection.measurement_method = cuboid.volume_method
            detection.measurement_quality = (
                "high" if cuboid.volume_confidence >= 0.65
                else "moderate" if cuboid.volume_confidence >= 0.35
                else "low"
            )
            detection.height_above_baseline_cm = round(cuboid.height_mm / 10.0, 1)
        else:
            detection.measurement_quality = "rejected-implausible-volume"
            detection.realsense_volume_l = None
            warnings.append(
                f"Rejected implausible RealSense box volume {cuboid.volume_liters:.1f} L "
                "from the table-relative cuboid measurement; the table-plane fit or object "
                "mask is likely contaminated"
            )

    def _assemble(
        self,
        frame: np.ndarray,
        depth_m: np.ndarray | None,
        intrinsics: CameraIntrinsics | None,
        source: str,
        timestamp: float,
        detections: list[Detection],
        predicted_depth: np.ndarray | None,
        inference_ms: float,
        persist: bool,
        peer_bag_present: bool = False,
        peer_box_present: bool = False,
    ) -> FrameAnalysis:
        warnings: list[str] = []
        if self.config.operating_mode == "geometry_validation":
            warnings.append(
                "Geometry validation mode is active; waste history and auto-deposit are disabled"
            )
        # Box-cuboid multi-frame aggregation bookkeeping (see the box-family
        # measurement block below and the post-tracking loop that consumes
        # these): keyed by `id(detection)` because `detection.track_id` is
        # not assigned until `self.tracker.update()` runs, later in this
        # same method call, well after every detection's own single-frame
        # cuboid measurement has already been computed.
        pending_box_attempted: set[int] = set()
        pending_box_measurements: dict[int, "BoxVolumeMeasurement"] = {}
        # Per-frame shape-router result, keyed like the cuboid stash above and
        # folded into the track's `GeometryLock` once tracking has run.
        pending_shapes: dict[int, ShapeGeometry] = {}
        pending_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if depth_m is not None and depth_m.shape != frame.shape[:2]:
            warnings.append("RealSense depth is not aligned to the RGB frame; hardware volume was skipped")
            depth_m = None
        self._validate_saved_baseline(frame, warnings)
        calibrated_prediction = (
            self.calibration.apply(predicted_depth)
            if predicted_depth is not None and self.calibration is not None
            else predicted_depth
        )
        bin_region = self._measurement_region(frame.shape)
        minimum_height_m = (
            self.config.geometry_validation_min_object_height_m
            if self.config.operating_mode == "geometry_validation" and self.camera_id != "logitech"
            else self.config.min_object_height_m
        )
        # Playbook section 12: a disallowed object is a mis-sort the operator
        # needs told about. `filter_waste_detections` below drops it -- rightly,
        # since it must never be tracked, measured or written to the ledger --
        # so the verdict is taken here, while the detector's own labels are
        # still in hand, and surfaced as a warning instead of vanishing.
        mis_sorted = sorted({
            family
            for detection in detections
            if detection.confidence >= self.config.detector_confidence
            and (family := mis_sort_family(detection.label)) is not None
        })
        logitech = self.camera_id == "logitech"
        counters = self.stage_counters
        counters["frames_processed"] += 1
        counters["raw_detections"] += len(detections)
        rejections: Counter = Counter()
        detections = filter_waste_detections(
            detections, frame.shape, bin_region, self.config, rejections=rejections,
            # Logitech-only: a can is a few hundred pixels in a 640x480 C920 frame.
            min_pixels=self.config.logitech_min_object_pixels if logitech else None,
            confidence=self.config.logitech_detector_confidence if logitech else None,
        )
        counters.update({f"rejected_{key}": value for key, value in rejections.items()})
        counters["after_class_confidence_roi_area"] += len(detections)
        self.last_stage_rejections = dict(rejections)
        for family in mis_sorted:
            warnings.append(f"MIS-SORT: a {family} object was detected; this bin does not accept it")

        # Logitech measurement cascade. Mode 1 is the calibrated path; without
        # an empty baseline Mode 2 uses the measured camera-to-floor distance,
        # and Mode 3 takes the support plane from the ROI's own median depth.
        # Missing intrinsics fall back to the configured field of view, so a
        # tracked object is never left without geometry to measure it with.
        measure_intrinsics = intrinsics or self.latest_intrinsics
        # Whether the *whole* calibrated chain can actually run -- not merely
        # whether a calibration object exists. A restored profile whose
        # monocular reference or support plane is missing has a calibration and
        # still cannot measure, which is what left every object pending.
        full_metric_ready = self._full_metric_ready(calibrated_prediction, measure_intrinsics)
        uncalibrated_logitech = (
            self.camera_id == "logitech" and predicted_depth is not None and not full_metric_ready
            # The hard mounting-tilt limit is a safety rule, not a calibration
            # step: past it the bin floor is barely visible and no mode may
            # report litres.
            and not self._logitech_tilt_invalid()
        )
        if uncalibrated_logitech:
            if measure_intrinsics is None:
                measure_intrinsics = self._field_of_view_intrinsics(frame.shape)
            source_depth = (
                calibrated_prediction
                if calibrated_prediction is not None and calibrated_prediction.shape == predicted_depth.shape
                else predicted_depth
            )
            source_depth = self._metric_from_relative(source_depth, bin_region)
            plane_values = source_depth[bin_region & np.isfinite(source_depth) & (source_depth > 0.1)]
            uncalibrated_logitech = measure_intrinsics is not None and plane_values.size >= 100
            if uncalibrated_logitech:
                calibrated_prediction = source_depth
                measured_distance = self.config.logitech_reference_distance_m
                plane_distance = measured_distance if measured_distance > 0 else float(np.median(plane_values))
                self._uncalibrated_plane = axis_aligned_plane(measure_intrinsics, plane_distance)
                self.calibration_mode = (
                    "reference-distance-estimate" if measured_distance > 0 else "uncalibrated-estimate"
                )
        if self.camera_id == "logitech":
            mask_debug: dict[str, Any] = {}
            depth_change = None
            if calibrated_prediction is not None and self.reference_monocular is not None \
                    and calibrated_prediction.shape == self.reference_monocular.shape:
                rise = self.reference_monocular.astype(np.float32) - calibrated_prediction
                depth_change = np.isfinite(rise) & (rise >= self.config.logitech_min_object_height_m)
            detections, segmentation_warnings = bound_logitech_detections(
                frame,
                self.reference_rgb,
                detections,
                bin_region,
                # Scaled to the frame, so a small can is not filtered out as
                # speckle by a pixel count chosen for a larger sensor.
                min_pixels=minimum_object_pixels(
                    frame.shape[:2],
                    min(self.config.min_component_pixels, self.config.logitech_min_object_pixels),
                ),
                foreground_threshold=self.config.foreground_threshold,
                max_scene_fraction=self.config.logitech_max_scene_fraction,
                max_expansion=self.config.logitech_max_mask_expansion,
                duplicate_overlap=self.config.logitech_duplicate_overlap,
                depth_change=depth_change,
                debug=mask_debug,
            )
            self.logitech_mask_debug = {"frame": frame, **mask_debug}
            # Startup, without asking the operator for anything: the floor plane
            # gives the camera height, and a still, empty view gives the
            # baseline. Both run every frame and both are cheap -- the plane is
            # cached, and the learner only counts.
            self._auto_camera_height(self._logitech_floor_plane(
                calibrated_prediction, measure_intrinsics, None, bin_region,
            ))
            self._maybe_learn_empty_baseline(
                frame, calibrated_prediction, bin_region, detections,
            )
            detections.extend(self._recovered_foreground_detections(
                frame, detections, mask_debug.get("foreground"), bin_region,
            ))
            counters["valid_masks"] += len(detections)
            counters["detector_only_masks"] += int(mask_debug.get("detector_only_masks", 0))
            for reason in mask_debug.get("reasons") or []:
                if reason != "detector_mask_without_foreground_verification":
                    counters[f"mask_rejected_{reason}"] += 1
            warnings.extend(segmentation_warnings)
            if self.config.logitech_stabilize_depth:
                calibrated_prediction, self.last_background_stabilization = stabilize_background_depth(
                    calibrated_prediction,
                    self.reference_monocular,
                    frame,
                    self.reference_rgb,
                    detections,
                    bin_region,
                    foreground_threshold=self.config.foreground_threshold,
                )
        scene_depth = calibrated_prediction if self.camera_id == "logitech" else depth_m
        scene_reference = self.reference_monocular if self.camera_id == "logitech" else self.reference_realsense

        if self.reference_rgb is not None:
            # Logitech's scene_depth is Depth-Anything-V2's monocular metric
            # PREDICTION, not a real depth measurement -- its background
            # jitter is much larger than RealSense's real stereo depth, so
            # reusing the shared (RealSense-tuned) min_object_height_m here
            # let background jitter register as "changed" and grow into
            # full-frame phantom boxes (see logitech_scene_min_height_m's
            # definition in config.py). Only this scene-object detection
            # (feeds phantom recovery) is camera-aware; actual volume/height
            # integration for confirmed detections is unaffected.
            scene_min_height_m = (
                self.config.logitech_scene_min_height_m
                if self.camera_id == "logitech"
                else minimum_height_m
            )
            scene_objects = detect_scene_objects(
                frame,
                self.reference_rgb,
                scene_depth,
                scene_reference,
                self.config.roi,
                threshold=self.config.foreground_threshold,
                min_area=self.config.min_component_pixels,
                min_height_m=scene_min_height_m,
                max_height_m=self.config.max_object_height_m,
                measurement_mask=bin_region,
            )
            original_count = len(detections)
            # BUGFIX (round 21): `allow_unclassified` used to also turn True
            # off `self.tracker.has_active_counted_track()` alone -- a plain
            # "is anything, anywhere, already counted?" boolean. Once any
            # real bag/box had been confirmed once, that let the single
            # largest leftover "changed" region ANYWHERE in the frame (a
            # couch cushion, a backpack, an office chair -- whatever a stale
            # baseline made read as "different from empty") be promoted to
            # its own tracked, displayed "unclassified object", drifting
            # across unrelated furniture as different regions won "largest"
            # from frame to frame. `counted_track_boxes()` below is passed
            # through instead so `fuse_scene_detections` can require the
            # candidate region be near an *already-counted track's own box*
            # -- bridging that specific object through a brief detector
            # dropout, never licensing an unrelated new one elsewhere.
            detections = fuse_scene_detections(
                frame,
                detections,
                scene_objects,
                # A peer camera saying "bag" is not enough to turn an
                # arbitrary changed-depth region into a new waste object.
                # The supplied real-hardware results showed pillows,
                # backpacks, bottles and empty patches entering through this
                # exact gate.  Unclassified regions may still bridge a nearby
                # already-counted track via counted_track_boxes below.
                allow_unclassified=self.config.allow_unclassified_foreground,
                bag_only=self.config.bag_only,
                require_scene_match=(
                    scene_depth is not None
                    and scene_reference is not None
                ),
                counted_track_boxes=self.tracker.counted_track_boxes(),
            )
            # Scene fusion can still hand back more than one box for the same
            # physical container (an unmatched fragment next to a fused
            # neural match, or two fused scene objects whose merged masks
            # still touch). Collapse those before tracking so one bag is
            # never counted or displayed twice.
            detections = deduplicate_overlapping_detections(detections, frame.shape[:2])
            if scene_objects and not original_count and (
                self.config.allow_unclassified_foreground
                or self.tracker.counted_track_boxes()
            ):
                warnings.append(
                    "Bag silhouette was recovered from fixed-bin depth; its semantic label was not confirmed"
                    if self.config.bag_only
                    else "Unclassified foreground was measured without a confirmed waste-object label"
                )
            elif scene_objects and not original_count:
                LOGGER.debug("Ignored %s unclassified foreground components", len(scene_objects))
            elif scene_objects and any(item.source == "yoloe-scene-fusion" for item in detections):
                LOGGER.debug("Fused %s neural detections with %s complete scene objects", original_count, len(scene_objects))
            for detection in detections:
                if not _is_phantom_detection(detection):
                    detection.accepted_class = accepted_object_class(
                        detection.label, self.config.operating_mode,
                    )
        if not detections and self.baseline_rgb is None:
            warnings.append("No object detected; capture an empty-scene baseline to enable fallback and volume")
        elif detections and self.baseline_rgb is None:
            warnings.append("Capture an empty-bin baseline to recover complete garbage bags and measure liters")

        intrinsics = intrinsics or self.latest_intrinsics
        if intrinsics is not None:
            self.latest_intrinsics = intrinsics
        if self.reference_plane is None and intrinsics is not None:
            floor = self.reference_monocular if self.camera_id == "logitech" else self.reference_realsense
            if floor is not None:
                fitted_plane = fit_reference_plane(floor, intrinsics, mask=bin_region)
                if reference_plane_is_usable(fitted_plane):
                    self.reference_plane = fitted_plane
                    self._calibration_id = uuid4().hex[:12]
                else:
                    warnings.append(
                        "The captured reference does not contain one reliable support plane; "
                        "metric dimensions were withheld"
                    )
        object_mask = combined_mask(detections, frame.shape[:2]) if detections else None

        # BUGFIX (round 24) -- the root blocker behind every "pending -
        # pending empty baseline" cell reported from real hardware, and the
        # reason several rounds of correct geometry work changed nothing on
        # the actual rig.
        #
        # Until now BOTH the support plane and the reference surface every
        # volume method measures against could only come from a separately
        # captured EMPTY-SCENE baseline. With no baseline captured,
        # `reference_plane` stayed None (so `estimate_box_volume_cuboid()`
        # returned None on its first guard) and `reference_realsense` stayed
        # None (so `estimate_volume()` had nothing to subtract) -- every
        # liters cell read "pending" forever, however correct the maths
        # underneath was. Capturing that baseline requires the scene to be
        # genuinely empty, which a real room with the object already in
        # shot never is, so on this rig the gate simply never opened.
        #
        # The build spec's section 9.1 prescribes the alternative directly:
        # fit the support plane from the live frame's own background points,
        # excluding the object masks with a ~10-15% margin. The floor
        # visible AROUND the object is real, measured, and always present.
        # From that plane the "empty floor" reference depth is then computed
        # in closed form rather than captured (`synthesize_plane_depth`),
        # which unblocks the per-pixel bag path too.
        #
        # A real captured baseline is still strictly preferred and is never
        # overwritten -- this only fills in when none exists. Results
        # measured this way are flagged `live_fitted_support_plane` so the
        # diagnostics stay honest about where the reference came from.
        captured_support_plane = self.reference_plane is not None and (
            self.reference_realsense is not None
            if self.camera_id != "logitech" else self.reference_monocular is not None
        )
        measurement_plane = self.reference_plane
        self.support_plane_source = "captured-baseline" if captured_support_plane else "none"
        effective_reference = self.reference_realsense
        if (
            self.camera_id != "logitech"
            and depth_m is not None
            and intrinsics is not None
            and (self.reference_plane is None or self.reference_realsense is None)
        ):
            live_plane = fit_support_plane_from_background(
                depth_m,
                intrinsics,
                object_mask=object_mask,
                region_mask=bin_region,
            )
            if reference_plane_is_usable(live_plane):
                # Use this one plane consistently for height, footprint and
                # synthetic reference depth in the current measurement.  The
                # old path kept the first live plane for height while creating
                # later synthetic references from newly-fitted planes.
                measurement_plane = live_plane
                self.reference_plane = live_plane
                self.support_plane_source = "live-frame-background"
                if effective_reference is None:
                    synthetic = synthesize_plane_depth(
                        depth_m.shape, intrinsics, measurement_plane.coefficients
                    )
                    if synthetic is not None:
                        effective_reference = synthetic
                        self.support_plane_source = "live-frame-background"
                        warnings.append(
                            "Measured against a support plane fitted from this frame's own background; "
                            "capture an empty-scene baseline for the most accurate results"
                        )
            elif live_plane is not None:
                warnings.append(
                    f"Rejected an unreliable live support plane "
                    f"({live_plane.residual_rmse_m * 1000.0:.1f} mm RMSE); "
                    "capture a genuinely empty-scene baseline before trusting dimensions"
                )
        # `hardware_total`/`monocular_total` (the per-camera aggregate liters
        # figure rendered as the dashboard's "CURRENT VOLUME" metric, and --
        # before the fix above -- also `calibrate_known_volume()`'s implicit
        # fallback) must never let an unconfirmed, zero-neural-confidence
        # phantom detection (see `_is_phantom_detection` above) blend its own
        # silhouette volume into a real confirmed object's number. This
        # project's own history (rounds 5-8) shows a phantom coexisting with
        # a real detection in the same frame is a routine occurrence, not an
        # edge case. When at least one confirmed detection is present,
        # `aggregate_detections` excludes every phantom; when the *only*
        # thing in view is a phantom silhouette, it still measures that
        # silhouette alone (unchanged from the existing, tested "recovered
        # bag silhouette measured honestly" behavior for an isolated
        # unconfirmed foreground region -- there is nothing for it to
        # contaminate in that case). `combined_mask()` safely returns an
        # all-False mask for an empty list, and `estimate_volume()` rejects
        # an all-False mask via its own pixel-count floor.
        deposit_change, deposit_rise = self._committed_scene_change(
            frame, depth_m, calibrated_prediction,
        )
        measurement_masks: dict[int, np.ndarray] = {}
        recovered_measurement_ids: set[int] = set()
        if (
            self.camera_id != "logitech"
            and depth_m is not None
            and intrinsics is not None
            and reference_plane_is_usable(measurement_plane)
        ):
            for detection in detections:
                if detection.accepted_class is None or _is_phantom_detection(detection):
                    continue
                seed = combined_mask([detection], frame.shape[:2])
                recovered = recover_elevated_object_mask(
                    depth_m,
                    intrinsics,
                    seed,
                    measurement_plane,
                    measurement_mask=bin_region,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                    min_points=min(60, self.config.min_component_pixels),
                )
                if recovered is not None and int(np.count_nonzero(recovered)) > int(np.count_nonzero(seed) * 1.08):
                    measurement_masks[id(detection)] = recovered
                    recovered_measurement_ids.add(id(detection))
                    # Re-evaluate colour on the full physical surface. This
                    # avoids a red logo, carpet halo, or small shaded centre
                    # patch deciding the colour of an otherwise white bag.
                    detection.color, detection.color_confidence = classify_color(
                        frame, recovered
                    )

        confirmed_detections = [item for item in detections if not _is_phantom_detection(item)]
        aggregate_detections = confirmed_detections if confirmed_detections else detections
        confirmed_object_mask = np.zeros(frame.shape[:2], dtype=bool)
        for item in aggregate_detections:
            confirmed_object_mask |= measurement_masks.get(
                id(item), combined_mask([item], frame.shape[:2]),
            )
        logitech_ready = self.camera_id != "logitech" or self._logitech_measurement_ready()
        provisional_logitech = (
            self.camera_id == "logitech" and self.calibration_mode == "model-metric-unverified"
        )
        precision = {
            "geometry_mode": self.config.volume_geometry,
            "calibration_factor": self.config.volume_calibration_factor,
            "systematic_error_fraction": max(
                self.config.systematic_error_fraction,
                self.config.logitech_provisional_systematic_error_fraction
                if provisional_logitech else 0.0,
                self._logitech_tilt_uncertainty_fraction(),
            ),
            "baseline_noise_m": self.baseline_noise_m,
            "baseline_noise_map": self.baseline_noise_map,
            "noise_sigma": self.config.depth_noise_sigma,
            "reject_outliers": self.config.reject_depth_outliers,
            "grid_size_m": self.config.volume_grid_size_m,
            "min_points_per_cell": self.config.volume_min_points_per_cell,
            "cell_height_percentile": self.config.volume_cell_height_percentile,
        }

        hardware_total = estimate_volume(
            depth_m,
            effective_reference,
            intrinsics,
            object_mask=confirmed_object_mask,
            roi=self.config.roi,
            min_height_m=minimum_height_m,
            max_height_m=self.config.max_object_height_m,
            min_pixels=min(50, self.config.min_component_pixels),
            method="realsense-aligned-depth",
            measurement_mask=bin_region,
            depth_noise_m=self.config.depth_noise_m,
            reference_plane=measurement_plane,
            **precision,
        ) if detections and self.camera_id != "logitech" else None

        occupancy_depth = calibrated_prediction if self.camera_id == "logitech" else depth_m
        occupancy_reference = self.baseline_monocular if self.camera_id == "logitech" else self.baseline_realsense
        occupancy_mask = None
        if self.camera_id == "logitech":
            occupancy_mask = np.zeros(frame.shape[:2], dtype=bool)
            if self._occupied_logitech_mask is not None and self._occupied_logitech_mask.shape == frame.shape[:2]:
                occupancy_mask |= self._occupied_logitech_mask
            if object_mask is not None:
                occupancy_mask |= object_mask
        bin_total = estimate_volume(
            occupancy_depth,
            occupancy_reference,
            intrinsics,
            object_mask=occupancy_mask,
            roi=self.config.roi,
            min_height_m=minimum_height_m,
            max_height_m=self.config.max_object_height_m,
            min_pixels=min(50, self.config.min_component_pixels),
            method=f"{self.camera_id}-total-bin-occupancy",
            measurement_mask=bin_region,
            depth_noise_m=self.config.depth_noise_m,
            reference_plane=measurement_plane,
            **precision,
        ) if occupancy_reference is not None and logitech_ready else None
        # One whole-bin height grid per frame, shared by the deposit-isolation
        # logic below. It is the same computation `bin_total` already performs,
        # kept as a grid so a committed scene can be differenced against it.
        scene_grid = None
        if (
            self.camera_id != "logitech"
            and depth_m is not None
            and intrinsics is not None
            and measurement_plane is not None
            and measurement_plane.coefficients is not None
        ):
            scene_grid = integrate_height_map(
                depth_m,
                intrinsics,
                plane_coefficients=measurement_plane.coefficients,
                mask=bin_region,
                settings=HeightMapSettings(
                    grid_size_m=self.config.volume_grid_size_m,
                    min_height_m=max(minimum_height_m, 1e-4),
                    max_height_m=self.config.max_object_height_m,
                    min_points_per_cell=self.config.volume_min_points_per_cell,
                    cell_height_percentile=self.config.volume_cell_height_percentile,
                    min_valid_depth_fraction=0.0,
                    max_fill_fraction=1.0,
                ),
            )
        if hardware_total is not None and hardware_total.coverage_ratio < self.config.minimum_depth_coverage:
            warnings.append(
                f"Only {hardware_total.coverage_ratio * 100:.0f}% of the bag has valid depth; "
                "the volume is uncertain"
            )

        if detections and self.camera_id != "logitech":
            depth_signal = summarize_depth_signal(depth_m)
            if depth_m is None:
                warnings.append("No RealSense depth is arriving; start the Raspberry Pi bridge with --source realsense")
            elif depth_signal["valid_pixels"] == 0:
                warnings.append("The RealSense depth image contains no valid distances; check the camera and USB 3 connection")
            elif intrinsics is None:
                warnings.append("RealSense camera intrinsics are missing; liters cannot be calculated")
            elif self.baseline_rgb is not None and self.baseline_realsense is None:
                warnings.append("The baseline has no RealSense depth; remove the object and capture the empty baseline again")
            elif self.baseline_realsense is not None and hardware_total is None:
                warnings.append(
                    "No measurable height above the empty baseline; remove the object, recapture the baseline, "
                    "then put the object back without moving the camera"
                )

        monocular_total = None
        if predicted_depth is not None:
            if detections and self.calibration is not None and logitech_ready:
                monocular_total = estimate_volume(
                    calibrated_prediction,
                    self.reference_monocular,
                    intrinsics,
                    object_mask=confirmed_object_mask,
                    roi=self.config.roi,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                    min_pixels=min(50, self.config.min_component_pixels),
                    method=("logitech-depth-anything-v2" if self.camera_id == "logitech"
                            else "monocular-depth-calibrated-against-realsense"),
                    measurement_mask=bin_region,
                    depth_noise_m=self.config.depth_noise_m,
                    reference_plane=measurement_plane,
                    **precision,
                )
            elif detections and self.baseline_monocular is not None and self.camera_id != "logitech":
                warnings.append(
                    "Monocular volume was withheld because an aligned RealSense depth calibration is missing"
                )
        if detections and self.camera_id == "logitech":
            if intrinsics is None:
                warnings.append("Logitech camera intrinsics are missing; calibrate its lens or configure its field of view")
            if (
                self.config.logitech_require_reference
                and self.calibration_mode != "independent-measured-distance"
                and not self.config.logitech_allow_provisional_metric
            ):
                warnings.append(
                    "Logitech liters are blocked until its own measured camera-to-empty-bin distance is entered"
                )
            elif self.calibration_mode == "model-metric-unverified":
                warnings.append(
                    "Logitech liters are provisional and include a large scale uncertainty; "
                    "enter a measured empty-bin distance for thesis-grade results"
                )
            if self._logitech_tilt_invalid():
                warnings.append(
                    f"Logitech mounting tilt {self.reference_plane.tilt_degrees:.1f}° exceeds "
                    f"the {self.config.logitech_hard_max_tilt_degrees:.1f}° hard limit; the bin floor "
                    "is barely visible at this angle, so liters cannot be reported. Mount the camera "
                    "above the bin and restrict its region to the bin floor"
                )
            elif self.reference_plane is not None and self.reference_plane.tilt_degrees > self.config.logitech_max_tilt_degrees:
                tilt_penalty_pct = self._logitech_tilt_uncertainty_fraction() * 100
                warnings.append(
                    f"Logitech mounting tilt {self.reference_plane.tilt_degrees:.1f}° is above the "
                    f"{self.config.logitech_max_tilt_degrees:.1f}° confident-mounting zone; liters are still "
                    f"reported but with roughly {tilt_penalty_pct:.0f}% added uncertainty from the steeper "
                    "angle. Mount closer to overhead for tighter numbers."
                )
            if self.baseline_rgb is not None and self.baseline_monocular is None:
                warnings.append("Capture an empty Logitech baseline before comparing volumes")

        for detection in detections:
            instance_mask = measurement_masks.get(
                id(detection), combined_mask([detection], frame.shape[:2]),
            )
            # The detector says what this is; the change since the committed
            # scene says where it ends. A mask that spilled across the floor,
            # a second carton and a chair is cut back to the island that was
            # actually deposited, and background is refused outright.
            instance_mask = self._deposit_measurement_mask(
                detection, instance_mask, bin_region, deposit_change, deposit_rise,
            )
            logitech_height_coherent = True
            if depth_m is not None:
                valid_distance = instance_mask & np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 20.0)
                if np.any(valid_distance):
                    detection.depth_distance_m = round(float(np.median(depth_m[valid_distance])), 3)
                if effective_reference is not None and effective_reference.shape == depth_m.shape:
                    height_m = effective_reference.astype(np.float32) - depth_m.astype(np.float32)
                    valid_height = (
                        valid_distance
                        & np.isfinite(effective_reference)
                        & (height_m >= minimum_height_m)
                        & (height_m <= self.config.max_object_height_m)
                    )
                    if np.any(valid_height):
                        detection.height_above_baseline_cm = round(float(np.median(height_m[valid_height]) * 100.0), 1)
            if calibrated_prediction is not None:
                valid_prediction = (
                    instance_mask
                    & np.isfinite(calibrated_prediction)
                    & (calibrated_prediction > 0.10)
                    & (calibrated_prediction < 20.0)
                )
                if np.any(valid_prediction):
                    detection.monocular_distance_m = round(float(np.median(calibrated_prediction[valid_prediction])), 3)
                    if self.camera_id == "logitech" and self.reference_monocular is not None:
                        relative_height = self.reference_monocular - calibrated_prediction
                        positive_height = valid_prediction & (
                            relative_height >= minimum_height_m
                        )
                        valid_height = positive_height & (
                            relative_height <= self.config.max_object_height_m
                        )
                        positive_pixels = int(np.count_nonzero(positive_height))
                        valid_pixels = int(np.count_nonzero(valid_height))
                        logitech_height_coherent = (
                            positive_pixels == 0
                            or valid_pixels / positive_pixels >= self.config.logitech_min_valid_height_fraction
                        )
                        if not logitech_height_coherent and logitech_ready:
                            warnings.append(
                                "Rejected Logitech object height beyond the physical limit; "
                                "check the measured distance, camera angle, and empty baseline"
                            )
                        if np.any(valid_height) and logitech_ready and logitech_height_coherent:
                            detection.height_above_baseline_cm = round(
                                float(np.median(relative_height[valid_height]) * 100), 1
                            )
            individual = estimate_volume(
                depth_m,
                effective_reference,
                intrinsics,
                # Always use the resolved instance region.  Some detectors
                # provide a box without a segmentation mask; passing None here
                # previously integrated the entire camera ROI as one object.
                object_mask=instance_mask,
                roi=self.config.roi,
                min_height_m=minimum_height_m,
                max_height_m=self.config.max_object_height_m,
                min_pixels=min(25, self.config.min_component_pixels),
                method="realsense-instance",
                measurement_mask=bin_region,
                depth_noise_m=self.config.depth_noise_m,
                reference_plane=measurement_plane,
                **precision,
            )
            if individual is not None:
                detection.depth_coverage_percent = round(individual.coverage_ratio * 100, 1)
                detection.volume_uncertainty_l = round(individual.uncertainty_l, 6)
                detection.measurement_method = individual.method
                detection.measurement_quality = individual.quality
                detection.calibration_mode = "factory-depth-plus-known-volume" if (
                    self.config.volume_calibration_factor != 1.0
                ) else "factory-calibrated-stereo-depth"
                foreground_fraction = individual.valid_pixels / max(1, individual.candidate_pixels)
                if foreground_fraction < self.config.minimum_foreground_fraction:
                    detection.measurement_quality = "rejected-sparse-height-inside-mask"
                    warnings.append(
                        "Rejected a volume whose measured height occupied too little of the detection mask"
                    )
                elif individual.liters > self.config.realsense_max_item_volume_l:
                    detection.measurement_quality = "rejected-implausible-volume"
                    warnings.append(
                        f"Rejected implausible RealSense item volume {individual.liters:.1f} L; "
                        "the empty reference or object mask is contaminated"
                    )
                else:
                    detection.realsense_volume_l = round(individual.liters, 6)
                    # Replace the raw median-over-the-whole-mask height above
                    # (line ~898) with the 90th-percentile height from this
                    # same accepted, hole-filled, outlier-rejected column
                    # field -- the exact pixels the liters figure came from.
                    # A dome-shaped or tapered object (a pillow, a slouched
                    # bag) has many low-height edge pixels that drag a
                    # whole-mask median far below the object's true, ruler-
                    # measured height; the near-top percentile does not.
                    detection.height_above_baseline_cm = round(individual.height_p90_m * 100.0, 1)
            # General RealSense-only physical dimensions for every accepted
            # plastic bag, paper bag, or cardboard box.  Bags expose a visible
            # support-plane footprint rather than pretending to be cuboids.
            dimensions = None
            if (
                self.camera_id != "logitech"
                and detection.accepted_class is not None
                and not _is_phantom_detection(detection)
            ):
                dimensions = estimate_object_dimensions(
                    depth_m,
                    intrinsics,
                    instance_mask,
                    measurement_plane,
                    measurement_mask=bin_region,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                    min_points=min(60, self.config.min_component_pixels),
                )
            if dimensions is not None:
                # Median over a short window, so one badly segmented frame
                # cannot decide the reported size. This is what stops a 2 cm
                # cream box reading 4 cm on the odd frame.
                smoothed_length, smoothed_width, smoothed_height = self._dimension_smoother.update(
                    detection.track_id,
                    dimensions.length_mm, dimensions.width_mm, dimensions.height_mm,
                )
                detection.footprint_length_mm = smoothed_length
                detection.footprint_width_mm = smoothed_width
                detection.physical_height_mm = smoothed_height
                detection.dimension_confidence = round(dimensions.confidence, 4)
                extra_dimension_flags = (
                    ("live_fitted_support_plane",)
                    if self.support_plane_source == "live-frame-background" else ()
                )
                if id(detection) in recovered_measurement_ids:
                    extra_dimension_flags += ("support_plane_mask_recovered",)
                detection.dimension_flags = tuple(dimensions.flags) + extra_dimension_flags
                detection.dimension_method = dimensions.method
                # Use the same plane-relative, elevated-point height shown in
                # the dimension triplet, not a camera-Z mask median.
                detection.height_above_baseline_cm = round(dimensions.height_mm / 10.0, 1)
            if dimensions is not None:
                plane_points = object_plane_points(
                    depth_m, intrinsics, instance_mask, measurement_plane,
                    measurement_mask=bin_region,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                )
                if plane_points is not None and self.diagnostics.enabled:
                    pending_points[id(detection)] = plane_points
                if plane_points is not None:
                    # Mesh volume is the height-map integral set just above,
                    # before any cuboid override below replaces it.
                    pending_shapes[id(detection)] = measure_shape(
                        *plane_points,
                        mesh_volume_l=detection.realsense_volume_l,
                        height_mm=dimensions.height_mm,
                        label=detection.label,
                    )
            # Table-relative cuboid measurement for box-family detections
            # (Revised Dual-Camera Volume Estimation recipe). RealSense only
            # -- Logitech never supplies metric geometry (PDF hard
            # requirement #2). This is a targeted override, not a
            # replacement of the block above: `individual` (the per-pixel
            # height*area sum) still runs and still populates the fallback
            # bag/general-object measurement; a box-family detection's own
            # `realsense_volume_l`/`height_above_baseline_cm` are only
            # overwritten when `estimate_box_volume_cuboid()` itself
            # succeeds, so a box with too little valid depth or no fitted
            # table plane still gets the pre-existing per-pixel behaviour
            # rather than silently losing its measurement.
            # A label from the other camera cannot safely promote an arbitrary
            # RealSense changed-depth blob: the two views are not pixel-
            # registered, so presence in the same frame is not object identity.
            # Require this RealSense detection itself to be an accepted
            # cardboard class before producing physical cuboid geometry.
            if (
                self.camera_id != "logitech"
                and (
                    detection.accepted_class == "cardboard_box"
                    or (
                        self.config.operating_mode == "geometry_validation"
                        and bool(
                            set(_normalized_label(detection.label).split())
                            & {"box", "boxes", "carton", "cartons", "parcel", "parcels", "package"}
                        )
                    )
                )
                and not _is_phantom_detection(detection)
                and depth_m is not None
                # A "cardboard box" made of fabric is measured as what it is:
                # the height-map path below, not a forced cuboid.
                and (conflict := classification_conflict(detection)) is None
            ):
                cuboid = estimate_box_volume_cuboid(
                    depth_m,
                    intrinsics,
                    instance_mask,
                    measurement_plane,
                    measurement_mask=bin_region,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                )
                if cuboid is not None:
                    extra_flags: tuple[str, ...] = ()
                    if self.support_plane_source == "live-frame-background":
                        extra_flags += ("live_fitted_support_plane",)
                    if id(detection) in recovered_measurement_ids:
                        extra_flags += ("support_plane_mask_recovered",)
                    if extra_flags:
                        cuboid = replace(cuboid, flags=tuple(cuboid.flags) + extra_flags)
                # `detection.track_id` is not assigned yet at this point in
                # `_assemble` (`self.tracker.update()` runs later, once every
                # detection in this frame has been measured) -- this frame's
                # own single-frame cuboid result is applied immediately below
                # so a value is never hidden while waiting for tracking, and
                # is separately stashed by object identity so the
                # post-tracking step further down (search
                # `pending_box_attempted`) can fold it into that now-known
                # track's multi-frame aggregation history.
                pending_box_attempted.add(id(detection))
                if cuboid is not None:
                    pending_box_measurements[id(detection)] = cuboid
                    self._apply_box_cuboid(detection, cuboid, warnings)
            detection.classification_note = classification_conflict(detection)
            if (
                (self.calibration is not None or uncalibrated_logitech)
                and (logitech_ready or uncalibrated_logitech) and logitech_height_coherent
                # A detector-only mask is never measured as a calibrated
                # result; in the uncalibrated mode it carries the estimate,
                # which is labelled as such everywhere it appears.
                and (detection.source != DETECTOR_ONLY_SOURCE or uncalibrated_logitech)
            ):
                metric_mask: np.ndarray | None = None
                metric_result: Any = None
                if self.camera_id == "logitech":
                    volume_mask = clip_to_region(restore_mask(instance_mask, frame.shape[:2]), bin_region)
                    volume_mask = _tracked_component(volume_mask, detection.box)
                    if _spans_the_background(volume_mask, bin_region):
                        # A mask filling the measurement area and running off
                        # several of its edges is the floor, a sofa or a wall.
                        # Measuring it produced the 13 L "objects"; it is
                        # refused here, after detection, so nothing about the
                        # detector or its masks changes.
                        detection.volume_rejection_reason = "background_region_not_measurable"
                        continue
                    static_reason = static_background_reason(
                        volume_mask, bin_region, change=deposit_change,
                    )
                    if static_reason is not None:
                        # The bed, the table and the floor strip are detected
                        # like anything else. A stale baseline used to let one
                        # through as a 35 L "deposit"; a mask that covers or
                        # spans the measurement zone is the room, not an object
                        # standing in it.
                        detection.volume_rejection_reason = static_reason
                        detection.measurement_quality = static_reason
                        self.stage_counters[f"logitech_static_background_{static_reason}"] += 1
                        continue
                    problems = frame_consistency(
                        frame, volume_mask, calibrated_prediction, measure_intrinsics,
                        region=bin_region,
                        foreign_intrinsics=self.peer_intrinsics,
                    )
                    # Sharing one estimated matrix between two cameras at the
                    # same resolution is legal; measuring with a mask or depth
                    # map from another coordinate system is not.
                    blocking = [item for item in problems
                                if item != "intrinsics_belong_to_the_other_camera"]
                    if blocking:
                        detection.volume_rejection_reason = blocking[0]
                        self.last_logitech_volume_diagnostics = {
                            "reason": blocking[0], "coordinate_problems": problems,
                            "label": detection.label,
                        }
                        continue
                    if self.config.logitech_volume_erode_px > 0:
                        # Classification keeps the full mask; volume uses the
                        # core, because the rim pixel is floor at object depth.
                        import cv2

                        eroded = cv2.erode(
                            volume_mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                            iterations=self.config.logitech_volume_erode_px,
                        ) > 0
                        if int(np.count_nonzero(eroded)) >= self.config.logitech_min_object_pixels:
                            volume_mask = eroded
                    plane_for_volume = measurement_plane
                    if plane_for_volume is None or plane_for_volume.coefficients is None:
                        # The floor this camera actually looks at, fitted from
                        # its own calibrated depth over the part of the zone no
                        # object stands on. Without it the fallback is a plane
                        # perpendicular to the optical axis, which is the floor
                        # only for a camera pointing straight down: on a tilted
                        # mount it reports every height times the cosine of the
                        # tilt. Fitting it here is what makes a floor-relative
                        # height and a floor-plane footprint possible at all.
                        plane_for_volume = self._logitech_floor_plane(
                            calibrated_prediction, measure_intrinsics, object_mask, bin_region,
                        )
                    if plane_for_volume is None or plane_for_volume.coefficients is None:
                        plane_for_volume = self._uncalibrated_plane
                    result = metric_object_volume(
                        calibrated_prediction, measure_intrinsics, volume_mask, plane_for_volume,
                        reference_depth_m=self.reference_monocular,
                        measurement_mask=bin_region,
                        min_height_m=self.config.logitech_min_object_height_m,
                        max_height_m=self.config.max_object_height_m,
                        min_pixels=min(25, self.config.logitech_min_object_pixels),
                        cell_size_m=self.config.logitech_height_map_cell_m,
                        # Measured or derived: geometry only needs the number.
                        camera_height_m=self.camera_height_m(),
                    )
                    individual_mono = result.measurement
                    prediction_values = (
                        predicted_depth[bin_region] if predicted_depth is not None else np.empty(0)
                    )
                    prediction_values = prediction_values[np.isfinite(prediction_values)]
                    self.last_logitech_volume_diagnostics = {
                        **result.diagnostics,
                        "reason": result.reason,
                        "label": detection.label,
                        "frame_resolution": [int(frame.shape[1]), int(frame.shape[0])],
                        "calibration_resolution": (
                            None if self.calibration is None or self.calibration.resolution is None
                            else list(self.calibration.resolution)
                        ),
                        "inference_resolution": (
                            None if predicted_depth is None
                            else [int(predicted_depth.shape[1]), int(predicted_depth.shape[0])]
                        ),
                        "inference_to_frame_scale": (
                            None if predicted_depth is None
                            else round(frame.shape[1] / predicted_depth.shape[1], 6)
                        ),
                        "intrinsics": None if intrinsics is None else {
                            "fx": intrinsics.fx, "fy": intrinsics.fy,
                            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
                        },
                        "camera_to_plane_m": self.config.logitech_reference_distance_m or None,
                        "relative_depth_stats": None if not prediction_values.size else {
                            "min": round(float(prediction_values.min()), 4),
                            "median": round(float(np.median(prediction_values)), 4),
                            "max": round(float(prediction_values.max()), 4),
                            "inverse_model": depth_output_kind(self.config.depth_model) != "metric",
                        },
                        "calibration": None if self.calibration is None else {
                            "scale": self.calibration.scale, "offset_m": self.calibration.offset_m,
                            "method": self.calibration.method, "id": self.calibration.calibration_id,
                        },
                        "live_volume_l": None if result.measurement is None else round(result.measurement.liters, 6),
                        "stable_volume_l": detection.stable_volume_l,
                        # The measurement chain for this one object, end to
                        # end, so a wrong number can be traced to the link that
                        # produced it without reading every frame's log.
                        "detection_box": list(detection.box),
                        "detector_mask_pixels": int(np.count_nonzero(instance_mask)),
                        "change_mask_pixels": (
                            0 if deposit_change is None
                            else int(np.count_nonzero(deposit_change & bin_region))
                        ),
                        "measurement_mask_pixels": int(np.count_nonzero(volume_mask)),
                        "measurement_mask_source": (self.last_measurement_mask or {}).get("source"),
                    }
                    if uncalibrated_logitech and result.measurement is not None:
                        detection.measurement_quality = self.calibration_mode
                        detection.calibration_mode = self.calibration_mode
                        # The dashboard's height column reads this field.
                        height_m = result.diagnostics.get("height_p90_m")
                        if height_m:
                            detection.height_above_baseline_cm = round(float(height_m) * 100.0, 1)
                        LOGGER.info(
                            "Logitech measurement (%s): track=%s mask=%s depth=%s baseline=%s "
                            "intrinsics=%s plane=%s litres=%.3f",
                            self.calibration_mode, detection.track_id, int(np.count_nonzero(volume_mask)),
                            calibrated_prediction is not None, self.reference_monocular is not None,
                            measure_intrinsics is not None, result.diagnostics.get("plane_source"),
                            result.measurement.liters,
                        )
                    if result.reason is not None:
                        detection.volume_rejection_reason = result.reason
                    elif result.diagnostics.get("length_mm"):
                        # Logitech dimensions are the object's own cross-section
                        # in floor coordinates, measured from the same point
                        # cloud the volume was integrated over. A short rolling
                        # median follows, so a settled object stops changing
                        # size every frame.
                        stable = self._logitech_geometry.update(
                            detection.track_id,
                            length_mm=float(result.diagnostics["length_mm"]),
                            width_mm=float(result.diagnostics["width_mm"]),
                            height_mm=float(result.diagnostics["height_p90_m"]) * 1000.0,
                            volume_l=None if result.measurement is None else result.measurement.liters,
                        )
                        detection.footprint_length_mm = stable.length_mm
                        detection.footprint_width_mm = stable.width_mm
                        detection.physical_height_mm = stable.height_mm
                        detection.dimension_method = str(
                            result.diagnostics.get("dimension_method")
                            or "logitech_support_plane_footprint"
                        )
                        detection.dimension_flags = tuple(detection.dimension_flags or ()) + tuple(
                            flag for flag in (
                                None if result.diagnostics.get("plane_is_floor", True)
                                else "height_not_floor_relative",
                                None if result.diagnostics.get("geometry_consistent", True) is not False
                                else "volume_disagrees_with_dimensions",
                                None if stable.settled else "geometry_not_settled",
                            ) if flag is not None
                        )
                        detection.dimension_confidence = round(
                            float(min(0.6, result.measurement.coverage_ratio)), 4)
                    metric_mask, metric_result = volume_mask, result
                else:
                    individual_mono = estimate_volume(
                    calibrated_prediction,
                    self.reference_monocular,
                    intrinsics,
                    object_mask=instance_mask,
                    roi=self.config.roi,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                    min_pixels=min(25, self.config.min_component_pixels),
                    method="monocular-instance",
                    measurement_mask=bin_region,
                    depth_noise_m=self.config.depth_noise_m,
                    reference_plane=measurement_plane,
                    **precision,
                    )
                if individual_mono is not None:
                    if self.camera_id == "logitech":
                        detection.depth_coverage_percent = round(individual_mono.coverage_ratio * 100, 1)
                        detection.volume_uncertainty_l = round(individual_mono.uncertainty_l, 6)
                        detection.measurement_method = "logitech-depth-anything-v2"
                        # A provisional mode keeps its own name here: the
                        # dashboard, the row and the operator all need to see
                        # that this number is an estimate, not a calibration.
                        detection.measurement_quality = (
                            self.calibration_mode if uncalibrated_logitech else individual_mono.quality
                        )
                        detection.calibration_mode = self.calibration_mode
                    foreground_fraction = individual_mono.valid_pixels / max(1, individual_mono.candidate_pixels)
                    if foreground_fraction < self.config.minimum_foreground_fraction:
                        detection.measurement_quality = "rejected-sparse-height-inside-mask"
                    else:
                        detection.monocular_volume_l = round(individual_mono.liters, 6)
                        if self.camera_id == "logitech":
                            # Same reasoning as the RealSense branch above:
                            # report the near-top percentile height from the
                            # accepted volume's own column field, not the
                            # separate whole-mask median computed earlier.
                            detection.height_above_baseline_cm = round(
                                individual_mono.height_p90_m * 100.0, 1
                            )
                if metric_mask is not None:
                    # Last, so the calibrated centimetres are what survives:
                    # the provisional height and litres are computed above and
                    # stay on the detection for comparison.
                    self._apply_metric_calibration(
                        detection, metric_mask, calibrated_prediction, metric_result,
                    )
            if (
                self.camera_id == "logitech"
                and detection.monocular_volume_l is not None
                and self.calibration is not None
                and logitech_ready
                and logitech_height_coherent
            ):
                # Same router, fed by calibrated Depth Anything V2 depth over
                # the eroded mask: monocular depth bleeds across object edges,
                # so the outermost pixels are not trusted for geometry.
                plane_points = object_plane_points(
                    calibrated_prediction, intrinsics, instance_mask, measurement_plane,
                    measurement_mask=bin_region,
                    min_height_m=minimum_height_m,
                    max_height_m=self.config.max_object_height_m,
                    erode_px=LOGITECH_MASK_ERODE_PX,
                )
                if plane_points is not None and self.diagnostics.enabled:
                    pending_points[id(detection)] = plane_points
                if plane_points is not None:
                    pending_shapes[id(detection)] = measure_shape(
                        *plane_points,
                        mesh_volume_l=detection.monocular_volume_l,
                        label=detection.label,
                    )

        if self.camera_id == "logitech":
            rejected = []
            for detection in detections:
                if (detection.monocular_volume_l is not None
                        and detection.monocular_volume_l > self.config.logitech_max_item_volume_l):
                    rejected.append(float(detection.monocular_volume_l))
                    detection.monocular_volume_l = None
                    detection.volume_uncertainty_l = None
                    detection.measurement_quality = "rejected-implausible-volume"
            if rejected:
                monocular_total = None
                warnings.append(
                    f"Rejected implausible Logitech object volume {max(rejected):.1f} L; "
                    "check its object mask, empty baseline, lens calibration, and measured reference distance"
                )
            if detections and any(item.monocular_volume_l is None for item in detections):
                monocular_total = None
            if bin_total is not None and self.config.bin_capacity_l and bin_total.liters > self.config.bin_capacity_l:
                warnings.append(
                    "Rejected Logitech occupancy above the measured physical bin capacity; "
                    "check its region, calibration, and empty baseline"
                )
                bin_total = None

        # BUGFIX (LiveFix): every confirmed bag/box must be tracked and counted as
        # soon as it is seen, regardless of whether its volume is trustworthy yet.
        # The previous code ran the tracker only on the subset of detections that
        # already passed `_measurement_is_recordable`, so any object waiting on a
        # calibration step (Logitech tilt/reference distance, baseline restore,
        # depth coverage) never received a track_id. Without a track_id it was
        # invisible to color smoothing, volume smoothing, BAGS/BOXES SEEN, and the
        # live table (`#?`, blank color, blank liters) even while its bounding box
        # and raw distance were already being drawn on the video feed. That
        # mismatch between what the camera overlay showed and what the metrics
        # panel showed is the "volume/colors missing" behavior reported live.
        # Tracking is now always run on the full detection list so an object is
        # seen/counted/colored the instant it is recognized; only the *database
        # write* (`ledger.observe`, which starts the deposit/history record) stays
        # gated behind `_measurement_is_recordable`, exactly as the
        # LOCALLIFE_RECORD_ONLY_MEASURED docstring always promised ("shown live,
        # but never added to experiment databases").
        tracking_detections = detections
        new_ids = self.tracker.update(tracking_detections) if self.config.auto_count else []
        self.stage_counters["active_tracks_last_frame"] = sum(1 for item in detections if item.track_id is not None)
        self.stage_counters["confirmed_tracks_total"] = int(self.tracker.total_count)
        if self._previous_bin_total_l is not None:
            for track_id in new_ids:
                self._bin_total_before_track[track_id] = self._previous_bin_total_l
        self._release_expired_track_state(self.tracker.last_expired_ids)
        self._check_camera_placement(depth_m, intrinsics, bin_region, detections, warnings)

        # Box-cuboid multi-frame track aggregation (Revised Dual-Camera
        # Volume Estimation recipe, section 13), now that every detection in
        # this frame has a known `track_id`: fold this frame's own
        # single-frame cuboid result (stashed by `id(detection)` in the
        # box-family measurement block above, before tracking ran) into that
        # track's accepted-frame history, and once enough accepted frames
        # exist, recompute the reported box measurement from their median
        # L/W/H -- overwriting the single-frame values `_apply_box_cuboid`
        # already applied above. A track with fewer accepted frames than
        # `box_aggregation_min_frames` keeps its current single-frame
        # result, so early frames are never hidden.
        # What a diagnostic bundle saves if a measurement is persisted this frame.
        self._frame_context = {
            "frame": frame,
            "depth": (calibrated_prediction if calibrated_prediction is not None else predicted_depth) if self.camera_id == "logitech" else depth_m,
            "points": pending_points,
        }
        for detection in detections:
            self._verify_track_identity(detection)
            detection.shape_geometry = self._geometry_lock.update(
                detection.track_id, pending_shapes.get(id(detection)),
                _object_signature(detection, self.camera_id),
            )
            self._apply_cylinder_geometry(detection)
            detection.canonical_type = canonical_object_type(detection.label, detection.confidence)
            if self.camera_id == "logitech" and self.volume_factors is not None:
                # Raw geometry first, then the frozen factor -- both kept.
                raw = detection.raw_volume_l = detection.monocular_volume_l
                group = geometry_group(
                    None if detection.shape_geometry is None else detection.shape_geometry.geometry_method,
                    detection.canonical_type or detection.label,
                )
                corrected, applied = self.volume_factors.correct(raw, group)
                detection.monocular_volume_l = corrected
                self.last_logitech_volume_diagnostics = {
                    **self.last_logitech_volume_diagnostics,
                    "calibration_group": group, "empirical_factor": applied,
                }
            if self.camera_id == "logitech" and detection.track_id is not None:
                live = detection.monocular_volume_l
                if live is not None:
                    self._logitech_volume_samples[detection.track_id].append(float(live))
                stable, spread = stable_volume(list(self._logitech_volume_samples[detection.track_id]))
                detection.stable_volume_l = None if stable is None else round(stable, 6)
                self._logitech_volume_spread[detection.track_id] = spread
        for detection in detections:
            if detection.track_id is None or id(detection) not in pending_box_attempted:
                continue
            self._box_frames_considered[detection.track_id] += 1
            raw_cuboid = pending_box_measurements.get(id(detection))
            if raw_cuboid is None:
                continue
            history = self._box_measurement_history[detection.track_id]
            history.append(raw_cuboid)
            if len(history) >= self.config.box_aggregation_min_frames:
                aggregated = aggregate_box_measurements(
                    list(history), frames_considered=self._box_frames_considered[detection.track_id],
                )
                if aggregated is not None:
                    self._apply_box_cuboid(detection, aggregated, warnings)

        if self.config.record_only_measured_objects:
            recordable_new_ids = {
                detection.track_id for detection in detections
                if detection.track_id in new_ids and self._measurement_is_recordable(detection)
            }
            if new_ids and recordable_new_ids != set(new_ids):
                warnings.append(
                    "Unmeasured or unvalidated detections are shown live but are not added to the database"
                )
        for detection in detections:
            if detection.track_id is not None:
                colors = self._color_history[detection.track_id]
                # A tracked prediction repeats the previous frame's colour;
                # counting it as a new vote made one early brown/black error
                # impossible to correct after the detector recovered.
                if (
                    detection.color not in {"unknown", ""}
                    and detection.source != "tracked-prediction"
                    and not _is_phantom_detection(detection)
                ):
                    colors.append(detection.color)
                if colors:
                    detection.color = Counter(colors).most_common(1)[0][0]
            # Playbook section 12. Applied to every detection, tracked or not,
            # so a mis-sorted object that never earns a track is still called
            # out rather than silently dropping off the event record.
            verdict = classify_sorting(
                detection.label,
                confidence=detection.confidence,
                accepted_class=detection.accepted_class,
            )
            detection.sorting_status = verdict.status
            detection.sorting_reason = verdict.reason
            if detection.track_id is not None:
                canonical_material = {
                    "plastic_bag": "polythene bag",
                    "paper_bag": "paper bag",
                    "cardboard_box": "cardboard",
                }.get(detection.accepted_class)
                if canonical_material is not None:
                    # The accepted class is stronger material evidence than
                    # a generic crop classifier. This prevents a confirmed
                    # cardboard box being reported as plastic (image2_1).
                    detection.material = canonical_material
                    detection.material_confidence = 1.0
                elif self.material_classifier is not None and self.material_classifier.enabled:
                    materials = self._material_history[detection.track_id]
                    frame_count = self._material_frame_counts[detection.track_id]
                    due = frame_count % max(1, self.config.material_reclassify_frames) == 0
                    self._material_frame_counts[detection.track_id] = frame_count + 1
                    if due or not materials:
                        label, score = self.material_classifier.classify(frame, detection.mask, detection.box)
                        if label != "unknown" and score >= self.config.material_confidence_threshold:
                            materials.append(label)
                    if materials:
                        votes = Counter(materials)
                        best_material, best_count = votes.most_common(1)[0]
                        detection.material = best_material
                        detection.material_confidence = round(best_count / len(materials), 4)
            measured = self._detection_volume(detection)
            if detection.track_id is not None and measured is not None:
                history = self._volume_history[detection.track_id]
                history.append(measured)
                required = self.config.volume_stability_frames if self.config.record_only_measured_objects else 1
                smoothing_window = list(history)[-max(required, self.config.volume_window_frames) :]
                recent = np.asarray(smoothing_window[-required:], dtype=np.float64)
                median = float(np.median(recent))
                tolerance_l = max(0.20, median * max(0.15, self.config.settle_volume_tolerance))
                stable = len(recent) >= required and float(np.max(recent) - np.min(recent)) <= tolerance_l
                waited = self._measurement_frames[detection.track_id] = (
                    self._measurement_frames.get(detection.track_id, 0) + 1
                )
                if not stable and self.camera_id == "logitech" \
                        and waited >= self.config.logitech_measurement_timeout_frames and recent.size >= 2:
                    # A monocular volume can jitter past the tolerance for ever.
                    # After the window, the median of what this track has is
                    # its answer; the id and the method do not change.
                    stable = True
                    detection.measurement_quality = "median-after-stability-timeout"
                statistics = stable_statistics(
                    smoothing_window, min_frames=required,
                    tolerance=max(0.15, self.config.settle_volume_tolerance),
                )
                self._stability[detection.track_id] = statistics
                if not stable and len(recent) >= required and statistics.get("median") is not None:
                    # Valid numbers that will not sit still are still numbers.
                    # Blanking them is how a measured object ended up showing
                    # nothing at all; the median is shown and marked instead.
                    stabilized = round(float(statistics["median"]), 6)
                    if self.camera_id == "logitech":
                        detection.monocular_volume_l = stabilized
                    else:
                        detection.realsense_volume_l = stabilized
                    detection.stable_volume_l = stabilized
                    detection.measurement_quality = "low-confidence-unstable-median"
                elif not stable:
                    if self.camera_id == "logitech":
                        detection.monocular_volume_l = None
                    else:
                        detection.realsense_volume_l = None
                    detection.measurement_quality = (
                        "stabilizing-volume" if len(recent) < required else "rejected-unstable-volume"
                    )
                else:
                    stabilized = round(median, 6)
                    if self.camera_id == "logitech":
                        detection.monocular_volume_l = stabilized
                    else:
                        detection.realsense_volume_l = stabilized

            if detection.measurement_quality is None:
                detection.measurement_quality = self._pending_measurement_reason(
                    detection, depth_m=depth_m, intrinsics=intrinsics
                )

            if detection.track_id is not None:
                track = self.tracker.tracks.get(detection.track_id)
                if track is not None and track.counted:
                    detection.tracking_status = "confirmed"
                    previous = self._session_seen_tracks.get(detection.track_id, {})
                    self._session_seen_tracks[detection.track_id] = {
                        "track_id": detection.track_id,
                        "label": detection.label,
                        "object_type": waste_object_type(detection.label),
                        "color": detection.color or "unknown",
                        "first_seen_at": previous.get("first_seen_at", float(timestamp)),
                        "last_seen_at": float(timestamp),
                    }
                self.tracker.remember(detection)

        for detection in detections:
            track = self.tracker.tracks.get(detection.track_id) if detection.track_id is not None else None
            should_observe = bool(track is not None and track.counted)
            should_observe = should_observe and not _is_phantom_detection(detection)
            should_observe = should_observe and detection.accepted_class is not None
            if self.config.record_only_measured_objects:
                should_observe = should_observe and self._measurement_is_recordable(detection)
            # A track can become measurable several frames after it was born.
            # Check on every confirmed frame, not only when its ID is new.
            if self.config.operating_mode == "waste":
                if should_observe:
                    self.ledger.observe(detection, timestamp=timestamp)
                # Preserve production behavior: an already-observed record
                # may still receive improved color/material/volume on a frame
                # that is not itself eligible to create a new record.
                self.ledger.refresh(detection)

        newly_deposited: list[Detection] = []
        if self.config.auto_deposit and self.config.operating_mode == "waste":
            for detection in detections:
                if detection.track_id is None or self.ledger.is_deposited(detection.track_id):
                    continue
                if _is_phantom_detection(detection):
                    continue
                if detection.accepted_class is None:
                    continue
                track = self.tracker.tracks.get(detection.track_id)
                if track is None or not track.counted:
                    continue
                if (
                    detection.depth_coverage_percent is not None
                    and detection.depth_coverage_percent < self.config.minimum_depth_coverage * 100.0
                ):
                    continue
                history = list(self._volume_history.get(detection.track_id, ()))
                if len(history) < self.config.settle_frames:
                    continue
                recent = np.asarray(history[-self.config.settle_frames :], dtype=np.float64)
                median = float(np.median(recent))
                tolerance_l = max(0.15, median * self.config.settle_volume_tolerance)
                if float(np.max(recent) - np.min(recent)) <= tolerance_l:
                    if not self._record_added_volume(detection, bin_total, scene_grid):
                        warnings.append(
                            f"Deposit withheld: {detection.volume_rejection_reason}"
                        )
                        # A withheld deposit is a real outcome, not an absence:
                        # record it with its reason so a run's rejections are
                        # auditable instead of vanishing from the export.
                        self.persist_measurement_event(
                            detection, timestamp,
                            status=STATUS_REJECTED,
                            status_reason=detection.volume_rejection_reason,
                        )
                        continue
                    # Persist BEFORE the ledger marks the deposit complete, so
                    # the dashboard can never show a finalised row that was
                    # never written to disk.
                    self.persist_measurement_event(detection, timestamp)
                    self.ledger.deposit(detection, timestamp=timestamp)
                    self._committed_scene = scene_grid
                    self.deposit_state.committed()
                    newly_deposited.append(detection)
        # This frame's occupancy becomes the "before" state that whatever
        # appears next will be measured against.
        if bin_total is not None:
            self._previous_bin_total_l = float(bin_total.liters)
        elif occupancy_reference is not None and occupancy_depth is not None and logitech_ready:
            # A bin that was measured and found to hold nothing above the noise
            # floor occupies 0 L; treating that as unknown would deny the very
            # first deposit into an empty bin its "before" state.
            self._previous_bin_total_l = 0.0
        # Reconciled against `aggregate_detections` (not raw `detections`),
        # matching the phantom-excluding mask used to compute the estimate
        # above -- see the `confirmed_object_mask` comment for why.
        if hardware_total is not None and aggregate_detections and all(
            detection.realsense_volume_l is not None for detection in aggregate_detections
        ):
            hardware_total.liters = float(
                sum(detection.realsense_volume_l for detection in aggregate_detections)
            )
        if self.camera_id == "logitech" and monocular_total is not None and aggregate_detections and all(
            detection.monocular_volume_l is not None for detection in aggregate_detections
        ):
            monocular_total.liters = float(
                sum(detection.monocular_volume_l for detection in aggregate_detections)
            )
        if aggregate_detections and self.camera_id != "logitech" and any(
            detection.realsense_volume_l is None for detection in aggregate_detections
        ):
            hardware_total = None
        if aggregate_detections and self.camera_id == "logitech" and any(
            detection.monocular_volume_l is None for detection in aggregate_detections
        ):
            monocular_total = None
        display_detections = detections + self.tracker.predicted_detections(
            self.config.tracker_live_prediction_frames
        )
        analysis = FrameAnalysis(
            timestamp=float(timestamp),
            source=source,
            detections=display_detections,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            automatic_count=self.tracker.total_count,
            realsense_total=hardware_total,
            monocular_total=monocular_total,
            calibration=self.calibration,
            inference_ms=inference_ms,
            warnings=warnings,
            bin_total=bin_total,
        )
        if timestamp >= self.latest_frame_timestamp:
            self.latest_frame = frame.copy()
            self.latest_depth = None if depth_m is None else depth_m.copy()
            self.latest_frame_timestamp = float(timestamp)
        self.latest_monocular_depth = None if calibrated_prediction is None else calibrated_prediction.copy()
        if self.diagnostics.enabled:
            self.diagnostics.maybe_record_scene(
                frame=frame,
                depth=(calibrated_prediction if calibrated_prediction is not None else predicted_depth) if self.camera_id == "logitech" else depth_m,
                mask_overlay=self.logitech_mask_overlay() if self.camera_id == "logitech" else None,
                summary={
                    "detections": [item.to_dict() for item in detections],
                    "warnings": list(warnings),
                    "logitech": self.logitech_diagnostics() if self.camera_id == "logitech" else None,
                },
            )
        self._observe_deposit_state(detections, deposit_change, bin_region)
        self._log_measurement_chain()
        self.latest_analysis = analysis
        self.latest_processed_frame = frame.copy()
        self.latest_analysis_timestamp = float(timestamp)
        self.frames_processed += 1
        self.last_frame_processed_at = time.time()
        if depth_m is not None and not detections:
            self._recent_depth_frames.append(depth_m.copy())
        if predicted_depth is not None and not detections:
            self._recent_monocular_frames.append(predicted_depth.copy())
        self._consider_automatic_baseline(
            frame, depth_m, intrinsics, detections, warnings,
            peer_bag_present=peer_bag_present,
        )

        if persist:
            self.store.append_jsonl("frames.jsonl", analysis.to_dict())
            self._finalise_settled_measurements(detections, timestamp)
            starting_count = self.tracker.total_count - len(new_ids)
            for index, track_id in enumerate(new_ids, start=1):
                detection = next(item for item in detections if item.track_id == track_id)
                self.store.append_jsonl(
                    "events.jsonl",
                    {
                        "event": "confirmed-object",
                        "timestamp": timestamp,
                        "count": starting_count + index,
                        "detection": detection.to_dict(),
                    },
                )
        if (
            self.config.advance_reference_on_deposit
            and newly_deposited
            and len(newly_deposited) == len(detections)
        ):
            self._remember_occupied_objects(newly_deposited)
            self._advance_reference()
            self.committed_bags = self.ledger.summary()["deposited_bags"]
            self.tracker.tracks.clear()
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            self._geometry_lock.clear()
            self._track_signatures.clear()
            self._color_history.clear()
            self._material_history.clear()
            self._material_frame_counts.clear()
            LOGGER.info(
                "Automatically recorded %s settled waste item(s); updated the bin reference",
                len(newly_deposited),
            )
        return analysis

    def accept_current(self) -> dict[str, Any]:
        with self.lock:
            if self.latest_analysis is None or not self.latest_analysis.detections:
                raise ValueError("No detected object is available for acceptance")
            record = {
                "event": "manual-validation",
                "accepted_at": time.time(),
                "analysis": self.latest_analysis.to_dict(),
            }
            self.store.append_jsonl("validated_measurements.jsonl", record)
            return record

    def commit_current_bags(self) -> dict[str, Any]:
        """Use the settled contents as the reference for the next arriving bag."""
        with self.lock:
            if self.latest_analysis is None or not self.latest_analysis.detections:
                raise ValueError("Wait for a settled garbage bag before updating the bin reference")
            if self.camera_id == "logitech":
                if self.latest_monocular_depth is None or self.reference_monocular is None:
                    raise ValueError("Logitech estimated depth and an empty baseline are required before committing")
                if any(item.monocular_volume_l is None for item in self.latest_analysis.detections):
                    raise ValueError("An unmeasured or implausible Logitech volume cannot be committed")
            elif self.latest_depth is None or self.reference_realsense is None:
                raise ValueError("Aligned RealSense depth is required before committing a bag")
            if any(
                item.depth_coverage_percent is not None
                and item.depth_coverage_percent < self.config.minimum_depth_coverage * 100.0
                for item in self.latest_analysis.detections
            ):
                raise ValueError(
                    "Object depth coverage is below the reliable measurement threshold; "
                    "improve camera alignment or lighting before recording its volume"
                )
            record = self.accept_current()
            for detection in self.latest_analysis.detections:
                self.ledger.deposit(detection)
            if self.config.advance_reference_on_deposit:
                self._remember_occupied_objects(self.latest_analysis.detections)
                self._advance_reference()
            self.committed_bags = self.ledger.summary()["deposited_bags"]
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            self._geometry_lock.clear()
            self._track_signatures.clear()
            self._color_history.clear()
            self._material_history.clear()
            self._material_frame_counts.clear()
            self.tracker.tracks.clear()
            record["event"] = "committed-waste-item"
            record["committed_bags"] = self.committed_bags
            self.store.append_jsonl("bag_commits.jsonl", record)
            return record

    def _remember_occupied_objects(self, detections: list[Detection]) -> None:
        if self.camera_id != "logitech" or self.latest_frame is None:
            return
        shape = self.latest_frame.shape[:2]
        if self._occupied_logitech_mask is None or self._occupied_logitech_mask.shape != shape:
            self._occupied_logitech_mask = np.zeros(shape, dtype=bool)
        self._occupied_logitech_mask |= combined_mask(detections, shape)

    def _advance_reference(self) -> None:
        self.reference_rgb = self.latest_frame.copy() if self.latest_frame is not None else None
        self.reference_realsense = self.latest_depth.copy() if self.latest_depth is not None else None
        self.reference_monocular = (
            None if self.latest_monocular_depth is None else self.latest_monocular_depth.copy()
        )
        self._save_working_reference()

    def reset_live_tracking(self) -> None:
        with self.lock:
            next_id = self.tracker.next_track_id
            self.tracker.reset()
            self.tracker.next_track_id = next_id
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            self._geometry_lock.clear()
            self._track_signatures.clear()
            self._color_history.clear()
            self._material_history.clear()
            self._material_frame_counts.clear()
            self._session_seen_tracks.clear()

    def _consider_automatic_baseline(
        self,
        frame: np.ndarray,
        depth_m: np.ndarray | None,
        intrinsics: CameraIntrinsics | None,
        detections: list[Detection],
        warnings: list[str],
        *,
        peer_bag_present: bool = False,
    ) -> None:
        """Capture a temporal empty reference after a stable, object-free run.

        This removes dashboard calibration buttons, but it does not pretend the
        physical empty-scene requirement disappeared. A visible waste object,
        camera motion, missing intrinsics, or missing depth resets the countdown.
        """
        if not self.config.automatic_baseline:
            self._automatic_baseline_status = "disabled"
            return
        if self.baseline_restore_state == "validating":
            self._automatic_baseline_status = "validating-saved-profile"
            return
        if self.baseline_rgb is not None:
            self._automatic_baseline_status = "ready"
            return
        if detections or peer_bag_present:
            self._automatic_baseline_stable_frames = 0
            self._automatic_baseline_previous = None
            self._automatic_baseline_status = "waiting-for-empty-scene"
            return
        if intrinsics is None:
            self._automatic_baseline_status = "waiting-for-camera-intrinsics"
            return
        if self.camera_id == "realsense" and summarize_depth_signal(depth_m)["valid_pixels"] == 0:
            self._automatic_baseline_status = "waiting-for-realsense-depth"
            return
        # The emergency local CPU build intentionally has no monocular depth.
        # Logitech still receives an RGB baseline for tracking and colour; only
        # its independent litre estimate remains unavailable.

        region = fixed_bin_mask(frame.shape, self.config.roi, self.config.bin_polygon)
        previous = self._automatic_baseline_previous
        motion = 0.0
        if previous is not None and previous.shape == frame.shape:
            difference = np.mean(
                np.abs(frame.astype(np.float32) - previous.astype(np.float32)), axis=2
            )
            values = difference[region]
            motion = float(np.mean(values)) if values.size else float("inf")
        self._automatic_baseline_previous = frame.copy()
        if previous is not None and motion > self.config.automatic_baseline_motion_threshold:
            self._automatic_baseline_stable_frames = 0
            self._automatic_baseline_status = "waiting-for-camera-and-scene-to-settle"
            return

        self._automatic_baseline_stable_frames += 1
        required = self.config.automatic_baseline_frames
        self._automatic_baseline_status = (
            f"verifying-empty-scene-{self._automatic_baseline_stable_frames}-of-{required}"
        )
        if self._automatic_baseline_stable_frames < required:
            return
        try:
            self.set_baseline()
        except ValueError as exc:
            self._automatic_baseline_stable_frames = 0
            self._automatic_baseline_status = "automatic-setup-retrying"
            warnings.append(f"Automatic empty-scene setup is retrying: {exc}")
            return
        self._automatic_baseline_status = "ready"
        warnings.append("Automatic empty-scene setup completed and was saved")

    def _pending_measurement_reason(
        self,
        detection: Detection,
        *,
        depth_m: np.ndarray | None,
        intrinsics: CameraIntrinsics | None,
    ) -> str:
        """Return a precise live reason instead of an unexplained blank liters cell."""
        if self.baseline_restore_state == "validating":
            return "pending-baseline-validation"
        if intrinsics is None:
            return "pending-camera-intrinsics"
        if self.camera_id == "realsense":
            if depth_m is None:
                return "pending-realsense-depth"
            # Only genuinely blocked on a baseline when the live-background
            # support plane could not be fitted either (see the round-24
            # block in `_assemble`) -- otherwise a measurement is possible
            # and the real reason lies further down.
            if self.reference_realsense is None and self.reference_plane is None:
                return "pending-empty-baseline"
        else:
            if self.depth_estimator is None:
                return "unavailable-depth-anything-not-loaded"
            if self._detection_volume(detection) is not None:
                # A numeric result is never described as pending.
                return detection.measurement_quality or self.calibration_mode or "measured"
            if self.latest_monocular_depth is not None and not self._full_metric_ready(
                self.latest_monocular_depth, self.latest_intrinsics or self._field_of_view_intrinsics(
                    (0, 0) if self.latest_frame is None else self.latest_frame.shape)
            ):
                # The provisional cascade can run, so the object is measured by
                # it rather than waiting for a baseline that may never come.
                return self.calibration_mode if self.calibration_mode in (
                    "reference-distance-estimate", "uncalibrated-estimate") else "uncalibrated-estimate"
            if self.latest_monocular_depth is None and self.reference_monocular is None:
                return "pending-monocular-depth"
            if self.reference_monocular is None:
                return "pending-empty-baseline"
            if not self._logitech_measurement_ready():
                return "pending-logitech-calibration"
        if detection.depth_coverage_percent is not None and (
            detection.depth_coverage_percent < self.config.minimum_depth_coverage * 100.0
        ):
            return "pending-depth-coverage"
        return "pending-measurable-height"

    def _session_seen_summary(self) -> dict[str, Any]:
        records = list(self._session_seen_tracks.values())
        colors: dict[str, int] = {}
        for record in records:
            color = str(record.get("color") or "unknown")
            colors[color] = colors.get(color, 0) + 1
        return {
            "total": len(records),
            "bags": sum(item.get("object_type") == "bag" for item in records),
            "boxes": sum(item.get("object_type") == "box" for item in records),
            "colors": [
                {"color": color, "seen_count": count}
                for color, count in sorted(colors.items(), key=lambda item: (-item[1], item[0]))
            ],
        }

    def update_camera_roi(self, values: tuple[float, float, float, float]) -> dict[str, Any]:
        with self.lock:
            previous = self.config.roi
            self.config.roi = values
            try:
                self.config.validate()
            except ValueError:
                self.config.roi = previous
                raise
            self.store.save_json("calibration/roi.json", {"camera_id": self.camera_id, "roi": list(values)})
            self.baseline_rgb = None
            self.baseline_realsense = None
            self.baseline_monocular = None
            self.reference_rgb = None
            self.reference_realsense = None
            self.reference_monocular = None
            self._occupied_logitech_mask = None
            self.saved_profile_loaded = False
            self.baseline_restore_state = "roi-changed-setup-required"
            self.saved_baseline_changed_fraction = None
            self._invalidate_saved_baseline()
            self.reset_live_tracking()
            return {"camera_id": self.camera_id, "roi": list(values), "recapture_baseline": True}

    def clear_history(self) -> dict[str, Any]:
        with self.lock:
            previous = self.ledger.summary()
            ledger_path = self.config.results_dir / "waste_plant_ledger.jsonl"
            backup = None
            if ledger_path.is_file():
                backup_directory = self.config.results_dir / "history_backups"
                backup_directory.mkdir(parents=True, exist_ok=True)
                backup = backup_directory / f"waste_plant_ledger-{time.time_ns()}.jsonl"
                ledger_path.replace(backup)
            self.ledger = WastePlantLedger(
                self.store, color_streams=self.config.color_waste_streams,
                history_limit=self.config.history_limit, camera_id=self.camera_id,
            )
            self.committed_bags = 0
            if self._occupied_logitech_mask is not None:
                self._occupied_logitech_mask.fill(False)
            self.reset_live_tracking()
            return {"camera_id": self.camera_id, "removed_observations": previous["observed_count"],
                    "backup_created": backup is not None}

    def _detection_volume(self, detection: Detection) -> float | None:
        return detection.monocular_volume_l if self.camera_id == "logitech" else detection.realsense_volume_l

    def _measurement_is_recordable(self, detection: Detection) -> bool:
        if _is_phantom_detection(detection) or detection.accepted_class is None:
            return False
        if self.baseline_restore_state == "validating":
            return False
        measured = self._detection_volume(detection)
        if measured is None or not np.isfinite(measured) or measured <= 0:
            return False
        if self.camera_id == "logitech" and not self._logitech_measurement_ready():
            return False
        return not (
            detection.depth_coverage_percent is not None
            and detection.depth_coverage_percent < self.config.minimum_depth_coverage * 100.0
        )

    def _load_saved_baseline(self) -> None:
        if not self.config.restore_saved_baseline:
            return
        directory = self.config.results_dir / "baselines"
        metadata_path = directory / "metadata.json"
        rgb_path = directory / "rgb.npy"
        if not metadata_path.is_file() or not rgb_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("camera_id") not in {None, self.camera_id}:
                raise ValueError("baseline belongs to another camera")
            rgb = np.load(rgb_path, allow_pickle=False)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("saved RGB baseline has an invalid shape")

            def optional_array(name: str) -> np.ndarray | None:
                location = directory / name
                if not location.is_file():
                    return None
                array = np.load(location, allow_pickle=False).astype(np.float32)
                if array.ndim != 2 or array.shape != rgb.shape[:2]:
                    raise ValueError(f"saved {name} has an invalid shape")
                return array

            realsense = optional_array("realsense_depth_m.npy")
            monocular = optional_array("monocular_depth_m.npy")
            noise = optional_array("noise_m.npy")
            reference_rgb_path = directory / "reference_rgb.npy"
            reference_rgb = (
                np.load(reference_rgb_path, allow_pickle=False)
                if reference_rgb_path.is_file() else rgb
            )
            if reference_rgb.shape != rgb.shape:
                raise ValueError("saved working RGB reference has an invalid shape")
            reference_realsense = optional_array("reference_realsense_depth_m.npy")
            reference_monocular = optional_array("reference_monocular_depth_m.npy")
            occupied_mask = optional_array("occupied_logitech_mask.npy")
            calibration_payload = metadata.get("calibration")
            calibration = None
            if calibration_payload:
                calibration = DepthCalibration.from_dict(calibration_payload)
            self.baseline_rgb = rgb.astype(np.uint8, copy=False)
            self.reference_rgb = reference_rgb.astype(np.uint8, copy=False)
            self.baseline_realsense = realsense
            self.reference_realsense = (
                reference_realsense if reference_realsense is not None
                else None if realsense is None else realsense.copy()
            )
            self.baseline_monocular = monocular
            self.reference_monocular = (
                reference_monocular if reference_monocular is not None
                else None if monocular is None else monocular.copy()
            )
            self.baseline_noise_map = noise
            self.baseline_noise_m = float(metadata.get("baseline_noise_m", 0.0))
            self.baseline_frame_count = int(metadata.get("baseline_frame_count", 1))
            self.calibration = calibration
            self.calibration_mode = str(metadata.get("calibration_mode", "restored"))
            self.latest_intrinsics = CameraIntrinsics.from_dict(metadata.get("intrinsics"))
            self.latest_intrinsics_origin = "restored-installation-profile"
            self._occupied_logitech_mask = (
                occupied_mask.astype(bool) if self.camera_id == "logitech" and occupied_mask is not None
                else np.zeros(rgb.shape[:2], dtype=bool) if self.camera_id == "logitech" else None
            )
            self.saved_profile_loaded = True
            self.baseline_restore_state = "validating"
            LOGGER.info("Loaded saved %s installation profile; validating the live scene", self.camera_id)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._clear_baseline("invalid-saved-profile")
            LOGGER.warning("Could not restore %s saved baseline: %s", self.camera_id, exc)

    def _validate_saved_baseline(self, frame: np.ndarray, warnings: list[str]) -> None:
        if self.baseline_restore_state != "validating" or self.reference_rgb is None:
            return
        if frame.shape != self.reference_rgb.shape:
            self._clear_baseline("rejected-frame-shape-changed")
            warnings.append(
                "Saved setup was rejected because the camera resolution changed; capture a new empty baseline"
            )
            return
        region = fixed_bin_mask(frame.shape, self.config.roi, self.config.bin_polygon)
        difference = np.max(
            np.abs(frame.astype(np.int16) - self.reference_rgb.astype(np.int16)), axis=2
        )
        values = difference[region]
        changed_fraction = float(
            np.mean(values > self.config.saved_baseline_rgb_threshold)
        ) if values.size else 1.0
        self.saved_baseline_changed_fraction = changed_fraction
        if changed_fraction > self.config.saved_baseline_max_changed_fraction:
            self._clear_baseline("rejected-scene-changed")
            warnings.append(
                "Saved setup was rejected because the camera or bin scene changed; empty the bin and capture new baselines"
            )
            return
        self._saved_baseline_matching_frames += 1
        if self._saved_baseline_matching_frames >= self.config.saved_baseline_validation_frames:
            self.baseline_restore_state = "restored-and-validated"
            LOGGER.info("Validated saved %s installation profile", self.camera_id)
        else:
            warnings.append("Validating the saved installation profile; measurements are temporarily withheld")

    def _clear_baseline(self, state: str) -> None:
        self.baseline_rgb = None
        self.baseline_realsense = None
        self.baseline_monocular = None
        self.reference_rgb = None
        self.reference_realsense = None
        self.reference_monocular = None
        self.baseline_noise_map = None
        self.reference_plane = None
        self.saved_profile_loaded = False
        self.baseline_restore_state = state
        self._saved_baseline_matching_frames = 0
        self._automatic_baseline_stable_frames = 0
        self._automatic_baseline_previous = None
        self._automatic_baseline_status = "waiting-for-empty-stable-scene"

    def _save_working_reference(self) -> None:
        if self.reference_rgb is None:
            return
        directory = self.config.results_dir / "baselines"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "reference_rgb.npy", self.reference_rgb)
        for name, value in (
            ("reference_realsense_depth_m.npy", self.reference_realsense),
            ("reference_monocular_depth_m.npy", self.reference_monocular),
            ("occupied_logitech_mask.npy", self._occupied_logitech_mask),
        ):
            location = directory / name
            if value is None:
                location.unlink(missing_ok=True)
            else:
                np.save(location, value)
        self.store.save_json(
            "baselines/reference_metadata.json",
            {"camera_id": self.camera_id, "saved_at": time.time()},
        )

    def _invalidate_saved_baseline(self) -> None:
        directory = self.config.results_dir / "baselines"
        for name in (
            "rgb.npy", "realsense_depth_m.npy", "monocular_depth_m.npy", "noise_m.npy",
            "reference_rgb.npy", "reference_realsense_depth_m.npy",
            "reference_monocular_depth_m.npy", "occupied_logitech_mask.npy",
            "metadata.json", "reference_metadata.json",
        ):
            (directory / name).unlink(missing_ok=True)

    def _load_volume_calibration(self) -> None:
        roi_location = self.config.results_dir / "calibration" / "roi.json"
        if roi_location.is_file():
            try:
                payload = json.loads(roi_location.read_text(encoding="utf-8"))
                roi = tuple(float(value) for value in payload["roi"])
                if len(roi) != 4:
                    raise ValueError("ROI requires four normalized coordinates")
                previous = self.config.roi
                self.config.roi = roi
                try:
                    self.config.validate()
                except ValueError:
                    self.config.roi = previous
                    raise
            except (OSError, ValueError, KeyError, TypeError) as exc:
                LOGGER.warning("Could not restore %s camera ROI: %s", self.camera_id, exc)
        reference_location = self.config.results_dir / "calibration" / "reference_distance.json"
        if self.camera_id == "logitech" and reference_location.is_file():
            try:
                payload = json.loads(reference_location.read_text(encoding="utf-8"))
                distance = float(payload["distance_m"])
                if np.isfinite(distance) and distance > 0:
                    self.config.logitech_reference_distance_m = distance
            except (OSError, ValueError, KeyError, TypeError) as exc:
                LOGGER.warning("Could not restore Logitech reference-distance calibration: %s", exc)
        location = self.config.results_dir / "calibration" / "volume.json"
        if not location.is_file():
            return
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
            factor = float(payload["factor"])
            if np.isfinite(factor) and factor > 0:
                self.config.volume_calibration_factor = factor
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("Could not restore %s volume calibration: %s", self.camera_id, exc)

    def calibrate_known_volume(self, known_liters: float, observed_liters: float | None = None) -> dict[str, Any]:
        with self.lock:
            if not np.isfinite(known_liters) or known_liters <= 0:
                raise ValueError("Known reference volume must be a finite positive number of liters")
            if observed_liters is None:
                if self.latest_analysis is None:
                    raise ValueError("Measure a reference object before calibrating its volume")
                # Deliberately NOT `realsense_total`/`monocular_total`: those
                # are estimated over the union mask of every current
                # detection (`combined_mask()` in geometry.py), including any
                # unconfirmed phantom/depth-silhouette region still in frame
                # (a shadow, a background fold -- see `_is_phantom_detection`
                # above; this project's own history, rounds 5-8, shows these
                # appear often). The dashboard's "Calibrate from this object"
                # button never sends `observed_liters` explicitly, so before
                # this fix every calibration silently used that combined
                # figure -- if a phantom happened to share the frame with the
                # real known-volume object, the solved factor was fit against
                # "known object + phantom noise" instead of the known object
                # alone, and that wrong factor then multiplied into every
                # future reading on this camera. Anchoring on the single
                # confirmed detection's own already-displayed liters value
                # instead guarantees calibration is fit against exactly the
                # number the user is looking at on the dashboard.
                candidates = [
                    detection for detection in self.latest_analysis.detections
                    if not _is_phantom_detection(detection)
                    and detection.tracking_status in {"confirmed", "predicted"}
                    and (
                        detection.monocular_volume_l if self.camera_id == "logitech"
                        else detection.realsense_volume_l
                    ) is not None
                ]
                if not candidates:
                    raise ValueError(
                        "No confirmed, measured object is currently in view; place the known-volume "
                        "object where its liters value is shown in the live table, then calibrate"
                    )
                if len(candidates) > 1:
                    raise ValueError(
                        "More than one measured object is currently in view; remove everything except "
                        "the single known-volume reference object, then calibrate"
                    )
                observed_liters = (
                    candidates[0].monocular_volume_l if self.camera_id == "logitech"
                    else candidates[0].realsense_volume_l
                )
            if not np.isfinite(observed_liters) or observed_liters <= 0:
                raise ValueError("Observed reference volume must be a finite positive number of liters")
            previous = self.config.volume_calibration_factor
            factor = previous * known_liters / observed_liters
            if not 0.10 <= factor <= 10.0:
                raise ValueError("Calibration factor is implausible; verify the reference volume and camera geometry")
            self.config.volume_calibration_factor = float(factor)
            record = {
                "camera_id": self.camera_id,
                "known_liters": float(known_liters),
                "observed_liters": float(observed_liters),
                "previous_factor": float(previous),
                "factor": float(factor),
                "calibrated_at": time.time(),
                "warning": "Validate accuracy using separate objects not used to fit this factor.",
            }
            self.store.save_json("calibration/volume.json", record)
            # The factor changed, so litres measured under the old one are
            # stale and their history goes.
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            if self.camera_id != "logitech":
                self._geometry_lock.clear()
                self._track_signatures.clear()
            else:
                # A Logitech object's geometry is measured in metres and does
                # not depend on this factor at all, so throwing away its shape
                # lock and its track signatures only sent every object on
                # screen back to "not settled" with no dimensions -- which is
                # what entering a known volume looked like from the dashboard.
                record["geometry_preserved"] = True
            return record

    def state(self) -> dict[str, Any]:
        with self.lock:
            hardware = summarize_depth_signal(self.latest_depth)
            monocular = summarize_depth_signal(self.latest_monocular_depth)
            volume_status = self._volume_status(hardware)
            readiness = self._measurement_readiness()
            return {
                "camera_id": self.camera_id,
                "camera_name": "Intel RealSense D435" if self.camera_id == "realsense" else "Logitech C920",
                # Every prerequisite, one field each, with the next action.
                "measurement_readiness": readiness.to_dict(),
                "measurement_zone": (
                    None if self.measurement_zone is None else self.measurement_zone.describe()
                ),
                "measurement_zone_source": self.zone_source,
                "logitech_metric": self.metric_status(),
                "deposit_state": self.deposit_state.describe(),
                # How many objects this session has finalised into
                # measurements.csv. A download that comes back with only a
                # header is almost always this being zero because no empty-bin
                # baseline was captured, so the page can say so instead of
                # handing the operator a blank file with no explanation.
                "measurements_recorded": self.event_log.status()["events_persisted"],
                # Whether each finalised event actually reached disk, where the
                # file is, and whether a failure is waiting to be retried.
                "csv_persistence": self.event_log.status(),
                # The canonical measurement history: finalised events, read
                # back from the CSV the pipeline already wrote. A read-only
                # view -- it observes what tracking produced and never feeds
                # anything back into it.
                "measurement_history": self.event_log.recent(limit=30),
                "last_persisted_event": self._last_persist_result,
                "measurement_method": (
                    "hardware-stereo-depth" if self.camera_id == "realsense"
                    else "local-rgb-tracking" if self.depth_estimator is None
                    else "depth-anything-v2-monocular"
                ),
                "frames_processed": self.frames_processed,
                "frames_received": self.frames_received,
                "stream": {
                    "received": self.frames_received,
                    "processed": self.frames_processed,
                    "last_received_age_s": None if self.last_frame_received_at is None else round(
                        max(0.0, time.time() - self.last_frame_received_at), 2
                    ),
                    "last_processed_age_s": None if self.last_frame_processed_at is None else round(
                        max(0.0, time.time() - self.last_frame_processed_at), 2
                    ),
                    "analysis_lag_s": round(
                        max(0.0, self.latest_frame_timestamp - self.latest_analysis_timestamp), 3
                    ),
                },
                "automatic_count": self.tracker.total_count,
                "session_seen": self._session_seen_summary(),
                "baseline_ready": self.baseline_rgb is not None,
                "plug_and_play": True,
                "saved_profile_loaded": self.saved_profile_loaded,
                "baseline_restore_state": self.baseline_restore_state,
                "automatic_setup": {
                    "enabled": self.config.automatic_baseline,
                    "status": self._automatic_baseline_status,
                    "stable_frames": self._automatic_baseline_stable_frames,
                    "required_frames": self.config.automatic_baseline_frames,
                    "ready": (
                        self.baseline_rgb is not None
                        and self.baseline_restore_state != "validating"
                    ),
                },
                "saved_baseline_changed_fraction": self.saved_baseline_changed_fraction,
                "hardware_depth_baseline_ready": self.baseline_realsense is not None,
                "monocular_calibrated": (
                    self.calibration is not None
                    and (self.camera_id != "logitech" or self._logitech_measurement_ready())
                ),
                "realsense_depth": hardware,
                "monocular_depth": monocular,
                "camera_intrinsics_ready": self.latest_intrinsics is not None,
                "camera_intrinsics": None if self.latest_intrinsics is None else self.latest_intrinsics.to_dict(),
                "camera_intrinsics_origin": self.latest_intrinsics_origin,
                "volume_status": volume_status,
                "operating_mode": self.config.operating_mode,
                "waste_ledger_enabled": self.config.ledger_active,
                "allow_unclassified_foreground": self.config.allow_unclassified_foreground,
                "bag_only": self.config.bag_only,
                "baseline_frame_count": self.baseline_frame_count,
                "baseline_noise_m": round(self.baseline_noise_m, 6),
                "reference_plane": None if self.reference_plane is None else self.reference_plane.to_dict(),
                "calibration_mode": self.calibration_mode,
                "volume_calibration_factor": self.config.volume_calibration_factor,
                "volume_geometry": self.config.volume_geometry,
                "systematic_error_fraction": self.config.systematic_error_fraction,
                "background_stabilization": dict(self.last_background_stabilization),
                "max_item_volume_l": (
                    self.config.logitech_max_item_volume_l
                    if self.camera_id == "logitech" else self.config.realsense_max_item_volume_l
                ),
                "reference_distance_m": (
                    self.config.logitech_reference_distance_m or None if self.camera_id == "logitech" else None
                ),
                "maximum_tilt_degrees": (
                    self.config.logitech_max_tilt_degrees if self.camera_id == "logitech" else None
                ),
                "hard_maximum_tilt_degrees": (
                    self.config.logitech_hard_max_tilt_degrees if self.camera_id == "logitech" else None
                ),
                "tilt_uncertainty_fraction": (
                    self._logitech_tilt_uncertainty_fraction() if self.camera_id == "logitech" else None
                ),
                "independent_reference_required": self.camera_id == "logitech" and self.config.logitech_require_reference,
                "provisional_metric_allowed": (
                    self.camera_id == "logitech" and self.config.logitech_allow_provisional_metric
                ),
                "committed_bags": self.committed_bags,
                "plant": self.ledger.summary(),
                "auto_deposit": (
                    self.config.auto_deposit and self.config.operating_mode == "waste"
                ),
                "color_waste_streams": dict(self.config.color_waste_streams),
                "bin_capacity_l": self.config.bin_capacity_l or None,
                "bin_fill_percent": (
                    round(self.latest_analysis.bin_total.liters / self.config.bin_capacity_l * 100, 2)
                    if self.config.bin_capacity_l
                    and self.latest_analysis is not None
                    and self.latest_analysis.bin_total is not None
                    else None
                ),
                "bin_polygon": [list(vertex) for vertex in self.config.bin_polygon],
                "latest": None if self.latest_analysis is None else self.latest_analysis.to_dict(),
                "detector_model": self.config.detector_model,
                "depth_model": self.config.depth_model if self.depth_estimator else None,
                "runtime": getattr(self.detector, "runtime", {}),
                "roi": list(self.config.roi),
            }

    def _logitech_tilt_invalid(self) -> bool:
        # Hard block only past the *hard* ceiling: the per-pixel volume math
        # (surface-columns / ray-frustum / triangulated-surface) integrates a
        # swept volume along each camera ray between the baseline and object
        # depth, which is geometrically valid at any mounting angle -- it
        # never assumed a near-overhead view. What tilt actually costs is
        # rising self-occlusion (more of the object's far side is hidden)
        # and Depth Anything's own metric-scale prediction likely being less
        # reliable outside a roughly-overhead framing. Both degrade accuracy
        # continuously rather than "break" at one angle, so past the
        # confident-zone limit this is now a growing uncertainty penalty
        # (see `_logitech_tilt_uncertainty_fraction`), not a hard stop --
        # except past `logitech_hard_max_tilt_degrees`, where the bin floor
        # is barely visible at all and a reported number would be fiction.
        return bool(
            self.camera_id == "logitech"
            and self.config.logitech_require_overhead
            and self.reference_plane is not None
            and self.reference_plane.tilt_degrees > self.config.logitech_hard_max_tilt_degrees
        )

    # Written once per object, in every operating mode. The waste-ledger CSV
    # served by the dashboard is empty in geometry_validation mode because that
    # mode deliberately disables the ledger, which left validation runs with no
    # spreadsheet output at all -- this is the mode-independent record.
    # One schema, owned by the event log, shared by local and cloud modes so the
    # two are directly comparable. Kept as a station attribute because the
    # download route and the operator page both read it from here.
    MEASUREMENT_CSV_COLUMNS = MeasurementEventLog.COLUMNS

    def _is_settled(self, track_id: int | None) -> bool:
        """Has this track's volume held steady long enough to be final?

        Exactly the ledger's own settle rule (`settle_frames` samples within
        `settle_volume_tolerance`), reused rather than reinvented so a mode
        without a ledger finalises on the same evidence a deposit does.
        """
        history = list(self._volume_history.get(track_id, ()))
        if len(history) < self.config.settle_frames:
            return False
        recent = np.asarray(history[-self.config.settle_frames :], dtype=np.float64)
        median = float(np.median(recent))
        tolerance_l = max(0.15, median * self.config.settle_volume_tolerance)
        return float(np.max(recent) - np.min(recent)) <= tolerance_l

    def _finalise_settled_measurements(
        self, detections: list[Detection], timestamp: float | None,
    ) -> None:
        """Persist finalised rows in the modes that have no waste ledger.

        In `waste` mode the ledger's own acceptance is the canonical event and
        persistence happens there, at the moment of the deposit. Geometry
        validation deliberately disables the ledger, which is precisely why
        validation runs used to produce no spreadsheet at all -- so here a
        settled, non-phantom, measured track is the finalised event.
        """
        if self.config.operating_mode == "waste" and self.config.auto_deposit:
            # Waste mode persists at the moment the deposit is accepted; this
            # settle-based path is for every other mode, so that recording never
            # depends on the mode. Exactly one of the two runs.
            return
        for detection in detections:
            if detection.track_id is None or detection.track_id in self._csv_logged:
                continue
            if _is_phantom_detection(detection):
                continue
            if self._detection_volume(detection) is not None and self._is_settled(detection.track_id):
                self.persist_measurement_event(detection, timestamp)
                continue
            track = self.tracker.tracks.get(detection.track_id)
            if track is None or not track.counted:
                continue
            frames = self._unfinalised_frames[detection.track_id] = (
                self._unfinalised_frames.get(detection.track_id, 0) + 1
            )
            if frames >= self.config.finalise_max_frames:
                # An object that never settles (or never gets a valid volume)
                # is still a real outcome: one rejected row with its reason,
                # instead of silently producing no row at all.
                reason = detection.volume_rejection_reason or (
                    "unstable_volume" if self._detection_volume(detection) is not None
                    else "no_valid_measurement"
                )
                self.persist_measurement_event(
                    detection, timestamp, status=STATUS_REJECTED, status_reason=reason,
                )

    def persist_measurement_event(
        self,
        detection: Detection,
        timestamp: float | None,
        *,
        status: str = STATUS_ACCEPTED,
        status_reason: str | None = None,
    ) -> dict[str, Any]:
        """Write the canonical row for one finalised measurement.

        Called at the moment a deposit is accepted (or explicitly withheld) --
        never speculatively per frame -- so a pending object can never become an
        accepted row. Idempotent: the event log keys on a durable `event_id`, so
        a repeated frame, a dashboard refresh, a retry or a cloud reconnection
        resolves to the same id and appends nothing the second time.
        """
        # Keyed on the PERMANENT event id where one exists. Falling back to the
        # detector's track id would reintroduce the renumbering bug, so that
        # fallback only applies to objects that never reached stability (which
        # are rejections, and are meant to be recorded once each).
        permanent = self._permanent_event_ids.get(detection.track_id)
        event_id = resolve_event_id(
            self.event_log.session_id, self.camera_id,
            permanent if permanent is not None else detection.track_id,
            suffix=status if permanent is None else f"{status}-permanent",
        )
        row = self._measurement_row(detection, timestamp, event_id, status, status_reason)
        result = self.event_log.record(row)
        if result.written:
            self.stage_counters["finalised_measurements"] += 1
        if self.camera_id == "logitech" and result.ok and not result.duplicate:
            # A side-car for the accuracy benchmark, keyed by measurement id.
            # The measurement CSV and its schema are deliberately untouched.
            self.store.append_jsonl("logitech_raw_volumes.jsonl", {
                "measurement_id": result.event_id,
                "timestamp": row.get("timestamp"),
                "object_type": row.get("object_type"),
                "raw_volume_l": detection.raw_volume_l,
                "corrected_volume_l": detection.monocular_volume_l,
                "stable_volume_l": detection.stable_volume_l,
                "empirical_factor": (self.last_logitech_volume_diagnostics or {}).get("empirical_factor"),
                "geometry_method": None if detection.shape_geometry is None
                else detection.shape_geometry.geometry_method,
                "length_mm": detection.footprint_length_mm,
                "width_mm": detection.footprint_width_mm,
                "height_mm": detection.physical_height_mm,
            })
        if self.diagnostics.enabled and not result.duplicate:
            context = self._frame_context
            self.diagnostics.record_measurement(
                row=row, frame=context.get("frame"), mask=detection.mask, depth=context.get("depth"),
                points=(context.get("points") or {}).get(id(detection)),
                mask_overlay=self.logitech_mask_overlay() if self.camera_id == "logitech" else None,
                latency_ms=row.get("processing_time_ms"),
            )
        if detection.track_id is not None and (result.ok or result.error):
            # A failed write is queued in the event log for retry; the track
            # is finalised either way, so later frames do not re-create it.
            self._csv_logged.add(detection.track_id)
        if result.ok and self.measurement_listener is not None:
            # Paired comparison is a passive downstream consumer: whatever it
            # does, it must never interrupt this camera's own tracking.
            try:
                self.measurement_listener(row)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Comparison listener failed for %s", result.event_id)
        self._last_persist_result = {
            "event_id": result.event_id,
            "written": result.written,
            "duplicate": result.duplicate,
            "error": result.error,
            "csv_path": None if result.path is None else str(result.path),
        }
        return self._last_persist_result

    def _measurement_row(
        self,
        detection: Detection,
        timestamp: float | None,
        event_id: str,
        status: str,
        status_reason: str | None,
    ) -> dict[str, Any]:
        """The shared local/cloud schema, so both modes stay comparable."""
        shape = detection.shape_geometry
        analysis = self.latest_analysis
        return {
            "event_id": event_id,
            "status": status,
            "reason": status_reason,
            "processing_mode": self.config.processing_mode,
            "camera_source": self.camera_id,
            "diagnostic": self.config.diagnostic_mode,
            "model_version": self.config.detector_model,
            "pipeline_version": __version__,
            # One accepted event is one bag: that is exactly what the
            # deposit-isolation rule guarantees -- two touching bags are only
            # accepted once each has been isolated against the committed scene,
            # and a pair that cannot be separated is withheld rather than
            # merged. A withheld event contributes no bag.
            "bag_count": 1 if status == STATUS_ACCEPTED else 0,
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "operating_mode": self.config.operating_mode,
            "calibration_id": self._calibration_id,
            "track_id": detection.track_id,
            "label": detection.label,
            "object_type": detection.canonical_type or detection.label,
            "accepted_class": detection.accepted_class,
            "color": detection.color,
            "color_confidence": round(float(detection.color_confidence), 4),
            "sorting_status": detection.sorting_status,
            "material": detection.material,
            "material_confidence": round(float(detection.material_confidence), 4),
            # Each camera's own volume: Logitech's lives in monocular_volume_l,
            # and a finalised Logitech row carries its stable trimmed median.
            "volume_l": detection.stable_volume_l or self._detection_volume(detection),
            "live_volume_l": self._detection_volume(detection),
            "added_volume_l": detection.added_volume_l,
            "displaced_volume_l": detection.displaced_volume_l,
            "volume_before_l": detection.volume_before_l,
            "volume_after_l": detection.volume_after_l,
            "volume_uncertainty_l": detection.volume_uncertainty_l,
            "length_mm": detection.footprint_length_mm if shape is None else shape.length_mm,
            "width_mm": detection.footprint_width_mm if shape is None else shape.width_mm,
            "height_mm": detection.physical_height_mm if shape is None else shape.height_mm,
            "dimension_confidence": detection.dimension_confidence,
            "dimension_method": detection.dimension_method,
            "depth_coverage_percent": detection.depth_coverage_percent,
            "measurement_method": detection.measurement_method,
            "measurement_quality": detection.measurement_quality,
            "volume_rejection_reason": detection.volume_rejection_reason,
            "calibration_valid": self._calibration_valid,
            # Read by the paired comparison log (paired_events.py); the
            # per-camera CSV keeps its established columns.
            "shape_geometry": None if shape is None else shape.to_dict(),
            "raw_volume_l": detection.raw_volume_l,
            "geometry_method": None if shape is None else shape.geometry_method,
            "calibration_method": None if self.calibration is None else self.calibration.method,
            "processing_time_ms": None if analysis is None else round(float(analysis.inference_ms), 2),
        }

    def _check_camera_placement(
        self,
        depth_m: np.ndarray | None,
        intrinsics: CameraIntrinsics | None,
        bin_region: np.ndarray,
        detections: list[Detection],
        warnings: list[str],
    ) -> None:
        """Invalidate the calibration if the camera no longer matches its pose.

        A support plane is only meaningful for the pose it was fitted at. Moving
        or tilting the camera, or changing its distance to the surface, silently
        turns every height above that plane into a different quantity -- which
        is why a measurement captured at one placement cannot be compared with
        one captured at another. Checked only while the scene is empty, so an
        object in the bin is never mistaken for the floor having moved.
        """
        if (
            self.reference_plane is None
            or self.reference_plane.coefficients is None
            or depth_m is None
            or intrinsics is None
            or detections
        ):
            return
        live = fit_reference_plane(depth_m, intrinsics, mask=bin_region)
        if not reference_plane_is_usable(live) or live.coefficients is None:
            return
        tilt_change = abs(float(live.tilt_degrees) - float(self.reference_plane.tilt_degrees))
        # `c` is the plane's intercept: the camera-axis distance to the surface.
        distance_change = abs(
            float(live.coefficients[2]) - float(self.reference_plane.coefficients[2])
        )
        moved = (
            tilt_change > self.config.camera_move_max_tilt_deg
            or distance_change > self.config.camera_move_max_distance_m
        )
        if moved and self._calibration_valid:
            self._calibration_valid = False
            warnings.append(
                "camera_moved_recalibration_required: the support plane moved by "
                f"{tilt_change:.1f} degrees and {distance_change * 1000:.0f} mm since "
                "calibration; capture a new empty-scene baseline before trusting any volume"
            )
        elif not moved and not self._calibration_valid:
            self._calibration_valid = True

    def _measurement_region(self, shape: tuple[int, ...]) -> np.ndarray:
        """The pixels this camera is allowed to measure inside.

        The drawn mat if one was calibrated for *this* camera at a resolution
        that still matches, and the configured ROI otherwise. A zone drawn at
        another aspect ratio is not stretched onto the frame: it would measure
        a different part of the room, so it is reported as a mismatch and the
        configuration stands in.
        """
        zone = self.measurement_zone
        if zone is not None:
            mask = zone.mask(shape)
            if mask is not None and np.any(mask):
                self.zone_source = "saved-zone"
                return mask
            self.zone_source = "zone_resolution_mismatch"
        elif self.config.bin_polygon:
            self.zone_source = "configured-polygon"
        else:
            self.zone_source = "configured-roi"
        return fixed_bin_mask(shape, self.config.roi, self.config.bin_polygon)

    def set_measurement_zone(
        self, corners, shape: tuple[int, ...], *,
        near_edge_m: float = 0.0, depth_edge_m: float = 0.0,
    ) -> dict[str, Any]:
        """Store this camera's mat outline and, with its real size, its floor scale."""
        zone = MeasurementZone(
            camera=self.camera_id, corners=tuple(corners),
            width_px=int(shape[1]), height_px=int(shape[0]),
            near_edge_m=float(near_edge_m or 0.0), depth_edge_m=float(depth_edge_m or 0.0),
        )
        with self.lock:
            self.measurement_zone = self.zones.save(zone)
            self.zone_source = "saved-zone"
            # The zone defines what may be measured, so anything fitted or
            # smoothed under the previous one is no longer about this scene.
            self._geometry_lock.clear()
            self._track_signatures.clear()
            self.deposit_state.reset()
        return self.measurement_zone.describe()

    def clear_measurement_zone(self) -> None:
        with self.lock:
            self.zones.forget(self.camera_id)
            self.measurement_zone = None
            self.zone_source = "configured-roi"

    def _metric_from_relative(self, predicted: np.ndarray, region: np.ndarray) -> np.ndarray:
        """A relative checkpoint's output, scaled onto the one distance we know.

        A relative model reports an ordering, not metres: its numbers depend on
        whatever else is in the scene, so using them directly makes an object's
        height a function of the furniture behind it. When the configured
        checkpoint is relative, the prediction is scaled so that the floor of
        the measurement zone sits at the measured camera-to-floor distance, and
        the mode says the result is an estimate. Without that distance nothing
        can be scaled, and the reason is recorded rather than papered over.
        """
        self.relative_depth_reason = None
        if depth_output_kind(self.config.depth_model) == METRIC_OUTPUT:
            return predicted
        distance = float(self.config.logitech_reference_distance_m or 0.0)
        usable = region & np.isfinite(predicted) & (predicted > 1e-6)
        values = predicted[usable]
        if distance <= 0 or values.size < 100:
            self.relative_depth_reason = "relative_depth_without_reference_distance"
            return predicted
        # Relative Depth Anything V2 outputs inverse depth: larger is nearer,
        # so the floor's own value fixes the constant in Z = k / d.
        constant = float(np.median(values)) * distance
        scaled = np.full(predicted.shape, np.nan, dtype=np.float32)
        np.divide(constant, predicted, out=scaled, where=usable | (np.isfinite(predicted) & (predicted > 1e-6)))
        self.relative_depth_reason = "relative_depth_scaled_to_measured_floor_distance"
        self.calibration_mode = "relative-depth-estimate"
        return scaled

    def _minimum_rise_m(self) -> float:
        """How far above the committed scene something must stand to be an object.

        Half of the height the measurement itself demands, and for the same
        reason the mask stage uses a smaller pixel floor: the real threshold is
        applied once, on the measurement. Gating twice would throw away a thin
        object that the measurement would have accepted, while a shadow -- which
        raises nothing at all -- still fails this one.
        """
        if self.camera_id == "logitech":
            return 0.5 * float(self.config.logitech_min_object_height_m)
        threshold = (
            self.config.geometry_validation_min_object_height_m
            if self.config.operating_mode == "geometry_validation"
            else self.config.min_object_height_m
        )
        return 0.5 * float(threshold)

    def note_manual_reference_distance(self, distance_m: float) -> None:
        """Record an operator-measured camera height without disturbing anything.

        The height is derived from the fitted floor plane during normal use, so
        a typed value is a cross-check rather than a prerequisite. It is stored
        beside the derived one and the two are compared in the diagnostics; it
        never clears a baseline, a track, a calibration or a committed scene.
        """
        with self.lock:
            self.config.logitech_reference_distance_m = float(distance_m)
            self.logitech_distance_source = "operator_measured"

    def _auto_camera_height(self, plane: Any) -> float | None:
        """The camera's height above the floor, from the floor it just fitted.

        A tape measure was the only way to get this, and until it was typed in
        the readiness panel reported the camera-to-floor distance missing. The
        fitted plane already contains it: the perpendicular distance from the
        camera to that plane is the height.
        """
        if plane is None or getattr(plane, "coefficients", None) is None:
            return None
        height = camera_height_from_plane(plane.coefficients)
        if height is None:
            return None
        if self.logitech_derived_distance_m is None:
            LOGGER.info(
                "Logitech camera height derived from the fitted floor: %.3f m", height,
            )
        self.logitech_derived_distance_m = height
        # Deliberately NOT written into config.logitech_reference_distance_m.
        # That field means "an operator measured this installation", and the
        # research modes, the auto-deposit gate and the calibration-mode label
        # all key off it. A height this code derived from its own depth cannot
        # verify the installation, so it stays in its own field and is used for
        # geometry only. Conflating the two would have let a derived number be
        # reported as an independently measured distance.
        if not self.config.logitech_reference_distance_m:
            self.logitech_distance_source = "derived_from_floor_plane"
        return height

    def camera_height_m(self) -> float | None:
        """The camera's height above the floor, measured or derived.

        Geometry needs a number; provenance decides what may be claimed from
        it. This is the number.
        """
        measured = float(self.config.logitech_reference_distance_m or 0.0)
        if measured > 0:
            return measured
        return self.logitech_derived_distance_m

    def _maybe_learn_empty_baseline(
        self, frame: np.ndarray, depth: np.ndarray | None, region: np.ndarray | None,
        detections: list[Detection],
    ) -> None:
        """Capture the empty scene by itself, once the view is still and empty.

        Plug-and-play means the operator does not have to know that a baseline
        exists, let alone press a button at the right moment. What must never
        happen is capturing one with an object in the zone, so the learner only
        says yes after a run of consecutive frames that are both empty and
        still, and names which condition failed the rest of the time.
        """
        if self.camera_id != "logitech" or not self.config.logitech_auto_baseline:
            return
        if self.reference_rgb is not None and self.reference_monocular is not None:
            return
        inside = [
            item for item in detections
            if not _is_phantom_detection(item)
            and (region is None or bool(np.any(item.mask & region)))
        ]
        decision = self._baseline_learner.observe(
            depth=depth, region=region, objects_in_zone=len(inside),
        )
        self.auto_baseline_state = decision.to_dict()
        if not decision.capture:
            return
        self.set_baseline(frame=frame)
        self._baseline_learner.reset()
        self.stage_counters["logitech_auto_baseline_captured"] += 1
        LOGGER.info(
            "Logitech empty scene learned automatically after %s still frames",
            decision.stable_frames,
        )

    def _recovered_foreground_detections(
        self, frame: np.ndarray, detections: list[Detection],
        change: np.ndarray | None, region: np.ndarray | None,
    ) -> list[Detection]:
        """Objects the detector missed but the empty-scene change did not.

        The small can is visible, changes against the baseline and stands above
        the floor, yet no detector box is proposed for it, so nothing measured
        it. This adds one unknown-object detection per unclaimed island, with a
        low confidence so it reads as weaker evidence than a classified one.
        Nothing about the detector, its vocabulary or its masks changes: this
        only ever appends where the detector proposed nothing.
        """
        if self.camera_id != "logitech" or not self.config.logitech_recover_unclaimed:
            return []
        claimed = combined_mask(detections, frame.shape[:2]) if detections else None
        islands = unclaimed_foreground_islands(
            change, region, claimed,
            min_pixels=minimum_object_pixels(
                frame.shape[:2],
                min(self.config.min_component_pixels, self.config.logitech_min_object_pixels),
            ),
        )
        recovered: list[Detection] = []
        for island in islands:
            rows, columns = np.nonzero(island)
            recovered.append(Detection(
                label="unknown object",
                confidence=self.config.logitech_recovered_confidence,
                box=(int(columns.min()), int(rows.min()),
                     int(columns.max()) + 1, int(rows.max()) + 1),
                mask=island,
                source=CHANGE_RECOVERED_SOURCE,
                color=classify_color(frame, island)[0],
            ))
        if recovered:
            self.stage_counters["logitech_foreground_recovered_objects"] += len(recovered)
        return recovered

    def _logitech_floor_plane(
        self, depth_m: np.ndarray | None, intrinsics: Any, object_mask: np.ndarray | None,
        region: np.ndarray | None,
    ) -> Any:
        """The floor this fixed camera looks at, fitted from its own depth.

        Every Logitech height and footprint is measured relative to a plane. If
        that plane is the one perpendicular to the optical axis -- the fallback
        the calibrated camera height gives -- then on a tilted mount a height
        comes out multiplied by the cosine of the tilt, and the "footprint" is a
        frontal projection rather than a floor one. The camera does not move, so
        the plane is fitted once from the empty baseline where there is one, or
        from this frame's own background, and then reused.
        """
        if self.camera_id != "logitech" or depth_m is None or intrinsics is None:
            return None
        shape = tuple(int(value) for value in depth_m.shape[:2])
        baseline = self.reference_monocular
        has_baseline = baseline is not None and baseline.shape == depth_m.shape
        cached = self._logitech_plane_cache
        if cached is not None and cached[0] == shape:
            # A plane fitted from a live frame was fitted around whatever was
            # in the scene. Once an empty baseline exists it is the better
            # surface, so that one provisional fit is replaced exactly once.
            if self._logitech_plane_source == "empty_baseline" or not has_baseline:
                return cached[1]
        plane = None
        if has_baseline:
            plane = fit_reference_plane(baseline, intrinsics, mask=region)
        source = "empty_baseline"
        if not reference_plane_is_usable(plane):
            plane = fit_support_plane_from_background(
                depth_m, intrinsics, object_mask=object_mask, region_mask=region,
            )
            source = "live_frame_background"
        if not reference_plane_is_usable(plane):
            self._logitech_plane_reason = "no_coherent_floor_plane_in_logitech_depth"
            return None if cached is None else cached[1]
        self._logitech_plane_reason = None
        self._logitech_plane_source = source
        self._logitech_plane_cache = (shape, plane)
        LOGGER.info(
            "Logitech floor plane fitted: tilt=%.1f deg rmse=%.4f m inliers=%s",
            plane.tilt_degrees, plane.residual_rmse_m, plane.inlier_pixels,
        )
        return plane

    def _committed_scene_change(
        self, frame: np.ndarray, depth_m: np.ndarray | None, monocular_depth: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """What has changed since the scene was last committed, and by how much.

        The reference is advanced every time a deposit is committed, so this is
        the difference against everything already counted -- the mat, the
        furniture behind it and the carton deposited a minute ago. Colour finds
        what moved; depth says how far it stands above what was there, which is
        what tells an object from a shadow.
        """
        change = rgb_change(frame, self.reference_rgb, self.config.foreground_threshold)
        depth = monocular_depth if self.camera_id == "logitech" else depth_m
        reference = self.reference_monocular if self.camera_id == "logitech" else self.reference_realsense
        rise = None
        if depth is not None and reference is not None and depth.shape == reference.shape:
            rise = (reference.astype(np.float32) - depth.astype(np.float32))
            rise[~np.isfinite(rise)] = 0.0
            risen = rise >= self._minimum_rise_m()
            change = risen if change is None else (change | risen)
        return change, rise

    def _deposit_measurement_mask(
        self, detection: Detection, instance_mask: np.ndarray, region: np.ndarray,
        change: np.ndarray | None, rise: np.ndarray | None,
    ) -> np.ndarray:
        """The mask this detection is measured with, or an empty one with a reason.

        An empty mask leaves the object detected, tracked, classified and drawn
        -- only its volume is withheld, with the reason on the detection, which
        is what the operator needs to see.
        """
        # The floor here only sweeps away speckle: how large an object has to
        # be to count is decided downstream, on the measurement itself, and
        # applying that threshold twice would drop small objects entirely.
        speckle = max(4, min(self.config.min_component_pixels, self.config.logitech_min_object_pixels) // 8)
        choice = deposit_component(
            instance_mask, change, region, rise_m=rise,
            min_pixels=speckle, min_height_rise_m=self._minimum_rise_m(),
        )
        self.last_measurement_mask = {
            "label": detection.label, "track_id": detection.track_id,
            "source": choice.source, "reason": choice.reason, **choice.diagnostics,
        }
        if choice.reason:
            self.stage_counters[f"measurement_mask_rejected_{choice.reason}"] += 1
        track = detection.track_id
        if choice.measurable:
            if track is not None:
                self._deposit_refusals.pop(track, None)
            return choice.mask if choice.source == FOREGROUND_COMPONENT else instance_mask
        if choice.reason == NO_NEW_DEPOSIT and track is not None:
            refusals = self._deposit_refusals[track] = self._deposit_refusals.get(track, 0) + 1
            if refusals >= self.config.baseline_contains_object_frames:
                # A tracked object that never differs from the committed scene
                # is in that scene: the baseline was captured with it already
                # standing there. Refusing it for ever is how a detected
                # object stayed "pending" with no height and no litres, so it
                # is measured from its own mask and the reason is recorded.
                fallback = clean(instance_mask & region, max(4, self.config.min_component_pixels // 8))
                candidate = fallback if np.any(fallback) else instance_mask
                # Furniture reaches this branch by the same road an object
                # present at baseline capture does: it never changes, so its
                # refusals accumulate. The difference is physical -- a deposit
                # sits inside the zone and has an outside, the bed and the
                # floor reach the zone's own edges. Without this the fallback
                # measured the bed as a 35 L "unclassified object".
                static_reason = static_background_reason(candidate, region)
                if static_reason is not None:
                    detection.volume_rejection_reason = static_reason
                    detection.measurement_quality = static_reason
                    self.stage_counters[f"measurement_mask_static_{static_reason}"] += 1
                    return np.zeros_like(instance_mask)
                detection.measurement_quality = "baseline-contains-object"
                detection.volume_rejection_reason = None
                self.stage_counters["measurement_mask_baseline_contains_object"] += 1
                return candidate
        detection.volume_rejection_reason = choice.reason
        # The live table shows the measurement quality, so the precise reason
        # belongs there too: "pending - reference distance estimate" named the
        # cascade mode, never the thing that actually stopped the measurement.
        detection.measurement_quality = str(choice.reason)
        return np.zeros_like(instance_mask)

    def _observe_deposit_state(
        self, detections: list[Detection], change: np.ndarray | None, region: np.ndarray,
    ) -> None:
        """Tell the deposit machine what this frame saw."""
        available = max(int(np.count_nonzero(region)), 1)
        changed = 0.0 if change is None else int(np.count_nonzero(change & region)) / available
        measured = [
            self._detection_volume(item) for item in detections
            if self._detection_volume(item) is not None
        ]
        self.deposit_state.observe(FrameObservation(
            changed_fraction=changed,
            tracked_objects=sum(1 for item in detections if item.track_id is not None),
            measured_volume_l=measured[0] if measured else None,
            stable=bool(measured) and not any(
                item.volume_rejection_reason for item in detections
            ),
        ))

    def _measurement_readiness(self) -> MeasurementReadiness:
        """Each prerequisite for a metric measurement, ready or missing."""
        zone = self.measurement_zone
        logitech = self.camera_id == "logitech"
        mapping = True
        if logitech:
            mapping = self.calibration is not None and self.calibration_rejected_reason is None
        reason = self.relative_depth_reason or self.calibration_rejected_reason or ""
        if not reason and self.last_measurement_mask.get("reason"):
            reason = str(self.last_measurement_mask["reason"])
        samples = len(self.metric_store.samples(CALIBRATION_SET)) if logitech else 0
        height_ready = not logitech or self.height_calibration is not None
        if logitech and self.height_calibration_reason:
            reason = reason or self.height_calibration_reason
        return MeasurementReadiness(
            camera=self.camera_id,
            measurement_zone=zone is not None or bool(self.config.bin_polygon),
            empty_baseline=self.reference_rgb is not None,
            intrinsics=self.latest_intrinsics is not None,
            floor_scale=bool(zone is not None and zone.has_floor_scale),
            metric_depth_mapping=bool(mapping),
            # Satisfied by a height derived from the fitted floor as well as by
            # a measured one, so normal use never waits on a tape measure. The
            # source is reported separately, and the modes that require an
            # independently measured installation still check the config field
            # itself rather than this flag.
            camera_floor_distance=not logitech or bool(self.camera_height_m()),
            height_calibration=bool(height_ready),
            method=self.calibration_mode or getattr(self, "measurement_method", "") or "",
            reason=reason,
            depth_output=depth_output_kind(self.config.depth_model) if logitech else "hardware_depth",
            calibration_samples=samples,
            calibration_samples_required=RECOMMENDED_SAMPLES if logitech else 0,
            calibration_status=(
                "missing" if not logitech or self.height_calibration is None
                else self.height_calibration.status
            ),
        )

    # ------------------------------------------------ the Logitech metric layer
    def _camera_setup(self, shape: tuple[int, ...] | None = None) -> CameraSetup:
        """The installation this camera is in right now."""
        if shape is None:
            frame = self.latest_frame if self.latest_frame is not None else self.latest_processed_frame
            shape = (0, 0) if frame is None else frame.shape[:2]
        saved = self.metric_store.setup
        return CameraSetup(
            camera=self.camera_id, width_px=int(shape[1]), height_px=int(shape[0]),
            camera_floor_distance_cm=float(self.config.logitech_reference_distance_m or 0.0) * 100.0,
            zone_signature=zone_signature(self.measurement_zone),
            setup_id="" if saved is None else saved.setup_id,
            note="" if saved is None else saved.note,
        )

    def save_camera_setup(
        self, *, camera_floor_distance_cm: float, setup_id: str = "", note: str = "",
        shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Record the physical setup a calibration will belong to.

        The distance is from the camera's optical centre straight down to the
        empty measurement floor. It is part of the calibration's identity: move
        the camera and the mapping fitted under the old height is no longer
        about this scene.
        """
        distance = float(camera_floor_distance_cm)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Measure the camera-to-floor distance in centimetres")
        with self.lock:
            self.config.logitech_reference_distance_m = distance / 100.0
            setup = self._camera_setup(shape)
            setup = replace(setup, camera_floor_distance_cm=distance,
                            setup_id=setup_id or setup.setup_id, note=note or setup.note)
            stored = self.metric_store.save_setup(setup)
            self._validate_height_calibration(stored)
            # The restore path reads this file at startup. Writing only the
            # setup record left a station that had been given its distance
            # asking for it again after every restart -- and measuring as an
            # uncalibrated estimate until someone noticed.
            try:
                self.store.save_json("calibration/reference_distance.json",
                                     {"distance_m": distance / 100.0, "captured_at": time.time()})
            except (OSError, ValueError) as exc:  # pragma: no cover - disk issues
                LOGGER.warning("Could not persist the Logitech reference distance: %s", exc)
        return stored.to_dict()

    def _validate_height_calibration(self, setup: CameraSetup | None = None) -> None:
        """Refuse a calibration that belongs to another installation."""
        if self.camera_id != "logitech":
            return
        calibration = self.metric_store.calibration
        if calibration is None:
            self.height_calibration, self.height_calibration_reason = None, None
            return
        current = setup or self._camera_setup()
        # Judged against the installation the mapping was *fitted* in, not the
        # one saved a moment ago: saving a new setup is exactly the event that
        # should invalidate an older calibration.
        recorded = self.metric_store.setup
        if calibration.setup_snapshot:
            try:
                recorded = CameraSetup.from_dict(calibration.setup_snapshot)
            except (KeyError, TypeError, ValueError):
                pass
        difference = current.difference(recorded)
        if difference and difference != "no_saved_camera_setup":
            self.height_calibration = None
            self.height_calibration_reason = f"calibration_invalidated_{difference}"
            return
        self.height_calibration, self.height_calibration_reason = calibration, None

    def _metric_signal(
        self, mask: np.ndarray, prediction: np.ndarray | None,
    ) -> np.ndarray | None:
        """How far each object pixel stands above the empty floor, before calibration.

        The difference between the empty-scene prediction and this frame's, in
        the model's own units read as centimetres. It is a *signal*, not a
        height: turning it into centimetres of real object is exactly what the
        calibration does.
        """
        reference = self.reference_monocular
        if prediction is None or reference is None or prediction.shape != reference.shape:
            return None
        if mask.shape != prediction.shape:
            return None
        signal = (reference.astype(np.float32) - prediction.astype(np.float32)) * 100.0
        signal[~np.isfinite(signal)] = 0.0
        return signal

    def _apply_metric_calibration(
        self, detection: Detection, mask: np.ndarray, prediction: np.ndarray | None, result: Any,
    ) -> None:
        """Report calibrated centimetres, and keep the provisional numbers beside them.

        Length and width come from the object's contact with the mat, warped to
        a bird's-eye view through the mat's homography, so an object at the back
        of the mat is not reported smaller than the same object at the front.
        Only the contact band is warped: the homography maps the floor plane and
        nothing else, so warping a standing object's whole silhouette reports
        its shadow instead of its base. Height comes from the fitted mapping,
        and the volume is the per-pixel integral of the two.
        """
        if self.camera_id != "logitech":
            return
        signal = self._metric_signal(mask, prediction)
        if signal is None or not np.any(mask):
            return
        zone = self.measurement_zone
        area_cm2 = None
        if zone is not None and zone.has_floor_scale:
            metres = zone.pixel_area_m2(mask.shape)
            area_cm2 = None if metres is None else metres * 10000.0
        raw = robust_height_cm(signal[mask])
        # Only the pixels that actually touch the floor may be warped through
        # the mat homography. Warping the whole silhouette smeared a standing
        # object into its own shadow and reported a 5 cm can as 11.5 cm wide;
        # see logitech_footprint.py for the geometry.
        measured = measure_footprint(
            zone, mask,
            camera_height_m=self.camera_height_m(),
            zone_limits_m=_zone_limits_m(zone),
        )
        footprint = (
            (measured.length_m, measured.width_m, measured.area_m2 or 0.0)
            if measured.ok else None
        )
        self.last_footprint_result = measured
        self.last_metric_context = {
            "signal_cm": float(raw.get("top_cm", 0.0)),
            "signal_mean_cm": float(raw.get("mean_cm", 0.0)),
            "mask_pixels": int(np.count_nonzero(mask)),
            "length_cm": None if footprint is None else round(footprint[0] * 100.0, 2),
            "width_cm": None if footprint is None else round(footprint[1] * 100.0, 2),
            "mask": mask.copy(), "signal_map": signal, "prediction": prediction,
            "label": detection.label, "track_id": detection.track_id,
            # How the footprint was obtained, so an operator can tell a real
            # measurement from a refusal without reading the logs.
            "footprint_method": measured.method or None,
            "footprint_reason": measured.reason,
            "footprint_contact_pixels": measured.contact_pixels,
            "footprint_occlusion_corrected": measured.occlusion_corrected,
        }
        # Whatever the provisional path produced, kept for comparison.
        detection.uncalibrated_length_mm = detection.footprint_length_mm
        detection.uncalibrated_width_mm = detection.footprint_width_mm
        detection.uncalibrated_height_mm = detection.physical_height_mm
        detection.uncalibrated_volume_l = detection.monocular_volume_l

        if footprint is None and measured.reason:
            detection.volume_rejection_reason = measured.reason
            if detection.measurement_quality in (None, "", "measured"):
                detection.measurement_quality = measured.reason
        # The metric 3-D path measures a cross-section of the object in floor
        # coordinates; this one warps the mat. Where both ran, the 3-D one is
        # the measurement and this stays a cross-check, so its numbers are not
        # written over the better ones.
        has_metric_geometry = bool(
            result is not None and getattr(result, "diagnostics", None)
            and result.diagnostics.get("length_mm")
        )
        if footprint is not None and not has_metric_geometry:
            # The mat's own scale, so the same object measures the same at the
            # front and the back of the perspective view.
            detection.footprint_length_mm = round(footprint[0] * 1000.0, 2)
            detection.footprint_width_mm = round(footprint[1] * 1000.0, 2)
            detection.dimension_method = measured.method or "logitech_contact_band_footprint"
        elif footprint is not None:
            self.last_metric_context["homography_length_cm"] = round(footprint[0] * 100.0, 2)
            self.last_metric_context["homography_width_cm"] = round(footprint[1] * 100.0, 2)

        calibration = self.height_calibration
        if calibration is None:
            detection.calibration_version = None
            if detection.measurement_quality in (None, "", "measured"):
                detection.measurement_quality = self.calibration_mode or "uncalibrated-estimate"
            return

        heights_cm = np.asarray(calibration.apply(signal), dtype=np.float64)
        heights_cm = np.where(mask, heights_cm, 0.0)
        ceiling_cm = max(20.0, float(self.config.logitech_reference_distance_m or 0.0) * 100.0)
        statistics = robust_height_cm(heights_cm[mask], max_cm=ceiling_cm)
        if "top_cm" not in statistics:
            return
        detection.physical_height_mm = round(float(statistics["top_cm"]) * 10.0, 2)
        detection.height_above_baseline_cm = round(float(statistics["top_cm"]), 1)
        if not has_metric_geometry:
            detection.dimension_method = (
                "logitech_calibrated_homography" if footprint is not None
                else "logitech_calibrated_height"
            )
        detection.calibration_version = f"{calibration.mapping}:{calibration.setup_id or 'setup'}"
        refusal = self._implausible_measurement(detection, float(statistics["top_cm"]), zone)
        if refusal:
            detection.monocular_volume_l = None
            detection.physical_height_mm = None
            detection.height_above_baseline_cm = None
            detection.volume_rejection_reason = refusal
            detection.measurement_quality = refusal
            return
        litres = integrate_volume_l(np.clip(heights_cm, 0.0, ceiling_cm), area_cm2, mask)
        if litres is not None and litres > 0:
            detection.monocular_volume_l = round(litres, 6)
            detection.measurement_method = "logitech_calibrated_height_map"
            detection.measurement_quality = (
                "calibrated" if calibration.frozen else "provisional-calibration"
            )
        self.last_metric_context.update({
            "height_cm": float(statistics["top_cm"]),
            "volume_l": detection.monocular_volume_l,
        })

    def _implausible_measurement(
        self, detection: Detection, height_cm: float, zone: MeasurementZone | None,
    ) -> str:
        """Refuse a number the installation makes impossible.

        An object cannot be taller than the camera is high, wider than the mat
        it stands on, or hold more than the zone could contain. Publishing such
        a value and letting a reader notice later is worse than saying why.
        """
        distance_cm = float(self.config.logitech_reference_distance_m or 0.0) * 100.0
        if distance_cm and height_cm > distance_cm:
            return "height_exceeds_camera_floor_distance"
        if height_cm <= 0:
            return "no_positive_object_height"
        if zone is not None and zone.has_floor_scale:
            longest_cm = max(zone.near_edge_m, zone.depth_edge_m) * 100.0
            for value in (detection.footprint_length_mm, detection.footprint_width_mm):
                if value and value / 10.0 > 1.2 * longest_cm:
                    return "dimension_larger_than_measurement_zone"
            capacity_l = (zone.near_edge_m * zone.depth_edge_m
                          * max(height_cm / 100.0, 0.01)) * 1000.0
            if detection.monocular_volume_l and detection.monocular_volume_l > 1.5 * capacity_l:
                return "volume_exceeds_measurement_zone_capacity"
        return ""

    def _log_measurement_chain(self) -> None:
        """One line per state change, naming every link in the chain.

        Printed when something in it changes rather than every frame, so a
        session log shows the moment a calibration arrived or a resolution
        moved -- which is what turns "pending" into a diagnosis.
        """
        if self.camera_id != "logitech":
            return
        readiness = self._measurement_readiness()
        frame = self.latest_processed_frame
        zone = self.measurement_zone
        calibration = self.height_calibration
        chain = (
            self.camera_id,
            None if frame is None else (int(frame.shape[1]), int(frame.shape[0])),
            str(self.metric_store.directory), None if calibration is None else
            f"{calibration.mapping}:{calibration.status}",
            self.zone_source, None if zone is None else zone.has_floor_scale,
            round(float(self.config.logitech_reference_distance_m or 0.0), 3),
            self.reference_monocular is not None, self.config.depth_model,
            readiness.depth_output, self.calibration_mode, readiness.fully_calibrated,
            readiness.missing[:1], self.relative_depth_reason or self.height_calibration_reason or "",
        )
        if chain == self._last_chain:
            return
        self._last_chain = chain
        LOGGER.info(
            "Logitech chain: resolution=%s calibration_dir=%s height_calibration=%s zone=%s "
            "floor_scale=%s camera_height_m=%s empty_baseline=%s depth_model=%s output=%s "
            "mode=%s ready=%s missing=%s reason=%s",
            *chain[1:],
        )

    # -------------------------------------------------- operator calibration steps
    def capture_empty_zone(self) -> dict[str, Any]:
        """Capture the empty measurement zone, or say exactly why it was refused.

        A baseline taken while something is still in the zone poisons every
        later measurement -- that object becomes part of the floor. The checks
        are therefore explicit, and the baseline itself is the temporal median
        the existing capture already builds rather than one noisy frame.
        """
        with self.lock:
            frame = self.latest_frame
            if frame is None:
                return {"ok": False, "reason": "no_camera_frame_yet"}
            region = self._measurement_region(frame.shape)
            analysis = self.latest_analysis
            inside = [
                item for item in (analysis.detections if analysis is not None else [])
                if item.track_id is not None and not _is_phantom_detection(item)
            ]
            if inside:
                return {"ok": False, "reason": "object_inside_measurement_zone",
                        "objects": [item.label for item in inside]}
            depth = self.latest_monocular_depth if self.camera_id == "logitech" else self.latest_depth
            if depth is None:
                return {"ok": False, "reason": "no_depth_for_this_camera"}
            if depth.shape == region.shape:
                valid = np.isfinite(depth) & (depth > 0.05) & region
                coverage = int(np.count_nonzero(valid)) / max(int(np.count_nonzero(region)), 1)
                if coverage < 0.5:
                    return {"ok": False, "reason": "insufficient_valid_depth_in_zone",
                            "coverage": round(coverage, 3)}
            frames = [item for item in self._recent_depth_frames if item.shape == depth.shape]
            if len(frames) >= 3:
                spread = float(np.median(np.std(np.stack(frames[-5:]), axis=0)))
                if spread > max(0.05, 4.0 * float(self.config.depth_noise_m)):
                    return {"ok": False, "reason": "scene_not_stable", "spread_m": round(spread, 4)}
        captured = self.set_baseline()
        self._validate_height_calibration()
        return {"ok": True, "baseline": captured, "samples": len(self._recent_depth_frames)}

    def add_height_sample(
        self, *, name: str, true_length_cm: float, true_width_cm: float, true_height_cm: float,
        true_volume_l: float | None = None, kind: str = CALIBRATION_SET,
    ) -> dict[str, Any]:
        """Record the object standing in the zone, with the size a ruler says it is."""
        context = dict(self.last_metric_context)
        if not context or not context.get("signal_cm"):
            raise ValueError("Place the object in the measurement zone and wait for a measurement")
        if float(true_height_cm) <= 0:
            raise ValueError("The object's true height in centimetres is required")
        setup = self.metric_store.setup
        sample = HeightSample(
            name=str(name or context.get("label") or "object"),
            true_length_cm=float(true_length_cm or 0.0), true_width_cm=float(true_width_cm or 0.0),
            true_height_cm=float(true_height_cm), true_volume_l=true_volume_l,
            signal_cm=float(context["signal_cm"]), signal_mean_cm=float(context.get("signal_mean_cm", 0.0)),
            measured_length_cm=float(context.get("length_cm") or 0.0),
            measured_width_cm=float(context.get("width_cm") or 0.0),
            measured_height_cm=float(context.get("height_cm") or context["signal_cm"]),
            mask_pixels=int(context.get("mask_pixels", 0)),
            setup_id="" if setup is None else setup.setup_id,
            kind=kind if kind in (CALIBRATION_SET, EVALUATION_SET) else CALIBRATION_SET,
            artefacts=self._save_sample_artefacts(name, context),
        )
        stored = self.metric_store.add_sample(sample)
        return {"sample": stored.to_dict(), **self.metric_status()}

    def _save_sample_artefacts(self, name: str, context: dict[str, Any]) -> dict[str, str]:
        """Keep the evidence behind a calibration sample, so a fit can be audited."""
        directory = self.config.results_dir / "calibration" / "samples"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = directory / f"{stamp}_{''.join(ch for ch in str(name) if ch.isalnum() or ch in '-_')[:40]}"
        artefacts: dict[str, str] = {}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            frame = self.latest_processed_frame
            if frame is not None:
                import cv2

                cv2.imwrite(str(folder / "frame.png"), frame)
                artefacts["frame"] = str(folder / "frame.png")
            for key, array in (("mask", context.get("mask")),
                               ("prediction", context.get("prediction")),
                               ("baseline_relative", context.get("signal_map"))):
                if array is not None:
                    np.save(folder / f"{key}.npy", np.asarray(array))
                    artefacts[key] = str(folder / f"{key}.npy")
        except (OSError, ValueError, ImportError) as exc:  # pragma: no cover - disk/codec issues
            LOGGER.warning("Could not save calibration artefacts (%s)", exc)
        return artefacts

    def fit_height_calibration(self) -> dict[str, Any]:
        """Fit the mapping from the ruler-measured samples and keep the best."""
        samples = self.metric_store.samples(CALIBRATION_SET)
        setup = self.metric_store.setup
        calibration, reason = fit_height_calibration(
            samples, setup_id="" if setup is None else setup.setup_id, setup=setup,
        )
        if calibration is None:
            return {"ok": False, "reason": reason, **self.metric_status()}
        self.metric_store.save_calibration(calibration)
        self._validate_height_calibration()
        return {"ok": True, **self.metric_status()}

    def freeze_height_calibration(self) -> dict[str, Any]:
        """Freeze the mapping, so evaluation objects are measured by a fixed rule."""
        frozen = self.metric_store.freeze()
        self._validate_height_calibration()
        return {"ok": frozen is not None, **self.metric_status()}

    def reset_height_calibration(self, *, samples: bool = False) -> dict[str, Any]:
        with self.lock:
            self.metric_store.save_calibration(None)
            if samples:
                self.metric_store.clear_samples()
            self.height_calibration, self.height_calibration_reason = None, None
        return self.metric_status()

    def metric_status(self) -> dict[str, Any]:
        """Everything the calibration panel shows for this camera."""
        if self.camera_id != "logitech":
            return {"camera": self.camera_id, "applies": False}
        status = self.metric_store.status(required=RECOMMENDED_SAMPLES)
        zone = self.measurement_zone
        status.update({
            "camera": self.camera_id, "applies": True,
            "measurement_zone": None if zone is None else zone.describe(),
            "floor_dimensions": "ready" if zone is not None and zone.has_floor_scale else "missing",
            "camera_floor_distance_cm": round(
                float(self.config.logitech_reference_distance_m or 0.0) * 100.0, 1) or None,
            "empty_baseline": "ready" if self.reference_monocular is not None else "missing",
            "active": None if self.height_calibration is None else self.height_calibration.to_dict(),
            "invalidated_reason": self.height_calibration_reason,
            "measurement_method": getattr(self, "measurement_method", "") or "",
            # Normal use needs neither of these entered by hand. The panel says
            # where each came from, so a derived value is never read as a
            # measured one, and an operator's tape measure is visibly a
            # cross-check rather than a prerequisite.
            "camera_floor_distance_source": self.logitech_distance_source,
            "camera_floor_distance_derived_cm": (
                None if self.logitech_derived_distance_m is None
                else round(self.logitech_derived_distance_m * 100.0, 1)
            ),
            "floor_plane_source": self._logitech_plane_source,
            "floor_plane_reason": self._logitech_plane_reason,
            "auto_baseline": self.auto_baseline_state,
        })
        return status

    def _verify_track_identity(self, detection: Detection) -> None:
        """Drop this track's histories when the id has changed hands.

        The tracker reuses integer ids, and every smoother in the pipeline is
        keyed by one. Without this check a new object inherits the previous
        holder's volume samples, colour votes and frozen geometry, which is
        what repeated one object's dimensions on the next.
        """
        track_id = detection.track_id
        if track_id is None:
            return
        signature = _object_signature(detection, self.camera_id)
        known = self._track_signatures.get(track_id)
        if known is not None and not known.matches(signature, tolerance=2.0):
            self._release_expired_track_state([track_id])
        self._track_signatures[track_id] = signature

    def _release_expired_track_state(self, expired_ids: list[int]) -> None:
        """Drop every per-track buffer belonging to a track the tracker closed.

        These histories are keyed by track id and were previously only ever
        emptied wholesale at a baseline capture or an explicit reset, so a
        session that presented fifty objects carried fifty objects' worth of
        colour votes, material votes, volume samples and box measurements for
        as long as it ran. Freeing them at expiry bounds the pipeline's state by
        the number of *live* tracks rather than by the number ever seen, and
        guarantees a later track can never read another object's history --
        including after `reset_live_tracking()`, which restarts id allocation.
        `_session_seen_tracks` is deliberately excluded: it is the session
        summary the dashboard counts, not a measurement input.
        """
        for track_id in expired_ids:
            self._volume_history.pop(track_id, None)
            self._box_measurement_history.pop(track_id, None)
            self._box_frames_considered.pop(track_id, None)
            self._geometry_lock.forget(track_id)
            self._unfinalised_frames.pop(track_id, None)
            self._logitech_volume_samples.pop(track_id, None)
            self._measurement_frames.pop(track_id, None)
            self._stability.pop(track_id, None)
            self._deposit_refusals.pop(track_id, None)
            self._logitech_volume_spread.pop(track_id, None)
            self._color_history.pop(track_id, None)
            self._material_history.pop(track_id, None)
            self._material_frame_counts.pop(track_id, None)
            self._bin_total_before_track.pop(track_id, None)
            self._track_signatures.pop(track_id, None)

    def _record_added_volume(
        self,
        detection: Detection,
        bin_total: VolumeMeasurement | None,
        scene_grid: HeightMapVolume | None,
    ) -> bool:
        """Attach this deposit's own incremental volume; False withholds the deposit.

        Against the *committed* scene, cell by cell, not against the detection's
        own mask. Two touching black bags merge into one mask and one depth
        component -- nothing in colour or class can separate identical
        polythene -- so the first bag's cells already carry its height in the
        committed grid and differencing leaves only what the second bag raised.
        Without this the merged pair is remeasured as a single new ~29 L object.

        Returns False when the new material cannot be isolated (nothing changed,
        or the existing pile moved enough that its volume would be counted
        twice). The caller then withholds the deposit with an explicit reason
        rather than recording a combined figure.
        """
        before = self._bin_total_before_track.pop(detection.track_id, None)
        if bin_total is not None:
            detection.volume_after_l = round(float(bin_total.liters), 6)
        if before is not None:
            detection.volume_before_l = round(before, 6)

        if self._committed_scene is None:
            # First deposit into a bin with no committed scene: the object's own
            # measurement is the increment, and there is nothing to double-count.
            if detection.realsense_volume_l is not None:
                detection.added_volume_l = detection.realsense_volume_l
            elif before is not None and bin_total is not None:
                detection.added_volume_l = round(
                    max(0.0, float(bin_total.liters) - before), 6
                )
            return True
        increment = incremental_deposit(
            self._committed_scene,
            scene_grid,
            min_change_m=self.config.min_object_height_m,
            max_added_l=self.config.realsense_max_item_volume_l,
        )
        if increment is None:
            detection.volume_rejection_reason = "unstable_depth"
            return False
        if not increment.is_valid:
            detection.volume_rejection_reason = increment.rejection_reason
            detection.measurement_quality = increment.rejection_reason
            return False
        detection.added_volume_l = round(increment.added_liters, 6)
        detection.displaced_volume_l = round(increment.displaced_liters, 6)
        return True

    def _logitech_tilt_uncertainty_fraction(self) -> float:
        """Graduated confidence penalty for mounting tilt above the confident zone.

        0 below `logitech_max_tilt_degrees` (the well-tested "confident"
        limit); ramps linearly up to `logitech_max_tilt_uncertainty_fraction`
        at `logitech_hard_max_tilt_degrees` (where measurement is blocked
        entirely by `_logitech_tilt_invalid`).
        """
        if (
            self.camera_id != "logitech"
            or not self.config.logitech_require_overhead
            or self.reference_plane is None
        ):
            return 0.0
        tilt = self.reference_plane.tilt_degrees
        low = self.config.logitech_max_tilt_degrees
        high = self.config.logitech_hard_max_tilt_degrees
        if tilt <= low:
            return 0.0
        span = max(1e-6, high - low)
        return self.config.logitech_max_tilt_uncertainty_fraction * min(1.0, (tilt - low) / span)

    def prepare_input(
        self, frame: np.ndarray, intrinsics: CameraIntrinsics | None,
    ) -> tuple[np.ndarray, CameraIntrinsics | None]:
        """Lens-undistort a Logitech frame; every other camera passes through."""
        if self.logitech_lens is None:
            return frame, intrinsics
        return self.logitech_lens.prepare(frame, intrinsics)

    def predict_depth(self, frame: np.ndarray) -> np.ndarray | None:
        """Depth Anything V2 for one Logitech frame, at most every logitech_depth_interval_s."""
        if self.depth_estimator is None:
            return None
        now = time.monotonic()
        cached = self._depth_cache
        if (
            cached is not None and cached[1].shape == frame.shape[:2]
            and now - cached[0] < self.config.logitech_depth_interval_s
        ):
            return cached[1]
        self.stage_counters["da_v2_calls"] += 1
        try:
            prediction = self.depth_estimator.estimate_batch([frame])[0]
        except Exception as exc:  # noqa: BLE001
            # A model that fails at inference must say so, not silently
            # leave the camera "offline" forever.
            LOGGER.exception("Logitech Depth Anything V2 inference failed")
            self.depth_load_error = f"inference failed: {type(exc).__name__}: {exc}"
            return None
        self._depth_cache = (now, prediction)
        return prediction

    def calibrate_empty_scene(self, camera_height_m: float) -> dict[str, Any]:
        """One-time Logitech metric calibration from the empty scene and camera height.

        The measurement area must be empty: the tape-measured lens-to-floor
        distance fixes Depth Anything V2's scale on the empty support plane,
        the plane becomes the reference, and both are saved and reused on the
        next start. No known-volume object is involved.
        """
        if self.camera_id != "logitech":
            raise ValueError("Empty-scene metric calibration applies to the Logitech camera")
        if not np.isfinite(camera_height_m) or not 0.2 <= camera_height_m <= 5.0:
            raise ValueError("Enter the measured camera-to-empty-floor distance (0.2-5 m)")
        if self.depth_estimator is None:
            raise ValueError("Depth Anything V2 is not loaded: " + (self.depth_load_error or "depth disabled"))
        analysis = self.latest_analysis
        if analysis is not None and any(
            item.tracking_status != "tentative" and not _is_phantom_detection(item) for item in analysis.detections
        ):
            raise ValueError("Remove every object from the Logitech view before calibrating the empty scene")
        with self.lock:
            if not self._recent_monocular_frames:
                raise ValueError("No Logitech Depth Anything V2 prediction has been received yet")
            latest = self._recent_monocular_frames[-1]
            frames = [item for item in self._recent_monocular_frames if item.shape == latest.shape]
            predicted = np.median(np.stack(frames), axis=0).astype(np.float32)
            intrinsics = self.latest_intrinsics
            region = fixed_bin_mask(self.latest_frame.shape, self.config.roi, self.config.bin_polygon)
        if intrinsics is None:
            raise ValueError("The Logitech camera has not reported intrinsics yet")
        calibration, diagnostics = fit_plane_alignment(
            predicted, region, intrinsics, float(camera_height_m),
            inverse=depth_output_kind(self.config.depth_model) != "metric",
        )
        if calibration is None:
            raise ValueError(f"Empty-plane calibration failed: {diagnostics.get('reason')}")
        calibration.roi = tuple(self.config.roi)
        calibration.intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        calibration.device = str(getattr(self, "camera_device", "") or self.camera_id)
        self.config.logitech_reference_distance_m = float(camera_height_m)
        sample = self.logitech_calibration.set_calibration(calibration, diagnostics)
        baseline = self.set_baseline()
        self.store.save_json("calibration/reference_distance.json",
                             {"distance_m": float(camera_height_m), "captured_at": time.time()})
        return {"calibration": sample, "baseline": baseline, "diagnostics": diagnostics,
                "status": self.logitech_calibration_status()}

    def _apply_cylinder_geometry(self, detection: Detection) -> None:
        """Make a fitted cylinder the object's reported dimensions and volume.

        Before this, the router could say "cylinder" while the overlay, API and
        CSV still read the support-plane footprint rectangle (an upright bottle
        showed 83 x 23 mm: its visible arc) and the height-map litres. Every
        downstream reader uses these same fields, so they now carry the fitted
        diameter x diameter x height and pi r^2 h. A cylinder candidate whose
        fit was rejected stays pending with its reason -- never a box volume.
        Cuboids and irregular objects keep the existing measurement untouched.
        """
        shape = detection.shape_geometry
        if shape is None:
            return
        logitech = self.camera_id == "logitech"
        if shape.geometry_method == CYLINDER:
            detection.footprint_length_mm = round(float(shape.length_mm), 2)
            detection.footprint_width_mm = round(float(shape.width_mm), 2)
            detection.physical_height_mm = round(float(shape.height_mm), 2)
            detection.height_above_baseline_cm = round(float(shape.height_mm) / 10.0, 1)
            detection.dimension_confidence = round(float(shape.geometry_confidence), 4)
            detection.dimension_method = f"fitted_cylinder_{shape.cylinder_orientation}"
            detection.measurement_method = "cylinder_pi_r2_h"
            volume = round(float(shape.selected_volume_litres), 6)
            if logitech:
                detection.monocular_volume_l = volume
            else:
                detection.realsense_volume_l = volume
        elif shape.geometry_method == UNCERTAIN and shape.rejection_reason in CYLINDER_REJECTIONS:
            measured = detection.monocular_volume_l if logitech else detection.realsense_volume_l
            if measured is None:
                # Nothing was measured by any other means, so the rejection is
                # the whole story.
                detection.volume_rejection_reason = shape.rejection_reason
            # Otherwise the height-map litres already on the detection stand:
            # refusing a cylinder fit says the object is not a cylinder, not
            # that it could not be measured.

    def _full_metric_ready(
        self, calibrated_depth: np.ndarray | None, intrinsics: CameraIntrinsics | None,
    ) -> bool:
        """Can the calibrated Logitech path actually produce a measurement?

        Every prerequisite it dereferences, checked together: a calibration
        that survived the resolution/ROI checks, intrinsics, the empty-scene
        monocular reference at this frame's shape, and a usable support plane.
        """
        if self.camera_id != "logitech":
            return True
        return bool(
            self.calibration is not None
            and self.calibration_rejected_reason is None
            and calibrated_depth is not None
            and intrinsics is not None
            and self.reference_monocular is not None
            and self.reference_monocular.shape == calibrated_depth.shape
            and self.reference_plane is not None
            and self.reference_plane.coefficients is not None
        )

    def _field_of_view_intrinsics(self, shape: tuple[int, ...]) -> CameraIntrinsics | None:
        """Intrinsics from the configured field of view, for a camera that sends none.

        The Raspberry Pi normally reports the C920's intrinsics with every
        frame. When it does not, an object still has to be measurable, so the
        configured horizontal field of view stands in -- and the result is
        reported as an uncalibrated estimate.
        """
        try:
            from .edge_client import camera_intrinsics_from_fov

            estimate = camera_intrinsics_from_fov(
                int(shape[1]), int(shape[0]),
                horizontal_fov_deg=self.config.logitech_horizontal_fov_deg,
            )
        except (ImportError, ValueError):
            return None
        return CameraIntrinsics(**estimate)

    def _fitted_logitech_calibration(self, shape: tuple[int, ...]) -> DepthCalibration | None:
        """The stored fit, if it still describes this camera, crop and model."""
        store = self.logitech_calibration
        if store is None or store.calibration is None:
            return None
        calibration = store.calibration
        self.calibration_rejected_reason = None
        current = self.latest_intrinsics
        if calibration.intrinsics is not None and current is not None:
            fx, fy, ppx, ppy = calibration.intrinsics
            moved = (
                abs(fx - current.fx) > 0.01 * max(fx, 1.0)
                or abs(fy - current.fy) > 0.01 * max(fy, 1.0)
                or abs(ppx - current.ppx) > 2.0 or abs(ppy - current.ppy) > 2.0
            )
            if moved:
                # New lens geometry: the plane fit and the empty reference both
                # belong to the old one. Ask for a fresh empty-scene calibration.
                self.calibration_rejected_reason = "camera_geometry_changed_recalibrate_empty_scene"
                return None
        if calibration.resolution is not None and tuple(calibration.resolution) != (shape[1], shape[0]):
            # Depth predictions are resampled to the frame, so a fit made at a
            # different resolution does not describe this crop.
            self.calibration_rejected_reason = "resolution_changed_recalibrate_empty_scene"
            return None
        if calibration.roi is not None and tuple(calibration.roi) != tuple(self.config.roi):
            # The fit belongs to the region it was measured over.
            self.calibration_rejected_reason = "roi_changed_recalibrate_empty_scene"
            return None
        return calibration

    def add_logitech_calibration_sample(self, known_distance_m: float) -> dict[str, Any]:
        """Record one flat reference at a tape-measured distance (calibration data only).

        The reference must fill the fixed bin region: the empty bin floor, or a
        flat board placed in it. Evaluation objects must never be used here.
        """
        if self.camera_id != "logitech" or self.logitech_calibration is None:
            raise ValueError("Metric depth calibration samples apply to the Logitech camera only")
        with self.lock:
            if not self._recent_monocular_frames or self.latest_frame is None:
                raise ValueError("No Logitech Depth Anything V2 prediction has been received yet")
            latest_shape = self._recent_monocular_frames[-1].shape
            frames = [item for item in self._recent_monocular_frames if item.shape == latest_shape]
            predicted = np.median(np.stack(frames), axis=0).astype(np.float32)
            region = fixed_bin_mask(self.latest_frame.shape, self.config.roi, self.config.bin_polygon)
            status = self.logitech_calibration.add_sample(
                predicted, region, float(known_distance_m), depth_output_kind(self.config.depth_model),
            )
        status["apply"] = "Capture the empty baseline again to apply this calibration"
        return status

    def logitech_diagnostics(self) -> dict[str, Any]:
        """The operator-facing yes/no chain for Logitech + Depth Anything V2."""
        analysis = self.latest_analysis
        debug = self.logitech_mask_debug
        status = self._volume_status({}) if analysis is not None else {"message": "Waiting for camera frames"}
        reasons = list(debug.get("reasons") or [])
        detections = [] if analysis is None else analysis.detections
        rejection = next((item.volume_rejection_reason for item in detections if item.volume_rejection_reason), None)
        latency = None if analysis is None else round(float(analysis.inference_ms), 1)
        stream_age = None if analysis is None else max(0.0, time.time() - float(analysis.timestamp or 0))
        return {
            "camera_connected": analysis is not None and stream_age is not None and stream_age < 10.0,
            "detector_ready": self.detector is not None,
            "da_v2_model": self.config.depth_model if self.depth_estimator is not None else None,
            "da_v2_device": getattr(self.depth_estimator, "device", None),
            "da_v2_error": self.depth_load_error,
            "da_v2_loaded": bool(self.depth_estimator is not None and self._recent_monocular_frames),
            "calibration_loaded": bool(self._logitech_measurement_ready() and self.calibration is not None),
            "final_mask_valid": bool(debug.get("final_mask_valid")),
            "metric_volume_status": status.get("message"),
            "rejection_reason": rejection or (reasons[0] if reasons else None),
            "latency_ms": latency,
            "processing_fps": None if not latency else round(1000.0 / latency, 2),
            "volume_trace": self.last_logitech_volume_diagnostics,
        }

    def stage_report(self) -> dict[str, Any]:
        """Counters plus the single most likely reason nothing is reaching history."""
        counters = dict(self.stage_counters)
        if not counters.get("frames_processed"):
            reason = "no frames reached inference for this camera"
        elif not counters.get("raw_detections"):
            reason = "detector returned no predictions (check prompts / confidence)"
        elif not counters.get("after_class_confidence_roi_area"):
            top = max(self.last_stage_rejections.items(), key=lambda item: item[1], default=(None, 0))[0]
            reason = f"every prediction removed by application filters (last frame mostly: {top})"
        elif self.camera_id == "logitech" and not counters.get("valid_masks"):
            reason = "every mask rejected by the Logitech mask gate"
        elif not counters.get("confirmed_tracks_total"):
            reason = "masks exist but no track has been confirmed yet"
        elif not counters.get("finalised_measurements"):
            reason = "tracks exist; waiting for a stable volume (or metric calibration)"
        else:
            reason = None
        return {
            "counters": counters,
            "last_frame_rejections": self.last_stage_rejections,
            "blocking_reason": reason,
            "detector": {
                "model": self.config.detector_model,
                "device": getattr(self.detector, "device", None),
                "prompts": len(self.config.prompts),
                "image_size": self.config.image_size,
                "confidence": (self.config.logitech_detector_confidence if self.camera_id == "logitech"
                               else self.config.detector_confidence),
            },
        }

    def diagnose_detector_frame(self) -> dict[str, Any]:
        """Run the detector once on the latest frame, before and after every filter, and save it."""
        import cv2

        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            raise ValueError(f"No {self.camera_id} frame has been received yet")
        with self.inference_lock:
            raw = self.detector.detect_batch([frame])[0]
        region = fixed_bin_mask(frame.shape, self.config.roi, self.config.bin_polygon)
        verdicts = []
        for item in raw:
            single: Counter = Counter()
            kept = filter_waste_detections(
                [Detection(item.label, item.confidence, item.box, item.mask, source=item.source)],
                frame.shape, region, self.config, rejections=single,
                min_pixels=self.config.logitech_min_object_pixels if self.camera_id == "logitech" else None,
                confidence=self.config.logitech_detector_confidence if self.camera_id == "logitech" else None,
            )
            verdicts.append({
                "label": item.label, "confidence": round(float(item.confidence), 4), "box": list(item.box),
                "mask_pixels": int(item.area_pixels), "passed_filters": bool(kept),
                "rejected_by": next(iter(single), None),
            })
        accepted = filter_waste_detections(
            list(raw), frame.shape, region, self.config,
            min_pixels=self.config.logitech_min_object_pixels if self.camera_id == "logitech" else None,
            confidence=self.config.logitech_detector_confidence if self.camera_id == "logitech" else None,
        )
        debug: dict[str, Any] = {}
        final = accepted
        if self.camera_id == "logitech":
            final, _ = bound_logitech_detections(
                frame, self.reference_rgb, accepted, region,
                min_pixels=self.config.logitech_min_object_pixels,
                foreground_threshold=self.config.foreground_threshold,
                max_scene_fraction=self.config.logitech_max_scene_fraction,
                max_expansion=self.config.logitech_max_mask_expansion,
                duplicate_overlap=self.config.logitech_duplicate_overlap, debug=debug,
            )
        raw_view = frame.copy()
        for item, verdict in zip(raw, verdicts):
            colour = (0, 200, 0) if verdict["passed_filters"] else (0, 0, 255)
            if item.mask is not None and item.mask.shape == frame.shape[:2]:
                raw_view[item.mask] = (0.5 * raw_view[item.mask] + 0.5 * np.array(colour)).astype(np.uint8)
            x1, y1, x2, y2 = item.box
            cv2.rectangle(raw_view, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(raw_view, f"{item.label} {item.confidence:.2f} {verdict['rejected_by'] or 'ok'}",
                        (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        final_view = frame.copy()
        for item in final:
            final_view[item.mask] = (0.5 * final_view[item.mask] + np.array((0, 110, 0))).clip(0, 255).astype(np.uint8)
        directory = self.config.results_dir / "hardware_diagnostics" / self.camera_id / \
            (time.strftime("%Y%m%d-%H%M%S") + "_detector")
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "original.png"), frame)
        cv2.imwrite(str(directory / "raw_predictions.png"), raw_view)
        cv2.imwrite(str(directory / "final_masks.png"), final_view)
        depth = self.predict_depth(frame) if self.camera_id == "logitech" else None
        if depth is not None:
            from .diagnostics import _depth_image

            cv2.imwrite(str(directory / "relative_depth.png"), _depth_image(depth))
        summary = {
            "camera": self.camera_id, "frame_shape": list(frame.shape), "raw_predictions": verdicts,
            "after_filters": len(accepted), "final_masks": len(final),
            "detector_only_masks": int(debug.get("detector_only_masks", 0)),
            "mask_gate_reasons": debug.get("reasons"), "has_empty_reference": self.reference_rgb is not None,
            "stage_report": self.stage_report(), "saved_to": str(directory),
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def logitech_mask_overlay(self) -> np.ndarray | None:
        """Detector (blue), foreground (yellow), final object (green), rejected (red)."""
        debug = self.logitech_mask_debug
        frame = debug.get("frame")
        if frame is None:
            return None
        output = frame.copy()
        for key, colour in (("detector", (255, 120, 0)), ("foreground", (0, 220, 255)),
                            ("rejected", (0, 0, 255)), ("final", (0, 220, 0))):
            mask = debug.get(key)
            if mask is not None and mask.shape == output.shape[:2] and mask.any():
                tinted = output.copy()
                tinted[mask] = colour
                output = (0.55 * output + 0.45 * tinted).astype(np.uint8)
        return output

    def latest_raw_volume(self) -> tuple[float | None, str | None, str | None]:
        """The newest raw Logitech volume, its object name and calibration group."""
        analysis = self.latest_analysis
        if analysis is None:
            return None, None, None
        for detection in sorted(analysis.detections, key=lambda item: item.area_pixels, reverse=True):
            if detection.raw_volume_l:
                group = geometry_group(
                    None if detection.shape_geometry is None else detection.shape_geometry.geometry_method,
                    detection.canonical_type or detection.label,
                )
                return float(detection.raw_volume_l), detection.canonical_type or detection.label, group
        return None, None, None

    def logitech_calibration_status(self) -> dict[str, Any]:
        store = self.logitech_calibration
        active = self.calibration if self.camera_id == "logitech" else None
        ready = self._logitech_measurement_ready()
        return {
            "metric_ready": bool(ready and active is not None),
            "message": None if ready and active is not None else RELATIVE_ONLY_MESSAGE,
            "calibration_mode": self.calibration_mode,
            "active_calibration": None if active is None else active.to_dict(),
            "depth_output": depth_output_kind(self.config.depth_model),
            "calibration_rejected_reason": self.calibration_rejected_reason,
            "volume_factors": None if self.volume_factors is None else self.volume_factors.status(),
            "stored": None if store is None else store.status(),
            "diagnostics": self.logitech_diagnostics() if self.camera_id == "logitech" else None,
            "lens": None if self.logitech_lens is None else {
                "status": self.logitech_lens.status, "profile": self.logitech_lens.source,
            },
        }

    def _logitech_measurement_ready(self) -> bool:
        if self.camera_id != "logitech":
            return True
        if (
            self.config.logitech_require_reference
            and self.calibration_mode not in (
                "independent-measured-distance", "uncalibrated-estimate", "reference-distance-estimate")
            and not self.config.logitech_allow_provisional_metric
        ):
            return False
        return not self._logitech_tilt_invalid()

    def _volume_status(self, hardware: dict[str, Any]) -> dict[str, Any]:
        if self.latest_analysis is None:
            return {"ready": False, "code": "waiting_for_camera", "message": "Waiting for camera frames"}
        if self.camera_id == "logitech":
            if self.depth_estimator is None:
                return {
                    "ready": True,
                    "code": "local_rgb_only",
                    "message": "Depth Anything V2 is not loaded, so Logitech litres are unavailable: " + (
                        self.depth_load_error or "enable LOCALLIFE_ENABLE_DEPTH"
                    ),
                }
            if self.latest_monocular_depth is None:
                return {"ready": False, "code": "missing_monocular_depth", "message": "Waiting for Logitech RGB and Depth Anything inference"}
            if self.latest_intrinsics is None:
                return {"ready": False, "code": "missing_intrinsics", "message": "Logitech lens calibration or field of view is missing"}
            if self.baseline_monocular is None:
                return {
                    "ready": self.latest_monocular_depth is not None,
                    "code": "uncalibrated_estimate" if self.latest_monocular_depth is not None
                    else "missing_empty_baseline",
                    "message": (
                        ("REFERENCE-DISTANCE ESTIMATE — measured camera height, no empty baseline yet"
                         if self.calibration_mode == "reference-distance-estimate"
                         else "UNCALIBRATED ESTIMATE — capture the empty Logitech baseline for a calibrated result")
                        if self.latest_monocular_depth is not None
                        else "Clear the Logitech view; automatic empty-scene setup is running"
                    ),
                }
            if (
                self.config.logitech_require_reference
                and self.calibration_mode != "independent-measured-distance"
                and not self.config.logitech_allow_provisional_metric
            ):
                return {
                    "ready": False,
                    "code": "missing_reference_distance",
                    "message": (
                        "Tracked — metric calibration required"
                        if self.latest_analysis is not None and self.latest_analysis.detections
                        else RELATIVE_ONLY_MESSAGE + " (use Calibrate Empty Logitech Scene)"
                    ),
                }
            if self._logitech_tilt_invalid():
                return {
                    "ready": False,
                    "code": "excessive_camera_tilt",
                    "message": (
                        f"Logitech camera tilt is {self.reference_plane.tilt_degrees:.1f}°; "
                        f"mount it above the bin or restrict its region to the bin floor "
                        f"(hard maximum {self.config.logitech_hard_max_tilt_degrees:.1f}°)"
                    ),
                }
            if not self.latest_analysis.detections:
                return {"ready": True, "code": "waiting_for_object", "message": "Logitech baseline is ready; waiting for a bag or box"}
            if self.latest_analysis.monocular_total is None:
                return {"ready": False, "code": "no_valid_object_height", "message": "The Logitech depth model has not found measurable object height"}
            if self.calibration_mode == "model-metric-unverified":
                return {"ready": True, "code": "measuring_unverified", "message": "Estimated liters; enter a measured reference distance for thesis-grade calibration"}
            return {"ready": True, "code": "measuring", "message": "Independently calibrated Logitech monocular volume measurement is active"}
        if not hardware["available"]:
            return {
                "ready": False,
                "code": "no_realsense_depth",
                "message": "No RealSense depth is arriving; start the Pi with --source realsense",
            }
        if hardware["valid_pixels"] == 0:
            return {
                "ready": False,
                "code": "invalid_realsense_depth",
                "message": "The RealSense depth frame contains no valid distances",
            }
        if self.latest_intrinsics is None:
            return {
                "ready": False,
                "code": "missing_intrinsics",
                "message": "The Raspberry Pi did not send RealSense camera intrinsics",
            }
        if self.baseline_rgb is None:
            return {
                "ready": False,
                "code": "missing_empty_baseline",
                "message": "Clear the measurement area; automatic empty-scene setup is running",
            }
        if self.baseline_realsense is None:
            return {
                "ready": False,
                "code": "baseline_missing_depth",
                "message": "Recapture the empty baseline while RealSense depth is connected",
            }
        if not self.latest_analysis.detections:
            return {
                "ready": True,
                "code": "waiting_for_object",
                "message": "Depth and empty-bin reference are ready; deposit a garbage bag",
            }
        if self.latest_analysis.realsense_total is None:
            return {
                "ready": False,
                "code": "no_valid_object_height",
                "message": "No object height above baseline; recapture a genuinely empty scene",
            }
        if self.latest_analysis.realsense_total.coverage_ratio < self.config.minimum_depth_coverage:
            return {
                "ready": False,
                "code": "insufficient_depth_coverage",
                "message": (
                    f"RealSense sees only {self.latest_analysis.realsense_total.coverage_ratio * 100:.0f}% "
                    "of this object; improve depth coverage before recording its volume"
                ),
            }
        return {"ready": True, "code": "measuring", "message": "RealSense volume measurement is active"}
