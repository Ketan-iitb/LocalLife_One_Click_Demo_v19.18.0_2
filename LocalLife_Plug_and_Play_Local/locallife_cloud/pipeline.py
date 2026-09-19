"""End-to-end detection, tracking, calibration, and volume measurement."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from typing import Any

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
from .logitech import bound_logitech_detections, stabilize_background_depth
from .sorting_rules import classify_sorting, mis_sort_family
from .storage import ResultStore
from .tracking import ObjectTracker
from .box_templates import load_box_templates, match_box_template
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
    reference_plane_is_usable,
    recover_elevated_object_mask,
    synthesize_plane_depth,
)

LOGGER = logging.getLogger(__name__)


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
        min(config.min_component_pixels, max(1, int(region_area * 0.10))),
    )
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
            continue
        if config.operating_mode == "waste" and config.bag_only and not is_bag_detection(detection.label):
            continue
        if config.operating_mode == "waste" and not config.bag_only and not is_supported_waste_detection(detection.label):
            continue
        if detection.source.startswith("yolo") and detection.confidence < config.detector_confidence:
            continue
        x1, y1, x2, y2 = detection.box
        width, height = max(0, x2 - x1), max(0, y2 - y1)
        if width < frame_width * config.min_detection_side_fraction:
            continue
        if height < frame_height * config.min_detection_side_fraction:
            continue
        center_x = min(frame_width - 1, max(0, (x1 + x2) // 2))
        center_y = min(frame_height - 1, max(0, (y1 + y2) // 2))
        if not region[center_y, center_x]:
            continue
        mask = combined_mask([detection], (frame_height, frame_width)) & region
        area = int(np.count_nonzero(mask))
        if area < minimum_area or area / region_area > config.max_detection_area_fraction:
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
                    if self.config.logitech_reference_distance_m > 0:
                        region = fixed_bin_mask(image.shape, self.config.roi, self.config.bin_polygon)
                        values = predicted[region & np.isfinite(predicted) & (predicted > 0.10)]
                        if not values.size:
                            raise ValueError("The Logitech depth model returned no valid baseline pixels")
                        scale = self.config.logitech_reference_distance_m / float(np.median(values))
                        self.calibration = DepthCalibration(
                            scale=scale, offset_m=0.0, rmse_m=0.0, sample_pixels=int(values.size)
                        )
                        self.calibration_mode = "independent-measured-distance"
                    else:
                        self.calibration = DepthCalibration(
                            scale=1.0, offset_m=0.0, rmse_m=0.0, sample_pixels=int(predicted.size)
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
            self._occupied_logitech_mask = (
                np.zeros(image.shape[:2], dtype=bool) if self.camera_id == "logitech" else None
            )
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
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
        if depth_m is not None and depth_m.shape != frame.shape[:2]:
            warnings.append("RealSense depth is not aligned to the RGB frame; hardware volume was skipped")
            depth_m = None
        self._validate_saved_baseline(frame, warnings)
        calibrated_prediction = (
            self.calibration.apply(predicted_depth)
            if predicted_depth is not None and self.calibration is not None
            else predicted_depth
        )
        bin_region = fixed_bin_mask(frame.shape, self.config.roi, self.config.bin_polygon)
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
        detections = filter_waste_detections(detections, frame.shape, bin_region, self.config)
        for family in mis_sorted:
            warnings.append(f"MIS-SORT: a {family} object was detected; this bin does not accept it")

        if self.camera_id == "logitech":
            detections, segmentation_warnings = bound_logitech_detections(
                frame,
                self.reference_rgb,
                detections,
                bin_region,
                min_pixels=self.config.min_component_pixels,
                foreground_threshold=self.config.foreground_threshold,
                max_scene_fraction=self.config.logitech_max_scene_fraction,
                max_expansion=self.config.logitech_max_mask_expansion,
                duplicate_overlap=self.config.logitech_duplicate_overlap,
            )
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
                detection.footprint_length_mm = round(dimensions.length_mm, 2)
                detection.footprint_width_mm = round(dimensions.width_mm, 2)
                detection.physical_height_mm = round(dimensions.height_mm, 2)
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
            if self.calibration is not None and logitech_ready and logitech_height_coherent:
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
                        detection.measurement_quality = individual_mono.quality
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
        if self._previous_bin_total_l is not None:
            for track_id in new_ids:
                self._bin_total_before_track[track_id] = self._previous_bin_total_l

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
                if not stable:
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
                    self._record_added_volume(detection, bin_total)
                    self.ledger.deposit(detection, timestamp=timestamp)
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
        if newly_deposited and len(newly_deposited) == len(detections):
            self._remember_occupied_objects(newly_deposited)
            self._advance_reference()
            self.committed_bags = self.ledger.summary()["deposited_bags"]
            self.tracker.tracks.clear()
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
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
            self._remember_occupied_objects(self.latest_analysis.detections)
            self._advance_reference()
            self.committed_bags = self.ledger.summary()["deposited_bags"]
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
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
                return "unavailable-offline-local-rgb-only"
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
                calibration = DepthCalibration(
                    scale=float(calibration_payload["scale"]),
                    offset_m=float(calibration_payload["offset_m"]),
                    rmse_m=float(calibration_payload["rmse_m"]),
                    sample_pixels=int(calibration_payload["sample_pixels"]),
                )
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
            self._volume_history.clear()
            self._box_measurement_history.clear()
            self._box_frames_considered.clear()
            return record

    def state(self) -> dict[str, Any]:
        with self.lock:
            hardware = summarize_depth_signal(self.latest_depth)
            monocular = summarize_depth_signal(self.latest_monocular_depth)
            volume_status = self._volume_status(hardware)
            return {
                "camera_id": self.camera_id,
                "camera_name": "Intel RealSense D435" if self.camera_id == "realsense" else "Logitech C920",
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
                "waste_ledger_enabled": self.config.operating_mode == "waste",
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

    def _record_added_volume(
        self, detection: Detection, bin_total: VolumeMeasurement | None,
    ) -> None:
        """Attach this deposit's incremental occupied volume (playbook sections 5, 10).

        The difference between total bin occupancy before this object arrived
        and after it settled. Purely additive: it records what the bin gained
        without influencing whether the deposit happens or what the object's own
        measured volume is. Both totals have to be real measurements -- an
        absent one leaves the fields None rather than inventing a difference.
        """
        before = self._bin_total_before_track.pop(detection.track_id, None)
        if before is None or bin_total is None:
            return
        after = float(bin_total.liters)
        detection.volume_before_l = round(before, 6)
        detection.volume_after_l = round(after, 6)
        # Section 16: a small negative difference is sensor noise around an
        # unchanged scene, not a bin that shrank, so it reports as zero.
        detection.added_volume_l = round(max(0.0, after - before), 6)

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

    def _logitech_measurement_ready(self) -> bool:
        if self.camera_id != "logitech":
            return True
        if (
            self.config.logitech_require_reference
            and self.calibration_mode != "independent-measured-distance"
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
                    "message": "Local Logitech tracking and colour are ready; independent Logitech litres are unavailable offline",
                }
            if self.latest_monocular_depth is None:
                return {"ready": False, "code": "missing_monocular_depth", "message": "Waiting for Logitech RGB and Depth Anything inference"}
            if self.latest_intrinsics is None:
                return {"ready": False, "code": "missing_intrinsics", "message": "Logitech lens calibration or field of view is missing"}
            if self.baseline_monocular is None:
                return {"ready": False, "code": "missing_empty_baseline", "message": "Clear the Logitech view; automatic empty-scene setup is running"}
            if (
                self.config.logitech_require_reference
                and self.calibration_mode != "independent-measured-distance"
                and not self.config.logitech_allow_provisional_metric
            ):
                return {
                    "ready": False,
                    "code": "missing_reference_distance",
                    "message": "Waiting for an automatically learned metric reference; Logitech liters remain withheld",
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
