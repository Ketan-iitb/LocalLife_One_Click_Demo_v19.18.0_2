"""End-to-end orchestration of the dual-camera recipe -- v3 blueprint
("Zero-Flaw Implementation Blueprint"), superseding the v1/v2-era text
recipe's Steps 2-6 this module originally implemented.

Capture/synchronization (v3 §3) happens upstream, on the Raspberry Pi -- by
the time `process_object()` is called, the aligned RealSense color+depth
(optionally from two views for the box multi-view accuracy target) and the
Logitech color frame are already in hand.

Camera responsibilities follow v3 §8's fusion rule exactly, with no
cross-sensor value mixing: **volume <- RealSense, color <- Logitech,
material <- Logitech.** If Logitech did not itself detect the object (only
RealSense did), RealSense's own crop of its own color frame is used for
color/material instead, so an object visible to only one camera still gets
*a* color/material answer rather than "Other" by default -- everywhere else,
Logitech's crop is used exclusively for color and material.

Output matches v3 §9's exact JSON schema:
`volume_liters, volume_tolerance_liters, volume_confidence, color,
color_confidence, material, material_confidence, material_model,
object_type, views_used, timestamp, flags`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .pointcloud_volume import estimate_volume_recipe
from .recipe_calibration import CalibrationMap
from .recipe_color import classify_dominant_color
from .recipe_config import RecipeConfig
from .recipe_detect import RecipeDetection, RecipeDetector, frame_edge_clip_fraction, select_best
from .recipe_material import RecipeMaterialClassifier, RecipeMaterialFallbackClassifier, classify_material_cascade
from .types import CameraIntrinsics

# v3 edge-case matrix: "Object touches frame edge -> pad ROI; if mask
# clipped >30%, lower confidence." frame_edge_clip_fraction() returns the
# fraction of a box's 4 sides flush with the frame boundary (0, 0.25, 0.5,
# ...); >30% of *that* signal (i.e. at least 2 of 4 sides touching) is
# treated as "clipped enough to flag".
_FRAME_EDGE_CLIP_THRESHOLD = 0.30
_VOLUME_LOW_CONFIDENCE_THRESHOLD = 0.7
_OUTLIER_REMOVAL_HIGH_FRACTION = 0.10


def process_object(
    realsense_color_bgr: np.ndarray,
    realsense_depth_m: np.ndarray,
    realsense_intrinsics: CameraIntrinsics,
    logitech_color_bgr: np.ndarray | None,
    *,
    detector: RecipeDetector,
    material_classifier: RecipeMaterialClassifier,
    config: RecipeConfig | None = None,
    fallback_material_classifier: RecipeMaterialFallbackClassifier | None = None,
    volume_calibration: CalibrationMap | None = None,
    color_calibration: CalibrationMap | None = None,
    material_calibration: CalibrationMap | None = None,
    realsense_baseline_depth_m: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    realsense_depth_m_view2: np.ndarray | None = None,
    realsense_intrinsics_view2: CameraIntrinsics | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Run the recipe's full v3 pipeline and return its exact JSON schema.

    `detector` and `material_classifier` are passed in (rather than created
    here) so a caller -- the FastAPI service in `recipe_api.py`, or the
    existing dashboard pipeline wiring these in as an additional result --
    loads each heavy model once and reuses it across calls instead of paying
    model-load cost per object. `config` supplies every v3 §10 tuning knob
    (defaults to `RecipeConfig.default()` if not given).
    `realsense_depth_m_view2`/`realsense_intrinsics_view2`, if both given,
    are a second RealSense capture of the same box from a ~90-degree
    rotated view (v3 §3.4) -- the box path's accuracy target.
    `*_calibration`, if given, are v3 §8's fitted logistic calibration maps;
    omitted (the default in every real deployment right now -- see
    `recipe_calibration.py`'s own docstring) means every confidence in the
    result is raw/uncalibrated, and the result carries the
    `uncalibrated_confidences` flag accordingly.
    """
    config = config or RecipeConfig.default()
    conf_threshold = confidence_threshold if confidence_threshold is not None else config.segmentation.conf_threshold

    realsense_detections = detector.detect(realsense_color_bgr, confidence_threshold=conf_threshold)
    logitech_detections = (
        detector.detect(logitech_color_bgr, confidence_threshold=conf_threshold)
        if logitech_color_bgr is not None
        else []
    )

    realsense_frame_area = int(realsense_color_bgr.shape[0] * realsense_color_bgr.shape[1])
    select_kwargs = dict(
        target_class=config.segmentation.target_class,
        area_range=config.segmentation.area_range,
    )

    overall_best = select_best(
        realsense_detections, logitech_detections, frame_area=realsense_frame_area, **select_kwargs
    )
    if overall_best is None:
        return _empty_result()

    realsense_best = select_best(realsense_detections, frame_area=realsense_frame_area, **select_kwargs)
    logitech_best = select_best(
        logitech_detections,
        frame_area=int(logitech_color_bgr.shape[0] * logitech_color_bgr.shape[1]) if logitech_color_bgr is not None else None,
        **select_kwargs,
    )

    object_type = (realsense_best or overall_best).object_type

    flags: set[str] = set()

    if realsense_best is not None:
        volume_result = estimate_volume_recipe(
            realsense_depth_m,
            realsense_intrinsics,
            realsense_best.mask,
            object_type=object_type,
            baseline_depth_m=realsense_baseline_depth_m,
            roi_mask=roi_mask,
            mask_erode_px=config.depth.mask_erode_px,
            valid_range_m=config.depth.valid_range_m,
            depth_m_view2=realsense_depth_m_view2,
            intrinsics_view2=realsense_intrinsics_view2,
            mask_view2=realsense_best.mask if realsense_depth_m_view2 is not None else None,
        )
        volume_liters = volume_result.liters
        volume_tolerance_liters = volume_result.tolerance_liters
        volume_confidence = volume_result.confidence
        views_used = volume_result.views_used
        flags.update(volume_result.flags)

        clip_fraction = frame_edge_clip_fraction(realsense_best.box, realsense_color_bgr.shape[:2])
        if clip_fraction > _FRAME_EDGE_CLIP_THRESHOLD:
            flags.add("frame_edge_clipped")
            volume_confidence *= 0.7
    else:
        # RealSense never itself detected the object (only Logitech did) --
        # per v3's fusion rule, volume is RealSense-only, so there is
        # nothing valid to measure it from. Reporting 0.0/0.0 (rather than
        # silently reusing Logitech's own depth/mask) keeps the contract
        # honest: a caller can tell "measured as zero" apart from "not
        # measurable".
        volume_liters = 0.0
        volume_tolerance_liters = 0.0
        volume_confidence = 0.0
        views_used = 1
        flags.add("no_realsense_detection")

    if volume_confidence < _VOLUME_LOW_CONFIDENCE_THRESHOLD:
        flags.add("low_volume_conf")

    color_source_frame, color_source_detection = (
        (logitech_color_bgr, logitech_best) if logitech_best is not None
        else (realsense_color_bgr, realsense_best)
    )
    if color_source_detection is not None:
        color_result = classify_dominant_color(
            color_source_frame,
            color_source_detection.mask,
            kmeans_k=config.color.kmeans_k,
            center_crop_frac=config.color.center_crop_frac,
        )
        color_label = color_result.label
        color_confidence = color_result.confidence
    else:
        color_label = "Other"
        color_confidence = 0.0

    if color_source_detection is not None:
        material_label, material_confidence, material_model, ambiguous_material = classify_material_cascade(
            color_source_frame,
            color_source_detection.mask,
            color_source_detection.box,
            clip_classifier=material_classifier,
            fallback_classifier=fallback_material_classifier,
            clip_confidence_floor=config.material.clip_confidence_floor,
            fallback_confidence_floor=config.material.fallback_confidence_floor,
            plastic_thin_margin=config.material.plastic_thin_margin,
        )
        if ambiguous_material:
            flags.add("ambiguous_material")
    else:
        material_label, material_confidence, material_model = "Other", 0.0, "clip"

    volume_calibration = volume_calibration or CalibrationMap()
    color_calibration = color_calibration or CalibrationMap()
    material_calibration = material_calibration or CalibrationMap()
    if not (volume_calibration.fitted and color_calibration.fitted and material_calibration.fitted):
        flags.add("uncalibrated_confidences")

    volume_confidence = volume_calibration.apply(volume_confidence)
    color_confidence = color_calibration.apply(color_confidence)
    material_confidence = material_calibration.apply(material_confidence)

    return {
        "volume_liters": round(float(volume_liters), 2),
        "volume_tolerance_liters": round(float(volume_tolerance_liters), 2),
        "volume_confidence": round(float(volume_confidence), 2),
        "color": color_label,
        "color_confidence": round(float(color_confidence), 2),
        "material": material_label,
        "material_confidence": round(float(material_confidence), 2),
        "material_model": material_model,
        "object_type": object_type,
        "views_used": int(views_used),
        "timestamp": _timestamp(),
        "flags": sorted(flags),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "volume_liters": 0.0,
        "volume_tolerance_liters": 0.0,
        "volume_confidence": 0.0,
        "color": "Other",
        "color_confidence": 0.0,
        "material": "Other",
        "material_confidence": 0.0,
        "material_model": "clip",
        "object_type": "unknown",
        "views_used": 1,
        "timestamp": _timestamp(),
        "flags": ["no_object_found"],
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
