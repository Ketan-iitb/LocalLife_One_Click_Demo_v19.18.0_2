"""Foreground validation: what may become a mesh, and what may not.

Dark floor, shadow, furniture and bin walls were being meshed and reported as
4-6 L of waste. The temptation is to raise the detector's confidence threshold
globally, and that is the wrong fix: a black bin bag in poor light is exactly
the case a higher threshold removes first. The detector is not confused about
*whether* something is there -- it is right that pixels changed. What is missing
is the physical question: does this region stand above the calibrated support
plane by an amount a real object would?

So gating happens on geometry, before any mesh is built, and every rejection
carries a reason an operator can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Reasons, as reported in the CSV's `reason` column and on the live card.
OUTSIDE_ROI = "outside_measurement_roi"
BACKGROUND = "background_or_support_plane"
INSUFFICIENT_DEPTH = "insufficient_depth_difference"
LOW_VALID_DEPTH = "low_valid_depth"
UNSTABLE_MASK = "unstable_mask"
IMPLAUSIBLE = "implausible_dimensions"
INSUFFICIENT_NEW_VOLUME = "insufficient_new_volume"
CALIBRATION_CHANGED = "camera_or_baseline_changed"
TOO_SMALL = "component_too_small"
BACKGROUND_SCALE = "background_scale_mask"

REJECTION_REASONS = (
    OUTSIDE_ROI, BACKGROUND, INSUFFICIENT_DEPTH, LOW_VALID_DEPTH, UNSTABLE_MASK,
    IMPLAUSIBLE, INSUFFICIENT_NEW_VOLUME, CALIBRATION_CHANGED, TOO_SMALL,
    BACKGROUND_SCALE,
)


@dataclass
class ForegroundSettings:
    """Physical thresholds, sized from the bin and the sensor rather than taste.

    `min_height_m` is the smallest rise that a real deposit makes above the
    support plane; below it, depth noise and a slightly-off plane fit are
    indistinguishable from an object. 2 cm sits above the D435's noise at bin
    distance while still admitting a flattened bag.

    `max_area_fraction` catches the opposite failure: a mask covering most of
    the frame is the floor or a lighting change, never one deposit.
    """

    min_height_m: float = 0.02
    min_area_px: int = 900
    max_area_fraction: float = 0.60
    min_valid_depth_fraction: float = 0.35
    max_dimension_m: float = 2.0
    max_aspect_ratio: float = 12.0
    min_new_volume_l: float = 0.15


@dataclass
class GateResult:
    accepted: bool
    reason: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason, "detail": self.detail or {}}


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if not rows.any() or not columns.any():
        return None
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(columns)[0][[0, -1]]
    return int(left), int(top), int(right), int(bottom)


def validate_foreground(
    mask: np.ndarray,
    depth_m: np.ndarray | None,
    plane_depth_m: np.ndarray | None,
    *,
    roi_mask: np.ndarray | None = None,
    settings: ForegroundSettings | None = None,
    calibration_valid: bool = True,
) -> GateResult:
    """Decide whether this component may be measured at all.

    `plane_depth_m` is the calibrated support plane (or empty-bin baseline)
    sampled at every pixel. The decisive test is the *height above it*: a region
    that is level with the plane is the plane, however dark it looks and however
    confident the detector is.
    """
    options = settings or ForegroundSettings()

    if not calibration_valid:
        # A stale plane applied to a moved camera turns background into metres
        # of phantom height. Refuse rather than measure against it.
        return GateResult(False, CALIBRATION_CHANGED)

    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return GateResult(False, TOO_SMALL, {"area_px": 0})
    if area < options.min_area_px:
        return GateResult(False, TOO_SMALL, {"area_px": area})
    if area > mask.size * options.max_area_fraction:
        return GateResult(
            False, BACKGROUND_SCALE,
            {"area_fraction": round(area / mask.size, 3)},
        )

    if roi_mask is not None:
        inside = np.asarray(roi_mask, dtype=bool)
        overlap = int((mask & inside).sum())
        # Mostly-outside means the component belongs to the room, not the bin.
        if overlap < area * 0.5:
            return GateResult(
                False, OUTSIDE_ROI, {"inside_fraction": round(overlap / area, 3)},
            )
        mask = mask & inside
        area = int(mask.sum())
        if area < options.min_area_px:
            return GateResult(False, TOO_SMALL, {"area_px": area})

    if depth_m is None or plane_depth_m is None:
        # No depth means no physical check is possible. Colour and class alone
        # must never create an object -- that is how shadows became waste.
        return GateResult(False, LOW_VALID_DEPTH, {"reason": "no depth available"})

    depth = np.asarray(depth_m, dtype=np.float64)
    plane = np.asarray(plane_depth_m, dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(plane) & (plane > 0)
    valid_fraction = float(valid.sum()) / max(1, area)
    if valid_fraction < options.min_valid_depth_fraction:
        return GateResult(
            False, LOW_VALID_DEPTH, {"valid_depth_fraction": round(valid_fraction, 3)},
        )

    # Height above the plane: the plane is further from the camera, so a real
    # object reads as a smaller depth.
    heights = plane[valid] - depth[valid]
    # The 75th percentile, not the max: one speckle of noise should not qualify
    # a flat floor, and not the mean either, which a large flat region dilutes.
    height = float(np.percentile(heights, 75))
    if height < options.min_height_m:
        reason = BACKGROUND if height <= options.min_height_m * 0.25 else INSUFFICIENT_DEPTH
        return GateResult(False, reason, {"height_m": round(height, 4)})

    bounds = _mask_bounds(mask)
    if bounds is None:
        return GateResult(False, TOO_SMALL)
    left, top, right, bottom = bounds
    width_px, height_px = max(1, right - left), max(1, bottom - top)
    aspect = max(width_px / height_px, height_px / width_px)
    if aspect > options.max_aspect_ratio:
        # A long thin sliver is a bin edge, a cable or a shadow line.
        return GateResult(False, IMPLAUSIBLE, {"aspect_ratio": round(aspect, 2)})
    if height > options.max_dimension_m:
        return GateResult(False, IMPLAUSIBLE, {"height_m": round(height, 3)})

    return GateResult(True, None, {
        "height_m": round(height, 4),
        "area_px": area,
        "valid_depth_fraction": round(valid_fraction, 3),
    })


def has_new_volume(added_litres: float | None, settings: ForegroundSettings | None = None) -> GateResult:
    """Is the incremental change against the committed scene a real deposit?"""
    options = settings or ForegroundSettings()
    if added_litres is None:
        return GateResult(False, INSUFFICIENT_NEW_VOLUME, {"added_l": None})
    if added_litres < options.min_new_volume_l:
        return GateResult(False, INSUFFICIENT_NEW_VOLUME, {"added_l": round(added_litres, 4)})
    return GateResult(True, None, {"added_l": round(added_litres, 4)})
