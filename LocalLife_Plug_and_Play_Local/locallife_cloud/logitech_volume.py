"""Metric height map and volume for the Logitech + Depth Anything V2 camera.

Depth Anything V2 is monocular: its depth is affine-related to metres at best,
so it is aligned to the *empty support plane* whose geometry is known from the
fixed installation (camera intrinsics and one tape-measured camera height).

    Z_metric(u,v) ~= a * Z_pred(u,v) + b        fitted on the empty ROI, where
    Z_plane(u,v)   = height / cos(theta(u,v))   is the distance along each ray
                                                to a plane perpendicular to the
                                                optical axis at that height.

Object volume is then the masked integral of the filtered height above the
fitted plane, with each pixel's own metric footprint:

    V = sum( height(u,v) * Z(u,v)^2 / (fx * fy) )

Heights are clipped at zero, spike-filtered (median + MAD) and capped, because
monocular depth is smooth but locally wrong, and a handful of spikes used to
dominate the integral. Every intermediate statistic is returned so one object
can be traced end to end.

RealSense keeps its own, unchanged volume path: nothing here is used for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import CameraIntrinsics, DepthCalibration, VolumeMeasurement
from .volume import ReferencePlane, _plane_perpendicular_height

# The plane fit needs a real sample of the ROI, not a handful of pixels.
MIN_PLANE_FRACTION = 0.25
MIN_PLANE_PIXELS = 200
# Heights outside these bounds are sensor error, not objects.
MIN_VALID_HEIGHT_M = 0.004
MAX_HEIGHT_MAD = 6.0


def ray_plane_distance(
    shape: tuple[int, int], intrinsics: CameraIntrinsics, height_m: float,
) -> np.ndarray:
    """Distance from the camera to an overhead plane at `height_m`, per pixel."""
    rows, columns = np.indices(shape, dtype=np.float64)
    x = (columns - intrinsics.ppx) / intrinsics.fx
    y = (rows - intrinsics.ppy) / intrinsics.fy
    return height_m * np.sqrt(1.0 + x * x + y * y)


def fit_plane_alignment(
    predicted: np.ndarray,
    region: np.ndarray,
    intrinsics: CameraIntrinsics,
    height_m: float,
    *,
    inverse: bool = False,
) -> tuple[DepthCalibration | None, dict[str, Any]]:
    """Scale and offset that map an empty-scene prediction onto the known plane.

    Trimmed least squares over the ROI: the empty scene is one plane, so a
    handful of leftover objects or depth spikes must not tilt the fit.
    """
    target = ray_plane_distance(predicted.shape[:2], intrinsics, height_m)
    usable = region & np.isfinite(predicted) & (predicted > 0)
    required = max(MIN_PLANE_PIXELS, int(MIN_PLANE_FRACTION * np.count_nonzero(region)))
    diagnostics: dict[str, Any] = {
        "plane_pixels": int(np.count_nonzero(usable)),
        "required_pixels": required,
        "camera_height_m": float(height_m),
        "predicted_median_on_plane": None,
        "expected_median_on_plane": float(np.median(target[region])) if np.any(region) else None,
    }
    if diagnostics["plane_pixels"] < required:
        return None, {**diagnostics, "reason": "too_few_empty_plane_pixels"}
    source = predicted[usable].astype(np.float64)
    if inverse:
        source = np.where(source > 1e-6, 1.0 / source, np.nan)
        usable_values = np.isfinite(source)
        source = source[usable_values]
        expected = target[usable][usable_values]
    else:
        expected = target[usable]
    diagnostics["predicted_median_on_plane"] = float(np.median(source))
    scale, offset = 1.0, 0.0
    for _ in range(3):
        design = np.column_stack((source, np.ones_like(source)))
        solution, *_ = np.linalg.lstsq(design, expected, rcond=None)
        scale, offset = float(solution[0]), float(solution[1])
        residual = scale * source + offset - expected
        limit = 3.0 * 1.4826 * float(np.median(np.abs(residual - np.median(residual))) or 1e-4)
        keep = np.abs(residual) <= limit
        if keep.all() or int(keep.sum()) < required // 2:
            break
        source, expected = source[keep], expected[keep]
    if not np.isfinite(scale) or scale <= 0:
        return None, {**diagnostics, "reason": "unstable_monocular_scale"}
    rmse = float(np.sqrt(np.mean((scale * source + offset - expected) ** 2)))
    diagnostics.update({"scale": scale, "offset_m": offset, "plane_rmse_m": rmse,
                        "fitted_pixels": int(source.size)})
    return DepthCalibration(
        scale=scale, offset_m=offset, rmse_m=rmse, sample_pixels=int(source.size),
        method="empty-plane-ray-alignment", reference_distance_m=float(height_m),
        sample_count=1, resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
        inverse=inverse,
    ), diagnostics


@dataclass
class HeightMapResult:
    measurement: VolumeMeasurement | None
    diagnostics: dict[str, Any]
    reason: str | None = None


def metric_object_volume(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    object_mask: np.ndarray | None,
    reference_plane: ReferencePlane | None,
    *,
    reference_depth_m: np.ndarray | None = None,
    measurement_mask: np.ndarray | None = None,
    min_height_m: float = 0.01,
    max_height_m: float = 0.80,
    min_pixels: int = 60,
) -> HeightMapResult:
    """Masked height-map volume from calibrated monocular depth, with its statistics."""
    diagnostics: dict[str, Any] = {}
    if depth_m is None or intrinsics is None or object_mask is None:
        return HeightMapResult(None, diagnostics, "missing_depth_intrinsics_or_mask")
    candidate = object_mask.astype(bool)
    if measurement_mask is not None:
        candidate &= measurement_mask.astype(bool)
    diagnostics["mask_pixels"] = int(np.count_nonzero(candidate))
    if diagnostics["mask_pixels"] < min_pixels:
        return HeightMapResult(None, diagnostics, "mask_too_small")

    depth = depth_m.astype(np.float64, copy=False)
    height_map = None
    if reference_plane is not None and reference_plane.coefficients is not None:
        height_map = _plane_perpendicular_height(depth, intrinsics, reference_plane.coefficients)
    diagnostics["height_source"] = "fitted_support_plane"
    if height_map is None:
        # No usable support plane: the empty-scene prediction is the reference.
        if reference_depth_m is None or reference_depth_m.shape != depth.shape:
            return HeightMapResult(None, diagnostics, "no_support_plane_or_empty_reference")
        height_map = reference_depth_m.astype(np.float64) - depth
        diagnostics["height_source"] = "empty_scene_reference"
    elif reference_depth_m is not None and reference_depth_m.shape == depth.shape:
        # The empty-scene prediction is the other, independent reference: used
        # when the fitted plane leaves this object with no measurable height
        # (a steeply mounted camera, or a plane fitted off the object's side).
        from_reference = reference_depth_m.astype(np.float64) - depth
        plane_pixels = int(np.count_nonzero(candidate & np.isfinite(height_map) & (height_map >= min_height_m)))
        if plane_pixels < min_pixels:
            height_map = from_reference
            diagnostics["height_source"] = "empty_scene_reference"
    try:
        import cv2

        # Monocular depth is smooth but locally wrong; a small median kills
        # single-pixel spikes without eating a thin packet's edge.
        smoothed = cv2.medianBlur(height_map.astype(np.float32), 5)
    except Exception:  # noqa: BLE001
        smoothed = height_map.astype(np.float32)
    heights = np.where(np.isfinite(smoothed), smoothed, 0.0).astype(np.float64)
    heights = np.clip(heights, 0.0, None)  # below the plane is not negative volume
    valid = candidate & np.isfinite(depth) & (depth > 0.05) & (heights >= min_height_m) \
        & (heights <= max_height_m)
    diagnostics["above_plane_pixels"] = int(np.count_nonzero(valid))
    if diagnostics["above_plane_pixels"] < min_pixels:
        return HeightMapResult(None, diagnostics, "no_measurable_height_above_plane")

    values = heights[valid]
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    spike_limit = median + MAX_HEIGHT_MAD * max(mad, 0.004)
    accepted = valid & (heights <= spike_limit)
    if int(np.count_nonzero(accepted)) < min_pixels:
        accepted = valid
    values = heights[accepted]
    areas = depth[accepted] ** 2 / (intrinsics.fx * intrinsics.fy)
    litres = float(np.sum(values * areas) * 1000.0)
    diagnostics.update({
        "height_median_m": median,
        "height_mad_m": mad,
        "height_p90_m": float(np.percentile(values, 90)),
        "height_max_m": float(values.max()),
        "spike_limit_m": float(spike_limit),
        "integrated_pixels": int(values.size),
        "rejected_spike_pixels": int(np.count_nonzero(valid) - values.size),
        "metric_pixel_area_median_m2": float(np.median(areas)),
        "object_depth_median_m": float(np.median(depth[accepted])),
        "raw_volume_l": litres,
    })
    if not math.isfinite(litres) or litres <= 0:
        return HeightMapResult(None, diagnostics, "non_finite_volume")
    coverage = values.size / max(1, diagnostics["mask_pixels"])
    measurement = VolumeMeasurement(
        liters=litres,
        valid_pixels=int(values.size),
        mean_height_m=float(values.mean()),
        max_height_m=float(values.max()),
        projected_area_m2=float(np.sum(areas)),
        method="logitech-depth-anything-v2-height-map",
        candidate_pixels=diagnostics["mask_pixels"],
        coverage_ratio=coverage,
        uncertainty_l=litres * 0.35,
        geometry_mode="calibrated-height-map",
        quality="monocular-calibrated",
        height_p90_m=diagnostics["height_p90_m"],
    )
    return HeightMapResult(measurement, diagnostics)


def stable_volume(samples: list[float], *, tolerance: float = 0.20) -> tuple[float | None, float]:
    """Trimmed median of recent instantaneous volumes, and their spread.

    Returns (stable litres or None, relative spread). None until the spread is
    inside `tolerance`, so a still-settling object shows only its live value.
    """
    if len(samples) < 3:
        return None, math.inf
    values = np.sort(np.asarray(samples[-9:], dtype=np.float64))
    trimmed = values[1:-1] if values.size >= 5 else values
    median = float(np.median(trimmed))
    spread = float(trimmed.max() - trimmed.min()) / max(median, 1e-6)
    return (median if spread <= tolerance else None), spread
