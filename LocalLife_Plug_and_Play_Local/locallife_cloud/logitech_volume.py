"""Metric height map and volume for the Logitech + Depth Anything V2 camera.

Depth Anything V2 is monocular: its depth is affine-related to metres at best,
so it is aligned to the *empty support plane* whose geometry is known from the
fixed installation (camera intrinsics and one tape-measured camera height).

    Z_metric(u,v) ~= a * Z_pred(u,v) + b        fitted on the empty ROI, where
    Z_plane(u,v)   = height / cos(theta(u,v))   is the distance along each ray
                                                to a plane perpendicular to the
                                                optical axis at that height.

Object volume is then a support-plane height map, not an image-plane one.
Every mask pixel is backprojected with its own depth and dropped onto the
plane; the plane is celled, and each occupied cell contributes its own height:

    V = sum over cells( cell_area * height(cell) )

Integrating in the image plane instead (area = Z^2/(fx*fy) per pixel, times
that pixel's height) is what made an obliquely seen bottle read four times too
large: its silhouette -- the tall front face -- was being treated as floor
footprint, so a 85 x 253 mm silhouette times 0.24 m of height gave ~5 L for a
1.4 L bottle. The front face now falls into the few cells in front of the
object, where it belongs, and the top surface sets each cell's height.

Heights are clipped at zero, spike-filtered (median + MAD) and capped, because
monocular depth is smooth but locally wrong, and a handful of spikes used to
dominate the integral. Every intermediate statistic is returned so one object
can be traced end to end.

RealSense keeps its own, unchanged volume path: nothing here is used for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .footprint import estimate_extents
from .logitech_bin import envelope_fill, shape_model_for
from .logitech_geometry import robust_object_height_m
from .types import CameraIntrinsics, DepthCalibration, VolumeMeasurement
from .volume import ReferencePlane, _plane_perpendicular_height

# The plane fit needs a real sample of the ROI, not a handful of pixels.
MIN_PLANE_FRACTION = 0.25
MIN_PLANE_PIXELS = 200
# Heights outside these bounds are sensor error, not objects.
MIN_VALID_HEIGHT_M = 0.004
MAX_HEIGHT_MAD = 6.0
# Support-plane cell size for the height map. 5 mm resolves a small can's
# footprint while staying far coarser than monocular depth's own noise.
CELL_SIZE_M = 0.005
# Per cell, the height that represents it: near-top, so a noisy pixel cannot
# set a whole cell, and a thin top surface is not averaged away by its sides.
CELL_HEIGHT_PERCENTILE = 90.0


def ray_plane_distance(
    shape: tuple[int, int], intrinsics: CameraIntrinsics, height_m: float,
) -> np.ndarray:
    """Distance from the camera to an overhead plane at `height_m`, per pixel."""
    rows, columns = np.indices(shape, dtype=np.float64)
    x = (columns - intrinsics.ppx) / intrinsics.fx
    y = (rows - intrinsics.ppy) / intrinsics.fy
    return height_m * np.sqrt(1.0 + x * x + y * y)


def _trimmed_fit(source: np.ndarray, expected: np.ndarray, required: int) -> tuple[float, float]:
    """Least squares with three rounds of MAD trimming."""
    scale, offset = 1.0, 0.0
    for _ in range(3):
        design = np.column_stack((source, np.ones_like(source)))
        solution, *_ = np.linalg.lstsq(design, expected, rcond=None)
        scale, offset = float(solution[0]), float(solution[1])
        residual = scale * source + offset - expected
        limit = 3.0 * 1.4826 * float(np.median(np.abs(residual - np.median(residual))) or 1e-4)
        keep = np.abs(residual) <= limit
        if keep.all() or int(keep.sum()) < max(required // 2, 8):
            break
        source, expected = source[keep], expected[keep]
    return scale, offset


def _mapping_error(
    source: np.ndarray, expected: np.ndarray, *, inverse: bool, required: int, seed: int = 0,
) -> tuple[float, float, float]:
    """Held-out RMSE of one mapping, plus the coefficients it fitted.

    Half the pixels fit, the other half score: a mapping that only looks good
    on the points it was fitted to is not the one to ship.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(source.size)
    half = source.size // 2
    fit_index, test_index = order[:half], order[half:]
    # The inverse form is fitted the way DepthCalibration.apply evaluates it:
    # 1/Z = a*d + b, so the target is inverted, not the prediction.
    targets = np.where(expected > 1e-6, 1.0 / expected, np.nan) if inverse else expected
    finite = np.isfinite(targets) & np.isfinite(source)
    fit_index = fit_index[finite[fit_index]]
    test_index = test_index[finite[test_index]]
    if fit_index.size < 8 or test_index.size < 8:
        return float("inf"), 1.0, 0.0
    scale, offset = _trimmed_fit(source[fit_index], targets[fit_index], required)
    mapped = scale * source[test_index] + offset
    predicted = np.where(mapped > 1e-6, 1.0 / mapped, np.nan) if inverse else mapped
    error = predicted - expected[test_index]
    error = error[np.isfinite(error)]
    if error.size < 8:
        return float("inf"), scale, offset
    return float(np.sqrt(np.mean(error ** 2))), scale, offset


def fit_plane_alignment(
    predicted: np.ndarray,
    region: np.ndarray,
    intrinsics: CameraIntrinsics,
    height_m: float,
    *,
    inverse: bool = False,
) -> tuple[DepthCalibration | None, dict[str, Any]]:
    """Scale and offset that map an empty-scene prediction onto the known plane.

    Both candidate mappings -- metric_depth = a*d + b and the inverse-depth
    form -- are fitted on half the ROI and scored on the other half, and the
    one with the lower held-out error is kept. `inverse` is only the model's
    own hint; the data decides.
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
    expected = target[usable]
    diagnostics["predicted_median_on_plane"] = float(np.median(source))
    linear_error, linear_scale, linear_offset = _mapping_error(
        source, expected, inverse=False, required=required)
    inverse_error, inverse_scale, inverse_offset = _mapping_error(
        source, expected, inverse=True, required=required)
    use_inverse = inverse_error < linear_error
    diagnostics.update({
        "held_out_rmse_linear_m": None if not np.isfinite(linear_error) else round(linear_error, 5),
        "held_out_rmse_inverse_m": None if not np.isfinite(inverse_error) else round(inverse_error, 5),
        "mapping": "inverse_depth" if use_inverse else "linear_depth",
        "model_hint_inverse": bool(inverse),
    })
    spread = float(np.percentile(source, 95) - np.percentile(source, 5))
    if spread < 0.005 * max(abs(float(np.median(source))), 1e-6):
        # The prediction is flat across the ROI: a plane at one predicted value
        # fixes the scale and says nothing about an offset, so only the scale
        # is fitted rather than letting least squares invent a pair.
        scale = float(np.median(expected) / max(np.median(source), 1e-9))
        diagnostics.update({"scale": scale, "offset_m": 0.0, "plane_rmse_m": 0.0,
                            "fitted_pixels": int(source.size), "mapping": "scale_only",
                            "reason_note": "constant_prediction_over_plane"})
        return DepthCalibration(
            scale=scale, offset_m=0.0, rmse_m=0.0, sample_pixels=int(source.size),
            method="empty-plane-ray-alignment", reference_distance_m=float(height_m),
            sample_count=1, resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
            inverse=False,
        ), diagnostics
    inverse = use_inverse
    targets = np.where(expected > 1e-6, 1.0 / expected, np.nan) if inverse else expected
    finite = np.isfinite(targets) & np.isfinite(source)
    source, targets, expected = source[finite], targets[finite], expected[finite]
    scale, offset = _trimmed_fit(source, targets, required)
    if not np.isfinite(scale) or scale <= 0:
        return None, {**diagnostics, "reason": "unstable_monocular_scale"}
    mapped = scale * source + offset
    metric = np.where(mapped > 1e-6, 1.0 / mapped, np.nan) if inverse else mapped
    residual = (metric - expected)[np.isfinite(metric)]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    diagnostics.update({"scale": scale, "offset_m": offset, "plane_rmse_m": rmse,
                        "fitted_pixels": int(source.size)})
    return DepthCalibration(
        scale=scale, offset_m=offset, rmse_m=rmse, sample_pixels=int(source.size),
        method="empty-plane-ray-alignment", reference_distance_m=float(height_m),
        sample_count=1, resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
        inverse=inverse,
    ), diagnostics


def axis_aligned_plane(intrinsics: CameraIntrinsics, height_m: float) -> ReferencePlane:
    """A plane perpendicular to the optical axis at `height_m`.

    Used when no plane has been fitted from the scene yet: the fixed overhead
    installation's own geometry is still known from the calibration.
    """
    return ReferencePlane(
        tilt_degrees=0.0, residual_rmse_m=0.0, inlier_pixels=0,
        normal=(0.0, 0.0, 1.0), coefficients=(0.0, 0.0, float(height_m)),
    )


def plane_basis(coefficients: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit normal and two in-plane axes for z = a*x + b*y + c."""
    a, b, _ = coefficients
    normal = np.array((a, b, -1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    seed = np.array((1.0, 0.0, 0.0)) if abs(normal[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    u_hat = seed - float(np.dot(seed, normal)) * normal
    u_hat /= np.linalg.norm(u_hat)
    return normal, u_hat, np.cross(normal, u_hat)


def project_to_plane(
    depth: np.ndarray, intrinsics: CameraIntrinsics, mask: np.ndarray,
    coefficients: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mask pixels as (N, 2) support-plane coordinates and their heights (metres)."""
    rows, columns = np.nonzero(mask)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))
    normal, u_hat, v_hat = plane_basis(coefficients)
    a, b, c = coefficients
    # Perpendicular distance above z = a*x + b*y + c, in the same sense as
    # volume._plane_perpendicular_height: positive when nearer than the plane.
    heights = (a * points[:, 0] + b * points[:, 1] + c - points[:, 2]) / np.sqrt(a * a + b * b + 1.0)
    return np.column_stack((points @ u_hat, points @ v_hat)), heights


def cell_height_map(
    footprint: np.ndarray, heights: np.ndarray, *, cell_size_m: float = CELL_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Group plane points into cells; return each occupied cell's key and height."""
    keys = np.floor(footprint / cell_size_m).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.lexsort((heights, inverse))
    sorted_cells, sorted_heights = inverse[order], heights[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    counts = np.diff(np.r_[boundaries, sorted_cells.size])
    # The percentile entry of each cell's own sorted heights.
    picks = boundaries + np.floor(CELL_HEIGHT_PERCENTILE / 100.0 * (counts - 1)).astype(int)
    return unique, sorted_heights[picks]


def object_plane_component(
    cells: np.ndarray, heights: np.ndarray, *, bridge_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the footprint the tracked object stands on, drop detached patches.

    Mask leakage -- floor beyond a bag, a foot beside a backpack -- lands as
    separate islands once the points are on the support plane, where the
    object itself is one connected footprint. The island holding the tallest
    cell is the object (leakage lies near the plane, which is why it leaked),
    so the others are dropped before the volume is integrated. A one-cell
    closing keeps a genuine footprint whole across a missing row of points.
    """
    import cv2

    origin = cells.min(axis=0)
    grid = cells - origin
    shape = (int(grid[:, 1].max()) + 1, int(grid[:, 0].max()) + 1)
    if min(shape) < 3 or shape[0] * shape[1] > 4_000_000:
        return cells, heights, 0
    occupancy = np.zeros(shape, np.uint8)
    occupancy[grid[:, 1], grid[:, 0]] = 1
    joined = cv2.morphologyEx(
        occupancy, cv2.MORPH_CLOSE, np.ones((2 * bridge_cells + 1, 2 * bridge_cells + 1), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if count <= 2:
        return cells, heights, 0
    tallest = int(np.argmax(heights))
    keep_label = int(labels[grid[tallest, 1], grid[tallest, 0]])
    if keep_label == 0:
        keep_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels[grid[:, 1], grid[:, 0]] == keep_label
    if int(keep.sum()) < 4:
        return cells, heights, 0
    return cells[keep], heights[keep], int((~keep).sum())


def fill_occluded_cells(
    cells: np.ndarray, heights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Complete the footprint one camera cannot see all of.

    An obliquely viewed object hides its far side: those support-plane cells
    receive no points at all, and the volume reads low (a 60-degree view of a
    bottle lost ~60 %). The occupied cells' convex hull is the footprint the
    object actually stands on, so empty cells inside it take the height of the
    nearest measured cell -- the same convex completion the RealSense
    height-map path already performs.
    """
    import cv2

    origin = cells.min(axis=0)
    grid = cells - origin
    shape = (int(grid[:, 1].max()) + 1, int(grid[:, 0].max()) + 1)
    if min(shape) < 2 or shape[0] * shape[1] > 4_000_000:
        return cells, heights, 0
    occupancy = np.zeros(shape, np.uint8)
    occupancy[grid[:, 1], grid[:, 0]] = 255
    hull = cv2.convexHull(np.column_stack((grid[:, 0], grid[:, 1])).astype(np.int32))
    inside = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(inside, hull, 255)
    missing = (inside > 0) & (occupancy == 0)
    if not missing.any():
        return cells, heights, 0
    # Nearest measured cell for every hole, by distance transform labels.
    _, labels = cv2.distanceTransformWithLabels(
        (occupancy == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL,
    )
    # Each occupied cell is its own label seed; map label -> that cell's height,
    # then read the nearest seed's label at every hole.
    value_by_label = np.zeros(int(labels.max()) + 1, np.float64)
    height_grid = np.zeros(shape, np.float64)
    height_grid[grid[:, 1], grid[:, 0]] = heights
    value_by_label[labels[occupancy > 0]] = height_grid[occupancy > 0]
    filled_rows, filled_columns = np.nonzero(missing)
    filled_heights = value_by_label[labels[filled_rows, filled_columns]]
    keep = filled_heights > 0
    added = np.column_stack((filled_columns[keep], filled_rows[keep])) + origin
    return (np.vstack((cells, added)), np.r_[heights, filled_heights[keep]], int(keep.sum()))


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
    cell_size_m: float = CELL_SIZE_M,
    camera_height_m: float | None = None,
    fill_occlusion: bool = True,
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
    plane = reference_plane if reference_plane is not None and reference_plane.coefficients is not None else None
    if plane is None and camera_height_m:
        # No floor plane fitted: fall back to a plane perpendicular to the
        # optical axis at the calibrated camera height. That is only the floor
        # for a camera pointing straight down. On a tilted mount every height
        # measured against it is the true height times the cosine of the tilt
        # -- which is why a 203 mm bottle read 137 mm and a 124 mm box read
        # 89 mm on a camera tilted about 45 degrees. The fallback still
        # produces a number, but it is named here so it can never be mistaken
        # for a floor-relative measurement.
        plane = axis_aligned_plane(intrinsics, camera_height_m)
        diagnostics["plane_source"] = "optical_axis_plane_not_floor"
    elif plane is not None:
        diagnostics["plane_source"] = "fitted_support_plane"
    if plane is not None and plane.coefficients is not None:
        # z = a*x + b*y + c with a = b = 0 is perpendicular to the optical
        # axis. Whatever produced it, it is the floor only for a camera
        # pointing straight down, and every height measured against it on a
        # tilted mount is short by the cosine of that tilt.
        gradient_a, gradient_b, _ = plane.coefficients
        diagnostics["plane_is_floor"] = bool(abs(gradient_a) > 1e-6 or abs(gradient_b) > 1e-6)
    if plane is None:
        return HeightMapResult(None, diagnostics, "no_support_plane")
    plane_coefficients = plane.coefficients
    height_map = _plane_perpendicular_height(depth, intrinsics, plane_coefficients)
    diagnostics["height_source"] = "fitted_support_plane"
    if height_map is None:
        return HeightMapResult(None, diagnostics, "degenerate_support_plane")
    if reference_depth_m is not None and reference_depth_m.shape == depth.shape:
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

    # Drop every valid pixel onto the support plane with its own depth, so a
    # vertical face lands in front of the object rather than under it. The
    # object's own footprint is isolated BEFORE any robust statistic is taken:
    # leaked floor outnumbers a small object, and a median over both once
    # rejected the object itself as the outlier.
    footprint, plane_heights = project_to_plane(depth, intrinsics, valid, plane_coefficients)
    keep = plane_heights >= min_height_m
    footprint, plane_heights = footprint[keep], plane_heights[keep]
    if footprint.shape[0] < min_pixels:
        return HeightMapResult(None, diagnostics, "no_measurable_height_above_plane")
    cells, cell_heights = cell_height_map(footprint, plane_heights, cell_size_m=cell_size_m)
    measured_cells = int(cells.shape[0])
    cells, cell_heights, dropped_cells = object_plane_component(cells, cell_heights)
    # The object's cells before the spike gate. The gate exists to protect the
    # integral, and it is right to be aggressive there -- but a bottle's neck
    # is a real part of the object that sits far above its body's median, and
    # clipping it is how a 203 mm bottle came back 150 mm tall. The height is
    # therefore a percentile of this population: high enough that a handful of
    # monocular spikes cannot reach it, low enough to keep a narrow neck.
    unclipped_heights = cell_heights.copy()
    # Now that only the object's own cells remain, a median/MAD gate removes
    # the monocular depth spikes that used to dominate the integral.
    median = float(np.median(cell_heights))
    mad = 1.4826 * float(np.median(np.abs(cell_heights - median)))
    spike_limit = median + MAX_HEIGHT_MAD * max(mad, 0.004)
    within = cell_heights <= spike_limit
    spike_cells = int((~within).sum())
    if int(within.sum()) >= 4:
        cells, cell_heights = cells[within], cell_heights[within]
    # The height comes from the cells that actually received points: the convex
    # completion copies a neighbour's height into the cells it adds, and letting
    # those into a percentile is what pulled the reported height below the
    # object's real top.
    object_height_m = robust_object_height_m(unclipped_heights, noise_floor_m=min_height_m)
    if fill_occlusion:
        cells, cell_heights, filled = fill_occluded_cells(cells, cell_heights)
    else:
        filled = 0
    cell_area = cell_size_m * cell_size_m
    litres = float(np.sum(cell_heights) * cell_area * 1000.0)
    # The dimensions describe the same cells the volume was integrated over,
    # completion included. Measuring them over the cells the volume did not see
    # left the two describing different objects, and a volume can then come out
    # larger than the envelope reported around it -- which is impossible, and
    # which is what an envelope check has to be able to trust.
    #
    # A cell is a square of `cell_size_m`, not a point: a rectangle fitted to
    # the cell *centres* stops half a cell short on each side. That is nothing
    # across a bag of several hundred cells and a quarter of the answer on a
    # small object of sixteen, so the extent of the cells is used, not their
    # midpoints.
    cell_centres = (cells.astype(np.float64) + 0.5) * cell_size_m
    extents = estimate_extents(cell_centres - cell_centres.mean(axis=0))
    if extents is not None:
        extents = replace(
            extents,
            length_m=extents.length_m + cell_size_m,
            width_m=extents.width_m + cell_size_m,
        )
    diagnostics["dimension_method"] = "logitech_floor_plane_footprint"
    diagnostics.update({
        "object_height_m": object_height_m,
        "height_median_m": median,
        "height_mad_m": mad,
        # The reported height. `height_p90_cells_m` is what it used to be: the
        # percentile across every footprint cell, hull-filled ones included,
        # which is the value that under-read every standing object.
        "height_p90_m": float(object_height_m or np.percentile(cell_heights, 90)),
        "height_p90_cells_m": float(np.percentile(cell_heights, 90)),
        # How far above the floor fit's own residual noise this height stands.
        # A height only a couple of times the plane's RMSE is not distinguishable
        # from the floor being slightly wrong, whatever the number says.
        "height_over_plane_noise": (
            None if plane.residual_rmse_m is None or not np.isfinite(plane.residual_rmse_m)
            or plane.residual_rmse_m <= 0 or object_height_m is None
            else round(float(object_height_m) / float(plane.residual_rmse_m), 2)
        ),
        "height_max_m": float(cell_heights.max()),
        "spike_limit_m": float(spike_limit),
        "integrated_pixels": int(plane_heights.size),
        "rejected_spike_cells": spike_cells,
        "rejected_spike_pixels": spike_cells,
        "footprint_cells": int(cells.shape[0]),
        "measured_cells": measured_cells,
        "background_cells_dropped": dropped_cells,
        "occlusion_filled_cells": filled,
        "footprint_area_m2": float(cells.shape[0] * cell_area),
        "cell_size_m": cell_size_m,
        "object_depth_median_m": float(np.median(depth[valid])),
        "length_mm": None if extents is None else extents.length_mm,
        "width_mm": None if extents is None else extents.width_mm,
        "raw_volume_l": litres,
    })
    # How much of its own envelope the object occupies. A cuboid test used to
    # sit here and called every bag inconsistent: a filled bag fills about half
    # to two thirds of the box around it, which is what a bag is, not an error.
    # Only the physical bounds are enforced now -- see logitech_bin.
    diagnostics.update(envelope_fill(
        length_m=None if extents is None else extents.length_m,
        width_m=None if extents is None else extents.width_m,
        height_m=object_height_m, litres=litres,
        shape_model=shape_model_for(
            None, occlusion_corrected=bool(extents is not None and extents.occlusion_corrected),
        ),
    ).to_dict())
    diagnostics["occlusion_corrected"] = bool(
        extents is not None and extents.occlusion_corrected
    )
    coverage = plane_heights.size / max(1, diagnostics["mask_pixels"])
    measurement = VolumeMeasurement(
        liters=litres,
        valid_pixels=int(plane_heights.size),
        mean_height_m=float(cell_heights.mean()),
        max_height_m=float(plane_heights.max()),
        projected_area_m2=diagnostics["footprint_area_m2"],
        method="logitech-depth-anything-v2-height-map",
        candidate_pixels=diagnostics["mask_pixels"],
        coverage_ratio=coverage,
        uncertainty_l=litres * 0.35,
        geometry_mode="calibrated-support-plane-height-map",
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
