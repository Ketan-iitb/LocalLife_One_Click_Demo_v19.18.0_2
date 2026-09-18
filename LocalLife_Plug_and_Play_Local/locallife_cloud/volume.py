"""Calibrated depth integration; never invent metric volume without a baseline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .geometry import roi_mask
from .types import (
    BoxVolumeMeasurement,
    CameraIntrinsics,
    DepthCalibration,
    ObjectDimensions,
    VolumeMeasurement,
)


@dataclass(slots=True)
class ReferencePlane:
    """Robust empty-bin floor diagnostic -- and, via `coefficients`, a usable
    correction for the mounting tilt every wall/bracket-mounted rig has.

    `coefficients` are the raw least-squares (a, b, c) from the plane fit
    z = a*x + b*y + c (camera-frame 3-D metres). They let a caller compute
    the true perpendicular distance from any backprojected 3-D point to this
    plane, which is what `estimate_volume()` uses to correct object height
    for tilt. Defaults to None so every existing caller/test that constructs
    a `ReferencePlane` directly (without this field) keeps working exactly
    as before -- the correction is simply skipped when it is absent, falling
    back to the original raw camera-Z-difference height.
    """

    tilt_degrees: float
    residual_rmse_m: float
    inlier_pixels: int
    normal: tuple[float, float, float]
    coefficients: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "tilt_degrees": round(float(self.tilt_degrees), 4),
            "residual_rmse_m": round(float(self.residual_rmse_m), 6),
            "inlier_pixels": int(self.inlier_pixels),
            "normal": [round(float(value), 7) for value in self.normal],
        }


MAX_REFERENCE_PLANE_RMSE_M = 0.008


def reference_plane_is_usable(
    plane: ReferencePlane | None, *, maximum_rmse_m: float = MAX_REFERENCE_PLANE_RMSE_M,
) -> bool:
    """Whether a support plane is accurate enough to publish metric L/W/H."""
    return bool(
        plane is not None
        and plane.coefficients is not None
        and np.isfinite(plane.residual_rmse_m)
        and plane.residual_rmse_m <= maximum_rmse_m
    )


def fit_reference_plane(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    *,
    mask: np.ndarray | None = None,
    minimum_samples: int = 100,
    maximum_samples: int = 10_000,
) -> ReferencePlane | None:
    """Estimate camera-to-floor alignment from robust 3-D pinhole backprojection."""
    if depth_m is None or intrinsics is None or depth_m.ndim != 2:
        return None
    valid = np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 20.0)
    if mask is not None:
        if mask.shape != valid.shape:
            return None
        valid &= mask.astype(bool)
    rows, columns = np.nonzero(valid)
    if rows.size < minimum_samples:
        return None
    if rows.size > maximum_samples:
        sample = np.linspace(0, rows.size - 1, maximum_samples, dtype=np.int64)
        rows, columns = rows[sample], columns[sample]
    z = depth_m[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    design = np.column_stack((x, y, np.ones_like(z)))

    # A single least-squares fit across a room averages the floor, wall,
    # chair and cables into a plane that does not physically exist.  That is
    # exactly what the hardware screenshots exposed: 78-94 cm "height" for
    # a measured 21.5 cm box, accompanied by high_plane_rmse.  Use a small,
    # deterministic RANSAC search to find the dominant coherent surface,
    # then refine only its inliers.
    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float] = (-1, float("-inf"))
    ransac_gate_m = 0.012
    if z.size >= 3:
        for _ in range(72):
            indices = rng.choice(z.size, size=3, replace=False)
            seed_design = design[indices]
            if np.linalg.cond(seed_design) > 1e7:
                continue
            candidate = np.linalg.solve(seed_design, z[indices])
            if not np.all(np.isfinite(candidate)):
                continue
            absolute = np.abs(z - design @ candidate)
            inliers = absolute <= ransac_gate_m
            count = int(np.count_nonzero(inliers))
            median_error = float(np.median(absolute[inliers])) if count else float("inf")
            score = (count, -median_error)
            if score > best_score:
                best_score = score
                best_inliers = inliers

    minimum_inliers = max(minimum_samples, int(np.ceil(z.size * 0.18)))
    if best_inliers is None or int(np.count_nonzero(best_inliers)) < minimum_inliers:
        # Preserve a conservative fallback for small, already-clean masks.
        coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
        residual = z - design @ coefficients
        deviation = np.abs(residual - np.median(residual))
        robust_sigma = max(0.0005, 1.4826 * float(np.median(deviation)))
        inliers = deviation <= max(0.003, 3.5 * robust_sigma)
    else:
        coefficients = np.linalg.lstsq(design[best_inliers], z[best_inliers], rcond=None)[0]
        residual = z - design @ coefficients
        seed_residual = residual[best_inliers]
        robust_sigma = max(
            0.0005,
            1.4826 * float(np.median(np.abs(seed_residual - np.median(seed_residual)))),
        )
        inliers = np.abs(residual) <= max(0.004, min(ransac_gate_m, 3.5 * robust_sigma))

    if int(np.count_nonzero(inliers)) < minimum_samples:
        return None
    design, z = design[inliers], z[inliers]
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
    normal = np.array((-coefficients[0], -coefficients[1], 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    residual = z - design @ coefficients
    return ReferencePlane(
        tilt_degrees=float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))),
        residual_rmse_m=float(np.sqrt(np.mean(residual * residual))),
        inlier_pixels=int(z.size),
        normal=tuple(float(item) for item in normal),
        coefficients=tuple(float(item) for item in coefficients),
    )


def fit_support_plane_from_background(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    *,
    object_mask: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    margin_fraction: float = 0.12,
    minimum_samples: int = 100,
) -> ReferencePlane | None:
    """Fit the support (table/floor) plane from the LIVE frame's own background.

    This exists to remove a hard dependency that silently blocked every
    volume measurement on real hardware: until now the support plane could
    only be fitted from a separately captured *empty-scene* baseline depth
    frame. If no such baseline had ever been captured, `reference_plane`
    stayed None, `estimate_box_volume_cuboid()` returned None on its very
    first guard, `estimate_volume()` had no reference to subtract, and every
    liters cell on the dashboard read "pending - pending empty baseline" no
    matter how correct the geometry code underneath was. Capturing that
    baseline requires the scene to be genuinely empty, which a real room
    with the object already in shot never is.

    The build spec's section 9.1 prescribes exactly this alternative: "fit
    the support-plane from valid background points, not from object-mask
    points; remove all object masks and expanded bounding boxes with a
    10-15% margin". The floor visible *around* the object in the current
    frame is real, measured, and always available -- no empty scene needed.

    `object_mask` (the union of detection masks) is dilated by
    `margin_fraction` of its own size before exclusion, so the noisy depth
    fringe at a segmentation boundary never contaminates the fit. Returns
    None if too little background survives, in which case the caller keeps
    whatever plane it already had rather than trusting a weak fit.
    """
    if depth_m is None or intrinsics is None or depth_m.ndim != 2:
        return None

    background = np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 20.0)
    if region_mask is not None and region_mask.shape == background.shape:
        background &= region_mask.astype(bool)

    if object_mask is not None and object_mask.shape == background.shape:
        excluded = object_mask.astype(bool)
        if np.any(excluded):
            # The support surface for this installation is below/alongside
            # the object in image space. Prefer the lower band surrounding
            # the object so a large wall or doorway cannot outvote the floor.
            object_rows = np.nonzero(excluded)[0]
            lower_start = int(np.percentile(object_rows, 35.0))
            lower_band = background.copy()
            lower_band[:lower_start, :] = False
            if int(np.count_nonzero(lower_band & ~excluded)) >= minimum_samples:
                background = lower_band
            try:
                import cv2

                # Scale the margin to the object's own size rather than using
                # a fixed kernel: a small carton and a full-frame bag need
                # very different exclusion rings for the same 10-15% intent.
                object_area = max(1, int(np.count_nonzero(excluded)))
                kernel_size = max(3, int(round((object_area**0.5) * margin_fraction)) | 1)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                excluded = cv2.dilate(excluded.astype(np.uint8), kernel, iterations=1).astype(bool)
            except ImportError:
                pass
        background &= ~excluded

    if int(np.count_nonzero(background)) < minimum_samples:
        return None
    return fit_reference_plane(
        depth_m, intrinsics, mask=background, minimum_samples=minimum_samples
    )


def synthesize_plane_depth(
    shape: tuple[int, int],
    intrinsics: CameraIntrinsics | None,
    coefficients: tuple[float, float, float] | None,
) -> np.ndarray | None:
    """Render the fitted support plane as a synthetic "empty floor" depth map.

    Every existing volume path measures an object against a reference depth
    image of the empty scene (`height = reference - current`). Once the
    support plane is known, that reference no longer has to be *captured* --
    it can be *computed*, exactly, for every pixel: the empty floor is by
    definition the plane itself, so the reference depth at pixel (u, v) is
    simply where that pixel's viewing ray meets the plane.

    With the plane written z = a*x + b*y + c and the pinhole ray
    x = (u - ppx) * z / fx, y = (v - ppy) * z / fy, substituting gives a
    closed form with no iteration and no approximation:

        z = c / (1 - a * (u - ppx) / fx - b * (v - ppy) / fy)

    Pixels whose ray is (near) parallel to the plane have no finite
    intersection and are returned as NaN, which every downstream consumer
    already treats as "no reference here" rather than as a measurement.
    """
    if intrinsics is None or coefficients is None or len(shape) != 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return None

    a, b, c = (float(value) for value in coefficients)
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c) or c <= 0:
        return None

    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(columns, rows)
    denominator = (
        1.0
        - a * (grid_x - intrinsics.ppx) / intrinsics.fx
        - b * (grid_y - intrinsics.ppy) / intrinsics.fy
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = c / denominator
    depth[np.abs(denominator) < 1e-6] = np.nan
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.10) | (depth >= 20.0)] = np.nan
    return depth.astype(np.float32)


def newly_introduced_mask(
    depth_m: np.ndarray | None,
    baseline_depth_m: np.ndarray | None,
    *,
    region_mask: np.ndarray | None = None,
    min_change_m: float = 0.012,
    max_change_m: float = 5.0,
    noise_map_m: np.ndarray | None = None,
    min_points: int = 40,
) -> np.ndarray | None:
    """Pixels that now read measurably closer than the empty-scene baseline.

    This is the only evidence the system has that an object was *introduced*
    rather than always being part of the room. A bed, a duvet fold and a
    pillow are all elevated above the floor plane and all connected to each
    other, so elevation alone can never separate them from a bag placed on
    top; the bag is distinguished by the fact that the surface under it moved
    towards the camera relative to the captured empty baseline.

    Returns None when no usable baseline exists, so callers can keep their
    existing behaviour instead of silently treating "no evidence of change"
    as "nothing changed".
    """
    if (
        depth_m is None or baseline_depth_m is None
        or depth_m.ndim != 2 or baseline_depth_m.shape != depth_m.shape
    ):
        return None
    if min_change_m <= 0 or max_change_m <= min_change_m:
        return None

    current = depth_m.astype(np.float64, copy=False)
    baseline = baseline_depth_m.astype(np.float64, copy=False)
    valid = (
        np.isfinite(current) & (current > 0.10) & (current < 20.0)
        & np.isfinite(baseline) & (baseline > 0.10) & (baseline < 20.0)
    )
    if region_mask is not None and region_mask.shape == valid.shape:
        valid &= region_mask.astype(bool)

    threshold = np.full(current.shape, float(min_change_m), dtype=np.float64)
    if noise_map_m is not None and noise_map_m.shape == current.shape:
        # Where the empty baseline itself was noisy, demand a correspondingly
        # larger change before believing something is physically there.
        threshold = np.maximum(threshold, np.nan_to_num(
            noise_map_m.astype(np.float64, copy=False), nan=0.0, posinf=0.0,
        ) * 3.0)

    change = baseline - current
    introduced = valid & (change >= threshold) & (change <= max_change_m)

    try:
        import cv2

        introduced = cv2.morphologyEx(
            introduced.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=1,
        ).astype(bool)
    except ImportError:
        pass

    if int(np.count_nonzero(introduced)) < min_points:
        return np.zeros(current.shape, dtype=bool)
    return introduced


def recover_elevated_object_mask(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    seed_mask: np.ndarray | None,
    reference_plane: ReferencePlane | None,
    *,
    measurement_mask: np.ndarray | None = None,
    min_height_m: float = 0.025,
    max_height_m: float = 0.80,
    min_points: int = 60,
    max_expansion: float = 6.0,
    change_mask: np.ndarray | None = None,
    min_change_fraction: float = 0.0,
) -> np.ndarray | None:
    """Recover the complete depth surface anchored by a semantic detection.

    Open-vocabulary segmentation can cover only a printed logo or the centre
    panel of a pale paper/plastic bag. Measuring that partial mask produces a
    plausible distance but a severely undersized footprint and height. Once a
    trustworthy support plane exists, RealSense itself can identify all pixels
    physically elevated above that plane. This function returns only the
    connected elevated component that overlaps the semantic seed, so an
    unrelated raised object elsewhere in the frame is never promoted to waste.

    The result is a *measurement mask*, not a new classifier: without an
    accepted plastic-bag, paper-bag, or cardboard seed, nothing is recovered.

    v7: elevation above the support plane is necessary but nowhere near
    sufficient on a real surface. On a bed, the object, the duvet folds and a
    pillow form one *connected* elevated component, so the largest-overlap
    rule below happily returned the whole bed and the estimator measured it
    honestly (the reported ~1021 x 669 mm). When `change_mask` is supplied --
    the pixels that read closer than the captured empty baseline, see
    `newly_introduced_mask` -- elevation is intersected with it, so unchanged
    furniture can no longer be annexed into the measured surface. A recovered
    mask that is still mostly unchanged background fails
    `min_change_fraction` and is rejected rather than measured.
    """
    if (
        depth_m is None or intrinsics is None or seed_mask is None
        or depth_m.ndim != 2 or seed_mask.shape != depth_m.shape
        or not reference_plane_is_usable(reference_plane)
    ):
        return None
    if measurement_mask is not None and measurement_mask.shape != depth_m.shape:
        return None
    if min_points < 1 or max_expansion < 1.0:
        return None

    seed = seed_mask.astype(bool)
    region = np.ones(depth_m.shape, dtype=bool)
    if measurement_mask is not None:
        region &= measurement_mask.astype(bool)
    seed &= region
    seed_pixels = int(np.count_nonzero(seed))
    if seed_pixels < min_points:
        return None

    depth = depth_m.astype(np.float64, copy=False)
    height_map = _plane_perpendicular_height(
        depth, intrinsics, reference_plane.coefficients,
    )
    if height_map is None:
        return None
    elevated = (
        region
        & np.isfinite(depth)
        & (depth > 0.10)
        & (depth < 20.0)
        & np.isfinite(height_map)
        & (height_map >= min_height_m)
        & (height_map <= max_height_m)
    )
    introduced: np.ndarray | None = None
    if change_mask is not None and change_mask.shape == depth_m.shape:
        introduced = change_mask.astype(bool)
        constrained = elevated & introduced
        # Only honour the constraint when the changed region still explains
        # the seed. If the baseline is stale or the object was already present
        # when it was captured, falling back to plain elevation preserves the
        # previous behaviour instead of silently measuring nothing.
        if int(np.count_nonzero(constrained & seed)) >= max(min_points // 4, 12):
            elevated = constrained

    # Bridge only small RealSense holes on the same physical surface. A tight
    # close is sufficient for stereo speckle and cannot span the large gap to
    # a second object elsewhere in the scene.
    try:
        import cv2

        elevated = cv2.morphologyEx(
            elevated.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=1,
        ).astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            elevated.astype(np.uint8), connectivity=8,
        )
        components = [
            labels == index for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_points
        ]
    except ImportError:
        from .geometry import connected_components

        components = connected_components(elevated, min_area=min_points)

    best: np.ndarray | None = None
    best_overlap = 0
    for component in components:
        overlap = int(np.count_nonzero(component & seed))
        if overlap > best_overlap:
            best_overlap = overlap
            best = component.astype(bool)
    # Deliberately loose: v4 added this recovery precisely because a detector
    # can return only a printed logo or centre panel, and that seed is a small
    # fraction of the true object. Contamination is prevented by the change
    # constraint above, not by starving a legitimately partial seed.
    minimum_overlap = max(12, min_points // 4, int(np.ceil(seed_pixels * 0.015)))
    if best is None or best_overlap < minimum_overlap:
        return None

    recovered_pixels = int(np.count_nonzero(best))
    # Kept in force even when the change constraint applied. "Newly
    # introduced" is not automatically small: a stale baseline, or one
    # captured from a different camera pose, can mark most of the frame as
    # changed, and without this bound the recovered surface would then be
    # unbounded again. A seed too small to reach its object under this ratio
    # falls back to the semantic mask -- undersized, but never the furniture.
    if recovered_pixels > int(seed_pixels * max_expansion):
        # A huge jump is more likely a contaminated/incorrect plane than a
        # legitimately partial object mask. Keep the semantic seed instead.
        return None
    if introduced is not None and min_change_fraction > 0.0:
        changed_fraction = int(np.count_nonzero(best & introduced)) / max(1, recovered_pixels)
        if changed_fraction < min_change_fraction:
            # The recovered surface is mostly scenery that was already there
            # before the object arrived. Measuring it would report the
            # furniture's dimensions under the object's label.
            return None
    return best


def _plane_perpendicular_height(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    coefficients: tuple[float, float, float],
) -> np.ndarray | None:
    """True object height above a (possibly tilted) fitted floor plane.

    `estimate_volume()`'s original height field was `baseline_m - depth_m`:
    the camera-Z difference between the empty-bin baseline and the current
    reading, at the same pixel. That is only the object's true vertical
    height when the camera's optical axis is exactly perpendicular to the
    bin floor. A wall/bracket-mounted rig (this project's actual, documented
    setup -- see DUAL_CAMERA_THESIS.md) is essentially never mounted at a
    perfect 0.0deg tilt, and for two parallel planes (an object's flat top,
    resting on a flat floor) cut by a camera ray at tilt angle theta from
    their shared normal, the camera-Z difference overstates the true
    perpendicular distance between them by a factor of 1/cos(theta) -- a
    25-30deg mounting tilt (very plausible for a wall bracket "pointing
    downward into the bin", not a calibrated overhead gantry) already
    inflates every reported height, and therefore every liters figure, by
    10-40%. This is a real, geometry-derived bug, not a guess: it explains
    "distance is accurate but height/volume is not" exactly, since the raw
    per-pixel depth READING is unaffected -- only the height DERIVED from it
    was wrong.

    This backprojects the object's own (row, col, depth) into the same 3-D
    camera frame `fit_reference_plane()` used to fit the floor, then returns
    the perpendicular (not camera-Z-axis) distance from each such point to
    that plane -- the true, tilt-corrected object height. Fully vectorised;
    returns None only if the plane is degenerate (should not happen given
    `fit_reference_plane()`'s own normalization).
    """
    a, b, c = coefficients
    denom = float(np.sqrt(a * a + b * b + 1.0))
    if not np.isfinite(denom) or denom < 1e-9:
        return None
    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    z = depth_m.astype(np.float64, copy=False)
    x = (columns - intrinsics.ppx) * z / intrinsics.fx
    y = (rows - intrinsics.ppy) * z / intrinsics.fy
    return (a * x + b * y + c - z) / denom


def _reject_isolated_outliers(
    depth: np.ndarray,
    region: np.ndarray,
    *,
    depth_noise_m: float,
) -> tuple[np.ndarray, int]:
    """Repair isolated stereo spikes only within locally smooth valid surfaces."""
    valid = np.isfinite(depth) & (depth > 0.10)
    try:
        import cv2

        working = np.where(valid, depth, 0.0).astype(np.float32)
        local_median = cv2.medianBlur(working, 3).astype(np.float64)
        valid_neighbors = cv2.filter2D(valid.astype(np.float32), -1, np.ones((3, 3), dtype=np.float32))
        region_neighbors = cv2.filter2D(region.astype(np.float32), -1, np.ones((3, 3), dtype=np.float32))
        deviation = np.abs(depth - local_median)
        residual = np.where(valid, deviation, 0).astype(np.float32)
        local_deviation = cv2.medianBlur(residual, 3)
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view

        padded = np.pad(np.where(valid, depth, np.nan), 1, mode="edge")
        neighborhoods = sliding_window_view(padded, (3, 3))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            local_median = np.nanmedian(neighborhoods, axis=(-2, -1))
        valid_neighbors = np.sum(np.isfinite(neighborhoods), axis=(-2, -1))
        region_neighbors = np.sum(
            sliding_window_view(np.pad(region.astype(np.uint8), 1, mode="edge"), (3, 3)), axis=(-2, -1)
        )
        deviation = np.abs(depth - local_median)
        residual = np.pad(np.where(valid, deviation, 0), 1, mode="edge")
        local_deviation = np.median(sliding_window_view(residual, (3, 3)), axis=(-2, -1))
    threshold = max(0.020, depth_noise_m * 6.0)
    spikes = (
        region & valid & (valid_neighbors >= 8) & (region_neighbors >= 8) & (local_median > 0.10)
        & (deviation > threshold) & (local_deviation < max(0.008, depth_noise_m * 2.5))
    )
    count = int(np.count_nonzero(spikes))
    if count:
        depth[spikes] = local_median[spikes]
    return depth, count


def _fill_local_depth_holes(depth: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill only isolated 3x3 holes with at least five valid neighbours.

    The original implementation iterated over every invalid pixel in Python.
    A 640x480 RealSense image can contain tens of thousands of invalid pixels,
    and volume is evaluated several times per frame.  Keeping this operation
    vectorised removes a large, avoidable source of dashboard latency while
    preserving the same conservative five-neighbour rule.
    """
    valid = np.isfinite(depth) & (depth > 0.10) & (depth < 20.0)
    holes = region & ~valid
    if not np.any(holes):
        return depth, 0
    try:
        import cv2

        working = np.where(valid, depth, 0.0).astype(np.float32)
        local_median = cv2.medianBlur(working, 3).astype(np.float64)
        neighbour_count = cv2.filter2D(
            valid.astype(np.float32), -1, np.ones((3, 3), dtype=np.float32),
            borderType=cv2.BORDER_CONSTANT,
        )
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view

        padded = np.pad(np.where(valid, depth, np.nan), 1, mode="constant", constant_values=np.nan)
        windows = sliding_window_view(padded, (3, 3))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            local_median = np.nanmedian(windows, axis=(-2, -1))
        neighbour_count = np.sum(np.isfinite(windows), axis=(-2, -1))
    fill = holes & (neighbour_count >= 5) & np.isfinite(local_median) & (local_median > 0.10)
    count = int(np.count_nonzero(fill))
    if count:
        depth[fill] = local_median[fill]
    return depth, count


def _triangulated_height_field(
    height_m: np.ndarray,
    baseline_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a regular triangular mesh over the calibrated baseline plane.

    This is the real-time image-grid equivalent of VolPy's point -> triangle ->
    plane -> integral method. Each depth pixel contributes four calibrated 3-D
    corner vertices; the two planar triangles exactly integrate a linearly
    interpolated height field across that pixel footprint.
    """
    rows, columns = height_m.shape
    height_sum = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    depth_sum = np.zeros_like(height_sum)
    samples = np.zeros_like(height_sum)
    safe_height = np.where(valid, height_m, 0.0)
    safe_depth = np.where(valid, baseline_m, 0.0)
    weights = valid.astype(np.float64)
    for row_offset, column_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
        row_slice = slice(row_offset, row_offset + rows)
        column_slice = slice(column_offset, column_offset + columns)
        height_sum[row_slice, column_slice] += safe_height
        depth_sum[row_slice, column_slice] += safe_depth
        samples[row_slice, column_slice] += weights
    with np.errstate(divide="ignore", invalid="ignore"):
        corner_height = np.divide(height_sum, samples, out=np.zeros_like(height_sum), where=samples > 0)
        corner_depth = np.divide(depth_sum, samples, out=np.zeros_like(depth_sum), where=samples > 0)

    vertical, horizontal = np.indices(corner_depth.shape, dtype=np.float64)
    horizontal -= 0.5
    vertical -= 0.5
    corner_x = (horizontal - intrinsics.ppx) * corner_depth / intrinsics.fx
    corner_y = (vertical - intrinsics.ppy) * corner_depth / intrinsics.fy

    x_tl, y_tl, h_tl = corner_x[:-1, :-1], corner_y[:-1, :-1], corner_height[:-1, :-1]
    x_tr, y_tr, h_tr = corner_x[:-1, 1:], corner_y[:-1, 1:], corner_height[:-1, 1:]
    x_br, y_br, h_br = corner_x[1:, 1:], corner_y[1:, 1:], corner_height[1:, 1:]
    x_bl, y_bl, h_bl = corner_x[1:, :-1], corner_y[1:, :-1], corner_height[1:, :-1]
    first_area = 0.5 * np.abs((x_tr - x_tl) * (y_br - y_tl) - (y_tr - y_tl) * (x_br - x_tl))
    second_area = 0.5 * np.abs((x_br - x_tl) * (y_bl - y_tl) - (y_br - y_tl) * (x_bl - x_tl))
    cell_area = first_area + second_area
    cell_volume = (
        first_area * (h_tl + h_tr + h_br) / 3.0
        + second_area * (h_tl + h_br + h_bl) / 3.0
    )
    usable = valid & np.isfinite(cell_area) & np.isfinite(cell_volume) & (cell_area > 0)
    return cell_volume[usable], cell_area[usable]


def estimate_volume(
    depth_m: np.ndarray | None,
    baseline_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    *,
    object_mask: np.ndarray | None = None,
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    min_height_m: float = 0.015,
    max_height_m: float = 0.80,
    min_pixels: int = 25,
    method: str = "realsense",
    measurement_mask: np.ndarray | None = None,
    depth_noise_m: float = 0.004,
    fill_small_holes: bool = True,
    geometry_mode: str = "surface-columns",
    calibration_factor: float = 1.0,
    systematic_error_fraction: float = 0.0,
    baseline_noise_m: float = 0.0,
    baseline_noise_map: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    reject_outliers: bool = False,
    reference_plane: ReferencePlane | None = None,
) -> VolumeMeasurement | None:
    """Integrate projected pixel area times object height in cubic meters.

    This measures the visible height field above an empty-scene baseline. It is
    not a closed-mesh reconstruction and does not infer hidden object surfaces.
    """
    if depth_m is None or baseline_m is None or intrinsics is None:
        return None
    if depth_m.ndim != 2 or depth_m.shape != baseline_m.shape:
        return None
    if object_mask is not None and object_mask.shape != depth_m.shape:
        return None
    if measurement_mask is not None and measurement_mask.shape != depth_m.shape:
        return None
    if baseline_noise_map is not None and baseline_noise_map.shape != depth_m.shape:
        return None
    if geometry_mode not in {
        "surface-columns", "ray-frustum", "reference-plane", "triangulated-surface"
    }:
        raise ValueError("Unsupported volume integration geometry")
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        raise ValueError("Calibration factor must be finite and positive")

    depth = depth_m.astype(np.float64, copy=True)
    baseline = baseline_m.astype(np.float64, copy=False)
    region = roi_mask(depth.shape, roi)
    if measurement_mask is not None:
        region &= measurement_mask.astype(bool)
    if object_mask is not None:
        region &= object_mask.astype(bool)

    original_valid_depth = np.isfinite(depth) & (depth > 0.10) & (depth < 20.0)
    candidate_pixels = int(np.count_nonzero(region))
    observed_pixels = int(np.count_nonzero(region & original_valid_depth))
    coverage_ratio = observed_pixels / max(1, candidate_pixels)
    filled_pixels = 0
    rejected_pixels = 0
    # Dark, low-IR-reflectivity materials (a black backpack, a black bag)
    # routinely give the RealSense stereo depth sensor patchy, scattered
    # dropout across an otherwise perfectly real, compact object -- not one
    # large missing block, but many small holes throughout its surface. The
    # previous single-pass fill only ran above 70% raw coverage and only
    # closed 3x3 holes with 5+ valid neighbours, so a real object with, say,
    # 40% patchy dropout was refused entirely ("rejected sparse height
    # inside mask") even though most of its footprint was genuinely there.
    # Iterating the same conservative 5-neighbour rule closes holes one
    # ring at a time, growing inward from whatever *is* valid each pass,
    # while still leaving `coverage_ratio` (computed above, before any
    # filling) as the honest raw-sensor number that drives the uncertainty
    # budget and quality label below -- filling more aggressively lets a
    # real, patchy object produce a (correctly wider-uncertainty) volume
    # instead of nothing, without hiding how much of it was actually seen.
    # A large solid missing block (most of the object never sensed at all)
    # still cannot be filled by this rule: pixels deep inside it never reach
    # 5 valid neighbours in any pass, so nothing there is fabricated.
    if fill_small_holes and candidate_pixels and coverage_ratio >= 0.25:
        for _ in range(4):
            depth, newly_filled = _fill_local_depth_holes(depth, region)
            filled_pixels += newly_filled
            if newly_filled == 0:
                break
    if reject_outliers:
        depth, rejected_pixels = _reject_isolated_outliers(depth, region, depth_noise_m=depth_noise_m)

    height = baseline - depth
    # Tilt-corrected height, when a fitted floor plane is available (see
    # `_plane_perpendicular_height`'s docstring for why the raw `height`
    # above is systematically wrong for any non-zero mounting tilt). Falls
    # back to the original camera-Z-difference `height` when no plane was
    # fitted yet (e.g. before the first baseline) or a caller intentionally
    # omits it -- byte-identical to the pre-existing behaviour in that case.
    effective_height = height
    if reference_plane is not None and reference_plane.coefficients is not None:
        corrected = _plane_perpendicular_height(depth, intrinsics, reference_plane.coefficients)
        if corrected is not None and corrected.shape == height.shape:
            effective_height = corrected
    adaptive_minimum = min_height_m
    if noise_sigma > 0:
        if baseline_noise_map is not None:
            reference_noise = np.maximum(baseline_noise_map.astype(np.float64), baseline_noise_m)
        else:
            reference_noise = baseline_noise_m
        adaptive_minimum = np.maximum(
            min_height_m,
            noise_sigma * np.sqrt(depth_noise_m * depth_noise_m + reference_noise * reference_noise),
        )
    valid = (
        np.isfinite(depth)
        & np.isfinite(baseline)
        & (depth > 0.10)
        & (baseline > 0.10)
        & (effective_height >= adaptive_minimum)
        & (effective_height <= max_height_m)
        & region
    )
    pixel_count = int(np.count_nonzero(valid))
    if pixel_count < min_pixels:
        return None

    object_depth = depth[valid]
    reference_depth = baseline[valid]
    object_height = effective_height[valid]
    focal_product = intrinsics.fx * intrinsics.fy
    if geometry_mode == "ray-frustum":
        # Correction (round 16): this sum is computed directly from raw
        # camera-Z `object_depth`/`reference_depth` and does NOT read
        # `object_height` (== `effective_height[valid]`, the tilt-corrected
        # perpendicular height) anywhere in the volume math below -- it is
        # used only to back-solve `pixel_area_m2` for the uncertainty budget.
        # This mode's own volume total is therefore NOT tilt-corrected,
        # despite round 12's claim that it was exact regardless of mounting
        # tilt; that claim was wrong in a way real-hardware testing
        # confirmed (see config.py's `volume_geometry` comment for the full
        # story). Kept available for the documented thesis sensitivity
        # comparison, but it is no longer the default for exactly this
        # reason -- prefer "reference-plane" for a genuinely tilt-corrected
        # per-pixel sum.
        contributions_m3 = (reference_depth**3 - object_depth**3) / (3.0 * focal_product)
        pixel_area_m2 = contributions_m3 / object_height
    elif geometry_mode == "reference-plane":
        pixel_area_m2 = (reference_depth * reference_depth) / focal_product
        contributions_m3 = object_height * pixel_area_m2
    elif geometry_mode == "triangulated-surface":
        contributions_m3, pixel_area_m2 = _triangulated_height_field(
            effective_height, baseline, valid, intrinsics
        )
        if contributions_m3.size < min_pixels:
            return None
    else:
        pixel_area_m2 = (object_depth * object_depth) / focal_product
        contributions_m3 = object_height * pixel_area_m2
    raw_liters = float(np.sum(contributions_m3) * 1000.0)
    liters = raw_liters * calibration_factor
    unobserved_fraction = max(0.0, 1.0 - coverage_ratio)
    depth_variance = depth_noise_m * depth_noise_m + baseline_noise_m * baseline_noise_m
    if baseline_noise_map is not None:
        sample_noise = np.maximum(baseline_noise_map[valid].astype(np.float64), baseline_noise_m)
        depth_variance = depth_noise_m * depth_noise_m + sample_noise * sample_noise
    sensor_uncertainty_l = float(
        np.sqrt(np.sum((pixel_area_m2**2) * depth_variance)) * 1000.0 * calibration_factor
    )
    missing_depth_uncertainty_l = float(liters * unobserved_fraction)
    systematic_uncertainty_l = float(liters * systematic_error_fraction)
    uncertainty_l = float(np.sqrt(
        sensor_uncertainty_l**2 + missing_depth_uncertainty_l**2 + systematic_uncertainty_l**2
    ))
    relative_uncertainty = uncertainty_l / max(liters, 1e-12)
    quality = (
        "high" if coverage_ratio >= 0.95 and relative_uncertainty <= 0.05
        else "moderate" if coverage_ratio >= 0.80 and relative_uncertainty <= 0.15
        else "low"
    )
    # A person measures an object's "height" with a ruler against its
    # tallest point, not the average height across its whole footprint --
    # a dome-shaped pillow or a bag with sloped/tapered sides has plenty of
    # low-height pixels near its edges that pull a mean or median far below
    # what anyone would call its height. The 90th percentile of this same
    # already-cleaned column-height field (post hole-fill, post outlier
    # rejection, restricted to exactly the pixels this measurement's own
    # liters figure is integrated from) reports the near-top surface while
    # still ignoring a single hot noise spike the way a bare max() would not.
    height_p90 = float(np.percentile(object_height, 90)) if object_height.size else 0.0
    return VolumeMeasurement(
        liters=liters,
        valid_pixels=pixel_count,
        mean_height_m=float(np.mean(object_height)),
        max_height_m=float(np.max(object_height)),
        projected_area_m2=float(np.sum(pixel_area_m2)),
        method=method,
        candidate_pixels=candidate_pixels,
        filled_pixels=filled_pixels,
        coverage_ratio=coverage_ratio,
        uncertainty_l=uncertainty_l,
        raw_liters=raw_liters,
        geometry_mode=geometry_mode,
        calibration_factor=calibration_factor,
        random_uncertainty_l=sensor_uncertainty_l,
        systematic_uncertainty_l=systematic_uncertainty_l,
        baseline_noise_m=float(baseline_noise_m),
        rejected_pixels=rejected_pixels,
        quality=quality,
        height_p90_m=height_p90,
    )


def _erode_object_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a boolean mask by `iterations` pixels (PDF section 4.1: RGB-D
    object boundaries routinely contain mixed background/object depths, so
    the mask must be eroded before depth is sampled from it)."""
    if iterations <= 0:
        return mask
    try:
        import cv2

        return cv2.erode(
            mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=iterations
        ).astype(bool)
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view

        eroded = mask.astype(bool)
        for _ in range(iterations):
            windows = sliding_window_view(np.pad(eroded.astype(np.uint8), 1, mode="constant"), (3, 3))
            eroded = np.all(windows, axis=(-2, -1))
        return eroded


def _mask_touches_measurement_boundary(
    object_mask: np.ndarray,
    measurement_mask: np.ndarray | None = None,
) -> bool:
    """Detect clipping on the original mask, before edge erosion hides it."""
    mask = object_mask.astype(bool)
    if not np.any(mask):
        return False
    if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return True
    if measurement_mask is None or measurement_mask.shape != mask.shape:
        return False
    allowed = measurement_mask.astype(bool)
    interior = allowed.copy()
    interior[1:, :] &= allowed[:-1, :]
    interior[:-1, :] &= allowed[1:, :]
    interior[:, 1:] &= allowed[:, :-1]
    interior[:, :-1] &= allowed[:, 1:]
    return bool(np.any(mask & allowed & ~interior))


def _largest_connected_region(mask: np.ndarray) -> np.ndarray:
    """Keep one coherent elevated surface instead of disconnected noise."""
    if not np.any(mask):
        return mask.astype(bool)
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask.astype(bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest
    except ImportError:
        return mask.astype(bool)


def estimate_object_dimensions(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    object_mask: np.ndarray | None,
    reference_plane: ReferencePlane | None,
    *,
    measurement_mask: np.ndarray | None = None,
    min_height_m: float = 0.025,
    max_height_m: float = 0.80,
    min_points: int = 60,
    footprint_trim_percentile: float = 2.0,
    height_percentile: float = 95.0,
) -> ObjectDimensions | None:
    """Measure visible support-plane footprint and robust RealSense height.

    Only points measurably above the plane enter the footprint.  This prevents
    an RGB/YOLO mask's floor halo from inflating length, width, and volume.
    """
    if depth_m is None or intrinsics is None or object_mask is None:
        return None
    if reference_plane is None or reference_plane.coefficients is None:
        return None
    if depth_m.ndim != 2 or object_mask.shape != depth_m.shape:
        return None
    if measurement_mask is not None and measurement_mask.shape != depth_m.shape:
        return None
    if not (0.0 <= footprint_trim_percentile < 50.0):
        raise ValueError("Footprint trim percentile must be between 0 and 50")
    if not (50.0 <= height_percentile <= 100.0):
        raise ValueError("Height percentile must be between 50 and 100")

    candidate = object_mask.astype(bool).copy()
    if measurement_mask is not None:
        candidate &= measurement_mask.astype(bool)
    candidate_pixels = int(np.count_nonzero(candidate))
    if candidate_pixels < min_points:
        return None

    depth = depth_m.astype(np.float64, copy=False)
    depth_ok = np.isfinite(depth) & (depth > 0.10) & (depth < 20.0)
    sensed_pixels = int(np.count_nonzero(candidate & depth_ok))
    depth_valid_ratio = sensed_pixels / max(1, candidate_pixels)
    height_map = _plane_perpendicular_height(depth, intrinsics, reference_plane.coefficients)
    if height_map is None:
        return None
    elevated = (
        candidate
        & depth_ok
        & np.isfinite(height_map)
        & (height_map >= min_height_m)
        & (height_map <= max_height_m)
    )
    elevated = _largest_connected_region(elevated)
    valid_count = int(np.count_nonzero(elevated))
    if valid_count < min_points:
        return None

    rows, columns = np.nonzero(elevated)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))

    a, b, _ = reference_plane.coefficients
    normal = np.array((a, b, -1.0), dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-9:
        return None
    normal /= normal_norm
    seed = np.array((1.0, 0.0, 0.0)) if abs(normal[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    u_hat = seed - float(np.dot(seed, normal)) * normal
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(normal, u_hat)

    footprint = np.column_stack((points @ u_hat, points @ v_hat))
    footprint -= np.median(footprint, axis=0)
    covariance = np.cov(footprint, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    _, eigenvectors = np.linalg.eigh(covariance)
    principal = footprint @ eigenvectors
    low = np.percentile(principal, footprint_trim_percentile, axis=0)
    high = np.percentile(principal, 100.0 - footprint_trim_percentile, axis=0)
    dimensions_m = np.sort(np.maximum(0.0, high - low))[::-1]
    if dimensions_m.size != 2 or dimensions_m[1] <= 0:
        return None

    heights = height_map[elevated]
    height_m = float(np.percentile(heights, height_percentile))
    mask_clipped = _mask_touches_measurement_boundary(object_mask, measurement_mask)
    flags: list[str] = ["single_view_visible_footprint"]
    if depth_valid_ratio < 0.70:
        flags.append("low_valid_depth")
    if valid_count / max(1, sensed_pixels) < 0.35:
        flags.append("low_elevated_fraction")
    if reference_plane.residual_rmse_m * 1000.0 > 8.0:
        flags.append("high_plane_rmse")
    if mask_clipped:
        flags.append("mask_clipped")

    confidence = 0.85
    confidence -= 0.30 * max(0.0, 0.90 - depth_valid_ratio)
    confidence -= 0.15 if "low_elevated_fraction" in flags else 0.0
    confidence -= 0.15 if "high_plane_rmse" in flags else 0.0
    confidence -= 0.20 if mask_clipped else 0.0
    return ObjectDimensions(
        length_mm=float(dimensions_m[0] * 1000.0),
        width_mm=float(dimensions_m[1] * 1000.0),
        height_mm=height_m * 1000.0,
        confidence=float(np.clip(confidence, 0.05, 0.95)),
        depth_valid_ratio=depth_valid_ratio,
        object_points=valid_count,
        mask_clipped=mask_clipped,
        flags=tuple(flags),
    )


def estimate_box_volume_cuboid(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    object_mask: np.ndarray | None,
    reference_plane: ReferencePlane | None,
    *,
    measurement_mask: np.ndarray | None = None,
    mask_erosion_px: int = 0,
    min_points: int = 60,
    height_percentile: float = 90.0,
    fallback_height_percentile: float = 98.0,
    footprint_trim_percentile: float = 2.0,
    min_height_m: float = 0.025,
    max_height_m: float = 0.80,
) -> BoxVolumeMeasurement | None:
    """Table-relative rigid-box volume: robust height above the fitted table
    plane, times a robust footprint length/width, per the "Revised
    Dual-Camera Volume Estimation" engineering recipe (sections 2, 4.3, 9).

    This intentionally does NOT sum per-pixel height*area contributions the
    way `estimate_volume()` does. For a box, camera-axis depth treated as
    height (or even a per-pixel plane-corrected height *summed* over a noisy,
    partially-eroded, possibly mask-leaking segmentation) is exactly the kind
    of coordinate-frame and edge-noise sensitivity the PDF identifies as the
    root cause of "mesh looks good, height/volume is wrong": a box's true
    volume is three scalar dimensions (L, W, H), not an integral over noisy
    per-pixel columns. Measuring L, W, H robustly and multiplying is more
    defensible for a rigid box and matches the PDF's own explicit formula
    (volume_l = length_m * width_m * height_m * 1000).

    Returns None if there isn't a usable table plane, or too few valid
    object points survive optional mask erosion + depth filtering -- this never
    fabricates a box measurement from insufficient geometry.
    """
    if depth_m is None or intrinsics is None or object_mask is None:
        return None
    if reference_plane is None or reference_plane.coefficients is None:
        return None
    if object_mask.shape != depth_m.shape:
        return None
    if measurement_mask is not None and measurement_mask.shape != depth_m.shape:
        return None

    a, b, c = reference_plane.coefficients
    denom = float(np.sqrt(a * a + b * b + 1.0))
    if not np.isfinite(denom) or denom < 1e-9:
        return None

    # Clipping must be checked on the original mask.  Erosion deliberately
    # removes its outer pixels and previously hid real image-edge clipping.
    mask_clipped = _mask_touches_measurement_boundary(object_mask, measurement_mask)
    geometric_mask = object_mask.astype(bool)
    if measurement_mask is not None:
        geometric_mask &= measurement_mask.astype(bool)
    eroded_mask = _erode_object_mask(geometric_mask, mask_erosion_px)
    raw_object_pixels = int(np.count_nonzero(geometric_mask))
    depth = depth_m.astype(np.float64)
    depth_ok = np.isfinite(depth) & (depth > 0.10) & (depth < 20.0)
    height_map = _plane_perpendicular_height(depth, intrinsics, reference_plane.coefficients)
    if height_map is None:
        return None
    valid = (
        eroded_mask
        & depth_ok
        & np.isfinite(height_map)
        & (height_map >= min_height_m)
        & (height_map <= max_height_m)
    )
    valid = _largest_connected_region(valid)
    valid_count = int(np.count_nonzero(valid))
    depth_valid_ratio = valid_count / max(1, raw_object_pixels)
    if valid_count < min_points:
        return None

    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy

    # Per-point perpendicular height above the fitted table plane.  It was
    # calculated before footprint PCA so floor/background pixels inside a
    # loose semantic mask cannot expand the physical box dimensions.
    height = height_map[valid]

    # Robust top height: retain the median of the upper band as a diagnostic,
    # but report the bounded 98th percentile. Real D435 masks include lower
    # side-wall and bevel pixels near a rigid box's outline; their presence
    # made the old upper-decile median (roughly p95) systematically short in
    # the V3 ruler trials. p98 remains far less noise-sensitive than raw max.
    threshold = float(np.percentile(height, height_percentile))
    top_band = height[height >= threshold]
    if top_band.size < 5:
        threshold = float(np.percentile(height, fallback_height_percentile))
        top_band = height[height >= threshold]
    if top_band.size == 0:
        return None
    height_top_median_m = float(np.median(top_band))
    height_p98_m = float(np.percentile(height, fallback_height_percentile))
    height_m = height_p98_m
    if height_m <= 0:
        return None

    # In-plane orthonormal basis (u_hat, v_hat), perpendicular to the plane
    # normal. Any such basis works as the intermediate frame: the actual
    # object-aligned axes are recovered afterward via PCA, which is
    # rotation-independent within that plane.
    normal = np.array((a, b, -1.0), dtype=np.float64) / denom
    seed = np.array((1.0, 0.0, 0.0)) if abs(normal[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    u_hat = seed - float(np.dot(seed, normal)) * normal
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(normal, u_hat)

    points = np.column_stack((x, y, z))
    u = points @ u_hat
    v = points @ v_hat
    footprint = np.column_stack((u, v))
    footprint -= footprint.mean(axis=0)
    covariance = np.cov(footprint, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # eigh returns ascending eigenvalues; the largest-variance axis last.
    principal = footprint @ eigenvectors
    axis_a = principal[:, 1]
    axis_b = principal[:, 0]
    low_a, high_a = np.percentile(axis_a, (footprint_trim_percentile, 100 - footprint_trim_percentile))
    low_b, high_b = np.percentile(axis_b, (footprint_trim_percentile, 100 - footprint_trim_percentile))
    length_m = max(0.0, float(high_a - low_a))
    width_m = max(0.0, float(high_b - low_b))
    if length_m <= 0 or width_m <= 0:
        return None

    volume_l = length_m * width_m * height_m * 1000.0

    plane_rmse_mm = reference_plane.residual_rmse_m * 1000.0
    flags: list[str] = ["single_view_estimate"]
    if depth_valid_ratio < 0.70:
        flags.append("low_valid_depth")
    if plane_rmse_mm > 8.0:
        flags.append("high_plane_rmse")
    if mask_clipped:
        flags.append("mask_clipped")

    confidence = 0.80
    confidence -= 0.25 * max(0.0, 0.90 - depth_valid_ratio)
    confidence -= 0.15 if "high_plane_rmse" in flags else 0.0
    confidence -= 0.10 if mask_clipped else 0.0
    confidence = float(np.clip(confidence, 0.05, 0.95))

    return BoxVolumeMeasurement(
        volume_liters=volume_l,
        volume_confidence=confidence,
        volume_method="table_relative_cuboid",
        length_mm=length_m * 1000.0,
        width_mm=width_m * 1000.0,
        height_mm=height_m * 1000.0,
        depth_valid_ratio=depth_valid_ratio,
        object_points=valid_count,
        table_plane_inliers=reference_plane.inlier_pixels,
        table_plane_rmse_mm=plane_rmse_mm,
        height_p98_mm=height_p98_m * 1000.0,
        height_top_median_mm=height_top_median_m * 1000.0,
        mask_clipped=mask_clipped,
        flags=tuple(flags),
    )


_DIMENSION_INSTABILITY_FRACTION = 0.12


def aggregate_object_dimensions(
    measurements: list[ObjectDimensions],
    *,
    frames_considered: int | None = None,
    dimension_instability_fraction: float = _DIMENSION_INSTABILITY_FRACTION,
) -> ObjectDimensions | None:
    """Median-aggregate general RealSense dimensions for one tracked object.

    Household validation objects and deformable bags use the general
    support-plane estimator rather than the rigid cuboid path. They need the
    same protection against one noisy depth/mask frame that boxes already
    receive. This combines L/W/H independently and records the observed
    spread; it never changes object classification or uses Logitech geometry.
    """
    if not measurements:
        return None

    values = np.asarray(
        [[item.length_mm, item.width_mm, item.height_mm] for item in measurements],
        dtype=np.float64,
    )
    medians = np.median(values, axis=0)
    if np.any(~np.isfinite(medians)) or np.any(medians <= 0):
        return None
    spreads = np.std(values, axis=0)
    relative_spreads = spreads / np.maximum(medians, 1e-6)
    unstable = float(np.max(relative_spreads)) > dimension_instability_fraction

    flags = {flag for item in measurements for flag in item.flags}
    if unstable:
        flags.add("dimension_instability")
    confidence = float(np.median([item.confidence for item in measurements]))
    if unstable:
        confidence *= 0.6

    latest = measurements[-1]
    return ObjectDimensions(
        length_mm=float(medians[0]),
        width_mm=float(medians[1]),
        height_mm=float(medians[2]),
        confidence=float(np.clip(confidence, 0.0, 0.99)),
        depth_valid_ratio=float(np.median([item.depth_valid_ratio for item in measurements])),
        object_points=int(np.median([item.object_points for item in measurements])),
        mask_clipped=any(item.mask_clipped for item in measurements),
        flags=tuple(sorted(flags)),
        method=latest.method,
        frames_considered=int(frames_considered if frames_considered is not None else len(measurements)),
        frames_accepted=len(measurements),
        dimension_std_mm=tuple(round(float(value), 3) for value in spreads),
    )


def aggregate_box_measurements(
    measurements: list[BoxVolumeMeasurement],
    *,
    frames_considered: int | None = None,
    dimension_instability_fraction: float = _DIMENSION_INSTABILITY_FRACTION,
) -> BoxVolumeMeasurement | None:
    """Track-based multi-frame aggregation, per the "Revised Dual-Camera
    Volume Estimation" recipe section 13: "Use the working track IDs to
    improve accuracy. Accept frames only after quality gates. Estimate
    L/W/H per accepted frame, aggregate dimensions using median, and
    calculate final volume once. Never sum per-frame volumes."

    `measurements` is the caller's own already-accepted set of per-frame
    `estimate_box_volume_cuboid()` results for one confirmed track (each one
    already passed that function's own internal validity checks -- this
    function does not re-derive geometry, only combines already-valid
    single-frame results). `frames_considered`, if given, is the wider
    count of frames the caller attempted for this track including ones that
    returned `None` or were otherwise dropped before reaching this function
    (defaults to `len(measurements)` when the caller does not track that).

    Returns `None` only if `measurements` is empty or the aggregated median
    dimensions are non-positive (should not happen given valid inputs, but
    this never fabricates a measurement from nothing).
    """
    if not measurements:
        return None

    lengths = np.asarray([m.length_mm for m in measurements], dtype=np.float64)
    widths = np.asarray([m.width_mm for m in measurements], dtype=np.float64)
    heights = np.asarray([m.height_mm for m in measurements], dtype=np.float64)

    length_mm = float(np.median(lengths))
    width_mm = float(np.median(widths))
    height_mm = float(np.median(heights))
    if length_mm <= 0 or width_mm <= 0 or height_mm <= 0:
        return None

    # V = L * W * H, computed once from the already-aggregated median
    # dimensions -- deliberately not a mean/median of each frame's own
    # already-multiplied volume_liters, and never a sum across frames.
    volume_liters = (length_mm / 1000.0) * (width_mm / 1000.0) * (height_mm / 1000.0) * 1000.0

    length_std = float(np.std(lengths))
    width_std = float(np.std(widths))
    height_std = float(np.std(heights))

    flags: set[str] = set()
    for measurement in measurements:
        flags.update(measurement.flags)

    # PDF section 13's quality-gate table: "Dimension spread too high over
    # accepted frames -> Return low confidence plus dimension_instability
    # flag." Relative spread (std / median) is used so the same fractional
    # threshold applies fairly to small and large boxes alike.
    relative_spreads = (
        length_std / max(length_mm, 1e-6),
        width_std / max(width_mm, 1e-6),
        height_std / max(height_mm, 1e-6),
    )
    unstable = max(relative_spreads) > dimension_instability_fraction
    if unstable:
        flags.add("dimension_instability")
    # Every accepted per-frame measurement already carries
    # "single_view_estimate" (estimate_box_volume_cuboid always sets it),
    # so the union above already preserves it -- this is just documenting
    # that the aggregate is still a single-camera-view estimate, not a
    # second, redundant assignment.

    confidence = float(np.median([measurement.volume_confidence for measurement in measurements]))
    if unstable:
        confidence *= 0.6
    confidence = float(np.clip(confidence, 0.0, 0.99))

    latest = measurements[-1]
    frames_accepted = len(measurements)

    return BoxVolumeMeasurement(
        volume_liters=volume_liters,
        volume_confidence=confidence,
        volume_method=latest.volume_method,
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        depth_valid_ratio=float(np.median([measurement.depth_valid_ratio for measurement in measurements])),
        object_points=int(np.median([measurement.object_points for measurement in measurements])),
        table_plane_inliers=latest.table_plane_inliers,
        table_plane_rmse_mm=float(np.median([measurement.table_plane_rmse_mm for measurement in measurements])),
        height_p98_mm=float(np.median([measurement.height_p98_mm for measurement in measurements])),
        height_top_median_mm=height_mm,
        mask_clipped=any(measurement.mask_clipped for measurement in measurements),
        flags=tuple(sorted(flags)),
        frames_considered=int(frames_considered if frames_considered is not None else frames_accepted),
        frames_accepted=frames_accepted,
        dimension_std_mm=(round(length_std, 3), round(width_std, 3), round(height_std, 3)),
        mesh_used_for_final_volume=False,
    )


def calibrate_monocular_depth(
    predicted_depth_m: np.ndarray | None,
    reference_depth_m: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
    minimum_samples: int = 100,
    maximum_samples: int = 50_000,
) -> DepthCalibration | None:
    """Fit a robust affine scale against aligned RealSense depth measurements."""
    if predicted_depth_m is None or reference_depth_m is None:
        return None
    if predicted_depth_m.shape != reference_depth_m.shape:
        return None

    valid = (
        np.isfinite(predicted_depth_m)
        & np.isfinite(reference_depth_m)
        & (predicted_depth_m > 0.05)
        & (reference_depth_m > 0.10)
        & (reference_depth_m < 10.0)
    )
    if mask is not None:
        if mask.shape != valid.shape:
            return None
        valid &= mask.astype(bool)

    prediction = predicted_depth_m[valid].astype(np.float64)
    reference = reference_depth_m[valid].astype(np.float64)
    if prediction.size < minimum_samples:
        return None
    if prediction.size > maximum_samples:
        indices = np.linspace(0, prediction.size - 1, maximum_samples, dtype=np.int64)
        prediction, reference = prediction[indices], reference[indices]

    prediction_range = float(np.ptp(prediction))
    reference_range = float(np.ptp(reference))
    if prediction_range < 1e-5:
        scale = 1.0
        offset = float(np.median(reference - prediction))
    elif reference_range < 1e-3:
        # A flat empty bin is common. Its nearly constant reference depth cannot
        # identify both affine parameters, so use a stable positive scale only.
        scale = float(np.median(reference / prediction))
        offset = 0.0
    else:
        design = np.column_stack((prediction, np.ones(prediction.size)))
        scale, offset = np.linalg.lstsq(design, reference, rcond=None)[0]
        residual = np.abs(reference - (scale * prediction + offset))
        cutoff = float(np.quantile(residual, 0.90))
        inliers = residual <= cutoff
        if int(np.count_nonzero(inliers)) >= minimum_samples:
            refined = np.column_stack((prediction[inliers], np.ones(np.count_nonzero(inliers))))
            scale, offset = np.linalg.lstsq(refined, reference[inliers], rcond=None)[0]
            prediction, reference = prediction[inliers], reference[inliers]

    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
        ratios = reference / prediction
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size < minimum_samples:
            return None
        scale, offset = float(np.median(ratios)), 0.0
    residual = reference - (scale * prediction + offset)
    return DepthCalibration(
        scale=float(scale),
        offset_m=float(offset),
        rmse_m=float(np.sqrt(np.mean(residual * residual))),
        sample_pixels=int(prediction.size),
    )
