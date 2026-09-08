"""Project one camera's depth into another camera's frame, and fuse the two.

Phase 1 of the dual-camera fusion system (see `calibration.py` for how the
`DualCameraCalibration` this module consumes is produced). This module is
pure geometry/array math with no camera or network I/O, so it is fully
testable with synthetic data -- and it is deliberately not wired into
`pipeline.py` yet. Fusing two independently-tracked, independently-baselined
camera streams into one combined measurement touches tracking, the ledger,
and the dashboard's per-camera assumptions in ways that need their own
careful, separately-tested integration pass; shipping the geometry core
first, with its own correctness tests, keeps that follow-up bounded and
reviewable rather than one large, hard-to-verify change.

Design (matches the requested "RealSense primary, Logitech fills
occlusions"): the RealSense is the one metric-calibrated stereo sensor in
this rig, so it is always the reference/primary frame. The Logitech's depth
(from its own source, e.g. Depth Anything V2) is reprojected into that
frame; wherever the RealSense already has valid depth, the RealSense value
wins unchanged; the Logitech-derived value is used only to fill pixels the
RealSense could not measure (its own invalid/occluded/out-of-range pixels).
"""

from __future__ import annotations

import numpy as np

from .calibration import DualCameraCalibration
from .types import CameraIntrinsics


def backproject_to_camera_space(
    depth_m: np.ndarray, intrinsics: CameraIntrinsics, valid: np.ndarray
) -> np.ndarray:
    """Pinhole-backproject a depth image's valid pixels to 3-D points in that camera's own frame.

    Returns an (N, 3) array, one row per True entry of `valid`, in the same
    row-major pixel order as `np.nonzero(valid)`.
    """
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def project_depth_to_reference_frame(
    depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    calibration: DualCameraCalibration,
    target_intrinsics: CameraIntrinsics,
    target_shape: tuple[int, int],
    *,
    min_depth_m: float = 0.10,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Reproject a secondary camera's depth image into the primary camera's pixel grid.

    Every valid source pixel is backprojected to a 3-D point in the source
    (secondary) camera's frame, rigidly transformed into the primary
    camera's frame via `calibration.rotation`/`calibration.translation`,
    then projected through `target_intrinsics` (a pinhole model, matching
    the rest of this project's volume math -- lens distortion is corrected
    upstream by the calibration itself, not modeled again here). Where two
    or more source points land on the same target pixel, the nearer one
    (smaller target-frame Z) is kept, since it is the one actually visible
    from the target camera's viewpoint -- a basic z-buffer.

    Returns a (height, width) float32 array in the target camera's pixel
    grid: `np.nan` at every pixel with no valid reprojected sample.
    """
    height, width = target_shape
    output = np.full((height, width), np.nan, dtype=np.float32)

    valid = np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    if not np.any(valid):
        return output

    source_points = backproject_to_camera_space(depth_m, source_intrinsics, valid)
    target_points = source_points @ calibration.rotation.T + calibration.translation

    target_z = target_points[:, 2]
    in_front = target_z > min_depth_m
    if not np.any(in_front):
        return output
    target_points = target_points[in_front]
    target_z = target_z[in_front]

    target_columns = target_intrinsics.fx * target_points[:, 0] / target_z + target_intrinsics.ppx
    target_rows = target_intrinsics.fy * target_points[:, 1] / target_z + target_intrinsics.ppy
    pixel_columns = np.round(target_columns).astype(np.int64)
    pixel_rows = np.round(target_rows).astype(np.int64)

    in_bounds = (
        (pixel_columns >= 0) & (pixel_columns < width) & (pixel_rows >= 0) & (pixel_rows < height)
    )
    pixel_columns = pixel_columns[in_bounds]
    pixel_rows = pixel_rows[in_bounds]
    target_z = target_z[in_bounds]

    # z-buffer: sort far-to-near so the final (overwriting) write at each
    # target pixel is always the nearest sample.
    order = np.argsort(-target_z)
    output[pixel_rows[order], pixel_columns[order]] = target_z[order].astype(np.float32)
    return output


def fuse_primary_with_secondary_fill(
    primary_depth_m: np.ndarray,
    secondary_projected_depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.10,
    max_depth_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the primary (RealSense) camera's invalid/occluded pixels from the reprojected secondary.

    Never overwrites a valid primary pixel -- the RealSense's own factory-
    calibrated stereo depth is always trusted over a reprojected monocular
    estimate wherever both exist. Returns `(fused_depth_m, filled_from_secondary_mask)`.
    """
    if primary_depth_m.shape != secondary_projected_depth_m.shape:
        raise ValueError("Primary and reprojected-secondary depth maps must be the same shape")

    primary_valid = (
        np.isfinite(primary_depth_m) & (primary_depth_m > min_depth_m) & (primary_depth_m < max_depth_m)
    )
    secondary_valid = np.isfinite(secondary_projected_depth_m) & (secondary_projected_depth_m > min_depth_m)

    fused = primary_depth_m.astype(np.float32, copy=True)
    fill_mask = (~primary_valid) & secondary_valid
    fused[fill_mask] = secondary_projected_depth_m[fill_mask]
    return fused, fill_mask
