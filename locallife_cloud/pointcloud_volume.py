"""RealSense point-cloud volume estimation (Open3D), per the dual-camera
recipe -- upgraded to the v3 "Zero-Flaw Implementation Blueprint" (the user's
`dual_camera_estimation_recipe_v3.pdf`), superseding the v1/v2-era text
recipe this module originally implemented.

RealSense depth is the sole source of volume (recipe v3 fusion rule, §8).
The pipeline per v3 §5 is, in order:

1. **Depth masking -> point cloud** (§5.1): depth pixels outside
   `[100mm, 2500mm]` are invalid (RealSense fills unmeasurable pixels with
   Z=0 -- never trust it); the object mask is eroded by 2-3 px first to
   reject the unreliable depth fringe at segmentation edges, *then* applied.
2. **Cleanup, in order** (§5.2): statistical outlier removal, table-plane
   RANSAC removal, then **pose normalization** -- rotate the whole cloud so
   the table's plane normal becomes world +Z and the table sits at world
   Z=0, storing the rotation matrix. Every measurement after this point
   happens in this Z-up, table-at-zero world frame, never in raw camera
   coordinates (v3's non-negotiable rule #5: "pose normalization happens
   before any measurement, never after").
3. **Boxes -> plane-fit + rectangular assumption** (§5.3): top-face RANSAC
   plane -> height above table; front-face RANSAC vertical plane + point
   projection -> width; third dimension from a second view if one was
   captured, else the front face's own max in-plane chord with confidence
   dampened and a `single_view_volume` flag (v3 rule #6: single-view is a
   fallback, never "accurate"). This is `estimate_volume_planefit_box()`,
   the new v3-mandated default for boxes.
4. **Bags -> heightmap + discrete approximation** (§5.4): grid the table-XY
   plane at ~2-3mm cells, take the max point height per cell, convex-fill
   empty interior cells from their neighbors, integrate, and discretize to
   the nearest expected size class unless the raw value falls far outside
   it. This is `estimate_volume_heightmap()`, extended with convex-fill and
   `discretize_bag_volume()`.

`estimate_volume_obb()` (the oriented-bounding-box method this module
originally shipped with) is kept, unchanged, specifically because v3 §13.3
calls for it as an explicit ablation baseline ("boxes: single-view vs
two-view vs raw-OBB (show OBB fails on hidden dimension)") -- it is no
longer the box default, but remains available for that comparison and for
any caller that still wants a fast, shape-agnostic fallback.

One deliberate, documented deviation from a literal reading of v3 §5.2:
table-plane RANSAC is **not** run on the object's own segmentation-masked
cloud when a segmentation-only mask is all that's available (see
`remove_support_plane`'s docstring, and `fit_floor_plane_from_baseline`,
both carried over unchanged from the prior recipe round) -- that was found,
via this project's own synthetic testing, to make RANSAC mistake a box's own
flat face for "the table" and strip it. `estimate_volume_recipe_v3()` below
still performs full §5.2 pose normalization, but sources the table plane
preferentially from an empty-scene baseline depth frame (every pixel of
which really is table/background) when one is available, falling back to
RANSAC-on-the-object-cloud only when no baseline was captured, in which case
callers should expect the plane fit to be less reliable and treat results
with the caution v3's own rule #8 asks for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .types import CameraIntrinsics

LOGGER = logging.getLogger("locallife_cloud.pointcloud_volume")

# A real bag/box on a bin floor is never literally millimeter-flat -- allow a
# generous band around the fitted supporting plane before a point counts as
# "the object" rather than "the floor/background it's sitting on".
_DEFAULT_PLANE_DISTANCE_THRESHOLD_M = 0.01
_DEFAULT_HEIGHTMAP_CELL_SIZE_M = 0.0025  # v3 §5.4: "grid the XY plane (cell ~2-3mm)"
_MIN_POINTS_FOR_MEASUREMENT = 30

# v3 §5.1's "Non-Negotiable Correctness Rule" #2/#1: depth outside this
# millimeter band is invalid (holes RealSense fills with 0, or noise past
# its reliable range) -- expressed here in meters since the rest of this
# project's depth arrays are already in meters by convention.
DEFAULT_VALID_DEPTH_RANGE_M = (0.10, 2.50)
DEFAULT_MASK_ERODE_PX = 3

# v3 §5.3: "Top face: points with z in the top ~5% envelope."
_TOP_FACE_Z_PERCENTILE = 95.0
# A candidate plane counts as "vertical" (a side face, not a second top-like
# face) when its normal is close enough to horizontal in the pose-normalized,
# Z-up frame.
_VERTICAL_NORMAL_MAX_Z_COMPONENT = 0.35


@dataclass(slots=True)
class PointCloudVolumeResult:
    liters: float
    method: str
    point_count: int
    confidence: float
    mean_height_m: float
    max_height_m: float
    footprint_area_m2: float
    tolerance_liters: float = 0.0
    views_used: int = 1
    flags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "liters": round(float(self.liters), 6),
            "method": self.method,
            "point_count": int(self.point_count),
            "confidence": round(float(self.confidence), 4),
            "mean_height_m": round(float(self.mean_height_m), 6),
            "max_height_m": round(float(self.max_height_m), 6),
            "footprint_area_m2": round(float(self.footprint_area_m2), 8),
            "tolerance_liters": round(float(self.tolerance_liters), 6),
            "views_used": int(self.views_used),
            "flags": list(self.flags),
        }


def _require_open3d():
    try:
        import open3d as o3d  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise RuntimeError(
            "open3d is required for point-cloud volume estimation "
            "(pip install open3d) -- see requirements-*.txt"
        ) from exc
    return o3d


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """v3 §5.1 rule #3: "Depth edges are noisy. Erode the object mask by
    2-3px before applying it to depth so you reject the unreliable depth
    fringe." `px<=0` is a no-op (kept for callers/tests that intentionally
    want the raw mask)."""
    if px <= 0:
        return mask.astype(bool)
    import cv2

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=int(px))
    return eroded.astype(bool)


def backproject_to_points(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray,
    *,
    mask_erode_px: int = DEFAULT_MASK_ERODE_PX,
    valid_range_m: tuple[float, float] = DEFAULT_VALID_DEPTH_RANGE_M,
) -> np.ndarray:
    """Backproject the masked, valid-depth pixels of a depth image to 3-D points.

    Uses the same pinhole convention as the rest of this project (`volume.py`):
    X = (u - ppx) * Z / fx, Y = (v - ppy) * Z / fy, Z = depth. Returns an
    (N, 3) float64 array in the camera's own metric frame (meters).

    Per v3 §5.1: `mask` is eroded by `mask_erode_px` before use (rejects the
    noisy depth fringe at segmentation edges), and only depth within
    `valid_range_m` counts -- RealSense fills unmeasurable pixels with
    Z=0, which must never be trusted as "the object is touching the lens".
    """
    if depth_m.shape != mask.shape:
        raise ValueError("depth_m and mask must have the same shape")
    eroded_mask = erode_mask(mask, mask_erode_px)
    low, high = valid_range_m
    valid = eroded_mask & np.isfinite(depth_m) & (depth_m > low) & (depth_m < high)
    rows, columns = np.where(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth_m[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)


def _make_point_cloud(points: np.ndarray):
    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def clean_point_cloud(points: np.ndarray, *, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal, per v3 §5.2 step 1 (`nb_neighbors=20,
    std_ratio=2.0` are the recipe's own defaults). Falls back to returning
    the input unchanged if there are too few points for the neighbor
    statistics to be meaningful (open3d itself would raise)."""
    if points.shape[0] < max(nb_neighbors * 2, _MIN_POINTS_FOR_MEASUREMENT):
        return points
    pcd = _make_point_cloud(points)
    cleaned, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(inlier_indices) == 0:
        return points
    return np.asarray(cleaned.points)


def remove_support_plane(
    points: np.ndarray,
    *,
    distance_threshold: float = _DEFAULT_PLANE_DISTANCE_THRESHOLD_M,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """RANSAC-fit and strip the dominant plane from a point cloud.

    Only meaningful on a point cloud that actually contains background/floor
    points alongside the object -- e.g. a raw ROI point cloud before object
    segmentation is applied. **Do not call this on a cloud that has already
    been restricted to an object's own segmentation mask**: a rigid box's own
    flat top or front face is frequently the single largest planar patch in
    such a cloud, so RANSAC would identify and strip the box's own face as
    "the floor" instead (this was verified directly -- see
    `estimate_volume_recipe`'s docstring for why the floor plane is instead
    fit from the empty-scene baseline depth image when one is available).
    Returns (remaining_points, plane_equation) -- plane_equation is (a, b, c,
    d) for ax+by+cz+d=0, or None if there were too few points to fit a plane.
    """
    if points.shape[0] < max(ransac_n * 3, _MIN_POINTS_FOR_MEASUREMENT):
        return points, None
    pcd = _make_point_cloud(points)
    try:
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
        )
    except RuntimeError:
        # open3d raises if it cannot fit any plane at all (e.g. too few
        # points or a fully degenerate point set) -- treat as "no plane found".
        return points, None
    inlier_set = set(inlier_indices)
    if len(inlier_set) >= points.shape[0]:
        # The whole cloud was classified as "the plane" -- nothing object-like
        # remains; hand back the original points rather than an empty result,
        # so a caller measuring a genuinely flat/thin object (a folded bag)
        # doesn't lose it entirely to an overzealous plane fit.
        return points, tuple(float(v) for v in plane_model)
    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[list(inlier_set)] = False
    return points[keep_mask], tuple(float(v) for v in plane_model)


def fit_floor_plane_from_baseline(
    baseline_depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    roi_mask: np.ndarray | None = None,
    *,
    max_points: int = 20000,
    seed: int = 0,
) -> tuple[float, float, float, float] | None:
    """RANSAC-fit the supporting (floor/bin-bottom) plane from the
    empty-scene baseline depth image, not the object's own masked points.

    This is the preferred way to get the table plane v3 §5.2 wants removed
    and normalized against: every pixel in an empty-scene baseline really is
    background by definition, so there is no risk of RANSAC mistaking one of
    the object's own flat faces for the table (the failure mode
    `remove_support_plane`'s docstring documents). `roi_mask`, if given,
    restricts the fit to the fixed measurement area (recommended -- avoids
    fitting to an unrelated plane elsewhere in a wide frame, e.g. a wall).
    """
    valid = np.isfinite(baseline_depth_m) & (baseline_depth_m > 0.05) & (baseline_depth_m < 10.0)
    if roi_mask is not None and roi_mask.shape == baseline_depth_m.shape:
        valid &= roi_mask.astype(bool)
    rows, cols = np.where(valid)
    if rows.size < _MIN_POINTS_FOR_MEASUREMENT:
        return None
    if rows.size > max_points:
        chosen = np.random.default_rng(seed).choice(rows.size, size=max_points, replace=False)
        rows, cols = rows[chosen], cols[chosen]
    z = baseline_depth_m[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
    points = np.stack([x, y, z], axis=1)
    pcd = _make_point_cloud(points)
    try:
        plane_model, _inliers = pcd.segment_plane(
            distance_threshold=_DEFAULT_PLANE_DISTANCE_THRESHOLD_M, ransac_n=3, num_iterations=1000
        )
    except RuntimeError:
        return None
    return tuple(float(v) for v in plane_model)


def _pca_extents(points: np.ndarray) -> np.ndarray:
    """Extent of `points` along its own 3 principal axes (ascending order),
    via numpy eigendecomposition -- no qhull, so it never throws on a
    degenerate/flat point set the way `get_oriented_bounding_box()` can."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    projected = centered @ eigenvectors
    return np.sort(projected.max(axis=0) - projected.min(axis=0))


def estimate_volume_obb(points: np.ndarray, *, min_thickness_m: float = 0.02) -> PointCloudVolumeResult:
    """Oriented-bounding-box volume. Kept as v3 §13.3's own explicit ablation
    baseline ("raw-OBB ... show OBB fails on hidden dimension") -- no longer
    the box default (see `estimate_volume_planefit_box` for that), but a
    fast, shape-agnostic method any caller can still reach for directly.

    A single depth-camera view of a box only ever sees 1-2 of its 3 faces.
    When only ONE flat face is visible -- no front-face points at all -- the
    point cloud is mathematically flat, and open3d's qhull-based OBB cannot
    be built from it (raises `RuntimeError`, verified directly). That case
    falls back to a PCA-aligned bounding box with its thinnest axis floored
    at `min_thickness_m` (2 cm default), so a single visible face still
    returns a defensible, clearly-lower-confidence volume instead of
    crashing or silently reporting 0 L.
    """
    if points.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        return PointCloudVolumeResult(0.0, "oriented-bounding-box", int(points.shape[0]), 0.0, 0.0, 0.0, 0.0)

    heights = points[:, 2]
    degenerate = False
    extent: np.ndarray
    try:
        pcd = _make_point_cloud(points)
        obb = pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent)
        if float(np.min(extent)) < 1e-6:
            degenerate = True
    except RuntimeError:
        degenerate = True
        extent = np.zeros(3)

    if degenerate:
        extent = _pca_extents(points)
        extent[0] = max(float(extent[0]), min_thickness_m)

    volume_m3 = float(np.prod(extent))
    liters = max(0.0, volume_m3 * 1000.0)
    sorted_extent = np.sort(extent)
    footprint_area_m2 = float(sorted_extent[1] * sorted_extent[2])

    base_confidence = float(np.clip(points.shape[0] / 2000.0, 0.05, 0.98))
    confidence = base_confidence * (0.5 if degenerate else 1.0)

    return PointCloudVolumeResult(
        liters=liters,
        method="oriented-bounding-box",
        point_count=int(points.shape[0]),
        confidence=confidence,
        mean_height_m=float(np.mean(heights)) if heights.size else 0.0,
        max_height_m=float(np.max(heights)) if heights.size else 0.0,
        footprint_area_m2=footprint_area_m2,
        views_used=1,
        flags=("single_view_volume",) if not degenerate else ("single_view_volume", "degenerate_obb_fallback"),
    )


def _rotation_rows_for_normal(normal: np.ndarray) -> np.ndarray:
    """A 3x3 rotation matrix R whose third ROW is `normal` (unit vector) --
    i.e. `(R @ p)[2] == normal . p` for any point p, and R's other two rows
    are an arbitrary orthonormal basis of the plane orthogonal to `normal`.
    This is exactly what pose normalization needs (rotate so the table
    normal becomes +Z) without any Rodrigues-formula edge cases."""
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(arbitrary, n)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    return np.stack([u, v, n], axis=0)


def normalize_pose(
    points: np.ndarray,
    plane_equation: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """v3 §5.2 step 3: "fit the table plane ax+by+cz+d=0; rotate the whole
    cloud so that plane normal -> (0,0,1). Store the rotation matrix. Now
    Z-up and the table is XY at z~=0."

    The plane normal is oriented (flipped if needed) so the point cloud's
    own centroid ends up on the positive-Z side after the transform -- i.e.
    "above the table", not below it. Returns `(rotated_points, R,
    z_shift)`; `R` is the stored rotation matrix (v3's own instruction), and
    `z_shift` is the additional translation along the new Z axis that put
    the table at z=0 (kept separately so a caller can reconstruct the full
    rigid transform if needed, e.g. for a second view's extrinsics).
    """
    a, b, c, d = plane_equation
    normal = np.array([a, b, c], dtype=np.float64)
    norm_length = float(np.linalg.norm(normal))
    if norm_length < 1e-9:
        return points.copy(), np.eye(3), 0.0
    normal = normal / norm_length
    d = d / norm_length

    point_on_plane = -d * normal
    reference = points.mean(axis=0) if points.shape[0] else point_on_plane
    if np.dot(normal, reference - point_on_plane) < 0:
        normal = -normal
        d = -d

    rotation = _rotation_rows_for_normal(normal)
    rotated = points @ rotation.T
    if rotated.shape[0]:
        rotated[:, 2] += d
    return rotated, rotation, float(d)


def _fit_top_face(points: np.ndarray) -> tuple[float, np.ndarray | None]:
    """v3 §5.3: points with z in the top ~5% envelope; RANSAC-fit a plane;
    height = plane_z (distance above the table, which sits at z=0 after
    `normalize_pose`). Returns (height_m, top_face_inlier_points)."""
    if points.shape[0] == 0:
        return 0.0, None
    z_cutoff = float(np.percentile(points[:, 2], _TOP_FACE_Z_PERCENTILE))
    top_candidates = points[points[:, 2] >= z_cutoff]
    if top_candidates.shape[0] < max(10, _MIN_POINTS_FOR_MEASUREMENT // 3):
        # Not enough points in the thin top envelope to RANSAC a plane --
        # fall back to the cloud's own max height directly.
        return float(np.max(points[:, 2])), top_candidates

    _remaining, plane_equation = remove_support_plane(
        top_candidates, distance_threshold=_DEFAULT_PLANE_DISTANCE_THRESHOLD_M
    )
    if plane_equation is not None:
        a, b, c, plane_d = plane_equation
        n = np.array([a, b, c])
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-9:
            # Distance of the table origin's own Z axis (x=y=0) to the
            # plane, i.e. the plane's height directly above the table.
            height = float(-(plane_d / n_norm) / (c / n_norm)) if abs(c / n_norm) > 1e-6 else float(np.median(top_candidates[:, 2]))
            if height > 0:
                return height, top_candidates
    return float(np.median(top_candidates[:, 2])), top_candidates


def _fit_front_face(
    points: np.ndarray,
    top_face_points: np.ndarray | None,
) -> tuple[float, np.ndarray | None, tuple[float, float, float, float] | None]:
    """v3 §5.3: "Front face: take points on the most visible vertical face;
    fit a vertical plane; project all points onto it and take the span of
    projections along the face's horizontal axis -> width."

    "Most visible" is approximated here as "the largest vertical-normal
    planar patch left once the top face is excluded" -- a single masked
    object view usually shows exactly one dominant side face, so the
    largest-inlier vertical plane and the camera-facing one coincide in
    practice; this is a documented simplification of "most visible" (true
    camera-facing disambiguation would need the pre-normalization camera
    pose carried alongside every point, which single-view callers may not
    retain). Returns (width_m, front_face_inlier_points, plane_equation).
    """
    if points.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        return 0.0, None, None

    if top_face_points is not None and top_face_points.shape[0]:
        top_z_min = float(np.min(top_face_points[:, 2]))
        side_candidates = points[points[:, 2] < top_z_min - 1e-6]
    else:
        side_candidates = points

    if side_candidates.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        side_candidates = points

    remaining = side_candidates
    plane_equation = None
    for _attempt in range(2):
        candidate_remaining, candidate_plane = remove_support_plane(
            remaining, distance_threshold=_DEFAULT_PLANE_DISTANCE_THRESHOLD_M
        )
        if candidate_plane is None:
            break
        a, b, c, _d = candidate_plane
        normal = np.array([a, b, c])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-9:
            break
        z_component = abs(normal[2] / normal_norm)
        if z_component <= _VERTICAL_NORMAL_MAX_Z_COMPONENT:
            plane_equation = candidate_plane
            inlier_count = remaining.shape[0] - candidate_remaining.shape[0]
            face_points = remaining[:inlier_count] if inlier_count > 0 else remaining
            # Recover the actual inlier points (remove_support_plane only
            # hands back the *remainder*) by re-fitting the distance test
            # directly against the discovered plane.
            a, b, c, d = plane_equation
            n = np.array([a, b, c])
            n_norm = float(np.linalg.norm(n))
            distances = np.abs(remaining @ n + d) / max(n_norm, 1e-9)
            face_points = remaining[distances <= _DEFAULT_PLANE_DISTANCE_THRESHOLD_M * 1.5]
            break
        # That plane was near-horizontal (another top-like patch) -- drop
        # its inliers and try again on what's left, once.
        remaining = candidate_remaining
    else:
        face_points = None

    if plane_equation is None or face_points is None or face_points.shape[0] < 10:
        return 0.0, None, None

    a, b, c, d = plane_equation
    normal = np.array([a, b, c])
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    # Horizontal axis within the (near-)vertical face: cross the face normal
    # with world-up: since the face is vertical, this is close to horizontal
    # by construction and gives a stable, table-frame-relative "width" axis.
    world_up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(normal, world_up)
    horizontal_norm = float(np.linalg.norm(horizontal_axis))
    if horizontal_norm < 1e-6:
        return 0.0, face_points, plane_equation
    horizontal_axis /= horizontal_norm

    projections = face_points @ horizontal_axis
    width = float(np.max(projections) - np.min(projections))
    return width, face_points, plane_equation


def _max_in_plane_chord(face_points: np.ndarray, plane_equation: tuple[float, float, float, float]) -> float:
    """v3 §5.3 single-view depth fallback: "derive depth from the max
    in-plane chord of the front face." Projects the face's own points into
    its 2-D plane coordinates and returns the largest pairwise distance
    (the diagonal span of the visible face) as a depth proxy. Subsampled for
    large point counts since this is an O(n^2) pairwise-distance search."""
    if face_points.shape[0] < 2:
        return 0.0
    a, b, c, _d = plane_equation
    normal = np.array([a, b, c])
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(normal, arbitrary)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-9)
    basis_v = np.cross(normal, basis_u)

    points_2d = np.stack([face_points @ basis_u, face_points @ basis_v], axis=1)
    if points_2d.shape[0] > 800:
        chosen = np.random.default_rng(0).choice(points_2d.shape[0], size=800, replace=False)
        points_2d = points_2d[chosen]

    # Convex-hull extreme points bound the max pairwise distance far more
    # cheaply than a full O(n^2) search when there are many points; fall
    # back to full pairwise search for small point sets.
    try:
        from scipy.spatial import ConvexHull  # type: ignore

        hull = ConvexHull(points_2d)
        candidates = points_2d[hull.vertices]
    except Exception:
        candidates = points_2d

    diffs = candidates[:, None, :] - candidates[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=-1))
    return float(np.max(distances)) if distances.size else 0.0


def estimate_volume_planefit_box(
    points_view1: np.ndarray,
    *,
    points_view2: np.ndarray | None = None,
    single_view_confidence_scale: float = 0.6,
) -> PointCloudVolumeResult:
    """v3 §5.3: box volume via plane-fit + rectangular assumption. `points_*`
    must already be pose-normalized (`normalize_pose`) -- Z-up, table at
    z~=0 -- and cleaned (SOR). height x width x depth, liters = product*1000.

    `points_view2`, if given, is a second pose-normalized view of the same
    object (recipe §3.4: rotate the object ~90 deg or walk the camera ~90
    deg between captures) -- its own front-face width becomes the box's
    depth dimension directly (v3: "Two views available: fit the
    corresponding face in the second view -> depth"). Without a second view,
    depth is estimated from view 1's own front face max in-plane chord, and
    v3's rule #6 applies: `volume_confidence *= 0.6` and a
    `single_view_volume` flag is set -- this is reported as *estimated*, not
    *measured*.
    """
    if points_view1.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        return PointCloudVolumeResult(0.0, "plane-fit-box", int(points_view1.shape[0]), 0.0, 0.0, 0.0, 0.0)

    height_m, top_face_points = _fit_top_face(points_view1)
    width_m, front_face_points, front_plane = _fit_front_face(points_view1, top_face_points)

    flags: list[str] = []
    views_used = 1

    if points_view2 is not None and points_view2.shape[0] >= _MIN_POINTS_FOR_MEASUREMENT:
        height2_m, top_face2 = _fit_top_face(points_view2)
        depth_m_dim, _face2, plane2 = _fit_front_face(points_view2, top_face2)
        if depth_m_dim > 0:
            views_used = 2
            # v3 edge-case matrix: "Two views disagree strongly -> report
            # view with best plane residual; flag." A >25% relative height
            # disagreement between the two independent top-face fits is
            # treated as "disagree strongly".
            if height_m > 1e-6 and abs(height2_m - height_m) / height_m > 0.25:
                flags.append("views_disagree")
                # Prefer whichever view's front-face plane fit more points
                # (a cheap, available proxy for "best plane residual" -- the
                # RANSAC calls above don't currently return per-point
                # residuals, only inlier membership).
                if front_face_points is not None and (
                    _face2 is None or front_face_points.shape[0] >= _face2.shape[0]
                ):
                    height_m = height_m
                else:
                    height_m = height2_m
        else:
            depth_m_dim = _max_in_plane_chord(front_face_points, front_plane) if front_face_points is not None and front_plane is not None else 0.0
            flags.append("single_view_volume")
    elif front_face_points is not None and front_plane is not None:
        depth_m_dim = _max_in_plane_chord(front_face_points, front_plane)
        flags.append("single_view_volume")
    else:
        depth_m_dim = 0.0
        flags.append("single_view_volume")

    volume_m3 = max(0.0, height_m) * max(0.0, width_m) * max(0.0, depth_m_dim)
    liters = volume_m3 * 1000.0

    point_count = int(points_view1.shape[0]) + int(points_view2.shape[0] if points_view2 is not None else 0)
    base_confidence = float(np.clip(point_count / 2500.0, 0.05, 0.97))
    confidence = base_confidence * (single_view_confidence_scale if views_used == 1 else 1.0)
    if "views_disagree" in flags:
        confidence *= 0.75
    if height_m <= 0 or width_m <= 0 or depth_m_dim <= 0:
        confidence *= 0.3
        flags.append("incomplete_face_fit")

    heights = points_view1[:, 2]
    return PointCloudVolumeResult(
        liters=max(0.0, liters),
        method="plane-fit-box",
        point_count=point_count,
        confidence=float(np.clip(confidence, 0.0, 0.99)),
        mean_height_m=float(np.mean(heights)) if heights.size else 0.0,
        max_height_m=float(np.max(heights)) if heights.size else 0.0,
        footprint_area_m2=float(max(0.0, width_m) * max(0.0, depth_m_dim)),
        views_used=views_used,
        flags=tuple(flags),
    )


def _exterior_mask(occupied: np.ndarray) -> np.ndarray:
    """Every empty cell reachable from the grid's own border without
    crossing an occupied cell -- i.e. background outside the object's
    footprint, not a hole inside it. A plain BFS flood-fill from the border
    (4-connected); no scipy dependency needed for this."""
    height, width = occupied.shape
    exterior = np.zeros_like(occupied, dtype=bool)
    from collections import deque

    frontier: deque[tuple[int, int]] = deque()
    for c in range(width):
        for r in (0, height - 1):
            if not occupied[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                frontier.append((r, c))
    for r in range(height):
        for c in (0, width - 1):
            if not occupied[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                frontier.append((r, c))

    while frontier:
        r, c = frontier.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and not occupied[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                frontier.append((rr, cc))
    return exterior


def _convex_fill_heightmap(grid: np.ndarray, *, max_passes: int = 40) -> np.ndarray:
    """v3 §5.4 step 2: "Convex-fill the heightmap: any INTERIOR cell with no
    depth gets the average of its neighbors (fills the far/side holes from
    single view) -- improves the undercount."

    `grid` has NaN for empty cells, over a bounding grid that is generally
    larger than the object's own circular/irregular footprint -- the empty
    cells in its four corners are real background, not "holes", and must
    never be filled (that would silently inflate the footprint area to the
    whole bounding box, badly overestimating volume for anything that
    isn't already a perfect square). `_exterior_mask` tells the two apart
    via a border flood-fill; only genuine enclosed gaps get filled here, via
    repeated 8-neighbor averaging (any 1 already-filled neighbor seeds a
    cell, so the fill grows inward ring by ring across a whole run of empty
    cells rather than needing every gap already surrounded) run to
    convergence (bounded by `max_passes` as a safety cap, not a target).
    """
    occupied = ~np.isnan(grid)
    if not occupied.any():
        return grid.copy()
    # A sparse single-view point cloud (distant object, or a grid finer than
    # the sensor's real spatial resolution at range) can leave the object's
    # occupied cells too porous for a strict border flood-fill: gaps of a
    # cell or two between real points let "exterior" leak straight through
    # to the center, misclassifying the whole interior as background (found
    # via this module's own synthetic testing) and defeating the fill
    # entirely. A light dilation of the occupied mask closes those small
    # gaps for the purpose of finding the footprint's outline only -- the
    # actual heights summed below always come from `grid` itself (dilation
    # never invents a height value, only decides interior vs. exterior).
    import cv2

    dilated_occupied = cv2.dilate(
        occupied.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=2
    ).astype(bool)
    exterior = _exterior_mask(dilated_occupied)
    interior_hole = np.isnan(grid) & ~exterior

    filled = grid.copy()
    height, width = filled.shape
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _pass in range(max_passes):
        remaining_rows, remaining_cols = np.where(interior_hole & np.isnan(filled))
        if remaining_rows.size == 0:
            break
        updated = False
        next_grid = filled.copy()
        for r, c in zip(remaining_rows, remaining_cols):
            neighbor_values = []
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and not np.isnan(filled[rr, cc]):
                    neighbor_values.append(filled[rr, cc])
            if len(neighbor_values) >= 1:
                next_grid[r, c] = float(np.mean(neighbor_values))
                updated = True
        filled = next_grid
        if not updated:
            break
    return filled


def discretize_bag_volume(
    liters: float,
    *,
    class_sizes_l: tuple[float, ...] = (5.0, 10.0),
    window_frac: float = 0.30,
) -> tuple[float, bool]:
    """v3 §5.4 step 4: "Discretize to nearest expected class {5,10} unless
    integrated value <70% or >130% of nearest class -- else report the raw
    value and lower confidence." Returns `(reported_liters, was_discretized)`.
    """
    if not class_sizes_l or liters <= 0:
        return liters, False
    nearest = min(class_sizes_l, key=lambda size: abs(size - liters))
    low, high = nearest * (1.0 - window_frac), nearest * (1.0 + window_frac)
    if low <= liters <= high:
        return float(nearest), True
    return float(liters), False


def bag_repeatability_stats(liters_samples: list[float]) -> dict:
    """v3 §5.4 step 5 / §13.5: "capture the same bag 8-12x, report mean +-
    std liters. The std is part of the honest result." A thin offline/
    validation helper -- not called from the single-frame pipeline, since a
    single API/dashboard call only ever has one capture to work with; a
    caller doing the recipe's own repeatability study runs this over its own
    collected samples.
    """
    values = np.asarray([v for v in liters_samples if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {"mean_liters": 0.0, "std_liters": 0.0, "n": 0}
    return {
        "mean_liters": round(float(np.mean(values)), 4),
        "std_liters": round(float(np.std(values)), 4),
        "n": int(values.size),
    }


def estimate_volume_heightmap(
    points: np.ndarray,
    *,
    cell_size_m: float = _DEFAULT_HEIGHTMAP_CELL_SIZE_M,
    plane_equation: tuple[float, float, float, float] | None = None,
    class_sizes_l: tuple[float, ...] = (5.0, 10.0),
    tolerance_frac: float = 0.15,
    discretize_window_frac: float = 0.30,
    confidence_cap: tuple[float, float] = (0.5, 0.7),
) -> PointCloudVolumeResult:
    """Heightmap / grid-integration volume, for bags and other soft objects
    (v3 §5.4). Bins the footprint into `cell_size_m` grid cells, takes the
    tallest point per cell, **convex-fills empty interior cells** from their
    neighbors, integrates cell_area * cell_height, discretizes to the
    nearest expected size class when close enough, and always reports a
    `tolerance_liters` band -- bags are inherently non-rigid (v3 rule #7),
    so confidence is capped at `confidence_cap` regardless of point density.
    """
    if points.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        return PointCloudVolumeResult(0.0, "heightmap-grid-integration", int(points.shape[0]), 0.0, 0.0, 0.0, 0.0)

    if plane_equation is None:
        _, plane_equation = remove_support_plane(points)

    if plane_equation is None:
        normal = np.array([0.0, 0.0, 1.0])
        d = -float(np.percentile(points[:, 2], 5))
    else:
        a, b, c, d = plane_equation
        normal = np.array([a, b, c], dtype=np.float64)
        norm_length = float(np.linalg.norm(normal))
        if norm_length < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
            norm_length = 1.0
        normal = normal / norm_length
        d = d / norm_length

    heights = points @ normal + d
    if np.median(heights) < 0:
        normal = -normal
        d = -d
        heights = points @ normal + d
    heights = np.clip(heights, 0.0, None)

    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(normal, arbitrary)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-9)
    basis_v = np.cross(normal, basis_u)

    u = points @ basis_u
    v = points @ basis_v

    u_min, v_min = u.min(), v.min()
    u_bins = np.floor((u - u_min) / cell_size_m).astype(np.int64)
    v_bins = np.floor((v - v_min) / cell_size_m).astype(np.int64)
    grid_w = int(u_bins.max()) + 1
    grid_h = int(v_bins.max()) + 1

    # Cap the grid resolution: a very large object/very small cell could
    # otherwise build an unreasonably large dense array. 400x400 cells at
    # the 2.5mm default already covers a 1m x 1m footprint.
    max_cells_per_axis = 400
    if grid_w > max_cells_per_axis or grid_h > max_cells_per_axis:
        scale = max(grid_w, grid_h) / max_cells_per_axis
        cell_size_m = cell_size_m * scale
        u_bins = np.floor((u - u_min) / cell_size_m).astype(np.int64)
        v_bins = np.floor((v - v_min) / cell_size_m).astype(np.int64)
        grid_w = int(u_bins.max()) + 1
        grid_h = int(v_bins.max()) + 1

    # np.maximum.at against a NaN-seeded grid is a no-op everywhere -- IEEE
    # 754 defines max(nan, x) as nan, so every cell would stay nan forever.
    # Seed with -inf instead (a true "no data yet" sentinel for a max-reduce)
    # and convert untouched cells back to nan once the reduce is done.
    grid = np.full((grid_h, grid_w), -np.inf, dtype=np.float64)
    np.maximum.at(grid, (v_bins, u_bins), heights)
    grid[np.isneginf(grid)] = np.nan
    occupied_before_fill = int(np.count_nonzero(~np.isnan(grid)))

    filled_grid = _convex_fill_heightmap(grid)

    cell_area_m2 = cell_size_m * cell_size_m
    valid_cells = ~np.isnan(filled_grid)
    volume_m3 = float(np.nansum(filled_grid[valid_cells]) * cell_area_m2)
    raw_liters = max(0.0, volume_m3 * 1000.0)

    reported_liters, was_discretized = discretize_bag_volume(
        raw_liters, class_sizes_l=class_sizes_l, window_frac=discretize_window_frac
    )

    flags: list[str] = ["bag_discrete_approx"] if not was_discretized else []

    base_confidence = float(np.clip(points.shape[0] / 3000.0, confidence_cap[0], confidence_cap[1]))
    if not was_discretized:
        base_confidence *= 0.85

    tolerance_liters = reported_liters * tolerance_frac

    return PointCloudVolumeResult(
        liters=reported_liters,
        method="heightmap-grid-integration",
        point_count=int(points.shape[0]),
        confidence=float(np.clip(base_confidence, confidence_cap[0] * 0.5, confidence_cap[1])),
        mean_height_m=float(np.mean(heights)) if heights.size else 0.0,
        max_height_m=float(np.max(heights)) if heights.size else 0.0,
        footprint_area_m2=float(int(np.count_nonzero(valid_cells)) * cell_area_m2),
        tolerance_liters=tolerance_liters,
        views_used=1,
        flags=tuple(flags),
    )


def estimate_volume_recipe(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray,
    *,
    object_type: str = "box",
    baseline_depth_m: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    mask_erode_px: int = DEFAULT_MASK_ERODE_PX,
    valid_range_m: tuple[float, float] = DEFAULT_VALID_DEPTH_RANGE_M,
    depth_m_view2: np.ndarray | None = None,
    intrinsics_view2: CameraIntrinsics | None = None,
    mask_view2: np.ndarray | None = None,
) -> PointCloudVolumeResult:
    """End-to-end v3 §5: mask -> point cloud -> clean -> pose-normalize ->
    volume. `object_type` is `"box"` (plane-fit volume, v3 §5.3) or `"bag"`
    (heightmap/grid-integration volume, v3 §5.4) -- any other value is
    treated as `"bag"`, the more conservative/general-purpose of the two.

    `mask` must already be the OBJECT's own segmentation mask (from
    `recipe_detect.py`), not a wider ROI. The table plane used for pose
    normalization is fit from `baseline_depth_m` (the empty-scene depth
    image) when given -- every pixel of which really is background by
    definition -- restricted to `roi_mask` if given; without a baseline it
    falls back to RANSAC on the object-masked cloud itself (see this
    module's own top-level docstring for why that fallback is less
    reliable). `depth_m_view2`/`intrinsics_view2`/`mask_view2`, if all
    given, are a second RealSense capture of the same box from a ~90-degree
    rotated view (v3 §3.4) and feed the box path's two-view depth dimension.
    """
    points = backproject_to_points(
        depth_m, intrinsics, mask, mask_erode_px=mask_erode_px, valid_range_m=valid_range_m
    )
    if points.shape[0] < _MIN_POINTS_FOR_MEASUREMENT:
        method = "plane-fit-box" if object_type == "box" else "heightmap-grid-integration"
        return PointCloudVolumeResult(0.0, method, int(points.shape[0]), 0.0, 0.0, 0.0, 0.0)

    points = clean_point_cloud(points)

    plane_equation = None
    if baseline_depth_m is not None:
        plane_equation = fit_floor_plane_from_baseline(baseline_depth_m, intrinsics, roi_mask)
    if plane_equation is None:
        _, plane_equation = remove_support_plane(points)

    if object_type == "box":
        if plane_equation is not None:
            normalized_points, _rotation, _shift = normalize_pose(points, plane_equation)
        else:
            normalized_points = points

        normalized_view2 = None
        if depth_m_view2 is not None and intrinsics_view2 is not None and mask_view2 is not None:
            points_view2 = backproject_to_points(
                depth_m_view2, intrinsics_view2, mask_view2,
                mask_erode_px=mask_erode_px, valid_range_m=valid_range_m,
            )
            if points_view2.shape[0] >= _MIN_POINTS_FOR_MEASUREMENT:
                points_view2 = clean_point_cloud(points_view2)
                plane_equation_2 = plane_equation
                if plane_equation_2 is not None:
                    normalized_view2, _r2, _s2 = normalize_pose(points_view2, plane_equation_2)
                else:
                    normalized_view2 = points_view2

        return estimate_volume_planefit_box(normalized_points, points_view2=normalized_view2)

    # Bag path (heightmap): normalize pose too, when a plane is known, so
    # the grid is measured in the table's own XY frame at any camera tilt.
    if plane_equation is not None:
        normalized_points, _rotation, _shift = normalize_pose(points, plane_equation)
        # After normalization the table is exactly the world XY plane at
        # z=0, so the heightmap grid can integrate directly against
        # world-Z (0,0,1,0) rather than refitting a plane on the object
        # cloud alone (the very fallback this module avoids for boxes).
        return estimate_volume_heightmap(normalized_points, plane_equation=(0.0, 0.0, 1.0, 0.0))
    return estimate_volume_heightmap(points, plane_equation=None)
