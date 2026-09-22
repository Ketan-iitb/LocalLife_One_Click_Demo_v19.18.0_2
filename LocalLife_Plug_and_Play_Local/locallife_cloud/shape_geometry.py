"""Shape-aware volume routing: cuboid, cylinder, irregular, or uncertain.

Why this exists
---------------
Every object used to be reported as a support-plane footprint rectangle times
its height. That is the right answer for a shoe box and the wrong one for a
can: a 6 cm can standing 12 cm tall has a 6 x 6 x 12 cm bounding box (0.432 L)
but occupies pi * 3^2 * 12 = 0.339 L. For a backpack neither number is the
occupied volume; the height-map integral over its mask is.

The method is chosen from what the *points* look like, never from the
detector's label alone:

* cuboid           -- the footprint fills its minimum-area rectangle and the
                      top is a flat plateau;
* cylinder         -- the footprint is a disc (upright), a reconstructed arc
                      (upright, seen from the side), or a filled rectangle
                      whose height profile across the short side is a
                      half-circle as tall as it is wide (lying down);
* irregular_rigid /
  flexible_or_unknown -- anything else with a valid height map: its volume is
                      the segmented height-map integral. The label only picks
                      the wording (a flexible bag's volume is its *current
                      external occupied volume*), not the formula;
* uncertain        -- too few points, or implausible dimensions.

Bounding-box, shape and mesh/height-map volumes are different quantities and
are always reported separately; `selected_volume_litres` says which one the
method stands behind.

The router is per-frame. `GeometryLock` turns a sequence of per-frame results
into one frozen answer per track: the method must repeat for several frames
before it is accepted, and once accepted neither the method nor the numbers
change again.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from .footprint import _column_fill_ratio, _fit_circle

CUBOID = "cuboid"
CYLINDER = "cylinder"
IRREGULAR_RIGID = "irregular_rigid"
FLEXIBLE_OR_UNKNOWN = "flexible_or_unknown"
UNCERTAIN = "uncertain"
GEOMETRY_METHODS = (CUBOID, CYLINDER, IRREGULAR_RIGID, FLEXIBLE_OR_UNKNOWN)

# Label words that only change how an irregular volume is *described*. They
# never select cuboid or cylinder: that needs the geometry to agree.
# Accepted cylindrical categories. A label only makes the object a cylinder
# *candidate* -- the robust circle fit must still succeed on its points.
_CYLINDER_WORDS = {
    "can", "cans", "tin", "bottle", "bottles", "jar", "cup", "mug", "tube", "canister",
    "flask", "cream", "balm", "lotion", "container", "cylinder", "cylindrical", "soda", "beverage",
}
_FLEXIBLE_WORDS = {"bag", "bags", "sack", "pillow", "cushion", "backpack", "textile", "cloth", "toy", "plush"}

MIN_POINTS = 50
MIN_DIMENSION_MM = 5.0
MAX_DIMENSION_MM = 1500.0

# A disc fills pi/4 = 0.785 of its bounding square, a rectangle fills ~1.
DISC_FILL_RANGE = (0.68, 0.87)
RECT_FILL_MIN = 0.88
MAX_CIRCLE_RESIDUAL = 0.07
FLAT_TOP_MIN = 0.40
# Edge-to-centre height ratio across the short side below which the top is curved.
CURVED_PROFILE_MAX = 0.85
# Robust cylinder acceptance: RMS radial residual / radius, arc coverage, and
# the combined fit confidence below which a cylinder candidate stays pending.
MAX_CYLINDER_RESIDUAL = 0.10
CYLINDER_NOISE_FLOOR_M = 0.005
MIN_ARC_COVERAGE_DEG = 110.0
MIN_CYLINDER_FIT_CONFIDENCE = 0.35
# Reasons a cylinder candidate is held pending instead of measured.
CYLINDER_REJECTIONS = frozenset({
    "cylinder_fit_residual_too_high", "insufficient_arc_coverage",
    "cylinder_fit_low_confidence", "implausible_cylinder_diameter",
})


@dataclass(frozen=True)
class ShapeGeometry:
    geometry_method: str
    geometry_confidence: float
    length_mm: float | None
    width_mm: float | None
    height_mm: float | None
    bounding_box_volume_litres: float | None
    mesh_volume_litres: float | None
    selected_volume_litres: float | None
    volume_meaning: str
    cylinder_diameter_mm: float | None = None
    cylinder_height_mm: float | None = None
    cylinder_fit_residual: float | None = None
    cylinder_volume_litres: float | None = None
    cylinder_orientation: str | None = None
    radius_mm: float | None = None
    fit_confidence: float | None = None
    rejection_reason: str | None = None
    points: int = 0
    frames: int = 1
    frozen: bool = False
    flags: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _round(value: float | None, digits: int) -> float | None:
            return None if value is None else round(float(value), digits)

        return {
            "geometry_method": self.geometry_method,
            "geometry_confidence": round(float(self.geometry_confidence), 4),
            "length_mm": _round(self.length_mm, 2),
            "width_mm": _round(self.width_mm, 2),
            "height_mm": _round(self.height_mm, 2),
            "bounding_box_volume_litres": _round(self.bounding_box_volume_litres, 6),
            "mesh_volume_litres": _round(self.mesh_volume_litres, 6),
            "selected_volume_litres": _round(self.selected_volume_litres, 6),
            "volume_meaning": self.volume_meaning,
            "cylinder_diameter_mm": _round(self.cylinder_diameter_mm, 2),
            "cylinder_height_mm": _round(self.cylinder_height_mm, 2),
            "cylinder_fit_residual": _round(self.cylinder_fit_residual, 5),
            "cylinder_volume_litres": _round(self.cylinder_volume_litres, 6),
            "cylinder_orientation": self.cylinder_orientation,
            # Flat aliases the dashboard, API and CSV read directly.
            "diameter_mm": _round(self.cylinder_diameter_mm, 2),
            "radius_mm": _round(self.radius_mm, 2),
            "volume_liters": _round(self.selected_volume_litres, 6),
            "fit_confidence": _round(self.fit_confidence, 4),
            "rejection_reason": self.rejection_reason,
            "points": int(self.points),
            "frames": int(self.frames),
            "frozen": bool(self.frozen),
            "flags": list(self.flags),
            "features": {key: round(float(value), 4) for key, value in self.features.items()},
        }


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _reject_outliers(footprint: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop stray points (mask bleed, flying pixels) far from the object's body."""
    centre = np.median(footprint, axis=0)
    radial = np.linalg.norm(footprint - centre, axis=1)
    median = float(np.median(radial))
    mad = float(np.median(np.abs(radial - median))) * 1.4826
    keep = radial <= median + 4.0 * max(mad, 1e-4)
    return footprint[keep], heights[keep]


def _drop_sparse(footprint: np.ndarray, heights: np.ndarray, cell: float = 0.005,
                 min_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated flying pixels: points in footprint cells with almost no neighbours."""
    keys = np.floor(footprint / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inverse.reshape(-1)] >= min_count
    if int(keep.sum()) < MIN_POINTS:
        return footprint, heights
    return footprint[keep], heights[keep]


def _circle_residual(points: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    circle = _fit_circle(points)
    if circle is None:
        return None
    cx, cy, radius = circle
    if radius <= 0:
        return None
    distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    return radius, float(np.std(distances - radius) / radius), np.array((cx, cy))


@dataclass(frozen=True)
class CircleFit:
    centre: np.ndarray
    radius: float
    residual: float  # RMS radial error of inliers / radius
    inlier_ratio: float
    coverage_deg: float  # angular span of inliers around the centre


def robust_circle_fit(points: np.ndarray, *, samples: int = 256, seed: int = 0) -> CircleFit | None:
    """RANSAC circle (3-point hypotheses) refined by least squares with MAD rejection.

    Deterministic (fixed seed). Works on a full outline or on a partial arc --
    the visible front shell of an upright bottle, or the top profile of a
    lying one -- where an algebraic fit alone is dragged by outliers.
    """
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count < 12:
        return None
    rng = np.random.default_rng(seed)
    triples = points[rng.integers(0, count, size=(samples, 3))]
    a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    valid = np.abs(d) > 1e-12
    if not np.any(valid):
        return None
    a, b, c, d = a[valid], b[valid], c[valid], d[valid]
    sa, sb, sc = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
    ux = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - b[:, 1])) / d
    uy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (b[:, 0] - a[:, 0])) / d
    radii = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
    span = float(np.max(np.ptp(points, axis=0)))
    plausible = (radii > 1e-3) & (radii < 2.0 * max(span, 1e-3))
    if not np.any(plausible):
        return None
    ux, uy, radii = ux[plausible], uy[plausible], radii[plausible]
    distances = np.abs(np.hypot(points[None, :, 0] - ux[:, None], points[None, :, 1] - uy[:, None]) - radii[:, None])
    tolerance = np.maximum(0.002, 0.05 * radii)
    scores = (distances <= tolerance[:, None]).sum(axis=1)
    best = int(np.argmax(scores))
    inliers = distances[best] <= tolerance[best]
    for _ in range(3):
        if int(inliers.sum()) < 12:
            return None
        fitted = _fit_circle(points[inliers])
        if fitted is None:
            return None
        cx, cy, radius = fitted
        radial = np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius
        mad = 1.4826 * float(np.median(np.abs(radial[inliers] - np.median(radial[inliers]))))
        inliers = np.abs(radial) <= max(3.5 * mad, 0.001)
    kept = points[inliers]
    radial = np.hypot(kept[:, 0] - cx, kept[:, 1] - cy) - radius
    angles = np.sort(np.arctan2(kept[:, 1] - cy, kept[:, 0] - cx))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
    coverage = 360.0 - math.degrees(float(gaps.max()))
    return CircleFit(np.array((cx, cy)), float(radius), float(np.sqrt(np.mean(radial ** 2)) / radius),
                     float(inliers.mean()), coverage)


def _residual_limit_m(fit: CircleFit) -> float:
    """Allowed RMS radial error: stereo depth noise (a few mm) or a fraction of the radius."""
    return max(CYLINDER_NOISE_FLOOR_M, MAX_CYLINDER_RESIDUAL * fit.radius)


def _edge_corrected_fit(footprint: np.ndarray) -> CircleFit | None:
    """Circle through a filled footprint's outline, corrected for how it was sampled.

    Each sector's outermost point overshoots a noisy edge by about one edge
    sigma, and pixel-centre sampling undershoots a clean one by half a pixel.
    Both are measured from the data (outline residual spread, mean point
    spacing) and removed; on synthetic discs this keeps the diameter within
    about -5 %/+10 % at 0-3 mm edge noise, versus +5-17 % uncorrected.
    """
    outline = _boundary_points(footprint)
    fit = robust_circle_fit(outline) if len(outline) >= 12 else None
    if fit is None:
        return None
    residual = np.hypot(outline[:, 0] - fit.centre[0], outline[:, 1] - fit.centre[1]) - fit.radius
    edge_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    hull = cv2.convexHull(footprint.astype(np.float32))
    spacing = math.sqrt(max(float(cv2.contourArea(hull)), 1e-12) / len(footprint))
    radius = fit.radius - edge_sigma + 0.5 * spacing
    return replace(fit, radius=radius) if radius > 0 else fit


def _fit_confidence(fit: CircleFit) -> float:
    residual_term = 1.0 - min(1.0, fit.residual * fit.radius / _residual_limit_m(fit))
    return float(np.clip(
        fit.inlier_ratio * (0.4 + 0.6 * residual_term) * min(1.0, fit.coverage_deg / 150.0), 0.0, 1.0,
    ))


def _boundary_points(footprint: np.ndarray, bins: int = 72, quantile: float = 1.0) -> np.ndarray:
    """One edge point per angular sector: a dense outline, unlike a sparse hull.

    `quantile` < 1 takes a near-outermost point instead of the extreme one:
    the extreme of a noisy edge sits a couple of sigma outside the true
    boundary, which inflated fitted diameters by 5-15 % at 1-3 mm noise.
    """
    centre = footprint.mean(axis=0)
    offsets = footprint - centre
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    sectors = ((angles + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    order = np.lexsort((radii, sectors))
    ordered_sectors = sectors[order]
    # Each sector is a contiguous run sorted by radius; pick its quantile entry.
    last = np.flatnonzero(np.r_[ordered_sectors[1:] != ordered_sectors[:-1], True])
    first = np.r_[0, last[:-1] + 1]
    chosen = first + np.floor(quantile * (last - first)).astype(int)
    return footprint[order[chosen]].astype(np.float64)


def _cross_profile_ratio(local: np.ndarray, heights: np.ndarray, width: float) -> float | None:
    """Edge-to-centre height ratio across the short side (y). ~1 flat, <0.8 curved."""
    if width <= 0 or len(local) < 30:
        return None
    y = local[:, 1] - np.median(local[:, 1])
    centre = heights[np.abs(y) <= width / 6.0]
    edges = heights[np.abs(y) >= width / 3.0]
    if centre.size < 5 or edges.size < 5:
        return None
    centre_height = float(np.median(centre))
    return float(np.median(edges)) / centre_height if centre_height > 0 else None


def _min_area_rect(points: np.ndarray) -> tuple[float, float, float]:
    """(long side, short side, long-side angle in radians), via OpenCV.

    Same answer as footprint.minimum_area_extents, which is quadratic in hull
    size -- and every point of a visible arc is on the hull.
    """
    (_, _), (first, second), degrees = cv2.minAreaRect(points.astype(np.float32))
    if first >= second:
        return float(first), float(second), math.radians(degrees)
    return float(second), float(first), math.radians(degrees + 90.0)


def _litres(cubic_m: float) -> float:
    return cubic_m * 1000.0


def measure_shape(
    footprint_m: np.ndarray,
    heights_m: np.ndarray,
    *,
    mesh_volume_l: float | None = None,
    height_mm: float | None = None,
    label: str = "",
    min_points: int = MIN_POINTS,
) -> ShapeGeometry:
    """Route one frame's segmented, plane-relative points to a volume method.

    `footprint_m` is (N, 2) support-plane coordinates in metres; `heights_m`
    is each point's height above that plane. `mesh_volume_l` is the existing
    height-map integral over the same mask, reported as the secondary value.
    """
    footprint = np.asarray(footprint_m, dtype=np.float64)
    heights = np.asarray(heights_m, dtype=np.float64)
    finite = np.all(np.isfinite(footprint), axis=1) & np.isfinite(heights) if len(footprint) else np.zeros(0, bool)
    footprint, heights = footprint[finite], heights[finite]
    mesh = None if mesh_volume_l is None or not math.isfinite(mesh_volume_l) else float(mesh_volume_l)

    def _uncertain(reason: str, points: int, **dims: float | None) -> ShapeGeometry:
        return ShapeGeometry(
            geometry_method=UNCERTAIN, geometry_confidence=0.1,
            length_mm=dims.get("length"), width_mm=dims.get("width"), height_mm=dims.get("height"),
            bounding_box_volume_litres=None, mesh_volume_litres=mesh, selected_volume_litres=None,
            volume_meaning="unavailable", points=points, flags=(reason,),
        )

    if len(footprint) < min_points:
        return _uncertain("insufficient_visible_surface", len(footprint))
    footprint, heights = _reject_outliers(footprint, heights)
    footprint, heights = _drop_sparse(footprint, heights)
    points = len(footprint)
    if points < min_points:
        return _uncertain("insufficient_visible_surface", points)

    hull = cv2.convexHull(footprint.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    length_m, width_m, angle = _min_area_rect(footprint)
    top_m = float(np.percentile(heights, 95)) if height_mm is None else height_mm / 1000.0
    length_mm, width_mm, top_mm = length_m * 1000.0, width_m * 1000.0, top_m * 1000.0
    if not all(MIN_DIMENSION_MM <= value <= MAX_DIMENSION_MM for value in (length_mm, width_mm, top_mm)):
        return _uncertain("implausible_dimensions", points, length=length_mm, width=width_mm, height=top_mm)

    cos, sin = math.cos(-angle), math.sin(-angle)
    local = footprint @ np.array([[cos, -sin], [sin, cos]]).T
    local -= local.mean(axis=0)
    hull_area = _polygon_area(hull)
    rect_fill = hull_area / max(length_m * width_m, 1e-12)
    aspect = width_m / max(length_m, 1e-12)
    flat_top = float(np.mean(heights >= top_m - max(0.008, 0.10 * top_m)))
    column_fill = _column_fill_ratio(local, width_m)
    outline = _boundary_points(footprint)
    circle = _circle_residual(outline) if len(outline) >= 12 else None
    residual = None if circle is None else circle[1]
    profile = _cross_profile_ratio(local, heights, width_m)
    features = {
        "rect_fill": rect_fill, "aspect": aspect, "flat_top_fraction": flat_top,
        "column_fill": column_fill, "circle_residual": -1.0 if residual is None else residual,
        "cross_profile_ratio": -1.0 if profile is None else profile,
    }
    bbox_l = _litres(length_m * width_m * top_m)
    base = dict(length_mm=length_mm, width_mm=width_mm, height_mm=top_mm,
                bounding_box_volume_litres=bbox_l, mesh_volume_litres=mesh, points=points, features=features)

    cylinder_rejection: list[str] = []

    def _cylinder(fit: CircleFit, axis_m: float, orientation: str,
                  flags: tuple[str, ...]) -> ShapeGeometry | None:
        """A fitted cylinder, or None when the fit is not trustworthy.

        A rejected fit falls through to the existing height-map result
        (irregular), with the reason kept -- never a fake cylinder, never a box.
        """
        diameter_m = 2.0 * fit.radius
        confidence = _fit_confidence(fit)
        reason = None
        if fit.residual * fit.radius > _residual_limit_m(fit):
            reason = "cylinder_fit_residual_too_high"
        elif fit.coverage_deg < MIN_ARC_COVERAGE_DEG:
            reason = "insufficient_arc_coverage"
        elif confidence < MIN_CYLINDER_FIT_CONFIDENCE:
            reason = "cylinder_fit_low_confidence"
        elif not MIN_DIMENSION_MM <= diameter_m * 1000.0 <= MAX_DIMENSION_MM:
            reason = "implausible_cylinder_diameter"
        if reason is not None:
            cylinder_rejection.append(reason)
            return None
        volume = _litres(math.pi * fit.radius ** 2 * axis_m)
        if orientation == "upright":
            dims = dict(length_mm=diameter_m * 1000.0, width_mm=diameter_m * 1000.0, height_mm=axis_m * 1000.0)
        else:
            dims = dict(length_mm=axis_m * 1000.0, width_mm=diameter_m * 1000.0, height_mm=top_mm)
        dims_bbox = _litres(dims["length_mm"] * dims["width_mm"] * dims["height_mm"] / 1e9)
        return ShapeGeometry(
            geometry_method=CYLINDER, geometry_confidence=float(np.clip(confidence, 0.05, 0.95)),
            bounding_box_volume_litres=dims_bbox, mesh_volume_litres=mesh, selected_volume_litres=volume,
            volume_meaning="cylinder_volume_pi_r2_h", cylinder_diameter_mm=diameter_m * 1000.0,
            cylinder_height_mm=axis_m * 1000.0, cylinder_fit_residual=fit.residual,
            cylinder_volume_litres=volume, cylinder_orientation=orientation,
            radius_mm=fit.radius * 1000.0, fit_confidence=confidence,
            points=points, features=features, flags=flags, **dims,
        )

    # Upright cylinder. From above the footprint is a disc; from an angle only
    # the front shell is visible and projects to an arc (the 83 x 23 mm bottle:
    # the 23 mm was the arc's bulge, not a width). The axis is the support-plane
    # normal, so the circle is fitted in the footprint plane -- to the outline
    # of a filled disc, or to every point of a thin shell -- and the height is
    # measured along that axis.
    disc = aspect >= 0.85 and DISC_FILL_RANGE[0] <= rect_fill <= DISC_FILL_RANGE[1] and residual is not None \
        and residual <= MAX_CIRCLE_RESIDUAL
    shell = column_fill < 0.55 and aspect < 0.85
    words = set(label.lower().replace("_", " ").replace("-", " ").split())
    # A labelled can/bottle seen obliquely shows its top disc plus the front
    # shell, whose projection lands on the same circle: the union is a disc
    # that need not pass the strict disc test, so the label admits the fit.
    labelled = bool(words & _CYLINDER_WORDS) and aspect >= 0.6
    # Height along the vertical axis: the near-top of the points, not the p95
    # used for general objects (a side-seen shell is uniform in height).
    # A side-seen shell has uniformly spread heights (no plateau), so its top
    # is its extreme; a visible top disc is a plateau, where p98 avoids noise.
    axis_top_m = float(np.percentile(heights, 99.5 if shell else 98.0)) if height_mm is None else top_m
    if disc or shell or labelled:
        # Arc coverage and residual (inside _cylinder) guard against an
        # extrapolated fit; the rectangle's chord is not used, because a few
        # stray points widen it without changing the circle. A filled footprint
        # is fitted on its outline -- the widest body, so a bottle's narrower
        # neck and cap inside it do not shrink the diameter.
        fit = robust_circle_fit(footprint) if shell else _edge_corrected_fit(footprint)
        if fit is not None:
            flag = "disc_footprint" if disc else "arc_reconstructed_diameter" if shell else "labelled_cylinder_fit"
            fitted = _cylinder(fit, axis_top_m, "upright", (flag,))
            if fitted is not None:
                return fitted

    # Cylinder lying on its side: the axis is the footprint's principal
    # direction (PCA); the cross-section across it is the visible upper half of
    # the circle, fitted in (across-axis offset, height above the plane).
    if (rect_fill >= RECT_FILL_MIN and profile is not None and profile < CURVED_PROFILE_MAX
            and 0.75 <= top_m / max(width_m, 1e-9) <= 1.30):
        centred = footprint - footprint.mean(axis=0)
        _, _, axes = np.linalg.svd(centred[:: max(1, len(centred) // 5000)], full_matrices=False)
        along, across = centred @ axes[0], centred @ axes[1]
        fit = robust_circle_fit(np.column_stack((across, heights)))
        if fit is not None:
            axis_length = float(np.percentile(along, 99.5) - np.percentile(along, 0.5))
            fitted = _cylinder(fit, axis_length, "lying", ("curved_cross_profile",))
            if fitted is not None:
                return fitted

    # Cuboid: a filled rectangle with a flat top.
    if (rect_fill >= RECT_FILL_MIN and flat_top >= FLAT_TOP_MIN and column_fill >= 0.55
            and (profile is None or profile >= CURVED_PROFILE_MAX)):
        confidence = 0.55 + 0.25 * min(1.0, (rect_fill - RECT_FILL_MIN) / 0.1) + 0.15 * min(1.0, flat_top)
        return ShapeGeometry(
            geometry_method=CUBOID, geometry_confidence=float(np.clip(confidence, 0.05, 0.95)),
            selected_volume_litres=bbox_l, volume_meaning="cuboid_volume_l_w_h",
            flags=("oriented_support_plane_box",), **base,
        )

    # Everything else is measured by what it occupies, not by its box.
    if mesh is None or mesh <= 0:
        result = _uncertain("no_height_map_volume", points, length=length_mm, width=width_mm, height=top_mm)
        return replace(result, bounding_box_volume_litres=bbox_l, features=features)
    flexible = bool(words & _FLEXIBLE_WORDS)
    confidence = 0.6 if mesh <= bbox_l * 1.05 else 0.35
    rejected = tuple(f"cylinder_fit_rejected:{reason}" for reason in cylinder_rejection)
    return ShapeGeometry(
        geometry_method=FLEXIBLE_OR_UNKNOWN if flexible else IRREGULAR_RIGID,
        geometry_confidence=confidence, selected_volume_litres=mesh,
        volume_meaning="current_external_occupied_volume" if flexible else "segmented_height_map_volume",
        flags=("height_map_integral",) + (() if mesh <= bbox_l * 1.05 else ("mesh_exceeds_bounding_box",)) + rejected,
        rejection_reason=cylinder_rejection[0] if cylinder_rejection else None,
        **base,
    )


def _median(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.median(present)) if present else None


def aggregate_shapes(results: list[ShapeGeometry]) -> ShapeGeometry:
    """Median of several same-method frames, with volumes recomputed from the medians."""
    first = results[0]
    length = _median([item.length_mm for item in results])
    width = _median([item.width_mm for item in results])
    height = _median([item.height_mm for item in results])
    mesh = _median([item.mesh_volume_litres for item in results])
    bbox = None if None in (length, width, height) else length * width * height / 1e6
    selected, cylinder_volume = mesh, None
    diameter = _median([item.cylinder_diameter_mm for item in results])
    axis = _median([item.cylinder_height_mm for item in results])
    if first.geometry_method == CYLINDER and diameter is not None and axis is not None:
        cylinder_volume = math.pi * (diameter / 2.0) ** 2 * axis / 1e6
        selected = cylinder_volume
    elif first.geometry_method == CUBOID:
        selected = bbox
    diameters = [item.cylinder_diameter_mm for item in results if item.cylinder_diameter_mm]
    flags = set(first.flags)
    if len(diameters) >= 2 and np.std(diameters) / max(np.mean(diameters), 1e-9) > 0.08:
        flags.add("unstable_cylinder_radius")
    return replace(
        first,
        geometry_confidence=float(np.median([item.geometry_confidence for item in results])),
        length_mm=length, width_mm=width, height_mm=height,
        bounding_box_volume_litres=bbox, mesh_volume_litres=mesh, selected_volume_litres=selected,
        cylinder_diameter_mm=diameter, cylinder_height_mm=axis, cylinder_volume_litres=cylinder_volume,
        cylinder_fit_residual=_median([item.cylinder_fit_residual for item in results]),
        frames=len(results), flags=tuple(sorted(flags)),
    )


class GeometryLock:
    """Per-track method voting, then a permanent freeze.

    A method is accepted once it wins `required_frames` of the last
    `window` frames. Until then the provisional answer is the latest frame's.
    After acceptance the aggregated result is returned unchanged for the
    rest of the track's life, so a completed measurement cannot flip method.
    """

    def __init__(self, required_frames: int = 5, window: int = 9) -> None:
        self.required_frames = max(1, int(required_frames))
        self.window = max(self.required_frames, int(window))
        self._history: dict[int, deque[ShapeGeometry]] = {}
        self._frozen: dict[int, ShapeGeometry] = {}

    def update(self, track_id: int | None, result: ShapeGeometry | None) -> ShapeGeometry | None:
        if track_id is None:
            return result
        frozen = self._frozen.get(track_id)
        if frozen is not None:
            return frozen
        if result is None:
            return None
        history = self._history.setdefault(track_id, deque(maxlen=self.window))
        history.append(result)
        votes = Counter(item.geometry_method for item in history if item.geometry_method != UNCERTAIN)
        if votes:
            method, count = votes.most_common(1)[0]
            if count >= self.required_frames:
                chosen = [item for item in history if item.geometry_method == method]
                final = replace(aggregate_shapes(chosen), frozen=True)
                self._frozen[track_id] = final
                self._history.pop(track_id, None)
                return final
        return result

    def frozen(self, track_id: int | None) -> ShapeGeometry | None:
        return None if track_id is None else self._frozen.get(track_id)

    def forget(self, track_id: int | None) -> None:
        self._history.pop(track_id, None)
        self._frozen.pop(track_id, None)

    def clear(self) -> None:
        self._history.clear()
        self._frozen.clear()
