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
_FLEXIBLE_WORDS = {"bag", "bags", "sack", "pillow", "cushion", "backpack", "textile", "cloth", "toy", "plush"}

MIN_POINTS = 50
# A visible shell whose chord (noise widens it) spans this range of the fitted
# diameter is a plausibly half-seen cylinder; outside it the fit extrapolates.
ARC_CHORD_TO_DIAMETER = (0.70, 1.15)
MIN_DIMENSION_MM = 5.0
MAX_DIMENSION_MM = 1500.0

# A disc fills pi/4 = 0.785 of its bounding square, a rectangle fills ~1.
DISC_FILL_RANGE = (0.68, 0.87)
RECT_FILL_MIN = 0.88
MAX_CIRCLE_RESIDUAL = 0.07
FLAT_TOP_MIN = 0.40
# Edge-to-centre height ratio across the short side below which the top is curved.
CURVED_PROFILE_MAX = 0.85


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


def _circle_residual(points: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    circle = _fit_circle(points)
    if circle is None:
        return None
    cx, cy, radius = circle
    if radius <= 0:
        return None
    distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    return radius, float(np.std(distances - radius) / radius), np.array((cx, cy))


def _boundary_points(footprint: np.ndarray, bins: int = 72) -> np.ndarray:
    """Outermost point per angular sector: a dense outline, unlike a sparse hull."""
    centre = footprint.mean(axis=0)
    offsets = footprint - centre
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    sectors = ((angles + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    order = np.lexsort((radii, sectors))
    ordered_sectors = sectors[order]
    # Last entry of each sector run is that sector's largest radius.
    last = np.flatnonzero(np.r_[ordered_sectors[1:] != ordered_sectors[:-1], True])
    return footprint[order[last]].astype(np.float64)


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

    def _cylinder(diameter_m: float, axis_m: float, fit_residual: float, orientation: str,
                  confidence: float, flags: tuple[str, ...]) -> ShapeGeometry:
        volume = _litres(math.pi * (diameter_m / 2.0) ** 2 * axis_m)
        if orientation == "upright":
            dims = dict(length_mm=diameter_m * 1000.0, width_mm=diameter_m * 1000.0, height_mm=top_mm)
        else:
            dims = dict(length_mm=axis_m * 1000.0, width_mm=diameter_m * 1000.0, height_mm=top_mm)
        dims_bbox = _litres(dims["length_mm"] * dims["width_mm"] * dims["height_mm"] / 1e9)
        return ShapeGeometry(
            geometry_method=CYLINDER, geometry_confidence=float(np.clip(confidence, 0.05, 0.95)),
            bounding_box_volume_litres=dims_bbox, mesh_volume_litres=mesh, selected_volume_litres=volume,
            volume_meaning="cylinder_volume_pi_r2_h", cylinder_diameter_mm=diameter_m * 1000.0,
            cylinder_height_mm=axis_m * 1000.0, cylinder_fit_residual=fit_residual,
            cylinder_volume_litres=volume, cylinder_orientation=orientation,
            points=points, features=features, flags=flags, **dims,
        )

    # Upright cylinder seen from above: the footprint is a filled disc.
    if (circle is not None and aspect >= 0.85 and DISC_FILL_RANGE[0] <= rect_fill <= DISC_FILL_RANGE[1]
            and residual <= MAX_CIRCLE_RESIDUAL):
        diameter = 2.0 * circle[0]
        confidence = 0.9 - 4.0 * residual - 0.3 * max(0.0, 0.6 - flat_top)
        return _cylinder(diameter, top_m, residual, "upright", confidence, ("disc_footprint",))

    # Upright cylinder seen from the side: only the front shell is visible, a
    # thin arc whose circle fit restores the hidden half (as footprint.py does
    # for the reported extents).
    arc = _circle_residual(footprint) if column_fill < 0.55 else None
    if arc is not None and arc[1] <= MAX_CIRCLE_RESIDUAL:
        diameter = 2.0 * arc[0]
        if ARC_CHORD_TO_DIAMETER[0] <= length_m / diameter <= ARC_CHORD_TO_DIAMETER[1]:
            confidence = 0.75 - 3.0 * arc[1]
            return _cylinder(diameter, top_m, arc[1], "upright", confidence, ("arc_reconstructed_diameter",))

    # Cylinder lying on its side: a filled rectangle whose cross-section is a
    # half-circle -- as tall as it is wide, highest along the middle.
    if (rect_fill >= RECT_FILL_MIN and profile is not None and profile < CURVED_PROFILE_MAX
            and 0.75 <= top_m / max(width_m, 1e-9) <= 1.30):
        diameter = 0.5 * (width_m + top_m)
        fit_residual = abs(top_m - width_m) / diameter
        confidence = 0.75 - 1.5 * fit_residual
        return _cylinder(diameter, length_m, fit_residual, "lying", confidence, ("curved_cross_profile",))

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
    words = set(label.lower().replace("_", " ").split())
    flexible = bool(words & _FLEXIBLE_WORDS)
    confidence = 0.6 if mesh <= bbox_l * 1.05 else 0.35
    return ShapeGeometry(
        geometry_method=FLEXIBLE_OR_UNKNOWN if flexible else IRREGULAR_RIGID,
        geometry_confidence=confidence, selected_volume_litres=mesh,
        volume_meaning="current_external_occupied_volume" if flexible else "segmented_height_map_volume",
        flags=("height_map_integral",) + (() if mesh <= bbox_l * 1.05 else ("mesh_exceeds_bounding_box",)),
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
