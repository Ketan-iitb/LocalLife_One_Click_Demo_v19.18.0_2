"""Support-plane footprint extents for objects seen from a single viewpoint.

Why the old measurement was wrong
---------------------------------
A cylinder standing 5 cm across and 20 cm tall was measured as 5 x 2 cm. The
height was right; the footprint was not, and the reason is geometric rather
than numerical.

One camera sees one side. For an upright cylinder the visible surface is the
front half of a shell, so projecting those points onto the support plane gives
a *half*-disc: full diameter across the chord, but only the bulge depth -- a
couple of centimetres -- along the viewing direction. Taking the extent of the
visible points therefore reports the sagitta of an arc as if it were the
object's width. A box does not suffer from this because its flat front face
projects to a line and its top, when visible, fills the rectangle.

Two changes follow from that:

* extents come from a minimum-area rotated rectangle rather than the principal
  axes of the point spread, which is what a diagonally-placed object needs;
* when the footprint is a one-sided arc, the occluded half is reconstructed
  from the chord and the sagitta instead of being reported as the width.

Neither changes a fully-visible flat footprint, which is why large boxes -- the
case that already measured well -- are left alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

# An arc has to bulge by at least this fraction of its chord before a circle is
# fitted to it. Below that the points are nearly collinear, the radius estimate
# explodes, and the honest answer is "this is a flat face, not a curve".
MIN_SAGITTA_RATIO = 0.06

# A reconstructed diameter is only believed up to this multiple of the measured
# chord. A half-seen cylinder gives a diameter close to its chord; anything much
# larger means the arc fit has latched onto noise.
MAX_DIAMETER_GAIN = 2.5


@dataclass
class FootprintExtents:
    """The two support-plane extents, and how they were arrived at."""

    length_m: float
    width_m: float
    method: str = "rotated-rect"
    occlusion_corrected: bool = False
    # Populated when a circle was fitted: the radius it implies, in metres.
    fitted_radius_m: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def length_mm(self) -> float:
        return self.length_m * 1000.0

    @property
    def width_mm(self) -> float:
        return self.width_m * 1000.0


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull. Kept local so this module needs only numpy."""
    if len(points) < 3:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]

    def _half(sequence: np.ndarray) -> list[np.ndarray]:
        stack: list[np.ndarray] = []
        for point in sequence:
            while len(stack) >= 2:
                first, second = stack[-2], stack[-1]
                cross = ((second[0] - first[0]) * (point[1] - first[1])
                         - (second[1] - first[1]) * (point[0] - first[0]))
                if cross <= 0:
                    stack.pop()
                else:
                    break
            stack.append(point)
        return stack

    lower = _half(ordered)
    upper = _half(ordered[::-1])
    return np.array(lower[:-1] + upper[:-1]) if len(lower) + len(upper) > 3 else ordered


def minimum_area_extents(points: np.ndarray) -> tuple[float, float, float]:
    """Smallest enclosing rectangle: (long side, short side, angle in radians).

    Rotating calipers over the hull edges. The minimum-area rectangle always
    has a side flush with a hull edge, so testing each edge's direction finds
    it exactly -- unlike principal axes, which follow how the points are
    *distributed* and skew on a lopsided or partly occluded mask.
    """
    hull = _convex_hull(np.asarray(points, dtype=np.float64))
    if len(hull) < 2:
        return 0.0, 0.0, 0.0
    best = (math.inf, 0.0, 0.0, 0.0)
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        norm = float(np.hypot(edge[0], edge[1]))
        if norm < 1e-12:
            continue
        angle = math.atan2(edge[1], edge[0])
        cos, sin = math.cos(-angle), math.sin(-angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        rotated = hull @ rotation.T
        spans = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(spans[0] * spans[1])
        if area < best[0]:
            best = (area, float(spans[0]), float(spans[1]), angle)
    _, first, second, angle = best
    return max(first, second), min(first, second), angle


def _fit_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic (Kasa) circle fit: returns centre x, centre y, radius."""
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    target = x ** 2 + y ** 2
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre_x, centre_y, offset = solution
    squared = offset + centre_x ** 2 + centre_y ** 2
    if not np.isfinite(squared) or squared <= 0:
        return None
    return float(centre_x), float(centre_y), float(math.sqrt(squared))


def _column_fill_ratio(local: np.ndarray, short_side: float, bins: int = 12) -> float:
    """How much of the short side a typical column of the footprint spans.

    Near 1 for a filled rectangle, near 0 for a thin curve. The median across
    columns is used so a couple of ragged edge columns cannot decide it.
    """
    if short_side <= 0 or len(local) < bins:
        return 0.0
    x = local[:, 0]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    spans: list[float] = []
    for index in range(bins):
        in_bin = (x >= edges[index]) & (x <= edges[index + 1])
        column = local[in_bin, 1]
        if column.size >= 2:
            spans.append(float(column.max() - column.min()) / short_side)
    if len(spans) < max(3, bins // 3):
        return 0.0
    return float(median(spans))


def estimate_extents(
    points: np.ndarray,
    *,
    correct_self_occlusion: bool = True,
) -> FootprintExtents | None:
    """Footprint extents from support-plane points, occlusion-aware.

    `points` are the object's points already projected onto the support plane,
    as (N, 2) metres.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        return None
    array = array[np.all(np.isfinite(array), axis=1)]
    if len(array) < 3:
        return None

    long_side, short_side, angle = minimum_area_extents(array)
    if long_side <= 0 or short_side <= 0:
        return None
    flags: list[str] = []
    if not correct_self_occlusion:
        return FootprintExtents(long_side, short_side, flags=tuple(flags))

    # Work in the rectangle's own frame: x along the long side (the chord), y
    # along the short one (the direction the camera looks down).
    cos, sin = math.cos(-angle), math.sin(-angle)
    local = array @ np.array([[cos, -sin], [sin, cos]]).T
    local -= local.mean(axis=0)

    sagitta_ratio = short_side / long_side if long_side > 0 else 0.0
    if sagitta_ratio >= 0.85:
        # Already nearly square: nothing is hidden that matters.
        return FootprintExtents(long_side, short_side, flags=tuple(flags))

    # Is this a filled rectangle, or a one-sided shell?
    #
    # Slice the footprint into columns along the long axis and ask how much of
    # the short side each column spans. A box top fills its rectangle, so every
    # column spans nearly all of it. A cylinder's visible surface is a thin
    # curved shell, so each column is a point or two however far the curve
    # wanders overall. Measuring the *spread of the whole cloud* cannot tell
    # those apart -- a semicircular arc spreads about as much as a uniform fill
    # -- which is why this looks per column instead.
    if _column_fill_ratio(local, short_side) >= 0.55:
        flags.append("filled_footprint")
        return FootprintExtents(long_side, short_side, flags=tuple(flags))

    if sagitta_ratio < MIN_SAGITTA_RATIO:
        # Too flat to fit a circle to: report what was seen, and say so.
        flags.append("flat_face_only")
        return FootprintExtents(long_side, short_side, flags=tuple(flags))

    circle = _fit_circle(local)
    if circle is None:
        return FootprintExtents(long_side, short_side, flags=tuple(flags))
    _, _, radius = circle
    diameter = 2.0 * radius
    # The lower bound carries a tolerance on purpose: a circle fitted to a
    # perfect half-disc returns a diameter a rounding error below the measured
    # chord, and an exact `>=` rejected exactly the case this exists for.
    # Physically the same allowance covers a noisy arc fitting a hair small.
    if not (long_side * 0.98 <= diameter <= long_side * MAX_DIAMETER_GAIN):
        # The fit disagrees with the chord we actually measured; trust the
        # measurement rather than the extrapolation.
        flags.append("arc_fit_rejected")
        return FootprintExtents(long_side, short_side, flags=tuple(flags))

    # The occluded half restores a round cross-section: both extents become the
    # diameter, bounded below by what was directly observed.
    corrected = max(diameter, short_side)
    flags.append("self_occlusion_corrected")
    return FootprintExtents(
        length_m=max(long_side, corrected),
        width_m=min(max(long_side, short_side), corrected),
        method="rotated-rect+arc",
        occlusion_corrected=True,
        fitted_radius_m=radius,
        flags=tuple(flags),
    )


class DimensionSmoother:
    """Per-track median of the last N frames' dimensions.

    A single frame's segmentation decides the reported thickness of a thin
    object, which is why a 2 cm cream box read 4 cm on some frames. The median
    over a short window ignores the odd bad mask without lagging behind a real
    change, and it is what gets frozen when a measurement is finalised.
    """

    def __init__(self, window: int = 9) -> None:
        self.window = max(1, int(window))
        self._samples: dict[int, list[tuple[float, float, float]]] = {}

    def update(
        self, track_id: int | None, length_mm: float, width_mm: float, height_mm: float,
    ) -> tuple[float, float, float]:
        if track_id is None:
            return length_mm, width_mm, height_mm
        samples = self._samples.setdefault(track_id, [])
        samples.append((length_mm, width_mm, height_mm))
        if len(samples) > self.window:
            del samples[: len(samples) - self.window]
        return (
            round(median(item[0] for item in samples), 2),
            round(median(item[1] for item in samples), 2),
            round(median(item[2] for item in samples), 2),
        )

    def stability(self, track_id: int | None) -> float | None:
        """How much the smoothed dimensions are still moving, 0-1.

        None until there is enough history to say. Used to warn on small or
        thin objects, where a percentage point of mask error is millimetres of
        reported thickness.
        """
        samples = self._samples.get(track_id) if track_id is not None else None
        if not samples or len(samples) < 3:
            return None
        spreads = []
        for axis in range(3):
            values = [item[axis] for item in samples]
            centre = median(values)
            if centre > 0:
                spreads.append((max(values) - min(values)) / centre)
        return None if not spreads else round(1.0 - min(1.0, max(spreads)), 3)

    def forget(self, track_id: int | None) -> None:
        if track_id is not None:
            self._samples.pop(track_id, None)


def small_object_warning(
    length_mm: float, width_mm: float, height_mm: float,
    *, depth_noise_mm: float = 5.0,
) -> str | None:
    """Flag dimensions close enough to sensor noise to be unreliable.

    Not a rejection: a 2 cm object is still worth recording. It is a statement
    that the figure carries a relative error the operator should know about,
    since a few millimetres of depth noise is a large fraction of it.
    """
    smallest = min(length_mm, width_mm, height_mm)
    if smallest <= depth_noise_mm * 2.0:
        return "dimension_near_sensor_noise"
    if smallest <= depth_noise_mm * 4.0:
        return "small_object_low_precision"
    return None
