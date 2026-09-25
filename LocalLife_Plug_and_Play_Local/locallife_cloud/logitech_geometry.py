"""Object-only metric geometry for the Logitech camera.

Every Logitech dimension is measured relative to a plane, and the plane decides
whether the answer means anything. On a ray-traced scene with the rig's own
geometry -- a camera tilted 45 degrees, 0.95 m above the mat -- the existing
projection code recovers 335 x 255 x 124 mm for a 338 x 253 x 124 mm box and
75 x 70 mm for a 75 mm bottle *when it is given the fitted floor*. The same
code measured against `axis_aligned_plane`, the fallback taken whenever no
floor plane has been fitted, measures against a plane perpendicular to the
optical axis instead: the floor only for a camera pointing straight down, and
for this mount a plane standing across the scene. That is the difference
between a measurement and a number.

So the first repair is upstream of this module: fit the floor and measure
against it (`VisionPipeline._logitech_floor_plane`), and mark any measurement
that fell back (`plane_is_floor`) so it can never pass for a floor-relative
one. Whether that alone accounts for the 540 x 360 x 89 mm the hardware
reported is not established here -- it is a hypothesis the next hardware run
tests, not a proven cause.

What follows is what the plane repair does not cover:

* `robust_object_height_m` -- the top from the object's own measured cells,
  never from the cells the convex completion filled in with a neighbour's
  height, and taken before the spike gate that was clipping a bottle's neck
  off: the 203 mm bottle came back 150 mm tall, its body's height exactly,
  because the neck sat above median + 6 MAD and was discarded as an outlier.
* the envelope check that used to live here has moved to `logitech_bin`,
  where it compares the measured volume with the envelope containing it
  instead of assuming every object is a cuboid. The cuboid form called every
  bag in a real bin inconsistent while passing a volume 21 % larger than its
  own bounding box.
* `static_background_reason` -- the bed and the floor were being measured as
  35 L deposits. A deposit sits inside the zone and has an outside; a piece of
  the room reaches the zone's own edges.
* `unclaimed_foreground_islands` -- an object the detector never proposed is
  still an object, which is why the small can was never measured.
* `GeometryStabiliser` -- the short rolling median that stops a settled object
  changing size every frame.

Nothing here touches the RealSense pipeline, which has its own dimension and
volume path and does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np


# A height counts as the object's when this much of its own footprint reaches
# it -- never fewer than MIN_TOP_CELLS, so a small object is not measured by a
# single cell.
MIN_TOP_AREA_FRACTION = 0.03
MIN_TOP_CELLS = 4

# A mask covering more of the measurement zone than this is the scene, not a
# deposit standing in it.
MAX_ZONE_COVERAGE = 0.35
# ... and one reaching two opposite edges of the zone has no outside.
BORDER_TOLERANCE_PX = 2
# An object the detector never proposed is still an object, and it is marked so
# nothing mistakes it for a classified detection.
CHANGE_RECOVERED_SOURCE = "logitech-foreground-change-object"


def robust_object_height_m(
    heights: np.ndarray, *, noise_floor_m: float = 0.004,
    min_area_fraction: float = MIN_TOP_AREA_FRACTION, min_cells: int = MIN_TOP_CELLS,
) -> float | None:
    """The highest level a real part of the object actually reaches.

    Not a percentile chosen by taste. A height is believed when enough of the
    object's own footprint reaches it: at least `min_area_fraction` of its cells
    and never fewer than `min_cells`, whichever is larger. That single rule
    handles both failures that a plain percentile could not handle together.

    A bottle's neck is a twelfth of its footprint and 50 mm above its body, so
    it clears the bar and the bottle measures 203 mm rather than its body's
    150 mm. A handful of monocular depth spikes over a flat, shiny bag is a
    fraction of a percent of a large footprint, does not clear it, and cannot
    turn a 78 mm bag into a 228 mm one. On a small object, where three per cent
    of the footprint is less than a cell, `min_cells` takes over and the height
    still needs several cells to agree.

    Values at or below the sensor's noise floor are not a height at all.
    """
    values = np.asarray(heights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > noise_floor_m)]
    if values.size == 0:
        return None
    needed = max(int(min_cells), int(np.ceil(min_area_fraction * values.size)))
    needed = min(needed, values.size)
    # The needed-th largest value: the level that many cells reach or exceed.
    return float(np.sort(values)[-needed])


def static_background_reason(
    mask: np.ndarray,
    region: np.ndarray | None,
    *,
    change: np.ndarray | None = None,
    max_coverage: float = MAX_ZONE_COVERAGE,
    border_tolerance: int = BORDER_TOLERANCE_PX,
) -> str | None:
    """Why this mask is the room rather than something deposited in it, or None.

    The bed, the floor and the table are detected like anything else, and once
    a stale baseline let them through they were measured as 35 L deposits. An
    object placed in the zone occupies a modest part of it and has an outside;
    a piece of the room reaches the zone's own edges.
    """
    array = np.asarray(mask, dtype=bool)
    if not array.any():
        return None
    zone = None if region is None else np.asarray(region, dtype=bool)
    if zone is not None and zone.shape == array.shape and zone.any():
        inside = int(np.count_nonzero(array & zone))
        if inside == 0:
            return "outside_measurement_zone"
        if inside / int(np.count_nonzero(zone)) > max_coverage:
            return "covers_measurement_zone"
        rows, columns = np.nonzero(zone)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
    else:
        top, left = 0, 0
        bottom, right = array.shape[0] - 1, array.shape[1] - 1

    pad = max(0, int(border_tolerance))
    touches_top = array[top : top + pad + 1, left : right + 1].any()
    touches_bottom = array[max(top, bottom - pad) : bottom + 1, left : right + 1].any()
    touches_left = array[top : bottom + 1, left : left + pad + 1].any()
    touches_right = array[top : bottom + 1, max(left, right - pad) : right + 1].any()
    if (touches_top and touches_bottom) or (touches_left and touches_right):
        return "spans_measurement_zone"
    if change is not None:
        changed = np.asarray(change, dtype=bool)
        if changed.shape == array.shape and not (array & changed).any():
            return "unchanged_since_baseline"
    return None


@dataclass
class StableGeometry:
    length_mm: float
    width_mm: float
    height_mm: float
    volume_l: float | None
    samples: int
    settled: bool
    spread: float


class GeometryStabiliser:
    """Rolling median of one track's dimensions and volume.

    Monocular depth moves frame to frame, so a settled object was changing size
    on every redraw. A short median window is enough to remove a one-frame
    spike without lagging a real change, and the spread it reports is what says
    whether the measurement has settled yet.
    """

    def __init__(self, window: int = 9, minimum_frames: int = 3, tolerance: float = 0.25) -> None:
        self.window = max(1, int(window))
        self.minimum_frames = max(1, int(minimum_frames))
        self.tolerance = float(tolerance)
        self._samples: dict[int, list[tuple[float, float, float, float | None]]] = {}

    def update(
        self, track_id: int | None, *, length_mm: float, width_mm: float, height_mm: float,
        volume_l: float | None = None,
    ) -> StableGeometry:
        if track_id is None:
            return StableGeometry(length_mm, width_mm, height_mm, volume_l, 1, False, float("inf"))
        samples = self._samples.setdefault(int(track_id), [])
        samples.append((float(length_mm), float(width_mm), float(height_mm), volume_l))
        if len(samples) > self.window:
            del samples[: len(samples) - self.window]
        lengths = [item[0] for item in samples]
        widths = [item[1] for item in samples]
        heights = [item[2] for item in samples]
        volumes = [item[3] for item in samples if item[3] is not None]
        spread = 0.0
        for values in (lengths, widths, heights):
            centre = median(values)
            if centre > 0:
                spread = max(spread, (max(values) - min(values)) / centre)
        settled = len(samples) >= self.minimum_frames and spread <= self.tolerance
        return StableGeometry(
            length_mm=round(median(lengths), 2), width_mm=round(median(widths), 2),
            height_mm=round(median(heights), 2),
            volume_l=None if not volumes else round(median(volumes), 6),
            samples=len(samples), settled=settled, spread=round(spread, 4),
        )

    def forget(self, track_id: int | None) -> None:
        if track_id is not None:
            self._samples.pop(int(track_id), None)

    def clear(self) -> None:
        self._samples.clear()


def unclaimed_foreground_islands(
    change: np.ndarray | None,
    region: np.ndarray | None,
    claimed: np.ndarray | None,
    *,
    min_pixels: int,
    max_coverage: float = MAX_ZONE_COVERAGE,
    claimed_overlap: float = 0.25,
    limit: int = 3,
) -> list[np.ndarray]:
    """Foreground islands inside the zone that no detection claimed.

    A small can standing on the mat is plainly a change against the empty
    scene, but an open-vocabulary detector does not always propose a box for
    it, and an object nothing proposes is an object nothing measures. The
    change mask has already established that the island is new and inside the
    zone, which is what the volume path needs; a class label is not. Islands
    that reach two sides of the zone are the floor, and islands a detection
    already covers are not returned at all, so this can only ever add an object
    the detector missed -- it never modifies the detector or its masks.
    """
    if change is None or region is None:
        return []
    changed = np.asarray(change, dtype=bool)
    zone = np.asarray(region, dtype=bool)
    if changed.shape != zone.shape or not changed.any():
        return []
    taken = (
        np.zeros_like(zone) if claimed is None or np.asarray(claimed).shape != zone.shape
        else np.asarray(claimed, dtype=bool)
    )
    try:
        import cv2

        count, labels = cv2.connectedComponents((changed & zone).astype(np.uint8), connectivity=8)
    except Exception:  # noqa: BLE001 - without OpenCV there is no recovery, not a crash
        return []
    found: list[tuple[int, np.ndarray]] = []
    for index in range(1, count):
        island = labels == index
        pixels = int(np.count_nonzero(island))
        if pixels < max(4, int(min_pixels)):
            continue
        if static_background_reason(island, zone, max_coverage=max_coverage) is not None:
            continue
        if int(np.count_nonzero(island & taken)) > claimed_overlap * pixels:
            continue
        found.append((pixels, island))
    found.sort(key=lambda item: item[0], reverse=True)
    return [island for _, island in found[: max(0, int(limit))]]


def minimum_object_pixels(shape: tuple[int, ...] | None, configured: int) -> int:
    """A pixel floor that scales with the frame, so a small can is not excluded.

    150 pixels is a reasonable floor at 640x480 and far too many at 1920x1080,
    where the same can covers four times as much. Scaling by the frame's area
    keeps the physical size the floor represents roughly constant, and the
    configured value is still the ceiling: this only ever lowers the bar for a
    small frame, never raises it.
    """
    baseline = 640 * 480
    if not shape or len(shape) < 2:
        return int(configured)
    pixels = int(shape[0]) * int(shape[1])
    if pixels <= 0:
        return int(configured)
    scaled = int(round(configured * pixels / baseline))
    return max(40, min(int(configured), scaled)) if pixels < baseline else max(40, int(configured))
