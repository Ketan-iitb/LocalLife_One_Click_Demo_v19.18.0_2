"""Which pixels are the object that was just deposited.

The detector answers "what is this", not "where does it end". Its open
vocabulary masks spill: one "beverage carton" mask arrived covering two
cartons, the floor between them, a chair and part of the wall, and integrating
it reported a carton 339 mm tall and a 5.4 L blob of background. A mask that
size is not a carton, whatever the label says.

So the detector stops being the measurement mask. What is measured is the part
of the measurement zone that has *changed* since the scene was last committed
-- a real deposit raises the floor -- and the detector only says which of those
changed islands is the object it is tracking. Everything already in the
committed scene (the mat, the furniture behind it, the carton deposited a
minute ago) is background by construction, so the second carton is measured on
its own rather than as the pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# A candidate covering more of the zone than this is the zone, not an object.
MAX_ZONE_FRACTION = 0.6
# Running off this many of the zone's sides means floor, wall or furniture.
MAX_ZONE_EDGES = 2
# How much of the smaller of (detector mask, changed island) they must share
# before the island is accepted as the thing being tracked.
MIN_DETECTOR_OVERLAP = 0.2
# A deposit stands above the committed scene. Less than this is a shadow, a
# reflection or exposure drift.
MIN_HEIGHT_RISE_M = 0.01

DETECTOR_MASK = "detector_mask"
FOREGROUND_COMPONENT = "foreground_component"

OUTSIDE_ZONE = "detection_outside_measurement_zone"
NO_NEW_DEPOSIT = "no_new_deposit_under_detection"
COVERS_ZONE = "candidate_covers_the_measurement_zone"
SPANS_ZONE = "candidate_spans_the_measurement_zone"
NO_RISE = "no_height_rise_above_the_committed_scene"
TOO_SMALL = "candidate_below_minimum_size"

BACKGROUND_REASONS = frozenset({COVERS_ZONE, SPANS_ZONE, NO_RISE})


@dataclass(frozen=True)
class MaskChoice:
    """The measurement mask for one detection, or the reason there is none."""

    mask: np.ndarray | None = None
    source: str = ""
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def measurable(self) -> bool:
        return self.mask is not None and bool(np.any(self.mask))


def _zone_bounds(zone: np.ndarray | None, shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    if zone is not None and np.any(zone):
        rows, columns = np.nonzero(zone)
        return (int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max()),
                int(np.count_nonzero(zone)))
    return 0, int(shape[0]) - 1, 0, int(shape[1]) - 1, int(shape[0]) * int(shape[1])


def zone_edges_touched(mask: np.ndarray, zone: np.ndarray | None) -> int:
    """How many sides of the measurement zone this mask runs off."""
    if not np.any(mask):
        return 0
    top, bottom, left, right, _ = _zone_bounds(zone, mask.shape)
    rows, columns = np.nonzero(mask)
    margin = max(2, int(0.01 * max(bottom - top, right - left)))
    return int(sum((
        int(rows.min()) <= top + margin, int(rows.max()) >= bottom - margin,
        int(columns.min()) <= left + margin, int(columns.max()) >= right - margin,
    )))


def looks_like_background(
    mask: np.ndarray, zone: np.ndarray | None,
    *, max_fraction: float = MAX_ZONE_FRACTION, max_edges: int = MAX_ZONE_EDGES,
) -> str | None:
    """The floor, a wall or a sofa rather than something placed on the mat."""
    pixels = int(np.count_nonzero(mask))
    if pixels < 1:
        return TOO_SMALL
    top, bottom, left, right, available = _zone_bounds(zone, mask.shape)
    share = pixels / max(available, 1)
    rows, columns = np.nonzero(mask)
    across = (int(columns.max()) - int(columns.min())) / max(right - left, 1)
    down = (int(rows.max()) - int(rows.min())) / max(bottom - top, 1)
    if share >= max_fraction:
        return COVERS_ZONE
    edges = zone_edges_touched(mask, zone)
    if edges > max_edges and share >= 0.25:
        return SPANS_ZONE
    if across >= 0.9 and down >= 0.9 and share >= 0.3:
        return SPANS_ZONE
    return None


def rgb_change(frame: np.ndarray, committed: np.ndarray | None, threshold: int) -> np.ndarray | None:
    """Pixels whose colour no longer matches the committed scene."""
    if committed is None or committed.shape != frame.shape:
        return None
    difference = np.abs(frame.astype(np.int16) - committed.astype(np.int16)).max(axis=2)
    return difference >= max(1, int(threshold))


def height_change(
    depth_m: np.ndarray | None, committed_depth_m: np.ndarray | None,
    *, min_rise_m: float = MIN_HEIGHT_RISE_M,
) -> np.ndarray | None:
    """Pixels that stand above the committed scene by a real amount.

    Depth shrinks as something is placed in front of the camera, so a rise is
    a *decrease* in distance against the scene last committed.
    """
    if depth_m is None or committed_depth_m is None or depth_m.shape != committed_depth_m.shape:
        return None
    valid = np.isfinite(depth_m) & np.isfinite(committed_depth_m) & (depth_m > 0) & (committed_depth_m > 0)
    rise = np.zeros(depth_m.shape, dtype=np.float32)
    np.subtract(committed_depth_m, depth_m, out=rise, where=valid)
    return (rise >= float(min_rise_m)) & valid


def clean(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Open away speckle, close pinholes, fill what the object encloses."""
    import cv2

    binary = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        binary = filled
    if min_pixels > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < min_pixels:
                binary[labels == index] = 0
    return binary > 0


def deposit_component(
    detector_mask: np.ndarray,
    foreground_change: np.ndarray | None,
    zone: np.ndarray | None,
    *,
    rise_m: np.ndarray | None = None,
    min_pixels: int = 50,
    min_overlap: float = MIN_DETECTOR_OVERLAP,
    min_height_rise_m: float = MIN_HEIGHT_RISE_M,
    max_fraction: float = MAX_ZONE_FRACTION,
    max_edges: int = MAX_ZONE_EDGES,
) -> MaskChoice:
    """The measurement mask for one tracked detection.

    `foreground_change` is what has changed since the scene was committed;
    `rise_m` is how far each pixel stands above it, when depth can say. The
    detector's mask only chooses between the changed islands -- it never
    becomes the measurement mask while a changed island is available.
    """
    import cv2

    shape = detector_mask.shape
    in_zone = detector_mask if zone is None else (detector_mask & zone)
    diagnostics: dict[str, Any] = {
        "detector_pixels": int(np.count_nonzero(detector_mask)),
        "detector_pixels_in_zone": int(np.count_nonzero(in_zone)),
    }
    if not np.any(in_zone):
        return MaskChoice(reason=OUTSIDE_ZONE, diagnostics=diagnostics)

    diagnostics["foreground_change"] = foreground_change is not None
    if foreground_change is None:
        # No committed scene to difference against yet. The detector's own
        # mask still carries the provisional numeric result, but background
        # sized masks are refused here as they are on the measured path.
        fallback = clean(in_zone, min_pixels)
        reason = looks_like_background(fallback, zone, max_fraction=max_fraction, max_edges=max_edges)
        if reason is not None or not np.any(fallback):
            return MaskChoice(reason=reason or TOO_SMALL, diagnostics=diagnostics)
        return MaskChoice(mask=fallback, source=DETECTOR_MASK, diagnostics=diagnostics)

    candidate = foreground_change if zone is None else (foreground_change & zone)
    if not np.any(candidate):
        # There *is* a committed scene and nothing in the zone differs from
        # it: whatever the detector is pointing at was already there. An
        # empty difference is an answer, not a missing reference.
        return MaskChoice(reason=NO_NEW_DEPOSIT, diagnostics=diagnostics)

    cleaned = clean(candidate, min_pixels)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    diagnostics["changed_components"] = max(count - 1, 0)
    best_index, best_overlap, best_score = 0, 0, 0.0
    for index in range(1, count):
        island = labels == index
        overlap = int(np.count_nonzero(island & in_zone))
        if overlap == 0:
            continue
        smaller = max(min(int(stats[index, cv2.CC_STAT_AREA]), diagnostics["detector_pixels_in_zone"]), 1)
        score = overlap / smaller
        if overlap > best_overlap or (overlap == best_overlap and score > best_score):
            best_index, best_overlap, best_score = index, overlap, score
    diagnostics["detector_overlap_fraction"] = round(best_score, 4)
    if best_index == 0 or best_score < min_overlap:
        # Nothing new arrived where the detector is looking: it is pointing at
        # something that was already in the committed scene.
        return MaskChoice(reason=NO_NEW_DEPOSIT, diagnostics=diagnostics)

    chosen = labels == best_index
    diagnostics["component_pixels"] = int(np.count_nonzero(chosen))
    if diagnostics["component_pixels"] > 3 * max(diagnostics["detector_pixels_in_zone"], 1):
        # The change is far wider than the object: the room relit, the camera
        # moved, or several things arrived together. The changed island is then
        # only evidence that something is there, and the detector says which
        # part of it is this object.
        chosen = chosen & in_zone
        diagnostics["clipped_to_detector"] = True
        diagnostics["component_pixels"] = int(np.count_nonzero(chosen))
    diagnostics["zone_edges_touched"] = zone_edges_touched(chosen, zone)
    if rise_m is not None and rise_m.shape == shape:
        values = rise_m[chosen & np.isfinite(rise_m)]
        rise = float(np.percentile(values, 90)) if values.size else 0.0
        diagnostics["height_rise_m"] = round(rise, 4)
        if rise < min_height_rise_m:
            return MaskChoice(reason=NO_RISE, diagnostics=diagnostics)
    reason = looks_like_background(chosen, zone, max_fraction=max_fraction, max_edges=max_edges)
    if reason is not None:
        return MaskChoice(reason=reason, diagnostics=diagnostics)
    return MaskChoice(mask=chosen, source=FOREGROUND_COMPONENT, diagnostics=diagnostics)
