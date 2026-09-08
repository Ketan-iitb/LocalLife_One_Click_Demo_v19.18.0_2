"""Image geometry, robust foreground fallback, and object colour estimates."""

from __future__ import annotations

import colorsys
from collections import deque
from math import hypot
from typing import Iterable, Sequence

import numpy as np

from .types import Detection


def roi_pixels(
    shape: tuple[int, ...], roi: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x, y, roi_width, roi_height = roi
    x1 = int(np.clip(x, 0.0, 1.0) * width)
    y1 = int(np.clip(y, 0.0, 1.0) * height)
    x2 = int(np.clip(x + roi_width, 0.0, 1.0) * width)
    y2 = int(np.clip(y + roi_height, 0.0, 1.0) * height)
    return x1, y1, x2, y2


def roi_mask(shape: tuple[int, ...], roi: tuple[float, float, float, float]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    x1, y1, x2, y2 = roi_pixels(shape, roi)
    mask[y1:y2, x1:x2] = True
    return mask


def fixed_bin_mask(
    shape: tuple[int, ...],
    roi: tuple[float, float, float, float],
    polygon: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Limit measurements to the calibrated opening of the fixed garbage bin."""
    rectangle = roi_mask(shape, roi)
    if not polygon:
        return rectangle
    height, width = shape[:2]
    vertices = np.asarray(polygon, dtype=np.float64)
    vertices[:, 0] *= width
    vertices[:, 1] *= height
    ys, xs = np.indices((height, width), dtype=np.float64)
    xs += 0.5
    ys += 0.5
    inside = np.zeros((height, width), dtype=bool)
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > ys) != (y2 > ys)
        denominator = y2 - y1
        if abs(denominator) > 1e-12:
            intersection = (x2 - x1) * (ys - y1) / denominator + x1
            inside ^= crosses & (xs < intersection)
        previous = current
    return rectangle & inside


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return 0.0 if union <= 0 else intersection / union


def _numpy_components(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Dependency-free fallback used by tests and minimal installations."""
    visited = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    components: list[np.ndarray] = []
    for row, column in np.argwhere(binary):
        row, column = int(row), int(column)
        if visited[row, column]:
            continue
        queue: deque[tuple[int, int]] = deque([(row, column)])
        visited[row, column] = True
        coordinates: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            coordinates.append((current_row, current_column))
            for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                for next_column in range(max(0, current_column - 1), min(width, current_column + 2)):
                    if binary[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        if len(coordinates) >= min_area:
            mask = np.zeros(binary.shape, dtype=bool)
            ys, xs = zip(*coordinates)
            mask[np.asarray(ys), np.asarray(xs)] = True
            components.append(mask)
    return components


def connected_components(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    source = np.asarray(binary, dtype=np.uint8)
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
        return [
            labels == component
            for component in range(1, count)
            if int(stats[component, cv2.CC_STAT_AREA]) >= min_area
        ]
    except ImportError:
        return _numpy_components(source.astype(bool), min_area)


def detect_foreground_objects(
    frame: np.ndarray,
    baseline: np.ndarray | None,
    roi: tuple[float, float, float, float],
    *,
    threshold: int = 18,
    min_area: int = 250,
) -> list[Detection]:
    """Find every changed object instead of discarding all but the largest blob."""
    if baseline is None or frame.shape != baseline.shape:
        return []

    try:
        import cv2

        current = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        reference = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB)
        score = np.max(cv2.absdiff(current, reference), axis=2)
        changed = (score > threshold).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel, iterations=2)
    except ImportError:
        difference = np.abs(frame.astype(np.int16) - baseline.astype(np.int16))
        changed = (np.max(difference, axis=2) > threshold).astype(np.uint8)

    changed &= roi_mask(frame.shape, roi).astype(np.uint8)
    detections: list[Detection] = []
    for mask in connected_components(changed, min_area=min_area):
        ys, xs = np.where(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        detections.append(
            Detection(
                label="unclassified object",
                confidence=1.0,
                box=box,
                mask=mask,
                source="foreground-fallback",
                color=dominant_color(frame, mask),
            )
        )
    return sorted(detections, key=lambda detection: detection.area_pixels, reverse=True)


def detect_scene_objects(
    frame: np.ndarray,
    baseline: np.ndarray | None,
    depth_m: np.ndarray | None,
    baseline_depth_m: np.ndarray | None,
    roi: tuple[float, float, float, float],
    *,
    threshold: int = 18,
    min_area: int = 250,
    min_height_m: float = 0.015,
    max_height_m: float = 0.80,
    measurement_mask: np.ndarray | None = None,
) -> list[Detection]:
    """Recover whole changed objects using aligned depth and RGB together.

    A text label, shipping tape, or printed logo can be the only region YOLO
    recognizes. The empty-scene depth difference still describes the entire
    physical container and should therefore own its measurement mask.
    """
    if baseline is None or frame.shape != baseline.shape:
        return []

    difference = np.abs(frame.astype(np.int16) - baseline.astype(np.int16))
    rgb_changed = np.max(difference, axis=2) > threshold
    depth_changed = None
    if (
        depth_m is not None
        and baseline_depth_m is not None
        and depth_m.shape == frame.shape[:2]
        and baseline_depth_m.shape == depth_m.shape
    ):
        height = baseline_depth_m.astype(np.float32) - depth_m.astype(np.float32)
        depth_changed = (
            np.isfinite(depth_m)
            & np.isfinite(baseline_depth_m)
            & (depth_m > 0.10)
            & (baseline_depth_m > 0.10)
            & (height >= min_height_m)
            & (height <= max_height_m)
        )

    # A black or low-IR-reflectivity object routinely gives the depth sensor
    # scattered, patchy dropout across its own surface (documented in prior
    # rounds), so its honest depth_changed pixel count can land below
    # `min_area` even though real depth signal exists all over the object --
    # just not enough anywhere to individually be its own "component" later.
    # Previously any shortfall below `min_area` here abandoned depth
    # gating entirely and fell back to raw `rgb_changed`, which has no way
    # to tell a shadow from the object it belongs to (both are simply
    # "changed" in RGB). Anchoring on a much smaller floor -- enough to be
    # confident this is real sensor signal and not a handful of noise
    # pixels, not enough to itself pass as a reportable object -- keeps
    # shadow rejection active across exactly the patchy-depth cases that
    # previously lost it, and only falls back to unguarded RGB when there
    # is truly no usable depth information anywhere in the frame.
    depth_anchor_floor = max(20, min_area // 8)
    if depth_changed is not None and np.count_nonzero(depth_changed) >= depth_anchor_floor:
        changed = depth_changed.copy()
        try:
            import cv2

            # This union recovers an object's own weak-depth-signal edges
            # (RGB changed, right next to where depth confirms an object is
            # really there) -- it is deliberately a *tight* radius, not a
            # shadow classifier: a real object can itself be darker than the
            # background it's sitting on (a black bag, a dark backpack), so
            # judging "changed" pixels by how much darker they got would
            # exclude genuine object surface along with any nearby shadow.
            # Shadow/whole-background inflation is instead caught downstream
            # by the raw-density gate on each closed contour below, which
            # only cares whether a hull's *interior* is actually filled with
            # real changed pixels, not what shade any single pixel is.
            nearby_depth = cv2.dilate(
                depth_changed.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1
            ).astype(bool)
            changed |= rgb_changed & nearby_depth
        except ImportError:
            # Keep disconnected RGB changes out: shadows and stationary scene
            # decoration must not become part of the measured container.
            changed |= rgb_changed & depth_changed
    else:
        changed = rgb_changed

    valid_region = roi_mask(frame.shape, roi)
    if measurement_mask is not None and measurement_mask.shape == valid_region.shape:
        valid_region &= measurement_mask.astype(bool)
    changed &= valid_region
    try:
        import cv2

        # A real waste bag rarely shows one uniform "changed" blob: folds,
        # wrinkles, shadows, and patchy depth-sensor dropout on its own
        # surface routinely split one physical object into many small,
        # separated components. A 7x7 close (the previous kernel) only
        # bridges gaps a few pixels wide, so on real hardware this produced
        # a flood of tiny fragment detections for what was visually one
        # bag. Scale the merging kernel with the frame size (roughly 3% of
        # the shorter side) so nearby fragments of the same object are
        # bridged into a single connected region regardless of resolution,
        # then fill each resulting contour solid.
        short_side = min(frame.shape[0], frame.shape[1])
        merge_kernel_size = max(9, int(round(short_side * 0.035)) | 1)
        merge_kernel = np.ones((merge_kernel_size, merge_kernel_size), dtype=np.uint8)

        raw_changed_before_close = changed.copy()
        binary = changed.astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, merge_kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        # Solid-filling every closed contour's hull unconditionally is what
        # let a shadow (or any other faint, disconnected changed patch a few
        # pixels from the object) balloon into "the whole background is the
        # object": the close bridges a thin gap between the object and the
        # unrelated patch, and drawContours(FILLED) then claims the entire
        # empty area enclosed by that bridge too -- exactly how a modest
        # pillow was reported as 100+ L covering most of the frame. One real,
        # solid object (with only small internal holes from wrinkles/dropout)
        # is mostly "changed" inside its own hull once closed; a hull that
        # only *looks* solid because closing bridged two distant regions is
        # mostly untouched background inside that same hull. Only fill a hull
        # solid when the raw, pre-close changed pixels actually density it;
        # otherwise keep just the closed silhouette itself, never the empty
        # gap the bridge invented.
        solid_fill_density = 0.55
        for contour in contours:
            hull = np.zeros_like(closed)
            cv2.drawContours(hull, [contour], -1, color=1, thickness=cv2.FILLED)
            hull_area = int(np.count_nonzero(hull))
            if hull_area <= 0:
                continue
            hull_bool = hull.astype(bool)
            raw_density = int(np.count_nonzero(raw_changed_before_close & hull_bool)) / hull_area
            if raw_density >= solid_fill_density:
                filled |= hull
            else:
                filled |= (closed.astype(bool) & hull_bool).astype(np.uint8)
        changed = filled.astype(bool) & valid_region
    except ImportError:
        pass

    # Small fragments (a shadow edge, a single noisy pixel cluster, sensor
    # speckle) must not become their own "changed object" even after
    # merging. Scale the accepted minimum with how much of the measurement
    # region is available, in addition to the caller's raw pixel floor, so
    # this behaves consistently across camera resolutions.
    region_pixels = max(1, int(np.count_nonzero(valid_region)))
    effective_min_area = max(min_area, int(region_pixels * 0.01))

    detections: list[Detection] = []
    for mask in connected_components(changed, min_area=effective_min_area):
        ys, xs = np.where(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        detections.append(
            Detection(
                label="storage container",
                confidence=1.0,
                box=box,
                mask=mask,
                source="depth-scene-segmentation" if depth_changed is not None else "foreground-segmentation",
                color=dominant_color(frame, mask),
            )
        )
    return sorted(detections, key=lambda detection: detection.area_pixels, reverse=True)


def _detection_mask(detection: Detection, shape: tuple[int, int]) -> np.ndarray:
    if detection.mask is not None and detection.mask.shape == shape:
        return detection.mask.astype(bool)
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = detection.box
    mask[max(0, y1) : min(shape[0], y2), max(0, x1) : min(shape[1], x2)] = True
    return mask


def _container_priority(label: str) -> int:
    words = label.lower()
    if any(token in words for token in ("box", "parcel", "carton", "crate", "container", "bin", "basket")):
        return 3
    if any(token in words for token in ("bag", "sack", "tote", "pouch")):
        return 3
    if any(token in words for token in ("paper", "cardboard", "packaging", "label", "tape", "sticker")):
        return 0
    return 1


def _container_label(
    candidate: Detection, matches: list[Detection], best: Detection | None
) -> str:
    if best is None:
        return "unclassified object"
    if best is not None and _container_priority(best.label) >= 2:
        return best.label

    width = max(1, candidate.box[2] - candidate.box[0])
    height = max(1, candidate.box[3] - candidate.box[1])
    rectangularity = candidate.area_pixels / (width * height)
    labels = " ".join(detection.label.lower() for detection in matches)
    if any(token in labels for token in ("paper", "cardboard", "label", "parcel")):
        return "cardboard box" if rectangularity >= 0.45 else "paper bag"
    if candidate.color in {"brown", "orange", "yellow"} and rectangularity >= 0.60:
        return "cardboard box"
    return best.label


# A phantom (unclassified, 0%-confidence) recovery is the *only* place a
# "changed" region becomes a reported bag/box without any neural label
# vouching for it. Two independent shape signals distinguish "one compact
# container sitting in the bin" from "the whole scene changed" (a camera
# pan to the ceiling, a stale baseline after the rig moved, a lighting
# flip): how much of the frame it covers, and how many of the frame's four
# edges its bounding box touches. Either signal alone can have a false
# positive (a bag pushed into a corner touches two edges; a very close
# bag can fill most of the frame); both together are what real single
# waste containers essentially never produce.
_PHANTOM_MAX_AREA_FRACTION = 0.60
_PHANTOM_MAX_EDGE_TOUCHES_AT_HIGH_AREA = 2
_PHANTOM_EDGE_MARGIN_PX = 2


def _is_implausible_phantom_region(
    box: tuple[int, int, int, int], shape: tuple[int, int], frame_area: int
) -> bool:
    height, width = shape
    x1, y1, x2, y2 = box
    area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / frame_area
    edge_touches = sum(
        (
            x1 <= _PHANTOM_EDGE_MARGIN_PX,
            y1 <= _PHANTOM_EDGE_MARGIN_PX,
            x2 >= width - _PHANTOM_EDGE_MARGIN_PX,
            y2 >= height - _PHANTOM_EDGE_MARGIN_PX,
        )
    )
    if edge_touches >= 3:
        return True
    if area_fraction >= _PHANTOM_MAX_AREA_FRACTION:
        return True
    if area_fraction >= 0.35 and edge_touches >= _PHANTOM_MAX_EDGE_TOUCHES_AT_HIGH_AREA:
        return True
    return False


# A neural label only has to overlap 30% of a scene_object's mask to claim
# it (see the matching loop below), and the whole point of adopting that
# scene_object's full mask afterward is to recover a physical container's
# true extent when YOLO only tagged a small label/logo/tape patch on it --
# see `test_paper_label_expands_to_whole_cardboard_box`. That is safe when
# the scene_object is one compact blob (a real container is convex-ish and
# fills most of its own bounding box). It backfires when the scene_object is
# actually two or more physically distinct changed regions weakly bridged
# by the gap-merging morphological close in `detect_scene_objects` -- a real
# bag next to unrelated background clutter (a patterned blanket, a doorway
# edge) that happened to register a little "changed" noise of its own. That
# shape is sprawling: a large bounding box mostly empty of actual mask
# pixels. `_SPRAWLING_FILL_RATIO_THRESHOLD` catches it structurally (fill
# ratio = mask area / bounding-box area) without needing an area-ratio cap,
# which would also reject the legitimate small-label case above.
_SPRAWLING_FILL_RATIO_THRESHOLD = 0.35


# Every `Detection.source` a fused, unconfirmed (no neural label ever
# vouched for it) region can carry, regardless of `bag_only`: the literal
# "fixed-bin-depth-silhouette" phantom label used when `bag_only=True`, and
# the raw `detect_scene_objects` source strings that pass straight through
# unmatched when `bag_only=False` ("unclassified object"). A detection whose
# source is in this set carries zero semantic evidence -- only a changed
# depth/RGB blob does -- and must never be treated as confidently as a real
# neural-confirmed ("yoloe*") detection: not written to a durable ledger, not
# fused into a cross-camera result, and not tracked with the same patience as
# a confirmed object.
PHANTOM_DETECTION_SOURCES = frozenset(
    {"fixed-bin-depth-silhouette", "depth-scene-segmentation", "foreground-segmentation"}
)


def is_phantom_source(source: str) -> bool:
    return source in PHANTOM_DETECTION_SOURCES


# How `fuse_scene_detections` decides an unmatched region is "close enough"
# to an already-counted track to be that same object's own dropout gap
# rather than an unrelated new one: either box overlaps the track's box at
# all, or its center falls within one track-box-diagonal of the track's own
# center. That is generous enough to bridge a bag/box that has shifted,
# deformed, or grown a wrinkle/shadow fragment near itself, while still
# rejecting a background blob on the far side of the frame (a couch, a
# backpack, an office chair) -- the exact class of real-hardware report this
# was added for: once *any* object had been confirmed once, the old
# `has_active_counted_track()` boolean let the largest leftover "changed"
# region ANYWHERE in the frame be promoted to its own tracked "unclassified
# object", regardless of where that region actually was.
def _near_an_existing_track(
    box: tuple[int, int, int, int], track_boxes: Sequence[tuple[int, int, int, int]]
) -> bool:
    if not track_boxes:
        return False
    cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
    for track_box in track_boxes:
        if intersection_over_union(box, track_box) > 0.0:
            return True
        tcx, tcy = (track_box[0] + track_box[2]) * 0.5, (track_box[1] + track_box[3]) * 0.5
        diagonal = max(1.0, hypot(track_box[2] - track_box[0], track_box[3] - track_box[1]))
        if hypot(cx - tcx, cy - tcy) <= diagonal:
            return True
    return False


def _fused_mask_for_match(
    candidate: Detection, matches: list[Detection], shape: tuple[int, int]
) -> np.ndarray:
    candidate_mask = _detection_mask(candidate, shape)
    neural_mask = np.zeros(shape, dtype=bool)
    for match in matches:
        neural_mask |= _detection_mask(match, shape)

    box_width = max(1, candidate.box[2] - candidate.box[0])
    box_height = max(1, candidate.box[3] - candidate.box[1])
    fill_ratio = candidate.area_pixels / (box_width * box_height)
    if fill_ratio >= _SPRAWLING_FILL_RATIO_THRESHOLD or not np.any(neural_mask):
        return (candidate_mask | neural_mask).copy()

    try:
        import cv2

        neural_area = max(1, int(np.count_nonzero(neural_mask)))
        # Scale the recovery radius to the neural detection's own size (a
        # small label still recovers a modestly larger surrounding surface)
        # without reaching across an unrelated, merely-noise-bridged region
        # many times its size.
        kernel_size = max(5, int(round((neural_area**0.5) * 0.6)) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated_neural = cv2.dilate(neural_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        dilated_neural = neural_mask
    return ((candidate_mask & dilated_neural) | neural_mask).copy()


def fuse_scene_detections(
    frame: np.ndarray,
    detections: list[Detection],
    scene_objects: list[Detection],
    *,
    allow_unclassified: bool = False,
    bag_only: bool = False,
    require_scene_match: bool = False,
    counted_track_boxes: Sequence[tuple[int, int, int, int]] = (),
) -> list[Detection]:
    """Give neural labels to complete physical objects, not nested labels.

    Every neural mask is assigned to the changed component containing most of
    it. Multiple labels on one parcel therefore become one tracked container.
    Neural detections with no changed-scene match remain available unchanged.
    Unclassified movement is ignored by default because depth alone cannot
    distinguish a waste container from a person entering the camera view.
    When it is allowed (`allow_unclassified=True`, a peer camera vouches for
    "something is really there", or the region sits near an already-counted
    track's own box -- see `_near_an_existing_track`), at most ONE unmatched
    scene component -- the largest plausible one -- is ever recovered as a
    phantom detection. The fixed measurement bin holds one container at a
    time by design, so treating every leftover fragment (a shadow, a
    wrinkle, sensor speckle) as its own object is never correct and is what
    previously flooded the dashboard with duplicate boxes for a single
    physical bag. `counted_track_boxes` deliberately scopes the "an
    already-confirmed track vouches for it" case to a region near that
    track's own box, not the whole frame -- an already-confirmed bag/box
    bridges its OWN brief detector dropout, it does not license inventing an
    unrelated new object anywhere else in view.
    """
    if not scene_objects:
        return [] if require_scene_match else list(detections)

    shape = frame.shape[:2]
    assignments: dict[int, list[Detection]] = {index: [] for index in range(len(scene_objects))}
    unmatched: list[Detection] = []
    for detection in detections:
        neural_mask = _detection_mask(detection, shape)
        neural_area = max(1, int(np.count_nonzero(neural_mask)))
        scores = [
            int(np.count_nonzero(neural_mask & _detection_mask(candidate, shape))) / neural_area
            for candidate in scene_objects
        ]
        candidate_index = int(np.argmax(scores)) if scores else -1
        if candidate_index >= 0 and scores[candidate_index] >= 0.30:
            assignments[candidate_index].append(detection)
        else:
            unmatched.append(detection)

    largest_unmatched_index = -1
    largest_unmatched_area = -1
    frame_area = max(1, shape[0] * shape[1])
    for index, candidate in enumerate(scene_objects):
        if assignments[index] or candidate.area_pixels <= largest_unmatched_area:
            continue
        if _is_implausible_phantom_region(candidate.box, shape, frame_area):
            # A camera pan, a stale baseline, or a lighting change can make
            # almost the entire frame read as "changed" at once -- ceiling,
            # walls, floor together. A single physical bag/box sitting in
            # the fixed measurement bin never fills most of the frame while
            # also touching most of its edges; that shape means "the whole
            # scene moved," not "here is one container," so it must never
            # be promoted to a phantom detection even though it is the
            # largest unmatched region.
            continue
        largest_unmatched_area = candidate.area_pixels
        largest_unmatched_index = index

    fused: list[Detection] = []
    for index, candidate in enumerate(scene_objects):
        matches = assignments[index]
        if not matches:
            if index != largest_unmatched_index:
                continue
            if not (
                allow_unclassified
                or _near_an_existing_track(candidate.box, counted_track_boxes)
            ):
                continue
        best = (
            max(matches, key=lambda item: (_container_priority(item.label), item.area_pixels, item.confidence))
            if matches
            else None
        )
        mask = _fused_mask_for_match(candidate, matches, shape)
        ys, xs = np.where(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if best is None and bag_only:
            label = "garbage bag (depth silhouette)"
            confidence = 0.0
            detection_source = "fixed-bin-depth-silhouette"
        else:
            label = _container_label(candidate, matches, best)
            confidence = best.confidence if best is not None else candidate.confidence
            detection_source = "yoloe-scene-fusion" if best is not None else candidate.source
        fused.append(
            Detection(
                label=label,
                confidence=confidence,
                box=box,
                mask=mask,
                source=detection_source,
                color=dominant_color(frame, mask),
            )
        )

    retained_unmatched = [] if require_scene_match else unmatched
    return sorted(fused + retained_unmatched, key=lambda item: item.area_pixels, reverse=True)


def _lab_b_channel(blue: float, green: float, red: float) -> float:
    """OpenCV's 8-bit LAB b* channel for one BGR colour, correctly re-centred.

    OpenCV's `cv2.COLOR_BGR2LAB` encodes L in [0, 255] (not the conventional
    [0, 100]) and a/b in [0, 255] with 128 as the neutral midpoint (not the
    conventional roughly [-128, 127] centred on 0). Reading OpenCV's raw a/b
    bytes directly -- without subtracting 128 -- silently treats a strongly
    warm/cool colour as barely warm/cool at all, since every value is offset
    upward by 128. b* is the blue<->yellow axis: positive after this
    correction means "yellow-leaning", negative means "blue-leaning", and
    values near zero are genuinely chromatically neutral. `blue`/`green`/
    `red` are 0.0-1.0 floats (this function's callers already have those).
    Returns 0.0 (chromatically neutral) if OpenCV is unavailable, which
    degrades to the pre-existing HSV-only behaviour rather than crashing.
    """
    try:
        import cv2
    except ImportError:
        return 0.0
    pixel = np.array([[[blue * 255.0, green * 255.0, red * 255.0]]], dtype=np.float32)
    lab = cv2.cvtColor(np.clip(pixel, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB)
    return float(lab[0, 0, 2]) - 128.0


def dominant_color(frame_bgr: np.ndarray, mask: np.ndarray | None) -> str:
    if mask is None or frame_bgr.shape[:2] != mask.shape or np.count_nonzero(mask) < 20:
        return "unknown"

    material = mask.astype(bool)
    # Open waste bags expose their contents in the middle. Sampling the whole
    # instance therefore called a green bag full of orange/yellow material
    # "orange". The inner contour band is the visible bag skin/edge and is a
    # much better material-colour sample. Solid bags still produce the same
    # answer, while tiny masks safely retain the original full-mask fallback.
    area = int(np.count_nonzero(material))
    if area >= 120:
        try:
            import cv2

            radius = max(1, min(7, int(np.sqrt(area) * 0.035)))
            eroded = cv2.erode(
                material.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=radius
            ).astype(bool)
            boundary = material & ~eroded
            if int(np.count_nonzero(boundary)) >= max(20, int(area * 0.06)):
                material = boundary
        except ImportError:
            from numpy.lib.stride_tricks import sliding_window_view

            radius = max(1, min(7, int(np.sqrt(area) * 0.035)))
            eroded = material.copy()
            for _ in range(radius):
                windows = sliding_window_view(
                    np.pad(eroded.astype(np.uint8), 1, mode="constant"), (3, 3)
                )
                eroded = np.all(windows, axis=(-2, -1))
            boundary = material & ~eroded
            if int(np.count_nonzero(boundary)) >= max(20, int(area * 0.06)):
                material = boundary

    selected = frame_bgr[material].astype(np.float32)
    blue, green, red = (float(value) / 255.0 for value in np.median(selected, axis=0))
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    degrees = hue * 360.0

    if value < 0.18:
        return "black"
    if saturation < 0.16:
        # HSV saturation alone cannot tell "genuinely neutral grey/white"
        # from "a pale, washed-out warm surface" -- a cream milk carton or a
        # translucent yellow-tinted bag both have a small max-min channel
        # spread (hence low HSV saturation) while still being unmistakably
        # warm-toned, not neutral. This was silently returning "grey" for
        # exactly that case (reported directly: a yellow bag/box labelled
        # grey). LAB b* (see `_lab_b_channel`) is largely independent
        # evidence: a genuinely neutral grey/white/black surface has b*
        # close to zero regardless of exposure, while a pale yellow surface
        # still carries a real positive b* even at low HSV saturation. Only
        # fall through to neutral white/grey/black once LAB agrees there is
        # no real warm tint.
        if _lab_b_channel(blue, green, red) > 6.0 and value > 0.40:
            return "yellow"
        if value > 0.77:
            return "white"
        return "grey" if value > 0.33 else "black"
    if degrees < 12 or degrees >= 345:
        return "red"
    if degrees < 38:
        return "orange" if value >= 0.50 else "brown"
    if degrees < 70:
        return "yellow" if value >= 0.52 else "brown"
    if degrees < 165:
        return "green"
    if degrees < 195:
        return "cyan"
    if degrees < 260:
        return "blue"
    if degrees < 320:
        return "purple"
    return "pink"


def combined_mask(detections: Iterable[Detection], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for detection in detections:
        if detection.mask is not None and detection.mask.shape == shape:
            union |= detection.mask.astype(bool)
            continue
        x1, y1, x2, y2 = detection.box
        union[max(0, y1) : min(shape[0], y2), max(0, x1) : min(shape[1], x2)] = True
    return union
