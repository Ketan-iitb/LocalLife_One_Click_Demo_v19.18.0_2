"""Permanent measurement identity, independent of the detector's track numbers.

The problem this solves
-----------------------
The detector renumbers a stationary object: a pillow observed continuously went
from track 10 to track 52, a can from 54 to 63. Anything keyed on that number
therefore split one physical object into several "measurements" -- several
history rows, several CSV rows, an inflated bag count.

A detector track id is a per-frame association hint. It is not an identity, and
it is never used as one here.

Two levels
----------
*Provisional*: while an object is being watched, detections are matched to a
candidate by physical continuity -- mask overlap, centroid distance, depth,
footprint size, and time since last seen. Deliberately not by class label or
colour: those are exactly the attributes that flicker frame to frame, and
matching on them would re-create the problem.

*Permanent*: once a candidate has held still and measured consistently for a
configured window, it is accepted and given a monotonically increasing
`event_id`. That id, and the robust median of its measurements, are then
**frozen**. A committed object survives the detector renumbering it, changing
its mind about the class, briefly losing confidence, or reshaping the mask.

The live card may keep showing the latest noisy estimate; committed history does
not move.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

# candidate -> tracking -> stable -> accepted/rejected -> persisted -> committed
LIFECYCLE = ("candidate", "tracking", "stable", "accepted", "rejected", "persisted", "committed")


@dataclass
class StabilitySettings:
    """Tuned from the cameras actually in use, not from round numbers.

    The RealSense D435 runs at 848x480 here, so a centroid drift of 40 px is
    about 5% of the frame width -- large enough to absorb mask jitter on a
    deformable bag, small enough that a genuinely moved object fails it. The
    depth bound of 25 mm sits above the sensor's own noise at bin distance
    (~1.5 m) without accepting a lift or a slump.
    """

    window_frames: int = 12
    min_valid_stable_frames: int = 6
    max_centroid_shift_px: float = 40.0
    min_mask_iou: float = 0.45
    max_depth_change_mm: float = 25.0
    max_volume_variation_percent: float = 12.0
    finalisation_hold_seconds: float = 1.0
    # How long a committed object may be unseen before its slot is released.
    # Short enough that a genuine removal frees the scene, long enough that a
    # few dropped frames do not.
    absent_release_seconds: float = 3.0

    def validate(self) -> None:
        if self.min_valid_stable_frames > self.window_frames:
            raise ValueError("min_valid_stable_frames cannot exceed window_frames")
        if self.window_frames < 2 or self.min_valid_stable_frames < 2:
            raise ValueError("a stability window needs at least two frames")


@dataclass
class Observation:
    """One frame's view of a candidate. Only physical quantities."""

    timestamp: float
    centroid: tuple[float, float]
    box: tuple[float, float, float, float]
    depth_mm: float | None = None
    volume_l: float | None = None
    length_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    label: str | None = None
    color: str | None = None
    material: str | None = None
    sorting_status: str | None = None
    color_confidence: float | None = None
    material_confidence: float | None = None
    confidence: float | None = None
    detector_track_id: int | None = None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - overlap
    return 0.0 if union <= 0 else overlap / union


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _robust(values: list[float]) -> float | None:
    """Median, after dropping values more than half a median away.

    A single badly segmented frame can otherwise drag the frozen volume; the
    median alone resists that, and the outlier drop resists a run of them.
    """
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    centre = median(usable)
    if centre == 0:
        return round(centre, 6)
    kept = [value for value in usable if abs(value - centre) <= abs(centre) * 0.5]
    return round(median(kept or usable), 6)


@dataclass
class StableObject:
    """One physical object, from first sighting to committed measurement."""

    internal_id: int
    state: str = "candidate"
    event_id: int | None = None
    observations: list[Observation] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    frozen: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None
    # Every detector track number this object has worn. Kept because it is the
    # evidence that renumbering happened, and it makes the behaviour auditable.
    detector_track_ids: list[int] = field(default_factory=list)

    @property
    def committed(self) -> bool:
        return self.state in {"accepted", "persisted", "committed"}

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.last_seen = observation.timestamp
        if not self.first_seen:
            self.first_seen = observation.timestamp
        track_id = observation.detector_track_id
        if track_id is not None and track_id not in self.detector_track_ids:
            self.detector_track_ids.append(track_id)
        if self.state == "candidate" and len(self.observations) >= 2:
            self.state = "tracking"

    def window(self, size: int) -> list[Observation]:
        return self.observations[-size:]

    def is_stable(self, settings: StabilitySettings) -> tuple[bool, str | None]:
        """Has this held still and measured consistently for long enough?

        Returns the reason it is *not* stable, so a caller can say why nothing
        was recorded rather than leaving the operator guessing.
        """
        window = self.window(settings.window_frames)
        if len(window) < settings.min_valid_stable_frames:
            return False, "insufficient_stable_frames"
        if window[-1].timestamp - window[0].timestamp < settings.finalisation_hold_seconds:
            return False, "finalisation_hold_not_met"
        reference = window[0]
        for observation in window[1:]:
            if _distance(observation.centroid, reference.centroid) > settings.max_centroid_shift_px:
                return False, "unstable_position"
            if _iou(observation.box, reference.box) < settings.min_mask_iou:
                return False, "unstable_mask"
            if (
                observation.depth_mm is not None
                and reference.depth_mm is not None
                and abs(observation.depth_mm - reference.depth_mm) > settings.max_depth_change_mm
            ):
                return False, "unstable_depth"
        volumes = [item.volume_l for item in window if item.volume_l is not None]
        if len(volumes) < settings.min_valid_stable_frames:
            return False, "insufficient_volume_samples"
        centre = median(volumes)
        if centre > 0:
            spread = (max(volumes) - min(volumes)) / centre * 100.0
            if spread > settings.max_volume_variation_percent:
                return False, "unstable_volume"
        return True, None

    def freeze(self, settings: StabilitySettings, event_id: int) -> dict[str, Any]:
        """Lock the permanent id and the robust final values.

        Frozen from the median of the window, not from the latest frame: the
        last frame before acceptance is not more truthful than the others, and
        it is just as likely to be the noisy one.
        """
        window = self.window(settings.window_frames)

        def _numeric(attribute: str) -> float | None:
            return _robust([
                getattr(item, attribute) for item in window
                if getattr(item, attribute) is not None
            ])

        def _mode(attribute: str) -> Any:
            values = [getattr(item, attribute) for item in window if getattr(item, attribute)]
            if not values:
                return None
            return max(set(values), key=values.count)

        self.event_id = event_id
        self.state = "accepted"
        self.frozen = {
            "event_id": event_id,
            "volume_l": _numeric("volume_l"),
            "length_mm": _numeric("length_mm"),
            "width_mm": _numeric("width_mm"),
            "height_mm": _numeric("height_mm"),
            "depth_mm": _numeric("depth_mm"),
            "confidence": _numeric("confidence"),
            "color_confidence": _numeric("color_confidence"),
            "material_confidence": _numeric("material_confidence"),
            # The value the window agreed on most often, not the last one the
            # detector happened to emit.
            "object_type": _mode("label"),
            "color": _mode("color"),
            "material": _mode("material"),
            "sorting_status": _mode("sorting_status"),
            "frames_used": len(window),
            "detector_track_ids": list(self.detector_track_ids),
        }
        return dict(self.frozen)


class StableObjectRegistry:
    """Associates detections to physical objects and owns the permanent ids.

    Holds no image data: callers pass geometry that has already been computed,
    which keeps this testable without a camera and keeps the measurement
    pipeline unchanged.
    """

    def __init__(self, settings: StabilitySettings | None = None, *, session_id: str = "") -> None:
        self.settings = settings or StabilitySettings()
        self.settings.validate()
        self.session_id = session_id
        self.objects: dict[int, StableObject] = {}
        self._next_internal = 1
        self._next_event = 1

    # ------------------------------------------------------------ matching
    def _match(self, observation: Observation) -> StableObject | None:
        """Best physical match, or None for a genuinely new object.

        Scored on overlap first, then centroid proximity: two candidates rarely
        tie on both, and overlap is the more reliable of the two for a
        deformable bag whose centroid wanders as it settles.
        """
        best: tuple[float, StableObject] | None = None
        for candidate in self.objects.values():
            if not candidate.observations:
                continue
            if observation.timestamp - candidate.last_seen > self.settings.absent_release_seconds:
                continue
            last = candidate.observations[-1]
            overlap = _iou(observation.box, last.box)
            distance = _distance(observation.centroid, last.centroid)
            if overlap < self.settings.min_mask_iou and distance > self.settings.max_centroid_shift_px:
                continue
            if (
                observation.depth_mm is not None
                and last.depth_mm is not None
                and abs(observation.depth_mm - last.depth_mm) > self.settings.max_depth_change_mm * 3
            ):
                continue
            score = overlap + max(0.0, 1.0 - distance / max(1.0, self.settings.max_centroid_shift_px))
            if best is None or score > best[0]:
                best = (score, candidate)
        return None if best is None else best[1]

    def observe(self, observation: Observation) -> StableObject:
        """Feed one detection. Returns the object it belongs to."""
        matched = self._match(observation)
        if matched is None:
            matched = StableObject(internal_id=self._next_internal)
            self.objects[self._next_internal] = matched
            self._next_internal += 1
        matched.add(observation)
        return matched

    # ------------------------------------------------------------ accepting
    def accept(self, item: StableObject) -> dict[str, Any] | None:
        """Assign the permanent id and freeze, if it is ready and not already done.

        Returns the frozen values only on the transition, so a caller can
        persist exactly once. An object that is already committed returns None
        however many frames it is offered on.
        """
        if item.committed:
            return None
        stable, reason = item.is_stable(self.settings)
        if not stable:
            item.rejection_reason = reason
            return None
        item.rejection_reason = None
        frozen = item.freeze(self.settings, self._next_event)
        self._next_event += 1
        return frozen

    def mark_persisted(self, item: StableObject) -> None:
        if item.state == "accepted":
            item.state = "persisted"

    def mark_committed(self, item: StableObject) -> None:
        if item.state in {"accepted", "persisted"}:
            item.state = "committed"

    def reject(self, item: StableObject, reason: str) -> None:
        item.state = "rejected"
        item.rejection_reason = reason

    # ------------------------------------------------------------- cleanup
    def release_absent(self, now: float | None = None) -> list[StableObject]:
        """Drop objects that have genuinely left the scene.

        A committed object keeps its slot while it is still visible -- that is
        what stops a stationary deposit being counted again -- and is released
        only once it has been gone for `absent_release_seconds`.
        """
        moment = time.time() if now is None else now
        released = []
        for internal_id, item in list(self.objects.items()):
            if moment - item.last_seen > self.settings.absent_release_seconds:
                released.append(self.objects.pop(internal_id))
        return released

    def reset(self) -> None:
        """Session reset. Permanent ids keep counting up; they are never reused."""
        self.objects.clear()

    # -------------------------------------------------------------- reading
    def committed_objects(self) -> list[StableObject]:
        return [item for item in self.objects.values() if item.committed]

    def accepted_count(self) -> int:
        """Bag count: accepted deposit events, never frame detections."""
        return self._next_event - 1

    def state(self) -> dict[str, Any]:
        return {
            "tracked": len(self.objects),
            "committed": len(self.committed_objects()),
            "accepted_events": self.accepted_count(),
            "settings": {
                "STABILITY_WINDOW_FRAMES": self.settings.window_frames,
                "MIN_VALID_STABLE_FRAMES": self.settings.min_valid_stable_frames,
                "MAX_CENTROID_SHIFT_PX": self.settings.max_centroid_shift_px,
                "MIN_MASK_IOU": self.settings.min_mask_iou,
                "MAX_DEPTH_CHANGE_MM": self.settings.max_depth_change_mm,
                "MAX_VOLUME_VARIATION_PERCENT": self.settings.max_volume_variation_percent,
                "FINALISATION_HOLD_SECONDS": self.settings.finalisation_hold_seconds,
            },
        }


def observations_from(detections: Iterable[Any], timestamp: float) -> list[Observation]:
    """Adapt pipeline Detections without importing them, so this stays testable."""
    result = []
    for detection in detections:
        box = tuple(float(value) for value in detection.box)
        result.append(Observation(
            timestamp=timestamp,
            centroid=((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
            box=box,  # type: ignore[arg-type]
            depth_mm=getattr(detection, "physical_height_mm", None),
            volume_l=getattr(detection, "realsense_volume_l", None),
            length_mm=getattr(detection, "footprint_length_mm", None),
            width_mm=getattr(detection, "footprint_width_mm", None),
            height_mm=getattr(detection, "physical_height_mm", None),
            label=getattr(detection, "label", None),
            color=getattr(detection, "color", None),
            material=getattr(detection, "material", None),
            sorting_status=getattr(detection, "sorting_status", None),
            color_confidence=getattr(detection, "color_confidence", None),
            material_confidence=getattr(detection, "material_confidence", None),
            confidence=getattr(detection, "confidence", None),
            detector_track_id=getattr(detection, "track_id", None),
        ))
    return result
