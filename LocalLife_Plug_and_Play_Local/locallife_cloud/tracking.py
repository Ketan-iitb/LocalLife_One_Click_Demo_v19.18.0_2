"""Dropout-resistant tracking that counts each physical object once."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, log

from .geometry import intersection_over_union, is_phantom_source
from .types import Detection


def _family(label: str) -> str:
    words = set(label.lower().replace("-", " ").replace("_", " ").split())
    if words & {"bag", "bags", "sack", "sacks", "tote", "pouch"}:
        return "bag"
    if words & {"box", "boxes", "cardboard", "carton", "parcel"}:
        return "box"
    return "other"


def _detection_family(detection: Detection) -> str:
    # Geometry-validation prompts can call the same rigid object "book",
    # "storage container", or "cardboard box" on adjacent frames. They are
    # all intentionally accepted as one generic measurement class, so do not
    # break the track (and discard its dimension history) when only that raw
    # open-vocabulary label changes.
    if detection.accepted_class == "measurement_object":
        words = set(detection.label.lower().replace("-", " ").replace("_", " ").split())
        if words & {
            "box", "boxes", "carton", "cartons", "parcel", "parcels",
            "package", "packages", "container", "containers", "book", "books",
            "shoebox",
        }:
            return "measurement_rigid"
        return f"measurement_{_family(detection.label)}"
    return _family(detection.label)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _area(box: tuple[int, int, int, int]) -> float:
    return float(max(1, box[2] - box[0]) * max(1, box[3] - box[1]))


def _clone_detection(detection: Detection) -> Detection:
    return replace(detection, mask=None if detection.mask is None else detection.mask.copy())


# A phantom/silhouette detection (see geometry.fuse_scene_detections) carries
# zero semantic confirmation -- no neural label ever vouched for it, only a
# changed-depth blob did. Tracking it at all exists purely to bridge a real,
# already-confirmed bag through a brief detector dropout. Letting it live for
# the same ~24-frame grace window as a real, neural-confirmed track is what
# let unrelated background noise (a blanket fold, a doorway edge) accumulate
# into its own long-lived "object" alongside the real one, since a *different*
# region can be the single largest unmatched blob from one frame to the next
# and each becomes its own track. Phantom tracks therefore decay fast, and at
# most one is ever kept alive at a time -- the fixed measurement bin holds one
# physical object, so two simultaneous phantom boxes are never both real.
def _is_phantom(track: "Track") -> bool:
    return track.last_detection is not None and is_phantom_source(track.last_detection.source)


@dataclass(slots=True)
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    label: str
    hits: int = 1
    missing: int = 0
    counted: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    last_detection: Detection | None = None


class ObjectTracker:
    """Associate deforming bags through scale changes and brief detector gaps."""

    def __init__(
        self,
        *,
        confirmation_frames: int = 2,
        max_missing_frames: int = 24,
        minimum_iou: float = 0.08,
        maximum_center_distance: float = 0.38,
        phantom_max_missing_frames: int = 2,
    ) -> None:
        self.confirmation_frames = max(1, confirmation_frames)
        self.max_missing_frames = max(0, max_missing_frames)
        self.minimum_iou = max(0.0, float(minimum_iou))
        self.maximum_center_distance = max(0.01, float(maximum_center_distance))
        self.phantom_max_missing_frames = max(0, min(phantom_max_missing_frames, self.max_missing_frames))
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.total_count = 0

    def _association(self, track: Track, detection: Detection) -> tuple[bool, float]:
        track_family = (
            _detection_family(track.last_detection)
            if track.last_detection is not None else _family(track.label)
        )
        if track_family != _detection_family(detection):
            return False, -1.0
        overlap = intersection_over_union(track.box, detection.box)
        tcx, tcy = _center(track.box)
        dcx, dcy = _center(detection.box)
        diagonal = max(1.0, hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]))
        distance = hypot(dcx - (tcx + track.velocity_x), dcy - (tcy + track.velocity_y)) / diagonal
        scale_penalty = abs(log(_area(detection.box) / _area(track.box)))
        allowed = (
            overlap >= self.minimum_iou
            or (distance <= self.maximum_center_distance and scale_penalty <= 2.30)
        )
        score = overlap * 2.4 + max(0.0, 1.0 - distance) - 0.20 * scale_penalty
        return allowed, score

    def update(self, detections: list[Detection]) -> list[int]:
        for track in self.tracks.values():
            track.missing += 1

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                allowed, score = self._association(track, detection)
                if allowed:
                    candidates.append((score, track_id, index))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            track = self.tracks[track_id]
            old_x, old_y = _center(track.box)
            new_x, new_y = _center(detection.box)
            track.velocity_x = 0.65 * track.velocity_x + 0.35 * (new_x - old_x)
            track.velocity_y = 0.65 * track.velocity_y + 0.35 * (new_y - old_y)
            track.box = detection.box
            track.label = detection.label
            track.hits += 1
            track.missing = 0
            detection.track_id = track_id
            track.last_detection = _clone_detection(detection)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track = Track(
                self.next_track_id,
                detection.box,
                detection.label,
                last_detection=_clone_detection(detection),
            )
            self.tracks[track.track_id] = track
            detection.track_id = track.track_id
            track.last_detection.track_id = track.track_id
            self.next_track_id += 1

        # At most one phantom (unconfirmed depth-silhouette) track is ever
        # allowed to stay alive. `fuse_scene_detections` already caps *new*
        # phantom promotions to one per frame, but without this a phantom
        # track from an earlier frame (still inside its own missing-frame
        # grace period) can keep existing at the same time a *different*
        # region gets promoted this frame, producing two simultaneous
        # unconfirmed boxes for what is never more than one physical object.
        fresh_phantom_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.missing == 0 and _is_phantom(track)
        ]
        if fresh_phantom_ids:
            keep = fresh_phantom_ids[-1]
            for track_id in list(self.tracks.keys()):
                if track_id == keep:
                    continue
                track = self.tracks[track_id]
                if track.missing > 0 and _is_phantom(track):
                    del self.tracks[track_id]

        new_ids: list[int] = []
        for track_id, track in list(self.tracks.items()):
            limit = self.phantom_max_missing_frames if _is_phantom(track) else self.max_missing_frames
            if track.missing > limit:
                del self.tracks[track_id]
                continue
            if not track.counted and track.hits >= self.confirmation_frames:
                track.counted = True
                self.total_count += 1
                new_ids.append(track_id)
        return new_ids

    def predicted_detections(self, maximum_missing: int) -> list[Detection]:
        predicted: list[Detection] = []
        for track in self.tracks.values():
            if not track.counted or track.last_detection is None or not 1 <= track.missing <= maximum_missing:
                continue
            item = _clone_detection(track.last_detection)
            item.track_id = track.track_id
            item.source = "tracked-prediction"
            item.tracking_status = "predicted"
            item.confidence = max(0.0, item.confidence * (0.88 ** track.missing))
            item.measurement_quality = "tracking-through-brief-detector-dropout"
            predicted.append(item)
        return predicted

    def remember(self, detection: Detection) -> None:
        """Store the post-smoothed display values for future dropout recovery."""
        if detection.track_id is None:
            return
        track = self.tracks.get(detection.track_id)
        if track is not None and track.missing == 0:
            track.last_detection = _clone_detection(detection)

    def has_active_counted_track(self) -> bool:
        return any(track.counted and track.missing <= self.max_missing_frames for track in self.tracks.values())

    def counted_track_boxes(self) -> list[tuple[int, int, int, int]]:
        """Boxes of every currently counted (already-confirmed) track still
        inside its normal missing-frame grace window.

        This exists so phantom/unclassified-foreground recovery in
        `geometry.fuse_scene_detections` can be scoped to *bridging that
        specific object* through a brief detector dropout -- never to
        license inventing an unrelated "changed" region anywhere else in
        the frame just because something, somewhere, is already being
        tracked. `has_active_counted_track()` above answers a plain yes/no
        question and was, for a while, the only signal `fuse_scene_detections`
        had to go on; using it alone let a real, once-confirmed bag/box
        "vouch" for an unrelated background blob (a couch cushion, a
        backpack, a chair) clear across the frame, which is exactly the
        drifting/mislabelled "unclassified object" boxes reported from real
        hardware. Callers should test proximity to one of these boxes, not
        just call `has_active_counted_track()`, before promoting a phantom.
        """
        return [
            track.box
            for track in self.tracks.values()
            if track.counted and track.missing <= self.max_missing_frames
        ]

    def reset(self) -> None:
        self.tracks.clear()
        self.total_count = 0
        self.next_track_id = 1
