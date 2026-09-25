"""How full the bin was before a bag went in, and how full it is after.

This is the thesis measurement, and the system did not have it. What it had was
a per-object volume: the litres of the bag the detector is looking at right now.
That is a different quantity, and in a bin that already holds waste it is not
the one that matters. A bag dropped onto a pile compresses it; another slides
into a gap and raises the surface by less than its own size; a third lands on
top and raises it by more. The occupancy change is what the bin experienced.
The bag's own envelope is what the bag is. Both are reported, and they are
never presented as the same number.

Occupied volume, operationally
------------------------------
The volume enclosed below the visible waste surface, measured against the bin's
own floor plane inside its own ROI, in that camera's own coordinates. It is an
estimate from one viewpoint. It is not the solid material volume, not the
printed capacity of the bags, and it cannot see air pockets under the surface --
a bag bridging a gap reads as though the space beneath it were full. Every
number this module produces carries that meaning and no other.

Absolute occupancy needs the empty bin's floor plane, saved once when the
installation is set up. Without it the surface can still be compared against
itself frame to frame, so the *change* is measurable while the absolute figure
is not. That case is reported as exactly that, rather than by inventing a floor
under the waste.

The event
---------
    STABLE_PRE -> DEPOSIT_IN_PROGRESS -> SETTLING -> STABLE_POST
               -> FINALIZED -> STABLE_PRE

The pre-state is frozen the moment a deposit starts, so the frames of the bag
falling cannot creep into the "before" reading -- which is how a before/after
pair silently becomes an after/after one. Finalisation happens once per event,
after several sufficiently still frames, on the median of those frames. The
committed pre-state then advances to the finalised post-state, so the next
deposit is measured against the bin as it now is, and the waste already there
is never counted again.

A negative or near-zero delta is kept as it is measured. Waste settles and
compresses, and an event that lowered the surface is a real observation about
the bin; clamping it to a positive "bag volume" would turn the measurement into
an assumption.

Each camera runs its own tracker over its own depth, plane and ROI. Nothing
here reads another camera's data; events are paired afterwards by timestamp for
comparison only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

STABLE_PRE = "STABLE_PRE"
DEPOSIT_IN_PROGRESS = "DEPOSIT_IN_PROGRESS"
SETTLING = "SETTLING"
STABLE_POST = "STABLE_POST"
FINALIZED = "FINALIZED"

STATES = (STABLE_PRE, DEPOSIT_IN_PROGRESS, SETTLING, STABLE_POST, FINALIZED)

MESSAGES = {
    STABLE_PRE: "Bin is settled; occupancy is being tracked",
    DEPOSIT_IN_PROGRESS: "Something is entering the bin",
    SETTLING: "Waiting for the waste to settle",
    STABLE_POST: "Settled; aggregating the after reading",
    FINALIZED: "Event finalised",
}

# Why an event could not be finalised. Each names the thing that is wrong, so
# the dashboard never shows a bare "pending".
NO_OCCUPANCY = "occupancy_unavailable_no_valid_surface"
NO_ABSOLUTE = "absolute_occupancy_unavailable_no_bin_profile"
VIEW_OBSTRUCTED = "view_obstructed_during_the_deposit"
SCENE_UNSETTLED = "waste_has_not_settled"
CAMERA_MOVED = "camera_or_bin_geometry_moved"
NO_PRE_STATE = "no_committed_before_reading"

# A surface reading needs this much of the bin ROI to carry valid depth before
# it is used as an occupancy figure at all.
MIN_VALID_FRACTION = 0.35
# Frames of a still surface before the post reading is taken.
REQUIRED_STABLE_FRAMES = 5
# How much the occupancy may wander between frames and still count as settled,
# as a fraction of the reading itself plus a floor for a nearly empty bin.
SETTLE_TOLERANCE = 0.06
SETTLE_FLOOR_L = 0.25
# A change smaller than this is the room breathing rather than a deposit.
ENTER_CHANGED_FRACTION = 0.02


@dataclass(frozen=True)
class OccupancyReading:
    """One frame's view of how full the bin is."""

    litres: float | None
    valid_fraction: float = 0.0
    absolute: bool = False
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return (
            self.litres is not None
            and self.reason is None
            and self.valid_fraction >= MIN_VALID_FRACTION
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "litres": None if self.litres is None else round(self.litres, 3),
            "valid_fraction": round(self.valid_fraction, 3),
            "absolute": self.absolute,
            "reason": self.reason,
        }


@dataclass
class OccupancyEvent:
    """One finalised deposit, as the bin experienced it."""

    event_id: str
    camera: str
    started_at: float
    finalized_at: float
    occupied_before_l: float | None
    occupied_after_l: float | None
    delta_occupancy_l: float | None
    absolute: bool
    status: str
    reason: str | None = None
    frames_aggregated: int = 0
    label: str | None = None
    color: str | None = None
    track_id: int | None = None
    # The deposited bag's own outer size, when one object was measured. This is
    # NOT the occupancy change and is never added to it.
    envelope_mm: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera": self.camera,
            "started_at": self.started_at,
            "finalized_at": self.finalized_at,
            "occupied_before_l": self.occupied_before_l,
            "occupied_after_l": self.occupied_after_l,
            "delta_occupancy_l": self.delta_occupancy_l,
            "absolute_occupancy": self.absolute,
            "status": self.status,
            "reason": self.reason,
            "frames_aggregated": self.frames_aggregated,
            "object_class": self.label,
            "object_color": self.color,
            "track_id": self.track_id,
            "object_envelope_mm": (
                None if self.envelope_mm is None else list(self.envelope_mm)
            ),
            # Said in every record, because the number is meaningless without it.
            "occupancy_definition": (
                "volume enclosed below the visible waste surface, relative to the "
                "bin floor, from this camera's viewpoint; hidden voids are not visible"
            ),
        }


@dataclass
class DepositObservation:
    """What one frame saw of the bin and of whatever is being put into it."""

    reading: OccupancyReading
    changed_fraction: float = 0.0
    tracked_objects: int = 0
    obstructed: bool = False
    geometry_moved: bool = False
    label: str | None = None
    color: str | None = None
    track_id: int | None = None
    envelope_mm: tuple[float, float, float] | None = None
    timestamp: float | None = None


class BinOccupancyTracker:
    """One camera's before/after occupancy, as a state machine.

    The tracker never reads a frame itself. The pipeline hands it one
    observation per frame and reads back the state, the committed before
    reading, and any event that has just finalised.
    """

    def __init__(
        self,
        camera: str,
        *,
        required_stable_frames: int = REQUIRED_STABLE_FRAMES,
        settle_tolerance: float = SETTLE_TOLERANCE,
        enter_changed_fraction: float = ENTER_CHANGED_FRACTION,
    ) -> None:
        self.camera = camera
        self.required_stable_frames = max(1, int(required_stable_frames))
        self.settle_tolerance = float(settle_tolerance)
        self.enter_changed_fraction = float(enter_changed_fraction)
        self.state = STABLE_PRE
        # The bin as it stood before the deposit now in progress. Frozen on
        # entry, so the frames of the bag falling cannot replace it.
        self.committed_before: OccupancyReading | None = None
        self.frozen_before: OccupancyReading | None = None
        self.pending_reason: str | None = NO_PRE_STATE
        self.events: list[OccupancyEvent] = []
        self._post_samples: list[float] = []
        self._recent: list[float] = []
        self._started_at: float | None = None
        self._last_object: dict[str, Any] = {}
        self._sequence = 0
        self._obstructed_during_event = False

    # ---------------------------------------------------------------- helpers
    def _settled(self, litres: float) -> bool:
        """Has the surface stopped moving?"""
        if len(self._recent) < 2:
            return False
        window = self._recent[-self.required_stable_frames:]
        if len(window) < min(2, self.required_stable_frames):
            return False
        centre = median(window)
        allowed = max(SETTLE_FLOOR_L, self.settle_tolerance * abs(centre))
        return (max(window) - min(window)) <= allowed

    def _new_event_id(self, timestamp: float) -> str:
        self._sequence += 1
        return f"{self.camera}-occ-{int(timestamp)}-{self._sequence:03d}"

    def _finalize(self, timestamp: float) -> OccupancyEvent:
        before = self.frozen_before or self.committed_before
        after_l = median(self._post_samples) if self._post_samples else None
        status = "finalized"
        reason: str | None = None
        delta: float | None = None
        if self._obstructed_during_event:
            status, reason = "pending", VIEW_OBSTRUCTED
        elif before is None or before.litres is None:
            status, reason = "relative_unavailable", NO_PRE_STATE
        elif after_l is None:
            status, reason = "pending", NO_OCCUPANCY
        else:
            delta = float(after_l) - float(before.litres)
        absolute = bool(before is not None and before.absolute
                        and after_l is not None)
        if status == "finalized" and not absolute:
            # The change is measurable, the absolute figure is not.
            reason = NO_ABSOLUTE
        event = OccupancyEvent(
            event_id=self._new_event_id(timestamp),
            camera=self.camera,
            started_at=self._started_at or timestamp,
            finalized_at=timestamp,
            occupied_before_l=(
                None if before is None or before.litres is None
                else round(before.litres, 3)
            ),
            occupied_after_l=None if after_l is None else round(float(after_l), 3),
            delta_occupancy_l=None if delta is None else round(delta, 3),
            absolute=absolute,
            status=status,
            reason=reason,
            frames_aggregated=len(self._post_samples),
            label=self._last_object.get("label"),
            color=self._last_object.get("color"),
            track_id=self._last_object.get("track_id"),
            envelope_mm=self._last_object.get("envelope_mm"),
        )
        self.events.append(event)
        del self.events[:-64]
        # The bin as it now is becomes the reference for the next deposit, so
        # what has just been counted is never counted again.
        if after_l is not None:
            self.committed_before = OccupancyReading(
                litres=float(after_l),
                valid_fraction=1.0,
                absolute=absolute,
            )
        self.frozen_before = None
        self._post_samples.clear()
        self._obstructed_during_event = False
        self.pending_reason = reason
        return event

    # ------------------------------------------------------------------- main
    def observe(self, observation: DepositObservation) -> OccupancyEvent | None:
        """Advance one frame. Returns an event only on the frame it finalises."""
        timestamp = observation.timestamp if observation.timestamp is not None else time.time()
        reading = observation.reading
        if observation.geometry_moved:
            self.reset(reason=CAMERA_MOVED)
            return None
        if reading.usable:
            self._recent.append(float(reading.litres))
            del self._recent[:-32]
        arriving = (
            observation.changed_fraction >= self.enter_changed_fraction
            or observation.tracked_objects > 0
        )
        if observation.label:
            self._last_object = {
                "label": observation.label, "color": observation.color,
                "track_id": observation.track_id, "envelope_mm": observation.envelope_mm,
            }
        if observation.obstructed and self.state != STABLE_PRE:
            self._obstructed_during_event = True

        finalised: OccupancyEvent | None = None
        if self.state == STABLE_PRE:
            if reading.usable and not arriving:
                # Refreshed only while nothing is arriving. The frame that
                # starts a deposit already contains part of it -- the bag is in
                # shot and the surface has begun to rise -- so letting it update
                # the committed reading turns the before/after pair into an
                # after/after one, quietly and with no symptom but a small
                # delta. The bin as it last stood undisturbed is the before.
                self.committed_before = reading
                self.pending_reason = None
            elif not reading.usable and self.committed_before is None:
                self.pending_reason = reading.reason or NO_OCCUPANCY
            if arriving:
                self.frozen_before = self.committed_before
                self._started_at = timestamp
                self._obstructed_during_event = observation.obstructed
                self.state = DEPOSIT_IN_PROGRESS
        elif self.state == DEPOSIT_IN_PROGRESS:
            if not arriving:
                self.state = SETTLING
            self.pending_reason = SCENE_UNSETTLED
        elif self.state == SETTLING:
            if arriving:
                self.state = DEPOSIT_IN_PROGRESS
            elif reading.usable and self._settled(float(reading.litres)):
                self._post_samples = [float(reading.litres)]
                self.state = STABLE_POST
            else:
                self.pending_reason = (
                    SCENE_UNSETTLED if reading.usable else (reading.reason or NO_OCCUPANCY)
                )
        elif self.state == STABLE_POST:
            if arriving:
                # Another bag arrived before this one was finalised: the event
                # continues rather than splitting into two.
                self._post_samples.clear()
                self.state = DEPOSIT_IN_PROGRESS
            elif reading.usable:
                self._post_samples.append(float(reading.litres))
                if len(self._post_samples) >= self.required_stable_frames:
                    finalised = self._finalize(timestamp)
                    self.state = FINALIZED
        elif self.state == FINALIZED:
            self.state = STABLE_PRE
        return finalised

    def reset(self, *, reason: str | None = None) -> None:
        self.state = STABLE_PRE
        self.frozen_before = None
        self._post_samples.clear()
        self._recent.clear()
        self._obstructed_during_event = False
        self._started_at = None
        self.pending_reason = reason

    def describe(self) -> dict[str, Any]:
        latest = self.events[-1] if self.events else None
        return {
            "camera": self.camera,
            "state": self.state,
            "message": MESSAGES.get(self.state, ""),
            "pending_reason": self.pending_reason,
            "committed_before_l": (
                None if self.committed_before is None or self.committed_before.litres is None
                else round(self.committed_before.litres, 3)
            ),
            "absolute_occupancy": bool(
                self.committed_before is not None and self.committed_before.absolute
            ),
            "stable_frames": len(self._post_samples),
            "required_stable_frames": self.required_stable_frames,
            "latest_event": None if latest is None else latest.to_dict(),
            "events": [event.to_dict() for event in self.events[-10:]],
        }


def pair_events(
    first: list[OccupancyEvent], second: list[OccupancyEvent], *, window_s: float = 6.0,
) -> list[tuple[OccupancyEvent | None, OccupancyEvent | None]]:
    """Line the two cameras' events up in time, for comparison only.

    Nothing is copied between them: each event keeps the litres its own camera
    measured, and the pairing exists so the thesis can put the two numbers side
    by side and report the difference honestly.
    """
    remaining = list(second)
    pairs: list[tuple[OccupancyEvent | None, OccupancyEvent | None]] = []
    for event in first:
        match = None
        for candidate in remaining:
            if abs(candidate.finalized_at - event.finalized_at) <= window_s:
                match = candidate
                break
        if match is not None:
            remaining.remove(match)
        pairs.append((event, match))
    pairs.extend((None, leftover) for leftover in remaining)
    return pairs
