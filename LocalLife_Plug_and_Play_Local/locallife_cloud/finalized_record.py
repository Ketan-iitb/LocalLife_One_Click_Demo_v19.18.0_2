"""One finalised measurement, and only one, for every surface that shows it.

The field screenshots show one object reported three ways at once: a live
overlay drawn with one triplet, a detection table showing another, and a
history row holding a third. All three are honest readings of the same object
-- they are simply readings of *different frames*. The overlay is painted from
whatever the detection holds at the instant the frame is drawn; the history row
was written from whatever it held at the instant the event was finalised; and
the detection keeps being re-measured after that, so the two drift apart and
keep drifting.

That is not a rounding difference. A reader comparing the picture with the row
sees two different objects, and the thesis cannot cite a number that changes
depending on where it was read.

So finalisation produces a record, and the record is the answer. Once a track
is finalised, `FinalizedRegistry` holds its event id, timestamp, dimensions,
volume and status, and every later frame republishes those values onto the
detection instead of the frame's own fresh measurement. The overlay, the table,
the history row and the exported file then read one immutable object.

Re-measurement does not stop -- the pipeline keeps computing, and the live
values stay available under their own names for diagnostics -- but what is
*published* for a finalised deposit no longer moves.

A track id can be reused by the tracker for a different physical object. The
registry is therefore keyed on the permanent event identity where one exists,
and `release` drops a track's record when the identity changes, so a new bag
can never inherit the previous holder's finalised size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The published fields a finalised record owns. Everything else on a detection
# (colour votes, tracking status, per-frame diagnostics) keeps moving.
FROZEN_FIELDS = (
    "footprint_length_mm",
    "footprint_width_mm",
    "physical_height_mm",
    "height_above_baseline_cm",
    "monocular_volume_l",
    "realsense_volume_l",
    "measurement_quality",
    "volume_rejection_reason",
    "dimension_method",
)


@dataclass(frozen=True)
class FinalizedMeasurement:
    """What was recorded when this deposit was finalised, verbatim."""

    event_id: str
    camera_id: str
    track_id: int | None
    timestamp: float | None
    status: str
    reason: str | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "reason": self.reason,
            **{name: self.values.get(name) for name in FROZEN_FIELDS},
        }


class FinalizedRegistry:
    """Every finalised measurement this camera has published, by identity.

    Keyed on the permanent event identity when the pipeline has assigned one,
    and on the detector's track id only for objects that never reached it. The
    distinction matters: detector track ids are reused, and keying solely on
    them would hand a new bag the finalised size of whatever held that number
    before.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._by_identity: dict[str, FinalizedMeasurement] = {}
        self._identity_of_track: dict[int, str] = {}

    @staticmethod
    def _identity(permanent_id: Any, track_id: int | None) -> str | None:
        if permanent_id is not None:
            return f"permanent:{permanent_id}"
        if track_id is not None:
            return f"track:{int(track_id)}"
        return None

    def remember(
        self,
        detection: Any,
        *,
        event_id: str,
        timestamp: float | None,
        status: str,
        reason: str | None = None,
        permanent_id: Any = None,
    ) -> FinalizedMeasurement | None:
        """Snapshot the values this event was recorded with."""
        identity = self._identity(permanent_id, getattr(detection, "track_id", None))
        if identity is None:
            return None
        existing = self._by_identity.get(identity)
        if existing is not None:
            # Finalisation is once per deposit. A repeated call is the same
            # event arriving again -- a retry, a refresh, a duplicate frame --
            # and it must not rewrite what was published.
            return existing
        record = FinalizedMeasurement(
            event_id=event_id,
            camera_id=self.camera_id,
            track_id=getattr(detection, "track_id", None),
            timestamp=timestamp,
            status=status,
            reason=reason,
            values={
                name: getattr(detection, name, None) for name in FROZEN_FIELDS
            },
        )
        self._by_identity[identity] = record
        track_id = getattr(detection, "track_id", None)
        if track_id is not None:
            self._identity_of_track[int(track_id)] = identity
        return record

    def lookup(self, permanent_id: Any, track_id: int | None) -> FinalizedMeasurement | None:
        identity = self._identity(permanent_id, track_id)
        return None if identity is None else self._by_identity.get(identity)

    def republish(self, detection: Any, permanent_id: Any = None) -> FinalizedMeasurement | None:
        """Put the finalised values back onto a detection that has moved on.

        This is the whole point of the module: after a deposit is finalised the
        pipeline keeps measuring, and what it measures keeps changing. The
        overlay must not show that drift, because the row and the export cannot.
        """
        record = self.lookup(permanent_id, getattr(detection, "track_id", None))
        if record is None:
            return None
        for name, value in record.values.items():
            if value is not None:
                setattr(detection, name, value)
        setattr(detection, "finalized_event_id", record.event_id)
        return record

    def release(self, track_id: int | None) -> None:
        """Forget a track whose number has been handed to another object."""
        if track_id is None:
            return
        identity = self._identity_of_track.pop(int(track_id), None)
        if identity is not None and identity.startswith("track:"):
            self._by_identity.pop(identity, None)

    def clear(self) -> None:
        self._by_identity.clear()
        self._identity_of_track.clear()

    def describe(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._by_identity.values()]
