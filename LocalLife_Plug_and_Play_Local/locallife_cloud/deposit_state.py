"""The life of one deposit, as a state machine.

A measurement is only meaningful against a scene that is understood: what was
already there, what has just arrived, and whether it has stopped moving. The
pipeline used to answer those questions with a scatter of counters, so a
second carton placed beside the first could be measured as though both had
just arrived. Here the sequence is explicit, one camera at a time:

    WAITING_FOR_CHANGE -> OBJECT_ENTERING -> WAITING_FOR_STABILITY
    -> MEASURING -> FINALIZED -> (commit the scene) -> WAITING_FOR_CHANGE

The machine decides nothing on its own: the pipeline reports what it sees each
frame and reads the state back, so the dashboard can say which step the
station is on and why nothing has been recorded yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WAITING_FOR_CHANGE = "WAITING_FOR_CHANGE"
OBJECT_ENTERING = "OBJECT_ENTERING"
WAITING_FOR_STABILITY = "WAITING_FOR_STABILITY"
MEASURING = "MEASURING"
FINALIZED = "FINALIZED"

STATES = (WAITING_FOR_CHANGE, OBJECT_ENTERING, WAITING_FOR_STABILITY, MEASURING, FINALIZED)

MESSAGES = {
    WAITING_FOR_CHANGE: "Waiting for an object to enter the measurement zone",
    OBJECT_ENTERING: "An object is arriving; hold it still to measure it",
    WAITING_FOR_STABILITY: "Waiting for the object to settle",
    MEASURING: "Measuring the object",
    FINALIZED: "Measured; the scene is being committed for the next deposit",
}


@dataclass(frozen=True)
class FrameObservation:
    """What one frame saw inside the measurement zone."""

    changed_fraction: float = 0.0
    tracked_objects: int = 0
    measured_volume_l: float | None = None
    stable: bool = False
    finalised: bool = False


@dataclass
class DepositStateMachine:
    """Where this camera is in the deposit it is currently watching."""

    # A change smaller than this is the room breathing: exposure drift, a
    # shadow, sensor noise. Larger, and something is in the zone.
    enter_fraction: float = 0.01
    # Frames the change must stop growing before the object counts as placed.
    settle_frames: int = 3
    state: str = WAITING_FOR_CHANGE
    frames_in_state: int = 0
    last_changed_fraction: float = 0.0
    awaiting_commit: bool = False
    history: list[str] = field(default_factory=list)

    def observe(self, observation: FrameObservation) -> str:
        """Advance by one frame and return the state the camera is now in."""
        previous = self.state
        occupied = (observation.changed_fraction >= self.enter_fraction
                    or observation.tracked_objects > 0)
        if self.state == WAITING_FOR_CHANGE:
            if occupied:
                self.state = OBJECT_ENTERING
        elif self.state == OBJECT_ENTERING:
            if not occupied:
                self.state = WAITING_FOR_CHANGE
            elif abs(observation.changed_fraction - self.last_changed_fraction) <= 0.2 * max(
                self.last_changed_fraction, self.enter_fraction
            ):
                if self.frames_in_state + 1 >= self.settle_frames:
                    self.state = WAITING_FOR_STABILITY
            else:
                self.frames_in_state = -1  # the object is still arriving
        elif self.state == WAITING_FOR_STABILITY:
            if not occupied:
                self.state = WAITING_FOR_CHANGE
            elif observation.measured_volume_l is not None:
                self.state = MEASURING
        elif self.state == MEASURING:
            if observation.finalised or observation.stable:
                self.state = FINALIZED
                self.awaiting_commit = True
            elif not occupied:
                self.state = WAITING_FOR_CHANGE
        elif self.state == FINALIZED:
            if not self.awaiting_commit:
                # The scene was committed, so whatever is in the zone now is
                # part of the background and the next deposit starts clean.
                self.state = WAITING_FOR_CHANGE
        self.last_changed_fraction = observation.changed_fraction
        self.frames_in_state = 0 if self.state != previous else self.frames_in_state + 1
        if self.state != previous:
            self.history.append(self.state)
            del self.history[:-32]
        return self.state

    def committed(self) -> None:
        """The scene now includes this deposit; the next one is measured against it."""
        self.awaiting_commit = False
        if self.state == FINALIZED:
            self.state = WAITING_FOR_CHANGE
            self.frames_in_state = 0
            self.history.append(self.state)
            del self.history[:-32]

    def reset(self) -> None:
        self.state = WAITING_FOR_CHANGE
        self.frames_in_state = 0
        self.last_changed_fraction = 0.0
        self.awaiting_commit = False
        self.history.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "frames_in_state": self.frames_in_state,
            "message": MESSAGES.get(self.state, ""),
            "changed_fraction": round(self.last_changed_fraction, 4),
            "awaiting_commit": self.awaiting_commit,
        }
