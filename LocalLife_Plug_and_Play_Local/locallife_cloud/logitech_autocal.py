"""Starting the Logitech camera without asking the operator for anything.

Two things the Logitech measurement needed, and used to demand by hand:

*The camera's height above the floor.* It was typed in from a tape measure and
stored as `logitech_reference_distance_m`, and until it existed the readiness
panel said the camera-to-floor distance was missing. But a floor plane has
already been fitted from the camera's own depth, and the perpendicular distance
from the camera to that plane *is* the height. `camera_height_from_plane`
returns it, so the tape measure becomes an optional cross-check rather than a
prerequisite.

*An empty-scene baseline.* It was captured by a button press, and the operator
was expected to press it at the right moment. `BaselineLearner` watches instead:
when the view has held still for a run of frames with nothing detected inside
the measurement zone, that is an empty scene and it can be captured. When it
cannot, the learner says which of the two conditions failed, so the instruction
on screen is "the zone is not empty" or "the view is still moving" rather than
a bare "pending".

The one thing this must never do is capture a baseline containing an object.
That object becomes part of the floor, nothing ever differs from the reference
again, and every later measurement is refused for lack of a foreground change.
That is exactly what happened when typing a reference distance quietly
recaptured the baseline from whatever was in front of the camera at the time,
and it is why the learner refuses far more readily than it accepts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

# Frames of a still, empty view before a baseline is taken by itself. At the
# usual frame rate this is a second or two -- long enough that a hand pulling
# back out of shot cannot be mistaken for an empty bin.
REQUIRED_STABLE_FRAMES = 12
# How much the depth of the scene may move, frame to frame, and still count as
# still. A multiple of the sensor's own noise, not a fixed number of metres.
STABILITY_NOISE_MULTIPLE = 3.0
MINIMUM_STABILITY_M = 0.01

ZONE_NOT_EMPTY = "measurement_zone_is_not_empty"
VIEW_NOT_STILL = "camera_view_is_still_moving"
NO_DEPTH = "no_depth_for_this_camera_yet"
WAITING = "learning_the_empty_scene"
LEARNED = "empty_scene_learned_automatically"


def camera_height_from_plane(coefficients: tuple[float, float, float] | None) -> float | None:
    """Perpendicular distance from the camera to the fitted floor, in metres.

    The plane is z = a*x + b*y + c in camera coordinates, so the distance from
    the origin to it is |c| / sqrt(a^2 + b^2 + 1). A plane fitted to a wall,
    or one that came out behind the camera, gives a distance outside anything a
    fixed installation can be, and None is returned rather than a number that
    would then be published as a calibration.
    """
    if coefficients is None:
        return None
    a, b, c = (float(value) for value in coefficients)
    if not all(np.isfinite(value) for value in (a, b, c)):
        return None
    distance = abs(c) / float(np.sqrt(a * a + b * b + 1.0))
    if not np.isfinite(distance) or not 0.15 <= distance <= 6.0:
        return None
    return distance


@dataclass
class BaselineDecision:
    capture: bool
    reason: str
    stable_frames: int = 0
    spread_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture, "reason": self.reason,
            "stable_frames": self.stable_frames,
            "spread_m": None if self.spread_m is None else round(self.spread_m, 5),
        }


class BaselineLearner:
    """Decides when the view is an empty scene worth remembering.

    It counts consecutive frames that are both empty and still, and only says
    yes once the run is long enough. Any frame that fails either condition
    resets the count, so a baseline is never assembled out of a sequence that
    merely averaged to empty.
    """

    def __init__(
        self, required_frames: int = REQUIRED_STABLE_FRAMES, depth_noise_m: float = 0.004,
    ) -> None:
        self.required_frames = max(1, int(required_frames))
        self.depth_noise_m = float(depth_noise_m)
        self.stable_frames = 0
        self.last_reason = WAITING
        self._recent: deque[np.ndarray] = deque(maxlen=5)

    def reset(self) -> None:
        self.stable_frames = 0
        self.last_reason = WAITING
        self._recent.clear()

    def observe(
        self, *, depth: np.ndarray | None, region: np.ndarray | None, objects_in_zone: int,
    ) -> BaselineDecision:
        if depth is None or not np.isfinite(depth).any():
            self.stable_frames = 0
            self._recent.clear()
            self.last_reason = NO_DEPTH
            return BaselineDecision(False, NO_DEPTH)
        if objects_in_zone > 0:
            # Something is standing in the zone. Capturing now would make it
            # part of the floor for ever.
            self.stable_frames = 0
            self._recent.clear()
            self.last_reason = ZONE_NOT_EMPTY
            return BaselineDecision(False, ZONE_NOT_EMPTY)

        frame = np.asarray(depth, dtype=np.float32)
        if self._recent and self._recent[-1].shape != frame.shape:
            self._recent.clear()
        self._recent.append(frame)
        spread: float | None = None
        if len(self._recent) >= 3:
            stack = np.stack(self._recent)
            inside = (
                np.ones(frame.shape, dtype=bool) if region is None or region.shape != frame.shape
                else np.asarray(region, dtype=bool)
            )
            valid = inside & np.isfinite(stack).all(axis=0)
            spread = (
                None if not valid.any()
                else float(np.median(np.std(stack, axis=0)[valid]))
            )
        limit = max(MINIMUM_STABILITY_M, STABILITY_NOISE_MULTIPLE * self.depth_noise_m)
        if spread is not None and spread > limit:
            self.stable_frames = 0
            self.last_reason = VIEW_NOT_STILL
            return BaselineDecision(False, VIEW_NOT_STILL, spread_m=spread)

        self.stable_frames += 1
        if self.stable_frames < self.required_frames:
            self.last_reason = WAITING
            return BaselineDecision(False, WAITING, self.stable_frames, spread)
        self.last_reason = LEARNED
        return BaselineDecision(True, LEARNED, self.stable_frames, spread)
