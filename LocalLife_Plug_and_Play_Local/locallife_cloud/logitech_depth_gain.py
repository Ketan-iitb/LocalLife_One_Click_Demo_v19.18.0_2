"""One number, measured once in the fixed pose, that seats Logitech objects at their true depth.

Depth Anything places objects a fraction of their distance nearer (or further)
than the floor they stand on. A point moved k % along its ray rises or sinks
by k % of its height below the camera, whatever the viewing angle -- so from a
camera on a table, 30-40 cm up, a 15 % error is 3 cm of height, and from a
bird's-eye tripod 1.3 m up it is 16 cm: a 200 mm bag read as 365 mm in the
ray-traced rig. That is why the Logitech was right from the table and two to
three times too tall from above.

Per-object cues cannot undo it: the one that looked promising (an object
touches its support along its lower silhouette edge) is wrong for anything
that bulges, and filled bags bulge -- a bag widest 5 cm above the floor lost a
third of its height. What is stable is the error itself, for one camera in one
pose: a depth gain. It is measured with the wizard's known-height objects:
for each, the gain that makes its measured height equal the ruler is found by
bisection, and the median is applied to every object's pixels afterwards.
Length, width and height move together, because it is the depth that is
corrected, not any one output. The gain belongs to the pose it was measured
in: a moved camera suspends it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

GAIN_RANGE = (0.5, 2.0)
MAX_SAMPLE_SPREAD = 0.10       # relative MAD across samples before the gain is trusted


def fit_object_gain(height_at: Callable[[float], float | None], true_height_m: float,
                    *, iterations: int = 40) -> float | None:
    """The object-depth gain at which `height_at(gain)` equals the true height.

    Height falls monotonically as the object is pushed away from the camera,
    so a bisection over the plausible range finds it, or proves it is not in
    that range.
    """
    low, high = GAIN_RANGE
    h_low, h_high = height_at(low), height_at(high)
    if h_low is None or h_high is None or not (h_high <= true_height_m <= h_low):
        return None
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value = height_at(middle)
        if value is None:
            return None
        if value > true_height_m:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass
class GainSample:
    name: str
    gain: float
    true_height_m: float
    measured_height_m: float
    pose: dict | None
    recorded_at: float = field(default_factory=time.time)


class DepthGainStore:
    """The gain samples on disk and the gain they give for the current pose."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.samples: list[GainSample] = []
        if path is not None and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.samples = [GainSample(**item) for item in payload.get("samples", [])]
            except (OSError, TypeError, ValueError):
                self.samples = []

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"samples": [asdict(item) for item in self.samples]}, indent=2),
                                 encoding="utf-8")
        except OSError:  # pragma: no cover - disk issues
            pass

    def add(self, sample: GainSample) -> None:
        self.samples.append(sample)
        self._save()

    def reset(self) -> None:
        self.samples = []
        self._save()

    def active(self, pose_matches: Callable[[dict | None], bool]) -> tuple[float | None, dict[str, Any]]:
        """(gain or None, status) from the samples recorded in the current pose."""
        usable = [item for item in self.samples if pose_matches(item.pose)]
        status: dict[str, Any] = {"samples": len(self.samples), "samples_in_this_pose": len(usable)}
        if not usable:
            return None, {**status, "state": "no_sample_in_this_pose" if self.samples else "not_measured"}
        gains = np.asarray([item.gain for item in usable])
        gain = float(np.median(gains))
        spread = float(np.median(np.abs(gains - gain))) * 1.4826 / gain if len(gains) > 1 else 0.0
        status.update({"gain": round(gain, 4), "relative_spread": round(spread, 4),
                       "per_sample": [round(float(value), 4) for value in gains]})
        if spread > MAX_SAMPLE_SPREAD:
            return None, {**status, "state": "samples_disagree_add_or_remeasure"}
        return gain, {**status, "state": "active" if len(usable) > 1 else "active_single_sample"}
