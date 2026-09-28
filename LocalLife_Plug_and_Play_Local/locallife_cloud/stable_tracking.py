"""Keep one ID per physical object while the detector changes its mind about the name.

The open-vocabulary detector relabels the same object from frame to frame: one
bag is "handbag" in one frame and "plastic garbage bag" in the next, a blanket
is "folded clothing" and then "textile item". `ObjectTracker` treats the
object family -- bag, box, other -- as a hard gate, so every such flip refused
the existing track and opened a new one. Worse, the old track stays alive for
its grace period, so the flips alternate: the same box at the same place came
back as ID 1, 2, 1, 2 as its label wavered. That is the "IDs keep changing"
the dashboard shows.

Here the family becomes a soft cue instead of a hard gate. A detection of a
different family can still continue a track, but only where the geometry says
it is unmistakably the same object -- the boxes overlap by at least half -- and
it scores below any same-family candidate, so where a genuinely different
object of the right family is also nearby, that one still wins. A new object
arriving elsewhere, or after the old one has left, still gets a new ID.

`tracking.py` is unchanged; this subclasses it and replaces only the
association rule.
"""

from __future__ import annotations

from .geometry import intersection_over_union
from .tracking import ObjectTracker, Track, _family
from .types import Detection

# A different-family detection continues a track only when the two boxes are
# this much the same box.
CROSS_FAMILY_MIN_IOU = 0.50
# ... and it always ranks below a same-family match for the same track.
CROSS_FAMILY_PENALTY = 1.0


class LabelTolerantTracker(ObjectTracker):
    """`ObjectTracker` whose association survives a relabelled detection."""

    def _association(self, track: Track, detection: Detection) -> tuple[bool, float]:
        if _family(track.label) == _family(detection.label):
            return super()._association(track, detection)
        overlap = intersection_over_union(track.box, detection.box)
        if overlap < CROSS_FAMILY_MIN_IOU:
            return False, -1.0
        # Score it as though the families matched, then penalise it, so the
        # geometry decides whether this is the same object and the name only
        # breaks ties.
        original = track.label
        try:
            track.label = detection.label
            allowed, score = super()._association(track, detection)
        finally:
            track.label = original
        return allowed, score - CROSS_FAMILY_PENALTY
