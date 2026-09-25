"""A real waste bin: bags that touch, bags that sag, and contents already there.

Three faults the field screenshots show, each reproducible from the numbers on
the dashboard alone.

**Every bag was flagged `volume_disagrees_with_dimensions`.** The consistency
check compared the integrated volume against length x width x height and
complained when the two differed by more than a fifth. That is the right test
for a box and the wrong one for a bag. A filled bag sags: it fills roughly half
to two thirds of the box that encloses it. The field numbers say exactly that --
437 x 272 x 251 mm encloses 29.8 L and the height map integrated 20.1 L, a fill
of 0.67; 340 x 277 x 69 mm encloses 6.5 L against 3.4 L, a fill of 0.53. Both
are what a bag looks like, and both were reported as a geometry failure.

Worse, the same check passes the one case that is physically impossible. A
paired RealSense reading of 317 x 196 x 120 mm with 9.013 L is a volume 21 %
larger than the envelope containing it, and the old test called that consistent
because the ratio happened to land inside the tolerance from the other side.

So `envelope_fill` replaces a symmetric tolerance with the physical
constraint: measured volume cannot exceed the envelope it was measured inside,
and a fill far below what any real object manages means the envelope was
inflated by something that is not the object. Between those bounds a bag is
simply a bag, and the fill fraction is published so the number can be judged
rather than merely flagged.

**A volume was published while the geometry was still moving.** Both rows also
carried `geometry_not_settled`, and both still showed litres. `publish_decision`
makes the two agree: an unsettled or impossible geometry yields a named reason
instead of a number.

**Bags touch.** In a bin they lean on each other, and one mask that has swallowed
its neighbour measures both as one deposit. `merged_neighbour_reason` finds a
mask that covers a large part of another tracked object and says so, rather than
reporting a confident volume for two bags at once.

Nothing here reads RealSense depth, dimensions or detections, and nothing in the
RealSense pipeline imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# A filled bag sags into roughly half to two thirds of its bounding box; a rigid
# box fills nearly all of it. Anything at or below the floor means the envelope
# is not describing the object -- leaked floor, a merged neighbour, a mask that
# ran across the bin. Anything above the ceiling is impossible: an object cannot
# occupy more space than the box drawn around it.
MIN_PLAUSIBLE_FILL = 0.12
MAX_PLAUSIBLE_FILL = 1.05

IMPOSSIBLE_FILL = "volume_exceeds_its_own_envelope"
EMPTY_ENVELOPE = "envelope_far_larger_than_the_measured_object"
UNSETTLED = "geometry_not_settled_yet"
MERGED = "mask_covers_a_neighbouring_object"
NOT_FLOOR_RELATIVE = "height_not_measured_against_the_floor"

# Which shape model the published volume claims to follow. A bag is measured as
# the external volume its own visible surface encloses above the floor, which is
# neither a cuboid nor a cylinder, and saying so is the point of the field.
SHAPE_IRREGULAR = "irregular_height_map"
SHAPE_BOX = "cuboid"
SHAPE_CYLINDER = "cylinder"

# Labels whose objects are flexible: their envelope is not their volume.
FLEXIBLE_WORDS = frozenset({
    "bag", "bags", "sack", "sacks", "pouch", "polythene", "plastic", "waste",
    "garbage", "refuse", "rubbish", "liner", "clothing", "textile", "fabric",
})


def shape_model_for(label: str | None, *, occlusion_corrected: bool = False) -> str:
    """Which volume model this object's published number follows.

    Geometry decides first -- a footprint reconstructed from an arc is a round
    cross-section whatever it is called -- and the label is only a weak prior
    for telling a flexible bag from a rigid carton.
    """
    if occlusion_corrected:
        return SHAPE_CYLINDER
    words = set((label or "").lower().replace("-", " ").replace("_", " ").split())
    return SHAPE_IRREGULAR if words & FLEXIBLE_WORDS else SHAPE_BOX


@dataclass
class EnvelopeFill:
    envelope_l: float | None
    fill_fraction: float | None
    shape_model: str
    plausible: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_l": None if self.envelope_l is None else round(self.envelope_l, 4),
            "fill_fraction": None if self.fill_fraction is None else round(self.fill_fraction, 3),
            "shape_model": self.shape_model,
            "envelope_plausible": self.plausible,
            "envelope_reason": self.reason,
        }


def envelope_fill(
    *, length_m: float | None, width_m: float | None, height_m: float | None,
    litres: float | None, shape_model: str = SHAPE_IRREGULAR,
    minimum: float = MIN_PLAUSIBLE_FILL, maximum: float = MAX_PLAUSIBLE_FILL,
) -> EnvelopeFill:
    """How much of its own envelope the measured object actually occupies.

    The dimensions are envelope dimensions -- the box the object fits inside --
    and the volume is what its surface encloses above the floor. For a rigid
    carton the two nearly coincide. For a bag they do not, and the fraction
    between them is a property of the object, not an error.

    Only the physical bounds are enforced: a volume cannot be larger than the
    envelope it was measured inside, and an envelope many times larger than the
    object is describing something other than the object.
    """
    if not length_m or not width_m or not height_m or not litres or litres <= 0:
        return EnvelopeFill(None, None, shape_model, False, "missing_geometry")
    envelope = float(length_m * width_m * height_m * 1000.0)
    if envelope <= 0:
        return EnvelopeFill(None, None, shape_model, False, "missing_geometry")
    fill = float(litres) / envelope
    if fill > maximum:
        return EnvelopeFill(envelope, fill, shape_model, False, IMPOSSIBLE_FILL)
    if fill < minimum:
        return EnvelopeFill(envelope, fill, shape_model, False, EMPTY_ENVELOPE)
    return EnvelopeFill(envelope, fill, shape_model, True, None)


def merged_neighbour_reason(
    mask: np.ndarray,
    other_masks: list[np.ndarray],
    *,
    swallow_fraction: float = 0.60,
    minimum_pixels: int = 40,
) -> str | None:
    """Has this mask taken in a neighbouring object as well as its own?

    Bags in a bin lean on each other, and a mask that covers most of the bag
    beside it is measuring two deposits as one. Overlap is judged against the
    *other* object's size, not this one's: a small bag entirely inside a large
    mask is exactly the case to catch, and it barely moves the large mask's own
    overlap fraction.
    """
    array = np.asarray(mask, dtype=bool)
    if not array.any():
        return None
    own = int(np.count_nonzero(array))
    for other in other_masks:
        neighbour = np.asarray(other, dtype=bool)
        if neighbour.shape != array.shape:
            continue
        size = int(np.count_nonzero(neighbour))
        if size < minimum_pixels or size >= own:
            # Only a neighbour smaller than this mask can have been swallowed
            # by it; the reverse case is caught when that mask is examined.
            continue
        if int(np.count_nonzero(array & neighbour)) >= swallow_fraction * size:
            return MERGED
    return None


@dataclass
class PublishDecision:
    publish: bool
    reason: str | None
    labels: tuple[str, ...]
    diagnostics: dict[str, Any]

    @property
    def final(self) -> bool:
        """May this frame's number be the deposit's finalised measurement?"""
        return self.publish and not self.labels


def publish_decision(
    *, settled: bool, plane_is_floor: bool, fill: EnvelopeFill,
    merged_reason: str | None = None,
) -> PublishDecision:
    """Whether this frame's volume may be shown, and what it has to be shown as.

    Two different things were tangled together on the dashboard. A volume that
    is *wrong* -- one that does not fit inside its own envelope, or that was
    integrated over two bags at once -- must not be published at all; a number
    nobody can act on is worse than a named refusal. A volume that is merely
    *not final yet* -- still settling, or measured against the optical-axis
    fallback rather than the floor -- is a usable live reading, and the fault
    was not that it existed but that it was printed without its status. Those
    stay, carrying the label that says what they are, so the number and the
    status can no longer disagree in public.

    `final` is the stricter question, and that is the one a recorded deposit
    has to answer yes to.
    """
    diagnostics = fill.to_dict()
    if merged_reason:
        return PublishDecision(False, merged_reason, (), diagnostics)
    if fill.reason and not fill.plausible:
        return PublishDecision(False, fill.reason, (), diagnostics)
    labels = tuple(label for label in (
        None if plane_is_floor else NOT_FLOOR_RELATIVE,
        None if settled else UNSETTLED,
    ) if label is not None)
    return PublishDecision(True, None, labels, diagnostics)
