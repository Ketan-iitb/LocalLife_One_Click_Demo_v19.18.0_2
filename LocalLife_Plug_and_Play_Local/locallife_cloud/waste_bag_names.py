"""One name per waste bag, whatever the detector called it.

The real-bin screenshots show identical bags captioned "filled plastic waste
bag", "plastic garbage bag" and "plastic trash bag" in one frame -- three names
for one kind of object, which reads to anyone looking as three kinds. The
open-vocabulary detector is not wrong about any of them; it simply phrases the
same thing differently from frame to frame and bag to bag.

This sits in front of `vocabulary.object_type` rather than inside it, so the
existing vocabulary -- frozen since V32, and relied on for every other object --
is untouched. Anything that is not a waste bag passes straight through to it.

Only the material survives the mapping, because sorting depends on it: paper
and plastic bags go to different streams. Filled, black, garbage, trash,
refuse, bin, transparent -- all the same object described differently. Colour
is reported separately and is never lost by this, and the detector's raw label
stays on the detection and in every record.

Words are matched whole, never as substrings: "handbag" and "laptop bag" are
not waste bags, and a substring test would say they are.
"""

from __future__ import annotations

from .vocabulary import object_type as vocabulary_object_type

BAG_WORDS = frozenset({"bag", "bags", "sack", "sacks", "liner", "liners", "binbag", "binbags"})
NOT_WASTE_BAG_WORDS = frozenset({
    "handbag", "backpack", "rucksack", "laptop", "school", "tote", "duffel",
    "sleeping", "tea", "bean",
})
PAPER_WORDS = frozenset({"paper", "kraft", "carrier"})
PLASTIC_WASTE_BAG = "plastic waste bag"
PAPER_WASTE_BAG = "paper waste bag"
# What an unsure bag is reported as: still honestly a bag, and far more useful
# in a waste bin than the vocabulary's generic "packaging object".
UNSURE_BAG = "waste bag"
# The same threshold the vocabulary uses for every other specific name.
SPECIFIC_CONFIDENCE = 0.45


def _normalise(label: str | None) -> str:
    return " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())


def waste_bag_name(label: str | None) -> str | None:
    """The canonical waste-bag name, or None when this is not a waste bag."""
    words = set(_normalise(label).split())
    if not words & BAG_WORDS or words & NOT_WASTE_BAG_WORDS:
        return None
    return PAPER_WASTE_BAG if words & PAPER_WORDS else PLASTIC_WASTE_BAG


def object_type(label: str, confidence: float) -> str:
    """The name to report: one per waste bag, the vocabulary's for anything else."""
    bag = waste_bag_name(label)
    if bag is None:
        return vocabulary_object_type(label, confidence)
    return bag if confidence >= SPECIFIC_CONFIDENCE else UNSURE_BAG
