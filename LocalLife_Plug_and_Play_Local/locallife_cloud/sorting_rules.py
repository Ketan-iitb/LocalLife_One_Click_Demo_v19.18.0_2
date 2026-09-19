"""Deterministic allowed / mis-sort / unknown rules (Final Implementation Playbook section 12).

The playbook is explicit that this stays a rule table over the labels the
existing detector already produces: no new waste model is to be trained during
the final sprint. The table's one non-negotiable behaviour is the last row --
an unrecognised or low-confidence detection is reported as UNKNOWN for manual
check, never guessed into either of the other two answers. Calling real waste a
mis-sort annoys an operator; calling a mis-sort correct defeats the point of
the system, and quietly guessing does both at random.
"""

from __future__ import annotations

from dataclasses import dataclass

CORRECT = "correct"
MIS_SORT = "mis_sort"
UNKNOWN = "unknown"

_DASHBOARD_TEXT = {
    CORRECT: "CORRECT",
    MIS_SORT: "MIS-SORT",
    UNKNOWN: "UNKNOWN / MANUAL CHECK",
}

# Object families the bin does not accept, keyed by the group name reported in
# the event log so a confusion matrix can be built per family (section 21).
MIS_SORT_FAMILIES: dict[str, frozenset[str]] = {
    "footwear": frozenset({
        "shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "boot", "boots",
        "sandal", "sandals", "slipper", "slippers", "flipflop", "flipflops", "footwear",
    }),
    "tool": frozenset({
        "drill", "hammer", "screwdriver", "wrench", "spanner", "saw", "pliers",
        "tool", "tools", "toolbox", "grinder", "sander",
    }),
    "appliance": frozenset({
        "vacuum", "hoover", "appliance", "microwave", "toaster", "kettle", "blender",
        "iron", "television", "tv", "monitor", "laptop", "computer", "printer",
        "charger", "adapter", "cable", "electrical", "electronic", "electronics",
    }),
    "furniture": frozenset({
        "chair", "stool", "sofa", "couch", "armchair", "table", "desk", "bench",
        "cabinet", "shelf", "wardrobe", "mattress", "furniture",
    }),
    "textile": frozenset({
        "pillow", "cushion", "blanket", "bedding", "duvet", "curtain", "drape",
        "clothing", "garment", "towel", "rug", "mat",
    }),
    "person": frozenset({"person", "hand", "foot", "arm", "leg", "face"}),
}

# Detector classes the bin does accept. `cardboard_box` is here because rigid
# boxes are the playbook's calibration and verification objects (section 1) and
# must pass through the same event path as a bag rather than tripping a
# mis-sort on every accuracy run.
ALLOWED_CLASSES: frozenset[str] = frozenset({"plastic_bag", "paper_bag", "cardboard_box"})

# Words that identify allowed waste directly from the detector's own label,
# for callers that have no mapped class to hand.
ALLOWED_WORDS: frozenset[str] = frozenset({
    "garbage", "trash", "refuse", "rubbish", "waste", "polythene", "polyethylene",
    "bin", "kraft",
})


@dataclass(frozen=True, slots=True)
class SortingVerdict:
    status: str
    reason: str
    family: str | None = None

    @property
    def dashboard_text(self) -> str:
        return _DASHBOARD_TEXT[self.status]

    @property
    def is_mis_sort(self) -> bool:
        return self.status == MIS_SORT

    def to_dict(self) -> dict[str, object]:
        return {
            "sorting_status": self.status,
            "sorting_text": self.dashboard_text,
            "sorting_reason": self.reason,
            "sorting_family": self.family,
        }


def normalize_label(label: str | None) -> str:
    if not label:
        return ""
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def label_words(label: str | None) -> set[str]:
    normalized = normalize_label(label).replace("(", " ").replace(")", " ")
    return set(normalized.split())


def mis_sort_family(label: str | None) -> str | None:
    """The disallowed family a label belongs to, if any."""
    words = label_words(label)
    for family, vocabulary in MIS_SORT_FAMILIES.items():
        if words & vocabulary:
            return family
    return None


def classify_sorting(
    label: str | None,
    *,
    confidence: float = 1.0,
    accepted_class: str | None = None,
    min_confidence: float = 0.35,
) -> SortingVerdict:
    """Allowed / mis-sort / unknown for one detection.

    `accepted_class` is the pipeline's own mapped family when it has one
    (`plastic_bag`, `paper_bag`, `cardboard_box`); `label` is the detector's raw
    text. A disallowed family is checked first and wins over both, because a
    label like "vacuum cleaner bag" contains "bag" and must not be waved
    through on that alone.
    """
    words = label_words(label)
    if not words:
        return SortingVerdict(UNKNOWN, "no object class was reported")
    if confidence < min_confidence:
        return SortingVerdict(
            UNKNOWN,
            f"detector confidence {confidence:.2f} is below the {min_confidence:.2f} "
            "threshold for a sorting decision",
        )

    family = mis_sort_family(label)
    if family is not None:
        return SortingVerdict(
            MIS_SORT, f"{family} is not accepted by this bin", family=family,
        )
    if accepted_class in ALLOWED_CLASSES:
        return SortingVerdict(CORRECT, f"{accepted_class} is accepted waste")
    if words & ALLOWED_WORDS and words & {"bag", "bags", "sack", "sacks"}:
        return SortingVerdict(CORRECT, "waste bag is accepted waste")
    return SortingVerdict(
        UNKNOWN, f"'{normalize_label(label)}' matches no allowed or disallowed rule",
    )
