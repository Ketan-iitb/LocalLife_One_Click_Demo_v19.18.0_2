"""V51 descriptive recognition: object name, visible material and colour as SEPARATE attributes.

The pipeline's internal fields keep their contracts: `label` (raw detector prompt),
`accepted_class`, `resolved_label` and the legacy `material` feed tracking, geometry dispatch,
counting and the sorting policy and are not changed here. This module only derives the
descriptive outputs shown to people and written to events/history/CSV:

* object_name     -- what the object is (backpack, plastic waste bag, shoe ...), from the track's
                     resolved class; a broader name or "unknown object" when the evidence is weak.
* visible_material -- the exterior material seen in the object's pixels, in one vocabulary:
                     plastic, paper/cardboard, metal, textile, glass, mixed, unknown. From the visual
                     classifier's votes over the track; the object class is only a fallback prior for
                     physically certain cases, and is labelled as such. Contents of an opaque bag are
                     never inferred.
* colour / colour_secondary -- from colour_evidence.py (foreground pixels only).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

UNKNOWN_OBJECT = "unknown object"
MATERIALS = ("plastic", "paper/cardboard", "metal", "textile", "glass", "mixed", "unknown")

# Detector prompt / resolved class -> reported object name. Only true synonyms collapse.
_NAME_OF: dict[str, str] = {}
for _name, _labels in {
    "plastic waste bag": ("plastic bag", "plastic garbage bag", "plastic trash bag", "filled plastic waste bag",
                          "polythene waste bag", "transparent plastic waste bag", "plastic waste bag",
                          "waste bag", "garbage bag", "trash bag", "bin bag", "bin liner"),
    "paper bag": ("paper bag", "paper waste bag", "kraft paper bag", "paper shopping bag"),
    "backpack": ("backpack", "rucksack", "school bag", "bagpack"),
    "handbag": ("handbag", "purse", "tote bag"),
    "laptop bag": ("laptop bag", "laptop sleeve", "briefcase"),
    "sports bag": ("duffel bag", "sports bag", "gym bag"),
    "shoe": ("shoe", "sneaker", "boot", "trainer", "running shoe"),
    "slipper": ("slipper", "sandal", "flip flop", "flip-flop"),
    "pillow": ("pillow", "cushion"),
    "lamp": ("lamp", "table lamp", "desk lamp", "floor lamp"),
    "cardboard box": ("cardboard box", "cardboard shipping box", "carton box", "shipping box", "parcel",
                      "package", "shoe box"),
    "book": ("book", "notebook"),
    "bottle": ("bottle", "plastic bottle", "glass bottle", "water bottle", "cosmetic bottle", "cream bottle",
               "lotion bottle"),
    "drink carton": ("drink carton", "milk carton", "beverage carton", "juice carton"),
    "can": ("can", "soda can", "tin can", "drink can", "metal can", "aluminium can", "aluminum can",
            "aluminium drink can"),
    "clothing": ("clothing", "folded clothing", "shirt", "jacket", "sock"),
    "blanket": ("blanket", "bedding", "towel"),
    "laundry basket": ("laundry basket", "laundry hamper", "fabric storage basket"),
}.items():
    for _label in _labels:
        _NAME_OF[_label] = _name

# Catch-all prompts name a category, not an object.
_BROAD: dict[str, str] = {
    "textile item": "textile item", "flexible household object": "soft object",
    "rigid household object": "rigid object", "packaging object": "packaging",
    "electronic item": "electronic item", "electrical item": "electrical item",
    "decorative object": "decorative object", "unknown deposited object": UNKNOWN_OBJECT,
}

# Visual classifier label (material.py) -> visible-material vocabulary.
_VISIBLE_OF: dict[str, str] = {
    "plastic": "plastic", "polythene bag": "plastic", "paper": "paper/cardboard", "cardboard": "paper/cardboard",
    "fabric or textile": "textile", "textile": "textile", "metal": "metal", "glass": "glass",
    "mixed or general waste": "mixed",
    # Organic matter is not an exterior packaging material; never guessed as one.
    "food or organic waste": "unknown",
}

# Fallback ONLY when no visual evidence exists: classes whose exterior material is physically
# certain. A bag (plastic or textile), bottle (plastic or glass) or lamp (mixed) is not listed.
_CERTAIN_MATERIAL: dict[str, str] = {
    "cardboard box": "paper/cardboard", "book": "paper/cardboard", "paper bag": "paper/cardboard",
    "drink carton": "paper/cardboard", "can": "metal",
}

MIN_NAME_SHARE = 0.5          # resolved-class share needed to name the object specifically
MIN_MATERIAL_VOTES = 2        # visual material needs repeated agreement ...
MIN_MATERIAL_AGREEMENT = 0.6  # ... from most of its samples


def canonical_name(label: str | None) -> str | None:
    text = " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())
    if not text:
        return None
    if text in _NAME_OF:
        return _NAME_OF[text]
    if text in _BROAD:
        return _BROAD[text]
    padded = f" {text} "
    for key in sorted(_NAME_OF, key=len, reverse=True):     # whole words: "candle" is not a "can"
        if f" {key} " in padded:
            return _NAME_OF[key]
    return text


def object_name(label: str | None, resolved_label: str | None, resolved_share: float | None) -> tuple[str, str]:
    """(name, basis). The track's resolved class when its evidence is clear; otherwise the frame's
    detector label only if it is specific, else a broad category -- never a confident wrong name."""
    if resolved_label and (resolved_share or 0.0) >= MIN_NAME_SHARE:
        return canonical_name(resolved_label) or UNKNOWN_OBJECT, "resolved over the track"
    name = canonical_name(label)
    if not name or name in _BROAD.values():
        return name or UNKNOWN_OBJECT, "broad category: class not resolved yet"
    return name, "detector label (not yet confirmed over the track)"


def visible_material(scores: Iterable[tuple[str, float]], name: str | None) -> dict:
    """Visible exterior material from the track's visual votes; class prior only as a labelled fallback."""
    votes = [_VISIBLE_OF.get(str(label).lower(), "unknown") for label, _ in scores]
    votes = [v for v in votes if v != "unknown"]
    if votes:
        best, count = Counter(votes).most_common(1)[0]
        agreement = count / len(votes)
        if len(votes) >= MIN_MATERIAL_VOTES and agreement >= MIN_MATERIAL_AGREEMENT:
            return {"visible_material": best, "source": "visual classifier over the track",
                    "agreement": round(agreement, 4), "samples": len(votes)}
        return {"visible_material": "unknown", "source": "visual votes too few or split",
                "agreement": round(agreement, 4), "samples": len(votes)}
    prior = _CERTAIN_MATERIAL.get(name or "")
    if prior:
        return {"visible_material": prior, "source": f"object-class prior ({name}); not observed",
                "agreement": None, "samples": 0}
    return {"visible_material": "unknown", "source": "no visual material evidence", "agreement": None, "samples": 0}
