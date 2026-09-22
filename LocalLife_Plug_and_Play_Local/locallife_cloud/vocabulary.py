"""Canonical object names for the open-vocabulary detector.

The detector proposes many near-synonyms ("headset", "earphones"), and when it
is unsure it still returns one confident-looking specific label -- a folded
cloth came back as "cardboard box" and a carton as "toy". Reporting a parent
category ("textile item", "packaging object") is honest and still useful; a
wrong specific label is neither.

Geometry never reads any of this: shape_geometry.py decides from the points.
"""

from __future__ import annotations

# label fragment -> canonical name
SYNONYMS: dict[str, str] = {
    "headset": "headphones", "earphones": "headphones", "earbuds": "headphones",
    "ear cups": "headphones", "headphone": "headphones",
    "drill machine": "electric drill", "power drill": "electric drill", "cordless drill": "electric drill",
    "led bulb": "light bulb", "bulb": "light bulb", "lightbulb": "light bulb",
    "picture": "painting or picture frame", "framed artwork": "painting or picture frame",
    "picture frame": "painting or picture frame", "painting": "painting or picture frame",
    "rucksack": "backpack", "knapsack": "backpack",
    "power adapter": "charger", "power supply": "charger", "charging adapter": "charger",
    "charging cable": "cable", "electrical cable": "cable", "usb cable": "cable",
    "mobile phone": "phone", "smartphone": "phone", "cell phone": "phone",
    "computer mouse": "mouse", "remote": "remote control",
    "beverage carton": "carton", "milk carton": "carton", "juice carton": "carton",
    "aluminium can": "can", "aluminum can": "can", "metal can": "can", "drink can": "can",
    "soda can": "can", "tin can": "can",
    "cream bottle": "cosmetic bottle", "lotion bottle": "cosmetic bottle", "balm container": "cosmetic bottle",
    "folded clothing": "folded textile", "clothing": "textile item", "cloth": "textile item",
}

# canonical name -> parent category, used when the detector is unsure.
PARENTS: dict[str, str] = {
    "headphones": "electronic item", "charger": "electrical item", "cable": "electrical item",
    "phone": "electronic item", "laptop": "electronic item", "mouse": "electronic item",
    "keyboard": "electronic item", "remote control": "electronic item", "battery": "electrical item",
    "circuit board": "electronic item", "light bulb": "electrical item", "lamp": "electrical item",
    "table lamp": "electrical item", "torch": "electrical item", "flashlight": "electrical item",
    "electric drill": "power tool", "carton": "packaging object", "can": "packaging object",
    "plastic bottle": "packaging object", "glass bottle": "packaging object", "jar": "packaging object",
    "cup": "packaging object", "packet": "packaging object", "wrapper": "packaging object",
    "package": "packaging object", "cardboard box": "packaging object", "shipping box": "packaging object",
    "shoe box": "packaging object", "food container": "packaging object",
    "plastic bag": "packaging object", "paper bag": "packaging object", "filled waste bag": "packaging object",
    "backpack": "rigid household object", "handbag": "rigid household object", "book": "rigid household object",
    "shoe": "rigid household object", "slipper": "rigid household object",
    "painting or picture frame": "rigid household object", "cosmetic bottle": "packaging object",
    "toy": "rigid household object", "textile item": "textile item", "folded textile": "textile item",
    "pillow": "flexible household object", "cushion": "flexible household object",
}

PARENT_CATEGORIES = frozenset(PARENTS.values()) | {"unknown deposited object"}

# Below this the specific name is not trustworthy; the parent is reported.
SPECIFIC_CONFIDENCE = 0.45


def canonical_name(label: str) -> str:
    """One name per physical thing, whatever synonym the detector proposed."""
    text = " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())
    if not text:
        return "unknown deposited object"
    if text in SYNONYMS:
        return SYNONYMS[text]
    for fragment, canonical in SYNONYMS.items():
        if fragment in text:
            return canonical
    return text


def parent_category(label: str) -> str:
    canonical = canonical_name(label)
    if canonical in PARENT_CATEGORIES:
        return canonical
    for name, parent in PARENTS.items():
        if name in canonical:
            return parent
    return "unknown deposited object"


def object_type(label: str, confidence: float) -> str:
    """The name to report: the specific one when it is trustworthy, else its parent."""
    return canonical_name(label) if confidence >= SPECIFIC_CONFIDENCE else parent_category(label)
