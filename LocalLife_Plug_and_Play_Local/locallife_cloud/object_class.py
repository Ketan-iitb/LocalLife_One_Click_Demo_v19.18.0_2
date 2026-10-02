"""V49: one resolved object class per TRACK, from all of the detector's evidence for it.

Why: the open-vocabulary bank (config.DEFAULT_GEOMETRY_VALIDATION_PROMPTS, ~110 prompts) offers
"cardboard box", "cardboard shipping box", "carton box", "parcel", "package" AND "book", "pillow",
"textile item", "flexible household object"... YOLOE keeps one box per prompt (per-class NMS)
and the pipeline's de-duplication kept only the single most confident one, so the label shown
for one shipping box flipped book / pillow / cardboard shipping box frame to frame, and the
three box synonyms split the box's evidence three ways.

Here every overlapping proposal of a frame (kept + suppressed duplicates) is added to the track:
synonyms count as ONE class (max per frame, so a class with many prompts is not favoured),
catch-all parent prompts count half, and soft-object
classes (pillow, textile...) count half once the track's own depth shows a flat rigid top.
The raw detector label is never changed; `resolved_label` is the class to report.
"""

from __future__ import annotations

from typing import Iterable

# synonyms -> one reported class (only true synonyms; "book" stays a book)
CLASS_OF: dict[str, str] = {
    "cardboard box": "cardboard box", "cardboard shipping box": "cardboard box", "carton box": "cardboard box",
    "shipping box": "cardboard box", "parcel": "cardboard box", "package": "cardboard box",
    "plastic garbage bag": "plastic bag", "plastic trash bag": "plastic bag", "filled plastic waste bag": "plastic bag",
    "polythene waste bag": "plastic bag", "transparent plastic waste bag": "plastic bag",
    "plastic waste bag": "plastic bag", "waste bag": "plastic bag", "garbage bag": "plastic bag",
    "trash bag": "plastic bag", "plastic bag": "plastic bag",
    "paper waste bag": "paper bag", "kraft paper bag": "paper bag", "paper shopping bag": "paper bag",
    "milk carton": "drink carton", "drink carton": "drink carton", "beverage carton": "drink carton",
    "power drill": "electric drill", "electric drill": "electric drill",
}
# Parent / fallback prompts: honest when nothing better exists, weak evidence otherwise.
CATCH_ALL = frozenset({"textile item", "flexible household object", "rigid household object", "packaging object",
                       "unknown deposited object", "electronic item", "electrical item", "decorative object"})
# Soft, deformable classes: contradicted by a flat rigid top seen in the depth.
SOFT = frozenset({"pillow", "cushion", "blanket", "bedding", "clothing", "folded clothing", "textile",
                  "textile item", "curtain", "drape", "flexible household object", "plastic bag"})
# Class -> material it implies (material.py label names). Only physical certainties.
MATERIAL_OF: dict[str, str] = {
    "cardboard box": "cardboard", "plastic bag": "polythene bag", "paper bag": "paper",
    "drink carton": "cardboard", "can": "metal", "soda can": "metal", "tin can": "metal",
    "aluminium drink can": "metal", "plastic bottle": "plastic", "pillow": "fabric or textile",
    "cushion": "fabric or textile", "folded clothing": "fabric or textile", "clothing": "fabric or textile",
    "blanket": "fabric or textile", "book": "paper",
}

# Closed rigid objects with flat faces: measured as a cuboid / slab.
FLAT_FACED = frozenset({"cardboard box", "book", "drink carton", "shoe box"})

# Generous physical size limits (cm): a class whose measured object exceeds them is implausible
# and its evidence counts 0.3. Only classes with a well-known size range are listed.
SIZE_LIMIT_CM: dict[str, tuple[float, float]] = {   # class -> (max longest side, max thickness)
    "book": (40.0, 8.0),
    "soda can": (20.0, 10.0), "can": (20.0, 10.0),
}

DECAY = 0.96                # per frame: old frames fade, ~25 frames half-life
MIN_EVIDENCE = 1.0          # summed confidence before a class is "resolved"
MIN_SHARE = 0.45            # winning share of the evidence


def class_of(label: str | None) -> str:
    text = " ".join(str(label or "").lower().replace("_", " ").replace("-", " ").split())
    return CLASS_OF.get(text, text or "unknown")


class ClassResolver:
    def __init__(self) -> None:
        self.scores: dict[int, dict[str, float]] = {}
        self.rigid: dict[int, bool] = {}
        self.size: dict[int, tuple[float, float]] = {}

    def update(self, track_id: int, candidates: Iterable[tuple[str, float]]) -> None:
        scores = self.scores.setdefault(track_id, {})
        for key in scores:
            scores[key] *= DECAY
        best: dict[str, float] = {}
        for label, confidence in candidates:           # one vote per class per frame (max over synonyms)
            name = class_of(label)
            weight = 0.5 if name in CATCH_ALL else 1.0
            best[name] = max(best.get(name, 0.0), float(confidence) * weight)
        for name, value in best.items():
            scores[name] = scores.get(name, 0.0) + value
        if len(self.scores) > 400:
            for key in list(self.scores)[:200]:
                self.scores.pop(key, None)
                self.rigid.pop(key, None)
                self.size.pop(key, None)

    def set_rigid(self, track_id: int, rigid: bool) -> None:
        if rigid:
            self.rigid[track_id] = True

    def set_size(self, track_id: int, length_cm: float | None, height_cm: float | None) -> None:
        """This track's own measured size (longest side, thickness above its support)."""
        if length_cm and height_cm:
            self.size[track_id] = (float(length_cm), float(height_cm))

    def resolve(self, track_id: int | None) -> tuple[str | None, float, float]:
        """(class, share of evidence, total evidence); class None until evidence suffices."""
        scores = self.scores.get(track_id) if track_id is not None else None
        if not scores:
            return None, 0.0, 0.0
        rigid = self.rigid.get(track_id, False)
        size = self.size.get(track_id)

        def weight(name: str) -> float:
            w = 0.5 if rigid and name in SOFT else 1.0
            limit = SIZE_LIMIT_CM.get(name)
            if size is not None and limit is not None and (size[0] > limit[0] or size[1] > limit[1]):
                w *= 0.3
            return w
        weighted = {name: value * weight(name) for name, value in scores.items()}
        total = sum(weighted.values())
        name = max(weighted, key=weighted.get)
        share = weighted[name] / total if total > 0 else 0.0
        if total < MIN_EVIDENCE or share < MIN_SHARE:
            return None, round(share, 2), round(total, 2)
        return name, round(share, 2), round(total, 2)

    def forget(self) -> None:
        self.scores.clear()
        self.rigid.clear()
        self.size.clear()
