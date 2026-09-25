"""One physical bag should be one detection.

The bin screenshots show a red bag carrying two boxes at once: "#10 test object
(filled plastic waste bag) (tracked)" and "#11 unclassified object [red]",
drawn over each other on the same object. The unclassified one comes from the
scene-fusion bridge, which exists to carry an already-counted track through a
frame where the detector drops it. When the detector has *not* dropped it, the
bridge and the detection both survive and the same bag is presented twice: two
rows, two ids, and a bag count that is one too many.

The rule here is deliberately narrow. An unclassified region is removed only
when a classified detection already covers most of it -- so the classified
object is always the one kept, the real bag count cannot fall, and a genuine
unclassified region standing on its own is untouched. Nothing about the
detector, its vocabulary, its masks or its confidences changes.

Overlap is judged against the *smaller* box. A small bridge region sitting
inside a large bag's box barely moves that box's own overlap fraction, and it
is exactly the case to catch.
"""

from __future__ import annotations

from typing import Any

# What fraction of the smaller box must lie inside the larger before the two
# are the same physical object.
DUPLICATE_OVERLAP = 0.60

# Labels that carry no class: these are regions the fusion step proposed, not
# objects the detector recognised.
UNCLASSIFIED_LABELS = frozenset({
    "unclassified object", "unknown", "unknown object", "foreground object",
})


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def is_unclassified(label: str | None) -> bool:
    return (label or "").strip().lower() in UNCLASSIFIED_LABELS


def drop_unclassified_duplicates(
    detections: list[Any], *, overlap: float = DUPLICATE_OVERLAP,
) -> tuple[list[Any], int]:
    """Remove unclassified regions a classified detection already covers.

    Returns the surviving detections and how many were dropped, so the caller
    can count what happened rather than have it disappear silently.
    """
    classified = [
        item for item in detections
        if not is_unclassified(getattr(item, "label", None))
        and getattr(item, "box", None) is not None
    ]
    if not classified:
        return detections, 0
    kept: list[Any] = []
    dropped = 0
    for item in detections:
        box = getattr(item, "box", None)
        if box is None or not is_unclassified(getattr(item, "label", None)):
            kept.append(item)
            continue
        own = _area(tuple(float(value) for value in box))
        duplicate = False
        for other in classified:
            other_box = tuple(float(value) for value in other.box)
            smaller = min(own, _area(other_box))
            if smaller <= 0:
                continue
            if _intersection(tuple(float(v) for v in box), other_box) >= overlap * smaller:
                duplicate = True
                break
        if duplicate:
            dropped += 1
        else:
            kept.append(item)
    return kept, dropped
