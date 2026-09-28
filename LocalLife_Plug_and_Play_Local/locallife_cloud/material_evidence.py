"""Keep what an object *is*, what its outside is made of, and what it holds apart.

The crop classifier (material.py, unchanged) votes for one of a few material
names per sampled frame, and the most common vote was published with its vote
share as confidence. Three things went wrong with that:

* identity leaked into material: a handbag's crop scores high for
  "polythene bag" because it *is a bag*, and was published as one. A handbag
  is not a waste bag; what its outside is made of is plastic at most;
* a translucent bag's crop shows its contents, so the vote describes the
  contents as often as the bag;
* a split vote -- two frames "paper", two "fabric", one "plastic" -- was still
  published as a material, at 40 %.

This decides the published material from the votes and the object's
identity, and says "unknown", with the confidence it did have, when the
evidence does not settle it. Sorting rules are not touched: they read the
detector class, not this field.
"""

from __future__ import annotations

from collections import Counter

from .waste_bag_names import NOT_WASTE_BAG_WORDS

UNKNOWN = "unknown"
MIN_VOTES_FOR_SPLIT_CHECK = 3
MIN_AGREEMENT = 0.5
EXTERIOR_OF_A_NON_WASTE_BAG = {"polythene bag": "plastic", "mixed or general waste": UNKNOWN,
                               "food or organic waste": UNKNOWN}


def _words(label: str | None) -> set[str]:
    return set((label or "").lower().replace("-", " ").replace("_", " ").split())


def reconcile_material(
    label: str | None, votes: list[str], *, colour_state: str | None = None,
) -> tuple[str, float, dict]:
    """Published material, its confidence and the reasoning, from the classifier's votes."""
    evidence: dict = {"votes": dict(Counter(votes)), "notes": []}
    if not votes:
        return UNKNOWN, 0.0, {**evidence, "notes": ["no_confident_classifier_vote"]}
    best, count = Counter(votes).most_common(1)[0]
    share = count / len(votes)
    material = best
    if len(votes) >= MIN_VOTES_FOR_SPLIT_CHECK and share < MIN_AGREEMENT:
        evidence["notes"].append("material_votes_split")
        material = UNKNOWN
    words = _words(label)
    if words & NOT_WASTE_BAG_WORDS and best == "polythene bag" and share >= 0.6 \
            and len(votes) >= MIN_VOTES_FOR_SPLIT_CHECK:
        # The detector's name and the material disagree: a dark waste bag is
        # often called a "handbag" by appearance alone. Reported; the name is
        # the detector's and the sorting rules still read it.
        evidence["notes"].append("identity_conflict_material_suggests_plastic_waste_bag")
    if words & NOT_WASTE_BAG_WORDS and material in EXTERIOR_OF_A_NON_WASTE_BAG:
        # Identity is not material: the classifier recognised a bag.
        evidence["notes"].append("identity_is_not_a_waste_bag")
        material = EXTERIOR_OF_A_NON_WASTE_BAG[material]
    if colour_state == "see_through_or_background_coloured":
        # The crop shows the contents through the bag; the material named is
        # the bag's own, and contents are not classified.
        evidence["notes"].append("translucent_contents_visible_not_classified")
    evidence["identity"] = label
    evidence["exterior_material"] = material
    return material, round(float(share), 4), evidence
