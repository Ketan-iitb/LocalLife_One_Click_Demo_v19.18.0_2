"""V49 bin policy: this bin accepts PLASTIC BAGS only; anything else is a MIS-SORT.

`sorting_rules.py` (protected) keeps the playbook table, where a cardboard box is an allowed
calibration object and every pillow/textile label is a mis-sort. On this rig the detector reads
most plastic bags as "textile item" or "pillow", so the deployed verdict here combines the label
with the material classifier:

* a clearly non-bag label (box, book, shoe, drill, bottle, ...)       -> MIS-SORT
* a bag label (bag, sack, garbage, ...) unless paper/cardboard        -> CORRECT
* a soft label (textile, pillow, household, clothing, object ...):
  plastic / polythene material -> CORRECT; paper, cardboard, metal, organic or
  confident fabric -> MIS-SORT; otherwise CHECK (shown amber, never guessed).
"""

from __future__ import annotations

from .sorting_rules import MIS_SORT_FAMILIES, label_words

CORRECT, MIS_SORT, CHECK, IGNORE = "correct", "mis_sort", "check", "ignore"
TEXT = {CORRECT: "OK (plastic bag)", MIS_SORT: "MIS-SORT", CHECK: "CHECK", IGNORE: "—"}

_BOXLIKE = {"box", "boxes", "carton", "cartons", "cardboard", "shipping", "parcel", "package", "packaging", "crate"}
_HARD = set().union(*(v for k, v in MIS_SORT_FAMILIES.items() if k not in {"textile", "person"})) | _BOXLIKE | {
    "bottle", "bottles", "can", "cans", "tin", "jar", "glass", "newspaper", "magazine", "toy", "ball",
    "helmet", "shoebox", "plate", "cup", "mug", "phone", "keyboard", "mouse", "remote", "umbrella",
    "suitcase", "backpack", "handbag", "metal", "wood", "wooden", "brick", "stone", "book", "books",
}
_BAG = {"bag", "bags", "sack", "sacks", "binbag", "garbage", "trash", "rubbish", "refuse", "polythene", "plastic"}
_PERSON = MIS_SORT_FAMILIES["person"] | {"person", "people", "hands"}
_PLASTIC = {"plastic", "polythene bag", "polythene", "plastic bag"}
_NON_BAG_MATERIAL = {"paper", "cardboard", "metal", "food or organic waste"}


def verdict(label: str | None, material: str | None = None, material_confidence: float | None = None,
            colour: str | None = None) -> dict[str, str]:
    """{"status", "text", "reason", "object", "material"} for one detection or deposit."""
    words = label_words(label)
    mat = str(material or "").strip().lower()
    conf = 1.0 if material_confidence is None else float(material_confidence)
    if words & _PERSON:
        return _out(IGNORE, "person/hand, not waste", label, material)
    if words & _HARD:
        if words & _BOXLIKE and (mat in {"", "unknown", "paper", "cardboard"} or str(colour or "").lower() == "brown"):
            return _out(MIS_SORT, f"cardboard box (detector: {label}) — not a plastic bag", "cardboard box", "cardboard")
        return _out(MIS_SORT, f"{label} — not a plastic bag", label, material)
    if words & _BAG:
        if "paper" in words or (mat in {"paper", "cardboard"} and conf >= 0.5):
            return _out(MIS_SORT, f"paper bag (detector: {label})", "paper bag", "paper")
        return _out(CORRECT, "bag", label, material)
    if mat in _PLASTIC:
        return _out(CORRECT, f"plastic material (detector said {label})", "plastic bag", material)
    if mat in _NON_BAG_MATERIAL and conf >= 0.5:
        return _out(MIS_SORT, f"{mat} item (detector: {label})", label, material)
    if mat in {"fabric or textile", "textile"} and conf >= 0.6:
        return _out(MIS_SORT, f"textile item (material {round(conf * 100)} %)", label, material)
    return _out(CHECK, f"{label or 'object'}: material {mat or 'unknown'} — check by eye", label, material)


def _out(status: str, reason: str, obj: str | None, material: str | None) -> dict[str, str]:
    return {"status": status, "text": TEXT[status], "reason": reason, "object": obj or "unknown",
            "material": material or "unknown"}


def apply_to_event(event: dict) -> dict:
    """Sorting fields of a deposit row from its RESOLVED object class and material; idempotent.

    A foreground change alone proves a deposit, not its class: no track, or a track whose class is
    not resolved yet, stays CHECK -- never a guessed OK or MIS-SORT."""
    resolved = event.get("object_class")
    if event.get("track_id") is None:
        v = _out(CHECK, "image change only — no detection to classify", None, None)
    elif not resolved:
        v = _out(CHECK, f"object class not resolved (detector: {event.get('detector_label')})", None, None)
    else:
        v = verdict(resolved, event.get("material"), None, event.get("colour"))
    event["sorting"] = v["text"]
    event["sorting_status"] = v["status"]
    event["sorting_reason"] = v["reason"]
    return event
