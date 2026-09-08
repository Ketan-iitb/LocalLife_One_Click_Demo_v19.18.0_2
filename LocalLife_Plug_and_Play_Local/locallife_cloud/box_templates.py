"""Known-box template matching (Revised Dual-Camera Volume Estimation recipe,
sections 5, 14, and 16 item 10).

Template mode is the PDF's recommended primary method for the 1 L / 1.5 L /
2 L milk boxes the user is testing with: measure the box's own dimensions
geometrically (`estimate_box_volume_cuboid()` in volume.py), then check
whether they match a known, physically-measured package. A match raises
confidence substantially over a bare geometric measurement, because the true
dimensions are known exactly rather than estimated from noisy single-view
depth.

Per the PDF's explicit instruction ("Record the real box dimensions with a
ruler/caliper; do not let code or an LLM invent them"), the shipped
`box_templates.yaml` ships with every dimension a `0` placeholder and
`measured: false`. `match_box_template()` only ever matches a template with
`measured: true` -- this module never fabricates a "close enough" match
against a template it knows is unmeasured. Replace the placeholders in
`box_templates.yaml` with your own caliper measurements of the exact milk
boxes you are testing with, and flip `measured: true`, to enable matching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_YAML_FILENAME = "box_templates.yaml"


@dataclass(slots=True)
class BoxTemplate:
    id: str
    nominal_volume_liters: float
    length_mm: float
    width_mm: float
    height_mm: float
    tolerance_mm: float
    measured: bool = False


@dataclass(slots=True)
class BoxTemplateMatch:
    template: BoxTemplate
    length_error_mm: float
    width_error_mm: float
    height_error_mm: float
    max_error_mm: float


def load_box_templates(path: str | None = None) -> list[BoxTemplate]:
    """Load templates from YAML. Missing file, missing PyYAML, or a malformed
    entry all degrade to an empty/partial list rather than raising -- box
    template matching is an optional confidence booster, not a hard
    dependency of the volume pipeline."""
    resolved_path = path or os.path.join(os.path.dirname(__file__), _DEFAULT_YAML_FILENAME)
    if not os.path.isfile(resolved_path):
        return []
    try:
        import yaml
    except ImportError:
        return []
    try:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return []
    templates: list[BoxTemplate] = []
    for entry in payload.get("boxes", []) or []:
        try:
            dimensions = entry["dimensions_mm"]
            templates.append(BoxTemplate(
                id=str(entry["id"]),
                nominal_volume_liters=float(entry["nominal_volume_liters"]),
                length_mm=float(dimensions["length"]),
                width_mm=float(dimensions["width"]),
                height_mm=float(dimensions["height"]),
                tolerance_mm=float(entry.get("tolerance_mm", 8.0)),
                measured=bool(entry.get("measured", False)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return templates


def match_box_template(
    length_mm: float,
    width_mm: float,
    height_mm: float,
    templates: list[BoxTemplate],
) -> BoxTemplateMatch | None:
    """Match measured dimensions against known templates.

    The footprint's length/width assignment relative to the camera isn't
    known in advance (the box could be rotated either way), so both
    length<->width pairings are tried and the better one kept; height is
    always the vertical (table-perpendicular) measurement and is never
    swapped with a horizontal dimension. Returns the closest template within
    its own tolerance, or None if nothing matches (including when every
    candidate template is `measured: false` -- see this module's docstring).
    """
    best: BoxTemplateMatch | None = None
    for template in templates:
        if not template.measured:
            continue
        for length_target, width_target in (
            (template.length_mm, template.width_mm),
            (template.width_mm, template.length_mm),
        ):
            length_error = abs(length_mm - length_target)
            width_error = abs(width_mm - width_target)
            height_error = abs(height_mm - template.height_mm)
            max_error = max(length_error, width_error, height_error)
            if max_error > template.tolerance_mm:
                continue
            if best is None or max_error < best.max_error_mm:
                best = BoxTemplateMatch(
                    template=template,
                    length_error_mm=length_error,
                    width_error_mm=width_error,
                    height_error_mm=height_error,
                    max_error_mm=max_error,
                )
    return best
