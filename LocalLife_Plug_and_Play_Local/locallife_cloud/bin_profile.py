"""Per-camera, physically measured bin profile and the physical bounds it implies.

Each camera (RealSense, Logitech) has its OWN profile: the tripods put the two
optical centres at different heights and tilts, so they never share a
camera-to-floor distance or calibration.

Until someone tapes the real bin, the profile is the NOMINAL 660 L four-wheel
container (typical EN 840 inner size), marked `unverified`. Nominal geometry
is only a broad sanity bound -- a bag cannot be longer than the bin is wide --
and is never used as a scale or a volume formula; the nominal capacity is
never turned into L x W x H.

A profile becomes `measured` only when every required tape measurement is
present AND a reference-object check (rigid objects of known size placed at
several spots in the bin, measured by this camera) stays within tolerance.
Anything less is stored with the reason it was not accepted.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Typical inner geometry of a 660 L EN 840 container (flat lid). NOT measured on
# site: the operator must replace it with tape measurements of the actual bin.
NOMINAL_660L = {
    "capacity_label_l": 660.0,
    "inner_length_top_m": 1.25, "inner_width_top_m": 0.72,
    "inner_length_floor_m": 1.10, "inner_width_floor_m": 0.60,
    "inner_depth_m": 1.00,
    "source": "nominal EN 840 660 L four-wheel container (typical inner size) -- UNVERIFIED, measure on site",
}

REQUIRED_MEASUREMENTS = (
    "optical_center_to_empty_floor_m", "optical_center_above_rim_m", "inner_depth_m",
    "inner_length_top_m", "inner_width_top_m", "inner_length_floor_m", "inner_width_floor_m",
    "tilt_deg",
)
MIN_REFERENCE_PLACEMENTS = 5          # >= 3 objects, several positions
MIN_REFERENCE_OBJECTS = 3
MAX_MEDIAN_SCALE_ERROR = 0.05         # 5 % median dimension error
MAX_WORST_SCALE_ERROR = 0.12          # 12 % worst single dimension
POSE_TILT_TOLERANCE_DEG = 3.0
POSE_DISTANCE_TOLERANCE = 0.05        # 5 % of the measured optical-centre-to-floor distance
OVERFILL_ALLOWANCE_M = 0.30           # waste may stand above the rim of an overfull bin


@dataclass
class ReferencePlacement:
    """One rigid object of known size, placed in the bin and measured by this camera."""
    object_name: str
    position: str                      # e.g. "centre", "near-left", "far-right"
    true_mm: tuple[float, float, float]
    measured_mm: tuple[float, float, float] | None


@dataclass
class BinProfile:
    camera_id: str
    measurements: dict[str, float] = field(default_factory=dict)
    uncertainty_mm: dict[str, float] = field(default_factory=dict)
    method: str = ""
    measured_by: str = ""
    measured_at: float | None = None
    capacity_label: str = "bin capacity unverified (check the bin label / product)"
    references: list[ReferencePlacement] = field(default_factory=list)
    status: str = "unverified-nominal"
    status_reason: str = "no on-site measurements recorded; using nominal 660 L bounds only"
    validation: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- geometry
    def geometry(self) -> dict[str, float]:
        """Measured geometry when valid, else the nominal 660 L bounds (labelled)."""
        if self.status == "measured":
            return {k: float(self.measurements[k]) for k in self.measurements}
        return {k: v for k, v in NOMINAL_660L.items() if isinstance(v, float)}

    def bounds(self) -> dict[str, Any]:
        g = self.geometry()
        diagonal = math.hypot(g["inner_length_top_m"], g["inner_width_top_m"])
        return {
            "max_footprint_m": round(diagonal, 3),
            "max_height_m": round(g["inner_depth_m"] + OVERFILL_ALLOWANCE_M, 3),
            "source": "measured bin profile" if self.status == "measured" else NOMINAL_660L["source"],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bounds"] = self.bounds()
        return payload


def validate(profile: BinProfile) -> BinProfile:
    """Accept a profile as `measured` only on complete tape data and a passing reference check."""
    missing = [k for k in REQUIRED_MEASUREMENTS if not _positive_or_angle(k, profile.measurements.get(k))]
    if missing:
        profile.status, profile.status_reason = "rejected", "missing or invalid measurements: " + ", ".join(missing)
        return profile
    m = profile.measurements
    if m["optical_center_to_empty_floor_m"] <= m["inner_depth_m"] * 0.5:
        profile.status, profile.status_reason = "rejected", (
            "optical-centre-to-floor distance is less than half the inner depth; re-measure")
        return profile
    if m["inner_length_floor_m"] > m["inner_length_top_m"] * 1.05 or m["inner_width_floor_m"] > m["inner_width_top_m"] * 1.05:
        profile.status_reason = "note: floor wider than rim -- check the taper measurements"
    placements = [r for r in profile.references if r.measured_mm is not None]
    objects = {r.object_name for r in placements}
    if len(placements) < MIN_REFERENCE_PLACEMENTS or len(objects) < MIN_REFERENCE_OBJECTS:
        profile.status, profile.status_reason = "rejected", (
            f"reference check needs >= {MIN_REFERENCE_OBJECTS} rigid objects over >= {MIN_REFERENCE_PLACEMENTS} "
            f"placements measured by this camera (have {len(objects)} objects, {len(placements)} placements)")
        return profile
    errors = []
    for placement in placements:
        for true, got in zip(sorted(placement.true_mm), sorted(placement.measured_mm)):
            if true > 0:
                errors.append(abs(got - true) / true)
    median, worst = statistics.median(errors), max(errors)
    profile.validation = {"dimension_errors": len(errors), "median_rel_error": round(median, 4),
                          "worst_rel_error": round(worst, 4), "checked_at": time.time()}
    if median > MAX_MEDIAN_SCALE_ERROR or worst > MAX_WORST_SCALE_ERROR:
        profile.status, profile.status_reason = "rejected", (
            f"reference check failed: median error {median:.1%} (limit {MAX_MEDIAN_SCALE_ERROR:.0%}), "
            f"worst {worst:.1%} (limit {MAX_WORST_SCALE_ERROR:.0%})")
        return profile
    profile.status = "measured"
    if not profile.status_reason.startswith("note:"):
        profile.status_reason = f"validated: median error {median:.1%}, worst {worst:.1%}"
    return profile


def _positive_or_angle(key: str, value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    return -89.0 < number < 89.0 if key == "tilt_deg" else number > 0


def pose_check(profile: BinProfile, live_tilt_deg: float | None, live_floor_distance_m: float | None,
               *, empty_bin: bool) -> dict[str, Any]:
    """Compare the live fitted floor with the measured pose; only an empty-bin frame is decisive."""
    if profile.status != "measured":
        return {"state": "no_measured_profile"}
    if not empty_bin:
        return {"state": "not_checked", "reason": "the floor is covered by waste; a crowded frame cannot verify the pose"}
    problems = []
    if live_tilt_deg is not None and abs(live_tilt_deg - profile.measurements["tilt_deg"]) > POSE_TILT_TOLERANCE_DEG:
        problems.append(f"tilt {live_tilt_deg:.1f}° vs measured {profile.measurements['tilt_deg']:.1f}°")
    expected = profile.measurements["optical_center_to_empty_floor_m"]
    if live_floor_distance_m is not None and abs(live_floor_distance_m - expected) > POSE_DISTANCE_TOLERANCE * expected:
        problems.append(f"floor distance {live_floor_distance_m:.3f} m vs measured {expected:.3f} m")
    return {"state": "moved" if problems else "unchanged", "problems": problems}


def physical_check(bounds: dict[str, Any], length_mm: float | None, width_mm: float | None,
                   height_mm: float | None) -> str | None:
    """A reason when dimensions cannot exist inside this bin, else None. Never clamps."""
    if height_mm is not None and height_mm <= 0:
        return "physically_impossible_non_positive_height"
    if height_mm is not None and height_mm / 1000.0 > bounds["max_height_m"]:
        return "physically_impossible_height_exceeds_bin_depth"
    for value in (length_mm, width_mm):
        if value is not None and value / 1000.0 > bounds["max_footprint_m"]:
            return "physically_impossible_footprint_exceeds_bin"
    return None


class BinProfileStore:
    """One JSON file per camera, written atomically."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, camera_id: str) -> Path:
        return self.directory / f"{camera_id}.json"

    def load(self, camera_id: str) -> BinProfile:
        try:
            data = json.loads(self.path(camera_id).read_text(encoding="utf-8"))
            data["references"] = [ReferencePlacement(**{**r, "true_mm": tuple(r["true_mm"]),
                                                         "measured_mm": None if r.get("measured_mm") is None
                                                         else tuple(r["measured_mm"])})
                                  for r in data.get("references", [])]
            return BinProfile(**{k: v for k, v in data.items() if k in BinProfile.__dataclass_fields__})
        except (OSError, ValueError, TypeError, KeyError):
            return BinProfile(camera_id=camera_id)

    def save(self, profile: BinProfile) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(profile.camera_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
        os.replace(temporary, target)
