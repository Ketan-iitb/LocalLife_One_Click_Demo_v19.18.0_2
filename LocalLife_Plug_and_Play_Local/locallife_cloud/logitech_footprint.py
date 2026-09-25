"""Footprint of a standing object seen by one fixed, tilted camera.

The bug this replaces
---------------------
`MeasurementZone.footprint_m` warped *every* mask pixel through the mat
homography and fitted a rectangle to the result. A homography maps the floor
plane, and only the floor plane. A pixel belonging to the top of a can is not on
the floor, so warping it does not give that point's ground position -- it gives
where the camera's ray through it *would* meet the floor, which is further away
from the camera. The silhouette is therefore smeared into a long shadow, and the
rectangle fitted to that shadow is the object's footprint plus its height,
leaning away from the lens.

That predicts exactly the field results:

    object          true L x W      reported L x W    height
    shoe box        333 x 262       334 x 294         low, wide
    bag             259 x 237       282 x 275         tall but wide
    can              47 x  28       115 x  85          84 mm
    cosmetic bottle   75 x  32       227 x 133         204 mm

Flat and wide objects barely move. The taller and narrower the object, the worse
it gets -- a 3x error on the bottle -- because the smear is proportional to
height and the object's own width no longer hides it.

What replaces it
----------------
Only the pixels where the object actually meets the floor may be warped. For a
camera looking down from the front, that is the lower boundary of the
silhouette: the contact line. Warping just that band gives the object's real
footprint.

A contact band of a round object is an arc, not a rectangle -- the far side of
the base is hidden behind the object itself -- so the extents come from
`footprint.estimate_extents`, which already reconstructs a circular
cross-section from a visible arc. The same geometry that made a RealSense
cylinder read 5 x 2 cm applies here.

Where a per-pixel height and the camera height are both known, the smear can be
undone exactly rather than avoided: a point at height h, seen from a camera at
height H, lands on the plane displaced away from the camera's ground point by
H / (H - h). That path is used when the nadir is calibrated, and the contact
band is the fallback that needs nothing but the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .footprint import estimate_extents

# Reasons a footprint could not be measured. Each names the missing thing, so an
# operator is never shown the bare word "pending".
CALIBRATION_MISSING = "calibration_missing"
RESOLUTION_MISMATCH = "resolution_mismatch"
INSUFFICIENT_CONTACT = "insufficient_contact_pixels"
INSUFFICIENT_OBJECT_DEPTH = "insufficient_object_depth"
OBJECT_BELOW_NOISE_FLOOR = "object_below_noise_floor"
MASK_DEPTH_SHAPE_MISMATCH = "mask_depth_shape_mismatch"
IMPLAUSIBLE_FOOTPRINT = "implausible_footprint"

# The contact band is a fraction of the silhouette's own pixel height, so it
# scales with the object: a tall bottle contributes a proportionally thin base
# band, a flat box almost all of itself.
CONTACT_BAND_FRACTION = 0.22
MIN_CONTACT_ROWS = 3


@dataclass
class FootprintResult:
    length_m: float | None = None
    width_m: float | None = None
    area_m2: float | None = None
    method: str = ""
    reason: str | None = None
    contact_pixels: int = 0
    occlusion_corrected: bool = False
    diagnostics: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.length_m is not None and self.width_m is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_mm": None if self.length_m is None else round(self.length_m * 1000.0, 2),
            "width_mm": None if self.width_m is None else round(self.width_m * 1000.0, 2),
            "area_m2": self.area_m2,
            "method": self.method,
            "reason": self.reason,
            "contact_pixels": self.contact_pixels,
            "occlusion_corrected": self.occlusion_corrected,
            "diagnostics": self.diagnostics or {},
        }


def contact_band_mask(mask: np.ndarray, fraction: float = CONTACT_BAND_FRACTION) -> np.ndarray:
    """The lower band of a silhouette: where the object meets the floor.

    Rows increase downward in an image, and a camera mounted above and in front
    of the mat sees an object's base at the bottom of its silhouette. Taking a
    fraction of the silhouette's own row extent keeps the band proportional to
    the object rather than fixed in pixels, so it does not swallow a short box
    or miss the base of a tall bottle.
    """
    array = np.asarray(mask, dtype=bool)
    rows = np.nonzero(array.any(axis=1))[0]
    if rows.size == 0:
        return np.zeros_like(array)
    top, bottom = int(rows[0]), int(rows[-1])
    span = bottom - top + 1
    band = max(MIN_CONTACT_ROWS, int(round(span * max(0.01, fraction))))
    cut = max(top, bottom - band + 1)
    result = np.zeros_like(array)
    result[cut : bottom + 1, :] = array[cut : bottom + 1, :]
    return result


def _shrink_towards_nadir(
    points: np.ndarray, heights_m: np.ndarray, nadir: np.ndarray, camera_height_m: float,
) -> np.ndarray | None:
    """Undo the height smear exactly, given the camera's ground point.

    A point at height h above the plane projects along the camera ray to a place
    on the plane that is further from the nadir by H / (H - h). Dividing that
    back out returns the point's true ground position, so a standing object's
    whole silhouette -- not just its base -- can be used.
    """
    if camera_height_m <= 0:
        return None
    usable = heights_m < camera_height_m * 0.9
    if not np.any(usable):
        return None
    scale = (camera_height_m - heights_m[usable]) / camera_height_m
    offsets = points[usable] - nadir
    return nadir + offsets * scale[:, None]


def measure_footprint(
    zone: Any,
    mask: np.ndarray,
    *,
    heights_m: np.ndarray | None = None,
    camera_height_m: float | None = None,
    nadir_xy_m: tuple[float, float] | None = None,
    zone_limits_m: tuple[float, float] | None = None,
) -> FootprintResult:
    """The object's footprint on the mat, without the height smear.

    `heights_m` and `nadir_xy_m` enable the exact correction; without them the
    contact band is used, which needs only the mask.
    """
    if zone is None or not getattr(zone, "has_floor_scale", False):
        return FootprintResult(reason=CALIBRATION_MISSING)
    array = np.asarray(mask, dtype=bool)
    if not array.any():
        return FootprintResult(reason=INSUFFICIENT_CONTACT)
    if heights_m is not None and np.asarray(heights_m).shape != array.shape:
        return FootprintResult(reason=MASK_DEPTH_SHAPE_MISMATCH)

    diagnostics: dict[str, Any] = {"mask_pixels": int(array.sum())}
    points: np.ndarray | None = None
    method = ""

    # Preferred: correct every pixel back to its true ground position.
    if (
        heights_m is not None
        and nadir_xy_m is not None
        and camera_height_m is not None
        and camera_height_m > 0
    ):
        rows, columns = np.nonzero(array)
        warped = zone.ground_points(rows, columns, array.shape)
        if warped is not None and len(warped) >= 3:
            corrected = _shrink_towards_nadir(
                np.asarray(warped, dtype=np.float64),
                np.asarray(heights_m, dtype=np.float64)[array],
                np.asarray(nadir_xy_m, dtype=np.float64),
                float(camera_height_m),
            )
            if corrected is not None and len(corrected) >= 3:
                points = corrected
                method = "logitech_height_corrected_footprint"
                diagnostics["corrected_pixels"] = int(len(corrected))

    # Fallback: only the contact band, which is genuinely on the floor.
    if points is None:
        band = contact_band_mask(array)
        rows, columns = np.nonzero(band)
        diagnostics["contact_pixels"] = int(rows.size)
        if rows.size < 3:
            return FootprintResult(reason=INSUFFICIENT_CONTACT, diagnostics=diagnostics)
        warped = zone.ground_points(rows, columns, array.shape)
        if warped is None or len(warped) < 3:
            return FootprintResult(reason=INSUFFICIENT_CONTACT, diagnostics=diagnostics)
        points = np.asarray(warped, dtype=np.float64)
        method = "logitech_contact_band_footprint"

    # The visible contact of a round base is an arc: the far side is hidden
    # behind the object. estimate_extents reconstructs the full cross-section,
    # and leaves a filled rectangle -- a box's base -- untouched.
    extents = estimate_extents(points)
    if extents is None:
        return FootprintResult(reason=INSUFFICIENT_CONTACT, diagnostics=diagnostics)

    length, width = extents.length_m, extents.width_m
    if zone_limits_m is not None:
        limit = max(zone_limits_m)
        if length > limit or width > limit:
            # Larger than the calibrated mat is not a measurement, it is a
            # failure -- reported rather than clamped into looking plausible.
            diagnostics["zone_limit_m"] = limit
            return FootprintResult(
                reason=IMPLAUSIBLE_FOOTPRINT, method=method, diagnostics=diagnostics,
            )

    area = None
    try:
        area_map = zone.pixel_area_m2(array.shape)
        if area_map is not None:
            area = float(area_map[array].sum())
    except Exception:  # noqa: BLE001 - area is a diagnostic, never a blocker
        area = None
    diagnostics["extent_flags"] = list(extents.flags)
    return FootprintResult(
        length_m=length, width_m=width, area_m2=area, method=method,
        contact_pixels=int(diagnostics.get("contact_pixels", diagnostics["mask_pixels"])),
        occlusion_corrected=extents.occlusion_corrected,
        diagnostics=diagnostics,
    )


def plausible_height_m(
    height_m: float | None, camera_height_m: float | None, noise_floor_m: float = 0.008,
) -> tuple[bool, str | None]:
    """Is this height physically possible, and above the sensor's own noise?

    Separate reasons, because "too small to distinguish from noise" and "taller
    than the camera is mounted" call for completely different actions.
    """
    if height_m is None:
        return False, INSUFFICIENT_OBJECT_DEPTH
    if height_m <= noise_floor_m:
        return False, OBJECT_BELOW_NOISE_FLOOR
    if camera_height_m is not None and camera_height_m > 0 and height_m >= camera_height_m:
        return False, IMPLAUSIBLE_FOOTPRINT
    return True, None
