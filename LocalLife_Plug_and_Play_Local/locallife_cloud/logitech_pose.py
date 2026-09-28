"""The Logitech's pose: a scale anchor that is right on a tilted mount, and a guard against moving it.

Two scale anchors turned the operator's tape-measured camera height into a
depth scale, and both assumed the camera looks straight down:

* the reference-distance anchor divided the height by the *median depth* over
  the bin. On a tilted camera the depth to the floor is longer than the
  perpendicular height, by 1/cos(tilt) at the image centre and more towards
  the far edge, so every Logitech dimension came out too small -- by about a
  tenth at 25 degrees from straight down and by half at 70 degrees;
* the empty-scene alignment (`fit_plane_alignment`) fitted the prediction to a
  floor *perpendicular to the optical axis*. A tilted floor cannot be mapped
  onto that by a scale and an offset, so the fit flattened the depth map and
  no object had any height left.

The fix fits the floor plane in the prediction itself and scales so that the
plane's perpendicular distance from the camera equals the measured height. A
plane is a plane whatever its tilt, so the anchor is the same at every angle.
Only a scale is fitted: one plane at one known distance fixes a scale and
nothing else.

The guard records the floor the setup was made against -- its normal and the
camera's height above it -- and compares later fits to it. A calibration made
in one pose (the fitted height mapping, the empirical volume factors) does not
describe another, so a moved camera makes them provisional instead of silently
reusing them.

Nothing here reads RealSense data.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .logitech_autocal import camera_height_from_plane
from .types import CameraIntrinsics, DepthCalibration
from .volume import fit_reference_plane, reference_plane_is_usable

PLANE_SCALE_METHOD = "empty-plane-perpendicular-height"
# Past this the axis-aligned alignment cannot describe the floor.
AXIS_ALIGNMENT_MAX_TILT_DEG = 5.0
# A pose differing by more than this is a different installation.
POSE_TILT_TOLERANCE_DEG = 6.0
POSE_HEIGHT_TOLERANCE = 0.10
# Consecutive disagreeing live fits before a pose change is declared, so one
# noisy monocular frame does not revoke a calibration.
POSE_CHANGE_FRAMES = 3


def plane_height_scale(
    predicted: np.ndarray,
    region: np.ndarray | None,
    intrinsics: CameraIntrinsics,
    height_m: float,
) -> tuple[DepthCalibration | None, dict[str, Any]]:
    """Scale that puts the empty floor at the measured perpendicular height."""
    diagnostics: dict[str, Any] = {"camera_height_m": float(height_m), "method": PLANE_SCALE_METHOD}
    if not math.isfinite(height_m) or height_m <= 0:
        return None, {**diagnostics, "reason": "no_measured_camera_height"}
    plane = fit_reference_plane(predicted, intrinsics, mask=region)
    if not reference_plane_is_usable(plane):
        return None, {**diagnostics, "reason": "no_coherent_floor_plane_in_prediction"}
    fitted = camera_height_from_plane(plane.coefficients)
    if fitted is None or fitted <= 0:
        return None, {**diagnostics, "reason": "floor_plane_behind_camera"}
    scale = float(height_m) / float(fitted)
    diagnostics.update({
        "predicted_camera_height_m": round(float(fitted), 5),
        "scale": round(scale, 6),
        "floor_tilt_deg": round(float(plane.tilt_degrees), 2),
        "plane_rmse_m": round(float(plane.residual_rmse_m), 5),
        "plane_inliers": int(plane.inlier_pixels),
    })
    return DepthCalibration(
        scale=scale, offset_m=0.0, rmse_m=float(plane.residual_rmse_m) * scale,
        sample_pixels=int(plane.inlier_pixels), method=PLANE_SCALE_METHOD,
        calibrated_at=time.time(), reference_distance_m=float(height_m), sample_count=1,
        resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
    ), diagnostics


def floor_tilt_deg(predicted: np.ndarray, region: np.ndarray | None,
                   intrinsics: CameraIntrinsics) -> float | None:
    plane = fit_reference_plane(predicted, intrinsics, mask=region)
    return float(plane.tilt_degrees) if reference_plane_is_usable(plane) else None


@dataclass
class PoseFingerprint:
    normal: tuple[float, float, float]
    camera_height_m: float
    resolution: tuple[int, int]
    recorded_at: float

    @classmethod
    def from_plane(cls, plane: Any, resolution: tuple[int, int]) -> "PoseFingerprint | None":
        if plane is None or getattr(plane, "coefficients", None) is None:
            return None
        height = camera_height_from_plane(plane.coefficients)
        if height is None:
            return None
        a, b, _ = (float(value) for value in plane.coefficients)
        norm = math.sqrt(a * a + b * b + 1.0)
        return cls((a / norm, b / norm, -1.0 / norm), float(height),
                   (int(resolution[0]), int(resolution[1])), time.time())


def pose_difference(recorded: PoseFingerprint, current: PoseFingerprint) -> str | None:
    """Why `current` is not the pose `recorded` describes, or None."""
    if tuple(recorded.resolution) != tuple(current.resolution):
        return "resolution_changed"
    cosine = float(np.clip(np.dot(recorded.normal, current.normal), -1.0, 1.0))
    if math.degrees(math.acos(cosine)) > POSE_TILT_TOLERANCE_DEG:
        return "camera_tilt_changed"
    if abs(current.camera_height_m - recorded.camera_height_m) > POSE_HEIGHT_TOLERANCE * recorded.camera_height_m:
        return "camera_height_changed"
    return None


class PoseGuard:
    """The pose the setup was made in, on disk, and whether the camera is still in it."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.recorded: PoseFingerprint | None = None
        self.pending = False
        self.changed_reason: str | None = None
        self._disagreements = 0
        self.last_difference: str | None = None
        if path is not None and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.recorded = PoseFingerprint(
                    tuple(payload["normal"]), float(payload["camera_height_m"]),
                    tuple(payload["resolution"]), float(payload.get("recorded_at", 0.0)))
            except (OSError, KeyError, TypeError, ValueError):
                self.recorded = None

    def expect(self) -> None:
        """The operator just set the camera up: the next floor fit is its pose."""
        self.pending = True

    def record(self, plane: Any, resolution: tuple[int, int]) -> bool:
        fingerprint = PoseFingerprint.from_plane(plane, resolution)
        if fingerprint is None:
            return False
        self.recorded, self.pending = fingerprint, False
        self.changed_reason, self._disagreements = None, 0
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(asdict(fingerprint), indent=2), encoding="utf-8")
            except OSError:  # pragma: no cover - disk issues
                pass
        return True

    def observe(self, plane: Any, resolution: tuple[int, int], *, immediate: bool = False) -> str | None:
        """Compare a fresh floor fit to the recorded pose; returns the standing reason."""
        if self.pending:
            self.record(plane, resolution)
            return None
        if self.recorded is None:
            return None
        current = PoseFingerprint.from_plane(plane, resolution)
        if current is None:
            return self.changed_reason
        difference = pose_difference(self.recorded, current)
        self.last_difference = difference
        if difference is None:
            self._disagreements = 0
            self.changed_reason = None
        else:
            self._disagreements += 1
            if immediate or self._disagreements >= POSE_CHANGE_FRAMES:
                self.changed_reason = f"camera_pose_changed_{difference}"
        return self.changed_reason

    @property
    def state(self) -> str:
        if self.changed_reason:
            return "changed"
        if self.pending:
            return "awaiting_floor_fit"
        return "recorded" if self.recorded is not None else "not_recorded"

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state, "reason": self.changed_reason,
            "recorded": None if self.recorded is None else asdict(self.recorded),
        }


# ------------------------------------------------------------ result sanity flags
THIN_LABEL_WORDS = ("cable", "wire", "cord", "charger", "string", "rope", "lace")
THIN_OBJECT_MAX_HEIGHT_MM = 40.0
BORDER_MARGIN_PX = 2


def touches_frame_border(mask: np.ndarray | None, margin: int = BORDER_MARGIN_PX) -> bool:
    if mask is None or not np.any(mask):
        return False
    return bool(mask[:margin].any() or mask[-margin:].any()
                or mask[:, :margin].any() or mask[:, -margin:].any())


def sanity_flags(
    label: str, *, length_mm: float | None, width_mm: float | None, height_mm: float | None,
    mask: np.ndarray | None = None, scale_anchored: bool = True,
    pose_reason: str | None = None, local_floor_offset_m: float | None = None,
) -> tuple[str, ...]:
    """Reasons a result deserves suspicion. Flags only: nothing is clamped."""
    flags: list[str] = []
    if not scale_anchored:
        flags.append("metric_scale_unanchored")
    if pose_reason:
        flags.append(pose_reason)
    if touches_frame_border(mask):
        flags.append("object_cropped_by_frame_border")
    words = (label or "").lower()
    if height_mm is not None and any(word in words for word in THIN_LABEL_WORDS) \
            and height_mm > THIN_OBJECT_MAX_HEIGHT_MM:
        flags.append("thin_object_height_implausible")
    footprint = [value for value in (length_mm, width_mm) if value]
    if height_mm and footprint and height_mm > 3.0 * max(footprint):
        flags.append("height_exceeds_three_times_footprint")
    if local_floor_offset_m is not None and abs(local_floor_offset_m) >= 0.015:
        # Standing on earlier waste, or a floor fit that is off: either way the
        # support surface is not the floor, and the height is relative to it.
        flags.append("support_surface_not_floor_uncertain")
    return tuple(flags)


def result_record(
    *, diagnostics: dict[str, Any], depth: np.ndarray | None, mask: np.ndarray | None,
    depth_type: str, pose: dict[str, Any], calibration_valid: bool,
    raw_volume_l: float | None, calibrated_volume_l: float | None,
    dimensions_mm: tuple[float | None, float | None, float | None], rejection: str | None,
) -> dict[str, Any]:
    """One Logitech result, with everything needed to judge it later."""
    raw_range = None
    valid_fraction = None
    if depth is not None and mask is not None and depth.shape == mask.shape and np.any(mask):
        values = depth[mask]
        finite = values[np.isfinite(values) & (values > 0)]
        valid_fraction = round(float(finite.size) / float(values.size), 4)
        if finite.size:
            raw_range = [round(float(finite.min()), 4), round(float(finite.max()), 4)]
    return {
        "pose_state": pose.get("state"), "pose_reason": pose.get("reason"),
        "calibration_valid": bool(calibration_valid),
        "mask_area_px": None if mask is None else int(np.count_nonzero(mask)),
        "support_source": diagnostics.get("height_source") or diagnostics.get("plane_source"),
        "depth_type": depth_type,
        "valid_depth_fraction": valid_fraction,
        "raw_depth_range_m": raw_range,
        "height_distribution_m": {
            key: diagnostics.get(key) for key in ("height_median_m", "height_mad_m", "height_p90_m",
                                                  "height_max_m") if key in diagnostics
        },
        "dimensions_mm": list(dimensions_mm),
        "raw_volume_l": raw_volume_l, "calibrated_volume_l": calibrated_volume_l,
        "rejection_reason": rejection,
    }
