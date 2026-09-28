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


def fingerprint_coefficients(pose: PoseFingerprint) -> tuple[float, float, float]:
    """The recorded floor as z = a*x + b*y + c in camera coordinates."""
    nx, ny, nz = pose.normal
    a, b = -nx / nz, -ny / nz
    return a, b, pose.camera_height_m * math.sqrt(a * a + b * b + 1.0)


def expected_floor_depth(
    shape: tuple[int, int], intrinsics: CameraIntrinsics, coefficients: tuple[float, float, float],
) -> np.ndarray:
    """Axial depth at which each pixel's ray meets the recorded floor."""
    a, b, c = coefficients
    rows, columns = np.indices(shape, dtype=np.float64)
    x = (columns - intrinsics.ppx) / intrinsics.fx
    y = (rows - intrinsics.ppy) / intrinsics.fy
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = c / (1.0 - a * x - b * y)
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)


# ------------------------------------------------------ per-frame floor scale
FLOOR_MIN_PIXELS = 400
FLOOR_MIN_FRACTION = 0.05
FLOOR_MAX_NEAR_FAR_DISAGREEMENT = 0.06
FLOOR_MAX_RELATIVE_RESIDUAL = 0.05
FLOOR_SCALE_MAX_AGE_FRAMES = 150


class FloorScaleTracker:
    """Re-anchor each frame's monocular depth on the floor it can still see.

    Depth Anything's scale is not fixed: it changes with what is in the view,
    so a bag entering the bin can rescale the whole frame, and a scale fixed
    once on the empty scene then measures every object in the wrong units.
    Each frame, the floor pixels no object covers are compared with where the
    recorded floor says they must be; the ratio is the frame's scale. The
    near and the far floor must agree, because a single scale cannot fix a
    prediction that is wrong in shape rather than in size -- such a frame is
    refused, not averaged. Too little floor: the last validated scale is used
    for a while, at reduced confidence, and then nothing is.
    """

    def __init__(self, max_age_frames: int = FLOOR_SCALE_MAX_AGE_FRAMES) -> None:
        self.max_age_frames = max_age_frames
        self.frame = 0
        self.last_valid: tuple[float, int] | None = None
        self.last: dict[str, Any] = {"state": "not_run"}

    def update(
        self, depth: np.ndarray, intrinsics: CameraIntrinsics, floor_mask: np.ndarray,
        pose: PoseFingerprint,
    ) -> tuple[float | None, dict[str, Any]]:
        self.frame += 1
        expected = expected_floor_depth(depth.shape[:2], intrinsics, fingerprint_coefficients(pose))
        usable = floor_mask & np.isfinite(depth) & (depth > 0) & np.isfinite(expected)
        count = int(np.count_nonzero(usable))
        needed = max(FLOOR_MIN_PIXELS, int(FLOOR_MIN_FRACTION * depth.shape[0] * depth.shape[1]))
        record: dict[str, Any] = {"visible_floor_pixels": count, "required_floor_pixels": needed}
        reason = None
        scale = None
        if count < needed:
            reason = "too_little_visible_floor"
        else:
            ratio = expected[usable] / depth[usable]
            scale = float(np.median(ratio))
            residual = float(np.median(np.abs(ratio - scale))) * 1.4826 / scale
            # The nearest and the farthest quarter of the visible floor.
            low, high = np.percentile(expected[usable], (25, 75))
            near = ratio[expected[usable] <= low]
            far = ratio[expected[usable] >= high]
            near_scale, far_scale = float(np.median(near)), float(np.median(far))
            disagreement = abs(near_scale - far_scale) / scale
            record.update({
                "scale": round(scale, 5), "relative_residual": round(residual, 5),
                "near_scale": round(near_scale, 5), "far_scale": round(far_scale, 5),
                "near_far_disagreement": round(disagreement, 5),
                "floor_depth_predicted_p10_p50_p90_m": [
                    round(float(v), 4) for v in np.percentile(depth[usable], (10, 50, 90))],
                "floor_depth_expected_p10_p50_p90_m": [
                    round(float(v), 4) for v in np.percentile(expected[usable], (10, 50, 90))],
            })
            if disagreement > FLOOR_MAX_NEAR_FAR_DISAGREEMENT:
                reason = "floor_scale_differs_near_to_far"
            elif residual > FLOOR_MAX_RELATIVE_RESIDUAL:
                reason = "floor_scale_residual_too_high"
        if reason is None:
            self.last_valid = (scale, self.frame)
            record.update({"state": "validated_this_frame", "applied_scale": round(scale, 5)})
        elif self.last_valid is not None and self.frame - self.last_valid[1] <= self.max_age_frames:
            record.update({"state": "reused_previous_scale", "reason": reason,
                           "applied_scale": round(self.last_valid[0], 5),
                           "frames_since_validation": self.frame - self.last_valid[1]})
            scale = self.last_valid[0]
        else:
            record.update({"state": "unavailable", "reason": reason, "applied_scale": None})
            scale = None
        self.last = record
        return scale, record


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
    context: dict[str, Any] | None = None,
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
            object_percentiles = [round(float(v), 4) for v in np.percentile(finite, (10, 50, 90))]
        else:
            object_percentiles = None
    else:
        object_percentiles = None
    return {
        # Camera, checkpoint, calibration, detection, floor scale: whatever
        # the caller knows about this frame (see the pipeline).
        **(context or {}),
        "object_depth_p10_p50_p90_m": object_percentiles,
        "cropped_by_frame_border": touches_frame_border(mask),
        "raw_height_p90_m": diagnostics.get("height_p90_cells_m"),
        "filtered_height_p90_m": diagnostics.get("height_p90_m"),
        "footprint_area_m2": diagnostics.get("footprint_area_m2"),
        "local_floor_offset_m": diagnostics.get("local_floor_offset_m"),
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


# ------------------------------------------------------------ view-change guard
VIEW_DOWNSCALE_WIDTH = 160
VIEW_MIN_CORRELATION = 0.45
VIEW_CHANGE_FRAMES = 3
VIEW_IMMEDIATE_CORRELATION = 0.25
VIEW_MIN_USABLE_FRACTION = 0.25
VIEW_MIN_EDGE_STD = 2.0


def _edge_image(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame_bgr.shape[:2]
    scale = VIEW_DOWNSCALE_WIDTH / float(width)
    small = cv2.resize(frame_bgr, (VIEW_DOWNSCALE_WIDTH, max(8, int(round(height * scale)))),
                       interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


class ViewChangeGuard:
    """Has the Logitech been moved since its empty-scene reference was taken?

    The Logitech's empty-scene image, depth, floor plane and zone all belong to
    the pose they were captured in. Nothing checked that pose afterwards: the
    placement check runs on hardware depth, which the Logitech does not have.
    A camera turned from a front view to a downward view kept measuring against
    the old floor, which is how one object read 168 mm tall in one pose and
    299 mm in the next.

    The check compares the structure of the view -- edge strength, which
    survives a change of lighting -- with the reference image, away from
    anything detected. Deposited objects are excluded; a moved camera moves
    every edge of the room.
    """

    def __init__(self) -> None:
        self._reference_token: Any = None
        self._reference_edges: np.ndarray | None = None
        self._low = 0
        self.reason: str | None = None
        self.last: dict[str, Any] = {"state": "no_reference"}

    def update(self, frame_bgr: np.ndarray, reference_bgr: np.ndarray | None,
               exclude: np.ndarray | None = None) -> str | None:
        import cv2

        if reference_bgr is None or reference_bgr.shape != frame_bgr.shape:
            self.last = {"state": "no_reference" if reference_bgr is None else "reference_resolution_differs"}
            self.reason = None if reference_bgr is None else "camera_view_resolution_changed"
            return self.reason
        if reference_bgr is not self._reference_token:
            # A new empty-scene reference: whatever moved before, this is now the view.
            self._reference_token = reference_bgr
            self._reference_edges = _edge_image(reference_bgr)
            self._low, self.reason = 0, None
        current = _edge_image(frame_bgr)
        usable = np.ones(current.shape, dtype=bool)
        if exclude is not None and exclude.shape == frame_bgr.shape[:2]:
            small = cv2.resize(exclude.astype(np.uint8), current.shape[::-1], interpolation=cv2.INTER_NEAREST)
            usable = ~(cv2.dilate(small, np.ones((5, 5), np.uint8)) > 0)
        if usable.mean() < VIEW_MIN_USABLE_FRACTION:
            self.last = {"state": "too_much_of_the_view_occupied", "reason": self.reason}
            return self.reason
        a = self._reference_edges[usable]
        b = current[usable]
        if a.std() < VIEW_MIN_EDGE_STD or b.std() < VIEW_MIN_EDGE_STD:
            # A featureless view (a blank wall, a covered lens, darkness)
            # cannot show a move; say so rather than guess either way.
            self.last = {"state": "too_little_structure_to_compare", "reason": self.reason}
            return self.reason
        a, b = a - a.mean(), b - b.mean()
        correlation = float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))
        if correlation < VIEW_MIN_CORRELATION:
            self._low += 1
            # A view that shares almost no structure with the reference has
            # moved; waiting frames only lets wrong measurements through.
            if self._low >= VIEW_CHANGE_FRAMES or correlation < VIEW_IMMEDIATE_CORRELATION:
                self.reason = "camera_view_changed_since_empty_scene"
        else:
            self._low, self.reason = 0, None
        self.last = {"state": "moved" if self.reason else "same_view",
                     "edge_correlation": round(correlation, 3), "reason": self.reason}
        return self.reason
