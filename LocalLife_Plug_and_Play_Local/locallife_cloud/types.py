"""Small, explicit data contracts used throughout the measurement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float = 0.0
    ppy: float = 0.0
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.fx) or not np.isfinite(self.fy) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsics fx and fy must be finite positive values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "ppx": float(self.ppx),
            "ppy": float(self.ppy),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CameraIntrinsics | None":
        if not payload:
            return None
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            ppx=float(payload.get("ppx", 0)),
            ppy=float(payload.get("ppy", 0)),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
        )


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray | None = field(default=None, repr=False)
    source: str = "yoloe"
    # Stable application-level class.  The detector's raw free-form label is
    # intentionally retained in ``label`` for diagnostics, while this field
    # is restricted to the three object families the installation accepts.
    accepted_class: str | None = None
    color: str = "unknown"
    # Share of the object's own masked pixels that agree with `color`
    # (playbook section 11.6). A low value means the object genuinely is not
    # one colour, which is why `color` becomes "unknown" rather than a guess.
    color_confidence: float = 0.0
    # Deterministic allowed / mis-sort / unknown verdict (playbook section 12).
    sorting_status: str = "unknown"
    sorting_reason: str = ""
    material: str = "unknown"
    material_confidence: float = 0.0
    track_id: int | None = None
    depth_distance_m: float | None = None
    monocular_distance_m: float | None = None
    height_above_baseline_cm: float | None = None
    realsense_volume_l: float | None = None
    monocular_volume_l: float | None = None
    # Incremental occupied volume across this object's own deposit (playbook
    # sections 5 and 10): total bin occupancy just before the object appeared,
    # the total once it settled, and the difference. Populated at the moment of
    # deposit, and only when both totals were actually measured -- the bin does
    # not have to be emptied between bags for these to mean something.
    volume_before_l: float | None = None
    volume_after_l: float | None = None
    added_volume_l: float | None = None
    # Volume the committed contents lost while this object arrived (a bag
    # settling under a new one). Large values mean the old pile moved, which is
    # rejected rather than counted -- see `heightmap_volume.incremental_deposit`.
    displaced_volume_l: float | None = None
    # Why a measurement is still pending or was refused, as a stable code
    # (collecting_frames, object_moving, unstable_depth, insufficient_depth_coverage,
    # camera_moved_recalibration_required, possible_existing_object_movement,
    # new_deposit_not_isolatable, measurement_timeout).
    volume_rejection_reason: str | None = None
    depth_coverage_percent: float | None = None
    volume_uncertainty_l: float | None = None
    measurement_method: str | None = None
    measurement_quality: str | None = None
    calibration_mode: str | None = None
    tracking_status: str = "tentative"
    # Table-relative cuboid measurement (Revised Dual-Camera Volume
    # Estimation recipe; see volume.py's `estimate_box_volume_cuboid` and
    # pipeline.py's box-family wiring). Populated only for RealSense
    # detections whose label matches the box/carton/parcel family and for
    # which a usable table-plane fit exists -- None for bags and for every
    # Logitech detection, since Logitech never supplies metric geometry.
    box_length_mm: float | None = None
    box_width_mm: float | None = None
    box_height_mm: float | None = None
    box_volume_confidence: float | None = None
    box_volume_flags: tuple[str, ...] = ()
    box_template_id: str | None = None
    box_template_nominal_volume_liters: float | None = None
    # Multi-frame track aggregation diagnostics (see
    # `volume.aggregate_box_measurements`) -- default to "one frame, no
    # spread yet" so a fresh detection with no box measurement at all still
    # serializes sensible values.
    box_frames_considered: int = 1
    box_frames_accepted: int = 1
    box_dimension_std_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # General RealSense-only 3-D dimensions.  Unlike the legacy box fields,
    # these are also populated for filled plastic/paper bags.  For deformable
    # bags length/width describe the visible support-plane footprint, not the
    # flat manufactured bag size.
    footprint_length_mm: float | None = None
    footprint_width_mm: float | None = None
    physical_height_mm: float | None = None
    dimension_confidence: float | None = None
    dimension_flags: tuple[str, ...] = ()
    dimension_method: str | None = None

    @property
    def area_pixels(self) -> int:
        if self.mask is not None:
            return int(np.count_nonzero(self.mask))
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box_xyxy": list(self.box),
            "area_pixels": self.area_pixels,
            "source": self.source,
            "accepted_class": self.accepted_class,
            "color": self.color,
            "color_confidence": round(float(self.color_confidence), 4),
            "sorting_status": self.sorting_status,
            "sorting_reason": self.sorting_reason,
            "material": self.material,
            "material_confidence": round(float(self.material_confidence), 4),
            "track_id": self.track_id,
            "depth_distance_m": self.depth_distance_m,
            "monocular_distance_m": self.monocular_distance_m,
            "height_above_baseline_cm": self.height_above_baseline_cm,
            "realsense_volume_l": self.realsense_volume_l,
            "monocular_volume_l": self.monocular_volume_l,
            "volume_before_l": self.volume_before_l,
            "volume_after_l": self.volume_after_l,
            "added_volume_l": self.added_volume_l,
            "displaced_volume_l": self.displaced_volume_l,
            "volume_rejection_reason": self.volume_rejection_reason,
            "depth_coverage_percent": self.depth_coverage_percent,
            "volume_uncertainty_l": self.volume_uncertainty_l,
            "measurement_method": self.measurement_method,
            "measurement_quality": self.measurement_quality,
            "calibration_mode": self.calibration_mode,
            "tracking_status": self.tracking_status,
            "box_dimensions_mm": None if self.box_length_mm is None else {
                "length": round(float(self.box_length_mm), 2),
                "width": round(float(self.box_width_mm), 2),
                "height": round(float(self.box_height_mm), 2),
            },
            "box_volume_confidence": self.box_volume_confidence,
            "box_volume_flags": list(self.box_volume_flags),
            "box_template_id": self.box_template_id,
            "box_template_nominal_volume_liters": self.box_template_nominal_volume_liters,
            "box_frames_considered": int(self.box_frames_considered),
            "box_frames_accepted": int(self.box_frames_accepted),
            "box_dimension_std_mm": {
                "length": round(float(self.box_dimension_std_mm[0]), 3),
                "width": round(float(self.box_dimension_std_mm[1]), 3),
                "height": round(float(self.box_dimension_std_mm[2]), 3),
            } if self.box_length_mm is not None else None,
            "dimensions_mm": None if self.footprint_length_mm is None else {
                "footprint_length": round(float(self.footprint_length_mm), 2),
                "footprint_width": round(float(self.footprint_width_mm), 2),
                "height": round(float(self.physical_height_mm), 2),
            },
            "dimension_confidence": self.dimension_confidence,
            "dimension_flags": list(self.dimension_flags),
            "dimension_method": self.dimension_method,
        }


@dataclass(slots=True)
class VolumeMeasurement:
    liters: float
    valid_pixels: int
    mean_height_m: float
    max_height_m: float
    projected_area_m2: float
    method: str
    candidate_pixels: int = 0
    filled_pixels: int = 0
    coverage_ratio: float = 1.0
    uncertainty_l: float = 0.0
    raw_liters: float | None = None
    geometry_mode: str = "surface-columns"
    calibration_factor: float = 1.0
    random_uncertainty_l: float = 0.0
    systematic_uncertainty_l: float = 0.0
    baseline_noise_m: float = 0.0
    rejected_pixels: int = 0
    quality: str = "unverified"
    height_p90_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "liters": round(float(self.liters), 6),
            "valid_pixels": int(self.valid_pixels),
            "mean_height_m": round(float(self.mean_height_m), 6),
            "max_height_m": round(float(self.max_height_m), 6),
            "projected_area_m2": round(float(self.projected_area_m2), 8),
            "method": self.method,
            "candidate_pixels": int(self.candidate_pixels),
            "filled_pixels": int(self.filled_pixels),
            "coverage_ratio": round(float(self.coverage_ratio), 6),
            "uncertainty_l": round(float(self.uncertainty_l), 6),
            "raw_liters": None if self.raw_liters is None else round(float(self.raw_liters), 6),
            "geometry_mode": self.geometry_mode,
            "calibration_factor": round(float(self.calibration_factor), 8),
            "random_uncertainty_l": round(float(self.random_uncertainty_l), 6),
            "systematic_uncertainty_l": round(float(self.systematic_uncertainty_l), 6),
            "baseline_noise_m": round(float(self.baseline_noise_m), 6),
            "rejected_pixels": int(self.rejected_pixels),
            "quality": self.quality,
            "height_p90_m": round(float(self.height_p90_m), 6),
        }


@dataclass(slots=True)
class ObjectDimensions:
    """RealSense-only visible 3-D footprint and robust height.

    The footprint is measured in the fitted support-plane coordinate system.
    It is suitable for rigid cardboard containers and as an explicitly
    approximate *filled footprint* for deformable plastic/paper bags.
    """

    length_mm: float
    width_mm: float
    height_mm: float
    confidence: float
    depth_valid_ratio: float
    object_points: int
    mask_clipped: bool
    flags: tuple[str, ...] = ()
    method: str = "realsense_support_plane_footprint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions_mm": {
                "footprint_length": round(float(self.length_mm), 2),
                "footprint_width": round(float(self.width_mm), 2),
                "height": round(float(self.height_mm), 2),
            },
            "confidence": round(float(self.confidence), 4),
            "depth_valid_ratio": round(float(self.depth_valid_ratio), 4),
            "object_points": int(self.object_points),
            "mask_clipped": bool(self.mask_clipped),
            "flags": list(self.flags),
            "method": self.method,
        }


@dataclass(slots=True)
class BoxVolumeMeasurement:
    """Table-relative L*W*H box measurement (Revised Dual-Camera Volume
    Estimation recipe, sections 2/4.3/9). Distinct from `VolumeMeasurement`
    (a per-pixel height*area integral) because a rigid box's volume is
    measured from three robust scalar dimensions -- height above the table
    plane, and footprint length/width -- rather than summed pixel-by-pixel.
    Field names and shape intentionally mirror the PDF's own section 9 JSON
    diagnostics contract so a future FastAPI endpoint can emit it directly.
    """

    volume_liters: float
    volume_confidence: float
    volume_method: str
    length_mm: float
    width_mm: float
    height_mm: float
    depth_valid_ratio: float
    object_points: int
    table_plane_inliers: int
    table_plane_rmse_mm: float
    height_p98_mm: float
    height_top_median_mm: float
    mask_clipped: bool
    flags: tuple[str, ...] = ()
    template_id: str | None = None
    template_nominal_volume_liters: float | None = None
    # Multi-frame aggregation diagnostics (Revised Dual-Camera Volume
    # Estimation recipe, section 13: "estimate L/W/H per accepted frame,
    # aggregate dimensions using median, and calculate final volume once").
    # A single-frame `estimate_box_volume_cuboid()` result defaults these to
    # "one frame, no spread yet" so it stays a valid, self-describing result
    # on its own; `volume.aggregate_box_measurements()` fills in the real
    # numbers once several accepted frames exist for a track.
    frames_considered: int = 1
    frames_accepted: int = 1
    dimension_std_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_used_for_final_volume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_liters": round(float(self.volume_liters), 6),
            "volume_confidence": round(float(self.volume_confidence), 4),
            "volume_method": self.volume_method,
            "dimensions_mm": {
                "length": round(float(self.length_mm), 2),
                "width": round(float(self.width_mm), 2),
                "height": round(float(self.height_mm), 2),
            },
            "diagnostics": {
                "depth_valid_ratio": round(float(self.depth_valid_ratio), 4),
                "object_points": int(self.object_points),
                "table_plane_inliers": int(self.table_plane_inliers),
                "table_plane_rmse_mm": round(float(self.table_plane_rmse_mm), 3),
                "height_p98_mm": round(float(self.height_p98_mm), 2),
                "height_top_median_mm": round(float(self.height_top_median_mm), 2),
                "mask_clipped": bool(self.mask_clipped),
                "frames_considered": int(self.frames_considered),
                "frames_accepted": int(self.frames_accepted),
                "dimension_std_mm": {
                    "length": round(float(self.dimension_std_mm[0]), 3),
                    "width": round(float(self.dimension_std_mm[1]), 3),
                    "height": round(float(self.dimension_std_mm[2]), 3),
                },
                "mesh_used_for_final_volume": bool(self.mesh_used_for_final_volume),
            },
            "flags": list(self.flags),
            "template_id": self.template_id,
            "template_nominal_volume_liters": self.template_nominal_volume_liters,
        }


@dataclass(slots=True)
class DepthCalibration:
    scale: float
    offset_m: float
    rmse_m: float
    sample_pixels: int

    def apply(self, prediction_m: np.ndarray) -> np.ndarray:
        return prediction_m.astype(np.float32) * self.scale + self.offset_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": round(float(self.scale), 8),
            "offset_m": round(float(self.offset_m), 8),
            "rmse_m": round(float(self.rmse_m), 6),
            "sample_pixels": int(self.sample_pixels),
        }


@dataclass(slots=True)
class FrameAnalysis:
    timestamp: float
    source: str
    detections: list[Detection]
    frame_width: int
    frame_height: int
    automatic_count: int
    realsense_total: VolumeMeasurement | None = None
    monocular_total: VolumeMeasurement | None = None
    calibration: DepthCalibration | None = None
    inference_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    bin_total: VolumeMeasurement | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "detections": [detection.to_dict() for detection in self.detections],
            "visible_objects": len(self.detections),
            "automatic_count": self.automatic_count,
            "realsense_volume_l": None if self.realsense_total is None else self.realsense_total.liters,
            "monocular_volume_l": None if self.monocular_total is None else self.monocular_total.liters,
            "realsense_measurement": None if self.realsense_total is None else self.realsense_total.to_dict(),
            "monocular_measurement": None if self.monocular_total is None else self.monocular_total.to_dict(),
            "bin_total_volume_l": None if self.bin_total is None else self.bin_total.liters,
            "bin_total_measurement": None if self.bin_total is None else self.bin_total.to_dict(),
            "depth_calibration": None if self.calibration is None else self.calibration.to_dict(),
            "inference_ms": round(float(self.inference_ms), 2),
            "warnings": self.warnings,
        }
