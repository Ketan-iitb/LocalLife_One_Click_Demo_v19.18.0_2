"""Metric calibration of Logitech Depth Anything V2 depth.

Depth Anything V2 is monocular: its output is correct in *shape* but not in
absolute scale, so it cannot be reported in millimetres or litres until it is
tied to something physically measured. The camera is fixed above a known
measurement area, which provides that tie:

* one flat reference (the empty bin floor, or a board) at a tape-measured
  distance fixes the scale of the metric model:  Z = a * Z_pred;
* two or more such references at different distances also fix an offset:
  Z = a * Z_pred + b  (least squares over the samples);
* the relative (non-metric) Depth Anything V2 checkpoints predict inverse
  depth, so for them the same fit is made on 1/Z:  1/Z = a * d_pred + b, and
  needs at least two distances.

Samples are *calibration* data. They are stored separately from, and must
never include, the evaluation objects whose accuracy is reported later. A
RealSense reading is not accepted as a sample: the comparison would then be
measuring the Logitech against a Logitech tuned to agree with the RealSense.

Without a valid fit, Logitech depth is relative only and no litres are
reported (see `RELATIVE_ONLY_MESSAGE`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .types import DepthCalibration

RELATIVE_ONLY_MESSAGE = "CALIBRATION REQUIRED — metric volume unavailable"

METRIC_OUTPUT = "metric"
RELATIVE_INVERSE_OUTPUT = "relative_inverse"

METHOD_SINGLE_DISTANCE = "reference-distance-scale"
METHOD_MULTI_DISTANCE = "multi-distance-scale-shift"
METHOD_INVERSE_MULTI_DISTANCE = "multi-distance-inverse-depth-scale-shift"

# Two samples closer than this cannot separate scale from shift.
MIN_DISTANCE_SEPARATION_M = 0.05


def depth_output_kind(model_name: str) -> str:
    """Metric checkpoints are named "...-Metric-..."; the others predict inverse depth."""
    return METRIC_OUTPUT if "metric" in model_name.lower() else RELATIVE_INVERSE_OUTPUT


def robust_prediction(predicted: np.ndarray, region: np.ndarray | None) -> tuple[float, int] | None:
    """Median model output over a flat reference region, and how many pixels it used."""
    values = predicted if region is None else predicted[region]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 100:
        return None
    low, high = np.percentile(values, (5, 95))
    trimmed = values[(values >= low) & (values <= high)]
    return float(np.median(trimmed)), int(trimmed.size)


@dataclass
class CalibrationSample:
    predicted: float
    known_distance_m: float
    pixels: int
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted,
            "known_distance_m": self.known_distance_m,
            "pixels": self.pixels,
            "captured_at": self.captured_at,
        }


def fit_calibration(
    samples: list[CalibrationSample],
    output_kind: str,
    *,
    resolution: tuple[int, int] | None = None,
) -> tuple[DepthCalibration | None, str | None]:
    """Fit the metric mapping, or say why it cannot be fitted."""
    if not samples:
        return None, "no_calibration_samples"
    predicted = np.array([sample.predicted for sample in samples], dtype=np.float64)
    known = np.array([sample.known_distance_m for sample in samples], dtype=np.float64)
    distinct = float(known.max() - known.min()) >= MIN_DISTANCE_SEPARATION_M
    pixels = int(sum(sample.pixels for sample in samples))
    common = dict(
        calibration_id=uuid4().hex[:12],
        calibrated_at=time.time(),
        reference_distance_m=float(np.median(known)),
        sample_count=len(samples),
        resolution=resolution,
    )
    if output_kind == RELATIVE_INVERSE_OUTPUT:
        if not distinct:
            return None, "relative_model_needs_two_reference_distances"
        design = np.column_stack((predicted, np.ones_like(predicted)))
        (scale, shift), *_ = np.linalg.lstsq(design, 1.0 / known, rcond=None)
        if scale <= 0:
            return None, "unstable_monocular_scale"
        fitted = 1.0 / (scale * predicted + shift)
        rmse = float(np.sqrt(np.mean((fitted - known) ** 2)))
        return DepthCalibration(
            scale=float(scale), offset_m=float(shift), rmse_m=rmse, sample_pixels=pixels,
            method=METHOD_INVERSE_MULTI_DISTANCE, inverse=True, **common,
        ), None
    if distinct:
        design = np.column_stack((predicted, np.ones_like(predicted)))
        (scale, shift), *_ = np.linalg.lstsq(design, known, rcond=None)
        method = METHOD_MULTI_DISTANCE
    else:
        scale, shift = float(np.median(known / predicted)), 0.0
        method = METHOD_SINGLE_DISTANCE
    if scale <= 0:
        return None, "unstable_monocular_scale"
    rmse = float(np.sqrt(np.mean((scale * predicted + shift - known) ** 2)))
    return DepthCalibration(
        scale=float(scale), offset_m=float(shift), rmse_m=rmse, sample_pixels=pixels,
        method=method, **common,
    ), None


class LogitechLens:
    """Checkerboard lens profile for the Logitech C920, applied before any geometry.

    Loads a `MonoCalibration` (``logitech_lens.json``) or the Logitech half of
    a `DualCameraCalibration` written by ``scripts/calibrate_dual_camera.py``.
    A profile is only applied at its own resolution, or at another resolution
    with the same aspect ratio after scaling the intrinsics; any other crop is
    left untouched and reported, because distortion coefficients do not
    transfer to a different field of view.
    """

    def __init__(self, *paths: Path) -> None:
        self.calibration = None
        self.source: str | None = None
        self.status = "lens_uncalibrated"
        self._maps: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
        for path in paths:
            if self._load(Path(path)):
                break

    def _load(self, path: Path) -> bool:
        from .calibration import MonoCalibration

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        try:
            if "secondary" in payload:
                if payload.get("primary_camera_id") != "realsense":
                    return False
                payload = payload["secondary"]
            self.calibration = MonoCalibration.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return False
        self.source = str(path)
        return True

    def prepare(self, frame: np.ndarray, intrinsics: Any) -> tuple[np.ndarray, Any]:
        """Undistorted frame and the intrinsics that describe it."""
        import cv2

        from .types import CameraIntrinsics

        if self.calibration is None:
            self.status = "lens_uncalibrated"
            return frame, intrinsics
        height, width = frame.shape[:2]
        profile = self.calibration.intrinsics
        if not profile.width or not profile.height:
            self.status = "lens_profile_without_resolution"
            return frame, intrinsics
        if abs(width / height - profile.width / profile.height) > 1e-3:
            self.status = "lens_profile_resolution_mismatch"
            return frame, intrinsics
        key = (width, height)
        if key not in self._maps:
            factor = width / profile.width
            scaled = CameraIntrinsics(
                fx=profile.fx * factor, fy=profile.fy * factor,
                ppx=profile.ppx * factor, ppy=profile.ppy * factor, width=width, height=height,
            )
            matrix = np.array([[scaled.fx, 0, scaled.ppx], [0, scaled.fy, scaled.ppy], [0, 0, 1]], dtype=np.float64)
            distortion = np.asarray(self.calibration.distortion, dtype=np.float64)
            map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix, key, cv2.CV_16SC2)
            self._maps[key] = (map_x, map_y, scaled)
        map_x, map_y, scaled = self._maps[key]
        self.status = "undistorted"
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR), scaled


class LogitechCalibrationStore:
    """Calibration samples and the fitted mapping, persisted as one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.samples: list[CalibrationSample] = []
        self.calibration: DepthCalibration | None = None
        self.reason: str | None = "no_calibration_samples"
        self.diagnostics: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.samples = [
            CalibrationSample(
                predicted=float(item["predicted"]),
                known_distance_m=float(item["known_distance_m"]),
                pixels=int(item.get("pixels", 0)),
                captured_at=float(item.get("captured_at", 0.0)),
            )
            for item in payload.get("samples", [])
        ]
        self.calibration = DepthCalibration.from_dict(payload.get("calibration"))
        self.reason = payload.get("reason")
        self.diagnostics = payload.get("diagnostics") or {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "purpose": "calibration-only; never evaluation objects",
            "samples": [sample.to_dict() for sample in self.samples],
            "calibration": None if self.calibration is None else self.calibration.to_dict(),
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def add_sample(
        self,
        predicted: np.ndarray,
        region: np.ndarray | None,
        known_distance_m: float,
        output_kind: str,
    ) -> dict[str, Any]:
        if not 0.05 <= known_distance_m <= 10.0:
            raise ValueError("The measured reference distance must be between 0.05 and 10 m")
        robust = robust_prediction(predicted, region)
        if robust is None:
            raise ValueError("The Logitech depth model returned too few valid reference pixels")
        value, pixels = robust
        self.samples.append(CalibrationSample(value, float(known_distance_m), pixels))
        self.calibration, self.reason = fit_calibration(
            self.samples, output_kind, resolution=(int(predicted.shape[1]), int(predicted.shape[0])),
        )
        self._save()
        return self.status()

    def set_calibration(self, calibration: DepthCalibration, diagnostics: dict[str, Any]) -> dict[str, Any]:
        """Store a plane-alignment fit (not a sample list) and persist it."""
        self.samples = []
        self.calibration = calibration
        self.reason = None
        self.diagnostics = diagnostics
        self._save()
        return self.status()

    def clear(self) -> dict[str, Any]:
        self.samples = []
        self.calibration = None
        self.reason = "no_calibration_samples"
        self._save()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "valid": self.calibration is not None,
            "message": None if self.calibration is not None else RELATIVE_ONLY_MESSAGE,
            "reason": self.reason,
            "samples": [sample.to_dict() for sample in self.samples],
            "calibration": None if self.calibration is None else self.calibration.to_dict(),
            "diagnostics": self.diagnostics,
        }
