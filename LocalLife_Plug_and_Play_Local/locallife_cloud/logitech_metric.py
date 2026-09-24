"""Turning the Logitech's monocular signal into centimetres.

Depth Anything V2 answers "which of these is nearer", not "how many
centimetres above the floor". V32 made the mask trustworthy, and the hardware
run then showed what was left: one bag read 3.6 L against 13.4 L, and an
object measured 435 x 162 x 389 mm on the RealSense came out 380 x 355 x 117 mm
here -- a volume that happens to look close can be three dimensions that are
each wrong in compensating directions, so height, length and width are
calibrated and reported separately.

Three things live here:

* the *setup identity* -- resolution, mat, camera-to-floor distance -- because
  a calibration belongs to the installation it was measured in and must be
  refused once that installation changes;
* the *height calibration* -- a mapping from the baseline-relative monocular
  signal to real centimetres, fitted on objects whose height was measured with
  a ruler and chosen by held-out error rather than by the checkpoint's name;
* the *robust runtime statistics* -- what to report from a noisy per-pixel
  height map, and when a sequence of frames has settled enough to be final.

Nothing here reads the RealSense. The two cameras are compared in the thesis;
they are not fitted to each other.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SETUP_FILENAME = "logitech_setup.json"
CALIBRATION_FILENAME = "logitech_height_calibration.json"
SAMPLE_FILENAME = "logitech_height_samples.json"

CALIBRATION_SET = "calibration"
EVALUATION_SET = "evaluation"

LINEAR = "linear"
INVERSE = "inverse"
PIECEWISE = "piecewise"

# Fewer than three known heights cannot be checked against a held-out sample,
# so a fit below this is offered only as provisional.
RECOMMENDED_SAMPLES = 3
# A camera that has been moved by more than this is a different installation.
DISTANCE_TOLERANCE_CM = 2.0


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else round(float(value), digits)


# --------------------------------------------------------------------------- setup


@dataclass(frozen=True)
class CameraSetup:
    """The physical installation a calibration belongs to."""

    camera: str
    width_px: int
    height_px: int
    camera_floor_distance_cm: float
    zone_signature: str = ""
    setup_id: str = ""
    note: str = ""
    created_at: float = 0.0

    def matches(self, other: "CameraSetup | None") -> bool:
        if other is None:
            return False
        return (
            self.camera == other.camera
            and (self.width_px, self.height_px) == (other.width_px, other.height_px)
            and self.zone_signature == other.zone_signature
            and (self.setup_id or "") == (other.setup_id or "")
            and abs(self.camera_floor_distance_cm - other.camera_floor_distance_cm)
            <= DISTANCE_TOLERANCE_CM
        )

    def difference(self, other: "CameraSetup | None") -> str:
        """Why a stored calibration no longer belongs to this installation."""
        if other is None:
            return "no_saved_camera_setup"
        if (self.width_px, self.height_px) != (other.width_px, other.height_px):
            return "resolution_changed"
        if self.zone_signature != other.zone_signature:
            return "measurement_zone_changed"
        if (self.setup_id or "") != (other.setup_id or ""):
            return "camera_setup_id_changed"
        if abs(self.camera_floor_distance_cm - other.camera_floor_distance_cm) > DISTANCE_TOLERANCE_CM:
            return "camera_floor_distance_changed"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera, "width_px": self.width_px, "height_px": self.height_px,
            "camera_floor_distance_cm": _round(self.camera_floor_distance_cm, 2),
            "zone_signature": self.zone_signature, "setup_id": self.setup_id,
            "note": self.note, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraSetup":
        return cls(
            camera=str(payload["camera"]), width_px=int(payload["width_px"]),
            height_px=int(payload["height_px"]),
            camera_floor_distance_cm=float(payload.get("camera_floor_distance_cm", 0.0) or 0.0),
            zone_signature=str(payload.get("zone_signature", "")),
            setup_id=str(payload.get("setup_id", "")), note=str(payload.get("note", "")),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
        )


def zone_signature(zone: Any) -> str:
    """A short fingerprint of the mat a calibration was measured on."""
    if zone is None:
        return ""
    corners = ";".join(f"{x:.1f},{y:.1f}" for x, y in zone.corners)
    payload = f"{zone.camera}|{corners}|{zone.near_edge_m:.4f}|{zone.depth_edge_m:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------- calibration samples


@dataclass(frozen=True)
class HeightSample:
    """One object whose real size was measured with a ruler."""

    name: str
    true_length_cm: float
    true_width_cm: float
    true_height_cm: float
    signal_cm: float = 0.0
    signal_mean_cm: float = 0.0
    measured_length_cm: float = 0.0
    measured_width_cm: float = 0.0
    measured_height_cm: float = 0.0
    true_volume_l: float | None = None
    mask_pixels: int = 0
    setup_id: str = ""
    kind: str = CALIBRATION_SET
    captured_at: float = 0.0
    artefacts: dict[str, str] | None = None

    @property
    def usable(self) -> bool:
        return self.true_height_cm > 0 and self.signal_cm > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "true_length_cm": _round(self.true_length_cm, 2),
            "true_width_cm": _round(self.true_width_cm, 2),
            "true_height_cm": _round(self.true_height_cm, 2),
            "true_volume_l": _round(self.true_volume_l, 4),
            "signal_cm": _round(self.signal_cm, 4), "signal_mean_cm": _round(self.signal_mean_cm, 4),
            "measured_length_cm": _round(self.measured_length_cm, 2),
            "measured_width_cm": _round(self.measured_width_cm, 2),
            "measured_height_cm": _round(self.measured_height_cm, 2),
            "mask_pixels": int(self.mask_pixels), "setup_id": self.setup_id,
            "kind": self.kind, "captured_at": self.captured_at,
            "artefacts": dict(self.artefacts or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HeightSample":
        return cls(
            name=str(payload.get("name", "")),
            true_length_cm=float(payload.get("true_length_cm", 0) or 0),
            true_width_cm=float(payload.get("true_width_cm", 0) or 0),
            true_height_cm=float(payload.get("true_height_cm", 0) or 0),
            signal_cm=float(payload.get("signal_cm", 0) or 0),
            signal_mean_cm=float(payload.get("signal_mean_cm", 0) or 0),
            measured_length_cm=float(payload.get("measured_length_cm", 0) or 0),
            measured_width_cm=float(payload.get("measured_width_cm", 0) or 0),
            measured_height_cm=float(payload.get("measured_height_cm", 0) or 0),
            true_volume_l=(None if payload.get("true_volume_l") in (None, "")
                           else float(payload["true_volume_l"])),
            mask_pixels=int(payload.get("mask_pixels", 0) or 0),
            setup_id=str(payload.get("setup_id", "")),
            kind=str(payload.get("kind", CALIBRATION_SET)),
            captured_at=float(payload.get("captured_at", 0.0) or 0.0),
            artefacts=dict(payload.get("artefacts") or {}),
        )


# ------------------------------------------------------------------- the mapping


@dataclass(frozen=True)
class HeightCalibration:
    """Baseline-relative monocular signal to centimetres."""

    mapping: str
    coefficients: tuple[float, ...]
    sample_count: int = 0
    median_abs_error_cm: float = 0.0
    max_error_cm: float = 0.0
    min_height_cm: float = 0.0
    max_height_cm: float = 0.0
    setup_id: str = ""
    frozen: bool = False
    created_at: float = 0.0
    selection: str = ""
    # The installation as it stood when this mapping was fitted, so a later
    # change can be named rather than merely noticed.
    setup_snapshot: dict[str, Any] | None = None

    def apply(self, signal_cm: np.ndarray | float) -> np.ndarray | float:
        """Centimetres of real height for this signal."""
        values = np.asarray(signal_cm, dtype=np.float64)
        if self.mapping == LINEAR:
            scale, offset = self.coefficients[0], self.coefficients[1]
            heights = scale * values + offset
        elif self.mapping == INVERSE:
            # Fitted as 1/h = a/s + b, which inverts to h = s / (a + b*s):
            # the form the signal takes when it comes from inverse depth.
            a, b = self.coefficients[0], self.coefficients[1]
            denominator = a + b * values
            heights = np.divide(values, denominator, out=np.zeros_like(values),
                                where=np.abs(denominator) > 1e-9)
        elif self.mapping == PIECEWISE:
            knots = np.asarray(self.coefficients, dtype=np.float64).reshape(-1, 2)
            heights = _piecewise(values, knots)
        else:
            heights = values
        heights = np.where(np.isfinite(heights), heights, 0.0)
        return float(heights) if np.isscalar(signal_cm) or heights.ndim == 0 else heights

    @property
    def status(self) -> str:
        if self.frozen:
            return "frozen"
        return "provisional" if self.sample_count else "missing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping, "coefficients": [float(value) for value in self.coefficients],
            "sample_count": self.sample_count,
            "median_abs_error_cm": _round(self.median_abs_error_cm, 3),
            "max_error_cm": _round(self.max_error_cm, 3),
            "calibration_range_cm": [_round(self.min_height_cm, 2), _round(self.max_height_cm, 2)],
            "setup_id": self.setup_id, "frozen": self.frozen, "status": self.status,
            "created_at": self.created_at, "selection": self.selection,
            "setup_snapshot": self.setup_snapshot,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HeightCalibration":
        span = payload.get("calibration_range_cm") or [0.0, 0.0]
        return cls(
            mapping=str(payload["mapping"]),
            coefficients=tuple(float(value) for value in payload.get("coefficients", ())),
            sample_count=int(payload.get("sample_count", 0) or 0),
            median_abs_error_cm=float(payload.get("median_abs_error_cm", 0) or 0),
            max_error_cm=float(payload.get("max_error_cm", 0) or 0),
            min_height_cm=float(span[0] or 0), max_height_cm=float(span[1] or 0),
            setup_id=str(payload.get("setup_id", "")), frozen=bool(payload.get("frozen", False)),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            selection=str(payload.get("selection", "")),
            setup_snapshot=payload.get("setup_snapshot") or None,
        )


def _piecewise(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Monotonic interpolation through the knots, extended by the end slopes."""
    xs, ys = knots[:, 0], knots[:, 1]
    heights = np.interp(values, xs, ys)
    if len(xs) >= 2:
        low_slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
        high_slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
        heights = np.where(values < xs[0], ys[0] + low_slope * (values - xs[0]), heights)
        heights = np.where(values > xs[-1], ys[-1] + high_slope * (values - xs[-1]), heights)
    return heights


def _fit_linear(signals: np.ndarray, heights: np.ndarray) -> HeightCalibration | None:
    if signals.size == 0:
        return None
    if signals.size == 1 or np.ptp(signals) < 1e-6:
        # One distance, or several at the same one: the offset cannot be seen,
        # so only the scale is fitted rather than invented.
        scale = float(np.sum(signals * heights) / max(float(np.sum(signals ** 2)), 1e-9))
        return HeightCalibration(mapping=LINEAR, coefficients=(scale, 0.0))
    matrix = np.column_stack((signals, np.ones_like(signals)))
    solution, *_ = np.linalg.lstsq(matrix, heights, rcond=None)
    return HeightCalibration(mapping=LINEAR, coefficients=(float(solution[0]), float(solution[1])))


def _fit_inverse(signals: np.ndarray, heights: np.ndarray) -> HeightCalibration | None:
    usable = (signals > 1e-6) & (heights > 1e-6)
    if np.count_nonzero(usable) < 2:
        return None
    x, y = 1.0 / signals[usable], 1.0 / heights[usable]
    if np.ptp(x) < 1e-9:
        return None
    matrix = np.column_stack((x, np.ones_like(x)))
    solution, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return HeightCalibration(mapping=INVERSE, coefficients=(float(solution[0]), float(solution[1])))


def _fit_piecewise(signals: np.ndarray, heights: np.ndarray) -> HeightCalibration | None:
    if signals.size < 3:
        return None
    order = np.argsort(signals)
    xs, ys = signals[order], heights[order]
    # Collapse repeated signals and keep the mapping monotonic: a taller
    # object can never map below a shorter one.
    knots: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if knots and abs(x - knots[-1][0]) < 1e-6:
            knots[-1] = (knots[-1][0], (knots[-1][1] + y) / 2.0)
            continue
        if knots and y < knots[-1][1]:
            y = knots[-1][1]
        knots.append((float(x), float(y)))
    if len(knots) < 3:
        return None
    flat = tuple(value for knot in knots for value in knot)
    return HeightCalibration(mapping=PIECEWISE, coefficients=flat)


_FITTERS = ((LINEAR, _fit_linear), (INVERSE, _fit_inverse), (PIECEWISE, _fit_piecewise))


def fit_height_calibration(
    samples: Sequence[HeightSample], *, setup_id: str = "", setup: CameraSetup | None = None,
) -> tuple[HeightCalibration | None, str]:
    """Choose the mapping that predicts a *held-out* object best.

    Each candidate is refitted with one sample left out and scored on that
    sample, so a flexible mapping cannot win by memorising the calibration
    set. With fewer than three samples there is nothing to hold out and the
    fit is reported as provisional, with its in-sample error.
    """
    usable = [sample for sample in samples if sample.usable]
    if not usable:
        return None, "no_usable_calibration_samples"
    signals = np.asarray([sample.signal_cm for sample in usable], dtype=np.float64)
    heights = np.asarray([sample.true_height_cm for sample in usable], dtype=np.float64)

    best: tuple[float, HeightCalibration] | None = None
    scores: dict[str, float] = {}
    for name, fitter in _FITTERS:
        candidate = fitter(signals, heights)
        if candidate is None:
            continue
        errors = _held_out_errors(fitter, signals, heights)
        if errors is None:
            continue
        score = float(np.median(errors))
        scores[name] = round(score, 3)
        # Ties go to the simpler mapping: piecewise interpolation reproduces a
        # straight line exactly, and calling that a piecewise response would
        # claim structure the objects never showed.
        if best is None or score < best[0] - 1e-6:
            best = (score, candidate)
    if best is None:
        return None, "calibration_fit_failed"

    _, chosen = best
    predicted = np.asarray(chosen.apply(signals), dtype=np.float64)
    residuals = np.abs(predicted - heights)
    selection = ", ".join(f"{name}={value} cm" for name, value in sorted(scores.items()))
    calibration = replace(
        chosen,
        sample_count=len(usable),
        median_abs_error_cm=float(np.median(residuals)),
        max_error_cm=float(np.max(residuals)),
        min_height_cm=float(np.min(heights)), max_height_cm=float(np.max(heights)),
        setup_id=setup_id or ("" if setup is None else setup.setup_id), created_at=time.time(),
        setup_snapshot=None if setup is None else setup.to_dict(),
        selection=f"held-out median error: {selection}" if len(usable) >= RECOMMENDED_SAMPLES
        else f"in-sample error: {selection} (add {RECOMMENDED_SAMPLES - len(usable)} more object(s))",
    )
    return calibration, ""


def _held_out_errors(fitter: Any, signals: np.ndarray, heights: np.ndarray) -> np.ndarray | None:
    if signals.size < RECOMMENDED_SAMPLES:
        candidate = fitter(signals, heights)
        if candidate is None:
            return None
        return np.abs(np.asarray(candidate.apply(signals), dtype=np.float64) - heights)
    errors: list[float] = []
    for index in range(signals.size):
        keep = np.ones(signals.size, dtype=bool)
        keep[index] = False
        candidate = fitter(signals[keep], heights[keep])
        if candidate is None:
            return None
        predicted = float(np.asarray(candidate.apply(signals[index : index + 1]))[0])
        errors.append(abs(predicted - float(heights[index])))
    return np.asarray(errors, dtype=np.float64)


# ------------------------------------------------------------- runtime statistics


def robust_height_cm(
    heights_cm: np.ndarray, *, min_cm: float = 0.5, max_cm: float = 200.0,
    spike_mad: float = 6.0,
) -> dict[str, float | int]:
    """What to report from a noisy per-pixel height map.

    A single bright pixel at the mask's edge is depth bleeding from the floor
    behind the object, not the top of it, so the reported maximum is a high
    percentile of the pixels that survive a median-absolute-deviation gate --
    never one extreme value.
    """
    values = np.asarray(heights_cm, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    values = values[(values >= min_cm) & (values <= max_cm)]
    if values.size < 5:
        return {"valid_pixels": int(values.size)}
    median = float(np.median(values))
    # The deviation is floored at half a centimetre: on a flat top every pixel
    # agrees, the measured spread collapses to zero, and without a floor the
    # gate would let a 140 cm edge pixel through untouched.
    deviation = max(float(np.median(np.abs(values - median))), 0.5)
    values = values[np.abs(values - median) <= spike_mad * deviation]
    if values.size < 5:
        return {"valid_pixels": int(values.size)}
    trimmed = values[(values >= np.percentile(values, 10)) & (values <= np.percentile(values, 90))]
    return {
        "top_cm": float(np.percentile(values, 98)),
        "mean_cm": float(np.mean(trimmed)) if trimmed.size else float(np.mean(values)),
        "median_cm": float(np.median(values)),
        "valid_pixels": int(values.size),
        "spike_pixels": int(np.asarray(heights_cm).size - values.size),
    }


def integrate_volume_l(
    heights_cm: np.ndarray, pixel_area_cm2: np.ndarray | None, mask: np.ndarray,
) -> float | None:
    """Per-pixel integration with each pixel's own ground area.

    The primary volume: every pixel contributes the prism that stands on it,
    so a perspective view's far pixels -- which cover more ground -- count for
    more. Analytic solids only ever cross-check this number.
    """
    if pixel_area_cm2 is None or pixel_area_cm2.shape != heights_cm.shape:
        return None
    valid = mask & np.isfinite(heights_cm) & (heights_cm > 0)
    if not np.any(valid):
        return None
    return float(np.sum(heights_cm[valid] * pixel_area_cm2[valid]) / 1000.0)


def stable_statistics(
    samples: Iterable[float], *, min_frames: int = 5, tolerance: float = 0.15,
    spike_mad: float = 4.0,
) -> dict[str, Any]:
    """Has this track settled, and what is its answer?

    The median of the recent frames, with outliers dropped by their distance
    from it, plus the coefficient of variation the dashboard shows as
    confidence. A track that never settles still has an answer -- the median
    of what it has -- which is reported as unstable rather than as nothing.
    """
    values = np.asarray([value for value in samples if value is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    result: dict[str, Any] = {"frames": int(values.size), "stable": False}
    if values.size == 0:
        return result
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    kept = values if deviation <= 1e-9 else values[np.abs(values - median) <= spike_mad * deviation]
    if kept.size == 0:
        kept = values
    median = float(np.median(kept))
    spread = float(np.std(kept))
    result.update({
        "median": median,
        "mad_cm": round(deviation, 4),
        "coefficient_of_variation": round(spread / abs(median), 4) if abs(median) > 1e-9 else None,
        "outliers": int(values.size - kept.size),
        "stable": bool(kept.size >= min_frames
                       and abs(median) > 1e-9 and (spread / abs(median)) <= tolerance),
    })
    return result


# ------------------------------------------------------------------------ storage


class LogitechMetricStore:
    """The setup, the ruler-measured samples and the fitted mapping, on disk."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._setup: CameraSetup | None = None
        self._calibration: HeightCalibration | None = None
        self._samples: list[HeightSample] | None = None

    # -- files
    def _read(self, name: str) -> Any:
        try:
            return json.loads((self.directory / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, name: str, payload: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- setup
    @property
    def setup(self) -> CameraSetup | None:
        if self._setup is None:
            payload = self._read(SETUP_FILENAME)
            if payload:
                try:
                    self._setup = CameraSetup.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    self._setup = None
        return self._setup

    def save_setup(self, setup: CameraSetup) -> CameraSetup:
        stored = replace(setup, created_at=setup.created_at or time.time())
        self._setup = stored
        self._write(SETUP_FILENAME, stored.to_dict())
        return stored

    # -- samples
    def samples(self, kind: str = CALIBRATION_SET) -> list[HeightSample]:
        if self._samples is None:
            payload = self._read(SAMPLE_FILENAME) or []
            self._samples = []
            for record in payload:
                try:
                    self._samples.append(HeightSample.from_dict(record))
                except (KeyError, TypeError, ValueError):
                    continue
        return [sample for sample in self._samples if sample.kind == kind]

    def add_sample(self, sample: HeightSample) -> HeightSample:
        self.samples()  # load
        stored = replace(sample, captured_at=sample.captured_at or time.time())
        assert self._samples is not None
        self._samples.append(stored)
        self._write(SAMPLE_FILENAME, [item.to_dict() for item in self._samples])
        return stored

    def clear_samples(self, kind: str | None = None) -> None:
        self.samples()
        assert self._samples is not None
        self._samples = [] if kind is None else [item for item in self._samples if item.kind != kind]
        self._write(SAMPLE_FILENAME, [item.to_dict() for item in self._samples])

    # -- calibration
    @property
    def calibration(self) -> HeightCalibration | None:
        if self._calibration is None:
            payload = self._read(CALIBRATION_FILENAME)
            if payload:
                try:
                    self._calibration = HeightCalibration.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    self._calibration = None
        return self._calibration

    def save_calibration(self, calibration: HeightCalibration | None) -> HeightCalibration | None:
        self._calibration = calibration
        self._write(CALIBRATION_FILENAME, None if calibration is None else calibration.to_dict())
        return calibration

    def freeze(self) -> HeightCalibration | None:
        current = self.calibration
        if current is None:
            return None
        return self.save_calibration(replace(current, frozen=True))

    def status(self, *, required: int = RECOMMENDED_SAMPLES) -> dict[str, Any]:
        calibration = self.calibration
        samples = self.samples(CALIBRATION_SET)
        return {
            "setup": None if self.setup is None else self.setup.to_dict(),
            "calibration": None if calibration is None else calibration.to_dict(),
            "calibration_status": "missing" if calibration is None else calibration.status,
            "samples": len(samples),
            "samples_required": required,
            "evaluation_samples": len(self.samples(EVALUATION_SET)),
            "sample_names": [sample.name for sample in samples],
        }
