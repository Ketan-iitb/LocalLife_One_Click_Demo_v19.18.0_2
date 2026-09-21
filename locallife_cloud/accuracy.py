"""Accuracy-oriented utilities for the fixed overhead dumpster installation.

The functions in this module are deliberately model-agnostic. They improve the
measurement signal after inference without forcing the research pipelines to
share geometry or ground-truth information.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .types import DepthCalibration


def temporal_median_depth(frames: Iterable[np.ndarray], newest: np.ndarray | None = None) -> np.ndarray | None:
    """Return a per-pixel median while preserving the newest valid sample.

    Invalid depth is represented as NaN during the reduction. This works well
    for RealSense speckle and also damps frame-to-frame monocular flicker.
    """
    items = [np.asarray(item, dtype=np.float32) for item in frames if item is not None]
    if newest is not None:
        newest = np.asarray(newest, dtype=np.float32)
        if not items or items[-1] is not newest:
            items.append(newest)
    if not items:
        return None
    shape = items[-1].shape
    items = [item for item in items if item.shape == shape]
    if not items:
        return newest.copy() if newest is not None else None
    stack = np.stack([np.where(np.isfinite(item) & (item > 0.10), item, np.nan) for item in items])
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        result = np.nanmedian(stack, axis=0)
    fallback = items[-1]
    return np.where(np.isfinite(result), result, fallback).astype(np.float32)


def refine_measurement_mask(
    mask: np.ndarray | None,
    current_depth: np.ndarray | None,
    reference_depth: np.ndarray | None,
    *,
    min_height_m: float,
    max_height_m: float,
    noise_map: np.ndarray | None = None,
    depth_noise_m: float = 0.004,
    noise_sigma: float = 3.0,
    erode_pixels: int = 1,
) -> np.ndarray | None:
    """Combine the semantic instance with physically plausible depth change.

    The semantic mask remains the outer authority; depth is only used to reject
    background/edge pixels. If depth gating would destroy most of the semantic
    object, the conservative eroded semantic mask is returned instead.
    """
    if mask is None:
        return None
    result = mask.astype(bool).copy()
    original_area = int(np.count_nonzero(result))
    if original_area < 20:
        return result

    try:
        import cv2
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        if erode_pixels > 0:
            cleaned = cv2.erode(cleaned, kernel, iterations=int(erode_pixels))
        semantic = cleaned.astype(bool)
    except ImportError:
        semantic = result

    if int(np.count_nonzero(semantic)) < max(20, int(original_area * 0.35)):
        semantic = result

    if current_depth is None or reference_depth is None or current_depth.shape != semantic.shape or reference_depth.shape != semantic.shape:
        return semantic
    height = reference_depth.astype(np.float32) - current_depth.astype(np.float32)
    threshold: np.ndarray | float = float(min_height_m)
    if noise_sigma > 0 and noise_map is not None and noise_map.shape == semantic.shape:
        threshold = np.maximum(min_height_m, noise_sigma * np.sqrt(depth_noise_m**2 + noise_map.astype(np.float32)**2))
    physical = (
        np.isfinite(current_depth) & np.isfinite(reference_depth)
        & (current_depth > 0.10) & (reference_depth > 0.10)
        & (height >= threshold) & (height <= max_height_m)
    )
    gated = semantic & physical
    # Depth sensors often lose a narrow strip around shiny/crumpled plastic. Do
    # not let that turn a good semantic segmentation into a tiny fragment.
    if int(np.count_nonzero(gated)) >= max(25, int(np.count_nonzero(semantic) * 0.45)):
        return gated
    return semantic


def fit_scalar_depth_calibration(predicted: list[float], actual: list[float]) -> DepthCalibration:
    """Fit a scalar multi-point calibration, quadratic when data supports it."""
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.05) & (y > 0.05)
    x, y = x[valid], y[valid]
    if x.size < 2:
        raise ValueError("At least two valid calibration points are required")
    degree = 2 if x.size >= 3 and float(np.ptp(x)) >= 0.08 else 1
    coeff = np.polyfit(x, y, degree)
    if degree == 2:
        quadratic, scale, offset = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
        mode = "quadratic-multipoint"
    else:
        quadratic, scale, offset = 0.0, float(coeff[0]), float(coeff[1])
        mode = "affine-multipoint"
    fitted = quadratic * x * x + scale * x + offset
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    # Guard against pathological fits that invert metric depth over the working range.
    probe = np.linspace(max(0.1, float(np.min(x))), float(np.max(x)), 20)
    derivative = 2.0 * quadratic * probe + scale
    if np.any(derivative <= 0):
        coeff = np.polyfit(x, y, 1)
        quadratic, scale, offset = 0.0, float(coeff[0]), float(coeff[1])
        fitted = scale * x + offset
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        mode = "affine-multipoint"
    return DepthCalibration(scale=scale, offset_m=offset, rmse_m=rmse,
                            sample_pixels=int(x.size), quadratic=quadratic, mode=mode)


def measurement_reliability(record: dict) -> float:
    """0..1 operational reliability score from persisted measurement evidence."""
    coverage = record.get("depth_coverage_percent")
    coverage_score = 0.65 if coverage is None else min(1.0, max(0.0, float(coverage) / 100.0))
    volume = record.get("volume_l")
    uncertainty = record.get("volume_uncertainty_l")
    if volume in (None, 0) or uncertainty is None:
        uncertainty_score = 0.55
    else:
        relative = abs(float(uncertainty)) / max(abs(float(volume)), 0.25)
        uncertainty_score = max(0.0, 1.0 - min(1.0, relative / 0.30))
    quality = str(record.get("measurement_quality") or "").lower()
    quality_score = {"high": 1.0, "moderate": 0.78, "low": 0.45}.get(quality, 0.60)
    confidence = float(record.get("confidence") or 0.0)
    detection_score = min(1.0, max(0.25, confidence))
    return float(0.38 * coverage_score + 0.32 * uncertainty_score + 0.20 * quality_score + 0.10 * detection_score)


def fuse_pair_volume(left: dict, right: dict) -> dict:
    """Confidence/uncertainty weighted operational estimate for one matched drop."""
    lv, rv = left.get("volume_l"), right.get("volume_l")
    if lv is None and rv is None:
        return {"volume_l": None, "reliability": 0.0, "realsense_weight": 0.0, "logitech_weight": 0.0}
    if rv is None:
        return {"volume_l": float(lv), "reliability": measurement_reliability(left),
                "realsense_weight": 1.0, "logitech_weight": 0.0}
    if lv is None:
        return {"volume_l": float(rv), "reliability": measurement_reliability(right),
                "realsense_weight": 0.0, "logitech_weight": 1.0}
    lq, rq = measurement_reliability(left), measurement_reliability(right)
    lu = max(0.20, float(left.get("volume_uncertainty_l") or max(0.05 * float(lv), 0.5)))
    ru = max(0.35, float(right.get("volume_uncertainty_l") or max(0.10 * float(rv), 0.8)))
    # Dedicated stereo depth gets a mild prior advantage; uncertainty still dominates.
    lw = 1.20 * lq / (lu * lu)
    rw = 1.00 * rq / (ru * ru)
    total = lw + rw
    fused = (lw * float(lv) + rw * float(rv)) / total
    reliability = min(1.0, (lq * lw + rq * rw) / total)
    return {"volume_l": round(float(fused), 6), "reliability": round(float(reliability), 4),
            "realsense_weight": round(float(lw / total), 4), "logitech_weight": round(float(rw / total), 4)}


def stable_mode(values: Iterable[str]) -> tuple[str, float]:
    cleaned = [str(v).lower() for v in values if v and str(v).lower() != "unknown"]
    if not cleaned:
        return "unknown", 0.0
    value, count = Counter(cleaned).most_common(1)[0]
    return value, count / len(cleaned)
