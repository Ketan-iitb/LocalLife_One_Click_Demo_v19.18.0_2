"""Conservative Logitech segmentation, color sampling, and depth stabilization."""

from __future__ import annotations

import colorsys
from typing import Any

import numpy as np

from .geometry import combined_mask, connected_components, dominant_color
from .types import Detection
from .volume import calibrate_monocular_depth


def _seed_mask(detection: Detection, shape: tuple[int, int]) -> np.ndarray:
    return combined_mask([detection], shape)


def _interior(mask: np.ndarray) -> np.ndarray:
    if int(np.count_nonzero(mask)) < 80:
        return mask
    try:
        import cv2

        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(np.pad(mask.astype(np.uint8), 1, mode="constant"), (3, 3))
        eroded = np.all(windows, axis=(-2, -1))
    return eroded if int(np.count_nonzero(eroded)) >= max(25, int(np.count_nonzero(mask)) // 3) else mask


def object_color(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    label: str = "",
    foreground_threshold: int = 18,
) -> str:
    """Sample the object's changed interior; avoid walls, highlights, and edges."""
    if mask.shape != frame.shape[:2] or int(np.count_nonzero(mask)) < 20:
        return "unknown"
    normalized = label.lower()
    is_bag = any(word in normalized.replace("-", " ").split() for word in ("bag", "sack", "tote"))
    candidate = mask.astype(bool) if is_bag else _interior(mask.astype(bool))
    if baseline is not None and baseline.shape == frame.shape:
        changed = np.max(np.abs(frame.astype(np.int16) - baseline.astype(np.int16)), axis=2)
        object_changed = candidate & (changed >= foreground_threshold)
        if int(np.count_nonzero(object_changed)) >= max(20, int(np.count_nonzero(candidate)) // 5):
            candidate = object_changed
    pixels = frame[candidate].astype(np.float32) / 255.0
    if pixels.shape[0] > 20_000:
        pixels = pixels[np.linspace(0, pixels.shape[0] - 1, 20_000, dtype=np.int64)]
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-8)
    chromatic = (saturation >= 0.18) & (maximum >= 0.16)
    # Ignore a minority of achromatic glare if enough real object pigment exists.
    if int(np.count_nonzero(chromatic)) >= max(20, int(pixels.shape[0] * 0.18)):
        pixels = pixels[chromatic]
    blue, green, red = (float(value) for value in np.median(pixels, axis=0))
    hue, sat, value = colorsys.rgb_to_hsv(red, green, blue)
    degrees = hue * 360.0
    # Dark bags often pick up green/blue camera tint while remaining visually black.
    if value < 0.34 and sat < 0.58:
        return "black"
    if any(word in normalized for word in ("box", "cardboard", "carton", "parcel")):
        if 10 <= degrees <= 60 and sat >= 0.18 and value < 0.82:
            return "brown"
    if is_bag:
        return dominant_color(frame, candidate)
    synthetic = np.array([[[blue * 255, green * 255, red * 255]]], dtype=np.float32)
    tiled = np.tile(synthetic, (5, 5, 1)).astype(np.uint8)
    return dominant_color(tiled, np.ones((5, 5), dtype=bool))


def _foreground_change(
    frame: np.ndarray,
    baseline: np.ndarray | None,
    region: np.ndarray,
    detections: list[Detection],
    threshold: int,
) -> np.ndarray | None:
    if baseline is None or baseline.shape != frame.shape:
        return None
    difference = frame.astype(np.int16) - baseline.astype(np.int16)
    excluded = combined_mask(detections, frame.shape[:2])
    anchors = region & ~excluded
    if int(np.count_nonzero(anchors)) >= max(30, int(np.count_nonzero(region)) // 20):
        shift = np.median(difference[anchors], axis=0)
        shift = np.clip(shift, -35, 35)
        difference = difference - shift[None, None, :]
    return (np.max(np.abs(difference), axis=2) >= threshold) & region


def bound_logitech_detections(
    frame: np.ndarray,
    baseline: np.ndarray | None,
    detections: list[Detection],
    region: np.ndarray,
    *,
    min_pixels: int = 25,
    foreground_threshold: int = 18,
    max_scene_fraction: float = 0.45,
    max_expansion: float = 2.0,
    duplicate_overlap: float = 0.55,
) -> tuple[list[Detection], list[str]]:
    """Never allow monocular depth drift to replace a semantic instance mask."""
    region_area = max(1, int(np.count_nonzero(region)))
    changed = _foreground_change(frame, baseline, region, detections, foreground_threshold)
    components = connected_components(changed, min_area=min_pixels) if changed is not None else []
    retained: list[Detection] = []
    warnings: list[str] = []
    for detected in detections:
        seed = _seed_mask(detected, frame.shape[:2]) & region
        seed_area = int(np.count_nonzero(seed))
        if seed_area < min_pixels:
            continue
        core = seed.copy()
        if changed is not None:
            foreground_seed = seed & changed
            foreground_area = int(np.count_nonzero(foreground_seed))
            if foreground_area >= min_pixels and (
                foreground_area >= int(seed_area * 0.20)
                or seed_area / region_area > max_scene_fraction
            ):
                core = foreground_seed
        core_area = int(np.count_nonzero(core))
        if core_area / region_area > max_scene_fraction:
            warnings.append("Rejected a Logitech detection covering most of the scene; tighten its camera ROI")
            continue

        best_component = None
        best_overlap = 0
        for component in components:
            overlap = int(np.count_nonzero(component & core))
            if overlap > best_overlap and overlap >= max(10, int(core_area * 0.12)):
                best_component, best_overlap = component, overlap
        measured = core
        if best_component is not None:
            expanded = best_component | core
            expanded_area = int(np.count_nonzero(expanded))
            if expanded_area <= core_area * max_expansion and expanded_area / region_area <= max_scene_fraction:
                measured = expanded
        pixels = int(np.count_nonzero(measured))
        if pixels < min_pixels:
            continue
        rows, columns = np.nonzero(measured)
        box = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        footprint = (box[2] - box[0]) * (box[3] - box[1])
        if footprint / region_area > max_scene_fraction:
            warnings.append(
                "Rejected a Logitech object mask spanning the wall or floor; "
                "restrict its region to the garbage-bin opening"
            )
            continue
        retained.append(Detection(
            label=detected.label,
            confidence=detected.confidence,
            box=box,
            mask=measured,
            source="yoloe-logitech-object-mask",
            color=object_color(frame, measured, baseline=baseline, label=detected.label,
                               foreground_threshold=foreground_threshold),
        ))
    unique: list[Detection] = []
    # Nested open-vocabulary prompts often label one cardboard box several times.
    # Start with the tighter object footprint so the surrounding wall cannot win.
    for candidate in sorted(retained, key=lambda item: (item.area_pixels, -item.confidence)):
        duplicate = False
        for existing in unique:
            if _object_family(candidate.label) != _object_family(existing.label):
                continue
            if _nested_overlap(candidate, existing) >= duplicate_overlap:
                duplicate = True
                break
        if duplicate:
            warnings.append("Merged overlapping Logitech detections of the same physical object")
        else:
            unique.append(candidate)
    return sorted(unique, key=lambda item: item.area_pixels, reverse=True), warnings


def _object_family(label: str) -> str:
    words = set(label.lower().replace("-", " ").replace("_", " ").split())
    if words & {"bag", "bags", "sack", "sacks", "tote"}:
        return "bag"
    if words & {"box", "boxes", "cardboard", "carton", "parcel"}:
        return "box"
    return "other"


def _nested_overlap(first: Detection, second: Detection) -> float:
    if first.mask is not None and second.mask is not None and first.mask.shape == second.mask.shape:
        smaller = min(int(np.count_nonzero(first.mask)), int(np.count_nonzero(second.mask)))
        if smaller:
            return float(np.count_nonzero(first.mask & second.mask)) / smaller
    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller if smaller else 0.0


def stabilize_background_depth(
    current: np.ndarray | None,
    reference: np.ndarray | None,
    frame: np.ndarray,
    baseline_rgb: np.ndarray | None,
    detections: list[Detection],
    region: np.ndarray,
    *,
    foreground_threshold: int = 18,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Align monocular scale/offset using only unchanged Logitech background."""
    diagnostics: dict[str, Any] = {
        "applied": False, "scale": 1.0, "offset_m": 0.0, "anchor_pixels": 0,
        "background_error_before_m": None, "background_error_after_m": None,
    }
    if current is None or reference is None or current.shape != reference.shape:
        return current, diagnostics
    excluded = combined_mask(detections, current.shape)
    # Expand semantic boxes slightly so object-edge depth leakage never anchors the fit.
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        margin = max(3, int(max(x2 - x1, y2 - y1) * 0.08))
        excluded[max(0, y1 - margin): min(current.shape[0], y2 + margin),
                 max(0, x1 - margin): min(current.shape[1], x2 + margin)] = True
    anchors = region & ~excluded & np.isfinite(current) & np.isfinite(reference)
    anchors &= (current > 0.10) & (reference > 0.10)
    if baseline_rgb is not None and baseline_rgb.shape == frame.shape:
        rgb_change = np.max(np.abs(frame.astype(np.int16) - baseline_rgb.astype(np.int16)), axis=2)
        anchors &= rgb_change <= max(foreground_threshold * 2, 28)
    count = int(np.count_nonzero(anchors))
    diagnostics["anchor_pixels"] = count
    minimum = min(100, max(30, int(np.count_nonzero(region)) // 20))
    if count < minimum:
        return current, diagnostics
    calibration = calibrate_monocular_depth(current, reference, mask=anchors, minimum_samples=minimum)
    if calibration is None or not 0.70 <= calibration.scale <= 1.35 or abs(calibration.offset_m) > 0.30:
        return current, diagnostics
    before = float(np.median(np.abs(reference[anchors] - current[anchors])))
    stabilized = calibration.apply(current)
    after = float(np.median(np.abs(reference[anchors] - stabilized[anchors])))
    diagnostics["background_error_before_m"] = round(before, 6)
    diagnostics["background_error_after_m"] = round(after, 6)
    if before < 0.003 or after > before * 0.85:
        return current, diagnostics
    diagnostics.update(applied=True, scale=round(calibration.scale, 8),
                       offset_m=round(calibration.offset_m, 8))
    return stabilized, diagnostics
