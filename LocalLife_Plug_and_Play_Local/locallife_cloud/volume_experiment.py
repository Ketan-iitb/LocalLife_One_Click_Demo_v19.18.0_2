"""V54 volume experiment: one per-pixel volume definition for both cameras, with baselines,
one calibration procedure, trial recording, replay and evaluation (NumPy/OpenCV only).

Quantity measured (``quantity = "visible_surface_volume_above_empty_bin"``)
    The volume between the object's VISIBLE top surface and the session's empty-bin surface,
    integrated on a grid in the empty-bin support plane:  V_m3 = sum(rise_m * cell_m^2), V_L = 1000 V_m3.
    Space under the visible surface is assumed filled (a box's or bag's underside is not seen);
    cells with no visible sample are NOT filled in -- they lower `coverage` and can make the frame invalid.
    A curved object's far slope hidden behind its own crest is not counted either, and a loss at the
    footprint's edge cannot show in `coverage`: for bags seen obliquely the value is a LOWER BOUND
    (synthetic paraboloid bag: -1 % at 15 deg, -11 % at 30 deg, -26 % at 45 deg from vertical).
    `occluding_boundary_fraction` records how much of the outline hides something behind it.
    It is not the nominal container capacity, not an L x W x H enclosing cuboid and not a change in bin
    occupancy; those are other quantities and are never substituted for this one.

Per-pixel geometry (both cameras, independent depth sources)
    1. depth (metres; Logitech: model output x calibration scale) is deprojected with intrinsics scaled
       to the depth map's own resolution (pixel-centre convention); invalid depth stays invalid.
    2. The support plane is fitted (RANSAC + SVD refit, fixed seed) to the empty-bin baseline inside the
       ROI. Height = signed perpendicular distance to that plane (not a camera-axis depth difference).
    3. Each object pixel i with valid depth contributes rise_i x A_i, where
         rise_i = its height above the plane - the baseline surface height in its support-plane cell
                  (baseline cell = mean of the empty-bin samples; plane height 0 where the baseline has
                  none, counted in `baseline_cells_missing`), and
         A_i    = the area its surface patch covers ON THE SUPPORT PLANE, the exact Jacobian
                  (P_u x P_v) . n with P = z q(u, v):  A = z [z (q_u x q_v) + z_u (q x q_v) + z_v (q_u x q)] . n.
       A is linear in the depth derivatives (central differences; one-sided across depth edges), so
       zero-mean depth noise does not bias it; a side face projects to ~0 area, a tilted top to its true
       footprint, and no projected area is counted twice. A camera-facing pixel area is never used.
    4. Pixels with rise >= MIN_RISE_M are integrated (mask leakage onto the empty floor rises ~0 and
       drops out; its signed residual is kept). Coverage uses square support-plane cells (CELL_M, or
       2.5 x the sample spacing if coarser): footprint-hull cells with object samples / all hull cells.
    5. Validity rule (per frame): enough object pixels with valid depth, coverage >= MIN_COVERAGE,
       footprint >= MIN_CELLS, background (ROI outside the mask) within BACKGROUND_TOL of the baseline.
       An invalid frame has volume_l = None (never 0, never a previous value); `volume_l_partial`
       keeps the integral for diagnosis only.

Calibration (one procedure, camera-specific parameters; see `fit_depth_mapping`)
    The raw-depth -> metres mapping chosen by the model's output semantics (metric checkpoint:
    Z = s raw + t; relative inverse depth: Z = 1/(a d + b)), fitted ONLY to independently measured
    geometry of calibration placements -- top-surface heights at several bin positions and, optionally,
    the perpendicular camera-to-floor distance. No volumes are fitted. Frozen with its residuals and its
    validated raw-depth range; outside that range a measurement is not reported.
    Per-frame background alignment (median depth ratio to the baseline) is OFF by default; when enabled
    it is bounded (0.7-1.4), logged per frame and part of the calibration record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MEASUREMENT_VERSION = "V54.3"   # bump on ANY change that can alter a measurement; evaluate per version
QUANTITY = "visible_surface_volume_above_empty_bin"
CELL_M = 0.005
TOP_BAND_M = 0.015
MIN_RISE_M = 0.010
MIN_COVERAGE = 0.85
MIN_CELLS = 20
MIN_VALID_DEPTH_FRACTION = 0.60
BACKGROUND_TOL_M = 0.010
BASELINE_MAX_PLANE_RMSE_M = 0.015
BASELINE_MIN_VALID_FRACTION = 0.50
BASELINE_MAX_TEMPORAL_STD_M = 0.020
POSE_MAX_ANGLE_DEG = 3.0
POSE_MAX_DISTANCE_REL = 0.05
MIN_VALID_FRAMES_PER_TRIAL = 3
MAX_FRAME_IQR_REL = 0.15          # settled-window frame volumes must agree this well (IQR / median)
CAMERAS = ("realsense", "logitech")
DESIGNATIONS = ("calibration", "test")
CONDITIONS = ("isolated", "dark_bag", "overlapping", "falling", "other")
REFERENCE_STATUSES = ("measured", "pending", "unverified")
REFERENCE_QUANTITIES = ("external_geometric", "displacement", "enclosing_box", "printed_capacity", "other")
EXTERNAL_VOLUME_REFERENCES = ("external_geometric", "displacement")   # accuracy is scored only against these


# --------------------------------------------------------------------------------------- geometry
def intrinsics_dict(intrinsics: Any) -> dict[str, Any] | None:
    if intrinsics is None or isinstance(intrinsics, dict):
        return intrinsics
    return {k: getattr(intrinsics, k) for k in ("fx", "fy", "ppx", "ppy", "width", "height") if hasattr(intrinsics, k)}


def scaled_intrinsics(intrinsics: Any, shape: tuple[int, int], fallback_shape: tuple[int, int] | None = None
                      ) -> dict[str, float]:
    """fx, fy, cx, cy for an image of `shape` (rows, cols); the intrinsics' own width/height (or
    `fallback_shape` when they carry none) is the resolution they were given for."""
    d = intrinsics if isinstance(intrinsics, dict) else intrinsics_dict(intrinsics)
    rows, cols = shape
    base_w = int(d.get("width") or 0) or (fallback_shape[1] if fallback_shape else cols)
    base_h = int(d.get("height") or 0) or (fallback_shape[0] if fallback_shape else rows)
    sx, sy = cols / base_w, rows / base_h
    return {"fx": float(d["fx"]) * sx, "fy": float(d["fy"]) * sy,
            "cx": (float(d.get("ppx", d.get("cx", 0.0))) + 0.5) * sx - 0.5,
            "cy": (float(d.get("ppy", d.get("cy", 0.0))) + 0.5) * sy - 0.5,
            "width": cols, "height": rows}


def deproject(depth: np.ndarray, k: dict[str, float]) -> np.ndarray:
    """(rows, cols, 3) camera-frame metres; NaN where depth is invalid."""
    z = np.where(np.isfinite(depth) & (depth > 0.05), depth, np.nan).astype(np.float64)
    v, u = np.indices(depth.shape)
    return np.dstack([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z])


def fit_plane(points: np.ndarray, seed: int = 0) -> dict[str, Any] | None:
    """Robust plane n.p + d = 0 with d > 0 (camera on the positive side). RANSAC, then SVD refit."""
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 200:
        return None
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    tol = float(np.clip(0.01 * np.median(sample[:, 2]), 0.006, 0.03))
    best, best_count = None, 0
    for _ in range(150):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        count = int(np.count_nonzero(np.abs((sample - a) @ n) < tol))
        if count > best_count:
            best, best_count = (n, a), count
    if best is None:
        return None
    n, a = best
    inliers = points[np.abs((points - a) @ n) < tol]
    centre = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centre, full_matrices=False)
    n = vt[2]
    d = -float(n @ centre)
    if d < 0:
        n, d = -n, -d
    residual = inliers @ n + d
    return {"normal": n, "d": d, "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
            "inlier_fraction": float(len(inliers)) / len(points), "tolerance_m": tol}


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _cells(points: np.ndarray, plane: dict[str, Any], cell: float):
    """(cell keys, signed heights) of points in the plane's own coordinates."""
    n, d = plane["normal"], plane["d"]
    u, v = plane_basis(n)
    ix = np.floor(points @ u / cell).astype(np.int64)
    iy = np.floor(points @ v / cell).astype(np.int64)
    return ix, iy, points @ n + d


def _depth_geometry(depth: np.ndarray, k: dict[str, float]):
    """z, dz/du, dz/dv (central; one-sided on the pixel's own side across a > 2 cm + 2 % depth edge),
    the per-pixel ray q and its derivatives q_u, q_v."""
    z = np.where(np.isfinite(depth) & (depth > 0.05), depth, np.nan).astype(np.float64)

    def derivative(axis: int) -> np.ndarray:
        fwd = np.diff(z, axis=axis, append=np.nan)               # z[i+1] - z[i]
        bwd = np.diff(z, axis=axis, prepend=np.nan)              # z[i] - z[i-1]
        central = 0.5 * (fwd + bwd)
        edge = np.abs(fwd + bwd) > 0.02 + 0.02 * z
        one = np.where(np.abs(fwd) <= np.abs(bwd), fwd, bwd)
        one = np.where(np.isnan(fwd), bwd, np.where(np.isnan(bwd), fwd, one))
        return np.where(edge | np.isnan(central), one, central)

    v, u = np.indices(z.shape)
    q = np.dstack([(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], np.ones(z.shape)])
    return z, derivative(1), derivative(0), q, np.array([1.0 / k["fx"], 0.0, 0.0]), np.array([0.0, 1.0 / k["fy"], 0.0])


def _support_area(depth: np.ndarray, k: dict[str, float], normal: np.ndarray) -> np.ndarray:
    """Per-pixel area (m^2) of the imaged surface patch projected onto the support plane (normal n):
    (P_u x P_v) . n for P = z q(u, v), q = ((u-cx)/fx, (v-cy)/fy, 1). Central differences; across a depth
    edge (> 2 cm + 2 % between the two neighbours) the one-sided difference on the pixel's own side is
    used. NaN where no difference is available. Sign: positive for surfaces facing the camera and up."""
    z, z_u, z_v, q, qu, qv = _depth_geometry(depth, k)
    n = np.asarray(normal, dtype=np.float64)
    # (q x qv).n and (qu x q).n per pixel; (qu x qv).n is constant
    t1 = np.cross(q, qv) @ n
    t2 = np.cross(qu, q) @ n
    t0 = float(np.cross(qu, qv) @ n)
    area = z * (z * t0 + z_u * t1 + z_v * t2)
    return -area if t0 < 0 else area                       # camera on the plane's + side: upward faces > 0


def occluding_boundary_fraction(depth: np.ndarray, obj: np.ndarray) -> float | None:
    """Share of the object's boundary pixels whose outside neighbour lies > 2 cm deeper: there the object
    hides what is behind it. For a curved object (bag) seen obliquely its far slope continues out of
    sight there and is NOT counted (visible-surface volume = lower bound); a box's hidden far side face
    has no footprint, so it loses nothing. Diagnostic only: per-pixel normals at a contour are too noisy
    to tell the two apart reliably (tested), so this does not gate validity."""
    import cv2
    z = np.nan_to_num(np.where(np.isfinite(depth) & (depth > 0.05), depth, np.nan), nan=0.0)
    boundary = obj & ~cv2.erode(obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if boundary.sum() < 10:
        return None
    deeper = np.zeros_like(obj)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        deeper |= ~np.roll(obj, (dy, dx), axis=(0, 1)) & (np.roll(z, (dy, dx), axis=(0, 1)) > z + 0.02)
    return round(float(np.count_nonzero(boundary & deeper)) / int(boundary.sum()), 4)


MASK_COMPLETION_MARGIN = 0.15     # growth stays inside the detection box enlarged by this share per side
MASK_COMPLETION_MAX_GROWTH = 4.0  # a raised region > 4x the mask is a neighbour/pile, not this object
MASK_TRUNCATION_GROWTH = 1.15     # connected raised surface > 15 % beyond the mask = the mask is truncated


def complete_object_mask(rise: np.ndarray, valid: np.ndarray, z: np.ndarray, mask: np.ndarray,
                         box: tuple[int, int, int, int] | None = None, exclude: np.ndarray | None = None,
                         min_rise: float = 0.008):
    """The object's mask completed from GEOMETRY: detector masks often cover only part of an object (a
    carton's printed panel, an eroded outline), and integrating only those pixels under-read a 1.62 L
    box as 0.0-1.2 L depending on placement. Added: pixels raised >= `min_rise` above the support,
    connected to the mask without crossing a depth edge (> 2 cm + 2 %), inside the detection box +15 %,
    not in another detection (`exclude`). Not completed (mask kept, flagged) when the raised region is
    > 4x the mask or runs to the edge of that zone: a pile or neighbour, not this object.
    Returns (completed mask or None, info)."""
    import cv2
    h, w = rise.shape
    mask = mask.astype(bool)
    info = {"mask_pixels": int(mask.sum()), "completed_pixels": int(mask.sum()), "completed": False, "reason": None}
    if not mask.any():
        return None, info
    if box is None:
        rows, cols = np.nonzero(mask)
        box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    mx, my = int((x2 - x1) * MASK_COMPLETION_MARGIN) + 1, int((y2 - y1) * MASK_COMPLETION_MARGIN) + 1
    zx1, zy1, zx2, zy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    zone = np.zeros((h, w), bool)
    zone[zy1:zy2, zx1:zx2] = True
    risen = valid & zone & np.isfinite(rise) & (np.nan_to_num(rise, nan=-1.0) >= min_rise)
    if exclude is not None:
        risen &= ~exclude.astype(bool)
    zz = np.nan_to_num(np.where(valid, z, np.nan), nan=0.0)
    edge = np.zeros((h, w), bool)
    for axis in (0, 1):
        d = np.abs(np.diff(zz, axis=axis))
        big = d > 0.02 + 0.02 * (zz[:-1] if axis == 0 else zz[:, :-1])
        if axis == 0:
            edge[:-1] |= big
            edge[1:] |= big
        else:
            edge[:, :-1] |= big
            edge[:, 1:] |= big
    count, labels = cv2.connectedComponents((risen & ~edge).astype(np.uint8), connectivity=4)
    ids = np.unique(labels[mask & (labels > 0)])
    if not ids.size:
        info["reason"] = "no raised surface connected to the mask"
        return None, info
    grown = np.isin(labels, ids)
    grown |= cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & risen
    completed = mask | grown
    border = np.zeros((h, w), bool)
    border[zy1, zx1:zx2] = border[zy2 - 1, zx1:zx2] = border[zy1:zy2, zx1] = border[zy1:zy2, zx2 - 1] = True
    image_edge = np.zeros((h, w), bool)
    image_edge[0, :] = image_edge[-1, :] = image_edge[:, 0] = image_edge[:, -1] = True
    border &= ~image_edge                                   # image-edge contact is the clipping check's job
    info["completed_pixels"] = int(completed.sum())
    info["growth"] = round(info["completed_pixels"] / max(1, info["mask_pixels"]), 3)
    if info["growth"] > MASK_COMPLETION_MAX_GROWTH:
        info["reason"] = f"raised region {info['growth']:.1f}x the mask: a neighbour or pile, not completed"
        return None, info
    if (grown & border & ~mask).any():
        info["reason"] = "raised surface runs past the detection box: not completed (neighbour or larger object)"
        return None, info
    info["completed"] = info["completed_pixels"] > info["mask_pixels"]
    return completed, info


def top_height(rise: np.ndarray) -> float:
    """Height of the object's top surface: median of the samples within the top band (rise >= 80 % of
    the 95th percentile). For a flat-topped box this is its top face, unaffected by side-face samples."""
    if rise.size == 0:
        return float("nan")
    p95 = float(np.percentile(rise, 95))
    return float(np.median(rise[rise >= 0.8 * p95]))


def apply_depth_mapping(raw: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    """Raw camera depth -> metres. kind 'scale': Z = scale x raw (RealSense: 1). 'metric_affine'
    (metric checkpoint): Z = s x raw + t. 'inverse_affine' (relative inverse-depth output d):
    Z = 1 / (a x d + b). Non-positive results are invalid (NaN)."""
    raw = raw.astype(np.float64)
    kind = mapping.get("kind", "scale")
    if kind == "scale":
        z = raw * float(mapping.get("scale", 1.0))
    elif kind == "metric_affine":
        z = float(mapping["s"]) * raw + float(mapping["t"])
    elif kind == "inverse_affine":
        with np.errstate(divide="ignore", invalid="ignore"):
            z = 1.0 / (float(mapping["a"]) * raw + float(mapping["b"]))
    else:
        raise ValueError(f"unknown depth mapping {kind!r}")
    return np.where(np.isfinite(raw) & (raw > 0) & np.isfinite(z) & (z > 0.05), z, np.nan)


def undistort_inputs(k: dict[str, float], distortion: list[float], depth, baseline, mask, roi):
    """Remap depth, baseline, mask and ROI from the distorted image to the pinhole model with the same
    camera matrix (nearest neighbour: depth values are never blended across an edge)."""
    import cv2
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float64)
    h, w = depth.shape
    mx, my = cv2.initUndistortRectifyMap(K, np.asarray(distortion, np.float64), None, K, (w, h), cv2.CV_32FC1)

    def remap(img, fill):
        return cv2.remap(img, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    return (remap(depth.astype(np.float32), float("nan")), remap(baseline.astype(np.float32), float("nan")),
            remap(mask.astype(np.uint8), 0).astype(bool), remap(roi.astype(np.uint8), 0).astype(bool))


def _per_cell_mean(keys: np.ndarray, values: np.ndarray):
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inv, values) / np.bincount(inv)


def _per_cell_top(keys: np.ndarray, heights: np.ndarray):
    uniq, inv = np.unique(keys, return_inverse=True)
    top = np.full(len(uniq), -np.inf)
    np.maximum.at(top, inv, heights)
    band = heights >= top[inv] - TOP_BAND_M
    return uniq, (np.bincount(inv[band], heights[band], minlength=len(uniq))
                  / np.maximum(1, np.bincount(inv[band], minlength=len(uniq))))


def _key(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (ix + (1 << 20)) * (1 << 21) + (iy + (1 << 20))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    import cv2
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def measure_frame(depth: np.ndarray, intrinsics: Any, mask: np.ndarray, baseline_depth: np.ndarray,
                  roi: np.ndarray | None = None, *, scale: float = 1.0, align_background: bool = False,
                  fallback_shape: tuple[int, int] | None = None, cell: float = CELL_M,
                  mapping: dict[str, Any] | None = None, distortion: list[float] | None = None,
                  complete_mask: bool = True, exclude: np.ndarray | None = None,
                  box: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
    """Per-pixel visible-surface volume of `mask` above the empty-bin baseline. See module docstring.

    `mapping` (from the frozen calibration) turns the camera's raw depth into metres; without it the raw
    depth is multiplied by `scale` (RealSense: already metres, scale 1). `distortion` (OpenCV k1 k2 p1 p2
    [k3], measured for this camera) undistorts depth, baseline, mask and ROI to the pinhole model first."""
    mapping = mapping or {"kind": "scale", "scale": scale}
    reasons: list[str] = []
    out: dict[str, Any] = {"quantity": QUANTITY, "measurement_version": MEASUREMENT_VERSION, "volume_l": None,
                           "volume_l_partial": None, "status": "invalid", "reasons": reasons,
                           "depth_mapping": mapping, "distortion_modelled": bool(distortion),
                           "cell_m": cell, "background_ratio": None}
    if depth is None or baseline_depth is None or depth.shape != baseline_depth.shape:
        reasons.append("depth and baseline missing or of different resolution")
        return out
    shape = depth.shape
    mask = _resize_mask(mask, shape)
    roi = np.ones(shape, bool) if roi is None else _resize_mask(roi, shape)
    k = scaled_intrinsics(intrinsics, shape, fallback_shape)
    # clipping is judged on the ORIGINAL mask (no erosion): touching the image edge or the ROI boundary
    edge = np.zeros(shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    if (mask & edge).any():
        reasons.append("object clipped at the image edge: part of it is outside the view")
    elif roi is not None and not roi.all() and (mask & _dilate(~roi, 1)).any():
        reasons.append("object reaches the ROI boundary: part of it may be outside the measured region")
    if distortion:
        depth, baseline_depth, mask, roi = undistort_inputs(k, distortion, depth, baseline_depth, mask, roi)
    raw_obj = depth[mask & roi & np.isfinite(depth)]
    depth = apply_depth_mapping(depth.astype(np.float64), mapping)
    base = apply_depth_mapping(baseline_depth.astype(np.float64), mapping)
    valid_range = mapping.get("valid_raw_depth")
    if valid_range and raw_obj.size:
        median_raw = float(np.median(raw_obj))
        lo, hi = valid_range
        if not lo <= median_raw <= hi:
            reasons.append(f"object depth {median_raw:.3f} (raw) outside the calibrated range "
                           f"[{lo:.3f}, {hi:.3f}]: calibration not validated here")
    valid_now = np.isfinite(depth) & (depth > 0.05)
    valid_base = np.isfinite(base) & (base > 0.05)
    obj = mask & roi
    background = roi & ~_dilate(obj, 4) & valid_now & valid_base
    if align_background:
        if np.count_nonzero(background) < 500:
            reasons.append("too little ROI background to align this frame to the baseline")
            return out
        ratio = float(np.median(base[background] / depth[background]))
        if not 0.7 <= ratio <= 1.4:
            reasons.append(f"background alignment ratio {ratio:.3f} is implausible (scene or model changed)")
            out["background_ratio"] = round(ratio, 6)
            return out
        depth = depth * ratio
        out["background_ratio"] = round(ratio, 6)
    P, B = deproject(depth, k), deproject(base, k)
    plane = fit_plane(B[roi & valid_base])
    if plane is None:
        reasons.append("no support plane in the baseline ROI")
        return out
    out["plane"] = {"normal": [round(float(x), 6) for x in plane["normal"]], "d_m": round(plane["d"], 5),
                    "rmse_m": round(plane["rmse_m"], 5)}
    area = _support_area(depth, k, plane["normal"])
    n_obj = int(np.count_nonzero(obj))
    sel = obj & valid_now & np.isfinite(area)
    out.update(object_pixels=n_obj, valid_depth_fraction=round(np.count_nonzero(sel) / max(1, n_obj), 4))
    if n_obj == 0:
        reasons.append("no object mask")
        return out
    if np.count_nonzero(sel) / n_obj < MIN_VALID_DEPTH_FRACTION:
        reasons.append(f"only {np.count_nonzero(sel) / n_obj:.0%} of the object's pixels have valid depth")
    if not np.any(sel):
        return out
    spacing = float(np.sqrt(np.percentile(np.abs(area[sel]), 75)))   # side faces project to ~0: not the top spacing
    cell = max(cell, 2.5 * spacing)
    out["cell_m"] = round(cell, 5)
    bx, by, bh = _cells(B[roi & valid_base], plane, cell)
    base_keys, base_h = _per_cell_mean(_key(bx, by), bh)

    def base_at(keys):
        pos = np.clip(np.searchsorted(base_keys, keys), 0, len(base_keys) - 1)
        hit = base_keys[pos] == keys
        return np.where(hit, base_h[pos], 0.0), hit

    if complete_mask:
        # complete a partial detector mask from the raised surface connected to it (relative to the empty
        # baseline, so floor never qualifies); recorded, and refused for piles/neighbours
        allv = roi & valid_now
        ax, ay, ah = _cells(P[allv], plane, cell)
        rise_all = np.full(shape, np.nan)
        rise_all[allv] = ah - base_at(_key(ax, ay))[0]
        ex = None if exclude is None else _resize_mask(exclude, shape)
        box_px = None
        if box is not None and fallback_shape is not None:
            # detector boxes are in RGB pixels; the depth map may have another resolution
            sx, sy = shape[1] / fallback_shape[1], shape[0] / fallback_shape[0]
            box_px = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        elif box is not None:
            box_px = box
        completed, info = complete_object_mask(rise_all, allv, depth, obj, box_px, ex)
        out["mask_completion"] = info
        if completed is not None and info["completed"]:
            obj = completed & roi
            sel = obj & valid_now & np.isfinite(area)
            out["object_pixels_completed"] = int(np.count_nonzero(obj))
            background = roi & ~_dilate(obj, 4) & valid_now & valid_base
        elif info.get("growth", 1.0) > MASK_TRUNCATION_GROWTH:
            reasons.append(f"the mask covers only part of the raised object ({info['growth']:.2f}x larger), "
                           f"and it could not be completed: {info.get('reason')}")
    ox, oy, oh = _cells(P[sel], plane, cell)
    okeys = _key(ox, oy)
    base_o, has_base = base_at(okeys)
    rise = oh - base_o
    a = area[sel]
    up = rise >= MIN_RISE_M
    # background agreement with the baseline: camera moved, depth drift or a wrong scale shows here
    bgx, bgy, bgh = _cells(P[background], plane, cell)
    if len(bgh):
        bkeys, bmean = _per_cell_mean(_key(bgx, bgy), bgh)
        bb, bhit = base_at(bkeys)
        bres = (bmean - bb)[bhit]
    else:
        bres = np.empty(0)
    out["background_residual_median_m"] = None if not bres.size else round(float(np.median(bres)), 5)
    out["background_residual_p95_abs_m"] = None if not bres.size else round(float(np.percentile(np.abs(bres), 95)), 5)
    low = rise[~up]
    out.update(integrated_pixels=int(up.sum()), baseline_cells_missing=int(len(np.unique(okeys[up & ~has_base]))),
               below_plane_pixels=int((low < -MIN_RISE_M).sum()),
               below_plane_mean_m=None if not (low < -MIN_RISE_M).any() else round(float(low[low < -MIN_RISE_M].mean()), 5),
               leakage_pixels=int(((low >= -MIN_RISE_M) & (low < MIN_RISE_M)).sum()))
    keys = np.unique(okeys)
    foot_keys = np.unique(okeys[up])
    out["footprint_cells"] = int(len(foot_keys))
    if len(foot_keys) < MIN_CELLS:
        reasons.append(f"footprint of {len(foot_keys)} cells above {MIN_RISE_M * 100:.0f} cm is too small")
        out["coverage"] = None
        return out
    volume_l = float(np.sum(rise[up] * a[up])) * 1000.0
    out["volume_l_partial"] = round(volume_l, 5)
    out["height_max_m"] = round(float(np.percentile(rise[up], 99)), 4)
    out["top_height_m"] = round(top_height(rise[up]), 5)
    # share of the object's pixels on a depth edge (> 2 cm + 2 % between neighbours): stereo "flying
    # pixels" between an object's edge and the floor behind it spread the footprint along the view
    zz = np.where(valid_now, depth, np.nan)
    jump = np.zeros(shape, bool)
    for axis in (0, 1):
        dz = np.abs(np.diff(zz, axis=axis))
        big = np.nan_to_num(dz, nan=0.0) > 0.02 + 0.02 * np.nan_to_num(zz[:-1] if axis == 0 else zz[:, :-1], nan=0.0)
        if axis == 0:
            jump[:-1] |= big
            jump[1:] |= big
        else:
            jump[:, :-1] |= big
            jump[:, 1:] |= big
    out["depth_edge_fraction"] = round(float(np.count_nonzero(jump & obj)) / max(1, int(np.count_nonzero(obj))), 4)
    # share of the integrated footprint rising < 25 % of the top: floor counted as object (mask leakage,
    # monocular depth smeared over the object's edge) shows here; a box has ~0, a domed bag up to ~25 %
    top95 = float(np.percentile(rise[up], 95))
    out["low_skirt_area_fraction"] = round(float(np.sum(a[up][rise[up] < 0.25 * top95]) / max(np.sum(a[up]), 1e-12)), 4)
    out["plane_distance_m"] = round(float(plane["d"]), 5)
    # coverage: cells of the footprint's convex hull that were actually seen
    import cv2
    fx_, fy_ = foot_keys // (1 << 21), foot_keys % (1 << 21)
    x0, y0 = int(fx_.min()), int(fy_.min())
    raster = np.zeros((int(fy_.max()) - y0 + 1, int(fx_.max()) - x0 + 1), np.uint8)
    hull = cv2.convexHull(np.c_[fx_ - x0, fy_ - y0].astype(np.int32))
    cv2.fillConvexPoly(raster, hull, 1)
    seen = np.zeros_like(raster)
    ax_, ay_ = keys // (1 << 21) - x0, keys % (1 << 21) - y0
    inside = (ax_ >= 0) & (ay_ >= 0) & (ax_ < raster.shape[1]) & (ay_ < raster.shape[0])
    seen[ay_[inside], ax_[inside]] = 1
    # the hull's own rasterised edge cuts cells that hold no surface; holes are judged in its interior
    interior = cv2.erode(raster, np.ones((3, 3), np.uint8))
    region = interior if interior.sum() >= 10 else raster
    coverage = float(np.count_nonzero(seen & region)) / max(1, int(region.sum()))
    out["coverage"] = round(coverage, 4)
    out["hull_cells"] = int(raster.sum())
    if coverage < MIN_COVERAGE:
        reasons.append(f"coverage {coverage:.0%} < {MIN_COVERAGE:.0%}: part of the surface was not seen")
    out["occluding_boundary_fraction"] = occluding_boundary_fraction(depth, obj & valid_now)
    if out["background_residual_median_m"] is None:
        reasons.append("no ROI background to check the baseline against")
    elif abs(out["background_residual_median_m"]) > BACKGROUND_TOL_M:
        reasons.append(f"background differs from the baseline by {out['background_residual_median_m'] * 100:+.1f} cm "
                       "(camera moved, depth drift or wrong depth scale)")
    if not reasons:
        out.update(status="ok", volume_l=round(volume_l, 5))
    return out


def measure_objects(depth, intrinsics, masks: list[np.ndarray], baseline_depth, roi=None, boxes=None,
                    **kw) -> dict[str, Any]:
    """Aggregate volume of the union of `masks`, plus individual volumes only for masks that touch no
    other mask (overlapping bags have no observable boundary or support between them)."""
    shape = depth.shape
    resized = [_resize_mask(m, shape) for m in masks]
    union = np.zeros(shape, bool)
    for m in resized:
        union |= m
    union_box = None
    if boxes:
        b = np.asarray(boxes, float)
        union_box = (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))
    aggregate = measure_frame(depth, intrinsics, union, baseline_depth, roi, box=union_box, **kw)
    individual = []
    for i, m in enumerate(resized):
        others = np.zeros(shape, bool)
        for j, o in enumerate(resized):
            if j != i:
                others |= o
        if (_dilate(m, 3) & others).any():
            individual.append({"index": i, "volume_l": None, "status": "unavailable",
                               "reasons": ["touches or overlaps another object: no individual boundary/support"]})
        else:
            r = measure_frame(depth, intrinsics, m, baseline_depth, roi, exclude=others,
                              box=None if not boxes or i >= len(boxes) else tuple(boxes[i]), **kw)
            individual.append({"index": i, "volume_l": r["volume_l"], "status": r["status"], "reasons": r["reasons"]})
    aggregate["objects_in_mask"] = len(masks)
    aggregate["individual"] = individual
    return aggregate


# --------------------------------------------------------------------------------------- baseline
def build_baseline(depths: list[np.ndarray], intrinsics: Any, roi: np.ndarray | None, *, camera: str,
                   meta: dict[str, Any], objects_present: int = 0, fallback_shape=None) -> tuple[dict, np.ndarray | None]:
    """Median of N empty-bin frames + quality checks. Returns (record, median depth or None if rejected)."""
    stack = np.stack([d.astype(np.float32) for d in depths]) if depths else None
    record: dict[str, Any] = {"camera": camera, "created_at": time.time(), "frames": len(depths),
                              "aggregation": "per-pixel median of the frames (invalid where < half the frames valid)",
                              "status": "rejected", "reasons": [], **meta}
    if stack is None or len(depths) < 5:
        record["reasons"].append("fewer than 5 baseline frames")
        return record, None
    valid = np.isfinite(stack) & (stack > 0.05)
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(valid, stack, np.nan), axis=0).astype(np.float32)
        std = np.nanstd(np.where(valid, stack, np.nan), axis=0)
    median[valid.sum(axis=0) < len(depths) / 2] = np.nan
    roi_m = np.ones(median.shape, bool) if roi is None else _resize_mask(roi, median.shape)
    ok = roi_m & np.isfinite(median)
    k = scaled_intrinsics(intrinsics, median.shape, fallback_shape)
    plane = fit_plane(deproject(median, k)[ok])
    record.update(shape=list(median.shape), intrinsics=k, roi_pixels=int(roi_m.sum()),
                  valid_fraction=round(float(ok.sum()) / max(1, int(roi_m.sum())), 4),
                  temporal_std_median_m=None if not ok.any() else round(float(np.nanmedian(std[ok])), 5),
                  objects_present=objects_present)
    if plane is not None:
        record["plane"] = {"normal": [round(float(x), 6) for x in plane["normal"]], "d": round(plane["d"], 5),
                           "rmse_m": round(plane["rmse_m"], 5), "inlier_fraction": round(plane["inlier_fraction"], 4),
                           "tilt_from_view_axis_deg": round(math.degrees(math.acos(abs(float(plane["normal"][2])))), 2)}
    reasons = record["reasons"]
    if objects_present:
        reasons.append(f"{objects_present} object(s) detected in the ROI: the bin is not empty")
    if record["valid_fraction"] < BASELINE_MIN_VALID_FRACTION:
        reasons.append(f"only {record['valid_fraction']:.0%} of the ROI has valid depth")
    if plane is None:
        reasons.append("no support plane found")
    elif plane["rmse_m"] > BASELINE_MAX_PLANE_RMSE_M * max(1.0, plane["d"]):
        reasons.append(f"support plane RMSE {plane['rmse_m'] * 100:.1f} cm: floor not flat/visible")
    if record["temporal_std_median_m"] is not None and record["temporal_std_median_m"] > BASELINE_MAX_TEMPORAL_STD_M:
        reasons.append("the scene moved during the baseline")
    if reasons:
        return record, None
    record["status"] = "ok"
    record["baseline_id"] = f"{camera}-bl-{hashlib.sha1(median.tobytes()).hexdigest()[:10]}"
    return record, median


def compatibility(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Why `current` camera state (shape, model, intrinsics, plane) cannot use `reference` (baseline or
    calibration). Empty list = compatible."""
    reasons = []
    if reference.get("shape") and current.get("shape") and list(reference["shape"]) != list(current["shape"]):
        reasons.append(f"resolution changed {reference['shape']} -> {current['shape']}")
    for key in ("depth_model", "output_kind", "intrinsics_source"):
        if reference.get(key) != current.get(key):
            reasons.append(f"{key} changed {reference.get(key)} -> {current.get(key)}")
    ki, kc = reference.get("intrinsics") or {}, current.get("intrinsics") or {}
    for key in ("fx", "fy", "cx", "cy"):
        if key in ki and key in kc and abs(float(ki[key]) - float(kc[key])) > 0.01 * max(1.0, abs(float(ki[key]))):
            reasons.append(f"intrinsics {key} changed")
            break
    pr, pc = reference.get("plane"), current.get("plane")
    if pr and pc:
        angle = math.degrees(math.acos(float(np.clip(np.dot(pr["normal"], pc["normal"]), -1, 1))))
        if angle > POSE_MAX_ANGLE_DEG:
            reasons.append(f"mount pose changed: support plane rotated {angle:.1f} deg")
        if abs(pr["d"] - pc["d"]) > POSE_MAX_DISTANCE_REL * pr["d"]:
            reasons.append(f"mount pose changed: plane distance {pr['d']:.3f} -> {pc['d']:.3f}")
    return reasons


# ------------------------------------------------------------------------------------ calibration
def _nelder_mead(f, x0, step, iterations=200, tol=1e-9):
    """Minimal Nelder-Mead (NumPy only)."""
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(len(x0))[i] * step[i] for i in range(len(x0))]
    vals = [f(p) for p in pts]
    for _ in range(iterations):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centre = np.mean(pts[:-1], axis=0)
        reflect = centre + (centre - pts[-1])
        fr = f(reflect)
        if fr < vals[0]:
            expand = centre + 2 * (centre - pts[-1])
            fe = f(expand)
            pts[-1], vals[-1] = (expand, fe) if fe < fr else (reflect, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = reflect, fr
        else:
            contract = centre + 0.5 * (pts[-1] - centre)
            fc = f(contract)
            if fc < vals[-1]:
                pts[-1], vals[-1] = contract, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    best = int(np.argmin(vals))
    return pts[best], vals[best]


def _mapping_from(kind: str, x: np.ndarray, offset_free: bool) -> dict[str, Any]:
    if kind == "metric_affine":
        return {"kind": kind, "s": float(math.exp(x[0])), "t": float(x[1]) if offset_free else 0.0}
    return {"kind": kind, "a": float(math.exp(x[0])), "b": float(x[1]) if offset_free else 0.0}


def fit_depth_mapping(samples: list[dict[str, Any]], *, camera: str, output_kind: str,
                      floor_distance_m: float | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """THE calibration procedure (one per camera). Fits the raw-depth -> metres mapping chosen by the
    model's output semantics -- metric checkpoint: Z = s x raw + t; relative inverse depth d:
    Z = 1 / (a x d + b); RealSense hardware depth may be checked the same way (metric_affine) -- to
    INDEPENDENT geometric references only:
      * each calibration placement's tape-measured top-surface height above the bin floor
        (`reference_top_height_m`), measured through the same per-pixel geometry (`top_height`), at
        several bin positions/depths;
      * optionally the tape-measured PERPENDICULAR camera-to-floor distance (`floor_distance_m`).
    No volume is fitted, so evaluation objects' volumes never enter. The offset is fitted only when the
    perpendicular floor distance is given (heights alone do not identify it); otherwise only the scale is
    fitted and the record says so.
    samples: [{trial_id, object_id, depth (raw), baseline (raw), mask, roi, intrinsics, distortion,
               fallback_shape, reference_top_height_m}] from CALIBRATION trials only."""
    usable = [smp for smp in samples if smp.get("reference_top_height_m") and smp["reference_top_height_m"] > 0]
    if not usable:
        raise ValueError("calibration needs calibration placements with a measured top-surface height")
    kind = "metric_affine" if output_kind == "metric" else "inverse_affine"
    if kind == "inverse_affine" and not floor_distance_m:
        raise ValueError("relative (inverse-depth) output needs the measured perpendicular floor distance too")
    heights = [smp["reference_top_height_m"] for smp in usable]
    # A depth offset moves an object's top and the floor below it almost equally, so heights pin the SCALE
    # and barely the offset (synthetic: s 1.30 / t -0.08 m came back as s 1.24 from heights alone). Only an
    # absolute distance -- the measured perpendicular floor distance -- identifies the offset.
    offset_free = bool(floor_distance_m)

    def predict(mapping):
        out = []
        for smp in usable:
            r = measure_frame(smp["depth"][::2, ::2], smp["intrinsics"], smp["mask"][::2, ::2],
                              smp["baseline"][::2, ::2], None if smp.get("roi") is None else smp["roi"][::2, ::2],
                              mapping=mapping, distortion=None, fallback_shape=smp.get("fallback_shape"))
            out.append((r.get("top_height_m"), r.get("plane_distance_m")))
        return out

    def cost(x):
        mapping = _mapping_from(kind, x, offset_free)
        total = 0.0
        for (h, d), smp in zip(predict(mapping), usable):
            if h is None or not np.isfinite(h):
                return 1e6
            total += (h - smp["reference_top_height_m"]) ** 2
            if floor_distance_m and d is not None:
                total += (d - floor_distance_m) ** 2 / len(usable)
        return total

    if kind == "metric_affine":
        x0 = [0.0, 0.0]
    else:
        floor_raw = float(np.nanmedian(usable[0]["baseline"]))
        x0 = [math.log(1.0 / (floor_raw * floor_distance_m)), 0.0]
    if offset_free:
        x, _ = _nelder_mead(cost, x0, [0.2, 0.05 if kind == "metric_affine" else 0.05 * abs(math.exp(x0[0]))])
    else:
        x, _ = _nelder_mead(lambda v: cost(np.array([v[0], 0.0])), [x0[0]], [0.2])
        x = np.array([x[0], 0.0])
    mapping = _mapping_from(kind, x, offset_free)
    preds = predict(mapping)
    residuals = [{"trial_id": smp.get("trial_id"), "object_id": smp.get("object_id"),
                  "reference_top_height_m": smp["reference_top_height_m"],
                  "fitted_top_height_m": None if h is None else round(h, 5),
                  "residual_m": None if h is None else round(h - smp["reference_top_height_m"], 5),
                  "residual_pct": None if h is None else round(100 * (h - smp["reference_top_height_m"])
                                                                / smp["reference_top_height_m"], 3)}
                 for (h, _), smp in zip(preds, usable)]
    all_raw = np.concatenate([smp["depth"][smp["mask"] & np.isfinite(smp["depth"])] for smp in usable])
    lo, hi = np.percentile(all_raw, (5, 95))
    mapping["valid_raw_depth"] = [round(float(lo) * 0.9, 4), round(float(hi) * 1.1, 4)]
    floor_check = None
    if floor_distance_m:
        d_fit = [d for _, d in preds if d is not None]
        floor_check = {"measured_perpendicular_m": floor_distance_m,
                       "fitted_m": None if not d_fit else round(float(np.median(d_fit)), 4)}
    body = {"camera": camera, "procedure": "depth mapping fitted to measured top heights (+ perpendicular floor "
                                          "distance) of calibration placements; no volumes fitted",
            "measurement_version": MEASUREMENT_VERSION, "mapping": mapping, "output_kind": output_kind,
            "offset_identifiable": offset_free,
            "note": None if offset_free else ("no measured perpendicular floor distance: offset not identifiable, "
                                              "fixed at 0; scale fitted to the heights"),
            "align_background": False, "fitted_on": [{k: smp.get(k) for k in ("trial_id", "object_id",
                                                                              "reference_top_height_m")} for smp in usable],
            "calibration_objects": sorted({smp.get("object_id") for smp in usable}),
            "fit_quality": {"n_placements": len(usable), "residuals": residuals, "floor_check": floor_check,
                            "rms_height_residual_m": round(float(np.sqrt(np.nanmean(
                                [r["residual_m"] ** 2 for r in residuals if r["residual_m"] is not None]))), 5),
                            "reference_heights_m": [min(heights), max(heights)],
                            "raw_depth_range": [round(float(lo), 4), round(float(hi), 4)]},
            "created_at": time.time(), "frozen": True, **(meta or {})}
    body["calibration_id"] = f"{camera}-cal-{hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:10]}"
    return body


# ------------------------------------------------------------------------------------ evaluation
def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    a = np.asarray(values, float)
    return {"mean": round(float(a.mean()), 5), "median": round(float(np.median(a)), 5),
            "std": round(float(a.std(ddof=1)), 5) if len(a) > 1 else None}


def evaluate(trials: list[dict[str, Any]], *, calibration_objects: dict[str, set[str]] | None = None,
             criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per camera/object/condition/motion statistics, held-out separation and paired comparison.
    trials: trial records with `cameras[camera].volume_l` (None = failed) and the reference."""
    calibration_objects = calibration_objects or {}
    criteria = criteria or {}
    rows, groups = [], {}
    for t in trials:
        ref = t.get("reference_volume_l")
        for camera, res in (t.get("cameras") or {}).items():
            held_out = t.get("designation") == "test" and t.get("object_id") not in calibration_objects.get(camera, set())
            key = (camera, t.get("object_id"), t.get("condition", "isolated"), res.get("motion_state", "settled"),
                   "held_out" if held_out else ("calibration" if t.get("designation") == "calibration"
                                                else "test_object_used_in_calibration"))
            groups.setdefault(key, []).append((t, res, ref))
    for (camera, obj, condition, motion, split), items in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        est = [r.get("volume_l") for _, r, _ in items]
        refs = {ref for _, _, ref in items}
        ref = next(iter(refs)) if len(refs) == 1 else None
        valid = [v for v in est if v is not None]
        fails: dict[str, int] = {}
        for _, r, _ in items:
            if r.get("volume_l") is None:
                reason = (r.get("reasons") or [r.get("status") or "unknown"])[0]
                fails[reason] = fails.get(reason, 0) + 1
        row = {"camera": camera, "object_id": obj, "condition": condition, "motion_state": motion, "split": split,
               "attempted": len(items), "valid": len(valid), "failed": len(items) - len(valid),
               "availability": round(len(valid) / len(items), 4), "failure_reasons": fails,
               "reference_l": ref, **{f"estimate_{k}_l": v for k, v in _stats(valid).items()}}
        quantities = {(t.get("reference") or {}).get("reference_quantity") for t, _, _ in items}
        row["reference_quantity"] = next(iter(quantities)) if len(quantities) == 1 else sorted(map(str, quantities))
        if not quantities <= set(EXTERNAL_VOLUME_REFERENCES):
            row["error_note"] = (f"reference is {row['reference_quantity']}, not a measured external volume: "
                                 "errors not computed (record an external measurement to score accuracy)")
        elif ref is None or ref <= 0:
            row["error_note"] = "no single positive reference volume: errors not computed"
        elif valid:
            err = np.asarray(valid) - ref                        # per trial, before averaging
            pct = 100 * err / ref
            row.update(mean_signed_error_l=round(float(err.mean()), 5), mean_signed_error_pct=round(float(pct.mean()), 3),
                       mae_l=round(float(np.abs(err).mean()), 5), rmse_l=round(float(np.sqrt((err ** 2).mean())), 5),
                       mape_pct=round(float(np.abs(pct).mean()), 3),
                       std_signed_error_pct=round(float(pct.std(ddof=1)), 3) if len(pct) > 1 else None)
        acc, avail = criteria.get("max_mape_pct"), criteria.get("min_availability")
        if acc is None or avail is None:
            row["criteria"] = "not agreed"
        elif split != "held_out":
            row["criteria"] = "not applicable (not a held-out trial)"
        else:
            row["criteria"] = "pass" if (row.get("mape_pct") is not None and row["mape_pct"] <= acc
                                         and row["availability"] >= avail) else "fail"
        rows.append(row)
    paired = []
    for t in trials:
        cams = t.get("cameras") or {}
        rs, lg = (cams.get("realsense") or {}).get("volume_l"), (cams.get("logitech") or {}).get("volume_l")
        if rs is not None and lg is not None:
            paired.append({"trial_id": t["trial_id"], "object_id": t.get("object_id"), "realsense_l": rs,
                           "logitech_l": lg, "reference_l": t.get("reference_volume_l")})
    diff = [p["logitech_l"] - p["realsense_l"] for p in paired]
    return {"rows": rows, "paired": {"n_paired_valid": len(paired), "trials": paired,
                                     "logitech_minus_realsense_l": _stats(diff),
                                     "note": "agreement between cameras, not accuracy against the reference"},
            "criteria": criteria or "not agreed"}


# -------------------------------------------------------------------------------------- recorder
def software_version() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=str(Path(__file__).resolve().parent)).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                                    timeout=5, cwd=str(Path(__file__).resolve().parent)).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    return {"commit": commit, "uncommitted_changes": dirty, "measurement_version": MEASUREMENT_VERSION}


@dataclass
class FramePayload:
    camera: str
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray | None                 # RealSense metres / Logitech raw model output
    intrinsics: dict[str, Any] | None        # for `rgb`'s resolution
    masks: list[np.ndarray]
    track_ids: list[int | None]
    labels: list[str]
    roi: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)   # detector boxes, RGB pixels   # depth_model, output_kind, intrinsics_source, depth_units


class ExperimentRecorder:
    """Sessions, baselines, trials and their raw data under `root` (results/experiment)."""

    def __init__(self, root: Path, *, frames_per_trial: int = 10, settle_s: float = 1.5, window_s: float = 4.0,
                 timeout_s: float = 25.0, baseline_frames: int = 15) -> None:
        self.root = Path(root)
        self.frames_per_trial, self.settle_s, self.window_s, self.timeout_s = frames_per_trial, settle_s, window_s, timeout_s
        self.baseline_frames = baseline_frames
        self.lock = threading.Lock()
        self.session: dict[str, Any] | None = None
        self.baselines: dict[str, dict[str, Any]] = {}
        self._baseline_depth: dict[str, np.ndarray] = {}
        self._capture: dict[str, list[FramePayload]] | None = None
        self.trial: dict[str, Any] | None = None
        self._motion: dict[str, dict[str, Any]] = {}
        self.last_trial: dict[str, Any] | None = None
        self.config_meta: dict[str, Any] = {}      # effective app configuration, set by the coordinator

    # paths
    @property
    def session_dir(self) -> Path:
        assert self.session is not None
        return self.root / "sessions" / self.session["session_id"]

    def calibration_path(self, camera: str) -> Path:
        return self.root / "calibration" / f"{camera}.json"

    def objects_path(self) -> Path:
        return self.root / "objects.json"

    # objects
    def objects(self) -> dict[str, Any]:
        try:
            return json.loads(self.objects_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def add_object(self, object_id: str, *, reference_volume_l: float | None, reference_method: str,
                   reference_quantity: str, reference_status: str, reference_uncertainty_l: float | None = None,
                   notes: str = "") -> dict[str, Any]:
        if reference_status not in REFERENCE_STATUSES or reference_quantity not in REFERENCE_QUANTITIES:
            raise ValueError(f"status in {REFERENCE_STATUSES}, quantity in {REFERENCE_QUANTITIES}")
        if reference_status == "measured" and (reference_volume_l is None or reference_volume_l <= 0 or not reference_method):
            raise ValueError("a measured reference needs a positive volume and its measurement method")
        objects = self.objects()
        objects[object_id] = {"reference_volume_l": reference_volume_l, "reference_method": reference_method,
                              "reference_quantity": reference_quantity, "reference_status": reference_status,
                              "reference_uncertainty_l": reference_uncertainty_l, "notes": notes,
                              "recorded_at": time.time()}
        self.objects_path().parent.mkdir(parents=True, exist_ok=True)
        self.objects_path().write_text(json.dumps(objects, indent=1), encoding="utf-8")
        return objects[object_id]

    def calibration(self, camera: str) -> dict[str, Any] | None:
        try:
            return json.loads(self.calibration_path(camera).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # session / baseline
    def start_session(self, note: str = "") -> dict[str, Any]:
        with self.lock:
            sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            self.session = {"session_id": sid, "started_at": time.time(), "note": note, "software": software_version(),
                            "app_config": dict(self.config_meta),
                            "settings": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))}}
            self.baselines, self._baseline_depth, self.trial, self._capture = {}, {}, None, None
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "session.json").write_text(json.dumps(self.session, indent=1), encoding="utf-8")
            return self.session

    def capture_baseline(self, cameras: tuple[str, ...] = CAMERAS) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise RuntimeError("start a session first")
            if self.trial is not None:
                raise RuntimeError("a trial is running")
            self._capture = {c: [] for c in cameras}
            for c in cameras:
                self.baselines.pop(c, None)
                self._baseline_depth.pop(c, None)
            return {"capturing": list(cameras), "frames_needed": self.baseline_frames}

    def _finish_baseline(self, camera: str, frames: list[FramePayload]) -> None:
        last = frames[-1]
        meta = {k: last.meta.get(k) for k in ("depth_model", "output_kind", "intrinsics_source", "depth_units")}
        record, median = build_baseline([f.depth for f in frames], last.intrinsics, last.roi, camera=camera, meta=meta,
                                        objects_present=max(len(f.masks) for f in frames),
                                        fallback_shape=last.rgb.shape[:2])
        record["session_id"] = self.session["session_id"]
        directory = self.session_dir / "baselines"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = f"{camera}_{int(record['created_at'])}"
        np.savez_compressed(directory / f"{stamp}.npz", median=median if median is not None else np.zeros(0),
                            roi=np.zeros(0) if last.roi is None else last.roi, rgb=last.rgb,
                            frames=np.stack([f.depth.astype(np.float32) for f in frames]))
        record["file"] = f"baselines/{stamp}.npz"
        (directory / f"{stamp}.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
        self.baselines[camera] = record
        if median is not None:
            self._baseline_depth[camera] = median

    # trials
    def start_trial(self, *, object_id: str, designation: str, placement: str = "", condition: str = "isolated",
                    motion: str = "settled", trial_id: str | None = None,
                    top_height_m: float | None = None) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise RuntimeError("start a session first")
            if self.trial is not None:
                raise RuntimeError(f"trial {self.trial['trial_id']} is still running")
            if designation not in DESIGNATIONS or condition not in CONDITIONS or motion not in ("settled", "moving"):
                raise ValueError(f"designation {DESIGNATIONS}, condition {CONDITIONS}, motion settled|moving")
            obj = self.objects().get(object_id)
            if obj is None:
                raise ValueError(f"unknown object {object_id!r}: add it with its reference first")
            cal_objects = {c: set((self.calibration(c) or {}).get("calibration_objects", [])) for c in CAMERAS}
            if designation == "test" and any(object_id in s for s in cal_objects.values()):
                raise ValueError(f"{object_id} was used to fit a calibration: it cannot be a held-out test object")
            tid = trial_id or f"T{len(list((self.session_dir / 'trials').glob('*'))) + 1:03d}-{uuid.uuid4().hex[:4]}"
            self.trial = {"trial_id": tid, "session_id": self.session["session_id"], "object_id": object_id,
                          "designation": designation, "placement": placement, "condition": condition,
                          "motion_requested": motion, "started_at": time.time(),
                          "reference_volume_l": obj["reference_volume_l"], "reference": obj,
                          # tape-measured top-surface height above the floor in THIS placement (calibration)
                          "reference_top_height_m": top_height_m,
                          "software": self.session["software"], "cameras": {}, "_frames": {c: [] for c in CAMERAS},
                          "_settled_at": {}, "_still_since": {}}
            self._motion = {}
            return {k: v for k, v in self.trial.items() if not k.startswith("_")}

    def stop_trial(self, reason: str = "stopped by operator") -> dict[str, Any] | None:
        with self.lock:
            return self._finish_trial(reason)

    def _camera_setup(self, camera: str, payload: FramePayload) -> tuple[dict[str, Any] | None, list[str]]:
        """(scale/alignment settings, blocking reasons) for this camera now."""
        reasons: list[str] = []
        baseline = self.baselines.get(camera)
        if baseline is None or baseline.get("status") != "ok" or camera not in self._baseline_depth:
            reasons.append("no accepted empty-bin baseline for this camera in this session")
            return None, reasons
        if payload.depth is None:
            reasons.append("no depth for this frame")
            return None, reasons
        current = {"shape": list(payload.depth.shape),
                   "intrinsics": scaled_intrinsics(payload.intrinsics, payload.depth.shape, payload.rgb.shape[:2]),
                   **{k: payload.meta.get(k) for k in ("depth_model", "output_kind", "intrinsics_source", "depth_units")}}
        reasons += [f"baseline invalid: {r}" for r in compatibility(baseline, current)]
        calibration = self.calibration(camera)
        distortion = payload.meta.get("distortion")
        if calibration is None:
            if camera == "logitech":
                reasons.append("no frozen Logitech calibration (run the calibration procedure)")
                return None, reasons
            setup = {"mapping": {"kind": "scale", "scale": 1.0}, "align_background": False,
                     "calibration_id": "factory-depth-scale", "distortion": distortion}
        else:
            reasons += [f"calibration invalid: {r}" for r in compatibility(calibration, {**current, "plane": baseline.get("plane")})]
            setup = {"mapping": calibration["mapping"], "align_background": calibration.get("align_background", False),
                     "calibration_id": calibration["calibration_id"], "distortion": distortion}
        if payload.meta.get("output_kind") not in (None, "metric") and setup["mapping"].get("kind") != "inverse_affine":
            reasons.append("relative (inverse) depth output needs an inverse-depth calibration")
        return (None if reasons else setup), reasons

    def wants_frames(self) -> bool:
        """Cheap check for the pipeline: only a baseline capture or a running trial needs frames."""
        return self._capture is not None or self.trial is not None

    def cameras_path(self) -> Path:
        return self.root / "cameras.json"

    def _apply_camera_config(self, payload: FramePayload) -> None:
        """Measured intrinsics/distortion for this camera (cameras.json, e.g. from the checkerboard
        command) replace the pipeline's (Logitech: field-of-view estimate); recorded as the source."""
        try:
            config = json.loads(self.cameras_path().read_text(encoding="utf-8")).get(payload.camera)
        except (OSError, ValueError):
            config = None
        if config:
            payload.intrinsics = dict(config["intrinsics"])
            payload.meta["intrinsics_source"] = f"measured: {config.get('source', 'cameras.json')}"
            payload.meta["distortion"] = config.get("distortion")
        payload.meta.setdefault("distortion", None)

    def observe(self, payload: FramePayload) -> None:
        """Called for every processed frame of each camera."""
        self._apply_camera_config(payload)
        with self.lock:
            if self._capture is not None and payload.camera in self._capture and payload.depth is not None:
                frames = self._capture[payload.camera]
                frames.append(payload)
                if len(frames) >= self.baseline_frames:
                    self._finish_baseline(payload.camera, frames)
                    del self._capture[payload.camera]
                    if not self._capture:
                        self._capture = None
                return
            trial = self.trial
            if trial is None or payload.camera not in trial["_frames"] or payload.timestamp < trial["started_at"]:
                return                                       # never a frame from before the trial started
            motion = self._settle(payload)
            moving_trial = trial["motion_requested"] == "moving"
            settled_at = trial["_settled_at"].get(payload.camera)
            phase = "moving" if moving_trial else ("settled" if settled_at is not None else "settling")
            frames = trial["_frames"][payload.camera]
            in_window = moving_trial or settled_at is not None
            if in_window and sum(f["phase"] != "settling" for f in frames) < self.frames_per_trial:
                frames.append(self._record_frame(payload, phase, motion))
            elif not in_window and len(frames) < 3 * self.frames_per_trial and len(frames) % 3 == 0:
                frames.append(self._record_frame(payload, phase, motion))     # settling attempts, thinned
            elif not in_window:
                frames.append({"phase": "settling", "timestamp": payload.timestamp, "motion": motion, "saved": False})
            done = all(sum(f["phase"] != "settling" for f in fr) >= self.frames_per_trial for fr in trial["_frames"].values())
            if done:
                self._finish_trial("frame window complete")
            elif payload.timestamp - trial["started_at"] > self.timeout_s:     # frame clock
                self._finish_trial("timeout")

    def _settle(self, payload: FramePayload) -> float | None:
        """Changed-pixel fraction in the object's (dilated) region since this camera's previous frame;
        settled after `settle_s` below 1 %. Background motion outside the object region is ignored."""
        import cv2
        grey = cv2.cvtColor(payload.rgb, cv2.COLOR_BGR2GRAY) if payload.rgb.ndim == 3 else payload.rgb
        grey = cv2.resize(grey, (160, 120), interpolation=cv2.INTER_AREA)
        region = None
        if payload.masks:
            union = np.zeros(payload.masks[0].shape, bool)
            for m in payload.masks:
                union |= m.astype(bool)
            region = _dilate(_resize_mask(union, (120, 160)), 4)
        elif payload.roi is not None:
            region = _resize_mask(payload.roi, (120, 160))
        prev = self._motion.get(payload.camera)
        self._motion[payload.camera] = {"grey": grey}
        if prev is None or region is None or not region.any():
            return None
        changed = np.abs(grey.astype(np.int16) - prev["grey"].astype(np.int16)) > 20
        motion = float(np.count_nonzero(changed & region)) / int(np.count_nonzero(region))
        trial = self.trial
        if motion < 0.01:
            trial["_still_since"].setdefault(payload.camera, payload.timestamp)
            if payload.timestamp - trial["_still_since"][payload.camera] >= self.settle_s:
                trial["_settled_at"].setdefault(payload.camera, payload.timestamp)
        else:
            trial["_still_since"].pop(payload.camera, None)
        return round(motion, 5)

    def _record_frame(self, payload: FramePayload, phase: str, motion: float | None) -> dict[str, Any]:
        trial = self.trial
        index = len(trial["_frames"][payload.camera])
        directory = self.session_dir / "trials" / trial["trial_id"] / payload.camera
        directory.mkdir(parents=True, exist_ok=True)
        setup, reasons = self._camera_setup(payload.camera, payload)
        if setup is not None and payload.masks:
            result = measure_objects(payload.depth, payload.intrinsics, payload.masks,
                                     self._baseline_depth[payload.camera], payload.roi, boxes=payload.boxes,
                                     **_setup_kwargs(setup),
                                     fallback_shape=payload.rgb.shape[:2])
        else:
            result = {"volume_l": None, "status": "unavailable",
                      "reasons": reasons or ["no object mask in this frame"]}
        name = f"frame_{index:03d}.npz"
        empty = np.zeros(0)
        np.savez_compressed(directory / name, rgb=payload.rgb,
                            depth=empty if payload.depth is None else payload.depth.astype(np.float32),
                            masks=np.stack(payload.masks).astype(bool) if payload.masks else empty,
                            roi=empty if payload.roi is None else payload.roi.astype(bool),
                            boxes=np.asarray(payload.boxes, np.float32) if payload.boxes else empty)
        frame = {"index": index, "file": f"{payload.camera}/{name}", "timestamp": payload.timestamp, "phase": phase,
                 "motion": motion, "saved": True, "track_ids": payload.track_ids, "labels": payload.labels,
                 "intrinsics": payload.intrinsics, "meta": payload.meta, "setup": setup,
                 # conditions no calibration can lift (replay with another calibration keeps them)
                 "setup_blockers": [r for r in reasons if not r.startswith(("no frozen", "calibration invalid"))],
                 "baseline_id": (self.baselines.get(payload.camera) or {}).get("baseline_id"),
                 "baseline_file": (self.baselines.get(payload.camera) or {}).get("file"),
                 "volume_l": result.get("volume_l"), "status": result.get("status"), "reasons": result.get("reasons"),
                 "result": {k: v for k, v in result.items() if k not in ("reasons",)}}
        return frame

    def _finish_trial(self, reason: str) -> dict[str, Any] | None:
        trial = self.trial
        if trial is None:
            return None
        self.trial = None
        record = {k: v for k, v in trial.items() if not k.startswith("_")}
        record.update(finished_at=time.time(), end_reason=reason, cameras={})
        for camera, frames in trial["_frames"].items():
            record["cameras"][camera] = aggregate_trial(frames, trial["motion_requested"],
                                                        settled=camera in trial["_settled_at"])
            record["cameras"][camera]["frames"] = frames
        directory = self.session_dir / "trials" / trial["trial_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trial.json").write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
        _append_csv(self.session_dir / "trials.csv", trial_rows(record))
        self.last_trial = record
        return record

    def status(self) -> dict[str, Any]:
        with self.lock:
            trial = None if self.trial is None else {
                "trial_id": self.trial["trial_id"], "object_id": self.trial["object_id"],
                "frames": {c: sum(f.get("saved", False) for f in fr) for c, fr in self.trial["_frames"].items()},
                "settled": sorted(self.trial["_settled_at"])}
            return {"session": self.session, "baselines": self.baselines,
                    "baseline_capture": None if self._capture is None else {c: len(f) for c, f in self._capture.items()},
                    "trial": trial, "calibration": {c: (self.calibration(c) or {}).get("calibration_id") for c in CAMERAS},
                    "last_trial": None if self.last_trial is None else trial_rows(self.last_trial)}


def _setup_kwargs(setup: dict[str, Any]) -> dict[str, Any]:
    """measure_frame keyword arguments from a recorded setup (older records carry depth_scale)."""
    mapping = setup.get("mapping") or {"kind": "scale", "scale": float(setup.get("depth_scale", 1.0))}
    return {"mapping": mapping, "align_background": bool(setup.get("align_background", False)),
            "distortion": setup.get("distortion")}


def aggregate_trial(frames: list[dict[str, Any]], motion_requested: str, settled: bool) -> dict[str, Any]:
    """Trial value = MEDIAN of the valid frame volumes inside the trial's window (settled frames, or all
    frames of a 'moving' trial); fewer than MIN_VALID_FRAMES_PER_TRIAL valid frames -> unavailable."""
    window = [f for f in frames if f.get("phase") == ("moving" if motion_requested == "moving" else "settled")]
    vols = [f["volume_l"] for f in window if f.get("volume_l") is not None]
    reasons: dict[str, int] = {}
    for f in window:
        if f.get("volume_l") is None:
            for r in f.get("reasons") or ["unknown"]:
                reasons[r] = reasons.get(r, 0) + 1
    if motion_requested == "moving":
        motion_state = "moving"
    else:
        motion_state = "settled" if settled else "timeout_moving"
    out = {"motion_state": motion_state, "window_frames": len(window), "valid_frames": len(vols),
           "frame_reasons": reasons, "aggregation": "median of valid frame volumes in the window"}
    if not frames:
        out.update(motion_state="no_frames", volume_l=None, status="no_frames",
                   reasons=["this camera delivered no frames during the trial"])
    elif motion_state == "timeout_moving":
        out.update(volume_l=None, status="timeout_moving",
                   reasons=["object region never settled before the timeout; settling frames kept"])
    elif len(vols) < MIN_VALID_FRAMES_PER_TRIAL:
        out.update(volume_l=None, status="insufficient_valid_frames",
                   reasons=[f"{len(vols)} valid frames < {MIN_VALID_FRAMES_PER_TRIAL}"] + sorted(reasons, key=reasons.get, reverse=True)[:1])
    else:
        a = np.asarray(vols)
        median = float(np.median(a))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        out.update(frame_iqr_l=round(iqr, 5), frame_iqr_rel=round(iqr / median, 4) if median > 0 else None)
        tops = [f.get("result", {}).get("top_height_m") for f in window if f.get("volume_l") is not None]
        tops = [t for t in tops if t is not None]
        out["top_height_m"] = round(float(np.median(tops)), 5) if tops else None
        if median > 0 and iqr / median > MAX_FRAME_IQR_REL:
            out.update(volume_l=None, status="unstable", volume_l_unstable_median=round(median, 5),
                       reasons=[f"frame volumes vary by {iqr / median:.0%} (IQR/median) in the settled window "
                                f"(> {MAX_FRAME_IQR_REL:.0%})"])
        else:
            out.update(volume_l=round(median, 5), status="ok", reasons=[])
    return out


def trial_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for camera, res in record.get("cameras", {}).items():
        setups = [f.get("setup") for f in res.get("frames", []) if f.get("setup")]
        saved = [f for f in res.get("frames", []) if f.get("saved")]
        last = saved[-1] if saved else {}
        meta = last.get("meta") or {}
        mapping = (setups[-1] or {}).get("mapping") if setups else None
        rows.append({"session_id": record["session_id"], "trial_id": record["trial_id"], "camera": camera,
                     "object_id": record["object_id"], "designation": record["designation"],
                     "condition": record["condition"], "placement": record["placement"],
                     "motion_state": res.get("motion_state"), "reference_volume_l": record.get("reference_volume_l"),
                     "reference_status": (record.get("reference") or {}).get("reference_status"),
                     "volume_l": res.get("volume_l"), "status": res.get("status"),
                     "valid_frames": res.get("valid_frames"), "window_frames": res.get("window_frames"),
                     "reasons": "; ".join(res.get("reasons") or []),
                     "calibration_id": setups[-1]["calibration_id"] if setups else None,
                     "depth_mapping": None if not mapping else json.dumps({k: v for k, v in mapping.items()
                                                                          if k != "valid_raw_depth"}),
                     "baseline_id": last.get("baseline_id"), "depth_model": meta.get("depth_model"),
                     "output_kind": meta.get("output_kind"), "intrinsics_source": meta.get("intrinsics_source"),
                     "distortion_modelled": bool(meta.get("distortion")),
                     "resolution": None if not last.get("intrinsics") else
                     f"{last['intrinsics'].get('width')}x{last['intrinsics'].get('height')}",
                     "frame_iqr_rel": res.get("frame_iqr_rel"), "top_height_m": res.get("top_height_m"),
                     "reference_top_height_m": record.get("reference_top_height_m"),
                     "quantity": QUANTITY, "measurement_version": MEASUREMENT_VERSION,
                     "commit": (record.get("software") or {}).get("commit")})
    return rows


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ----------------------------------------------------------------------------------------- replay
def load_trials(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(p.rglob("trial.json"))
        out += [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return out


def replay_trial(trial_dir: Path, setups: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Recompute every saved frame of a trial from its raw files and the recorded baseline. With
    `setups=None` the recorded scale/alignment is used (reproduction); otherwise per-camera overrides
    (e.g. a calibration being fitted or a newly frozen one)."""
    trial_dir = Path(trial_dir)
    record = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
    session_dir = trial_dir.parent.parent
    out = {"trial_id": record["trial_id"], "cameras": {}}
    for camera, res in record["cameras"].items():
        frames = []
        for f in res.get("frames", []):
            if not f.get("saved"):
                frames.append(f)
                continue
            override = (setups or {}).get(camera)
            setup = f.get("setup")
            if override is not None:
                # a new calibration replaces the mapping; the frame's own camera geometry (distortion) stays
                setup = {"distortion": (f.get("meta") or {}).get("distortion"), **override}
            g = dict(f)
            if f.get("setup_blockers"):
                g.update(volume_l=None, status="unavailable", reasons=f["setup_blockers"])
                frames.append(g)
                continue
            if setup is None or not f.get("baseline_file"):
                g.update(volume_l=None, status="unavailable", reasons=f.get("reasons") or ["no recorded setup"])
                frames.append(g)
                continue
            raw = np.load(trial_dir / f["file"])
            base = np.load(session_dir / f["baseline_file"])["median"]
            masks = list(raw["masks"]) if raw["masks"].size else []
            if not masks:
                g.update(volume_l=None, status="unavailable", reasons=["no object mask in this frame"])
            else:
                boxes = [tuple(b) for b in raw["boxes"]] if "boxes" in raw.files and raw["boxes"].size else None
                r = measure_objects(raw["depth"], f["intrinsics"], masks, base,
                                    raw["roi"] if raw["roi"].size else None, boxes=boxes, **_setup_kwargs(setup),
                                    fallback_shape=raw["rgb"].shape[:2])
                g.update(volume_l=r["volume_l"], status=r["status"], reasons=r["reasons"],
                         volume_l_partial=r.get("volume_l_partial"), setup=setup)
            frames.append(g)
        agg = aggregate_trial(frames, record.get("motion_requested", "settled"), res.get("motion_state") != "timeout_moving")
        agg["frames"] = frames
        agg["recorded_volume_l"] = res.get("volume_l")
        out["cameras"][camera] = agg
    return out


def calibration_samples(trial_dir: Path, camera: str) -> list[dict[str, Any]]:
    """Raw inputs of a CALIBRATION trial for `fit_depth_mapping`: its middle saved settled frame."""
    trial_dir = Path(trial_dir)
    record = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
    if record.get("designation") != "calibration" or camera not in record.get("cameras", {}):
        return []
    frames = [f for f in record["cameras"][camera].get("frames", [])
              if f.get("saved") and f.get("phase") == "settled" and f.get("baseline_file") and not f.get("setup_blockers")]
    if not frames:
        return []
    f = frames[len(frames) // 2]
    raw = np.load(trial_dir / f["file"])
    if not raw["masks"].size:
        return []
    base = np.load(trial_dir.parent.parent / f["baseline_file"])["median"]
    mask = np.zeros(raw["depth"].shape, bool)
    for m in raw["masks"]:
        mask |= _resize_mask(m, raw["depth"].shape)
    roi = _resize_mask(raw["roi"], raw["depth"].shape) if raw["roi"].size else None
    depth, baseline = raw["depth"], base
    k = scaled_intrinsics(f["intrinsics"], depth.shape, raw["rgb"].shape[:2])
    if (f.get("meta") or {}).get("distortion"):
        depth, baseline, mask, roi_u = undistort_inputs(k, f["meta"]["distortion"], depth, base, mask,
                                                         np.ones(depth.shape, bool) if roi is None else roi)
        roi = roi_u
    return [{"trial_id": record["trial_id"], "object_id": record["object_id"], "depth": depth, "baseline": baseline,
             "mask": mask, "roi": roi, "intrinsics": k, "fallback_shape": None,
             "reference_top_height_m": record.get("reference_top_height_m"), "baseline_file": f["baseline_file"],
             "meta": f.get("meta") or {}}]


def intrinsics_from_corners(object_points: list[np.ndarray], image_points: list[np.ndarray],
                            image_size: tuple[int, int], source: str) -> dict[str, Any]:
    """Measured pinhole intrinsics + distortion from checkerboard correspondences (cv2.calibrateCamera)."""
    import cv2
    rms, K, dist, _, _ = cv2.calibrateCamera([o.astype(np.float32) for o in object_points],
                                             [i.astype(np.float32) for i in image_points], image_size, None, None)
    return {"intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "ppx": float(K[0, 2]), "ppy": float(K[1, 2]),
                           "width": int(image_size[0]), "height": int(image_size[1])},
            "distortion": [float(v) for v in dist.ravel()[:5]], "reprojection_rms_px": float(rms),
            "views": len(image_points), "source": source, "created_at": time.time()}


def env_criteria() -> dict[str, Any]:
    """Supervisor-agreed criteria, if recorded: LOCALLIFE_EXPERIMENT_MAX_MAPE_PCT / _MIN_AVAILABILITY."""
    mape, avail = os.environ.get("LOCALLIFE_EXPERIMENT_MAX_MAPE_PCT"), os.environ.get("LOCALLIFE_EXPERIMENT_MIN_AVAILABILITY")
    return {} if mape is None or avail is None else {"max_mape_pct": float(mape), "min_availability": float(avail)}
