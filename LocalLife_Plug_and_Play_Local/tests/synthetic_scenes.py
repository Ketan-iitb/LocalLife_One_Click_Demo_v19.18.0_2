"""SYNTHETIC reference scenes for geometry tests (exact ray casting, no camera).

Everything here is closed-form: a pinhole camera pitched about its X axis
looks at a flat floor (world z = 0) on which rigid oriented boxes, vertical
cylinders or a height field stand. Depth is camera-Z (the RealSense depth
convention), not ray length. These scenes check that the geometry code is
self-consistent; they say nothing about physical camera accuracy (noise
models, IR dropout, flying pixels and segmentation errors of a real D435 are
only crudely imitated by the optional noise/dropout arguments).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from locallife_cloud.types import CameraIntrinsics


@dataclass
class Box:
    centre_xy: tuple[float, float]
    length: float
    width: float
    height: float
    yaw_deg: float = 0.0


@dataclass
class Cylinder:
    centre_xy: tuple[float, float]
    radius: float
    height: float


def intrinsics(width: int = 640, height: int = 480, focal: float = 600.0) -> CameraIntrinsics:
    return CameraIntrinsics(fx=focal, fy=focal, ppx=width / 2.0, ppy=height / 2.0, width=width, height=height)


def _rays(k: CameraIntrinsics, pitch_deg: float, camera_height_m: float):
    """Camera centre and per-pixel ray directions with camera-Z component 1."""
    b = math.radians(pitch_deg)
    rows, cols = np.mgrid[0:k.height, 0:k.width].astype(np.float64)
    xn, yn = (cols - k.ppx) / k.fx, (rows - k.ppy) / k.fy
    right = np.array([1.0, 0.0, 0.0])
    down = np.array([0.0, -math.cos(b), -math.sin(b)])
    forward = np.array([0.0, math.sin(b), -math.cos(b)])
    d = xn[..., None] * right + yn[..., None] * down + forward
    centre = np.array([0.0, -camera_height_m * math.tan(b), camera_height_m])
    return centre, d


def render(
    k: CameraIntrinsics,
    *,
    pitch_deg: float = 0.0,
    camera_height_m: float = 1.0,
    boxes: tuple[Box, ...] = (),
    cylinders: tuple[Cylinder, ...] = (),
    heightfield=None,
    noise_m: float = 0.0,
    dropout: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return (depth_with_objects, empty_floor_depth, per-object masks)."""
    centre, d = _rays(k, pitch_deg, camera_height_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_t = np.where(d[..., 2] < 0, -centre[2] / d[..., 2], np.inf)
    nearest = floor_t.copy()
    owners = np.full(nearest.shape, -1, dtype=np.int64)
    index = 0
    for box in boxes:
        yaw = math.radians(box.yaw_deg)
        axes = np.array([[math.cos(yaw), math.sin(yaw), 0.0], [-math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
        half = np.array([box.length / 2, box.width / 2, box.height / 2])
        origin = np.array([box.centre_xy[0], box.centre_xy[1], box.height / 2])
        o, dd = (centre - origin) @ axes.T, d @ axes.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / dd, (half - o) / dd
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < nearest)
        nearest[hit], owners[hit] = tmin[hit], index
        index += 1
    for cylinder in cylinders:
        cx, cy = cylinder.centre_xy
        ox, oy = centre[0] - cx, centre[1] - cy
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        bq = 2 * (ox * d[..., 0] + oy * d[..., 1])
        c = ox * ox + oy * oy - cylinder.radius ** 2
        disc = bq * bq - 4 * a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            side_t = (-bq - np.sqrt(disc)) / (2 * a)
            side_z = centre[2] + side_t * d[..., 2]
            side_ok = (disc >= 0) & (side_t > 0) & (side_z >= 0) & (side_z <= cylinder.height)
            top_t = (cylinder.height - centre[2]) / d[..., 2]
            px, py = centre[0] + top_t * d[..., 0] - cx, centre[1] + top_t * d[..., 1] - cy
            top_ok = (top_t > 0) & (px * px + py * py <= cylinder.radius ** 2)
        t = np.where(top_ok, top_t, np.inf)
        t = np.where(side_ok & (side_t < t), side_t, t)
        hit = np.isfinite(t) & (t < nearest)
        nearest[hit], owners[hit] = t[hit], index
        index += 1
    if heightfield is not None:
        field_t = np.full(nearest.shape, np.inf)
        for t in np.arange(0.2, float(np.nanmax(np.where(np.isfinite(floor_t), floor_t, 0))) + 0.01, 0.002):
            p = centre + t * d
            hit = np.isinf(field_t) & (p[..., 2] <= heightfield(p[..., 0], p[..., 1]))
            field_t[hit] = t
        hit = field_t < nearest
        nearest[hit], owners[hit] = field_t[hit], index
        index += 1
    depth = nearest.astype(np.float32)          # t is camera-Z because d has unit Z component
    empty = floor_t.astype(np.float32)
    rng = np.random.default_rng(seed)
    if noise_m > 0:
        depth = depth + rng.normal(0.0, noise_m, depth.shape).astype(np.float32)
    if dropout > 0:
        depth[rng.random(depth.shape) < dropout] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    empty[~np.isfinite(empty)] = 0.0
    return depth, empty, [owners == i for i in range(index)]
