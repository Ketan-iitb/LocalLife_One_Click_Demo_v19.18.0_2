"""Tell a ball from a can: the one refinement the shape router needed.

`shape_geometry.measure_shape` (protected, unchanged) routes each object to a
cuboid, cylinder, dome or height-map volume from its support-plane points,
with the label only admitting a candidate. On ray-traced scenes it separated
cans, cartons, boxes and flat bags correctly -- but a ball was routed to
*cylinder* (a round footprint and a height equal to its width look like a
short can), which reports pi r^2 h: 5.04 L for a 4.19 L ball.

What separates them is the top, seen from the floor. Every support-plane
point is dropped into 5 mm floor cells and each cell keeps its highest point.
A flat-topped object (can, box) has its top at full height over essentially
its whole footprint -- 95-100 % of the cells in ray-traced scenes at 15-60
degrees of tilt. A ball reaches full height only over a small central cap --
30-38 % at every tilt. That share does not depend on the viewing angle, which
is what made it usable where a radial height profile was not.

A domed top alone could still be a filled bag, so a sphere also has to look
like one: its footprint must fit inside a circle about its own height across
around the cap, it must be about as wide as it is tall, and the cell heights
near the cap must follow a sphere's surface. Anything else keeps the route
measure_shape gave it. The label is never consulted.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .shape_geometry import CYLINDER, ELLIPSOID, IRREGULAR_RIGID, SPHERE, ShapeGeometry

CELL_M = 0.005
MIN_CELLS = 30
MAX_PLATEAU_FOR_DOME = 0.60     # share of footprint cells at >= 93 % of full height
MAX_OUTSIDE_SHARE = 0.05        # cells further than 0.6 x height from the cap centre
MAX_CAP_RESIDUAL = 0.08         # median |height - sphere surface| / radius near the cap
WIDTH_TO_HEIGHT = (0.80, 1.25)
CANDIDATES = (CYLINDER, ELLIPSOID, IRREGULAR_RIGID)


def sphere_evidence(footprint_m: np.ndarray, heights_m: np.ndarray) -> dict | None:
    """Plateau share, compactness, width/height and cap fit of one object's points."""
    footprint = np.asarray(footprint_m, dtype=np.float64)
    heights = np.asarray(heights_m, dtype=np.float64)
    ok = np.all(np.isfinite(footprint), axis=1) & np.isfinite(heights) & (heights > 0)
    footprint, heights = footprint[ok], heights[ok]
    if len(footprint) < MIN_CELLS:
        return None
    full = float(np.percentile(heights, 99))
    keys = np.floor(footprint / CELL_M).astype(np.int64)
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    if len(cells) < MIN_CELLS or full <= 0:
        return None
    tops = np.full(len(cells), -np.inf)
    np.maximum.at(tops, inverse.ravel(), heights)
    centres = (cells + 0.5) * CELL_M
    plateau = tops >= 0.93 * full
    cap = centres[plateau].mean(axis=0)
    distance = np.linalg.norm(centres - cap, axis=1)
    radius = full / 2.0
    near = distance < 0.8 * radius
    surface = radius + np.sqrt(np.clip(radius * radius - distance[near] ** 2, 0.0, None))
    low, high = np.percentile(footprint, 2, axis=0), np.percentile(footprint, 98, axis=0)
    return {
        "full_height_m": full,
        "plateau_share": float(plateau.mean()),
        "outside_share": float((distance > 0.6 * full).mean()),
        "width_to_height": float(np.max(high - low)) / full,
        "cap_residual": float(np.median(np.abs(tops[near] - surface))) / radius if near.any() else 1.0,
    }


def refine_shape(shape: ShapeGeometry, footprint_m: np.ndarray, heights_m: np.ndarray) -> ShapeGeometry:
    """`shape`, or a sphere when its points show a dome, a ball's extent and a sphere's cap."""
    if shape is None or shape.geometry_method not in CANDIDATES:
        return shape
    evidence = sphere_evidence(footprint_m, heights_m)
    if evidence is None:
        return shape
    features = {**shape.features, **{f"sphere_{key}": value for key, value in evidence.items()}}
    is_sphere = (
        evidence["plateau_share"] <= MAX_PLATEAU_FOR_DOME
        and evidence["outside_share"] <= MAX_OUTSIDE_SHARE
        and WIDTH_TO_HEIGHT[0] <= evidence["width_to_height"] <= WIDTH_TO_HEIGHT[1]
        and evidence["cap_residual"] <= MAX_CAP_RESIDUAL
    )
    if not is_sphere:
        return replace(shape, features=features)
    # The radius from the height: the top above the floor is seen whole from
    # any angle, while the footprint of the visible surface falls short of
    # the diameter along the viewing direction.
    radius_mm = evidence["full_height_m"] * 1000.0 / 2.0
    litres = 4.0 / 3.0 * math.pi * (radius_mm / 1000.0) ** 3 * 1000.0
    return replace(
        shape, geometry_method=SPHERE, length_mm=2 * radius_mm, width_mm=2 * radius_mm,
        height_mm=2 * radius_mm, radius_mm=radius_mm, selected_volume_litres=litres,
        cylinder_volume_litres=None, cylinder_diameter_mm=None, cylinder_height_mm=None,
        volume_meaning="ideal sphere envelope, 4/3 pi r^3, from the fitted radius",
        flags=tuple(shape.flags) + ("dome_top_routed_to_sphere",), features=features,
    )
