"""2.5D height-map grid volume -- the Final Implementation Playbook's locked method.

The playbook (sections 3-8, 10 and 16) freezes exactly one primary volume
method for the final system:

    RealSense metric depth
  + one-time empty-bin reference/floor-plane calibration
  + 2.5D height-map integration over a *physical* XY grid
  + before/after deposit difference for the per-bag contribution

This module implements that method and nothing else. Per section 26 it is
deliberately isolated: it takes depth, intrinsics, a calibrated reference and a
mask, and returns litres plus quality metadata. It imports no camera, detector,
dashboard or configuration module, so it can be unit-tested against synthetic
scenes with exact closed-form ground truth.

Why a physical grid rather than the per-pixel sum this project used before
(`volume.estimate_volume`'s "reference-plane"/"surface-columns" modes): both
integrate `height * pixel_area` once per pixel, with `pixel_area = z^2/(fx*fy)`.
That per-pixel area is the footprint of a ray hitting a surface *perpendicular
to it*. A crumpled polythene bag -- the system's actual target -- is mostly
oblique micro-facets, where the true footprint is larger by 1/cos(theta), so the
per-pixel sum is biased, and every pixel's own depth noise enters the total at
full weight. Binning the backprojected 3-D points into fixed 10-15 mm cells on
the bin floor and taking a robust per-cell height instead makes the cell area an
exact constant and replaces tens of thousands of noisy samples with a few
hundred medians, which is what section 7 prescribes and why it survives wrinkled
surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import CameraIntrinsics

# Depth readings outside this band are not physically meaningful for a bin-side
# RealSense and are dropped before anything else (playbook section 8.7).
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 20.0


@dataclass(frozen=True, slots=True)
class HeightMapSettings:
    """Tunable configuration for the height-map method (playbook section 27).

    The defaults are the playbook's own starting points, not experimentally
    validated constants -- section 27 is explicit that final values must be
    recorded after tuning on the real camera and bin.
    """

    grid_size_m: float = 0.010
    min_height_m: float = 0.005
    max_height_m: float = 0.80
    min_points_per_cell: int = 3
    cell_height_percentile: float = 50.0
    min_valid_depth_fraction: float = 0.70
    fill_hole_cells: bool = True
    max_fill_fraction: float = 0.25
    min_cells: int = 4

    def __post_init__(self) -> None:
        if not 0.002 <= self.grid_size_m <= 0.20:
            raise ValueError("grid_size_m must be between 2 mm and 200 mm")
        if self.min_height_m < 0:
            raise ValueError("min_height_m must not be negative")
        if self.max_height_m <= self.min_height_m:
            raise ValueError("max_height_m must exceed min_height_m")
        if self.min_points_per_cell < 1:
            raise ValueError("min_points_per_cell must be at least 1")
        if not 0.0 <= self.cell_height_percentile <= 100.0:
            raise ValueError("cell_height_percentile must be a percentile")
        if not 0.0 <= self.min_valid_depth_fraction <= 1.0:
            raise ValueError("min_valid_depth_fraction must be a fraction")
        if not 0.0 <= self.max_fill_fraction <= 1.0:
            raise ValueError("max_fill_fraction must be a fraction")
        if self.min_cells < 1:
            raise ValueError("min_cells must be at least 1")


@dataclass(slots=True)
class HeightMapVolume:
    """One occupied-volume reading plus the evidence behind it.

    "Occupied volume" is the playbook's section 3.2 definition: the bin space
    beneath the observed waste surface. It is deliberately not a claim about the
    material volume of the plastic film or about hidden contents.
    """

    liters: float
    volume_m3: float
    cell_area_m2: float
    grid_size_m: float
    cell_count: int
    filled_cells: int
    occupied_area_m2: float
    mean_height_m: float
    max_height_m: float
    height_p90_m: float
    valid_depth_fraction: float
    fill_fraction: float
    quality: str
    rejection_reason: str | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)
    # The per-cell height map behind `liters`, plus where its (0, 0) sits in
    # absolute floor-cell coordinates. Retained so a committed scene can be
    # differenced against a later one; NaN marks a cell nothing was measured in.
    grid: np.ndarray | None = field(default=None, repr=False)
    origin_row: int = 0
    origin_column: int = 0

    @property
    def is_valid(self) -> bool:
        return self.quality == "valid"

    def to_dict(self) -> dict[str, object]:
        return {
            "liters": round(float(self.liters), 4),
            "volume_m3": round(float(self.volume_m3), 8),
            "cell_area_m2": round(float(self.cell_area_m2), 8),
            "grid_size_m": round(float(self.grid_size_m), 5),
            "cell_count": int(self.cell_count),
            "filled_cells": int(self.filled_cells),
            "occupied_area_m2": round(float(self.occupied_area_m2), 6),
            "mean_height_m": round(float(self.mean_height_m), 5),
            "max_height_m": round(float(self.max_height_m), 5),
            "height_p90_m": round(float(self.height_p90_m), 5),
            "valid_depth_fraction": round(float(self.valid_depth_fraction), 4),
            "fill_fraction": round(float(self.fill_fraction), 4),
            "quality": self.quality,
            "rejection_reason": self.rejection_reason,
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class DepositVolume:
    """Incremental occupied volume for one deposit (playbook sections 5 and 10)."""

    added_liters: float
    volume_before_l: float
    volume_after_l: float
    quality: str
    rejection_reason: str | None = None
    valid_depth_fraction: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "added_volume_l": round(float(self.added_liters), 4),
            "volume_before_l": round(float(self.volume_before_l), 4),
            "volume_after_l": round(float(self.volume_after_l), 4),
            "volume_quality": self.quality,
            "rejection_reason": self.rejection_reason,
            "valid_depth_fraction": round(float(self.valid_depth_fraction), 4),
        }


def valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    """Finite, positive, physically plausible depth readings."""
    depth = np.asarray(depth_m, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > MIN_VALID_DEPTH_M) & (depth < MAX_VALID_DEPTH_M)


def median_depth(frames: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray | None:
    """Per-pixel median across a stack of depth frames (playbook sections 4.5, 8.6).

    Invalid readings are excluded per pixel rather than per frame, so a pixel
    that dropped out in a few frames still gets the median of the frames where
    it was seen. A pixel invalid in every frame stays NaN -- the caller must not
    be handed a fabricated distance for a surface the sensor never saw.
    """
    if not frames:
        return None
    stack = np.stack([np.asarray(frame, dtype=np.float64) for frame in frames])
    if stack.ndim != 3:
        return None
    usable = valid_depth_mask(stack)
    if not np.any(usable):
        return None
    masked = np.where(usable, stack, np.nan)
    all_invalid = ~np.any(usable, axis=0)
    with np.errstate(invalid="ignore"):
        combined = np.nanmedian(np.where(all_invalid[None, :, :], 0.0, masked), axis=0)
    combined[all_invalid] = np.nan
    return combined


def build_reference_depth(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    minimum_frames: int = 40,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """One-time empty-bin reference depth map D_ref (playbook section 4).

    Returns the per-pixel median map and a report describing the capture. The
    playbook asks for 40-60 frames; fewer is allowed but reported, because a
    thin reference stack is a real accuracy risk that must not be silent.
    """
    reference = median_depth(frames)
    report: dict[str, object] = {
        "frames": len(frames),
        "sufficient_frames": len(frames) >= minimum_frames,
        "minimum_frames": int(minimum_frames),
    }
    if reference is None:
        report["valid_fraction"] = 0.0
        return None, report
    valid = np.isfinite(reference)
    report["valid_fraction"] = round(float(np.count_nonzero(valid) / valid.size), 4)
    return reference, report


def depth_change_m(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    *,
    region: np.ndarray | None = None,
    percentile: float = 90.0,
) -> float | None:
    """Absolute depth change between two frames (playbook section 8.1).

    Not the mean, which a handful of flickering dropout pixels would keep
    permanently elevated, and not the plain median either: a bag still settling
    usually covers well under half of the bin ROI, so the median reads exactly
    zero while it is visibly moving and would declare the scene stable too
    early. A high percentile ignores isolated dropout flicker while still
    responding to a bag-sized region in motion. Pass `region` to restrict this
    to the bin interior, which is what the percentile is calibrated against.
    """
    if previous is None or current is None:
        return None
    first = np.asarray(previous, dtype=np.float64)
    second = np.asarray(current, dtype=np.float64)
    if first.shape != second.shape:
        return None
    comparable = valid_depth_mask(first) & valid_depth_mask(second)
    if region is not None:
        if region.shape != first.shape:
            return None
        comparable &= region.astype(bool)
    if not np.any(comparable):
        return None
    return float(np.percentile(np.abs(first[comparable] - second[comparable]), percentile))


def scene_is_stable(
    changes: list[float | None] | tuple[float | None, ...],
    *,
    threshold_m: float,
    required_frames: int,
) -> bool:
    """True once the last `required_frames` changes all stayed under threshold.

    This is the section 8.1 trigger that stops V_after being measured while the
    bag is still falling or bouncing.
    """
    if required_frames < 1 or threshold_m <= 0:
        return False
    if len(changes) < required_frames:
        return False
    recent = changes[-required_frames:]
    return all(value is not None and value < threshold_m for value in recent)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane with the given unit normal."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(normal[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def _cell_percentile(
    cell_ids: np.ndarray, heights: np.ndarray, percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-rank percentile of `heights` within each cell (playbook section 7).

    Nearest-rank (rather than an interpolated percentile) keeps every reported
    cell height an actually observed measurement, which is the point of using a
    robust statistic on a wrinkled surface in the first place.
    """
    order = np.lexsort((heights, cell_ids))
    sorted_ids = cell_ids[order]
    sorted_heights = heights[order]
    unique_ids, start_index, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    offsets = np.rint((percentile / 100.0) * (counts - 1)).astype(np.int64)
    return unique_ids, sorted_heights[start_index + offsets], counts


# Hole cells are filled from the lower quartile of their measured neighbours,
# not the median. Most holes next to an object are its own occlusion shadow --
# the strip of bin floor hidden behind the object's near edge -- so the
# neighbourhood straddles a real height discontinuity, and the median resolves
# it to the *object* side. On a 25 cm reference box that fabricated a one-to-two
# cell ring of full-height cells right around the perimeter and inflated the
# reading by 12%; the lower quartile resolves the same neighbourhood to the
# floor side and brought it to 4%. Inside a genuine surface dropout every
# neighbour has a similar height, so the quartile and the median agree there and
# this costs nothing. Erring low is also the right direction for a measurement
# that must never invent occupied volume (playbook section 8.4: fill
# conservatively).
HOLE_FILL_PERCENTILE = 25.0


def _fill_interior_cells(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill only holes enclosed by enough measured neighbours (section 7).

    A cell with at least five of its eight neighbours measured is interior to
    the observed surface, so a neighbour-derived estimate is defensible. A large
    missing region is left missing: cells deep inside it never reach five
    measured neighbours, so nothing is fabricated there.
    """
    filled = grid.copy()
    missing = ~np.isfinite(filled)
    if not np.any(missing):
        return filled, 0
    padded = np.pad(filled, 1, mode="constant", constant_values=np.nan)
    neighbours = np.stack([
        padded[row : row + grid.shape[0], column : column + grid.shape[1]]
        for row in range(3)
        for column in range(3)
        if (row, column) != (1, 1)
    ])
    support = np.count_nonzero(np.isfinite(neighbours), axis=0)
    fillable = missing & (support >= 5)
    if not np.any(fillable):
        return filled, 0
    # Restricted to fillable cells because nanpercentile warns on every all-NaN
    # column it is handed, and most columns here are all-NaN.
    filled[fillable] = np.nanpercentile(
        neighbours[:, fillable], HOLE_FILL_PERCENTILE, axis=0
    )
    return filled, int(np.count_nonzero(fillable))


def integrate_height_map(
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    *,
    plane_coefficients: tuple[float, float, float] | None = None,
    reference_depth_m: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    settings: HeightMapSettings | None = None,
) -> HeightMapVolume | None:
    """Occupied volume by 2.5D height-map integration (playbook sections 6-7).

    Method A (preferred, used whenever `plane_coefficients` is supplied):
    backproject every valid depth pixel to 3-D, take its signed perpendicular
    distance above the calibrated floor plane as the height, and bin the points
    into a physical grid laid out *on that plane*, so the grid is unaffected by
    camera tilt.

    Method B (`reference_depth_m` only): the height is the calibrated depth
    difference `D_ref - D`, and the grid is laid out on the camera's XY plane.
    Section 6 flags this as more sensitive to tilt and perspective; it is the
    fallback for a rig with no fitted plane, and the result carries a
    `method-b-depth-difference` flag so a reader always knows which was used.

    Returns None only when the inputs cannot support a measurement at all. A
    measurement that is computable but untrustworthy comes back with
    `quality == "low_quality"` and a `rejection_reason`, never silently.
    """
    settings = settings or HeightMapSettings()
    if depth_m is None or intrinsics is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        return None
    if plane_coefficients is None and reference_depth_m is None:
        return None

    region = np.ones(depth.shape, dtype=bool)
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != depth.shape:
            return None
        region &= mask_array.astype(bool)
    candidate_pixels = int(np.count_nonzero(region))
    if candidate_pixels == 0:
        return None

    usable = region & valid_depth_mask(depth)
    reference = None
    if reference_depth_m is not None:
        reference = np.asarray(reference_depth_m, dtype=np.float64)
        if reference.shape != depth.shape:
            return None
        usable &= valid_depth_mask(reference)
    valid_depth_fraction = float(np.count_nonzero(usable) / candidate_pixels)
    if not np.any(usable):
        return None

    rows, columns = np.nonzero(usable)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy

    flags: list[str] = []
    if plane_coefficients is not None:
        a, b, c = (float(value) for value in plane_coefficients)
        scale = float(np.sqrt(a * a + b * b + 1.0))
        if not np.isfinite(scale) or scale < 1e-9:
            return None
        # Plane is z = a*x + b*y + c, i.e. a*x + b*y - z + c = 0. A point nearer
        # the camera than the plane sits above the bin floor, so this signed
        # distance is positive exactly there.
        heights = (a * x + b * y + c - z) / scale
        normal = np.array([a, b, -1.0]) / scale
        first_axis, second_axis = _plane_basis(normal)
        points = np.stack([x, y, z], axis=1)
        grid_u = points @ first_axis
        grid_v = points @ second_axis
    else:
        assert reference is not None
        heights = reference[rows, columns] - z
        flags.append("method-b-depth-difference")
        grid_u = x
        grid_v = y

    plausible = np.isfinite(heights) & (heights <= settings.max_height_m)
    if not np.any(plausible):
        return None
    heights = np.clip(heights[plausible], 0.0, None)
    grid_u = grid_u[plausible]
    grid_v = grid_v[plausible]

    # A height map must never be finer than the depth image that feeds it.
    # One pixel covers roughly `median(z)/fx` metres on the surface, so a cell
    # of side `g` can hold about `(g / spacing)^2` samples; asking for a 10 mm
    # cell from a camera whose pixels already span 17 mm leaves most cells empty
    # or single-sampled, and `min_points_per_cell` then discards the object
    # entirely. Growing the cell to whatever the sampling can actually support
    # keeps the configured size on real hardware (a RealSense at ~1.2 m has
    # ~2 mm pixels, far finer than 10 mm) while degrading gracefully on a
    # coarse, distant or low-resolution view instead of returning nothing.
    sample_spacing_m = float(np.median(z[plausible])) / max(intrinsics.fx, 1e-9)
    cell = max(settings.grid_size_m, sample_spacing_m * np.sqrt(settings.min_points_per_cell))
    samples_per_cell = (cell / sample_spacing_m) ** 2 if sample_spacing_m > 0 else np.inf
    min_points = max(1, min(settings.min_points_per_cell, int(samples_per_cell)))
    if cell > settings.grid_size_m * 1.001:
        flags.append("grid-coarsened-to-depth-resolution")

    column_index = np.floor(grid_u / cell).astype(np.int64)
    row_index = np.floor(grid_v / cell).astype(np.int64)
    # Absolute cell indices on the calibrated floor, kept so two grids captured
    # at different times can be aligned and differenced (see `align_grids`).
    origin_column = int(column_index.min())
    origin_row = int(row_index.min())
    column_index -= origin_column
    row_index -= origin_row
    width = int(column_index.max()) + 1
    cell_ids = row_index * width + column_index

    unique_ids, cell_heights, counts = _cell_percentile(
        cell_ids, heights, settings.cell_height_percentile
    )
    supported = counts >= min_points
    if not np.any(supported):
        return None
    unique_ids = unique_ids[supported]
    cell_heights = cell_heights[supported]

    height_grid = np.full((int(row_index.max()) + 1, width), np.nan)
    height_grid[unique_ids // width, unique_ids % width] = cell_heights
    filled_cells = 0
    if settings.fill_hole_cells:
        height_grid, filled_cells = _fill_interior_cells(height_grid)

    measured = np.isfinite(height_grid)
    # The noise floor (section 8.8) drops cells whose height is indistinguishable
    # from the bin floor, so an empty bin integrates to zero rather than to the
    # accumulated sensor noise over its whole area.
    occupied = measured & (height_grid >= settings.min_height_m)
    cell_count = int(np.count_nonzero(occupied))
    if cell_count < settings.min_cells:
        return None

    occupied_heights = height_grid[occupied]
    cell_area_m2 = cell * cell
    volume_m3 = float(np.sum(occupied_heights) * cell_area_m2)
    fill_fraction = filled_cells / max(1, int(np.count_nonzero(measured)))

    quality = "valid"
    rejection_reason: str | None = None
    if valid_depth_fraction < settings.min_valid_depth_fraction:
        quality = "low_quality"
        rejection_reason = (
            f"only {valid_depth_fraction * 100:.0f}% of the measured region has valid depth "
            f"(minimum {settings.min_valid_depth_fraction * 100:.0f}%)"
        )
    elif fill_fraction > settings.max_fill_fraction:
        quality = "low_quality"
        rejection_reason = (
            f"{fill_fraction * 100:.0f}% of the measured cells were interpolated "
            f"(maximum {settings.max_fill_fraction * 100:.0f}%)"
        )

    return HeightMapVolume(
        liters=volume_m3 * 1000.0,
        volume_m3=volume_m3,
        cell_area_m2=cell_area_m2,
        grid_size_m=cell,
        cell_count=cell_count,
        filled_cells=filled_cells,
        occupied_area_m2=cell_count * cell_area_m2,
        mean_height_m=float(np.mean(occupied_heights)),
        max_height_m=float(np.max(occupied_heights)),
        height_p90_m=float(np.percentile(occupied_heights, 90)),
        valid_depth_fraction=valid_depth_fraction,
        fill_fraction=fill_fraction,
        quality=quality,
        rejection_reason=rejection_reason,
        flags=tuple(flags),
        grid=np.where(occupied, height_grid, np.where(measured, 0.0, np.nan)),
        origin_row=origin_row,
        origin_column=origin_column,
    )


def align_grids(
    before: HeightMapVolume, after: HeightMapVolume,
) -> tuple[np.ndarray, np.ndarray]:
    """Two height grids on a common absolute floor-cell frame.

    Cells either grid never measured come back NaN, so a caller can tell
    "nothing there" from "never seen".
    """
    top = min(before.origin_row, after.origin_row)
    left = min(before.origin_column, after.origin_column)
    bottom = max(
        before.origin_row + before.grid.shape[0], after.origin_row + after.grid.shape[0]
    )
    right = max(
        before.origin_column + before.grid.shape[1],
        after.origin_column + after.grid.shape[1],
    )
    shape = (bottom - top, right - left)
    canvases = []
    for item in (before, after):
        canvas = np.full(shape, np.nan)
        row = item.origin_row - top
        column = item.origin_column - left
        canvas[row : row + item.grid.shape[0], column : column + item.grid.shape[1]] = item.grid
        canvases.append(canvas)
    return canvases[0], canvases[1]


@dataclass(slots=True)
class IncrementalDeposit:
    """What the bin gained from one deposit, measured against the committed scene."""

    added_liters: float
    displaced_liters: float
    changed_area_m2: float
    quality: str
    rejection_reason: str | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return self.quality == "valid"

    def to_dict(self) -> dict[str, object]:
        return {
            "added_volume_l": round(float(self.added_liters), 4),
            "displaced_volume_l": round(float(self.displaced_liters), 4),
            "changed_area_m2": round(float(self.changed_area_m2), 6),
            "volume_quality": self.quality,
            "rejection_reason": self.rejection_reason,
            "flags": list(self.flags),
        }


def incremental_deposit(
    committed: HeightMapVolume | None,
    current: HeightMapVolume | None,
    *,
    min_change_m: float = 0.015,
    min_changed_area_m2: float = 0.004,
    max_displaced_fraction: float = 0.35,
    max_added_l: float = 90.0,
) -> IncrementalDeposit | None:
    """Volume of the newly arrived object only, per-cell against the last commit.

    This is what stops two touching bags being reported as one 29 L object. The
    detector and the segmentation mask both merge adjacent same-coloured bags --
    no colour or class rule can separate identical black polythene -- so the
    separation is done on physical geometry instead: every cell the first bag
    occupies already holds its height in the committed grid, so differencing
    leaves only the cells the second bag actually raised.

    Semantics, kept deliberately distinct:
      * `added_liters` -- this deposit's own contribution (sum of *positive*
        per-cell change). This is what a new ledger entry records.
      * `displaced_liters` -- volume that went *down* since the commit. An
        arriving bag squashing the pile slightly is normal; a large drop means
        the old pile was moved or removed, and then the positive cells are not
        a new deposit at all but the same material somewhere else, which would
        double-count. That case is rejected rather than guessed.

    Only positive change is summed, never the whole scene, so a committed
    deposit can never be counted a second time.
    """
    if committed is None or current is None:
        return None
    if committed.grid is None or current.grid is None:
        return None
    before, after = align_grids(committed, current)
    # A cell neither grid measured contributes nothing; one measured on only one
    # side is treated as zero height there rather than unknown, since the floor
    # is the calibrated reference.
    seen = np.isfinite(before) | np.isfinite(after)
    if not np.any(seen):
        return None
    before = np.where(np.isfinite(before), before, 0.0)
    after = np.where(np.isfinite(after), after, 0.0)
    delta = np.where(seen, after - before, 0.0)

    cell_area_m2 = current.cell_area_m2
    risen = delta >= min_change_m
    fallen = delta <= -min_change_m
    added_liters = float(np.sum(delta[risen]) * cell_area_m2 * 1000.0)
    displaced_liters = float(-np.sum(delta[fallen]) * cell_area_m2 * 1000.0)
    changed_area_m2 = float(np.count_nonzero(risen) * cell_area_m2)

    flags: list[str] = []
    if displaced_liters > 0:
        flags.append("existing-contents-settled")
    if changed_area_m2 < min_changed_area_m2:
        return IncrementalDeposit(
            0.0, displaced_liters, changed_area_m2, "rejected",
            "new_deposit_not_isolatable", tuple(flags),
        )
    if displaced_liters > max(0.2, added_liters * max_displaced_fraction):
        return IncrementalDeposit(
            0.0, displaced_liters, changed_area_m2, "rejected",
            "possible_existing_object_movement", tuple(flags),
        )
    if added_liters > max_added_l:
        return IncrementalDeposit(
            0.0, displaced_liters, changed_area_m2, "rejected",
            f"added volume {added_liters:.1f} L exceeds the plausible deposit bound",
            tuple(flags),
        )
    return IncrementalDeposit(
        added_liters, displaced_liters, changed_area_m2, "valid", None, tuple(flags),
    )


def added_volume(
    before: HeightMapVolume | None,
    after: HeightMapVolume | None,
    *,
    negative_tolerance_l: float = 0.5,
    max_added_l: float = 90.0,
) -> DepositVolume | None:
    """Per-deposit incremental occupied volume (playbook sections 5, 10 and 16).

    `V_bag = V_after - V_before`, with the section 16 rejection rules applied:
    a strongly negative difference, a physically impossible one, or either side
    being low quality produces a flagged result rather than a confident litre
    value. A small negative difference is sensor noise around an unchanged
    scene and is reported as zero.
    """
    if before is None or after is None:
        return None
    difference = after.liters - before.liters
    coverage = min(before.valid_depth_fraction, after.valid_depth_fraction)
    quality = "valid"
    rejection_reason: str | None = None
    if before.quality != "valid" or after.quality != "valid":
        quality = "low_quality"
        rejection_reason = after.rejection_reason or before.rejection_reason or "low-quality depth"
    elif difference < -abs(negative_tolerance_l):
        quality = "rejected"
        rejection_reason = (
            f"occupied volume fell by {abs(difference):.2f} L across the deposit; "
            "the bin was disturbed or the reference is stale"
        )
    elif difference > max_added_l:
        quality = "rejected"
        rejection_reason = (
            f"added volume {difference:.1f} L exceeds the plausible deposit bound "
            f"of {max_added_l:.0f} L"
        )
    return DepositVolume(
        added_liters=max(0.0, difference),
        volume_before_l=before.liters,
        volume_after_l=after.liters,
        quality=quality,
        rejection_reason=rejection_reason,
        valid_depth_fraction=coverage,
    )
