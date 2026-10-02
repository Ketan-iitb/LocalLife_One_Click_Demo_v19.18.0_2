"""Bin fill level from one camera's own measured pose.

Each camera has its OWN fill profile (tripods differ): the tape-measured
distance from its optical centre to the EMPTY bin floor, whether that distance
is vertical or along the tilted optical axis, the tilt from vertical, and the
usable floor-to-rim height. Nothing is shared between cameras and nothing is
derived from the bin's nominal capacity.

From calibrated metric depth, each valid pixel inside the bin region gets a
height above the empty floor:  h = H + up . P, with P the back-projected
point, H the vertical optical-centre height and up = (0, -sin b, -cos b) for a
camera pitched b degrees from vertical (roll assumed zero). Pixels are pooled
into 5 cm floor cells (the cell's top surface). The reported quantities are
kept apart and labelled:

* maximum reliable fill height -- 95th percentile of the cell tops;
* height-based fill %          -- that height / usable height;
* rough litres                 -- capacity x that fraction, which assumes the
                                  bin is filled roughly evenly;
* estimated occupied volume    -- mean cell height x measured inner floor area,
                                  only when the interior is measured and most of
                                  it is seen.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CELL_M = 0.05
RIM_TOLERANCE_M = 0.15          # above rim + this: rim, tripod or outside the bin
MIN_COVERAGE = 0.30              # of bin-region pixels with valid depth
OCCUPIED_MIN_CELL_SHARE = 0.60   # of the measured inner floor area seen
BELOW_FLOOR_M = 0.10            # a pixel this far below the floor reference is not a surface
BELOW_FLOOR_SHARE = 0.05         # more than this share of such pixels -> visible quality warning
INCONSISTENT_M = 0.15            # a single frame this far from the recent median is not trusted
SMOOTH_FRAMES = 5
FLOOR_NOISE_M = 0.03             # surface this close to the fitted floor is floor
STABLE_MOTION = 0.02             # scene motion fraction below which a frame is "settled"
MOVED_EDGE_CORRELATION = 0.45    # thumbnail edge correlation outside the bin below this: camera moved
HEIGHT_FILL_LABEL = "rough height-based equivalent; assumes roughly uniform filling"
OCCUPIED_LABEL = "estimated occupied volume over the measured bin floor area (visible surface; voids unseen)"
# Installation defaults. The operator measured ~110 cm from ONE camera to the
# empty floor and estimated the camera sits ~10 cm above the rim (so ~100 cm
# usable). Neither pose nor tilt was measured: a profile built from these is
# "approximate", never "validated".
DEFAULT_FLOOR_DISTANCE_M = 1.10
DEFAULT_USABLE_HEIGHT_M = 1.00
DEFAULT_ABOVE_RIM_M = 0.10


@dataclass
class FillProfile:
    camera_id: str
    camera_to_empty_floor_m: float | None = None
    distance_kind: str = "unknown"            # "vertical" | "optical_axis" | "unknown"
    tilt_from_vertical_deg: float | None = None
    usable_height_m: float | None = None      # empty floor -> rim, measured
    camera_above_rim_m: float | None = None
    inner_length_m: float | None = None
    inner_width_m: float | None = None
    capacity_l: float = 660.0
    capacity_verified: bool = False
    saved_at: float | None = None
    pose_edges: list[int] | None = None       # 64x48 edge thumbnail outside the bin at save time
    floor_plane: list[float] | None = None    # (a, b, c): z = a x + b y + c, this camera's own empty floor
    floor_scale: float = 1.0                  # Logitech: model depth -> metres from the 110 cm reference
    floor_fitted_at: float | None = None
    floor_raw_distance: float | None = None   # in this camera's own depth units (the floor is the DEEPEST plane)
    outside_reference: float | None = None    # median raw depth OUTSIDE the bin region at fit time (walls/rim)
    source: str = "none"                      # "saved" | "default-provisional" | "none"
    notes: list[str] | None = None            # where each default came from

    def assumptions(self) -> list[str]:
        """What the reading assumes because it was not measured (approximate profile)."""
        items = list(self.notes or [])
        if self.tilt_from_vertical_deg is None:
            items.append("tilt not measured: camera assumed to look straight down")
        if self.distance_kind not in ("vertical", "optical_axis"):
            items.append("floor distance assumed vertical (not stated whether vertical or along the line of sight)")
        if not self.capacity_verified:
            items.append("capacity: nominal 660 L, not checked on the bin label")
        return items

    @property
    def status(self) -> str:
        if self.blocking():
            return "unavailable"
        exact = self.tilt_from_vertical_deg is not None and self.distance_kind in ("vertical", "optical_axis")
        return "measured" if exact and self.source == "saved" and not self.notes else "approximate"

    def blocking(self) -> list[str]:
        """Missing values without which no height can be computed at all."""
        issues = []
        if self.camera_to_empty_floor_m is None or self.camera_to_empty_floor_m <= 0:
            issues.append("camera-to-empty-floor distance not measured for this camera")
        if self.usable_height_m is None or self.usable_height_m <= 0:
            issues.append("usable floor-to-rim height not set")
        if self.tilt_from_vertical_deg is not None and not 0 <= self.tilt_from_vertical_deg < 80:
            issues.append("tilt must be 0-80 deg from vertical")
        if not issues and self.usable_height_m >= self.vertical_height_m():
            issues.append("usable height is not below the camera: re-check both measurements")
        return issues

    def vertical_height_m(self) -> float | None:
        """Vertical optical-centre height; unknown tilt/kind -> the approximate straight-down reading."""
        if self.camera_to_empty_floor_m is None:
            return None
        if self.distance_kind == "optical_axis" and self.tilt_from_vertical_deg is not None:
            return self.camera_to_empty_floor_m * math.cos(math.radians(self.tilt_from_vertical_deg))
        return self.camera_to_empty_floor_m

    def problems(self) -> list[str]:
        issues = []
        if self.camera_to_empty_floor_m is None or self.camera_to_empty_floor_m <= 0:
            issues.append("camera-to-empty-floor distance not measured")
        if self.distance_kind not in ("vertical", "optical_axis"):
            issues.append("say whether the camera-to-floor distance is vertical or along the optical axis")
        if self.tilt_from_vertical_deg is None or not 0 <= self.tilt_from_vertical_deg < 80:
            issues.append("camera tilt from vertical not measured (0-80 deg)")
        if self.usable_height_m is None or self.usable_height_m <= 0:
            issues.append("usable floor-to-rim height not measured")
        vertical = self.vertical_height_m()
        if not issues and vertical is not None and self.usable_height_m >= vertical:
            issues.append("usable height is not below the camera: re-check both measurements")
        if (not issues and self.camera_above_rim_m is not None and vertical is not None
                and abs(vertical - (self.usable_height_m + self.camera_above_rim_m)) > 0.05):
            issues.append(f"inconsistent: camera height {vertical:.2f} m vs rim {self.usable_height_m:.2f} m + "
                          f"offset {self.camera_above_rim_m:.2f} m (more than 5 cm apart)")
        return issues


def heights_above_floor(depth_m: np.ndarray, intrinsics: Any, profile: FillProfile,
                        region: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(height above empty floor, horizontal x, horizontal y) for valid pixels inside `region`."""
    vertical = profile.vertical_height_m()
    tilt = math.radians(profile.tilt_from_vertical_deg or 0.0)
    rows, cols = np.nonzero((depth_m > 0.05) & np.isfinite(depth_m) & (True if region is None else region))
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols - intrinsics.ppx) * z / intrinsics.fx
    y = (rows - intrinsics.ppy) * z / intrinsics.fy
    up = (0.0, -math.sin(tilt), -math.cos(tilt))
    forward = (0.0, math.cos(tilt), -math.sin(tilt))
    height = vertical + up[1] * y + up[2] * z
    horizontal_y = forward[1] * y + forward[2] * z
    return height, x, horizontal_y


def cell_tops(height: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> dict[tuple[int, int], float]:
    """Per 5 cm floor cell, the 90th-percentile height (one noisy pixel does not set a cell's top)."""
    if height.size == 0:
        return {}
    ix, iy = np.floor(hx / CELL_M).astype(np.int64), np.floor(hy / CELL_M).astype(np.int64)
    keys = (ix + 100_000) * 1_000_000 + (iy + 100_000)
    order = np.lexsort((height, keys))
    keys, values = keys[order], height[order]
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    ends = np.r_[starts[1:], len(keys)]
    cells: dict[tuple[int, int], float] = {}
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue                                # a lone pixel is not a surface
        top = float(values[a + int(0.9 * (b - a - 1))])
        key = int(keys[a])
        cells[(key // 1_000_000 - 100_000, key % 1_000_000 - 100_000)] = top
    return cells


def _close_gaps(cells: dict[tuple[int, int], float], cells_wide: int = 3) -> dict[tuple[int, int], float]:
    """Grey closing (15 cm): a crevice narrower than that between bags is not usable space.

    Wider open floor stays open, so an empty or half-empty bin is not filled in.
    """
    if len(cells) < 9:
        return cells
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return cells
    keys = np.array(list(cells.keys()))
    x0, y0 = keys.min(axis=0)
    grid = np.zeros(tuple(keys.max(axis=0) - (x0, y0) + 1)[::-1], np.float32)
    grid[keys[:, 1] - y0, keys[:, 0] - x0] = np.fromiter(cells.values(), dtype=np.float32)
    kernel = np.ones((cells_wide, cells_wide), np.uint8)
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)
    return {k: float(max(cells[k], closed[k[1] - y0, k[0] - x0])) for k in cells}


def heights_above_plane(depth_m: np.ndarray, intrinsics: Any, coefficients, region=None):
    """(height above the saved empty-bin floor plane, x, y) -- the measured pose, when one exists."""
    a, b, c = (float(v) for v in coefficients)
    rows, cols = np.nonzero((depth_m > 0.05) & np.isfinite(depth_m) & (True if region is None else region))
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols - intrinsics.ppx) * z / intrinsics.fx
    y = (rows - intrinsics.ppy) * z / intrinsics.fy
    return (a * x + b * y + c - z) / math.sqrt(a * a + b * b + 1.0), x, y


def fill_reading(cells: dict[tuple[int, int], float], coverage: float, profile: FillProfile,
                 fill_m: float | None = None) -> dict[str, Any]:
    """The labelled fill quantities from one settled grid of cell tops (heights in metres)."""
    usable = float(profile.usable_height_m)
    # Cells above the rim are waste heaped over it: the height fill saturates at 100 %.
    tops = np.clip(np.fromiter(_close_gaps(cells).values(), dtype=np.float64), 0.0, usable)
    # Fill = mean waste-surface height over the visible bin floor: a few bags in an empty bin
    # give a few per cent, not the height of the tallest bag.
    fill = float(tops.mean()) if fill_m is None else float(min(max(fill_m, 0.0), usable))
    fraction = fill / usable
    reading = {
        "status": "ok", "reason": None, "coverage_pct": round(100 * coverage, 1),
        "max_fill_height_cm": round(fill * 100, 1), "usable_height_cm": round(usable * 100, 1),
        "tallest_cm": round(float(np.percentile(tops, 95)) * 100, 1),
        "height_fill_pct": round(100 * fraction, 1),
        "remaining_height_cm": round((usable - fill) * 100, 1),
        "rough_litres": round(profile.capacity_l * fraction, 1),
        "rough_remaining_litres": round(profile.capacity_l * (1 - fraction), 1),
        "rough_litres_label": HEIGHT_FILL_LABEL,
        "capacity_l": profile.capacity_l,
        "capacity_note": "verified on the bin label" if profile.capacity_verified else "nominal 660 L, unverified",
        "occupied_l": None, "occupied_pct": None, "occupied_label": OCCUPIED_LABEL,
    }
    if profile.inner_length_m and profile.inner_width_m:
        area = profile.inner_length_m * profile.inner_width_m
        seen = len(cells) * CELL_M * CELL_M / area
        if seen >= OCCUPIED_MIN_CELL_SHARE:
            mean = float(tops.mean())
            reading["occupied_l"] = round(mean * area * 1000.0, 1)
            reading["occupied_pct"] = round(100 * mean / usable, 1)
        else:
            reading["occupied_reason"] = f"only {seen:.0%} of the measured bin floor area is visible"
    else:
        reading["occupied_reason"] = "inner bin length and width not measured"
    return reading


def edge_thumbnail(frame: np.ndarray, region: np.ndarray | None) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(cv2.GaussianBlur(grey, (5, 5), 0), (64, 48), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 40, 120).astype(np.float32)
    if region is not None:
        outside = cv2.resize(region.astype(np.uint8), (64, 48), interpolation=cv2.INTER_NEAREST) == 0
        edges[~outside] = 0.0        # waste inside the bin changes; walls and rim do not
    return edges


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


class FillEstimator:
    """Per-camera fill reading, refreshed only from settled frames."""

    def __init__(self, camera_id: str, directory: Path, default: FillProfile | None = None) -> None:
        self.camera_id = camera_id
        self.path = Path(directory) / f"fill_{camera_id}.json"
        self.last_processed_at: float | None = None
        self.last_valid_at: float | None = None
        self.default = default or FillProfile(camera_id=camera_id)
        self.profile = self._load()
        self.reading: dict[str, Any] = self._na("no settled frame processed yet")
        self.history: deque[tuple[float, dict[tuple[int, int], float]]] = deque(maxlen=60)
        self.fill_history: deque[tuple[float, float]] = deque(maxlen=SMOOTH_FRAMES)
        self._last_fit_try = -1e9
        self._last_refit = -1e9
        self.last_maps: tuple[Any, float] = (None, 1.0)
        self.inconsistent = 0

    def _load(self) -> FillProfile:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            saved = FillProfile(**{k: v for k, v in data.items() if k in FillProfile.__dataclass_fields__})
            saved.source = data.get("source") or "saved"     # an existing file always wins over the defaults
            return saved
        except FileNotFoundError:
            # First start: persist the defaults ONCE so later runs show the same values;
            # an existing file (saved or defaulted) is never overwritten by defaults.
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(asdict(self.default)), encoding="utf-8")
            except OSError:
                pass
            return self.default
        except (OSError, ValueError, TypeError):
            return self.default

    def save_profile(self, profile: FillProfile, frame: np.ndarray | None, region: np.ndarray | None) -> list[str]:
        problems = profile.problems()
        if frame is not None:
            edges = edge_thumbnail(frame, region)
            profile.pose_edges = None if edges is None else edges.astype(np.uint8).ravel().tolist()
        profile.saved_at = time.time()
        profile.source = "saved"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(profile)), encoding="utf-8")
        os.replace(temporary, self.path)
        self.profile = profile
        self.history.clear()
        self.fill_history.clear()
        self.reading = self._na("profile saved; waiting for a settled frame")
        return problems

    def _decorate(self, reading: dict[str, Any]) -> dict[str, Any]:
        reading["diagnostics"] = dict(getattr(self, "last_diag", {}) or {})
        reading.update(profile_status=self.profile.status, profile_source=self.profile.source,
                       assumptions=self.profile.assumptions(), last_processed_at=self.last_processed_at,
                       last_valid_at=self.last_valid_at)
        return reading

    def _na(self, reason: str) -> dict[str, Any]:
        return self._decorate({"status": "na", "reason": reason, "updated_at": None, "camera_id": self.camera_id,
                "rough_litres_label": HEIGHT_FILL_LABEL, "occupied_label": OCCUPIED_LABEL,
                "capacity_l": self.profile.capacity_l,
                "capacity_note": "verified on the bin label" if self.profile.capacity_verified
                else "nominal 660 L, unverified"})

    def camera_moved(self, frame: np.ndarray, region: np.ndarray | None) -> bool:
        if not self.profile.pose_edges:
            return False
        now = edge_thumbnail(frame, region)
        if now is None:
            return False
        saved = np.asarray(self.profile.pose_edges, dtype=np.float32).reshape(48, 64)
        return _correlation(saved, now) < MOVED_EDGE_CORRELATION

    def update(self, frame: np.ndarray, depth_m: np.ndarray | None, intrinsics: Any, region: np.ndarray | None,
               motion: float | None, timestamp: float, *, depth_reason: str | None = None,
               floor_plane=None, depth_label: str = "hardware depth",
               objects: np.ndarray | None = None) -> dict[str, Any]:
        self.last_processed_at = timestamp
        problems = self.profile.blocking()
        if problems:
            return self._hold("fill profile incomplete: " + "; ".join(problems))
        if depth_m is None or intrinsics is None:
            return self._hold(depth_reason or "no metric depth for this camera")
        if self.camera_moved(frame, region):
            return self._hold("camera or bin moved since the fill profile was saved: re-save it")
        # Moving scenes still refresh the surface map (per-bag volumes need it); only the FILL
        # reading waits for a settled frame.
        moving = motion is not None and motion > STABLE_MOTION
        step = max(1, int(depth_m.shape[1] // 160))
        objects = None if objects is None else objects[::step, ::step]
        depth_m, intrinsics, region = _subsample(depth_m, intrinsics, region)
        if objects is not None and not objects.any() and not moving and timestamp - self._last_refit >= 30.0:
            # Nothing detected in the bin: the deepest flat surface is the floor -- refresh it (plug and play).
            self._last_refit = timestamp
            self.recalibrate(depth_m, intrinsics, region, automatic=True)
        self.last_drift = self.drift(depth_m, region)
        depth_m = depth_m / self.last_drift
        plane, scale = floor_plane, 1.0
        if plane is not None:
            geometry = "saved empty-bin floor plane"
        else:
            if self.profile.floor_plane is None and not moving and timestamp - self._last_fit_try >= 5.0:
                self._last_fit_try = timestamp
                self.recalibrate(depth_m, intrinsics, region, automatic=True)
            plane, scale = self.profile.floor_plane, float(self.profile.floor_scale or 1.0)
            geometry = "auto-detected bin floor" if plane is not None else "camera assumed looking straight down"
        height, flat, valid = surface_maps(depth_m, intrinsics, self.profile, plane, scale)
        if region is not None:
            valid &= region
        flat_share = float(np.count_nonzero(flat & valid)) / max(1, int(np.count_nonzero(valid)))
        if flat_share < 0.30:
            # Real stereo noise can tilt most normals past the wall test; then the test is not
            # informative, and blocking the whole fill on it was wrong (fill stayed "unavailable").
            flat = np.ones_like(flat)
        z_all = depth_m * scale
        rows_, cols_ = np.indices(depth_m.shape)
        self.last_surface = {"at": timestamp, "step": step, "height": height, "ok": valid & flat, "valid": valid,
                             "area": (z_all / intrinsics.fx) * (z_all / intrinsics.fy),
                             "x": (cols_ - intrinsics.ppx) * z_all / intrinsics.fx,
                             "y": (rows_ - intrinsics.ppy) * z_all / intrinsics.fy, "z": z_all,
                             "up": self._up_vector(plane)}
        a_, b_, c_ = plane if plane is not None else (0.0, 0.0, None)
        self.last_diag = {
            "floor_source": geometry, "depth_source": depth_label,
            "floor_distance_cm": None if c_ is None else round(abs(c_) / math.sqrt(a_ * a_ + b_ * b_ + 1) * scale * 100, 1),
            "tilt_deg": round(math.degrees(math.atan(math.hypot(a_, b_))), 1) if plane is not None
            else self.profile.tilt_from_vertical_deg,
            "depth_scale": round(scale, 3), "scale_drift": round(float(self.last_drift), 3),
            "valid_depth_pct": round(100.0 * np.count_nonzero(valid) / max(1, valid.size if region is None
                                                                           else int(np.count_nonzero(region))), 1),
            "statistic": "mean over 5 cm floor cells of the surface top (p90 per cell), gaps < 15 cm closed; < 3 cm = floor",
            "usable_height_cm": round(self.profile.usable_height_m * 100, 1), "frame_at": timestamp,
            "upward_surface_pct": round(100 * flat_share, 1),
            "wall_filter": "on" if flat_share >= 0.30 else "off (normals too noisy)",
        }
        if moving:
            return self._hold("scene moving", stale_only=True)
        usable = self.profile.usable_height_m
        keep = valid & flat & (height > -BELOW_FLOOR_M) & (height < usable + RIM_TOLERANCE_M)
        coverage = float(np.count_nonzero(keep)) / max(1, int(np.count_nonzero(valid)))
        self.last_diag.update(surface_coverage_pct=round(100 * coverage, 1),
                              detected_area_pct=None if objects is None or not objects.size
                              else round(100.0 * np.count_nonzero(objects) / objects.size, 1))
        warnings: list[str] = []
        below_share = float(np.count_nonzero(valid & (height < -BELOW_FLOOR_M))) / max(1, int(np.count_nonzero(valid)))
        if below_share > BELOW_FLOOR_SHARE:
            warnings.append(f"{below_share:.0%} of depth below the floor reference")   # API only, not shown
        n_valid = max(1, int(np.count_nonzero(valid)))
        below_share = float(np.count_nonzero(valid & (height < -BELOW_FLOOR_M))) / n_valid
        above_share = float(np.count_nonzero(valid & (height > usable + RIM_TOLERANCE_M))) / n_valid
        self.last_diag.update(below_floor_pct=round(100 * below_share, 1), above_rim_pct=round(100 * above_share, 1))
        if coverage < MIN_COVERAGE or np.count_nonzero(keep) < 30:
            # Self-heal ONLY a provably wrong floor (most surface far below the floor or above the rim).
            # A full bin that merely hides the floor keeps its good floor.
            wrong = max(below_share, above_share) > 0.6
            self._blind_since = (getattr(self, "_blind_since", None) or timestamp) if wrong else None
            if wrong and self.profile.floor_plane is not None and timestamp - self._blind_since > 20.0:
                self.forget_floor()
                self._blind_since = None
            return self._hold("waiting for a clear view of the bin surface", warnings=warnings)
        self._blind_since = None
        rows, cols = np.nonzero(keep)
        z = depth_m[rows, cols] * scale
        hx = (cols - intrinsics.ppx) * z / intrinsics.fx
        hy = (rows - intrinsics.ppy) * z / intrinsics.fy
        tops = height[keep].astype(np.float64)
        # One physical definition for both cameras, independent of how many objects a detector
        # happens to box: every visible upward-facing surface counts; floor noise (< 3 cm) is zero.
        tops = np.where(tops >= FLOOR_NOISE_M, tops, 0.0)
        cells = cell_tops(tops, hx, hy)
        if len(cells) < 8:
            return self._hold("waiting for a clear view of the bin surface", warnings=warnings)
        reading = fill_reading(cells, coverage, self.profile)
        frame_fill = reading["max_fill_height_cm"] / 100.0    # mean surface height, metres
        recent = [v for _, v in self.fill_history]
        if len(recent) >= 3 and abs(frame_fill - float(np.median(recent))) > INCONSISTENT_M:
            self.inconsistent += 1
            if self.inconsistent < 5:           # a lasting change (5 frames) is accepted as real
                return self._hold(f"inconsistent depth frame ignored ({frame_fill * 100:.0f} cm vs "
                                  f"{np.median(recent) * 100:.0f} cm recent)", warnings=warnings, stale_only=True)
            self.fill_history.clear()
        self.inconsistent = 0
        self.fill_history.append((timestamp, frame_fill))
        smoothed = float(np.median([v for _, v in self.fill_history]))
        reading = fill_reading(cells, coverage, self.profile, fill_m=smoothed)
        self.last_maps = (plane, scale)
        self.history.append((timestamp, cells))
        self.last_valid_at = timestamp
        self.reading = {**reading, "camera_id": self.camera_id, "updated_at": timestamp, "stale": False,
                        "warnings": warnings, "geometry": geometry, "depth_source": depth_label,
                        "frames_smoothed": len(self.fill_history)}
        return self._decorate(self.reading)

    def recalibrate(self, depth_m: np.ndarray | None, intrinsics: Any, region: np.ndarray | None,
                    *, automatic: bool = False) -> dict[str, Any]:
        """Find this camera's EMPTY bin floor in its own depth and keep it (reset button / first start).

        RealSense depth is metric: the floor must lie 80-140 cm away (110 cm reference).
        Logitech depth is model depth: its scale is set so the floor sits at the reference.
        """
        if depth_m is None or intrinsics is None:
            return {"ok": False, "reason": "no depth frame yet"}
        depth_m, intrinsics, region = _subsample(depth_m, intrinsics, region)
        drift = self.drift(depth_m, region)
        depth_m = depth_m / drift
        fit = fit_floor_plane(depth_m, intrinsics, region)
        if fit is None:
            return {"ok": False, "reason": "bin floor not visible (cover <15% of the view) -- empty the bin and retry"}
        plane, distance, share = fit
        reference = float(self.profile.camera_to_empty_floor_m or DEFAULT_FLOOR_DISTANCE_M)
        known = self.profile.floor_raw_distance
        if automatic and known and distance < 0.97 * known:
            # Waste is always closer than the floor: a shallower "floor" is the top of the pile.
            return {"ok": False, "reason": "a deeper floor is already known (pile top is not the floor)"}
        if self.camera_id == "logitech":
            if automatic and share < (0.35 if self.profile.floor_plane is not None else 0.20):
                return {"ok": False, "reason": "floor not clearly visible yet (automatic fit needs an open floor)"}
            scale = reference / distance
        else:
            if automatic and abs(distance - reference) > 0.12:
                # A waste layer is flat too: unattended, only a surface at the floor reference is the floor.
                return {"ok": False, "reason": f"deepest flat surface at {distance * 100:.0f} cm, not the "
                                               f"{reference * 100:.0f} cm floor"}
            if not 0.80 <= distance <= 1.40:
                return {"ok": False, "reason": f"deepest flat surface is {distance * 100:.0f} cm away, not near the "
                                               f"{reference * 100:.0f} cm floor -- floor hidden by waste?"}
            scale = 1.0
        self.profile.floor_plane = [float(v) for v in plane]
        self.profile.floor_scale = float(scale)
        self.profile.floor_fitted_at = time.time()
        self.profile.floor_raw_distance = float(distance)
        if region is not None and region.shape == depth_m.shape:
            outside = (~region) & np.isfinite(depth_m) & (depth_m > 0.05)
            self.profile.outside_reference = (float(np.median(depth_m[outside])) * drift
                                              if np.count_nonzero(outside) >= 50 else None)
        a, b, _ = plane
        self.profile.tilt_from_vertical_deg = round(math.degrees(math.atan(math.hypot(a, b))), 1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(asdict(self.profile)), encoding="utf-8")
        except OSError:
            pass
        self.fill_history.clear()
        return {"ok": True, "floor_cm": round(distance * scale * 100, 1), "tilt_deg": self.profile.tilt_from_vertical_deg,
                "floor_share": round(share, 2), "scale": round(scale, 3), "automatic": automatic}

    def forget_floor(self) -> None:
        """Drop the fitted floor (reset / self-heal); it is re-found automatically from the next frames."""
        self.profile.floor_plane = None
        self.profile.floor_raw_distance = None
        self.profile.outside_reference = None
        self.profile.floor_scale = 1.0
        self.fill_history.clear()
        self._last_fit_try = -1e9
        try:
            self.path.write_text(json.dumps(asdict(self.profile)), encoding="utf-8")
        except OSError:
            pass

    def drift(self, depth_m: np.ndarray, region: np.ndarray | None) -> float:
        """Monocular depth rescales the whole scene from frame to frame. The walls/rim OUTSIDE the bin
        do not change, so their median depth now vs. at the floor fit gives this frame's scale drift."""
        ref = self.profile.outside_reference
        if self.camera_id != "logitech" or not ref or region is None or region.shape != depth_m.shape:
            return 1.0
        outside = (~region) & np.isfinite(depth_m) & (depth_m > 0.05)
        if np.count_nonzero(outside) < 50:
            return 1.0
        ratio = float(np.median(depth_m[outside])) / ref
        return ratio if 0.5 < ratio < 2.0 else 1.0

    def height_map_small(self, depth_m: np.ndarray, intrinsics: Any):
        """(height above floor, pixel area m^2, x, y metres) for the deposit counter's small view, or None."""
        plane, scale = self.profile.floor_plane, float(self.profile.floor_scale or 1.0)
        if self.profile.blocking():
            return None
        depth_m = depth_m / getattr(self, "last_drift", 1.0)
        height, _, valid = surface_maps(depth_m, intrinsics, self.profile, plane, scale)
        z = depth_m * scale
        v, u = np.indices(depth_m.shape)
        x, y = (u - intrinsics.ppx) * z / intrinsics.fx, (v - intrinsics.ppy) * z / intrinsics.fy
        area = (z / intrinsics.fx) * (z / intrinsics.fy)
        return (np.where(valid, height, np.nan).astype(np.float32), area.astype(np.float32),
                x.astype(np.float32), y.astype(np.float32))

    def object_volume(self, box, mask: np.ndarray | None, now: float,
                      rigid_hint: bool = False) -> tuple[float, float] | None:
        """(litres, height m) of one detection above the surface AROUND it, from this camera's own map.

        Volume = sum over the object's pixels of (height - local support) x pixel floor area. The local
        support is the MEDIAN height in a ring around the box (on a +-6 cm uneven pile: mean |error| 6.8 % vs 16.9 % with the 25th percentile, which always over-read), so a bag lying on the pile is
        measured from the pile, not from the bin floor. None when the map is stale or support unclear.
        """
        self.last_object = {}                            # never a previous object's geometry
        surface = getattr(self, "last_surface", None)
        if not surface or now - surface["at"] > 10.0:      # the pile changes slowly; settled maps stay valid
            return None
        step, height, ok, area = surface["step"], surface["height"], surface["ok"], surface["area"]
        valid = surface.get("valid", ok)
        h, w = height.shape
        x1, y1, x2, y2 = (int(round(v / step)) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        inside = np.zeros_like(ok)
        if mask is not None and mask.ndim == 2:
            small = mask[::step, ::step][:h, :w]
            inside[:small.shape[0], :small.shape[1]] = small
            inside[:, :x1] = inside[:, x2:] = False
            inside[:y1] = inside[y2:] = False
        if not inside.any():
            inside[y1:y2, x1:x2] = True
        pad_x, pad_y = max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4)
        ring = np.zeros_like(ok)
        ring[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)] = True
        ring &= ~inside
        support_px = height[ring & ok]                   # support: median of the surface around the bag
        top = inside & valid                             # the bag's own surface: any valid depth
        if support_px.size < 10 or np.count_nonzero(top) < 10:
            return None
        support = float(np.percentile(support_px, 50))
        # A closed rigid box shows a flat rectangular face: measure it as a cuboid from its own
        # face and an OBSERVED thickness (side face / support at its low edge), not as the volume
        # above a ring median -- on uneven bags that support cut the box's low end off (45 cm box
        # read 27 cm long, 17 cm "tall", 2.4 L).
        face = _box_face(surface, top, valid, height, support, float(np.percentile(support_px, 25)))
        if face is not None:
            self.last_object = {"length_m": face["length"], "width_m": face["width"],
                                "height_m": face.get("thickness"), "method": face["method"], "planar": True,
                                "litres": None if face.get("litres") is None else round(face["litres"], 2),
                                "tilt_deg": face["tilt_deg"], "face_share": face["share"],
                                "thickness_source": face.get("thickness_source"), "reason": face.get("reason")}
            if face.get("litres") is None:
                return None                              # L x W seen, thickness not observable: no number
            return round(face["litres"], 2), face["thickness"]
        rise_map = np.where(top, height - support, 0.0)
        risen = top & (rise_map > 0.02)
        # One object: the largest connected risen region. A box drawn around a diagonal parcel
        # also covers pile beside it; those pixels are a separate island and are left out.
        try:
            import cv2
            count, labels, stats, _ = cv2.connectedComponentsWithStats(risen.astype(np.uint8), 8)
            if count > 2:
                risen = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        except ImportError:  # pragma: no cover
            cv2 = None
        if np.count_nonzero(risen) < 10:
            return None
        litres = float(np.sum(rise_map[risen] * area[risen])) * 1000.0
        # Depth dropouts on the object: assume they resemble the valid pixels around them.
        missing = int(np.count_nonzero(cv2_dilate(risen) & inside & ~valid))
        if missing:
            litres *= 1.0 + missing / max(1, int(np.count_nonzero(risen)))
        tall = float(np.percentile(rise_map[risen], 90))
        if litres > 250.0 or tall > self.profile.usable_height_m + RIM_TOLERANCE_M:
            return None                                  # implausible: unavailable, never clamped
        length = width = None
        method = "surface rise above the local surface"
        if cv2 is not None and "x" in surface:
            pts = np.c_[surface["x"][risen], surface["y"][risen]].astype(np.float32)
            (_, _), (w1, w2), _ = cv2.minAreaRect(pts)     # oriented: a diagonal parcel stays long/thin
            length, width = float(max(w1, w2)), float(min(w1, w2))
        slab = _tilted_slab(surface, risen, rise_map, min_share=0.4 if rigid_hint else 0.7)
        planar = bool(slab and slab.get("planar") and slab.get("share", 0) >= 0.7)
        if slab is not None and slab.get("litres") is not None and slab["litres"] < litres:
            # A rigid flat-topped object (box, book) propped on the pile at an angle: integrating
            # the top surface down to the support counted the AIR under its raised end (a ~4 L box
            # read 8.3 L, 31 cm "tall"). Its own plane gives in-plane L x W and the thickness.
            litres, tall = slab["litres"], slab["thickness"]
            length, width = slab["length"], slab["width"]
            method = "tilted rigid slab: top-face L x W x thickness"
        self.last_object = {"length_m": length, "width_m": width, "height_m": tall, "litres": round(litres, 2),
                            "method": method, "planar": planar}
        return round(litres, 2), tall

    def _up_vector(self, plane) -> np.ndarray:
        if plane is not None:
            a, b, _ = plane
            return np.array([a, b, -1.0]) / math.sqrt(a * a + b * b + 1.0)
        tilt = math.radians(self.profile.tilt_from_vertical_deg or 0.0)
        return np.array([0.0, -math.sin(tilt), -math.cos(tilt)])

    def _hold(self, reason: str, *, warnings: list[str] | None = None, stale_only: bool = False) -> dict[str, Any]:
        """Keep the last valid reading (marked stale, with its age) instead of inventing a fresh one."""
        if self.reading.get("status") == "ok":
            self.reading.update(stale=True, stale_reason=reason, warnings=warnings or self.reading.get("warnings", []))
            return self._decorate(self.reading)
        if stale_only and self.reading.get("status") == "na" and self.reading.get("reason"):
            return self._decorate(self.reading)
        self.reading = self._na(reason)
        self.reading["warnings"] = warnings or []
        return self.reading

    def added_height_m(self, started_at: float, finalized_at: float) -> tuple[float | None, str | None]:
        """New bag height: AFTER top minus the BEFORE surface, over the cells that rose (its footprint)."""
        before = next((cells for ts, cells in reversed(self.history) if ts < started_at), None)
        after = next((cells for ts, cells in reversed(self.history) if ts >= finalized_at - 1.0), None)
        if before is None or after is None:
            return None, "no settled before/after surface pair around the deposit"
        rises = [after[key] - before[key] for key in after.keys() & before.keys() if after[key] - before[key] > 0.02]
        if len(rises) < 4:
            return None, "no measurable rise of the surface under the new bag"
        return float(np.percentile(rises, 90)), None


def default_profile(camera_id: str, config: Any) -> FillProfile:
    """Shared installation defaults, per camera (persisted once on first start).

    One bin, so the usable floor-to-rim height (100 cm) and capacity (660 L,
    nominal) are shared. The camera-to-floor reference is the RealSense's
    measured ~110 cm; the Logitech starts from the same value as an explicitly
    APPROXIMATE shared-installation assumption (its tripod may differ), unless
    its own operator-measured reference distance is configured. 100 cm usable
    height and 110 cm camera range are different quantities and stay separate.
    Intrinsics, pose and depth scaling stay per camera.
    """
    try:
        distance_m = float(os.environ.get("LOCALLIFE_FLOOR_DISTANCE_CM", "110")) / 100.0
    except ValueError:
        distance_m = DEFAULT_FLOOR_DISTANCE_M
    notes = [f"usable height {DEFAULT_USABLE_HEIGHT_M * 100:.0f} cm (shared bin value, approximate)"]
    logitech_m = float(getattr(config, "logitech_reference_distance_m", 0.0) or 0.0)
    if camera_id == "logitech" and logitech_m > 0:
        distance_m, why = logitech_m, "Logitech operator-measured reference distance"
    elif camera_id == "logitech":
        why = "approximate shared-installation assumption copied from the RealSense reference; edit if the tripods differ"
    else:
        why = "RealSense measured reference"
    return FillProfile(camera_id=camera_id, camera_to_empty_floor_m=distance_m, usable_height_m=DEFAULT_USABLE_HEIGHT_M,
                       camera_above_rim_m=DEFAULT_ABOVE_RIM_M, source="default-provisional",
                       notes=[f"floor reference {distance_m * 100:.0f} cm: {why}"] + notes)


def fit_floor_plane(depth_m: np.ndarray, intrinsics: Any, region: np.ndarray | None = None,
                    iterations: int = 150) -> tuple[tuple[float, float, float], float, float] | None:
    """RANSAC plane through the DEEPEST visible points: the empty bin floor, when it is visible.

    Returns ((a, b, c) with z = a x + b y + c, perpendicular distance, floor share of the view).
    """
    valid = (depth_m > 0.05) & np.isfinite(depth_m) & (True if region is None else region)
    rows, cols = np.nonzero(valid)
    if rows.size < 200:
        return None
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols - intrinsics.ppx) * z / intrinsics.fx
    y = (rows - intrinsics.ppy) * z / intrinsics.fy
    deep = z >= np.percentile(z, 70)
    P = np.c_[x[deep], y[deep], z[deep]]
    tolerance = 0.02 * float(np.median(z))
    rng = np.random.default_rng(0)
    best, best_count = None, 0
    for _ in range(iterations):
        sample = P[rng.choice(len(P), 3, replace=False)]
        A = np.c_[sample[:, :2], np.ones(3)]
        try:
            coef = np.linalg.solve(A, sample[:, 2])
        except np.linalg.LinAlgError:
            continue
        if math.hypot(coef[0], coef[1]) > 1.2:              # steeper than ~50 deg: a wall, not a floor
            continue
        count = int(np.count_nonzero(np.abs(np.c_[x, y, np.ones_like(x)] @ coef - z) < tolerance))
        if count > best_count:
            best, best_count = coef, count
    if best is None:
        return None
    inliers = np.abs(np.c_[x, y, np.ones_like(x)] @ best - z) < tolerance
    share = float(np.count_nonzero(inliers)) / rows.size
    if share < 0.15:
        return None
    coef, *_ = np.linalg.lstsq(np.c_[x[inliers], y[inliers], np.ones(int(inliers.sum()))], z[inliers], rcond=None)
    a, b, c = (float(v) for v in coef)
    return (a, b, c), abs(c) / math.sqrt(a * a + b * b + 1.0), share


def surface_maps(depth_m: np.ndarray, intrinsics: Any, profile: FillProfile, plane, scale: float = 1.0):
    """2-D (height above floor, upward-facing mask, valid mask). Walls (steep surfaces) are not waste tops."""
    valid = (depth_m > 0.05) & np.isfinite(depth_m)
    z = np.where(valid, depth_m, np.nan).astype(np.float64)
    v, u = np.indices(depth_m.shape)
    x = (u - intrinsics.ppx) * z / intrinsics.fx
    y = (v - intrinsics.ppy) * z / intrinsics.fy
    if plane is not None:
        a, b, c = plane
        norm = math.sqrt(a * a + b * b + 1.0)
        height = (a * x + b * y + c - z) / norm * scale
        up = np.array([a, b, -1.0]) / norm
    else:
        tilt = math.radians(profile.tilt_from_vertical_deg or 0.0)
        up = np.array([0.0, -math.sin(tilt), -math.cos(tilt)])
        height = profile.vertical_height_m() + up[1] * y + up[2] * z
    # Normals from depth smoothed over 5 x 5 px: per-pixel stereo noise (a few mm at ~3 mm pixel
    # spacing) otherwise tilts every normal and drops real bag tops as "walls" (volume undercount).
    zs = _smooth_nan(z, 5)
    xs = (u - intrinsics.ppx) * zs / intrinsics.fx
    ys = (v - intrinsics.ppy) * zs / intrinsics.fy
    gx = [np.gradient(m, axis=1) for m in (xs, ys, zs)]
    gy = [np.gradient(m, axis=0) for m in (xs, ys, zs)]
    normal = np.cross(np.stack(gx, -1), np.stack(gy, -1))
    length = np.linalg.norm(normal, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        facing = np.abs(normal @ up) / length
    flat = np.nan_to_num(facing) >= math.cos(math.radians(55))
    return np.nan_to_num(height, nan=-9.0), flat, valid & np.isfinite(height)


BOX_FACE_MIN_SHARE = 0.35       # of the object's valid mask cells on one flat face
BOX_FACE_MAX_RMS_M = 0.012
BOX_FACE_MIN_RECT = 0.82        # face hull / its min-area rectangle (an ellipse is 0.785)
BOX_FACE_MIN_SIDE_M = 0.05


def _box_face(surface: dict[str, Any], inside: np.ndarray, valid: np.ndarray, height: np.ndarray,
              support: float, support_low: float | None = None) -> dict[str, Any] | None:
    """Closed rigid box seen as one flat RECTANGULAR face -> oriented cuboid; else None.

    L x W: min-area rectangle of the face points IN the face's own plane (not camera axes,
    not the image box). Thickness, only where the view supports it:
      1. a visible side face (points below the face, on a plane ~perpendicular to it):
         thickness = their depth below the face;
      2. else, a tilted face: the surface just beyond its LOW edge is what it rests on;
         thickness = (face height at that edge - that surface) / cos(tilt);
      3. else, a face lying flat: its height above the surrounding surface.
    A tilted face with neither 1 nor 2 keeps L x W and reports the thickness as unobservable.
    """
    if "z" not in surface or "up" not in surface:
        return None
    cells = inside & valid
    rows, cols = np.nonzero(cells)
    if rows.size < 40:
        return None
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    # The object's face, not the pile around it: a flat stretch of the support inside a loose
    # mask is flat too, so a plane that is not above the surrounding surface is set aside once.
    face = None
    for _ in range(2):
        face = _face_patch(surface, cells, rows, cols, cv2)
        if face is None:
            return None
        keep_cells = face[0]
        # vs the LOWER quartile around it: bags higher than a flat box beside it must not hide it
        floor_ref = support if support_low is None else support_low
        if float(np.median(height[rows[keep_cells], cols[keep_cells]])) - floor_ref >= 0.02:
            break
        cells = cells.copy()
        cells[rows[keep_cells], cols[keep_cells]] = False
        rows, cols = np.nonzero(cells)
        face = None
        if rows.size < 40:
            return None
    if face is None:
        return None
    keep, face_img, pts, centre, axes, rms = face
    share = float(keep.sum()) / len(pts)
    if share < BOX_FACE_MIN_SHARE or rms > BOX_FACE_MAX_RMS_M:
        return None
    return _box_from_face(surface, cells, rows, cols, keep, face_img, pts, centre, axes, rms, share,
                          valid, height, support, cv2)


def _face_patch(surface, cells, rows, cols, cv2):
    """Largest connected flat patch of the object's cells: (keep, image mask, points, centre, axes, rms)."""
    pts = np.c_[surface["x"][cells], surface["y"][cells], surface["z"][cells]].astype(np.float64)
    fit = _ransac_plane(pts)
    if fit is None:
        return None
    keep, centre, axes, rms = fit
    patch = np.zeros(cells.shape, np.uint8)
    patch[rows[keep], cols[keep]] = 1
    # Opening cuts the thin bridges a bleeding mask makes onto neighbouring bags that cross the
    # face's extended plane; 4-connectivity keeps diagonal leaks out.
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(patch, connectivity=4)
    if count < 2:
        return None
    ids = labels[rows, cols]
    biggest = 1 + int(np.argmax(np.bincount(ids[keep], minlength=count)[1:]))
    keep = keep & (ids == biggest)
    if keep.sum() < 30:
        return None
    face_img = labels == biggest
    centre = pts[keep].mean(axis=0)
    _, sing, axes = np.linalg.svd(pts[keep] - centre, full_matrices=False)
    rms = float(sing[2] / math.sqrt(int(keep.sum())))
    return keep, face_img, pts, centre, axes, rms


def _box_from_face(surface, cells, rows, cols, keep, face_img, pts, centre, axes, rms, share,
                   valid, height, support, cv2) -> dict[str, Any] | None:
    up = np.asarray(surface["up"], dtype=np.float64)
    normal = axes[2] if float(axes[2] @ up) >= 0 else -axes[2]
    cos_tilt = float(np.clip(normal @ up, -1.0, 1.0))
    if cos_tilt < 0.26:                                  # a near-vertical "face" is a wall or a side
        return None
    uv = np.c_[(pts[keep] - centre) @ axes[0], (pts[keep] - centre) @ axes[1]].astype(np.float32)
    corners = cv2.boxPoints(cv2.minAreaRect(uv))          # rectangle axes, free of angle conventions
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    e1, e2 = e1 / max(1e-9, float(np.linalg.norm(e1))), e2 / max(1e-9, float(np.linalg.norm(e2)))
    rot = uv @ np.stack([e1, e2], axis=1).astype(np.float32)
    # trimmed extents along the rectangle's own axes: a few leaked cells must not widen the box
    r1, r2 = (float(np.percentile(rot[:, i], 99) - np.percentile(rot[:, i], 1)) for i in (0, 1))
    rect = float(r1 * r2)
    core = rot[np.all((rot >= np.percentile(rot, 1, axis=0)) & (rot <= np.percentile(rot, 99, axis=0)), axis=1)]
    hull = float(cv2.contourArea(cv2.convexHull(core.astype(np.float32)))) if len(core) >= 3 else 0.0
    # cell centres sit half a cell inside the true edges: add one cell pitch per axis
    pitch = float(np.sqrt(np.median(surface["area"][cells][keep]))) / max(cos_tilt, 0.3)
    length, width = float(max(r1, r2)) + pitch, float(min(r1, r2)) + pitch
    if rect <= 0 or hull / rect < BOX_FACE_MIN_RECT or width < BOX_FACE_MIN_SIDE_M:
        return None
    tilt = math.degrees(math.acos(cos_tilt))
    result: dict[str, Any] = {"length": length, "width": width, "share": round(share, 2),
                              "tilt_deg": round(tilt, 1), "rms_m": round(rms, 4)}
    below = (centre - pts) @ normal                      # metres below the face plane
    thickness, source = None, None
    # Side faces: a box's side is perpendicular to its top and contains one rectangle edge. Its
    # points sit ON the plane through that edge (in-plane coordinate = the edge's), below the face,
    # and run CONTIGUOUSLY down from the edge; the run stops at the first gap, so bags under a
    # bleeding mask that happen to touch the plane further down are not counted.
    uv_all = np.c_[(pts - centre) @ axes[0], (pts - centre) @ axes[1]] @ np.stack([e1, e2], axis=1)
    lo_e = np.percentile(rot, 1, axis=0)
    hi_e = np.percentile(rot, 99, axis=0)
    tol = max(0.012, pitch)
    # Detector masks bleed a cell or two onto the neighbours: look for side faces only inside the
    # eroded mask, so the bag a box leans on cannot extend its "side" downwards.
    core_cells = cv2.erode(cells.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)[rows, cols]

    def side_run(allowed: np.ndarray) -> float | None:
        runs = []
        for i in (0, 1):
            j = 1 - i
            along = (uv_all[:, j] >= lo_e[j]) & (uv_all[:, j] <= hi_e[j])
            for edge in (hi_e[i] + pitch / 2, lo_e[i] - pitch / 2):
                on = (~keep) & allowed & along & (np.abs(uv_all[:, i] - edge) <= tol) \
                    & (below > 0.003) & (below < 0.35)
                if on.sum() < 6:
                    continue
                depths = np.sort(below[on])
                gaps = np.flatnonzero(np.diff(depths) > 0.02)
                run = depths[: gaps[0] + 1] if gaps.size else depths
                # A side seen almost edge-on (the view ray grazing it) has no usable depth extent.
                basis = (e1, e2)[i]
                side_normal = basis[0] * axes[0] + basis[1] * axes[1]
                view = centre / max(1e-9, float(np.linalg.norm(centre)))
                if abs(float(view @ side_normal)) < math.sin(math.radians(8)):
                    continue
                if run[0] <= 0.02 and run.size >= 6:
                    runs.append(float(run[-1]))
        return max(runs) if runs else None

    core_run = side_run(core_cells)
    full_run = side_run(np.ones(len(pts), dtype=bool))
    # The outer ring adds at most the side's last cell (<= 3 cm in replay); more than that is a
    # neighbour under a bleeding mask, so the eroded-core run is kept.
    runs = [r for r in (core_run, full_run if full_run is not None and core_run is not None
                        and full_run - core_run <= 0.03 else None) if r is not None]
    if runs:
        thickness, source = max(runs), "visible side face"
    if thickness is None and tilt >= 5.0:
        downhill = -(up - (up @ normal) * normal)
        downhill /= max(1e-9, float(np.linalg.norm(downhill)))
        across = np.cross(normal, downhill)
        all_valid = valid & surface.get("valid", valid)
        vr, vc = np.nonzero(all_valid)
        allp = np.c_[surface["x"][all_valid], surface["y"][all_valid], surface["z"][all_valid]].astype(np.float64)
        s_all, a_all = (allp - centre) @ downhill, (allp - centre) @ across
        s_face, a_face = (pts[keep] - centre) @ downhill, (pts[keep] - centre) @ across
        s_edge = float(np.percentile(s_face, 99))
        band = (s_all > s_edge + 0.02) & (s_all < s_edge + 0.07) & \
            (a_all > np.percentile(a_face, 5)) & (a_all < np.percentile(a_face, 95)) & \
            (((centre - allp) @ normal) > 0.0)
        if band.sum() >= 8:
            face_h = height[rows[keep], cols[keep]]
            slope, offset = np.polyfit(s_face, face_h, 1)
            edge_h = float(slope * s_edge + offset)
            beside = height[vr[band], vc[band]]
            rest_h = float(np.percentile(beside, 50))
            # Only a level, uncluttered surface is evidence of where the box rests: on a pile of
            # bags beside its edge (spread > 2 cm) this under-read 7 cm as 3-4 cm in replay.
            level = float(np.percentile(beside, 75) - np.percentile(beside, 25)) <= 0.02
            if level and edge_h - rest_h > 0.005:
                thickness = (edge_h - rest_h) / max(cos_tilt, 0.3)
                source = "surface beside the low edge (assumes the box rests there)"
    if thickness is None and tilt < 5.0:
        face_h = float(np.median(height[rows[keep], cols[keep]]))
        if face_h - support > 0.005:
            thickness, source = face_h - support, "face height above the surrounding surface"
    if thickness is None:
        result.update({"method": "box face: L x W measured, thickness not observable",
                       "reason": ("tilted box with no visible side face and no level surface seen beside its "
                                  "low edge") if tilt >= 5.0 else "flat face with no visible side face and no "
                                  "surface around it lower than the face"})
        return result
    result.update({"thickness": thickness, "thickness_source": source,
                   "litres": length * width * thickness * 1000.0,
                   "method": f"box cuboid: face L x W (in its own plane) x thickness from {source}"})
    return result


def _ransac_plane(pts: np.ndarray, tol: float = 0.015, iterations: int = 120):
    """Dominant plane of 3-D points: (inlier mask, centre, svd axes, rms). Deterministic seed."""
    rng = np.random.default_rng(7)
    sample = pts if len(pts) <= 4000 else pts[rng.choice(len(pts), 4000, replace=False)]
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs((sample - a) @ normal) <= tol))
        if count > best_count:
            best, best_count = (a, normal), count
    if best is None:
        return None
    keep = np.abs((pts - best[0]) @ best[1]) <= tol
    for _ in range(2):                                   # refine on the inliers
        if keep.sum() < 3:
            return None
        centre = pts[keep].mean(axis=0)
        _, sing, axes = np.linalg.svd(pts[keep] - centre, full_matrices=False)
        keep = np.abs((pts - centre) @ axes[2]) <= tol
    centre = pts[keep].mean(axis=0)
    _, sing, axes = np.linalg.svd(pts[keep] - centre, full_matrices=False)
    return keep, centre, axes, float(sing[2] / math.sqrt(max(1, int(keep.sum()))))


def _tilted_slab(surface: dict[str, Any], risen: np.ndarray, rise_map: np.ndarray,
                 min_share: float = 0.7) -> dict[str, Any] | None:
    """A flat top face (rigid box / book) -> slab dims; None for rounded shapes (bags).

    RANSAC plane of the object's 3-D points; its inliers must be one connected patch holding
    >= `min_share` of the object (0.7 by default so a rounded bag never passes; lower only when
    the object already looks rigid -- box-like label or cardboard colour -- because a neighbour
    or the visible side face then shares its mask). "planar" is reported at any tilt; slab dims
    only past 5 deg: L x W = extents in the plane, thickness = rise at the face's LOW edge /
    cos(tilt) (a rigid box rests on its low edge).
    """
    if "z" not in surface or "up" not in surface or np.count_nonzero(risen) < 30:
        return None
    rows, cols = np.nonzero(risen)
    pts = np.c_[surface["x"][risen], surface["y"][risen], surface["z"][risen]].astype(np.float64)
    rises = rise_map[risen]
    fit = _ransac_plane(pts)
    if fit is None:
        return None
    keep, centre, axes, rms = fit
    try:                                                 # one connected face, not coplanar bits around
        import cv2
        patch = np.zeros(risen.shape, np.uint8)
        patch[rows[keep], cols[keep]] = 1
        count, labels = cv2.connectedComponents(patch, connectivity=8)
        if count > 2:
            ids = labels[rows, cols]
            biggest = 1 + int(np.argmax(np.bincount(ids[keep], minlength=count)[1:]))
            keep = keep & (ids == biggest)
            centre = pts[keep].mean(axis=0)
            _, sing, axes = np.linalg.svd(pts[keep] - centre, full_matrices=False)
            rms = float(sing[2] / math.sqrt(max(1, int(keep.sum()))))
    except ImportError:  # pragma: no cover
        pass
    share = float(keep.sum()) / len(pts)
    if keep.sum() < 30 or share < min_share or rms > 0.012:
        return None
    normal = axes[2]
    cos_tilt = abs(float(normal @ surface["up"]))
    result: dict[str, Any] = {"planar": True, "share": round(share, 2), "tilt_deg": round(math.degrees(math.acos(min(1.0, cos_tilt))), 1)}
    if cos_tilt > math.cos(math.radians(5)):
        return result                                    # flat-lying: the surface integral is right
    pts, rises = pts[keep], rises[keep]
    along = (pts - centre) @ axes[0]
    across = (pts - centre) @ axes[1]
    length = float(np.percentile(along, 98) - np.percentile(along, 2))
    width = float(np.percentile(across, 98) - np.percentile(across, 2))
    downhill = surface["up"] - (surface["up"] @ normal) * normal
    downhill = downhill / max(1e-9, float(np.linalg.norm(downhill)))
    s_hill = (pts - pts.mean(axis=0)) @ downhill
    slope, offset = np.polyfit(s_hill, rises, 1)
    low_end = float(np.percentile(s_hill, 2) if slope > 0 else np.percentile(s_hill, 98))
    thickness = float(slope * low_end + offset) / max(cos_tilt, 0.3)
    if thickness <= 0.005 or length <= 0 or width <= 0:
        return result
    result.update({"length": max(length, width), "width": min(length, width), "thickness": thickness,
                   "litres": length * width * thickness * 1000.0})
    return result


def cv2_dilate(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return mask
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _smooth_nan(values: np.ndarray, size: int) -> np.ndarray:
    """Box mean ignoring NaN (invalid depth); NaN stays NaN."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return values
    finite = np.isfinite(values)
    total = cv2.blur(np.where(finite, values, 0.0).astype(np.float64), (size, size))
    weight = cv2.blur(finite.astype(np.float64), (size, size))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    out[~finite] = np.nan
    return out


def _subsample(depth_m: np.ndarray, intrinsics: Any, region: np.ndarray | None, target: int = 160):
    """At most ~target px wide: the fill needs 5 cm cells, not every pixel (keeps each frame cheap)."""
    step = max(1, int(depth_m.shape[1] // target))
    if step == 1:
        return depth_m, intrinsics, region
    from types import SimpleNamespace
    small = depth_m[::step, ::step]
    geometry = SimpleNamespace(fx=intrinsics.fx / step, fy=intrinsics.fy / step,
                               ppx=intrinsics.ppx / step, ppy=intrinsics.ppy / step)
    return small, geometry, None if region is None else region[::step, ::step]


def height_map(depth_m: np.ndarray, intrinsics: Any, profile: FillProfile) -> np.ndarray:
    """2-D height above the empty floor (NaN where depth is invalid)."""
    out = np.full(depth_m.shape, np.nan, dtype=np.float32)
    valid = (depth_m > 0.05) & np.isfinite(depth_m)
    height, _, _ = heights_above_floor(depth_m, intrinsics, profile, valid)
    out[valid] = height
    return out
