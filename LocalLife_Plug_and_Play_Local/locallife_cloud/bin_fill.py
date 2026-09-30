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
    tops = np.clip(np.fromiter(cells.values(), dtype=np.float64), 0.0, usable)
    fill = float(np.percentile(tops, 95)) if fill_m is None else float(min(max(fill_m, 0.0), usable))
    fraction = fill / usable
    reading = {
        "status": "ok", "reason": None, "coverage_pct": round(100 * coverage, 1),
        "max_fill_height_cm": round(fill * 100, 1), "usable_height_cm": round(usable * 100, 1),
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
               floor_plane=None, depth_label: str = "hardware depth") -> dict[str, Any]:
        self.last_processed_at = timestamp
        problems = self.profile.blocking()
        if problems:
            return self._hold("fill profile incomplete: " + "; ".join(problems))
        if depth_m is None or intrinsics is None:
            return self._hold(depth_reason or "no metric depth for this camera")
        if self.camera_moved(frame, region):
            return self._hold("camera or bin moved since the fill profile was saved: re-save it")
        if motion is not None and motion > STABLE_MOTION:
            return self._hold("scene moving", stale_only=True)
        depth_m, intrinsics, region = _subsample(depth_m, intrinsics, region)
        if floor_plane is not None:
            height, hx, hy = heights_above_plane(depth_m, intrinsics, floor_plane, region)
            geometry = "saved empty-bin floor plane (measured pose)"
        else:
            height, hx, hy = heights_above_floor(depth_m, intrinsics, self.profile, region)
            geometry = ("camera assumed looking straight down (tilt not measured)"
                        if self.profile.tilt_from_vertical_deg is None else "measured tilt")
        total = int(np.count_nonzero(region)) if region is not None else depth_m.size
        usable = self.profile.usable_height_m
        below = height < -BELOW_FLOOR_M
        keep = ~below & (height < usable + RIM_TOLERANCE_M)
        coverage = float(np.count_nonzero(keep)) / max(1, total)
        warnings = []
        below_share = float(np.count_nonzero(below)) / max(1, height.size)
        if below_share > BELOW_FLOOR_SHARE:
            deep = float(-np.percentile(height[below], 50)) * 100
            warnings.append(f"depth quality: {below_share:.0%} of pixels lie ~{deep:.0f} cm below the assumed floor "
                            f"({geometry}; floor reference {self.profile.camera_to_empty_floor_m * 100:.0f} cm). "
                            "Likely a tilted camera read as straight down, or the floor reference is short -- "
                            "enter tilt or capture an empty-bin baseline. These pixels are excluded, not clamped.")
        if coverage < MIN_COVERAGE:
            return self._hold(f"only {coverage:.0%} of the bin region has valid depth", warnings=warnings)
        cells = cell_tops(height[keep], hx[keep], hy[keep])
        if len(cells) < 8:
            return self._hold("too few surface cells with valid depth", warnings=warnings)
        reading = fill_reading(cells, coverage, self.profile)
        frame_fill = reading["max_fill_height_cm"] / 100.0
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
        self.history.append((timestamp, cells))
        self.last_valid_at = timestamp
        self.reading = {**reading, "camera_id": self.camera_id, "updated_at": timestamp, "stale": False,
                        "warnings": warnings, "geometry": geometry, "depth_source": depth_label,
                        "frames_smoothed": len(self.fill_history)}
        return self._decorate(self.reading)

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
