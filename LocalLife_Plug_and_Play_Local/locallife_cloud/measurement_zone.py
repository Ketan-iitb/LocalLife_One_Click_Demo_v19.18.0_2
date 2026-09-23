"""The mat each camera measures on, and the scale of its floor.

Both cameras look at the same deposit mat from different places, so one set of
normalised coordinates cannot describe it for both: the Logitech ROI that
covers the mat in its view covers a bed and a chair in the RealSense view. A
zone is therefore stored per camera, with the resolution it was drawn at, and
is rejected rather than guessed at when that resolution no longer matches.

The four corners also carry the floor's real size, which turns the image into
a measurement: the homography from the corner quadrilateral to the mat's
metres gives every pixel its own ground area. One global pixels-per-centimetre
factor cannot do that in a perspective view, where a pixel at the far edge of
the mat covers several times the ground of a pixel at the near edge -- which is
how a carton at the back of the mat came out a third of a metre tall.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .geometry import fixed_bin_mask

ZONE_FILENAME = "measurement_zones.json"
# A zone drawn at one resolution is reused at another only when the aspect
# ratio is unchanged: the same lens, the same framing, a different stream size.
MAX_ASPECT_DRIFT = 0.02
FULL_FRAME = (0.0, 0.0, 1.0, 1.0)

_AREA_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _ordered(corners: Iterable[Iterable[float]]) -> tuple[tuple[float, float], ...]:
    """Corners as near-left, near-right, far-right, far-left.

    An operator clicks the mat's corners in whatever order they see them; the
    homography needs a consistent one, so they are sorted by image row (near
    is lower in the image) and then by column.
    """
    points = [(float(x), float(y)) for x, y in corners]
    if len(points) != 4:
        raise ValueError("A measurement zone is defined by exactly four corners")
    by_row = sorted(points, key=lambda point: point[1])
    far = sorted(by_row[:2], key=lambda point: point[0])
    near = sorted(by_row[2:], key=lambda point: point[0])
    return (near[0], near[1], far[1], far[0])


@dataclass(frozen=True)
class MeasurementZone:
    """The quadrilateral one camera measures inside, and what it spans."""

    camera: str
    corners: tuple[tuple[float, float], ...]
    width_px: int
    height_px: int
    # The mat's real size: the near edge and the near-to-far run. Without them
    # the zone still limits what is measured, but carries no floor scale.
    near_edge_m: float = 0.0
    depth_edge_m: float = 0.0
    created_at: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("A measurement zone needs the resolution it was drawn at")
        object.__setattr__(self, "corners", _ordered(self.corners))
        for x, y in self.corners:
            if not (-1 <= x <= self.width_px + 1 and -1 <= y <= self.height_px + 1):
                raise ValueError("Measurement zone corners must lie inside the image")
        if self._area_px() < 0.01 * self.width_px * self.height_px:
            raise ValueError("A measurement zone must cover at least 1 % of the image")

    # -- geometry -----------------------------------------------------------
    def _area_px(self) -> float:
        points = np.asarray(self.corners, dtype=np.float64)
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def has_floor_scale(self) -> bool:
        return self.near_edge_m > 0 and self.depth_edge_m > 0

    def normalised_polygon(self) -> tuple[tuple[float, float], ...]:
        return tuple((x / self.width_px, y / self.height_px) for x, y in self.corners)

    def for_shape(self, shape: tuple[int, ...]) -> "MeasurementZone | None":
        """This zone at another resolution, or None when it cannot be trusted.

        The same framing at a different stream size scales; a different aspect
        ratio means a different crop or a moved camera, and a zone that is
        merely stretched onto it would measure the wrong part of the room.
        """
        height, width = int(shape[0]), int(shape[1])
        if width == self.width_px and height == self.height_px:
            return self
        if width <= 0 or height <= 0:
            return None
        mine = self.width_px / self.height_px
        theirs = width / height
        if abs(mine - theirs) / mine > MAX_ASPECT_DRIFT:
            return None
        scale_x, scale_y = width / self.width_px, height / self.height_px
        return replace(
            self, width_px=width, height_px=height,
            corners=tuple((x * scale_x, y * scale_y) for x, y in self.corners),
        )

    def mask(self, shape: tuple[int, ...]) -> np.ndarray | None:
        """The pixels inside the zone, or None at an incompatible resolution."""
        zone = self.for_shape(shape)
        if zone is None:
            return None
        return fixed_bin_mask(shape, FULL_FRAME, zone.normalised_polygon())

    def homography(self, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
        """Image pixels to metres on the mat, or None without its real size."""
        if not self.has_floor_scale:
            return None
        zone = self if shape is None else self.for_shape(shape)
        if zone is None:
            return None
        import cv2

        source = np.asarray(zone.corners, dtype=np.float32)
        target = np.asarray([
            (0.0, 0.0), (self.near_edge_m, 0.0),
            (self.near_edge_m, self.depth_edge_m), (0.0, self.depth_edge_m),
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        return None if matrix is None or not np.all(np.isfinite(matrix)) else matrix

    def ground_points(self, rows: np.ndarray, columns: np.ndarray,
                      shape: tuple[int, ...]) -> np.ndarray | None:
        """Where these pixels sit on the mat, in metres."""
        matrix = self.homography(shape)
        if matrix is None or rows.size == 0:
            return None
        import cv2

        pixels = np.column_stack((columns.astype(np.float32), rows.astype(np.float32)))
        return cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    def pixel_area_m2(self, shape: tuple[int, ...]) -> np.ndarray | None:
        """Each pixel's own ground area, in square metres.

        A pixel's footprint grows with distance across a perspective view, so
        the area is taken from the homography at that pixel rather than from
        one factor for the whole image.
        """
        matrix = self.homography(shape)
        if matrix is None:
            return None
        height, width = int(shape[0]), int(shape[1])
        key = (self.camera, self.corners, self.near_edge_m, self.depth_edge_m, height, width)
        cached = _AREA_CACHE.get(key)
        if cached is not None:
            return cached
        import cv2

        rows, columns = np.indices((height, width), dtype=np.float32)
        def project(dx: float, dy: float) -> np.ndarray:
            pixels = np.stack((columns + dx, rows + dy), axis=-1).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(pixels, matrix).reshape(height, width, 2)

        origin = project(0.0, 0.0)
        along_x = project(1.0, 0.0) - origin
        along_y = project(0.0, 1.0) - origin
        area = np.abs(along_x[..., 0] * along_y[..., 1] - along_x[..., 1] * along_y[..., 0])
        area = np.where(np.isfinite(area), area, 0.0).astype(np.float32)
        if len(_AREA_CACHE) > 8:
            _AREA_CACHE.clear()
        _AREA_CACHE[key] = area
        return area

    def footprint_m(self, mask: np.ndarray) -> tuple[float, float, float] | None:
        """(length, width, area) of a mask on the mat, in metres and m^2.

        Measured where the object stands rather than in pixels, so the same
        carton reports the same footprint at the front and the back of the mat.
        """
        if not self.has_floor_scale or not np.any(mask):
            return None
        rows, columns = np.nonzero(mask)
        points = self.ground_points(rows, columns, mask.shape)
        if points is None or len(points) < 3:
            return None
        import cv2

        (_, _), (side_a, side_b), _ = cv2.minAreaRect(points.astype(np.float32))
        area_map = self.pixel_area_m2(mask.shape)
        area = float(area_map[mask].sum()) if area_map is not None else float(side_a * side_b)
        return (max(side_a, side_b), min(side_a, side_b), area)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera, "corners": [list(corner) for corner in self.corners],
            "width_px": self.width_px, "height_px": self.height_px,
            "near_edge_m": self.near_edge_m, "depth_edge_m": self.depth_edge_m,
            "created_at": self.created_at, "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeasurementZone":
        return cls(
            camera=str(payload["camera"]),
            corners=tuple(tuple(float(value) for value in corner) for corner in payload["corners"]),
            width_px=int(payload["width_px"]), height_px=int(payload["height_px"]),
            near_edge_m=float(payload.get("near_edge_m", 0.0) or 0.0),
            depth_edge_m=float(payload.get("depth_edge_m", 0.0) or 0.0),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            note=str(payload.get("note", "")),
        )

    def describe(self) -> dict[str, Any]:
        """What the dashboard shows about this zone."""
        return {
            "camera": self.camera, "resolution": f"{self.width_px}x{self.height_px}",
            "corners": [[round(x, 1), round(y, 1)] for x, y in self.corners],
            "mat_size_m": None if not self.has_floor_scale else
            [round(self.near_edge_m, 3), round(self.depth_edge_m, 3)],
            "floor_scale": "ready" if self.has_floor_scale else "missing",
            "captured": None if not self.created_at else
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.created_at)),
        }


class MeasurementZoneStore:
    """The zones on disk, one per camera."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / ZONE_FILENAME
        self._zones: dict[str, MeasurementZone] = {}
        self._loaded = False

    def _load(self) -> dict[str, MeasurementZone]:
        if self._loaded:
            return self._zones
        self._loaded = True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._zones
        for camera, record in (payload or {}).items():
            try:
                self._zones[str(camera)] = MeasurementZone.from_dict(record)
            except (KeyError, TypeError, ValueError):
                continue
        return self._zones

    def get(self, camera: str) -> MeasurementZone | None:
        return self._load().get(camera)

    def save(self, zone: MeasurementZone) -> MeasurementZone:
        zones = self._load()
        stored = replace(zone, created_at=zone.created_at or time.time())
        zones[zone.camera] = stored
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {camera: item.to_dict() for camera, item in zones.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return stored

    def forget(self, camera: str) -> None:
        zones = self._load()
        if zones.pop(camera, None) is not None:
            self.path.write_text(
                json.dumps({name: item.to_dict() for name, item in zones.items()}, indent=2),
                encoding="utf-8",
            )

    def describe(self) -> dict[str, Any]:
        return {camera: zone.describe() for camera, zone in self._load().items()}


def zone_from_rectangle(
    camera: str, roi: tuple[float, float, float, float], shape: tuple[int, ...],
    **sizes: float,
) -> MeasurementZone:
    """The configured ROI rectangle as a zone, for a camera not yet drawn on.

    Keeps the measurement area exactly where the configuration already put it,
    so a station without a drawn mat behaves as it did before.
    """
    height, width = int(shape[0]), int(shape[1])
    x, y, w, h = roi
    left, top = x * width, y * height
    right, bottom = (x + w) * width, (y + h) * height
    return MeasurementZone(
        camera=camera,
        corners=((left, bottom), (right, bottom), (right, top), (left, top)),
        width_px=width, height_px=height,
        near_edge_m=float(sizes.get("near_edge_m", 0.0) or 0.0),
        depth_edge_m=float(sizes.get("depth_edge_m", 0.0) or 0.0),
        note="configured-roi",
    )


def quadrilateral_is_sane(corners: Iterable[Iterable[float]]) -> bool:
    """Four corners that make a convex quadrilateral with area."""
    try:
        points = np.asarray(_ordered(corners), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    signs = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if math.isclose(cross, 0.0, abs_tol=1e-6):
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)
