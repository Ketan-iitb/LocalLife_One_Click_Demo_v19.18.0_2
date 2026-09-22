"""Real-hardware diagnostic bundles (LOCALLIFE_HARDWARE_DIAGNOSTIC=1).

Off by default and read-only with respect to measurement: it saves what each
camera saw and concluded, so a physical test run can be debugged afterwards
without guessing. Per finalised (or rejected) measurement it writes:

    original.png        the frame as processed (Logitech: lens-undistorted)
    object_mask.png     the final object mask
    mask_debug.png      Logitech only: detector / foreground / final / rejected
    depth.png           RealSense depth or calibrated/relative DA-V2 depth, colour-mapped
    points.npy          filtered support-plane points: columns u, v (m), height (m)
    result.json         geometry, dimensions, volume, rejection reason, latency, CSV row

Scene snapshots (no object needed -- empty bin, background movement) are
written every `interval_s` with the frame, depth, mask debug and warnings.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _depth_image(depth: np.ndarray | None) -> np.ndarray | None:
    import cv2

    if depth is None:
        return None
    values = depth[np.isfinite(depth) & (depth > 0)]
    if values.size == 0:
        return None
    low, high = np.percentile(values, (2, 98))
    scaled = np.clip((np.nan_to_num(depth, nan=high) - low) / max(high - low, 1e-6), 0, 1)
    return cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class HardwareDiagnostics:
    def __init__(self, directory: Path, camera_id: str, *, enabled: bool, interval_s: float = 5.0) -> None:
        self.directory = Path(directory).resolve() / camera_id
        self.camera_id = camera_id
        self.enabled = enabled
        self.interval_s = interval_s
        self._last_scene = 0.0
        self.bundles_written = 0

    def _bundle(self, kind: str, name: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stamp}_{kind}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, path: Path, images: dict[str, np.ndarray | None], payload: dict[str, Any],
               points: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        import cv2

        for name, image in images.items():
            if image is None:
                continue
            if image.dtype == bool:
                image = image.astype(np.uint8) * 255
            cv2.imwrite(str(path / f"{name}.png"), image)
        if points is not None:
            footprint, heights = points
            np.save(path / "points.npy", np.column_stack((footprint, heights)).astype(np.float32))
        (path / "result.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        self.bundles_written += 1

    def record_measurement(
        self,
        *,
        row: dict[str, Any],
        frame: np.ndarray | None,
        mask: np.ndarray | None,
        depth: np.ndarray | None,
        points: tuple[np.ndarray, np.ndarray] | None,
        mask_overlay: np.ndarray | None,
        latency_ms: float | None,
    ) -> Path | None:
        """Never raises: a diagnostic failure must not affect measurement."""
        if not self.enabled:
            return None
        try:
            path = self._bundle("measurement", str(row.get("event_id") or "unknown"))
            self._write(path, {
                "original": frame, "object_mask": mask, "mask_debug": mask_overlay, "depth": _depth_image(depth),
            }, {
                "camera": self.camera_id,
                "geometry": row.get("shape_geometry"),
                "dimensions_mm": [row.get("length_mm"), row.get("width_mm"), row.get("height_mm")],
                "volume_l": row.get("volume_l"),
                "status": row.get("status"),
                "rejection_reason": row.get("reason") or row.get("volume_rejection_reason"),
                "latency_ms": latency_ms,
                "points_saved": points is not None,
                "csv_row": row,
            }, points)
            return path
        except Exception:  # noqa: BLE001
            LOGGER.exception("Hardware diagnostic bundle failed for %s", self.camera_id)
            return None

    def maybe_record_scene(
        self,
        *,
        frame: np.ndarray,
        depth: np.ndarray | None,
        mask_overlay: np.ndarray | None,
        summary: dict[str, Any],
    ) -> Path | None:
        if not self.enabled or time.monotonic() - self._last_scene < self.interval_s:
            return None
        self._last_scene = time.monotonic()
        try:
            path = self._bundle("scene", str(int(time.time())))
            self._write(path, {"original": frame, "mask_debug": mask_overlay, "depth": _depth_image(depth)},
                        {"camera": self.camera_id, **summary})
            return path
        except Exception:  # noqa: BLE001
            LOGGER.exception("Hardware diagnostic scene snapshot failed for %s", self.camera_id)
            return None
