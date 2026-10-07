"""V49: one-click snapshot of what each camera saw (frame, depth, masks, detections) for offline tuning.

Written only when the operator presses the button; never touches calibration, counting or the cloud.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any

import numpy as np


def _intrinsics(k: Any) -> dict | None:
    if k is None:
        return None
    return {f: float(getattr(k, f)) for f in ("fx", "fy", "ppx", "ppy", "width", "height") if hasattr(k, f)}


def _detection(d: Any) -> dict:
    keys = ("label", "confidence", "box", "source", "track_id", "color", "material", "material_confidence",
            "accepted_class", "support_volume_l", "support_height_cm", "support_length_cm", "support_width_cm", "support_diagnostics",
            "realsense_volume_l", "monocular_volume_l")
    out = {}
    for key in keys:
        value = getattr(d, key, None)
        out[key] = list(value) if isinstance(value, tuple) else value
    return json.loads(json.dumps(out, default=lambda v: float(v) if np.isscalar(v) else str(v)))


def build_zip(stations: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for camera, station in stations.items():
            snap = getattr(station, "_snapshot_inputs", None)
            if not snap:
                archive.writestr(f"{camera}/MISSING.txt", "no processed frame yet")
                continue

            def put(name: str, array: Any) -> None:
                if array is not None:
                    raw = io.BytesIO()
                    np.save(raw, np.asarray(array))
                    archive.writestr(f"{camera}/{name}.npy", raw.getvalue())

            put("frame_bgr", snap.get("frame"))
            put("depth_m", None if snap.get("depth_m") is None else np.asarray(snap["depth_m"], np.float32))
            put("region", snap.get("region"))
            masks = {f"det{i}": d.mask for i, d in enumerate(snap.get("detections") or [])
                     if getattr(d, "mask", None) is not None}
            masks.update({f"raw{i}": d.mask for i, d in enumerate(snap.get("raw_detections") or [])
                          if getattr(d, "mask", None) is not None})
            if masks:
                raw = io.BytesIO()
                np.savez_compressed(raw, **{k: np.asarray(v, bool) for k, v in masks.items()})
                archive.writestr(f"{camera}/masks.npz", raw.getvalue())
            fill = getattr(station, "fill", None)
            meta = {
                "camera": camera, "at": snap.get("at"), "intrinsics": _intrinsics(snap.get("intrinsics")),
                "detections": [_detection(d) for d in snap.get("detections") or []],
                "raw_detections": [_detection(d) for d in snap.get("raw_detections") or []],
                "logitech_mask_reasons": (snap.get("mask_debug") or {}).get("reasons"),
                "fill_profile": getattr(getattr(fill, "profile", None), "__dict__", None),
                "last_object": getattr(fill, "last_object", None),
                "detector_confidence": getattr(getattr(station, "config", None), "logitech_detector_confidence", None),
            }
            archive.writestr(f"{camera}/meta.json", json.dumps(meta, indent=1, default=str))
        archive.writestr("README.txt", "LocalLife V49 debug snapshot, " + time.strftime("%Y-%m-%d %H:%M:%S"))
    return buffer.getvalue()
