"""Re-measure saved diagnostic bundles (Logitech or RealSense) with the current geometry code.

Record bundles on the rig with LOCALLIFE_HARDWARE_DIAGNOSTIC=1; each finalised
Logitech measurement writes a folder with depth_m.npy, reference_depth_m.npy,
object_mask.png and result.json. Then:

    python scripts/replay_logitech_bundle.py results/hardware_diagnostics/logitech/*_measurement_* \
        --truth truth.csv

truth.csv (optional) has columns: bundle,length_mm,width_mm,height_mm,volume_l
with ruler-measured values; the bundle column is the folder name. Per-axis
errors are printed for each object and summarised. Nothing is written back and
no number is invented: a bundle without depth arrays is reported as skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.logitech_volume import metric_object_volume  # noqa: E402
from locallife_cloud.types import CameraIntrinsics  # noqa: E402
from locallife_cloud.volume import fit_reference_plane  # noqa: E402


def replay(bundle: Path) -> dict:
    import cv2

    payload = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    record = payload.get("logitech_record") or {}
    intrinsics = record.get("intrinsics") or payload.get("intrinsics")
    depth_path, reference_path = bundle / "depth_m.npy", bundle / "reference_depth_m.npy"
    if not depth_path.exists() or not reference_path.exists() or not intrinsics:
        return {"bundle": bundle.name, "skipped": "no depth arrays or intrinsics in this bundle"}
    depth, reference = np.load(depth_path), np.load(reference_path)
    mask = cv2.imread(str(bundle / "object_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    fx, fy, ppx, ppy = intrinsics
    camera = CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy, width=depth.shape[1], height=depth.shape[0])
    plane = fit_reference_plane(reference, camera, mask=~mask)
    if payload.get("camera") == "realsense":
        # The RealSense's own path: measured depth, edge pixels dropped first.
        from locallife_cloud.depth_edges import drop_depth_edge_pixels
        from locallife_cloud.volume import estimate_object_dimensions

        kept, _ = drop_depth_edge_pixels(depth, mask)
        dims = estimate_object_dimensions(depth, camera, kept, plane, min_height_m=0.01,
                                          max_height_m=1.0, min_points=20)
        return {"bundle": bundle.name, "camera": "realsense",
                "length_mm": None if dims is None else round(dims.length_mm, 1),
                "width_mm": None if dims is None else round(dims.width_mm, 1),
                "height_mm": None if dims is None else round(dims.height_mm, 1),
                "saved_dimensions_mm": payload.get("dimensions_mm"),
                "saved_volume_l": payload.get("volume_l")}
    result = metric_object_volume(depth, camera, mask, plane, reference_depth_m=reference,
                                  min_height_m=0.01, min_pixels=25)
    d = result.diagnostics
    return {
        "bundle": bundle.name, "reason": result.reason,
        "length_mm": d.get("length_mm"), "width_mm": d.get("width_mm"),
        "height_mm": None if d.get("height_p90_m") is None else round(d["height_p90_m"] * 1000.0, 1),
        "volume_l": None if result.measurement is None else round(result.measurement.liters, 3),
        "saved_dimensions_mm": record.get("dimensions_mm"),
        "floor_scale": (record.get("floor_scale") or {}).get("state"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--truth", type=Path, help="CSV of ruler-measured objects")
    args = parser.parse_args()
    truth = {}
    if args.truth:
        with args.truth.open(newline="", encoding="utf-8") as handle:
            truth = {row["bundle"]: row for row in csv.DictReader(handle)}
    errors: dict[str, list[float]] = {"length_mm": [], "width_mm": [], "height_mm": [], "volume_l": []}
    for bundle in args.bundles:
        row = replay(bundle)
        known = truth.get(bundle.name)
        if known and not row.get("skipped"):
            for key in errors:
                if row.get(key) is not None and known.get(key):
                    error = (float(row[key]) - float(known[key])) / float(known[key])
                    row[f"{key}_error"] = round(error, 3)
                    errors[key].append(abs(error))
        print(json.dumps(row))
    for key, values in errors.items():
        if values:
            print(f"{key}: n={len(values)} median |error|={np.median(values):.1%} max={max(values):.1%}")


if __name__ == "__main__":
    main()
