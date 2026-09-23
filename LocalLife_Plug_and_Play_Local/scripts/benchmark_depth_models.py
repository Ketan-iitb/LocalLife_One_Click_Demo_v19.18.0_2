"""Score candidate metric-depth models on this rig's own saved frames.

Each provider predicts depth for the same empty-bin frame and the same object
frames; the empty frame gives the support plane the prediction is aligned to
(exactly as the live pipeline does), and each object's height-map volume is
compared with its measured reference volume. A model that cannot be imported
is reported with the reason, never silently dropped.

    python scripts/benchmark_depth_models.py \
        --empty captures/empty.png --objects captures/objects/*.png \
        --truth captures/truth.json --camera-height-m 1.05 \
        --intrinsics results/logitech/calibration/logitech_lens.json

`truth.json`: {"milk_carton.png": {"litres": 0.95, "length_mm": 95, "width_mm": 95, "height_mm": 190}, ...}

Writes results/benchmarks/depth_models_<timestamp>.json. It reads nothing from,
and writes nothing to, the measurement CSV.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.depth_providers import candidate_providers, timed_predict  # noqa: E402
from locallife_cloud.logitech_volume import fit_plane_alignment, metric_object_volume  # noqa: E402
from locallife_cloud.types import CameraIntrinsics  # noqa: E402
from locallife_cloud.volume import fit_reference_plane  # noqa: E402


def _load_intrinsics(path: str | None, shape: tuple[int, int]) -> CameraIntrinsics:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))["intrinsics"]
        return CameraIntrinsics(fx=payload["fx"], fy=payload["fy"], ppx=payload["ppx"],
                                ppy=payload["ppy"], width=payload["width"], height=payload["height"])
    from locallife_cloud.edge_client import camera_intrinsics_from_fov

    estimate = camera_intrinsics_from_fov(shape[1], shape[0])
    return CameraIntrinsics(**estimate)


def _object_mask(frame: np.ndarray, empty: np.ndarray, region: np.ndarray, threshold: int = 22) -> np.ndarray:
    import cv2

    difference = np.max(np.abs(frame.astype(np.int16) - empty.astype(np.int16)), axis=2)
    mask = (difference >= threshold) & region
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--empty", required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--camera-height-m", type=float, required=True)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/benchmarks")
    arguments = parser.parse_args(argv)

    empty = cv2.imread(arguments.empty)
    if empty is None:
        raise SystemExit(f"Could not read {arguments.empty}")
    paths = sorted(path for pattern in arguments.objects for path in glob.glob(pattern))
    truth = json.loads(Path(arguments.truth).read_text(encoding="utf-8"))
    intrinsics = _load_intrinsics(arguments.intrinsics, empty.shape[:2])
    region = np.ones(empty.shape[:2], dtype=bool)

    report: dict[str, object] = {
        "created_at": time.time(), "empty_frame": arguments.empty, "objects": len(paths),
        "camera_height_m": arguments.camera_height_m,
        "intrinsics": {"fx": intrinsics.fx, "fy": intrinsics.fy,
                       "ppx": intrinsics.ppx, "ppy": intrinsics.ppy},
        "models": [],
    }
    for provider in candidate_providers(arguments.device):
        usable, reason = provider.available()
        entry: dict[str, object] = {"name": provider.name, "available": usable, "reason": reason}
        if not usable:
            report["models"].append(entry)
            print(f"{provider.name}: skipped ({reason})")
            continue
        try:
            empty_depth, load_ms = timed_predict(provider, empty, intrinsics)
            calibration, plane_diagnostics = fit_plane_alignment(
                empty_depth, region, intrinsics, arguments.camera_height_m,
            )
            if calibration is None:
                entry.update({"available": False, "reason": f"plane alignment failed: {plane_diagnostics}"})
                report["models"].append(entry)
                continue
            plane = fit_reference_plane(calibration.apply(empty_depth), intrinsics)
            rows, latencies = [], [load_ms]
            for path in paths:
                frame = cv2.imread(path)
                reference = truth.get(Path(path).name, {})
                depth, latency = timed_predict(provider, frame, intrinsics)
                latencies.append(latency)
                mask = _object_mask(frame, empty, region)
                result = metric_object_volume(
                    calibration.apply(depth), intrinsics, mask, plane,
                    min_height_m=0.01, camera_height_m=arguments.camera_height_m,
                )
                measured = None if result.measurement is None else result.measurement.liters
                expected = reference.get("litres")
                rows.append({
                    "object": Path(path).name, "reference_litres": expected,
                    "litres": measured, "reason": result.reason,
                    "length_mm": result.diagnostics.get("length_mm"),
                    "width_mm": result.diagnostics.get("width_mm"),
                    "height_mm": None if result.diagnostics.get("height_p90_m") is None
                    else result.diagnostics["height_p90_m"] * 1000,
                    "percentage_error": None if not (expected and measured)
                    else round(abs(measured - expected) / expected * 100, 2),
                })
            errors = [row["percentage_error"] for row in rows if row["percentage_error"] is not None]
            entry.update({
                "plane_rmse_m": plane_diagnostics.get("plane_rmse_m"),
                "mapping": plane_diagnostics.get("mapping"),
                "median_percentage_error": None if not errors else round(float(np.median(errors)), 2),
                "valid_rate": round(sum(1 for row in rows if row["litres"]) / max(1, len(rows)), 3),
                "median_latency_ms": round(float(np.median(latencies)), 1),
                "rows": rows,
            })
            print(f"{provider.name}: median error {entry['median_percentage_error']}%, "
                  f"{entry['median_latency_ms']} ms/frame")
        except Exception as exc:  # noqa: BLE001
            entry.update({"available": False, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{provider.name}: failed ({exc})")
        report["models"].append(entry)

    directory = Path(arguments.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"depth_models_{int(time.time())}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {output}")
    scored = [item for item in report["models"] if item.get("median_percentage_error") is not None]
    if scored:
        best = min(scored, key=lambda item: item["median_percentage_error"])
        print(f"Lowest median volume error: {best['name']} ({best['median_percentage_error']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
