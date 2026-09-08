"""Process saved RGB/depth experiments in GPU batches for thesis evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from .config import AppConfig
from .dataset import IMAGE_SUFFIXES
from .pipeline import VisionPipeline
from .storage import BucketSync
from .types import CameraIntrinsics


LOGGER = logging.getLogger(__name__)


def _read_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "depth_m" if "depth_m" in archive.files else archive.files[0]
            return archive[key].astype(np.float32)
    return np.load(path, allow_pickle=False).astype(np.float32)


def _matching_depth(directory: Path | None, image: Path) -> np.ndarray | None:
    if directory is None:
        return None
    for suffix in (".npz", ".npy"):
        candidate = directory / (image.stem + suffix)
        if candidate.is_file():
            return _read_depth(candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process thesis images on the L4 GPU")
    parser.add_argument("--input", required=True, help="Directory of RGB images")
    parser.add_argument("--depth-dir", help="Optional directory of matching .npy/.npz metric depth maps")
    parser.add_argument("--baseline-image", help="Empty-scene RGB image")
    parser.add_argument("--baseline-depth", help="Aligned empty-scene RealSense .npy/.npz depth")
    parser.add_argument("--intrinsics", help="JSON file containing fx, fy, ppx, ppy, width, and height")
    parser.add_argument("--batch-size", type=int, help="Images processed per GPU batch")
    parser.add_argument("--output", help="Output CSV; defaults to artifacts/batch/measurements.csv")
    parser.add_argument("--disable-bucket-sync", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    config = AppConfig.from_env()
    if args.batch_size:
        config.batch_size = args.batch_size
    input_directory = Path(args.input).expanduser().resolve()
    if not input_directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_directory}")
    images = sorted(path for path in input_directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No supported images were found in {input_directory}")

    camera = None
    if args.intrinsics:
        camera = CameraIntrinsics.from_dict(json.loads(Path(args.intrinsics).read_text(encoding="utf-8")))
    depth_directory = Path(args.depth_dir) if args.depth_dir else None

    vision = VisionPipeline(config)
    LOGGER.info("Runtime ready: %s", vision.warmup())
    if args.baseline_image:
        empty_image = _read_image(Path(args.baseline_image))
        empty_depth = _read_depth(Path(args.baseline_depth)) if args.baseline_depth else None
        vision.set_baseline(empty_image, empty_depth, camera)

    destination = Path(args.output) if args.output else config.results_dir / "batch" / "measurements.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "image_path",
        "objects",
        "classes",
        "realsense_volume_l",
        "monocular_volume_l",
        "inference_ms",
        "warnings",
    ]

    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(images), config.batch_size):
            chunk = images[start : start + config.batch_size]
            frames = [_read_image(path) for path in chunk]
            depths = [_matching_depth(depth_directory, path) for path in chunk]
            analysis = vision.process_batch(
                frames,
                depths=depths,
                intrinsics=[camera] * len(chunk),
                sources=[str(path) for path in chunk],
            )
            for path, result in zip(chunk, analysis, strict=True):
                writer.writerow(
                    {
                        "sample_id": path.stem,
                        "image_path": str(path),
                        "objects": len(result.detections),
                        "classes": ";".join(detection.label for detection in result.detections),
                        "realsense_volume_l": "" if result.realsense_total is None else result.realsense_total.liters,
                        "monocular_volume_l": "" if result.monocular_total is None else result.monocular_total.liters,
                        "inference_ms": result.inference_ms,
                        "warnings": ";".join(result.warnings),
                    }
                )
            output.flush()
            LOGGER.info("Processed %s/%s images", min(start + len(chunk), len(images)), len(images))

    LOGGER.info("Measurements saved to %s", destination)
    if config.enable_bucket_sync and not args.disable_bucket_sync:
        BucketSync(config.results_dir, config.bucket).sync_once()


if __name__ == "__main__":
    main()
