"""Measure actual L4 throughput and GPU memory instead of guessing performance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import AppConfig
from .dataset import IMAGE_SUFFIXES
from .pipeline import VisionPipeline


def _read_images(directory: Path, limit: int) -> list[np.ndarray]:
    import cv2

    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)[:limit]
    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError(f"No readable benchmark images found in {directory}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark segmentation plus depth on the available GPU")
    parser.add_argument("--images", required=True, help="Directory containing representative RGB frames")
    parser.add_argument("--batch-sizes", default="1,2,4,6", help="Comma-separated candidate GPU batch sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()

    import torch

    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or min(batch_sizes) < 1 or args.repeats < 1:
        parser.error("Batch sizes and repeat count must be positive")
    images = _read_images(Path(args.images), max(batch_sizes))
    config = AppConfig.from_env()
    pipeline = VisionPipeline(config)
    runtime = pipeline.warmup()
    is_cuda = getattr(pipeline.detector, "device", "cpu").startswith("cuda")
    measurements: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        config.batch_size = batch_size
        batch = [images[index % len(images)] for index in range(batch_size)]
        pipeline.process_batch(batch, persist=False)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        elapsed: list[float] = []
        visible: list[int] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = pipeline.process_batch(batch, persist=False)
            if is_cuda:
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            visible.extend(len(item.detections) for item in result)

        average = sum(elapsed) / len(elapsed)
        measurements.append(
            {
                "batch_size": batch_size,
                "average_batch_seconds": round(average, 4),
                "frames_per_second": round(batch_size / average, 2),
                "milliseconds_per_frame": round(average * 1000 / batch_size, 2),
                "mean_visible_objects": round(sum(visible) / len(visible), 2),
                "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if is_cuda else None,
            }
        )

    report = {"runtime": runtime, "measurements": measurements}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    destination.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
