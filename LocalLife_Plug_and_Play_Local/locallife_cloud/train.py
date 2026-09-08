"""Spot-safe, audited YOLO training tailored to the project's single L4 GPU."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import AppConfig
from .dataset import audit_dataset
from .inference import configure_torch, resolve_device
from .storage import ResultStore


LOGGER = logging.getLogger(__name__)


class CheckpointUploader:
    """Copy durable checkpoints immediately after Ultralytics saves each epoch."""

    def __init__(self, bucket_uri: str, run_name: str) -> None:
        self.destination = bucket_uri.rstrip("/") + f"/cloud-v15/training/{run_name}"
        self.seen_modification_times: dict[str, float] = {}

    def __call__(self, trainer: Any) -> None:
        if not shutil.which("gcloud"):
            LOGGER.warning("gcloud is unavailable; checkpoint was saved locally only")
            return
        run_directory = Path(trainer.save_dir)
        candidates = [
            run_directory / "weights" / "last.pt",
            run_directory / "weights" / "best.pt",
            run_directory / "results.csv",
            run_directory / "args.yaml",
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            modified = candidate.stat().st_mtime
            if self.seen_modification_times.get(str(candidate)) == modified:
                continue
            relative = candidate.relative_to(run_directory).as_posix()
            destination = f"{self.destination}/{relative}"
            try:
                completed = subprocess.run(
                    ["gcloud", "storage", "cp", str(candidate), destination],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                LOGGER.warning("Checkpoint upload failed for %s: %s", relative, exc)
                continue
            if completed.returncode:
                LOGGER.warning("Checkpoint upload failed for %s: %s", relative, completed.stderr.strip())
                continue
            self.seen_modification_times[str(candidate)] = modified
        LOGGER.info("Durable checkpoint sync completed for epoch %s", getattr(trainer, "epoch", "?"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a waste-object detector or segmenter on the L4 VM")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    parser.add_argument("--name", default="waste-yolo11m", help="Experiment name")
    parser.add_argument("--model", help="Pretrained weights; automatically chosen from the label format")
    parser.add_argument("--resume", help="Path to a previously saved weights/last.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=float, default=-1, help="-1 auto-selects a GPU-safe batch size")
    parser.add_argument("--workers", type=int, default=2, help="Keep at two on the 4-vCPU VM")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-bucket-sync", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    config = AppConfig.from_env()
    report = audit_dataset(args.data)
    store = ResultStore(config.results_dir)
    store.save_json(f"training/{args.name}/dataset_audit.json", report)
    for warning in report["warnings"]:
        LOGGER.warning("Dataset: %s", warning)
    if not report["ready"]:
        for issue in report["issues"]:
            LOGGER.error("Dataset: %s", issue)
        raise SystemExit("Fix the dataset audit errors before starting GPU training")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-cloud.txt before training") from exc

    device = resolve_device(args.device)
    LOGGER.info("GPU runtime: %s", configure_torch(device))
    model_name = args.model or ("yolo11m-seg.pt" if report["task"] == "segment" else "yolo11m.pt")
    checkpoint = Path(args.resume).expanduser() if args.resume else None
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"Resume checkpoint does not exist: {checkpoint}")

    model = YOLO(str(checkpoint) if checkpoint is not None else model_name)
    if not args.disable_bucket_sync and config.enable_bucket_sync:
        model.add_callback("on_model_save", CheckpointUploader(config.bucket, args.name))

    if checkpoint is not None:
        LOGGER.info("Resuming interrupted training from %s", checkpoint)
        model.train(resume=True)
        return

    batch: int | float = int(args.batch) if args.batch.is_integer() else args.batch
    LOGGER.info("Training %s on %s with dataset task=%s", model_name, device, report["task"])
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        amp=device.startswith("cuda"),
        cache="disk",
        patience=args.patience,
        save=True,
        save_period=1,
        project=str(config.results_dir / "training"),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=42,
        verbose=True,
    )
    LOGGER.info("Training completed: %s", getattr(results, "results_dict", results))


if __name__ == "__main__":
    main()
