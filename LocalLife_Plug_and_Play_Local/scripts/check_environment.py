#!/usr/bin/env python3
"""Fail early when the VM is using a CPU-only interpreter or missing packages."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true", help="Load and cache both model checkpoints")
    parser.add_argument("--require-cuda", action="store_true", help="Exit nonzero when no GPU is available")
    args = parser.parse_args()

    modules = ["numpy", "cv2", "flask", "requests", "yaml", "torch", "ultralytics", "transformers"]
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in modules:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            missing.append(name)

    report: dict[str, object] = {"python": sys.version.split()[0], "packages": versions, "missing": missing}
    if "torch" not in missing:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

    if shutil.which("gcloud"):
        try:
            completed = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            report["gcp_project"] = completed.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["gcp_project_error"] = str(exc)

    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit("Missing packages: " + ", ".join(missing))
    if args.require_cuda and not report.get("cuda_available"):
        raise SystemExit("The interpreter cannot access the NVIDIA GPU")

    if args.download_models:
        from locallife_cloud.config import AppConfig
        from locallife_cloud.comparison import DualCameraCoordinator

        print(json.dumps(DualCameraCoordinator(AppConfig.from_env()).warmup(), indent=2))


if __name__ == "__main__":
    main()
