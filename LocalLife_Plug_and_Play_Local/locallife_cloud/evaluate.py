"""Produce thesis-ready error statistics from reference volume measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def metric_summary(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(predicted)
    truth = reference[valid]
    estimate = predicted[valid]
    if truth.size == 0:
        return {"samples": 0}
    errors = estimate - truth
    nonzero = np.abs(truth) > 1e-9
    correlation = float(np.corrcoef(truth, estimate)[0, 1]) if truth.size > 1 and np.std(truth) > 0 and np.std(estimate) > 0 else None
    return {
        "samples": int(truth.size),
        "mae_l": float(np.mean(np.abs(errors))),
        "rmse_l": float(np.sqrt(np.mean(errors * errors))),
        "bias_l": float(np.mean(errors)),
        "median_absolute_error_l": float(np.median(np.abs(errors))),
        "mape_percent": float(np.mean(np.abs(errors[nonzero] / truth[nonzero])) * 100) if np.any(nonzero) else None,
        "pearson_r": correlation,
        "bland_altman_lower_l": float(np.mean(errors) - 1.96 * np.std(errors, ddof=1)) if truth.size > 1 else None,
        "bland_altman_upper_l": float(np.mean(errors) + 1.96 * np.std(errors, ddof=1)) if truth.size > 1 else None,
    }


def _parse_float(value: str | None) -> float:
    if value is None or not value.strip():
        return float("nan")
    return float(value)


def evaluate_csv(path: str | Path) -> dict[str, object]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The evaluation CSV contains no measurements")
    if "ground_truth_l" not in rows[0]:
        raise ValueError("Add a ground_truth_l column containing physically measured reference volumes")

    truth = np.asarray([_parse_float(row.get("ground_truth_l")) for row in rows])
    hardware = np.asarray([_parse_float(row.get("realsense_volume_l")) for row in rows])
    mono = np.asarray([_parse_float(row.get("monocular_volume_l")) for row in rows])
    return {
        "total_rows": len(rows),
        "realsense": metric_summary(truth, hardware),
        "monocular_calibrated": metric_summary(truth, mono),
        "notes": [
            "Volume is a visible height-field estimate, not a closed 3D mesh.",
            "Reference volumes should come from measured container geometry or controlled displacement tests.",
            "Monocular results are valid only after aligned RealSense calibration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute thesis-ready volume comparison statistics")
    parser.add_argument("measurements", help="CSV containing ground_truth_l and estimated volumes")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    report = evaluate_csv(args.measurements)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
