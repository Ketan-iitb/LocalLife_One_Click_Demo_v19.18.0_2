"""Accuracy report for the Logitech pipeline, read-only over the existing records.

Reads the comparison CSV the system already writes plus a ground-truth file,
and reports rigid and irregular objects separately: reference volume, RealSense
volume, Logitech raw and calibrated volume, errors, dimension errors, valid and
small-object rates. Nothing here writes to the measurement CSV; the report goes
to its own directory.

    python scripts/logitech_benchmark.py --truth captures/truth.json \
        --csv results/comparison/comparison_measurements.csv

`truth.json`: {"milk carton": {"litres": 0.95, "length_mm": 95, "width_mm": 95,
               "height_mm": 190, "rigid": true, "small": false}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def _number(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _metrics(errors: list[float], references: list[float]) -> dict[str, object]:
    if not errors:
        return {"samples": 0}
    absolute = np.abs(np.asarray(errors, dtype=float))
    percentage = absolute / np.maximum(np.asarray(references, dtype=float), 1e-9) * 100
    return {
        "samples": int(absolute.size),
        "mae_litres": round(float(absolute.mean()), 4),
        "rmse_litres": round(float(np.sqrt((absolute ** 2).mean())), 4),
        "mape_percent": round(float(percentage.mean()), 2),
        "median_percentage_error": round(float(np.median(percentage)), 2),
        "within_35_percent": round(float((percentage <= 35).mean()), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="results/comparison/comparison_measurements.csv")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--raw-volumes", default="results/logitech/logitech_raw_volumes.jsonl",
                        help="side-car written by the pipeline; supplies the uncorrected litres")
    parser.add_argument("--output-dir", default="results/benchmarks")
    arguments = parser.parse_args(argv)

    path = Path(arguments.csv)
    if not path.is_file():
        raise SystemExit(f"No comparison CSV at {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    truth = {key.strip().lower(): value for key, value in
             json.loads(Path(arguments.truth).read_text(encoding="utf-8")).items()}

    raw_by_id: dict[str, dict] = {}
    raw_path = Path(arguments.raw_volumes)
    if raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("measurement_id"):
                raw_by_id[str(entry["measurement_id"])] = entry

    events: dict[str, dict[str, dict]] = {}
    for row in rows:
        events.setdefault(row.get("comparison_event_id", ""), {})[row.get("camera_source", "")] = row

    records, detections = [], {"logitech": 0, "logitech_small": 0, "small_total": 0}
    for event_id, cameras in events.items():
        logitech, realsense = cameras.get("logitech"), cameras.get("realsense")
        name = ((logitech or realsense or {}).get("object_type") or "").strip().lower()
        reference = truth.get(name)
        if reference is None:
            continue
        if reference.get("small"):
            detections["small_total"] += 1
        valid = bool(logitech and logitech.get("status") != "missing" and _number(logitech.get("volume_liters")))
        if valid:
            detections["logitech"] += 1
            if reference.get("small"):
                detections["logitech_small"] += 1
        record = {
            "comparison_event_id": event_id, "object": name,
            "rigid": bool(reference.get("rigid", True)),
            "reference_litres": _number(reference.get("litres")),
            "realsense_litres": None if realsense is None else _number(realsense.get("volume_liters")),
            "logitech_raw_litres": None if logitech is None else _number(
                (raw_by_id.get(str(logitech.get("measurement_id"))) or {}).get("raw_volume_l")),
            "logitech_litres": None if logitech is None else _number(logitech.get("volume_liters")),
            "logitech_dimensions_mm": None if logitech is None else [
                _number(logitech.get(key)) for key in ("length_mm", "width_mm", "height_mm")],
            "reference_dimensions_mm": [reference.get(key) for key in ("length_mm", "width_mm", "height_mm")],
            "processing_time_ms": None if logitech is None else _number(logitech.get("processing_time_ms")),
            "status": None if logitech is None else logitech.get("status"),
        }
        records.append(record)

    def _errors(selector, key):
        chosen = [item for item in records if selector(item) and item["reference_litres"] and item.get(key)]
        return _metrics([item[key] - item["reference_litres"] for item in chosen],
                        [item["reference_litres"] for item in chosen])

    dimension_errors = []
    for item in records:
        measured, expected = item["logitech_dimensions_mm"], item["reference_dimensions_mm"]
        if measured and expected and all(value for value in measured) and all(expected):
            dimension_errors.extend(abs(m - float(e)) / float(e) * 100 for m, e in zip(measured, expected))

    report = {
        "created_at": time.time(), "csv": str(path.resolve()), "events": len(records),
        "rigid": {
            "logitech_raw": _errors(lambda item: item["rigid"], "logitech_raw_litres"),
            "logitech_calibrated": _errors(lambda item: item["rigid"], "logitech_litres"),
            "realsense": _errors(lambda item: item["rigid"], "realsense_litres"),
        },
        "irregular": {
            "logitech_raw": _errors(lambda item: not item["rigid"], "logitech_raw_litres"),
            "logitech_calibrated": _errors(lambda item: not item["rigid"], "logitech_litres"),
        },
        "median_dimension_error_percent": None if not dimension_errors
        else round(float(np.median(dimension_errors)), 2),
        "valid_estimate_rate": round(detections["logitech"] / max(1, len(records)), 3),
        "small_object_detection_rate": (
            None if not detections["small_total"]
            else round(detections["logitech_small"] / detections["small_total"], 3)
        ),
        "records": records,
    }
    directory = Path(arguments.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"logitech_accuracy_{int(time.time())}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("events", "rigid", "irregular", "median_dimension_error_percent",
                       "valid_estimate_rate", "small_object_detection_rate")}, indent=2))
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
