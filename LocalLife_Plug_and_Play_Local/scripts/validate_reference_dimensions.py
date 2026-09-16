#!/usr/bin/env python3
"""Compare live RealSense dimensions with an annotated reference-object row.

This tool is for ``LOCALLIFE_OPERATING_MODE=geometry_validation``.  It reads
the physical LxWxH truth already stored in
``parameterised_objects/reference_objects.csv``.  The manifest's ``accepted``
column describes production waste classification; even a false row (for
example a backpack) is a valid geometry test object in validation mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "parameterised_objects" / "reference_objects.csv"
DEFAULT_LOG = PROJECT_ROOT / "data" / "reference_dimension_validation.jsonl"
PHANTOM_SOURCES = {
    "fixed-bin-depth-silhouette", "depth-scene-segmentation", "foreground-segmentation",
}


def load_references(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    """Load and type-check annotated reference objects without guessing values."""
    references: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item: dict[str, Any] = dict(row)
            item["accepted"] = str(row.get("accepted", "")).strip().lower() == "true"
            for field in ("footprint_length_mm", "footprint_width_mm", "height_mm"):
                value = str(row.get(field, "")).strip()
                item[field] = float(value) if value else None
            liters = str(row.get("reference_volume_liters", "")).strip()
            item["reference_volume_liters"] = float(liters) if liters else None
            if not item.get("file") or any(item[field] is None for field in (
                "footprint_length_mm", "footprint_width_mm", "height_mm",
            )):
                raise ValueError(f"Incomplete reference row in {path}: {row}")
            references.append(item)
    return references


def find_reference(references: list[dict[str, Any]], query: str) -> dict[str, Any]:
    normalized = query.strip().lower()
    exact = [item for item in references if str(item["file"]).lower() == normalized]
    if len(exact) == 1:
        return exact[0]
    partial = [
        item for item in references
        if normalized in str(item["file"]).lower()
        or normalized == Path(str(item["file"])).stem.lower()
    ]
    if len(partial) != 1:
        matches = ", ".join(str(item["file"]) for item in partial) or "none"
        raise ValueError(f"Reference '{query}' is not unique; matches: {matches}")
    return partial[0]


def current_dimension_observation(state: dict[str, Any]) -> dict[str, Any] | None:
    camera = (state.get("cameras") or {}).get("realsense") or {}
    latest = camera.get("latest") or {}
    candidates = [
        item for item in (latest.get("detections") or [])
        if item.get("tracking_status") in {"confirmed", "predicted"}
        and item.get("accepted_class") == "measurement_object"
        and item.get("source") not in PHANTOM_SOURCES
        and item.get("dimensions_mm")
    ]
    if len(candidates) != 1:
        return None
    item = candidates[0]
    dimensions = item["dimensions_mm"]
    return {
        "raw_label": item.get("label"),
        "color": item.get("color"),
        "length_mm": float(dimensions["footprint_length"]),
        "width_mm": float(dimensions["footprint_width"]),
        "height_mm": float(dimensions["height"]),
        "confidence": item.get("dimension_confidence"),
        "flags": item.get("dimension_flags") or [],
        "method": item.get("dimension_method"),
        "depth_coverage_percent": item.get("depth_coverage_percent"),
        "volume_liters": item.get("realsense_volume_l"),
        "build_version": state.get("build_version"),
        "operating_mode": state.get("operating_mode") or camera.get("operating_mode"),
    }


def dimension_error(actual: float, estimated: float) -> tuple[float, float]:
    error = estimated - actual
    return error, error / actual * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Full filename or unique part of a manifest filename")
    parser.add_argument("--list", action="store_true", help="List annotated reference objects and exit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="http://127.0.0.1:8100")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--placement", default="centre")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    references = load_references(args.manifest)
    if args.list:
        for item in references:
            volume = item["reference_volume_liters"]
            suffix = "dimensions only" if volume is None else f"{volume:g} L external cuboid"
            print(
                f"{item['file']}: {item['footprint_length_mm']:g} x "
                f"{item['footprint_width_mm']:g} x {item['height_mm']:g} mm; {suffix}"
            )
        return 0
    if not args.reference:
        parser.error("--reference is required unless --list is used")
    if args.samples < 1 or args.poll_interval_seconds < 0:
        parser.error("--samples must be positive and the poll interval cannot be negative")
    reference = find_reference(references, args.reference)

    try:
        import requests
    except ImportError:
        print("This tool needs requests (included in requirements-local.txt).")
        return 1

    observations: list[dict[str, Any]] = []
    attempts = 0
    maximum_attempts = max(args.samples * 4, args.samples + 6)
    while len(observations) < args.samples and attempts < maximum_attempts:
        attempts += 1
        try:
            response = requests.get(f"{args.host}/api/state", timeout=5)
            response.raise_for_status()
            state = response.json()
        except Exception as exc:  # noqa: BLE001 - interactive hardware utility
            print(f"Waiting for dashboard: {exc}")
            time.sleep(args.poll_interval_seconds)
            continue
        mode = state.get("operating_mode")
        if mode != "geometry_validation":
            print(
                "The dashboard is not in geometry_validation mode. Set "
                "LOCALLIFE_OPERATING_MODE=geometry_validation and restart it."
            )
            return 1
        observation = current_dimension_observation(state)
        if observation is None:
            print("Waiting for exactly one confirmed RealSense test object with LxWxH ...")
        else:
            observations.append(observation)
            print(
                f"Sample {len(observations)}/{args.samples}: "
                f"{observation['length_mm']:.1f} x {observation['width_mm']:.1f} x "
                f"{observation['height_mm']:.1f} mm"
            )
        if len(observations) < args.samples:
            time.sleep(args.poll_interval_seconds)
    if len(observations) < args.samples:
        print("Not enough valid samples. Keep one object still in the RealSense measurement area.")
        return 1

    estimated_length = statistics.median(item["length_mm"] for item in observations)
    estimated_width = statistics.median(item["width_mm"] for item in observations)
    estimated_height = statistics.median(item["height_mm"] for item in observations)
    # Footprint orientation can rotate in the image, so compare long side to
    # long side and short side to short side rather than image axes.
    actual_length, actual_width = sorted((
        float(reference["footprint_length_mm"]), float(reference["footprint_width_mm"]),
    ), reverse=True)
    estimated_length, estimated_width = sorted((estimated_length, estimated_width), reverse=True)
    length_error, length_percent = dimension_error(actual_length, estimated_length)
    width_error, width_percent = dimension_error(actual_width, estimated_width)
    height_error, height_percent = dimension_error(float(reference["height_mm"]), estimated_height)
    volumes = [float(item["volume_liters"]) for item in observations if item["volume_liters"] is not None]
    observed_volume = statistics.median(volumes) if volumes else None
    known_volume = reference["reference_volume_liters"]
    confidence_values = [
        float(item["confidence"]) for item in observations if item["confidence"] is not None
    ]
    method_values = [item["method"] for item in observations if item["method"]]

    print(f"\nReference: {reference['file']}")
    print(f"Ruler LxWxH:    {actual_length:.1f} x {actual_width:.1f} x {reference['height_mm']:.1f} mm")
    print(f"RealSense LxWxH:{estimated_length:.1f} x {estimated_width:.1f} x {estimated_height:.1f} mm")
    print(f"Errors: length {length_error:+.1f} mm ({length_percent:+.1f}%), "
          f"width {width_error:+.1f} mm ({width_percent:+.1f}%), "
          f"height {height_error:+.1f} mm ({height_percent:+.1f}%)")
    if known_volume is None:
        print("Volume truth: not provided; no volume-accuracy claim is made for this object.")
    elif observed_volume is not None:
        volume_error = (observed_volume - float(known_volume)) / float(known_volume) * 100.0
        print(f"External cuboid volume: {known_volume:.4f} L; observed {observed_volume:.4f} L ({volume_error:+.1f}%)")

    record = {
        "trial_id": f"dimension-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "timestamp": time.time(),
        "reference_file": reference["file"],
        "shape": reference.get("shape"),
        "production_expected_class": reference.get("expected_class"),
        "production_waste_accepted": reference["accepted"],
        "raw_label": statistics.mode([item["raw_label"] for item in observations]),
        "predicted_color": statistics.mode([item["color"] for item in observations]),
        "actual_length_mm": actual_length,
        "estimated_length_mm": estimated_length,
        "length_error_mm": length_error,
        "length_error_percent": length_percent,
        "actual_width_mm": actual_width,
        "estimated_width_mm": estimated_width,
        "width_error_mm": width_error,
        "width_error_percent": width_percent,
        "actual_height_mm": float(reference["height_mm"]),
        "estimated_height_mm": estimated_height,
        "height_error_mm": height_error,
        "height_error_percent": height_percent,
        "known_volume_liters": known_volume,
        "observed_volume_liters": observed_volume,
        "sample_count": len(observations),
        "dimension_confidence": statistics.median(confidence_values) if confidence_values else None,
        "dimension_method": statistics.mode(method_values) if method_values else None,
        "dimension_flags": sorted({flag for item in observations for flag in item["flags"]}),
        "build_version": observations[-1]["build_version"],
        "operating_mode": observations[-1]["operating_mode"],
        "placement": args.placement,
        "notes": args.notes,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"Trial saved to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
