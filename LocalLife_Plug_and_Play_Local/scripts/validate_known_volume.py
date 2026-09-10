#!/usr/bin/env python3
"""Validate real measured volume accuracy against a known-volume reference object.

WHAT THIS IS FOR
-----------------
The user's own thesis proposal names this as the core accuracy-validation
method: place an object of a precisely known real volume (their proposal's
own reference objects are a 1 L cube and a 2 L box) in front of a camera,
compare the system's reported liters against that known ground truth, and
track percent error across repeated trials -- median absolute percentage
error and 90th-percentile error, the same shape of statistic this script
reports.

This tool does NOT reimplement any volume math of its own. It talks to the
already-running local dashboard over its existing HTTP API (the same way
`scripts/calibrate_dual_camera.py` does) and reads back whatever
`VisionPipeline`/`estimate_volume`/`calibrate_known_volume` already
computed -- it only adds the bookkeeping (known-vs-observed comparison,
percent error, and aggregate statistics across trials) that those pieces do
not do on their own.

WHAT YOU NEED
-------------
1. The local dashboard already running (START_LOCAL_LIFE_DEMO.cmd, or
   `Start-LocalLife-Demo.ps1 -Mode Local`), with an empty-bin baseline
   already captured on the camera you are validating.
2. A reference object of a real, precisely known volume -- ideally measured
   independently of this system (e.g. a 1.000 L measuring cube, or a box
   whose internal dimensions you measured with a ruler and multiplied out).
   Do not eyeball "about 1 liter"; the whole point of this tool is an
   honest number to compare against.

HOW TO USE IT
-------------
Basic measurement (no calibration, just record how accurate the system
already is for this object):

    python scripts/validate_known_volume.py --camera realsense \\
        --known-liters 1.0 --label "1L-cube"

Run it again with the 2 L reference object:

    python scripts/validate_known_volume.py --camera realsense \\
        --known-liters 2.0 --label "2L-box"

Run it several times (ideally moving the object slightly, or across
separate sessions) to build up trials, then see the aggregate accuracy
picture -- MAPE, median absolute percentage error, and the 90th-percentile
error across every trial recorded so far for that camera:

    python scripts/validate_known_volume.py --camera realsense --report-only

To also SOLVE a per-installation calibration factor from a trial (this
calls the existing `/api/cameras/<id>/calibrate-volume` endpoint --
`calibrate_known_volume()` -- it does not compute anything itself), add
--calibrate. Per that endpoint's own guidance: calibrate once against one
object, then VALIDATE with a *different* object you did not calibrate
against -- fitting and validating against the same object will always look
perfect and proves nothing about general accuracy.

    python scripts/validate_known_volume.py --camera realsense \\
        --known-liters 1.0 --label "1L-cube-calibration" --calibrate

Every measured trial is appended to a small JSONL log
(default: data/known_volume_validation.jsonl) so accuracy can be tracked
over many sessions, not just the current run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _default_log_path() -> Path:
    return Path("data") / "known_volume_validation.jsonl"


ACCEPTED_CLASSES = {"plastic_bag", "paper_bag", "cardboard_box"}
PHANTOM_SOURCES = {
    "fixed-bin-depth-silhouette", "depth-scene-segmentation", "foreground-segmentation",
}
CSV_FIELDS = (
    "trial_id", "timestamp", "role", "camera", "label", "object_type",
    "predicted_object_type", "object_type_correct", "raw_label",
    "actual_color", "predicted_color", "color_correct", "known_liters",
    "observed_liters", "absolute_error_liters", "percent_error", "sample_count",
    "sample_min_liters", "sample_max_liters", "sample_std_liters",
    "actual_length_mm", "estimated_length_mm", "length_error_mm", "length_error_percent",
    "actual_width_mm", "estimated_width_mm", "width_error_mm", "width_error_percent",
    "actual_height_mm", "estimated_height_mm", "height_error_mm", "height_error_percent",
    "dimension_confidence", "depth_coverage_percent", "detection_confidence",
    "dimension_method", "dimension_flags", "measurement_method", "measurement_quality",
    "volume_uncertainty_l", "volume_calibration_factor_before", "inference_ms",
    "placement", "notes", "calibrated", "calibration_previous_factor", "calibration_new_factor",
)


def _fetch_state(session: "requests.Session", host: str) -> dict[str, Any] | None:
    try:
        response = session.get(f"{host}/api/state", timeout=5)
    except Exception as exc:  # noqa: BLE001 - CLI tool: report and keep polling
        print(f"  ! could not reach {host}: {exc}")
        return None
    if response.status_code != 200:
        print(f"  ! {host}/api/state returned HTTP {response.status_code}")
        return None
    return response.json()


def _current_observation(state: dict[str, Any], camera: str) -> dict[str, Any] | None:
    """Return one trustworthy live object plus its measurement diagnostics.

    A RealSense validation trial must contain exactly one accepted, confirmed
    non-phantom object. This matches the safety rule used by
    ``VisionPipeline.calibrate_known_volume`` and prevents a combined scene or
    unidentified depth silhouette from becoming calibration evidence.
    """
    if camera == "fused":
        fused = state.get("fused") or {}
        liters = fused.get("volume_l") if fused.get("available") else None
        if liters is None:
            return None
        return {
            "observed_liters": float(liters),
            "object_type": fused.get("object_type"),
            "predicted_color": fused.get("color"),
            "measurement_method": "camera-fusion-comparison",
        }

    cameras = state.get("cameras") or {}
    camera_state = cameras.get(camera) or {}
    latest = camera_state.get("latest") or {}
    key = "monocular_volume_l" if camera == "logitech" else "realsense_volume_l"

    if camera == "realsense":
        candidates = [
            item for item in (latest.get("detections") or [])
            if item.get("tracking_status") in {"confirmed", "predicted"}
            and item.get("accepted_class") in ACCEPTED_CLASSES
            and item.get("source") not in PHANTOM_SOURCES
            and item.get(key) is not None
        ]
        if len(candidates) != 1:
            return None
        item = candidates[0]
        dimensions = item.get("dimensions_mm") or {}
        return {
            "observed_liters": float(item[key]),
            "object_type": item.get("accepted_class"),
            "raw_label": item.get("label"),
            "predicted_color": item.get("color"),
            "estimated_length_mm": dimensions.get("footprint_length"),
            "estimated_width_mm": dimensions.get("footprint_width"),
            "estimated_height_mm": dimensions.get("height"),
            "dimension_confidence": item.get("dimension_confidence"),
            "dimension_flags": item.get("dimension_flags") or [],
            "dimension_method": item.get("dimension_method"),
            "depth_coverage_percent": item.get("depth_coverage_percent"),
            "detection_confidence": item.get("confidence"),
            "measurement_method": item.get("measurement_method"),
            "measurement_quality": item.get("measurement_quality"),
            "volume_uncertainty_l": item.get("volume_uncertainty_l"),
            "volume_calibration_factor": camera_state.get("volume_calibration_factor"),
            "inference_ms": latest.get("inference_ms"),
        }

    liters = latest.get(key)
    if liters is None:
        return None
    return {
        "observed_liters": float(liters),
        "measurement_method": "monocular-comparison-only",
        "volume_calibration_factor": camera_state.get("volume_calibration_factor"),
        "inference_ms": latest.get("inference_ms"),
    }


def _current_reading(state: dict[str, Any], camera: str) -> float | None:
    """Read whatever `VisionPipeline`/`DualCameraCoordinator` already computed.

    This never derives a liters figure itself -- it only picks the right
    already-computed field out of the existing `/api/state` payload, exactly
    the number the dashboard itself would show for the current object.
    """
    observation = _current_observation(state, camera)
    return None if observation is None else float(observation["observed_liters"])


def _collect_observations(
    session: "requests.Session", host: str, camera: str, count: int, interval: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(count * 3, count + 5)
    print(f"Reading {count} live sample(s) from '{camera}' at {host} ...")
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        state = _fetch_state(session, host)
        if state is not None:
            observation = _current_observation(state, camera)
            if observation is not None:
                samples.append(observation)
                print(f"  sample {len(samples)}/{count}: {observation['observed_liters']:.4f} L")
            else:
                print("  waiting for exactly one accepted, confirmed, measured object in view ...")
        time.sleep(interval)
    return samples


def _collect_samples(
    session: "requests.Session", host: str, camera: str, count: int, interval: float,
) -> list[float]:
    """Backward-compatible liters-only wrapper used by older callers."""
    return [
        float(item["observed_liters"])
        for item in _collect_observations(session, host, camera, count, interval)
    ]


def _median_numeric(observations: list[dict[str, Any]], field: str) -> float | None:
    values = [float(item[field]) for item in observations if item.get(field) is not None]
    return statistics.median(values) if values else None


def _common_value(observations: list[dict[str, Any]], field: str) -> Any:
    values = [item.get(field) for item in observations if item.get(field) not in (None, "")]
    return statistics.mode(values) if values else None


def _dimension_error(actual: float | None, estimated: float | None) -> float | None:
    return None if actual is None or estimated is None else estimated - actual


def _dimension_error_percent(actual: float | None, estimated: float | None) -> float | None:
    error = _dimension_error(actual, estimated)
    return None if error is None or actual == 0 else error / actual * 100.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _load_trials(
    log_path: Path, camera: str | None, role: str | None = None,
) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    trials = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_role = record.get("role", "validation")
            if (camera is None or record.get("camera") == camera) and (role is None or record_role == role):
                trials.append(record)
    return trials


def _export_csv(trials: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in trials:
            row = dict(item)
            flags = row.get("dimension_flags")
            if isinstance(flags, list):
                row["dimension_flags"] = "|".join(str(value) for value in flags)
            writer.writerow(row)


def _print_aggregate_report(trials: list[dict[str, Any]], camera: str, role: str = "validation") -> None:
    print(f"\n=== {role.title()} summary for '{camera}' ({len(trials)} trial(s) recorded) ===")
    if not trials:
        print("No trials recorded yet. Run this script with --known-liters to record one.")
        return
    errors = [abs(float(item["percent_error"])) for item in trials if item.get("percent_error") is not None]
    if not errors:
        print("Recorded trials have no usable percent-error values.")
        return
    mape = statistics.fmean(errors)
    median_ape = statistics.median(errors)
    p90_ape = _percentile(errors, 0.90)
    print(f"  MAPE (mean absolute percentage error):   {mape:6.2f}%")
    print(f"  Median absolute percentage error:        {median_ape:6.2f}%")
    print(f"  90th-percentile absolute percentage error:{p90_ape:6.2f}%")
    print(f"  Best trial:  {min(errors):6.2f}%   Worst trial: {max(errors):6.2f}%")
    print("\n  Per-trial detail:")
    for item in trials[-15:]:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("timestamp", 0)))
        calibrated_note = " (calibration run)" if item.get("calibrated") else ""
        known = item.get("known_liters")
        observed = item.get("observed_liters")
        error = item.get("percent_error")
        if known is None or observed is None or error is None:
            continue
        print(
            f"    {stamp}  {item.get('label', '(unlabeled)'):<18} "
            f"known={known:.3f} L  observed={observed:.3f} L  "
            f"error={error:+.1f}%{calibrated_note}"
        )
    if len(trials) > 15:
        print(f"    ... and {len(trials) - 15} earlier trial(s) not shown above")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://127.0.0.1:8100", help="Base URL of the running local dashboard")
    parser.add_argument("--camera", choices=("realsense", "logitech", "fused"), default="realsense",
                         help="Which reading to validate: a single camera, or the combined 'fused' result")
    parser.add_argument("--known-liters", type=float, default=None,
                         help="The reference object's real, independently-measured volume in liters "
                              "(e.g. 1.0 for a 1 L cube, 2.0 for a 2 L box). Required unless --report-only.")
    parser.add_argument("--label", default=None, help="Short name for this trial/object (e.g. '1L-cube')")
    parser.add_argument("--trial-id", default=None, help="Stable test ID; generated automatically when omitted")
    parser.add_argument("--role", choices=("validation", "calibration"), default="validation",
                         help="Keep factor-fitting trials separate from independent validation trials")
    parser.add_argument("--object-type", choices=tuple(sorted(ACCEPTED_CLASSES)), default=None,
                         help="Ground-truth waste class; otherwise use the RealSense accepted class")
    parser.add_argument("--actual-color", default=None, help="Ground-truth colour observed by the tester")
    parser.add_argument("--actual-length-mm", type=float, default=None,
                         help="Ruler-measured filled-object footprint length")
    parser.add_argument("--actual-width-mm", type=float, default=None,
                         help="Ruler-measured filled-object footprint width")
    parser.add_argument("--actual-height-mm", type=float, default=None,
                         help="Ruler-measured filled-object height")
    parser.add_argument("--placement", default=None, help="Placement note, e.g. centre, left, right, rotated")
    parser.add_argument("--notes", default=None, help="Lighting, fill material, or failure/setup notes")
    parser.add_argument("--samples", type=int, default=5,
                         help="Live readings to poll and take the median of for this trial (default: 5)")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0,
                         help="Delay between live samples while the object sits still in view")
    parser.add_argument("--calibrate", action="store_true",
                         help="Also solve a per-installation calibration factor from this trial via the "
                              "existing calibrate-volume endpoint (not valid with --camera fused)")
    parser.add_argument("--log", type=Path, default=None,
                         help="JSONL file trials accumulate in (default: data/known_volume_validation.jsonl)")
    parser.add_argument("--csv", type=Path, default=None,
                         help="CSV report path (default: same location/name as --log with .csv suffix)")
    parser.add_argument("--report-only", action="store_true",
                         help="Skip taking a new measurement; just print the aggregate accuracy report "
                              "from trials already recorded for --camera")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("This tool needs the 'requests' package (already in requirements-local.txt).")
        return 1

    log_path = args.log or _default_log_path()
    csv_path = args.csv or log_path.with_suffix(".csv")

    if not args.report_only:
        if args.known_liters is None:
            print("--known-liters is required unless --report-only is set.")
            return 1
        if args.known_liters <= 0:
            print("--known-liters must be a positive number.")
            return 1
        for name in ("actual_length_mm", "actual_width_mm", "actual_height_mm"):
            value = getattr(args, name)
            if value is not None and value <= 0:
                print(f"--{name.replace('_', '-')} must be a positive number when supplied.")
                return 1
        if args.calibrate and args.camera != "realsense":
            print("--calibrate is supported only for RealSense, the final metric-volume authority.")
            return 1
        if args.calibrate and args.role != "calibration":
            print("Use --role calibration with --calibrate so fitted trials cannot be mistaken for validation evidence.")
            return 1

        session = requests.Session()
        observations = _collect_observations(
            session, args.host, args.camera, args.samples, args.poll_interval_seconds,
        )
        if not observations:
            print(
                f"\nNo confirmed, measured reading was available on '{args.camera}' during this run. "
                "Make sure the empty-bin baseline is captured, the reference object is the only thing "
                "in view, and its liters value is already showing on the dashboard, then try again."
            )
            return 1

        samples = [float(item["observed_liters"]) for item in observations]
        observed_liters = statistics.median(samples)
        percent_error = (observed_liters - args.known_liters) / args.known_liters * 100.0
        print(f"\nKnown volume:    {args.known_liters:.4f} L")
        print(f"Observed volume: {observed_liters:.4f} L  (median of {len(samples)} sample(s), "
              f"range {min(samples):.4f}-{max(samples):.4f} L)")
        print(f"Percent error:   {percent_error:+.2f}%")

        estimated_length = _median_numeric(observations, "estimated_length_mm")
        estimated_width = _median_numeric(observations, "estimated_width_mm")
        estimated_height = _median_numeric(observations, "estimated_height_mm")
        if all(value is not None for value in (estimated_length, estimated_width, estimated_height)):
            print(
                f"RealSense L×W×H: {estimated_length:.1f} × {estimated_width:.1f} × "
                f"{estimated_height:.1f} mm"
            )
        else:
            print("RealSense L×W×H: unavailable in one or more samples; inspect dimension flags/coverage.")
        if all(value is not None for value in (
            args.actual_length_mm, args.actual_width_mm, args.actual_height_mm,
        )):
            print(
                f"Measured L×W×H:  {args.actual_length_mm:.1f} × {args.actual_width_mm:.1f} × "
                f"{args.actual_height_mm:.1f} mm"
            )

        calibrated = False
        calibration_previous_factor = None
        calibration_new_factor = None
        if args.calibrate:
            try:
                response = session.post(
                    f"{args.host}/api/cameras/{args.camera}/calibrate-volume",
                    json={"known_liters": args.known_liters},
                    timeout=5,
                )
                data = response.json()
                if response.status_code != 200:
                    print(f"\nCalibration request failed: {data.get('error', 'unknown error')}")
                else:
                    calibration = data["calibration"]
                    calibrated = True
                    calibration_previous_factor = calibration["previous_factor"]
                    calibration_new_factor = calibration["factor"]
                    factor_info = (
                        f" (factor {calibration['previous_factor']:.4f} -> {calibration['factor']:.4f}, "
                        f"solved from its own observed {calibration['observed_liters']:.4f} L)"
                    )
                    print(f"\nCalibrated '{args.camera}'{factor_info}")
                    print(calibration.get("warning", ""))
            except Exception as exc:  # noqa: BLE001 - CLI tool: report and continue
                print(f"\nCalibration request failed: {exc}")

        predicted_color = _common_value(observations, "predicted_color")
        detected_type = _common_value(observations, "object_type")
        object_type = args.object_type or detected_type
        if args.object_type is not None and detected_type != args.object_type:
            print(
                f"WARNING: ground-truth type '{args.object_type}' did not match detected type "
                f"'{detected_type}'. The mismatch is retained in the trial record."
            )
        record = {
            "trial_id": args.trial_id or f"trial-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "timestamp": time.time(),
            "role": args.role,
            "camera": args.camera,
            "label": args.label or f"{args.known_liters:g}L",
            "object_type": object_type,
            "detected_object_type": detected_type,
            "predicted_object_type": detected_type,
            "object_type_correct": None if args.object_type is None or detected_type is None else (
                args.object_type == detected_type
            ),
            "raw_label": _common_value(observations, "raw_label"),
            "actual_color": args.actual_color,
            "predicted_color": predicted_color,
            "color_correct": None if args.actual_color is None or predicted_color is None else (
                args.actual_color.strip().lower() == str(predicted_color).strip().lower()
            ),
            "known_liters": args.known_liters,
            "observed_liters": observed_liters,
            "absolute_error_liters": abs(observed_liters - args.known_liters),
            "percent_error": percent_error,
            "sample_count": len(samples),
            "sample_min_liters": min(samples),
            "sample_max_liters": max(samples),
            "sample_std_liters": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            "sample_values_liters": samples,
            "actual_length_mm": args.actual_length_mm,
            "estimated_length_mm": estimated_length,
            "length_error_mm": _dimension_error(args.actual_length_mm, estimated_length),
            "length_error_percent": _dimension_error_percent(args.actual_length_mm, estimated_length),
            "actual_width_mm": args.actual_width_mm,
            "estimated_width_mm": estimated_width,
            "width_error_mm": _dimension_error(args.actual_width_mm, estimated_width),
            "width_error_percent": _dimension_error_percent(args.actual_width_mm, estimated_width),
            "actual_height_mm": args.actual_height_mm,
            "estimated_height_mm": estimated_height,
            "height_error_mm": _dimension_error(args.actual_height_mm, estimated_height),
            "height_error_percent": _dimension_error_percent(args.actual_height_mm, estimated_height),
            "dimension_confidence": _median_numeric(observations, "dimension_confidence"),
            "dimension_flags": sorted({
                str(flag) for item in observations for flag in (item.get("dimension_flags") or [])
            }),
            "dimension_method": _common_value(observations, "dimension_method"),
            "depth_coverage_percent": _median_numeric(observations, "depth_coverage_percent"),
            "detection_confidence": _median_numeric(observations, "detection_confidence"),
            "measurement_method": _common_value(observations, "measurement_method"),
            "measurement_quality": _common_value(observations, "measurement_quality"),
            "volume_uncertainty_l": _median_numeric(observations, "volume_uncertainty_l"),
            "volume_calibration_factor_before": _median_numeric(
                observations, "volume_calibration_factor",
            ),
            "inference_ms": _median_numeric(observations, "inference_ms"),
            "placement": args.placement,
            "notes": args.notes,
            "calibrated": calibrated,
            "calibration_previous_factor": calibration_previous_factor,
            "calibration_new_factor": calibration_new_factor,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"\nTrial recorded to {log_path}")

    camera_trials = _load_trials(log_path, args.camera)
    trials = _load_trials(log_path, args.camera, args.role)
    _export_csv(camera_trials, csv_path)
    print(f"CSV report written to {csv_path}")
    _print_aggregate_report(trials, args.camera, args.role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
