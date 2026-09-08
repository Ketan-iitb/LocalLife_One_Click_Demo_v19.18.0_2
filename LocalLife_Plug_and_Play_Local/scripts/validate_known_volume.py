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

Every trial (measured or report-only run) is appended to a small JSONL log
(default: data/known_volume_validation.jsonl) so accuracy can be tracked
over many sessions, not just the current run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _default_log_path() -> Path:
    return Path("data") / "known_volume_validation.jsonl"


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


def _current_reading(state: dict[str, Any], camera: str) -> float | None:
    """Read whatever `VisionPipeline`/`DualCameraCoordinator` already computed.

    This never derives a liters figure itself -- it only picks the right
    already-computed field out of the existing `/api/state` payload, exactly
    the number the dashboard itself would show for the current object.
    """
    if camera == "fused":
        fused = state.get("fused") or {}
        if not fused.get("available"):
            return None
        return fused.get("volume_l")

    cameras = state.get("cameras") or {}
    camera_state = cameras.get(camera) or {}
    latest = camera_state.get("latest") or {}
    key = "monocular_volume_l" if camera == "logitech" else "realsense_volume_l"
    return latest.get(key)


def _collect_samples(
    session: "requests.Session", host: str, camera: str, count: int, interval: float,
) -> list[float]:
    samples: list[float] = []
    attempts = 0
    max_attempts = max(count * 3, count + 5)
    print(f"Reading {count} live sample(s) from '{camera}' at {host} ...")
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        state = _fetch_state(session, host)
        if state is not None:
            reading = _current_reading(state, camera)
            if reading is not None:
                samples.append(float(reading))
                print(f"  sample {len(samples)}/{count}: {reading:.4f} L")
            else:
                print("  waiting for a confirmed, measured object in view ...")
        time.sleep(interval)
    return samples


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


def _load_trials(log_path: Path, camera: str | None) -> list[dict[str, Any]]:
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
            if camera is None or record.get("camera") == camera:
                trials.append(record)
    return trials


def _print_aggregate_report(trials: list[dict[str, Any]], camera: str) -> None:
    print(f"\n=== Accuracy summary for '{camera}' ({len(trials)} trial(s) recorded) ===")
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
        print(
            f"    {stamp}  {item.get('label', '(unlabeled)'):<18} "
            f"known={item.get('known_liters'):.3f} L  observed={item.get('observed_liters'):.3f} L  "
            f"error={item.get('percent_error'):+.1f}%{calibrated_note}"
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
    parser.add_argument("--samples", type=int, default=5,
                         help="Live readings to poll and take the median of for this trial (default: 5)")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0,
                         help="Delay between live samples while the object sits still in view")
    parser.add_argument("--calibrate", action="store_true",
                         help="Also solve a per-installation calibration factor from this trial via the "
                              "existing calibrate-volume endpoint (not valid with --camera fused)")
    parser.add_argument("--log", type=Path, default=None,
                         help="JSONL file trials accumulate in (default: data/known_volume_validation.jsonl)")
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

    if not args.report_only:
        if args.known_liters is None:
            print("--known-liters is required unless --report-only is set.")
            return 1
        if args.known_liters <= 0:
            print("--known-liters must be a positive number.")
            return 1
        if args.calibrate and args.camera == "fused":
            print("--calibrate is not supported with --camera fused; calibrate 'realsense' or 'logitech' directly.")
            return 1

        session = requests.Session()
        samples = _collect_samples(session, args.host, args.camera, args.samples, args.poll_interval_seconds)
        if not samples:
            print(
                f"\nNo confirmed, measured reading was available on '{args.camera}' during this run. "
                "Make sure the empty-bin baseline is captured, the reference object is the only thing "
                "in view, and its liters value is already showing on the dashboard, then try again."
            )
            return 1

        observed_liters = statistics.median(samples)
        percent_error = (observed_liters - args.known_liters) / args.known_liters * 100.0
        print(f"\nKnown volume:    {args.known_liters:.4f} L")
        print(f"Observed volume: {observed_liters:.4f} L  (median of {len(samples)} sample(s), "
              f"range {min(samples):.4f}-{max(samples):.4f} L)")
        print(f"Percent error:   {percent_error:+.2f}%")

        calibrated = False
        factor_info = ""
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
                    factor_info = (
                        f" (factor {calibration['previous_factor']:.4f} -> {calibration['factor']:.4f}, "
                        f"solved from its own observed {calibration['observed_liters']:.4f} L)"
                    )
                    print(f"\nCalibrated '{args.camera}'{factor_info}")
                    print(calibration.get("warning", ""))
            except Exception as exc:  # noqa: BLE001 - CLI tool: report and continue
                print(f"\nCalibration request failed: {exc}")

        record = {
            "timestamp": time.time(),
            "camera": args.camera,
            "label": args.label or f"{args.known_liters:g}L",
            "known_liters": args.known_liters,
            "observed_liters": observed_liters,
            "percent_error": percent_error,
            "sample_count": len(samples),
            "calibrated": calibrated,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"\nTrial recorded to {log_path}")

    trials = _load_trials(log_path, args.camera)
    _print_aggregate_report(trials, args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
