"""Real-hardware acceptance check for the V26 paired comparison.

Run this against a live system: the laptop dashboard in local mode, or the
tunnelled cloud backend, with the Pi streaming both cameras. First place ONE
controlled object in the bin and wait until both cameras finalise it.

    python scripts/check_v26_acceptance.py --url http://127.0.0.1:8000 [--token TOKEN]

Each step prints PASS or FAIL with the evidence it saw. The script only reads
from the system: it never creates measurements, so passing it requires a real
object measured by the real cameras.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=None, help="LOCALLIFE_API_TOKEN, if the server requires one")
    parser.add_argument("--max-frame-age", type=float, default=3.0)
    args = parser.parse_args()
    headers = {"X-API-Token": args.token} if args.token else {}
    base = args.url.rstrip("/")
    failures = 0

    def check(name: str, passed: bool, evidence: object) -> None:
        nonlocal failures
        failures += 0 if passed else 1
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {evidence}")

    def get(path: str) -> requests.Response:
        return requests.get(base + path, headers=headers, timeout=15)

    health = get("/api/health")
    check("backend health", health.ok, health.status_code)
    state = get("/api/state").json()
    for camera in ("realsense", "logitech"):
        stream = (state.get("cameras", {}).get(camera) or {}).get("stream") or {}
        age = stream.get("last_received_age_s")
        check(f"{camera} frames arriving", age is not None and age < args.max_frame_age, f"last frame {age} s ago")
    calibration = get("/api/logitech/calibration").json()
    check("Logitech metric calibration", bool(calibration.get("metric_ready")),
          (calibration.get("active_calibration") or {}).get("method") or calibration.get("message"))
    check("Logitech lens profile", (calibration.get("lens") or {}).get("status") == "undistorted",
          calibration.get("lens"))

    events = get("/api/comparison/events?limit=1").json().get("events") or []
    latest = events[0] if events else {}
    cameras = latest.get("cameras") or {}
    check("latest comparison event has both cameras", set(cameras) >= {"realsense", "logitech"},
          f"{latest.get('comparison_event_id')}: {sorted(cameras)}")

    text = get("/api/comparison/measurements.csv").text.lstrip("﻿")
    rows = [row for row in csv.DictReader(io.StringIO(text))
            if row.get("comparison_event_id") == latest.get("comparison_event_id")]
    check("downloaded CSV contains that event's rows", len(rows) >= 2, f"{len(rows)} rows")
    for row in rows:
        empty = [column for column in ("camera_source", "length_mm", "width_mm", "height_mm",
                                       "selected_volume_litres", "geometry_method", "status")
                 if not row.get(column)]
        check(f"{row.get('camera_source')} row complete", not empty,
              f"{row.get('geometry_method')} {row.get('selected_volume_litres')} L" if not empty else f"empty: {empty}")

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
