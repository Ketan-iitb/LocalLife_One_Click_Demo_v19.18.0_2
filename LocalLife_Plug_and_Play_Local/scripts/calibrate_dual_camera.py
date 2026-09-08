#!/usr/bin/env python3
"""Capture synchronized checkerboard views and solve the RealSense<->Logitech calibration.

WHAT THIS IS FOR
-----------------
Phase 1 of the "System 2" dual-camera fusion feature: this produces the
calibration file (`dual_camera_calibration.json` by default) that a future
pipeline update will use to project the Logitech's depth into the
RealSense's coordinate frame (RealSense stays the primary/trusted metric
source; Logitech only fills gaps the RealSense could not see). Run this
ONCE after both cameras are mounted on the Raspberry Pi rig, and again any
time either camera is physically remounted -- the rig is otherwise rigid,
so the result does not change run to run.

WHAT YOU NEED
-------------
1. The local dashboard already running (`RUN_EVERYTHING_ONE_CLICK.cmd`, or
   `Start-PlugAndPlay-Experiment.ps1 -Role Local`) so this script can pull
   live frames from it over HTTP.
2. A printed checkerboard pattern with known, precisely-measured square
   size. A common one is a 10x7-square board (9x6 *internal* corners) --
   print at 100% scale (disable "fit to page" in your PDF viewer), then
   measure one square with a ruler and pass the real value in meters via
   --square-size-m. Getting this measurement right matters: it is the only
   thing that gives the whole calibration real-world scale.
3. A rigid, flat mount for the board (taped to a clipboard or piece of
   cardboard) so it does not flex while you move it.

HOW TO CAPTURE
---------------
Run this script, then slowly move/tilt the board so it is visible to BOTH
cameras at once, holding it still for a moment in each position. It
auto-captures a pair whenever both cameras see the board simultaneously,
then pauses briefly before looking for the next pose -- move the board to a
new position and angle during that pause. Aim for real variety: different
distances, different tilts left/right and up/down, and coverage of the
corners of both cameras' shared field of view, not just dead-center. 15-20
good pairs is a reasonable target; the script tells you when it has enough
and reports how well the fit converged.

USAGE
-----
    python scripts/calibrate_dual_camera.py \\
        --square-size-m 0.024 \\
        --columns 9 --rows 6 \\
        --pairs 18 \\
        --output data/dual_camera_calibration.json

Run with --help for every option (host/port, camera ids, save-images dir).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locallife_cloud.calibration import (  # noqa: E402
    calibrate_dual_camera,
    default_calibration_path,
    find_checkerboard_corners,
)


def _fetch_frame(session: "requests.Session", host: str, camera_id: str) -> np.ndarray | None:
    import cv2

    try:
        response = session.get(f"{host}/api/cameras/{camera_id}/raw-snapshot.jpg", timeout=5)
    except Exception as exc:  # noqa: BLE001 - this is a CLI tool; report and keep polling
        print(f"  ! could not reach {camera_id} at {host}: {exc}")
        return None
    if response.status_code != 200:
        return None
    data = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://127.0.0.1:8100", help="Base URL of the running local dashboard")
    parser.add_argument("--primary-camera", default="realsense", help="Camera id treated as the metric reference frame")
    parser.add_argument("--secondary-camera", default="logitech", help="Camera id whose depth will be reprojected into the primary frame")
    parser.add_argument("--columns", type=int, required=True, help="Internal checkerboard corners across (e.g. 9 for a 10-square-wide board)")
    parser.add_argument("--rows", type=int, required=True, help="Internal checkerboard corners down (e.g. 6 for a 7-square-tall board)")
    parser.add_argument("--square-size-m", type=float, required=True, help="Real, measured size of one checkerboard square, in meters")
    parser.add_argument("--pairs", type=int, default=15, help="How many synchronized checkerboard pairs to capture before solving")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.4, help="How often to check both cameras while waiting for the board")
    parser.add_argument("--cooldown-seconds", type=float, default=2.5, help="Pause after each capture so you can move the board to a new pose")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the calibration file (default: data/dual_camera_calibration.json)")
    parser.add_argument("--save-images", type=Path, default=None, help="Optional directory to also save each captured raw image pair, for your own audit trail")
    parser.add_argument("--max-reprojection-error-px", type=float, default=1.5, help="Warn (not fail) if the solved fit's reprojection error exceeds this")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("This tool needs the 'requests' package (already in requirements-local.txt).")
        return 1

    output_path = args.output or default_calibration_path(Path("data"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_images is not None:
        args.save_images.mkdir(parents=True, exist_ok=True)

    pattern_size = (args.columns, args.rows)
    session = requests.Session()

    primary_points: list[np.ndarray] = []
    secondary_points: list[np.ndarray] = []
    primary_shape: tuple[int, int] | None = None
    secondary_shape: tuple[int, int] | None = None

    print(f"Looking for a {args.columns}x{args.rows}-corner checkerboard in both "
          f"'{args.primary_camera}' and '{args.secondary_camera}' at {args.host} ...")
    print("Move the board into view of both cameras and hold it still. Press Ctrl+C to stop early.\n")

    last_status_print = 0.0
    try:
        while len(primary_points) < args.pairs:
            primary_frame = _fetch_frame(session, args.host, args.primary_camera)
            secondary_frame = _fetch_frame(session, args.host, args.secondary_camera)
            if primary_frame is None or secondary_frame is None:
                time.sleep(args.poll_interval_seconds)
                continue

            primary_corners = find_checkerboard_corners(primary_frame, pattern_size)
            secondary_corners = find_checkerboard_corners(secondary_frame, pattern_size)

            if primary_corners is not None and secondary_corners is not None:
                if primary_shape is None:
                    primary_shape = primary_frame.shape[:2]
                    secondary_shape = secondary_frame.shape[:2]
                if primary_frame.shape[:2] != primary_shape or secondary_frame.shape[:2] != secondary_shape:
                    print("  ! camera resolution changed mid-capture; skipping this pair")
                    time.sleep(args.poll_interval_seconds)
                    continue

                primary_points.append(primary_corners)
                secondary_points.append(secondary_corners)
                index = len(primary_points)
                print(f"  captured pair {index}/{args.pairs} -- now move the board to a new position/angle")

                if args.save_images is not None:
                    import cv2

                    cv2.imwrite(str(args.save_images / f"{index:02d}_{args.primary_camera}.jpg"), primary_frame)
                    cv2.imwrite(str(args.save_images / f"{index:02d}_{args.secondary_camera}.jpg"), secondary_frame)

                time.sleep(args.cooldown_seconds)
            else:
                now = time.monotonic()
                if now - last_status_print > 2.0:
                    seen = []
                    if primary_corners is not None:
                        seen.append(args.primary_camera)
                    if secondary_corners is not None:
                        seen.append(args.secondary_camera)
                    where = f"seen in: {', '.join(seen)}" if seen else "not seen in either view"
                    print(f"  waiting for the board in both views ({where})")
                    last_status_print = now
                time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        print(f"\nStopped early with {len(primary_points)} pair(s) captured.")

    if len(primary_points) < 4:
        print(f"\nOnly {len(primary_points)} pair(s) captured; at least 4 are needed and 12+ recommended. "
              "Run again and hold the board still for longer in each position.")
        return 1

    print(f"\nSolving calibration from {len(primary_points)} pairs ...")
    result = calibrate_dual_camera(
        primary_camera_id=args.primary_camera,
        primary_image_points=primary_points,
        secondary_image_points=secondary_points,
        pattern_size=pattern_size,
        square_size_m=args.square_size_m,
        primary_image_shape=primary_shape,
        secondary_image_shape=secondary_shape,
    )
    if result is None:
        print("Calibration failed to converge. Recapture with more pairs and more pose variety.")
        return 1

    print(f"Primary ({args.primary_camera}) reprojection error: {result.primary.reprojection_error_px:.3f} px")
    print(f"Secondary ({args.secondary_camera}) reprojection error: {result.secondary.reprojection_error_px:.3f} px")
    print(f"Stereo (relative pose) reprojection error: {result.stereo_reprojection_error_px:.3f} px")
    if result.stereo_reprojection_error_px > args.max_reprojection_error_px:
        print(f"\nWARNING: stereo reprojection error is above {args.max_reprojection_error_px} px -- "
              "this fit is saved, but consider recapturing with more pose variety (different distances, "
              "tilts, and positions across the shared field of view) for a tighter result.")

    result.save(output_path)
    print(f"\nSaved calibration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
