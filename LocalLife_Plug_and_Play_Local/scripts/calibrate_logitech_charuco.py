"""One-time physical calibration of the fixed Logitech C920.

Two steps, both from the same printed ChArUco board:

  intrinsics  several views of the board held at different angles ->
              camera matrix, distortion, reprojection error, saved as the
              lens profile the pipeline already loads
              (results/logitech/calibration/logitech_lens.json)

  plane       one view of the board lying flat in the bin ->
              camera pose, camera-to-plane distance and a plane->image
              homography, saved as logitech_plane.json

Run at the exact resolution the demo uses; a profile is refused at any other
resolution, and the camera, zoom and focus must not move afterwards.

    python scripts/calibrate_logitech_charuco.py intrinsics --camera 0 --width 1280 --height 720
    python scripts/calibrate_logitech_charuco.py intrinsics --images captures/*.png
    python scripts/calibrate_logitech_charuco.py plane --camera 0 --width 1280 --height 720

Needs the ArUco module: pip install opencv-contrib-python-headless
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M, MARKER_LENGTH_M = 0.04, 0.03
MAX_REPROJECTION_ERROR_PX = 1.0


def _aruco():
    import cv2

    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "OpenCV was built without the ArUco module.\n"
            "  pip install opencv-contrib-python-headless"
        )
    return cv2, cv2.aruco


def _board(aruco):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:
        return aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary), dictionary
    except AttributeError:  # OpenCV 4.6 and older
        return aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary), dictionary


def _detect(cv2, aruco, board, dictionary, image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(grey, dictionary)
    if ids is None or len(ids) < 4:
        return None, None
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, grey, board)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None
    return charuco_corners, charuco_ids


def _frames(arguments, cv2) -> list[np.ndarray]:
    if arguments.images:
        paths = sorted(path for pattern in arguments.images for path in glob.glob(pattern))
        if not paths:
            raise SystemExit("No images matched the given patterns")
        return [cv2.imread(path) for path in paths]
    capture = cv2.VideoCapture(arguments.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, arguments.height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {arguments.camera}")
    frames: list[np.ndarray] = []
    print(f"Capturing {arguments.views} views: move the board between captures.")
    try:
        for index in range(arguments.views):
            for _ in range(int(arguments.settle_frames)):
                capture.read()
            ok, image = capture.read()
            if not ok:
                raise SystemExit("The camera stopped returning frames")
            frames.append(image)
            print(f"  view {index + 1}/{arguments.views} captured")
            time.sleep(arguments.delay)
    finally:
        capture.release()
    return frames


def calibrate_intrinsics(arguments) -> int:
    cv2, aruco = _aruco()
    board, dictionary = _board(aruco)
    frames = _frames(arguments, cv2)
    all_corners, all_ids, shape = [], [], None
    for image in frames:
        if image is None:
            continue
        shape = image.shape[:2]
        corners, ids = _detect(cv2, aruco, board, dictionary, image)
        if corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)
    print(f"Board found in {len(all_corners)} of {len(frames)} views")
    if len(all_corners) < 5 or shape is None:
        raise SystemExit("At least five views with a visible board are needed")
    error, matrix, distortion, _, _ = aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, (shape[1], shape[0]), None, None,
    )
    print(f"Reprojection error: {error:.3f} px")
    if error > MAX_REPROJECTION_ERROR_PX:
        raise SystemExit(f"Reprojection error above {MAX_REPROJECTION_ERROR_PX} px; recapture with sharper views")
    profile = {
        "intrinsics": {"fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
                       "ppx": float(matrix[0, 2]), "ppy": float(matrix[1, 2]),
                       "width": int(shape[1]), "height": int(shape[0])},
        "distortion": [float(value) for value in np.asarray(distortion).reshape(-1)],
        "reprojection_error_px": float(error),
        "image_count": len(all_corners),
        "calibrated_at": time.time(),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Lens profile written to {output}")
    return 0


def calibrate_plane(arguments) -> int:
    cv2, aruco = _aruco()
    board, dictionary = _board(aruco)
    profile = json.loads(Path(arguments.lens).read_text(encoding="utf-8"))
    intrinsics = profile["intrinsics"]
    matrix = np.array([[intrinsics["fx"], 0, intrinsics["ppx"]],
                       [0, intrinsics["fy"], intrinsics["ppy"]], [0, 0, 1]], dtype=np.float64)
    distortion = np.array(profile["distortion"], dtype=np.float64)
    frames = _frames(argparse.Namespace(**{**vars(arguments), "views": 1}), cv2)
    image = frames[0]
    if image is None:
        raise SystemExit("No frame captured")
    if (image.shape[1], image.shape[0]) != (intrinsics["width"], intrinsics["height"]):
        raise SystemExit(
            f"Lens profile is for {intrinsics['width']}x{intrinsics['height']}, "
            f"this capture is {image.shape[1]}x{image.shape[0]}: recalibrate at the runtime resolution"
        )
    corners, ids = _detect(cv2, aruco, board, dictionary, image)
    if corners is None:
        raise SystemExit("The board was not found; place it flat in the bin, fully visible")
    ok, rvec, tvec = aruco.estimatePoseCharucoBoard(corners, ids, board, matrix, distortion, None, None)
    if not ok:
        raise SystemExit("Board pose could not be estimated")
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    distance = float(abs(np.dot(normal, tvec.reshape(3))))
    # Plane (z=0 of the board) -> image homography, for footprint drawing.
    homography = matrix @ np.column_stack((rotation[:, 0], rotation[:, 1], tvec.reshape(3)))
    payload = {
        "camera_to_plane_m": distance,
        "plane_normal_camera": [float(value) for value in normal],
        "rotation": [[float(value) for value in row] for row in rotation],
        "translation_m": [float(value) for value in tvec.reshape(3)],
        "plane_to_image_homography": [[float(value) for value in row] for row in homography],
        "board": {"squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                  "square_length_m": SQUARE_LENGTH_M, "marker_length_m": MARKER_LENGTH_M},
        "resolution": [int(image.shape[1]), int(image.shape[0])],
        "roi": [float(value) for value in arguments.roi.split(",")] if arguments.roi else None,
        "bin_width_m": arguments.bin_width, "bin_depth_m": arguments.bin_depth,
        "calibrated_at": time.time(),
        "lens_profile": str(Path(arguments.lens).resolve()),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Camera is {distance * 100:.1f} cm from the board plane; pose written to {output}")
    print("Now capture the empty-bin baseline in the dashboard: Calibrate Empty Logitech Scene")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("intrinsics", "plane"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--camera", type=int, default=0)
        sub.add_argument("--images", nargs="*", default=None)
        sub.add_argument("--width", type=int, default=1280)
        sub.add_argument("--height", type=int, default=720)
        sub.add_argument("--views", type=int, default=12)
        sub.add_argument("--delay", type=float, default=2.0)
        sub.add_argument("--settle-frames", type=int, default=5)
        if name == "intrinsics":
            sub.add_argument("--output", default="results/logitech/calibration/logitech_lens.json")
        else:
            sub.add_argument("--lens", default="results/logitech/calibration/logitech_lens.json")
            sub.add_argument("--output", default="results/logitech/calibration/logitech_plane.json")
            sub.add_argument("--roi", default=None, help="x,y,width,height as fractions of the frame")
            sub.add_argument("--bin-width", type=float, default=None)
            sub.add_argument("--bin-depth", type=float, default=None)
    arguments = parser.parse_args(argv)
    return calibrate_intrinsics(arguments) if arguments.command == "intrinsics" else calibrate_plane(arguments)


if __name__ == "__main__":
    sys.exit(main())
