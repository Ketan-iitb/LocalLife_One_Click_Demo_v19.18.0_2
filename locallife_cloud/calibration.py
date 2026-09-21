"""Checkerboard intrinsic/extrinsic calibration between the RealSense and Logitech cameras.

This is phase 1 of the dual-camera fusion system: it produces the
`DualCameraCalibration` file that `fusion.py` needs to project one camera's
depth into the other's coordinate frame. It does not run automatically --
the mounts on the Raspberry Pi are physically fixed once installed, so this
is a one-time (or "redo after remounting either camera") offline step the
user runs deliberately with `tools/calibrate_dual_camera.py`, not something
the live dashboard pipeline invokes per frame.

Nothing here changes behavior for anyone who does not run the capture tool:
`fusion.py` and the pipeline both treat a missing/unreadable calibration
file as "fusion is unavailable," and fall back to each camera's existing,
independent volume measurement -- unchanged from every prior version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import CameraIntrinsics


@dataclass(slots=True)
class MonoCalibration:
    """One camera's intrinsics plus lens distortion, in OpenCV's convention."""

    intrinsics: CameraIntrinsics
    distortion: tuple[float, ...]
    reprojection_error_px: float
    image_count: int

    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
                [0.0, self.intrinsics.fy, self.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "distortion": list(self.distortion),
            "reprojection_error_px": self.reprojection_error_px,
            "image_count": self.image_count,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "MonoCalibration":
        raw = payload["intrinsics"]
        intrinsics = CameraIntrinsics(
            fx=float(raw["fx"]), fy=float(raw["fy"]),
            ppx=float(raw.get("ppx", 0.0)), ppy=float(raw.get("ppy", 0.0)),
            width=int(raw.get("width", 0)), height=int(raw.get("height", 0)),
        )
        return MonoCalibration(
            intrinsics=intrinsics,
            distortion=tuple(float(value) for value in payload["distortion"]),
            reprojection_error_px=float(payload["reprojection_error_px"]),
            image_count=int(payload["image_count"]),
        )


@dataclass(slots=True)
class DualCameraCalibration:
    """Full RealSense<->Logitech calibration: both intrinsics plus their relative pose.

    `rotation`/`translation` map a 3-D point in the *secondary* camera's own
    coordinate frame into the *primary* camera's frame:
    `p_primary = rotation @ p_secondary + translation`. `primary_camera_id`
    records which physical camera that reference frame belongs to (this
    project always uses the RealSense, since it is the one metric-calibrated
    stereo sensor) so a stale file from a different rig layout is never
    silently misapplied.
    """

    primary_camera_id: str
    primary: MonoCalibration
    secondary: MonoCalibration
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3,  (meters)
    stereo_reprojection_error_px: float
    image_pair_count: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3):
            raise ValueError("Rotation must be a 3x3 matrix")
        if translation.shape != (3,):
            raise ValueError("Translation must be a 3-vector")
        # Reject anything that is not close to a proper rotation matrix
        # (orthonormal, determinant +1) -- a garbled or hand-edited
        # calibration file must never be silently used to warp depth.
        should_be_identity = rotation @ rotation.T
        if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
            raise ValueError("Rotation matrix is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            raise ValueError("Rotation matrix is not a proper rotation (determinant != 1)")
        self.rotation = rotation
        self.translation = translation

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "locallife-dual-camera-calibration-v1",
            "primary_camera_id": self.primary_camera_id,
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "stereo_reprojection_error_px": self.stereo_reprojection_error_px,
            "image_pair_count": self.image_pair_count,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "DualCameraCalibration":
        if payload.get("format") != "locallife-dual-camera-calibration-v1":
            raise ValueError("Unrecognized dual-camera calibration file format")
        return DualCameraCalibration(
            primary_camera_id=str(payload["primary_camera_id"]),
            primary=MonoCalibration.from_dict(payload["primary"]),
            secondary=MonoCalibration.from_dict(payload["secondary"]),
            rotation=np.array(payload["rotation"], dtype=np.float64),
            translation=np.array(payload["translation"], dtype=np.float64),
            stereo_reprojection_error_px=float(payload["stereo_reprojection_error_px"]),
            image_pair_count=int(payload["image_pair_count"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "DualCameraCalibration | None":
        """Never raises: a missing, corrupt, or stale-format file just means

        "fusion is unavailable," which every caller must be able to treat as
        a normal, expected state -- not a crash.
        """
        candidate = Path(path)
        if not candidate.is_file():
            return None
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return DualCameraCalibration.from_dict(payload)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None


def default_calibration_path(results_dir: str | Path) -> Path:
    """Where `tools/calibrate_dual_camera.py` writes by default, and where phase 2's

    live fusion wiring will look by default once it exists -- kept as one
    shared definition so the two never drift apart.
    """
    return Path(results_dir) / "dual_camera_calibration.json"


def find_checkerboard_corners(
    image_bgr: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    """Locate a checkerboard's internal-corner grid, refined to sub-pixel accuracy.

    `pattern_size` is (columns, rows) of *internal* corners -- for the
    common 9x6-square board, that is (8, 5). Returns an (N, 1, 2) float32
    array of pixel coordinates in OpenCV's `findChessboardCorners` order,
    or None if the board was not found in this image.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined


def checkerboard_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """The checkerboard's own flat-grid 3-D points, in the board's own coordinate frame (Z=0)."""
    columns, rows = pattern_size
    points = np.zeros((rows * columns, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    return points


def calibrate_mono_camera(
    image_points_per_frame: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_m: float,
    image_shape: tuple[int, int],
) -> MonoCalibration | None:
    """Fit one camera's intrinsics from several checkerboard views.

    `image_shape` is (height, width) in pixels. Returns None if too few
    usable views were supplied (OpenCV's `calibrateCamera` itself needs at
    least 3-4 independent poses of the board to be well-conditioned).
    """
    import cv2

    if len(image_points_per_frame) < 4:
        return None
    object_points = checkerboard_object_points(pattern_size, square_size_m)
    object_points_per_frame = [object_points.astype(np.float32) for _ in image_points_per_frame]
    height, width = image_shape
    reprojection_error, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points_per_frame,
        [points.astype(np.float32) for points in image_points_per_frame],
        (width, height),
        None,
        None,
    )
    intrinsics = CameraIntrinsics(
        fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]),
        ppx=float(camera_matrix[0, 2]), ppy=float(camera_matrix[1, 2]),
        width=width, height=height,
    )
    return MonoCalibration(
        intrinsics=intrinsics,
        distortion=tuple(float(value) for value in np.asarray(distortion).reshape(-1)),
        reprojection_error_px=float(reprojection_error),
        image_count=len(image_points_per_frame),
    )


def calibrate_dual_camera(
    *,
    primary_camera_id: str,
    primary_image_points: list[np.ndarray],
    secondary_image_points: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_m: float,
    primary_image_shape: tuple[int, int],
    secondary_image_shape: tuple[int, int],
) -> DualCameraCalibration | None:
    """Fit both cameras' intrinsics and their relative pose from synchronized checkerboard pairs.

    Every element of `primary_image_points`/`secondary_image_points` must be
    the *same physical checkerboard pose*, seen by each camera at the same
    moment -- that correspondence is what lets `cv2.stereoCalibrate` recover
    the fixed rigid transform between the two cameras (they are both bolted
    to the same Raspberry Pi mount, so this transform does not change once
    solved, unless a camera is physically remounted). Returns None if either
    camera's own intrinsic fit failed, or too few synchronized pairs remain.
    """
    import cv2

    if len(primary_image_points) != len(secondary_image_points):
        raise ValueError("Primary and secondary image-point lists must be the same length (synchronized pairs)")

    primary_mono = calibrate_mono_camera(primary_image_points, pattern_size, square_size_m, primary_image_shape)
    secondary_mono = calibrate_mono_camera(secondary_image_points, pattern_size, square_size_m, secondary_image_shape)
    if primary_mono is None or secondary_mono is None or len(primary_image_points) < 4:
        return None

    object_points = checkerboard_object_points(pattern_size, square_size_m).astype(np.float32)
    object_points_per_frame = [object_points for _ in primary_image_points]
    primary_shape = (primary_image_shape[1], primary_image_shape[0])  # (width, height)

    stereo_error, _cm1, _d1, _cm2, _d2, rotation, translation, _essential, _fundamental = cv2.stereoCalibrate(
        object_points_per_frame,
        [points.astype(np.float32) for points in primary_image_points],
        [points.astype(np.float32) for points in secondary_image_points],
        primary_mono.camera_matrix(),
        np.array(primary_mono.distortion, dtype=np.float64),
        secondary_mono.camera_matrix(),
        np.array(secondary_mono.distortion, dtype=np.float64),
        primary_shape,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    # cv2.stereoCalibrate's (R, T) map a point in camera-1 (primary) coords
    # into camera-2 (secondary) coords: p2 = R @ p1 + T. `fusion.py` wants
    # the inverse -- secondary-to-primary -- to bring the Logitech's depth
    # into the RealSense's (metric, primary) frame, so invert here once.
    rotation_secondary_to_primary = rotation.T
    translation_secondary_to_primary = -rotation.T @ translation.reshape(3)

    return DualCameraCalibration(
        primary_camera_id=primary_camera_id,
        primary=primary_mono,
        secondary=secondary_mono,
        rotation=rotation_secondary_to_primary,
        translation=translation_secondary_to_primary,
        stereo_reprojection_error_px=float(stereo_error),
        image_pair_count=len(primary_image_points),
    )
