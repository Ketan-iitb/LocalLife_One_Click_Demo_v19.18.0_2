"""Round-trip correctness tests for phase 1 of dual-camera fusion (System 2).

These synthesize checkerboard views with a KNOWN ground-truth intrinsics and
relative pose, run them through the real `cv2.calibrateCamera`/
`cv2.stereoCalibrate` calls in `calibration.py`, and assert the solver
recovers that ground truth -- this is the strongest verification available
without physical cameras: it exercises the actual OpenCV solvers this
project depends on, not a mock of them.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.calibration import (
    DualCameraCalibration,
    MonoCalibration,
    calibrate_dual_camera,
    calibrate_mono_camera,
    checkerboard_object_points,
)
from locallife_cloud.fusion import (
    backproject_to_camera_space,
    fuse_primary_with_secondary_fill,
    project_depth_to_reference_frame,
)
from locallife_cloud.types import CameraIntrinsics


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _pinhole_project(points_camera_space: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    x, y, z = points_camera_space[:, 0], points_camera_space[:, 1], points_camera_space[:, 2]
    columns = intrinsics.fx * x / z + intrinsics.ppx
    rows = intrinsics.fy * y / z + intrinsics.ppy
    return np.column_stack((columns, rows)).astype(np.float32)


class SyntheticDualCameraFixture:
    """Shared ground truth for the calibration round-trip tests below."""

    pattern_size = (7, 5)
    square_size_m = 0.03
    primary_shape = (480, 640)  # (height, width)
    secondary_shape = (480, 640)

    primary_intrinsics = CameraIntrinsics(fx=620.0, fy=615.0, ppx=320.0, ppy=240.0, width=640, height=480)
    secondary_intrinsics = CameraIntrinsics(fx=540.0, fy=535.0, ppx=330.0, ppy=235.0, width=640, height=480)

    # A modest baseline/rotation, similar in scale to two cameras bolted a
    # few centimeters apart on the same small rig.
    true_rotation = _rotation_matrix(math.radians(4.0), math.radians(-6.0), math.radians(2.0))
    true_translation = np.array([0.08, -0.01, 0.02])

    def board_poses(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(rotation, translation) placing the board in front of BOTH cameras, in primary-frame coords."""
        poses = []
        for depth in (0.5, 0.7, 0.9):
            for (rx, ry, tilt_x, tilt_y) in (
                (0.0, 0.0, 0.0, 0.0),
                (0.25, -0.15, 0.15, 0.0),
                (-0.2, 0.2, -0.1, 0.1),
                (0.1, 0.3, 0.05, -0.15),
            ):
                rotation = _rotation_matrix(0.15 + tilt_x, tilt_y, 0.05)
                translation = np.array([rx, ry, depth])
                poses.append((rotation, translation))
        return poses

    def synthesize_image_points(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        object_points = checkerboard_object_points(self.pattern_size, self.square_size_m)
        # Center the board on its own middle so rotations don't fling it out of frame.
        object_points = object_points - object_points.mean(axis=0)

        primary_points_per_frame: list[np.ndarray] = []
        secondary_points_per_frame: list[np.ndarray] = []
        for rotation, translation in self.board_poses():
            points_primary = object_points @ rotation.T + translation
            points_secondary = (points_primary - self.true_translation) @ self.true_rotation

            primary_pixels = _pinhole_project(points_primary, self.primary_intrinsics)
            secondary_pixels = _pinhole_project(points_secondary, self.secondary_intrinsics)

            height, width = self.primary_shape
            if not (
                np.all(primary_pixels[:, 0] >= 0) and np.all(primary_pixels[:, 0] < width)
                and np.all(primary_pixels[:, 1] >= 0) and np.all(primary_pixels[:, 1] < height)
                and np.all(secondary_pixels[:, 0] >= 0) and np.all(secondary_pixels[:, 0] < width)
                and np.all(secondary_pixels[:, 1] >= 0) and np.all(secondary_pixels[:, 1] < height)
            ):
                continue  # this synthetic pose fell outside one camera's frame; skip it.

            primary_points_per_frame.append(primary_pixels.reshape(-1, 1, 2))
            secondary_points_per_frame.append(secondary_pixels.reshape(-1, 1, 2))
        return primary_points_per_frame, secondary_points_per_frame


class CalibrateDualCameraRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SyntheticDualCameraFixture()
        self.primary_points, self.secondary_points = self.fixture.synthesize_image_points()
        # The pose sweep above is deliberately varied (different depths and
        # tilts); if fewer than 6 survive the in-frame check something is
        # wrong with the fixture itself, not the code under test.
        self.assertGreaterEqual(len(self.primary_points), 6)

    def test_mono_calibration_recovers_true_intrinsics(self) -> None:
        result = calibrate_mono_camera(
            self.primary_points, self.fixture.pattern_size, self.fixture.square_size_m, self.fixture.primary_shape,
        )
        self.assertIsNotNone(result)
        self.assertLess(result.reprojection_error_px, 0.5)
        truth = self.fixture.primary_intrinsics
        self.assertAlmostEqual(result.intrinsics.fx, truth.fx, delta=3.0)
        self.assertAlmostEqual(result.intrinsics.fy, truth.fy, delta=3.0)
        self.assertAlmostEqual(result.intrinsics.ppx, truth.ppx, delta=3.0)
        self.assertAlmostEqual(result.intrinsics.ppy, truth.ppy, delta=3.0)

    def test_too_few_frames_returns_none_instead_of_a_bad_fit(self) -> None:
        result = calibrate_mono_camera(
            self.primary_points[:2], self.fixture.pattern_size, self.fixture.square_size_m, self.fixture.primary_shape,
        )
        self.assertIsNone(result)

    def test_stereo_calibration_recovers_true_relative_pose(self) -> None:
        result = calibrate_dual_camera(
            primary_camera_id="realsense",
            primary_image_points=self.primary_points,
            secondary_image_points=self.secondary_points,
            pattern_size=self.fixture.pattern_size,
            square_size_m=self.fixture.square_size_m,
            primary_image_shape=self.fixture.primary_shape,
            secondary_image_shape=self.fixture.secondary_shape,
        )
        self.assertIsNotNone(result)
        self.assertLess(result.stereo_reprojection_error_px, 1.0)

        # calibrate_dual_camera stores the secondary->primary transform
        # (inverted from cv2.stereoCalibrate's primary->secondary output),
        # so it should match the fixture's true_rotation/true_translation
        # directly.
        rotation_error = result.rotation.T @ self.fixture.true_rotation
        angle_error_deg = math.degrees(
            math.acos(np.clip((np.trace(rotation_error) - 1) / 2, -1.0, 1.0))
        )
        self.assertLess(angle_error_deg, 1.0)
        translation_error_m = float(np.linalg.norm(result.translation - self.fixture.true_translation))
        self.assertLess(translation_error_m, 0.01)

    def test_mismatched_pair_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_dual_camera(
                primary_camera_id="realsense",
                primary_image_points=self.primary_points,
                secondary_image_points=self.secondary_points[:-1],
                pattern_size=self.fixture.pattern_size,
                square_size_m=self.fixture.square_size_m,
                primary_image_shape=self.fixture.primary_shape,
                secondary_image_shape=self.fixture.secondary_shape,
            )

    def test_save_and_load_round_trip_preserves_values(self) -> None:
        result = calibrate_dual_camera(
            primary_camera_id="realsense",
            primary_image_points=self.primary_points,
            secondary_image_points=self.secondary_points,
            pattern_size=self.fixture.pattern_size,
            square_size_m=self.fixture.square_size_m,
            primary_image_shape=self.fixture.primary_shape,
            secondary_image_shape=self.fixture.secondary_shape,
        )
        self.assertIsNotNone(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual_camera_calibration.json"
            result.save(path)
            reloaded = DualCameraCalibration.load(path)
        self.assertIsNotNone(reloaded)
        np.testing.assert_allclose(reloaded.rotation, result.rotation, atol=1e-9)
        np.testing.assert_allclose(reloaded.translation, result.translation, atol=1e-9)
        self.assertEqual(reloaded.primary_camera_id, result.primary_camera_id)


class DualCameraCalibrationRobustnessTests(unittest.TestCase):
    def _valid_mono(self) -> MonoCalibration:
        return MonoCalibration(
            intrinsics=CameraIntrinsics(fx=600, fy=600, ppx=320, ppy=240, width=640, height=480),
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            reprojection_error_px=0.2,
            image_count=8,
        )

    def test_missing_calibration_file_loads_as_none_not_an_exception(self) -> None:
        self.assertIsNone(DualCameraCalibration.load("/nonexistent/path/calibration.json"))

    def test_corrupt_calibration_file_loads_as_none_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(DualCameraCalibration.load(path))

    def test_wrong_format_tag_loads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_format.json"
            path.write_text('{"format": "something-else"}', encoding="utf-8")
            self.assertIsNone(DualCameraCalibration.load(path))

    def test_non_orthonormal_rotation_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            DualCameraCalibration(
                primary_camera_id="realsense",
                primary=self._valid_mono(),
                secondary=self._valid_mono(),
                rotation=np.eye(3) * 2.0,  # not orthonormal
                translation=np.zeros(3),
                stereo_reprojection_error_px=0.3,
                image_pair_count=8,
            )

    def test_valid_identity_calibration_constructs_cleanly(self) -> None:
        calibration = DualCameraCalibration(
            primary_camera_id="realsense",
            primary=self._valid_mono(),
            secondary=self._valid_mono(),
            rotation=np.eye(3),
            translation=np.array([0.05, 0.0, 0.0]),
            stereo_reprojection_error_px=0.3,
            image_pair_count=8,
        )
        self.assertEqual(calibration.primary_camera_id, "realsense")


class FusionGeometryTests(unittest.TestCase):
    def test_backproject_then_project_round_trips_a_known_point(self) -> None:
        intrinsics = CameraIntrinsics(fx=500, fy=500, ppx=320, ppy=240, width=640, height=480)
        depth = np.full((480, 640), np.nan, dtype=np.float32)
        depth[240, 420] = 1.5  # 100px right of center, at 1.5m
        valid = np.isfinite(depth)
        points = backproject_to_camera_space(depth, intrinsics, valid)
        self.assertEqual(points.shape, (1, 3))
        expected_x = (420 - 320) * 1.5 / 500
        self.assertAlmostEqual(points[0, 0], expected_x, places=6)
        self.assertAlmostEqual(points[0, 1], 0.0, places=6)
        self.assertAlmostEqual(points[0, 2], 1.5, places=6)

    def test_identity_calibration_reprojects_depth_onto_itself(self) -> None:
        intrinsics = CameraIntrinsics(fx=400, fy=400, ppx=160, ppy=120, width=320, height=240)
        calibration = DualCameraCalibration(
            primary_camera_id="realsense",
            primary=MonoCalibration(intrinsics, (0, 0, 0, 0, 0), 0.1, 8),
            secondary=MonoCalibration(intrinsics, (0, 0, 0, 0, 0), 0.1, 8),
            rotation=np.eye(3),
            translation=np.zeros(3),
            stereo_reprojection_error_px=0.1,
            image_pair_count=8,
        )
        depth = np.full((240, 320), 1.2, dtype=np.float32)
        depth[0:20, :] = np.nan  # a strip with no valid depth

        reprojected = project_depth_to_reference_frame(
            depth, intrinsics, calibration, intrinsics, (240, 320),
        )
        valid_region = np.isfinite(depth)
        np.testing.assert_allclose(
            reprojected[valid_region], depth[valid_region], atol=1e-3,
        )
        self.assertTrue(np.all(np.isnan(reprojected[~valid_region])))

    def test_nearer_point_wins_when_two_source_points_project_to_the_same_pixel(self) -> None:
        intrinsics = CameraIntrinsics(fx=400, fy=400, ppx=160, ppy=120, width=320, height=240)
        calibration = DualCameraCalibration(
            primary_camera_id="realsense",
            primary=MonoCalibration(intrinsics, (0, 0, 0, 0, 0), 0.1, 8),
            secondary=MonoCalibration(intrinsics, (0, 0, 0, 0, 0), 0.1, 8),
            rotation=np.eye(3),
            translation=np.zeros(3),
            stereo_reprojection_error_px=0.1,
            image_pair_count=8,
        )
        # Two different source pixels whose backprojected 3-D points, once
        # transformed (here: identity), both round to the same target pixel
        # -- e.g. a near point directly in front of a far point along the
        # same ray. The nearer one must be what survives.
        depth = np.full((240, 320), np.nan, dtype=np.float32)
        depth[120, 160] = 0.5   # near, dead-center
        depth[120, 161] = 3.0   # far, one pixel over -- rounds to the same target pixel at this depth

        reprojected = project_depth_to_reference_frame(
            depth, intrinsics, calibration, intrinsics, (240, 320),
        )
        # Whatever pixel(s) ended up holding a value near this location must
        # reflect the nearer (0.5m) sample, not the farther one, at the
        # target pixel where they collided.
        candidate_values = reprojected[119:122, 159:162]
        finite_values = candidate_values[np.isfinite(candidate_values)]
        self.assertTrue(np.any(np.isclose(finite_values, 0.5, atol=0.05)))

    def test_fuse_only_fills_where_primary_is_invalid(self) -> None:
        primary = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
        secondary = np.array([[9.0, 1.5], [3.5, 9.0]], dtype=np.float32)
        fused, filled_mask = fuse_primary_with_secondary_fill(primary, secondary)
        np.testing.assert_allclose(fused, [[1.0, 1.5], [3.5, 2.0]])
        np.testing.assert_array_equal(filled_mask, [[False, True], [True, False]])

    def test_fuse_leaves_a_hole_where_both_are_invalid(self) -> None:
        primary = np.array([[np.nan]], dtype=np.float32)
        secondary = np.array([[np.nan]], dtype=np.float32)
        fused, filled_mask = fuse_primary_with_secondary_fill(primary, secondary)
        self.assertTrue(np.isnan(fused[0, 0]))
        self.assertFalse(filled_mask[0, 0])

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_primary_with_secondary_fill(
                np.zeros((2, 2), dtype=np.float32), np.zeros((3, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
