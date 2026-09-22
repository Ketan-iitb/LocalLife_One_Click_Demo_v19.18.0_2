"""V26: shape-aware geometry, Logitech metric calibration, paired comparison CSV."""

from __future__ import annotations

import csv
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from locallife_cloud import paired_events
from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.logitech_calibration import (
    METHOD_INVERSE_MULTI_DISTANCE,
    METHOD_MULTI_DISTANCE,
    METHOD_SINGLE_DISTANCE,
    RELATIVE_INVERSE_OUTPUT,
    RELATIVE_ONLY_MESSAGE,
    CalibrationSample,
    LogitechCalibrationStore,
    fit_calibration,
)
from locallife_cloud.paired_events import PairedComparisonLog
from locallife_cloud.server import create_app
from locallife_cloud.shape_geometry import (
    CUBOID,
    CYLINDER,
    FLEXIBLE_OR_UNKNOWN,
    IRREGULAR_RIGID,
    UNCERTAIN,
    GeometryLock,
    measure_shape,
)
from locallife_cloud.types import CameraIntrinsics, DepthCalibration, Detection
from locallife_cloud.volume import fit_reference_plane, object_plane_points

RNG = np.random.default_rng(7)


def _grid(length: float, width: float, step: float = 0.002) -> np.ndarray:
    x, y = np.meshgrid(np.arange(-length / 2, length / 2, step), np.arange(-width / 2, width / 2, step))
    return np.column_stack((x.ravel(), y.ravel()))


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    return points @ np.array([[cos, -sin], [sin, cos]]).T


def _noise(count: int) -> np.ndarray:
    return RNG.normal(0.0, 0.0015, count)


def _box(length=0.20, width=0.12, height=0.10, angle=0.4):
    points = _rotate(_grid(length, width), angle)
    return points, height + _noise(len(points))


def _upright_can(diameter=0.066, height=0.12):
    points = _grid(diameter, diameter)
    points = points[np.hypot(points[:, 0], points[:, 1]) <= diameter / 2]
    return points, height + _noise(len(points))


class ShapeRouterTests(unittest.TestCase):
    def test_cuboid_selects_cuboid_method_with_oriented_dimensions(self) -> None:
        result = measure_shape(*_box(), mesh_volume_l=2.3)
        self.assertEqual(result.geometry_method, CUBOID)
        self.assertAlmostEqual(result.length_mm, 200, delta=6)
        self.assertAlmostEqual(result.width_mm, 120, delta=6)
        self.assertAlmostEqual(result.selected_volume_litres, result.bounding_box_volume_litres)
        # Mesh volume is kept as a separate diagnostic, never relabelled as L*W*H.
        self.assertEqual(result.mesh_volume_litres, 2.3)
        self.assertEqual(result.volume_meaning, "cuboid_volume_l_w_h")

    def test_upright_can_selects_cylinder_with_equal_diameters(self) -> None:
        result = measure_shape(*_upright_can(), mesh_volume_l=0.40)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertAlmostEqual(result.length_mm, result.width_mm)
        self.assertAlmostEqual(result.cylinder_diameter_mm, 66, delta=4)
        self.assertLess(result.cylinder_fit_residual, 0.07)

    def test_cylinder_volume_is_pi_r_squared_h_not_cuboid(self) -> None:
        result = measure_shape(*_upright_can(), mesh_volume_l=0.40)
        radius_m = result.cylinder_diameter_mm / 2000.0
        expected = math.pi * radius_m ** 2 * (result.height_mm / 1000.0) * 1000.0
        self.assertAlmostEqual(result.selected_volume_litres, expected, places=6)
        self.assertAlmostEqual(result.selected_volume_litres / result.bounding_box_volume_litres, math.pi / 4, places=3)

    def test_side_view_shell_reconstructs_cylinder_diameter(self) -> None:
        angles = RNG.uniform(0, math.pi, 5000)
        points = np.column_stack((0.033 * np.cos(angles), 0.033 * np.sin(angles))) + RNG.normal(0, 0.0008, (5000, 2))
        result = measure_shape(points, RNG.uniform(0.02, 0.12, 5000), mesh_volume_l=0.2)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertAlmostEqual(result.cylinder_diameter_mm, 66, delta=4)

    def test_lying_cylinder_uses_its_length_as_axis(self) -> None:
        points = _grid(0.20, 0.07)
        heights = 0.035 + np.sqrt(np.clip(0.035 ** 2 - points[:, 1] ** 2, 0, None)) + _noise(len(points))
        result = measure_shape(points, heights, mesh_volume_l=0.7)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertEqual(result.cylinder_orientation, "lying")
        self.assertAlmostEqual(result.selected_volume_litres, math.pi * 0.035 ** 2 * 0.2 * 1000, delta=0.05)

    def test_irregular_objects_use_height_map_volume(self) -> None:
        points = _grid(0.35, 0.28, 0.003)
        radial = (points[:, 0] / 0.175) ** 2 + (points[:, 1] / 0.14) ** 2
        points, radial = points[radial < 1], radial[radial < 1]
        heights = 0.15 * np.sqrt(1 - radial) + _noise(len(points))
        bag = measure_shape(points, heights, mesh_volume_l=7.5, label="plastic bag")
        self.assertEqual(bag.geometry_method, FLEXIBLE_OR_UNKNOWN)
        self.assertEqual(bag.selected_volume_litres, 7.5)
        self.assertEqual(bag.volume_meaning, "current_external_occupied_volume")
        self.assertGreater(bag.bounding_box_volume_litres, bag.selected_volume_litres)
        toy_points = np.vstack([_grid(0.1, 0.1, 0.003), _grid(0.05, 0.08, 0.003) + [0.09, 0.03]])
        toy_heights = np.where(toy_points[:, 0] > 0.06, 0.05, 0.12) + _noise(len(toy_points))
        toy = measure_shape(toy_points, toy_heights, mesh_volume_l=1.2)
        self.assertEqual(toy.geometry_method, IRREGULAR_RIGID)
        self.assertEqual(toy.selected_volume_litres, 1.2)

    def test_label_alone_never_selects_a_formula(self) -> None:
        # A box-shaped point cloud labelled "can" is still a cuboid.
        self.assertEqual(measure_shape(*_box(), label="soda can").geometry_method, CUBOID)

    def test_low_evidence_returns_uncertain_without_litres(self) -> None:
        result = measure_shape(_grid(0.01, 0.01), np.full(len(_grid(0.01, 0.01)), 0.05))
        self.assertEqual(result.geometry_method, UNCERTAIN)
        self.assertIsNone(result.selected_volume_litres)
        # An irregular shape without a height-map volume has nothing honest to report.
        toy_points = np.vstack([_grid(0.1, 0.1, 0.003), _grid(0.05, 0.08, 0.003) + [0.09, 0.03]])
        toy_heights = np.where(toy_points[:, 0] > 0.06, 0.05, 0.12)
        no_mesh = measure_shape(toy_points, toy_heights, mesh_volume_l=None)
        self.assertEqual(no_mesh.geometry_method, UNCERTAIN)
        self.assertIsNone(no_mesh.selected_volume_litres)
        self.assertIsNotNone(no_mesh.bounding_box_volume_litres)

    def test_geometry_method_is_frozen_after_acceptance(self) -> None:
        lock = GeometryLock(required_frames=3)
        box = measure_shape(*_box())
        can = measure_shape(*_upright_can(), mesh_volume_l=0.4)
        self.assertFalse(lock.update(1, box).frozen)
        lock.update(1, can)
        lock.update(1, box)
        accepted = lock.update(1, box)
        self.assertTrue(accepted.frozen)
        self.assertEqual(accepted.geometry_method, CUBOID)
        self.assertEqual(accepted.frames, 3)
        for _ in range(5):
            self.assertIs(lock.update(1, can), accepted)
        lock.forget(1)
        self.assertIsNone(lock.frozen(1))


class ObjectPlanePointTests(unittest.TestCase):
    def test_mask_points_exclude_background_and_erosion_trims_edges(self) -> None:
        depth = np.full((60, 60), 2.0, dtype=np.float32)
        depth[20:40, 20:40] = 1.8
        camera = CameraIntrinsics(fx=100, fy=100, ppx=30, ppy=30, width=60, height=60)
        plane = fit_reference_plane(np.full((60, 60), 2.0, dtype=np.float32), camera)
        mask = np.zeros((60, 60), dtype=bool)
        mask[15:45, 15:45] = True  # generous mask, includes floor around the object
        footprint, heights = object_plane_points(depth, camera, mask, plane, min_height_m=0.02)
        self.assertEqual(len(heights), 400)
        self.assertTrue(np.all(heights > 0.15))
        eroded = object_plane_points(depth, camera, mask & (depth < 1.9), plane, min_height_m=0.02, erode_px=2)
        self.assertEqual(len(eroded[1]), 16 * 16)


class LogitechCalibrationTests(unittest.TestCase):
    def test_single_distance_fits_scale_only_for_metric_model(self) -> None:
        calibration, reason = fit_calibration([CalibrationSample(1.6, 2.0, 5000)], "metric")
        self.assertIsNone(reason)
        self.assertEqual(calibration.method, METHOD_SINGLE_DISTANCE)
        self.assertAlmostEqual(calibration.scale, 1.25)
        self.assertEqual(calibration.offset_m, 0.0)

    def test_two_distances_fit_scale_and_shift(self) -> None:
        samples = [CalibrationSample(0.8 * z + 0.1, z, 5000) for z in (1.2, 1.6, 2.0)]
        calibration, _ = fit_calibration(samples, "metric")
        self.assertEqual(calibration.method, METHOD_MULTI_DISTANCE)
        self.assertAlmostEqual(float(calibration.apply(np.array([0.8 * 1.4 + 0.1]))[0]), 1.4, places=5)

    def test_relative_inverse_depth_model_needs_two_distances(self) -> None:
        calibration, reason = fit_calibration([CalibrationSample(5.0, 2.0, 5000)], RELATIVE_INVERSE_OUTPUT)
        self.assertIsNone(calibration)
        self.assertEqual(reason, "relative_model_needs_two_reference_distances")
        samples = [CalibrationSample(3.0 / z + 0.2, z, 5000) for z in (1.0, 2.0)]
        calibration, _ = fit_calibration(samples, RELATIVE_INVERSE_OUTPUT)
        self.assertEqual(calibration.method, METHOD_INVERSE_MULTI_DISTANCE)
        self.assertAlmostEqual(float(calibration.apply(np.array([3.0 / 1.5 + 0.2]))[0]), 1.5, places=5)

    def test_store_persists_parameters_and_reports_relative_only_until_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logitech_depth.json"
            store = LogitechCalibrationStore(path)
            self.assertEqual(store.status()["message"], RELATIVE_ONLY_MESSAGE)
            predicted = np.full((40, 40), 1.6, dtype=np.float32)
            status = store.add_sample(predicted, None, 2.0, "metric")
            self.assertTrue(status["valid"])
            saved = LogitechCalibrationStore(path).calibration
            self.assertEqual(saved.to_dict()["calibration_id"], status["calibration"]["calibration_id"])
            for key in ("scale", "offset_m", "method", "calibrated_at", "reference_distance_m"):
                self.assertIn(key, saved.to_dict())
            self.assertEqual(saved.resolution, (40, 40))

    def test_calibration_round_trips_through_dict(self) -> None:
        original = DepthCalibration(2.0, 0.1, 0.01, 99, method="m", calibration_id="abc", inverse=True)
        self.assertEqual(DepthCalibration.from_dict(original.to_dict()).to_dict(), original.to_dict())


class LogitechLensTests(unittest.TestCase):
    def _profile(self, directory: str, distortion=(0.0, 0.0, 0.0, 0.0, 0.0)) -> Path:
        from locallife_cloud.calibration import MonoCalibration

        path = Path(directory) / "logitech_lens.json"
        profile = MonoCalibration(
            intrinsics=CameraIntrinsics(fx=600, fy=600, ppx=320, ppy=240, width=640, height=480),
            distortion=distortion, reprojection_error_px=0.3, image_count=20,
        )
        path.write_text(__import__("json").dumps(profile.to_dict()), encoding="utf-8")
        return path

    def test_profile_is_scaled_for_same_aspect_and_refused_for_other_crops(self) -> None:
        from locallife_cloud.logitech_calibration import LogitechLens

        with tempfile.TemporaryDirectory() as directory:
            lens = LogitechLens(self._profile(directory))
            frame = RNG.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            output, intrinsics = lens.prepare(frame, None)
            self.assertEqual(lens.status, "undistorted")
            self.assertAlmostEqual(intrinsics.fx, 300)
            self.assertEqual((intrinsics.width, intrinsics.height), (320, 240))
            # Zero distortion: the image is unchanged apart from border interpolation.
            self.assertLess(np.abs(output[5:-5, 5:-5].astype(int) - frame[5:-5, 5:-5]).mean(), 1.0)
            square = np.zeros((300, 300, 3), dtype=np.uint8)
            self.assertIs(lens.prepare(square, "given")[1], "given")
            self.assertEqual(lens.status, "lens_profile_resolution_mismatch")

    def test_barrel_distortion_is_removed_before_geometry(self) -> None:
        from locallife_cloud.logitech_calibration import LogitechLens

        with tempfile.TemporaryDirectory() as directory:
            lens = LogitechLens(self._profile(directory, distortion=(-0.3, 0.1, 0.0, 0.0, 0.0)))
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, ::40] = 255
            output, _ = lens.prepare(frame, None)
            self.assertFalse(np.array_equal(output, frame))


def _measurement(camera: str, event_id: str, **extra) -> dict:
    row = {
        "event_id": event_id, "camera_source": camera, "status": "accepted", "label": "box",
        "timestamp": 1.0, "volume_l": 1.0, "length_mm": 100, "width_mm": 100, "height_mm": 100,
        "shape_geometry": {"geometry_method": "cuboid", "selected_volume_litres": 1.0,
                           "length_mm": 100.0, "width_mm": 100.0, "height_mm": 100.0},
    }
    row.update(extra)
    return row


class PairedComparisonLogTests(unittest.TestCase):
    def test_one_object_one_event_with_separate_measurement_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory))
            first = log.record_measurement(_measurement("realsense", "rs-1"), now=100)
            second = log.record_measurement(_measurement("logitech", "lg-1", volume_l=1.2), now=103)
            self.assertEqual(first["comparison_event_id"], second["comparison_event_id"])
            rows = log.rows()
            self.assertEqual([row["measurement_id"] for row in rows], ["rs-1", "lg-1"])
            self.assertEqual({row["camera_source"] for row in rows}, {"realsense", "logitech"})

    def test_repeats_restarts_and_retries_never_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory))
            log.record_measurement(_measurement("realsense", "rs-1"), now=100)
            self.assertTrue(log.record_measurement(_measurement("realsense", "rs-1"), now=101)["duplicate"])
            restarted = PairedComparisonLog(Path(directory))
            self.assertTrue(restarted.record_measurement(_measurement("realsense", "rs-1"), now=102)["duplicate"])
            self.assertEqual(len(restarted.rows()), 1)

    def test_missing_camera_is_recorded_once_and_never_substituted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory), window_seconds=10)
            log.record_measurement(_measurement("realsense", "rs-1"), now=100)
            self.assertEqual(log.flush_expired(now=120), 1)
            self.assertEqual(log.flush_expired(now=130), 0)
            missing = [row for row in log.rows() if row["camera_source"] == "logitech"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["status"], "missing")
            self.assertEqual(missing[0]["selected_volume_litres"], "")
            # A late Logitech result starts a new event instead of filling the closed one.
            late = log.record_measurement(_measurement("logitech", "lg-late"), now=131)
            self.assertNotEqual(late["comparison_event_id"], log.rows()[0]["comparison_event_id"])

    def test_single_camera_research_mode_reports_no_missing_partner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory), window_seconds=1, expected_cameras=("realsense",))
            log.record_measurement(_measurement("realsense", "rs-1"), now=100)
            self.assertEqual(log.flush_expired(now=200), 0)
            self.assertFalse(log.record_measurement(_measurement("logitech", "lg-1"), now=201)["written"])

    def test_ground_truth_is_shared_by_the_event_and_errors_are_per_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory))
            event = log.record_measurement(_measurement("realsense", "rs-1"), now=100)["comparison_event_id"]
            logitech = _measurement("logitech", "lg-1")
            logitech["shape_geometry"] = dict(logitech["shape_geometry"], selected_volume_litres=1.5)
            log.record_measurement(logitech, now=101)
            log.set_ground_truth(event, {"reference_volume_litres": 1.2, "ground_truth_method": "cuboid_dimensions"})
            errors = {row["camera_source"]: float(row["absolute_error_litres"]) for row in log.rows()}
            self.assertAlmostEqual(errors["realsense"], 0.2, places=6)
            self.assertAlmostEqual(errors["logitech"], 0.3, places=6)
            with self.assertRaises(ValueError):
                log.set_ground_truth(event, {"reference_volume_litres": 1, "ground_truth_method": "guess"})

    def test_write_failure_is_queued_and_retry_writes_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = PairedComparisonLog(Path(directory))
            with mock.patch.object(paired_events.os, "fsync", side_effect=OSError("disk full")):
                result = log.record_measurement(_measurement("realsense", "rs-1"), now=100)
            self.assertFalse(result["written"])
            self.assertEqual(log.status()["persistence_status"], "retry_pending")
            # The failed append may have left a partial line; the retry rewrites nothing twice.
            log.path.write_text("", encoding="utf-8")
            self.assertEqual(log.retry_failed()["recovered"], 1)
            self.assertEqual(log.retry_failed()["recovered"], 0)
            self.assertEqual([row["measurement_id"] for row in log.rows()], ["rs-1"])


class SharedDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class MetricDepth:
    """Stand-in for Depth Anything V2 metric output: brighter pixels are closer."""

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [2.0 - frame.max(axis=2).astype(np.float32) * 0.0015 for frame in frames]


class PairedEndToEndTests(unittest.TestCase):
    def _station(self, directory: str, **overrides) -> tuple[DualCameraCoordinator, SharedDetector]:
        detector = SharedDetector()
        settings = dict(
            results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
            tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
            auto_deposit=True, logitech_reference_distance_m=2.0,
        )
        settings.update(overrides)
        return DualCameraCoordinator(AppConfig(**settings), detector=detector, depth_estimator=MetricDepth()), detector

    def _scene(self, manager: DualCameraCoordinator):
        empty = np.zeros((40, 40, 3), dtype=np.uint8)
        floor = np.full((40, 40), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40)
        manager.camera("realsense").process_frame(empty, depth_m=floor, intrinsics=camera, persist=False)
        manager.camera("logitech").process_frame(empty, intrinsics=camera, persist=False)
        manager.camera("realsense").set_baseline()
        manager.camera("logitech").set_baseline()
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:25, 10:25] = True
        frame = empty.copy()
        frame[mask] = (200, 0, 0)
        depth = floor.copy()
        depth[mask] = 1.7
        return frame, depth, camera, Detection("blue garbage bag", 0.9, (10, 10, 25, 25), mask, color="blue")

    def test_controlled_object_reaches_download_as_one_row_per_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory)
            frame, depth, camera, bag = self._scene(manager)
            detector.items = [bag]
            for timestamp in (100.0, 101.0, 102.0, 103.0):
                manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=timestamp)
                manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=timestamp + 0.25)
            client = create_app(manager.config, pipeline=manager).test_client()
            response = client.get("/api/comparison/measurements.csv")
            self.assertEqual(response.status_code, 200)
            rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True).lstrip("﻿"))))
            self.assertEqual(sorted(row["camera_source"] for row in rows), ["logitech", "realsense"])
            self.assertEqual(len({row["comparison_event_id"] for row in rows}), 1)
            self.assertEqual(len({row["measurement_id"] for row in rows}), 2)
            for row in rows:
                for column in ("camera_source", "length_mm", "width_mm", "height_mm",
                               "selected_volume_litres", "geometry_method", "status"):
                    self.assertNotEqual(row[column], "", f"{row['camera_source']} {column} is empty")
                self.assertGreater(float(row["selected_volume_litres"]), 0)
            realsense = next(row for row in rows if row["camera_source"] == "realsense")
            self.assertEqual(realsense["geometry_method"], CUBOID)
            # Dashboard state and a repeated download agree and add nothing.
            events = client.get("/api/comparison/events").get_json()
            self.assertEqual(events["status"]["realsense_rows"], 1)
            self.assertEqual(events["status"]["logitech_rows"], 1)
            again = client.get("/api/comparison/measurements.csv").get_data(as_text=True)
            self.assertEqual(again, response.get_data(as_text=True))

    def test_acceptance_script_reads_the_live_endpoints(self) -> None:
        import importlib.util
        import contextlib

        spec = importlib.util.spec_from_file_location(
            "check_v26_acceptance", Path(__file__).resolve().parents[1] / "scripts" / "check_v26_acceptance.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory)
            frame, depth, camera, bag = self._scene(manager)
            detector.items = [bag]
            for timestamp in (100.0, 101.0, 102.0, 103.0):
                manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=timestamp)
                manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=timestamp + 0.25)
            client = create_app(manager.config, pipeline=manager).test_client()

            def fake_get(url, headers=None, timeout=None):
                response = client.get(url.replace("http://127.0.0.1:8000", ""), headers=headers)
                return mock.Mock(ok=response.status_code < 400, status_code=response.status_code,
                                 json=response.get_json, text=response.get_data(as_text=True))

            output = io.StringIO()
            with mock.patch.object(script.requests, "get", fake_get), \
                    mock.patch.object(sys, "argv", ["check"]), contextlib.redirect_stdout(output):
                script.main()
            report = output.getvalue()
            self.assertIn("[PASS] latest comparison event has both cameras", report)
            self.assertIn("[PASS] downloaded CSV contains that event's rows", report)
            self.assertIn("[PASS] realsense row complete", report)
            self.assertIn("[PASS] logitech row complete", report)
            # No live stream and no lens profile in a unit test: reported, not faked.
            self.assertIn("[FAIL] Logitech lens profile", report)

    def test_listener_failure_does_not_stop_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory)
            frame, depth, camera, bag = self._scene(manager)
            detector.items = [bag]
            with mock.patch.object(manager.paired_log, "record_measurement", side_effect=RuntimeError("boom")):
                for timestamp in (100.0, 101.0, 102.0):
                    result = manager.camera("realsense").process_frame(
                        frame, depth_m=depth, intrinsics=camera, timestamp=timestamp)
            self.assertEqual(manager.camera("realsense").ledger.summary()["deposited_bags"], 1)
            self.assertEqual(result.detections[0].track_id, 1)

    def test_uncalibrated_logitech_reports_relative_depth_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self._station(directory, logitech_reference_distance_m=0.0)
            self._scene(manager)
            status = manager.camera("logitech").logitech_calibration_status()
            self.assertFalse(status["metric_ready"])
            self.assertEqual(status["message"], RELATIVE_ONLY_MESSAGE)

    def test_logitech_calibration_sample_endpoint_stores_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self._station(directory, logitech_reference_distance_m=0.0)
            self._scene(manager)
            client = create_app(manager.config, pipeline=manager).test_client()
            response = client.post("/api/logitech/calibration/sample", json={"known_distance_m": 2.1})
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            body = response.get_json()
            self.assertTrue(body["valid"])
            self.assertAlmostEqual(body["calibration"]["scale"], 2.1 / 2.0, places=4)
            manager.camera("logitech").set_baseline()
            status = manager.camera("logitech").logitech_calibration_status()
            self.assertTrue(status["metric_ready"])
            self.assertEqual(status["active_calibration"]["calibration_id"], body["calibration"]["calibration_id"])


if __name__ == "__main__":
    unittest.main()
