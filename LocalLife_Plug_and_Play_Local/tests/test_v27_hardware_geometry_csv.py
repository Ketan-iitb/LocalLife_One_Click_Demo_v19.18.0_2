"""V27 regressions: fitted cylinders end to end, Logitech object masks, one CSV row per object.

These use deterministic synthetic depth and masks. They check the code path,
not the physical cameras: they are NOT hardware validation.
"""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.logitech import bound_logitech_detections
from locallife_cloud.logitech_calibration import RELATIVE_ONLY_MESSAGE
from locallife_cloud.server import create_app
from locallife_cloud.shape_geometry import CYLINDER, UNCERTAIN, measure_shape
from locallife_cloud.types import CameraIntrinsics, Detection

RADIUS_M = 0.0415
HEIGHT_M = 0.195


def _bottle_shell(span_deg: float = 170.0, noise: float = 0.0015, seed: int = 11):
    """Front shell of an upright bottle seen obliquely, plus stray elevated pixels."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(math.radians(90 - span_deg / 2), math.radians(90 + span_deg / 2), 6000)
    points = np.column_stack((RADIUS_M * np.cos(angles), RADIUS_M * np.sin(angles))) + rng.normal(0, noise, (6000, 2))
    heights = rng.uniform(0.015, HEIGHT_M, 6000)
    strays = rng.uniform(-0.06, 0.06, (150, 2))
    return np.vstack([points, strays]), np.r_[heights, rng.uniform(0.015, 0.05, 150)]


class CylinderFitTests(unittest.TestCase):
    def test_upright_bottle_shell_reports_fitted_diameter_not_its_visible_arc(self) -> None:
        result = measure_shape(*_bottle_shell(), mesh_volume_l=0.6)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertEqual(result.cylinder_orientation, "upright")
        self.assertAlmostEqual(result.cylinder_diameter_mm, 83, delta=3)
        self.assertEqual(result.length_mm, result.width_mm)  # diameter x diameter x height
        self.assertAlmostEqual(result.height_mm, 195, delta=3)
        self.assertAlmostEqual(result.radius_mm * 2, result.cylinder_diameter_mm)
        expected = math.pi * (result.radius_mm / 1000) ** 2 * (result.height_mm / 1000) * 1000
        self.assertAlmostEqual(result.selected_volume_litres, expected, places=6)
        self.assertGreater(result.fit_confidence, 0.5)
        data = result.to_dict()
        self.assertEqual(data["volume_liters"], data["selected_volume_litres"])
        self.assertEqual(data["diameter_mm"], data["cylinder_diameter_mm"])

    def test_horizontal_bottle_uses_axis_length_and_cross_section_circle(self) -> None:
        x, y = np.meshgrid(np.arange(-0.0975, 0.0975, 0.002), np.arange(-RADIUS_M, RADIUS_M, 0.002))
        points = np.column_stack((x.ravel(), y.ravel()))
        heights = RADIUS_M + np.sqrt(np.clip(RADIUS_M ** 2 - points[:, 1] ** 2, 0, None))
        heights += np.random.default_rng(3).normal(0, 0.0015, len(points))
        result = measure_shape(points, heights, mesh_volume_l=0.9)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertEqual(result.cylinder_orientation, "lying")
        self.assertAlmostEqual(result.cylinder_diameter_mm, 83, delta=5)
        self.assertAlmostEqual(result.cylinder_height_mm, 195, delta=5)
        self.assertAlmostEqual(result.selected_volume_litres, math.pi * RADIUS_M ** 2 * 0.195 * 1000, delta=0.08)

    def test_poor_cylinder_fit_is_pending_never_a_cuboid_volume(self) -> None:
        # Only a 60 degree sliver of the shell is visible: too little arc to trust.
        points, heights = _bottle_shell(span_deg=60.0, noise=0.001)
        result = measure_shape(points, heights, mesh_volume_l=0.6)
        self.assertEqual(result.geometry_method, UNCERTAIN)
        self.assertIsNone(result.selected_volume_litres)
        self.assertIsNone(result.bounding_box_volume_litres)
        self.assertIsNotNone(result.rejection_reason)


class SharedDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames):
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class NoDepth:
    def estimate_batch(self, frames):
        return [np.full(frame.shape[:2], 1.0, dtype=np.float32) for frame in frames]


def _can_scene(size: int = 160, fx: float = 400.0, floor_m: float = 1.0, radius_m: float = 0.04,
               height_m: float = 0.15):
    """Overhead RealSense view of an upright can: a flat disc top above the floor."""
    camera = CameraIntrinsics(fx=fx, fy=fx, ppx=size / 2, ppy=size / 2, width=size, height=size)
    floor = np.full((size, size), floor_m, dtype=np.float32)
    rows, columns = np.mgrid[0:size, 0:size]
    top = floor_m - height_m
    x = (columns - size / 2) * top / fx
    y = (rows - size / 2) * top / fx
    disc = np.hypot(x, y) <= radius_m
    depth = floor.copy()
    depth[disc] = top
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[disc] = (30, 30, 200)
    ys, xs = np.nonzero(disc)
    detection = Detection("soda can", 0.9, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                          disc, color="red")
    return camera, floor, depth, frame, detection


class RealSenseCylinderEndToEndTests(unittest.TestCase):
    def _station(self, directory: str, **overrides):
        detector = SharedDetector()
        settings = dict(
            results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
            tracker_confirm_frames=1, settle_frames=3, volume_window_frames=2,
            operating_mode="geometry_validation", auto_deposit=False, research_mode="realsense_only",
            enable_monocular_depth=False,
        )
        settings.update(overrides)
        manager = DualCameraCoordinator(AppConfig(**settings), detector=detector, depth_estimator=NoDepth())
        return manager, detector

    def _run(self, manager, detector, frames: int = 8):
        camera, floor, depth, frame, can = _can_scene()
        station = manager.camera("realsense")
        station.process_frame(np.zeros_like(frame), depth_m=floor, intrinsics=camera, persist=False)
        station.set_baseline()
        detector.items = [can]
        result = None
        for index in range(frames):
            result = station.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=100.0 + index)
        return station, result

    def test_one_finalised_can_is_one_row_with_matching_overlay_api_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory)
            station, result = self._run(manager, detector)
            detection = result.detections[0]
            shape = detection.shape_geometry
            self.assertEqual(shape.geometry_method, CYLINDER)
            self.assertAlmostEqual(shape.cylinder_diameter_mm, 80, delta=5)
            self.assertAlmostEqual(shape.height_mm, 150, delta=5)
            # Every downstream field carries the fitted cylinder, not the footprint box.
            self.assertEqual(detection.footprint_length_mm, detection.footprint_width_mm)
            self.assertAlmostEqual(detection.footprint_length_mm, shape.cylinder_diameter_mm, places=1)
            self.assertAlmostEqual(detection.realsense_volume_l, shape.selected_volume_litres, places=5)
            api = create_app(manager.config, pipeline=manager).test_client()
            state_detection = api.get("/api/cameras/realsense/state").get_json()["latest"]["detections"][0]
            self.assertEqual(state_detection["shape_geometry"]["geometry_method"], CYLINDER)
            self.assertAlmostEqual(state_detection["realsense_volume_l"], shape.selected_volume_litres, places=5)

            response = api.get("/api/comparison/measurements.csv")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            rows = [row for row in csv.DictReader(io.StringIO(response.get_data(as_text=True).lstrip("﻿")))
                    if any(value not in ("", None) for value in row.values())]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["geometry_method"], CYLINDER)
            self.assertAlmostEqual(float(row["diameter_mm"]), shape.cylinder_diameter_mm, places=1)
            self.assertAlmostEqual(float(row["volume_liters"]), shape.selected_volume_litres, places=5)
            for column in ("timestamp", "session_id", "comparison_event_id", "measurement_id", "camera",
                           "object_type", "colour", "height_mm", "status"):
                self.assertNotEqual(row[column], "", column)
            per_camera = list(csv.DictReader(io.StringIO(
                api.get("/api/cameras/realsense/measurements.csv").get_data(as_text=True).lstrip("﻿"))))
            self.assertEqual(len(per_camera), 1)
            self.assertEqual(per_camera[0]["event_id"], row["measurement_id"])

    def test_csv_failure_keeps_tracking_and_retry_writes_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory)
            station = manager.camera("realsense")
            with mock.patch.object(type(station.event_log), "_append", side_effect=OSError("disk full")):
                _, result = self._run(manager, detector)
            self.assertEqual(result.detections[0].track_id, 1)
            self.assertEqual(station.event_log.status()["persistence_failures"], 1)
            self.assertEqual(station.event_log.retry_failed()["recovered"], 1)
            self.assertEqual(station.event_log.retry_failed()["recovered"], 0)
            self.assertEqual(len(station.event_log.rows()), 1)

    def test_unsettled_object_is_recorded_once_as_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory, finalise_max_frames=4, settle_volume_tolerance=0.0)
            station = manager.camera("realsense")
            station._is_settled = lambda track_id: False  # a volume that never stabilises
            self._run(manager, detector, frames=10)
            rows = station.event_log.rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "rejected")
            self.assertEqual(rows[0]["reason"], "unstable_volume")

    def test_hardware_diagnostic_bundle_is_written_for_a_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self._station(directory, hardware_diagnostic=True)
            self._run(manager, detector)
            bundles = list((Path(directory) / "realsense" / "hardware_diagnostics" / "realsense").glob("*_measurement_*"))
            self.assertEqual(len(bundles), 1)
            names = {path.name for path in bundles[0].iterdir()}
            self.assertTrue({"original.png", "object_mask.png", "depth.png", "points.npy", "result.json"} <= names)
            result = json.loads((bundles[0] / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["geometry"]["geometry_method"], CYLINDER)
            self.assertIn("csv_row", result)
            scenes = list((Path(directory) / "realsense" / "hardware_diagnostics" / "realsense").glob("*_scene_*"))
            self.assertGreaterEqual(len(scenes), 1)


class LogitechObjectMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.baseline = rng.integers(60, 190, (120, 160, 3), dtype=np.uint8)  # textured floor and sofa
        self.region = np.ones((120, 160), dtype=bool)
        self.object = np.zeros((120, 160), dtype=bool)
        self.object[40:80, 60:100] = True
        self.frame = self.baseline.copy()
        self.frame[self.object] = (40, 85, 145)
        self.shadow = np.zeros_like(self.object)
        self.shadow[80:95, 60:100] = True
        self.frame[self.shadow] = (self.baseline[self.shadow] * 0.6).astype(np.uint8)
        self.whole_roi = Detection("backpack", 0.8, (0, 0, 160, 120), np.ones((120, 160), dtype=bool))

    def test_whole_roi_detector_mask_is_cut_to_the_deposited_object(self) -> None:
        debug: dict = {}
        kept, _ = bound_logitech_detections(self.frame, self.baseline, [self.whole_roi], self.region,
                                            min_pixels=20, debug=debug)
        self.assertEqual(len(kept), 1)
        mask = kept[0].mask
        self.assertGreater((mask & self.object).sum() / self.object.sum(), 0.9)
        self.assertLess((mask & ~self.object).sum(), 0.1 * self.object.sum())
        self.assertLess((mask & self.shadow).sum(), 0.2 * self.shadow.sum())
        self.assertTrue(debug["final_mask_valid"])
        for key in ("detector", "foreground", "final", "rejected"):
            self.assertEqual(debug[key].shape, self.object.shape)
        self.assertGreater(debug["rejected"].sum(), 0.8 * (~self.object).sum())

    def test_no_empty_reference_means_no_object_mask(self) -> None:
        debug: dict = {}
        kept, warnings = bound_logitech_detections(self.frame, None, [self.whole_roi], self.region,
                                                   min_pixels=20, debug=debug)
        self.assertEqual(kept, [])
        self.assertIn("missing_empty_reference", debug["reasons"])
        self.assertTrue(warnings)

    def test_uncalibrated_logitech_says_calibration_required(self) -> None:
        self.assertEqual(RELATIVE_ONLY_MESSAGE, "CALIBRATION REQUIRED — metric volume unavailable")
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(results_dir=Path(directory), roi=(0, 0, 1, 1), logitech_reference_distance_m=0.0)
            manager = DualCameraCoordinator(config, detector=SharedDetector(), depth_estimator=NoDepth())
            camera = CameraIntrinsics(fx=100, fy=100, ppx=80, ppy=60, width=160, height=120)
            logitech = manager.camera("logitech")
            logitech.process_frame(self.baseline, intrinsics=camera, persist=False)
            logitech.set_baseline()
            chain = logitech.logitech_diagnostics()
            self.assertTrue(chain["da_v2_loaded"])
            self.assertFalse(chain["calibration_loaded"])
            self.assertIn("CALIBRATION REQUIRED", logitech.logitech_calibration_status()["message"])


if __name__ == "__main__":
    unittest.main()
