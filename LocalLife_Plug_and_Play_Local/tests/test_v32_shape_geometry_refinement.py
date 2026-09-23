"""V32: a shape is fitted from evidence, and belongs to the object it was fitted on."""

from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import _object_signature, _spans_the_background, _tracked_component
from locallife_cloud.shape_geometry import (
    CUBOID,
    CYLINDER,
    FLEXIBLE_OR_UNKNOWN,
    UNCERTAIN,
    GeometryLock,
    ObjectSignature,
    ShapeGeometry,
    measure_shape,
)
from locallife_cloud.types import CameraIntrinsics, Detection

PROJECT = Path(__file__).resolve().parents[1]
V31_SHA = "20184e07c4b9cd49e291faf7d549420c584e59c3"
RNG = np.random.default_rng(32)


def _grid(length: float, width: float, step: float = 0.002) -> np.ndarray:
    x, y = np.meshgrid(np.arange(-length / 2, length / 2, step), np.arange(-width / 2, width / 2, step))
    return np.column_stack((x.ravel(), y.ravel()))


def _noise(count: int) -> np.ndarray:
    return RNG.normal(0.0, 0.0015, count)


def _disc(radius: float, step: float = 0.002) -> np.ndarray:
    points = _grid(2 * radius, 2 * radius, step)
    return points[np.hypot(points[:, 0], points[:, 1]) <= radius]


def _cylinder_shape(diameter_mm: float, height_mm: float) -> ShapeGeometry:
    radius = diameter_mm / 2000.0
    litres = math.pi * radius ** 2 * (height_mm / 1000.0) * 1000.0
    return ShapeGeometry(
        geometry_method=CYLINDER, geometry_confidence=0.8, length_mm=diameter_mm,
        width_mm=diameter_mm, height_mm=height_mm, bounding_box_volume_litres=litres * 1.27,
        mesh_volume_litres=litres, selected_volume_litres=litres,
        volume_meaning="cylinder_volume_pi_r2_h", cylinder_diameter_mm=diameter_mm,
        cylinder_height_mm=height_mm, cylinder_volume_litres=litres, cylinder_orientation="upright",
        cylinder_fit_residual=0.01, radius_mm=diameter_mm / 2.0, fit_confidence=0.8, points=900,
    )


def _irregular_shape() -> ShapeGeometry:
    return ShapeGeometry(
        geometry_method=FLEXIBLE_OR_UNKNOWN, geometry_confidence=0.5, length_mm=320.0,
        width_mm=240.0, height_mm=150.0, bounding_box_volume_litres=11.5,
        mesh_volume_litres=6.4, selected_volume_litres=6.4,
        volume_meaning="current_external_occupied_volume", points=4000,
    )


def _signature(**overrides) -> ObjectSignature:
    values = dict(camera="logitech", label="bottle", centre_x=100.0, centre_y=80.0,
                  area=4000.0, aspect=0.5)
    values.update(overrides)
    return ObjectSignature(**values)


class Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames):
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class MetricDepth:
    """1.5 m floor, object 12 cm closer -- a metric checkpoint's output."""

    device = "cpu"

    def estimate_batch(self, frames):
        return [np.where(frame.max(axis=2) > 0, 1.38, 1.5).astype(np.float32) for frame in frames]


def _station(directory: str, **overrides):
    detector = Detector()
    settings = dict(
        results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
        tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
        operating_mode="geometry_validation", auto_deposit=False,
        logitech_reference_distance_m=0.0, automatic_baseline=False,
    )
    settings.update(overrides)
    manager = DualCameraCoordinator(AppConfig(**settings), detector=detector,
                                    depth_estimator=MetricDepth())
    camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
    return manager, detector, camera


def _frame_and_detection(label: str = "cosmetic bottle", box=(60, 40, 110, 90)):
    mask = np.zeros((120, 160), dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[mask] = (40, 40, 210)
    return frame, Detection(label, 0.7, box, mask, color="red")


def _run(station, frame, camera, frames=6, start=10.0):
    result = None
    for index in range(frames):
        result = station.process_frame(frame, intrinsics=camera, timestamp=start + index)
    return result


class ShapeEvidenceTests(unittest.TestCase):
    """A label is a hint; the points decide."""

    def test_a_cylinder_word_on_a_box_stays_a_box(self) -> None:
        points = _grid(0.20, 0.12)
        heights = 0.10 + _noise(len(points))
        for label in ("cream container", "beverage carton", "body lotion", "balm container"):
            with self.subTest(label=label):
                result = measure_shape(points, heights, mesh_volume_l=2.3, label=label)
                self.assertEqual(result.geometry_method, CUBOID)
                self.assertIsNone(result.cylinder_diameter_mm)

    def test_flexible_classes_are_never_fitted_with_a_solid(self) -> None:
        points = _disc(0.14, 0.003)
        radial = np.hypot(points[:, 0], points[:, 1]) / 0.14
        heights = 0.18 * np.sqrt(np.clip(1 - radial ** 2, 0, None)) + _noise(len(points))
        for label in ("backpack", "rucksack", "handbag", "clothing bundle", "fabric roll"):
            with self.subTest(label=label):
                result = measure_shape(points, heights, mesh_volume_l=5.1, label=label)
                self.assertEqual(result.geometry_method, FLEXIBLE_OR_UNKNOWN)
                self.assertEqual(result.selected_volume_litres, 5.1)

    def test_a_real_upright_cylinder_is_still_measured_as_one(self) -> None:
        points = _disc(0.0445)
        heights = 0.181 + _noise(len(points))
        mesh = math.pi * 0.0445 ** 2 * 0.181 * 1000.0
        result = measure_shape(points, heights, mesh_volume_l=mesh, label="bottle")
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertAlmostEqual(result.cylinder_diameter_mm, 89, delta=6)
        self.assertAlmostEqual(result.height_mm, 181, delta=6)

    def test_a_fit_contradicting_the_height_map_is_refused(self) -> None:
        points = _disc(0.0445)
        heights = 0.181 + _noise(len(points))
        result = measure_shape(points, heights, mesh_volume_l=0.15, label="bottle")
        self.assertNotEqual(result.geometry_method, CYLINDER)
        self.assertIn("cylinder_fit_rejected:cylinder_disagrees_with_height_map", result.flags)
        self.assertIn("height_map_fallback", result.flags)
        self.assertEqual(result.selected_volume_litres, 0.15)

    def test_a_thin_fragment_is_not_read_as_a_side_on_cylinder(self) -> None:
        points = np.column_stack((np.linspace(-0.02, 0.02, 60), RNG.normal(0, 0.0005, 60)))
        result = measure_shape(points, 0.05 + _noise(len(points)), mesh_volume_l=0.03, min_points=30)
        self.assertNotEqual(result.geometry_method, CYLINDER)


class GeometryOwnershipTests(unittest.TestCase):
    """A frozen shape belongs to the object it was fitted on."""

    def _frozen_lock(self) -> GeometryLock:
        lock = GeometryLock(required_frames=3, window=5)
        bottle = _signature()
        for _ in range(3):
            lock.update(1, _cylinder_shape(89.0, 181.0), bottle)
        self.assertIsNotNone(lock.frozen(1))
        return lock

    def test_a_frozen_cylinder_cannot_leak_into_the_next_track(self) -> None:
        for label, signature in (
            ("backpack", _signature(label="backpack", area=48000.0, aspect=0.9)),
            ("carton", _signature(label="carton", centre_x=20.0, centre_y=20.0)),
            ("background", _signature(label="", area=90000.0, aspect=0.95)),
        ):
            with self.subTest(next_object=label):
                lock = self._frozen_lock()
                result = lock.update(1, _irregular_shape(), signature)
                self.assertEqual(result.geometry_method, FLEXIBLE_OR_UNKNOWN)
                self.assertNotEqual(round(result.length_mm), 89)
                self.assertNotEqual(round(result.height_mm), 181)
                self.assertIsNone(lock.frozen(1))

    def test_the_same_object_keeps_its_frozen_measurement(self) -> None:
        lock = self._frozen_lock()
        moved = _signature(centre_x=112.0, centre_y=86.0, area=4600.0, aspect=0.54)
        result = lock.update(1, _irregular_shape(), moved)
        self.assertEqual(result.geometry_method, CYLINDER)
        self.assertEqual(round(result.cylinder_diameter_mm), 89)

    def test_geometry_does_not_cross_between_cameras(self) -> None:
        lock = self._frozen_lock()
        result = lock.update(1, _irregular_shape(), _signature(camera="realsense"))
        self.assertEqual(result.geometry_method, FLEXIBLE_OR_UNKNOWN)

    def test_an_unstable_diameter_never_freezes(self) -> None:
        lock = GeometryLock(required_frames=3, window=6)
        signature = _signature()
        for diameter in (60.0, 120.0, 45.0, 150.0, 70.0, 130.0):
            lock.update(2, _cylinder_shape(diameter, 181.0), signature)
        self.assertIsNone(lock.frozen(2))

    def test_a_steady_diameter_still_freezes(self) -> None:
        lock = GeometryLock(required_frames=3, window=6)
        signature = _signature()
        for diameter in (88.0, 90.0, 89.0):
            lock.update(3, _cylinder_shape(diameter, 181.0), signature)
        self.assertIsNotNone(lock.frozen(3))

    def test_the_signature_is_taken_from_the_detection(self) -> None:
        _, detection = _frame_and_detection()
        signature = _object_signature(detection, "logitech")
        self.assertEqual(signature.camera, "logitech")
        self.assertEqual(signature.label, "bottle")
        self.assertAlmostEqual(signature.centre_x, 85.0)
        self.assertEqual(signature.area, float(np.count_nonzero(detection.mask)))
        # "plastic bottle" and "bottle" are one family, not two objects.
        detection.label = "plastic bottle"
        self.assertTrue(signature.matches(_object_signature(detection, "logitech")))


class MeasurementEligibilityTests(unittest.TestCase):
    """What is measured is the object, not the room it stands in."""

    def test_a_region_spanning_the_view_is_not_an_object(self) -> None:
        region = np.ones((120, 160), dtype=bool)
        background = np.zeros((120, 160), dtype=bool)
        background[55:120, 0:160] = True
        self.assertTrue(_spans_the_background(background, region))

    def test_an_object_inside_the_view_is_measurable(self) -> None:
        region = np.ones((120, 160), dtype=bool)
        mask = np.zeros((120, 160), dtype=bool)
        mask[40:90, 60:110] = True
        self.assertFalse(_spans_the_background(mask, region))

    def test_background_never_reaches_a_volume(self) -> None:
        """End to end, whichever layer refuses it first.

        The detector's class filter is untouched, so the furniture arrives
        under an accepted label -- which is how a floor and a sofa were
        reported as a 13.48 L object. It is wide, flat and runs off three
        sides of the view, and stays under the existing 62 % size filter.
        """
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _station(directory)
            frame, detection = _frame_and_detection("cosmetic bottle", (0, 55, 160, 120))
            detector.items = [detection]
            result = _run(manager.camera("logitech"), frame, camera)
            for measured in result.detections:
                self.assertIsNone(measured.monocular_volume_l)
                self.assertIsNotNone(measured.volume_rejection_reason)

    def test_a_normal_object_is_still_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _station(directory)
            frame, detection = _frame_and_detection()
            detector.items = [detection]
            measured = _run(manager.camera("logitech"), frame, camera).detections[0]
            self.assertIsNone(measured.volume_rejection_reason)
            self.assertIsNotNone(measured.monocular_volume_l)
            self.assertGreater(measured.monocular_volume_l, 0)

    def test_a_merged_mask_measures_only_the_tracked_object(self) -> None:
        mask = np.zeros((120, 160), dtype=bool)
        mask[40:90, 20:60] = True   # the tracked object
        mask[40:90, 90:150] = True  # a second object in the same mask
        kept = _tracked_component(mask, (20, 40, 60, 90))
        self.assertEqual(int(np.count_nonzero(kept)), 50 * 40)
        self.assertFalse(kept[60, 120])

    def test_a_rejected_analytic_shape_keeps_the_height_map_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            logitech = manager.camera("logitech")
            _, detection = _frame_and_detection()
            detection.monocular_volume_l = 1.42
            detection.shape_geometry = ShapeGeometry(
                geometry_method=UNCERTAIN, geometry_confidence=0.2, length_mm=89.0,
                width_mm=89.0, height_mm=181.0, bounding_box_volume_litres=1.43,
                mesh_volume_litres=None, selected_volume_litres=None, volume_meaning="unknown",
                rejection_reason="insufficient_arc_coverage", points=300,
            )
            logitech._apply_cylinder_geometry(detection)
            self.assertEqual(detection.monocular_volume_l, 1.42)
            self.assertIsNone(detection.volume_rejection_reason)

    def test_a_rejection_with_nothing_measured_still_states_its_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            logitech = manager.camera("logitech")
            _, detection = _frame_and_detection()
            detection.shape_geometry = ShapeGeometry(
                geometry_method=UNCERTAIN, geometry_confidence=0.2, length_mm=89.0,
                width_mm=89.0, height_mm=181.0, bounding_box_volume_litres=1.43,
                mesh_volume_litres=None, selected_volume_litres=None, volume_meaning="unknown",
                rejection_reason="insufficient_arc_coverage", points=300,
            )
            logitech._apply_cylinder_geometry(detection)
            self.assertIsNone(detection.monocular_volume_l)
            self.assertEqual(detection.volume_rejection_reason, "insufficient_arc_coverage")


class ProtectedSurfacesTests(unittest.TestCase):
    def test_csv_ledger_detector_and_cloud_are_unchanged_since_v31(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/heightmap_volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/footprint.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/tracking.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/inference.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(["git", "diff", "--name-only", V31_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V31 commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
