"""V39: edge pixels, background in the baseline, and telling a ball from a can.

Ray-traced scenes and synthetic depth only. They show what the code does with
known geometry; they are not hardware measurements.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from locallife_cloud.depth_edges import drop_depth_edge_pixels
from locallife_cloud.logitech_volume import metric_object_volume
from locallife_cloud.shape_geometry import CUBOID, CYLINDER, SPHERE, ShapeGeometry, measure_shape
from locallife_cloud.shape_router import refine_shape
from locallife_cloud.volume import estimate_object_dimensions, fit_reference_plane, object_plane_points

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v37_logitech_pose_and_colour as rig  # noqa: E402

CAMERA = rig.CAMERA


def _sphere(scene, centre, radius):
    origin, direction = scene.origin, scene.direction
    offset = origin - np.array((centre[0], centre[1], radius))
    b = 2.0 * (direction * offset).sum(-1)
    a = (direction * direction).sum(-1)
    c = (offset * offset).sum() - radius * radius
    disc = b * b - 4.0 * a * c
    t = (-b - np.sqrt(np.where(disc >= 0, disc, 0.0))) / (2.0 * a)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def _edge_ring(depth, mask, width=2, blend="smooth"):
    """Pixels just outside the silhouette given depths between object and floor."""
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((2 * width + 1,) * 2, np.uint8)) > 0) & ~mask
    near = cv2.erode(np.where(mask, depth, 9.0).astype(np.float32), np.ones((2 * width + 1,) * 2, np.uint8))
    if blend == "smooth":            # stereo interpolation / monocular blur
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.clip(distance / (width + 1.0), 0.0, 1.0)
    else:
        weight = np.random.default_rng(0).uniform(0.1, 0.9, depth.shape)
    out = depth.copy()
    out[ring] = ((1 - weight) * near + weight * depth)[ring]
    return out, mask | ring


def _dimensions(scene, depth, mask):
    plane = fit_reference_plane(scene.empty, CAMERA, mask=scene.region)
    result = estimate_object_dimensions(depth, CAMERA, mask, plane, min_height_m=0.01,
                                        max_height_m=1.0, min_points=20)
    return result.length_mm, result.width_mm, result.height_mm


OBJECTS = {
    "can": lambda s: [s.cylinder((0, 0), 0.035, 0.164)],
    "milk carton": lambda s: [s.box((0, 0), 0.07, 0.095, 0.23)],
    "flat laptop bag": lambda s: [s.box((0, 0), 0.40, 0.29, 0.02)],
    "irregular bag": lambda s: [s.box((0, 0), 0.20, 0.15, 0.12), s.box((0.12, 0), 0.10, 0.12, 0.06)],
}


class DepthEdgeTests(unittest.TestCase):
    """RealSense flying pixels and monocular edge blur smear the footprint."""

    def test_edge_pixels_inflate_the_footprint_and_the_filter_restores_it(self) -> None:
        for tilt in (25.0, 45.0, 60.0):
            for blend in ("smooth", "random"):
                with self.subTest(tilt=tilt, blend=blend):
                    scene = rig.Scene(tilt)
                    depth, mask = scene.render(*OBJECTS["can"](scene))
                    clean = _dimensions(scene, depth, mask)
                    smeared, wide = _edge_ring(depth, mask, blend=blend)
                    before = _dimensions(scene, smeared, wide)
                    self.assertGreater(before[0], 1.5 * clean[0])        # the artefact
                    kept, info = drop_depth_edge_pixels(smeared, wide)
                    after = _dimensions(scene, smeared, kept)
                    self.assertTrue(info["applied"])
                    self.assertLess(abs(after[0] - clean[0]) / clean[0], 0.10, (clean, before, after))
                    self.assertAlmostEqual(after[2], clean[2], delta=3.0)

    def test_previously_good_objects_are_not_changed(self) -> None:
        # Regression fixtures for the cases that already measured well.
        for tilt in (25.0, 45.0, 60.0):
            for name, build in OBJECTS.items():
                with self.subTest(tilt=tilt, object=name):
                    scene = rig.Scene(tilt)
                    depth, mask = scene.render(*build(scene))
                    clean = _dimensions(scene, depth, mask)
                    kept, _ = drop_depth_edge_pixels(depth, mask)
                    after = _dimensions(scene, depth, kept)
                    for value, reference in zip(after, clean):
                        self.assertLess(abs(value - reference) / reference, 0.05, (name, clean, after))

    def test_a_mask_that_is_mostly_edge_is_left_alone(self) -> None:
        depth = np.tile(np.linspace(1.0, 1.4, 40, dtype=np.float32), (40, 1))
        depth[:, ::2] += 0.05                       # every other column steps
        mask = np.ones_like(depth, dtype=bool)
        kept, info = drop_depth_edge_pixels(depth, mask)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "would_remove_too_much_of_the_mask")
        self.assertTrue(np.array_equal(kept, mask))

    def test_monocular_edge_blur_widens_a_carton_and_is_removed(self) -> None:
        scene = rig.Scene(25.0)
        depth, mask = scene.render(*OBJECTS["milk carton"](scene))
        blurred, wide = _edge_ring(depth, mask, width=3)
        plane = fit_reference_plane(scene.empty, CAMERA, mask=scene.region)

        def breadth(m):
            result = metric_object_volume(blurred, CAMERA, m, plane, reference_depth_m=scene.empty,
                                          min_height_m=0.004, min_pixels=25, cell_size_m=0.005)
            return result.diagnostics["length_mm"], result.diagnostics["width_mm"]

        before = breadth(wide)
        kept, _ = drop_depth_edge_pixels(blurred, wide, radius=3)
        after = breadth(kept)
        self.assertGreater(max(before), 1.5 * 95.0)
        self.assertLess(max(after), 95.0 * 1.15)


class _FixedDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, detection):
        self.detection = detection

    def detect_batch(self, frames):
        from locallife_cloud.types import Detection

        return [[Detection(self.detection.label, self.detection.confidence, self.detection.box,
                           mask=self.detection.mask.copy())] for _ in frames]


class RealSensePipelineEdgeTests(unittest.TestCase):
    """The RealSense path itself, with a ring of flying pixels around a box."""

    def _run(self, with_ring: bool, label: str = "cosmetic bottle"):
        from locallife_cloud.config import AppConfig
        from locallife_cloud.pipeline import VisionPipeline
        from locallife_cloud.types import CameraIntrinsics, Detection

        size, floor = 200, 1.2
        mask = np.zeros((size, size), dtype=bool)
        mask[80:120, 70:130] = True
        detector_mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(results_dir=Path(directory), enable_monocular_depth=False,
                               enable_material_classification=False, roi=(0, 0, 1, 1),
                               tracker_confirm_frames=1, operating_mode="geometry_validation")
            pipeline = VisionPipeline(config, detector=_FixedDetector(
                Detection(label, 0.9, (67, 77, 133, 123), mask=detector_mask)))
            camera = CameraIntrinsics(fx=600, fy=600, ppx=size / 2, ppy=size / 2, width=size, height=size)
            empty = np.zeros((size, size, 3), dtype=np.uint8)
            baseline = np.full((size, size), floor, dtype=np.float32)
            pipeline.set_baseline(empty, baseline, camera)
            depth = baseline.copy()
            depth[mask] = floor - 0.10
            if with_ring:
                ring = detector_mask & ~mask
                distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
                weight = np.clip(distance / 4.0, 0, 1)
                depth[ring] = ((1 - weight) * (floor - 0.10) + weight * floor)[ring]
            frame = empty.copy()
            frame[detector_mask] = (40, 40, 200)
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=1.0)
        return [item for item in result.detections if item.confidence > 0][0]

    def test_flying_pixels_do_not_change_the_realsense_dimensions(self) -> None:
        # Truth: 60 x 40 px at 1.1 m with f = 600 px is about 110 x 73 x 100 mm.
        for label in ("cosmetic bottle", "box"):       # support-plane path, box-cuboid path
            with self.subTest(label=label):
                clean = self._run(False, label)
                ringed = self._run(True, label)
                for name in ("footprint_length_mm", "footprint_width_mm", "physical_height_mm"):
                    a, b = getattr(clean, name), getattr(ringed, name)
                    self.assertIsNotNone(a, name)
                    self.assertLess(abs(a - b) / a, 0.10, (name, a, b))

    def test_without_the_filter_the_support_plane_path_is_smeared(self) -> None:
        clean = self._run(False)
        with mock.patch("locallife_cloud.pipeline.drop_depth_edge_pixels",
                        lambda depth, mask, **kwargs: (mask, {"applied": False})):
            ringed = self._run(True)
        self.assertGreater(ringed.footprint_length_mm, 1.1 * clean.footprint_length_mm)


class BackgroundFallbackTests(unittest.TestCase):
    """An unchanged object that runs out of the picture is the room."""

    def test_a_blanket_entering_from_the_frame_edge_is_refused(self) -> None:
        from test_v34_independent_dual_camera import SHAPE, _manager, _object

        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory, baseline_contains_object_frames=2)
            station = manager.camera("realsense")
            _, detection = _object()
            detection.track_id = 9
            blanket = np.zeros(SHAPE, dtype=bool)
            blanket[: SHAPE[0] // 2, : SHAPE[1] // 3] = True   # enters from the top-left corner
            detection.mask = blanket
            region = np.ones(SHAPE, dtype=bool)
            unchanged = np.zeros(SHAPE, dtype=bool)
            for _ in range(3):
                kept = station._deposit_measurement_mask(detection, blanket.copy(), region, unchanged, None)
            self.assertFalse(np.any(kept))
            self.assertEqual(detection.volume_rejection_reason, "unchanged_object_leaves_the_frame")


class ShapeRouterTests(unittest.TestCase):
    def _route(self, tilt, build, label="object"):
        scene = rig.Scene(tilt)
        depth, mask = scene.render(*build(scene))
        plane = fit_reference_plane(scene.empty, CAMERA, mask=scene.region)
        points = object_plane_points(depth, CAMERA, mask, plane, min_height_m=0.004, max_height_m=1.0)
        mesh = metric_object_volume(depth, CAMERA, mask, plane, reference_depth_m=scene.empty,
                                    min_height_m=0.004, min_pixels=25, cell_size_m=0.005).measurement.liters
        return measure_shape(*points, mesh_volume_l=mesh, label=label), refine_shape(
            measure_shape(*points, mesh_volume_l=mesh, label=label), *points)

    def test_a_ball_was_a_cylinder_and_is_now_a_sphere(self) -> None:
        for tilt in (15.0, 25.0, 45.0, 60.0):
            for radius in (0.05, 0.10):
                with self.subTest(tilt=tilt, radius=radius):
                    before, after = self._route(tilt, lambda s: [_sphere(s, (0, 0), radius)])
                    self.assertEqual(after.geometry_method, SPHERE)
                    truth = 4.0 / 3.0 * np.pi * radius ** 3 * 1000.0
                    self.assertLess(abs(after.selected_volume_litres - truth) / truth, 0.05)
        before, _ = self._route(25.0, lambda s: [_sphere(s, (0, 0), 0.10)])
        self.assertEqual(before.geometry_method, CYLINDER)     # the defect, kept as evidence

    def test_flat_topped_objects_are_never_spheres(self) -> None:
        cases = {
            "can": (lambda s: [s.cylinder((0, 0), 0.035, 0.164)], CYLINDER),
            "squat can": (lambda s: [s.cylinder((0, 0), 0.04, 0.08)], CYLINDER),
            "cube": (lambda s: [s.box((0, 0), 0.2, 0.2, 0.2)], None),
            "carton": (lambda s: [s.box((0, 0), 0.07, 0.095, 0.23)], None),
        }
        for tilt in (15.0, 25.0, 45.0, 60.0):
            for name, (build, expected) in cases.items():
                with self.subTest(tilt=tilt, object=name):
                    _, after = self._route(tilt, build)
                    self.assertNotEqual(after.geometry_method, SPHERE)
                    if expected:
                        self.assertEqual(after.geometry_method, expected)

    def test_lumpy_bags_are_not_spheres(self) -> None:
        bags = {
            "lumpy": lambda s: [s.box((0, 0), 0.20, 0.15, 0.12), s.box((0.12, 0), 0.10, 0.12, 0.06),
                                _sphere(s, (-0.05, 0.03), 0.07)],
            "ball beside a box": lambda s: [_sphere(s, (0, 0), 0.10), s.box((0.1, 0.05), 0.12, 0.1, 0.14)],
        }
        for tilt in (25.0, 45.0, 60.0):
            for name, build in bags.items():
                with self.subTest(tilt=tilt, object=name):
                    _, after = self._route(tilt, build)
                    self.assertNotEqual(after.geometry_method, SPHERE)

    def test_the_label_does_not_decide(self) -> None:
        _, ball = self._route(25.0, lambda s: [_sphere(s, (0, 0), 0.10)], label="can")
        self.assertEqual(ball.geometry_method, SPHERE)
        _, can = self._route(25.0, lambda s: [s.cylinder((0, 0), 0.035, 0.164)], label="ball")
        self.assertEqual(can.geometry_method, CYLINDER)

    def test_the_pipeline_publishes_a_sphere_from_its_median_height(self) -> None:
        from locallife_cloud.pipeline import VisionPipeline
        from locallife_cloud.types import Detection

        station = VisionPipeline.__new__(VisionPipeline)
        station.camera_id = "realsense"
        shape = ShapeGeometry(SPHERE, 0.8, 200.0, 200.0, 200.0, None, 4.9, 4.9, "sphere",
                              radius_mm=100.0)
        detection = Detection("ball", 0.9, (0, 0, 10, 10))
        detection.shape_geometry = shape
        station._apply_cylinder_geometry(detection)
        self.assertEqual(detection.dimension_method, "fitted_sphere")
        self.assertAlmostEqual(detection.realsense_volume_l, 4.18879, places=3)
        self.assertEqual(detection.physical_height_mm, 200.0)


if __name__ == "__main__":
    unittest.main()
