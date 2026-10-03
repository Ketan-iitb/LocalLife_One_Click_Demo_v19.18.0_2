"""Measurement-integrity regression tests (improvement/locallife-measurement-integrity).

All scenes here are SYNTHETIC and exact (tests/synthetic_scenes.py): they check
that the geometry is self-consistent and that reported numbers carry the right
provenance. They do not establish physical accuracy of a RealSense or Logitech.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_scenes import Box, intrinsics, render  # noqa: E402

from locallife_cloud.box_templates import BoxTemplate  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.types import CameraIntrinsics, Detection  # noqa: E402
from locallife_cloud.volume import (  # noqa: E402
    aggregate_box_measurements,
    estimate_box_volume_cuboid,
    estimate_volume,
    fit_reference_plane,
)

K = intrinsics()


def _cuboid(box: Box, pitch: float = 0.0, height: float = 1.0, **render_kw):
    depth, empty, masks = render(K, pitch_deg=pitch, camera_height_m=height, boxes=(box,), **render_kw)
    plane = fit_reference_plane(empty, K)
    return estimate_box_volume_cuboid(depth, K, masks[0], plane), depth, empty, masks[0], plane


class ElevatedFootprintTests(unittest.TestCase):
    """The controlled case from the review: flat top at 0.8 m, floor at 1.0 m."""

    def setUp(self) -> None:
        self.camera = CameraIntrinsics(fx=100.0, fy=100.0, ppx=15.0, ppy=15.0, width=30, height=30)
        self.baseline = np.full((30, 30), 1.0, np.float32)
        self.depth = np.full((30, 30), 0.8, np.float32)   # 900 px, 0.0576 m^2 of top face

    def _litres(self, mode: str) -> float:
        return estimate_volume(self.depth, self.baseline, self.camera, geometry_mode=mode).liters

    def test_the_three_modes_reproduce_and_reference_plane_no_longer_inflates(self) -> None:
        self.assertAlmostEqual(self._litres("surface-columns"), 11.52, places=3)   # 0.0576 m^2 x 0.2 m
        self.assertAlmostEqual(self._litres("ray-frustum"), 14.64, places=3)       # frustum incl. occluded shadow
        # Previously 18.00 L: the footprint was taken at the 1.0 m floor, (1.0/0.8)^2 too large.
        self.assertAlmostEqual(self._litres("reference-plane"), 11.52, places=3)


class SyntheticSceneGeometryTests(unittest.TestCase):
    def test_height_map_grid_has_no_boundary_cell_inflation(self) -> None:
        # A 7 x 7 x 23 cm carton covered 8 x 8 ten-millimetre cells: +31 % before
        # boundary cells were weighted by the floor area their samples cover.
        for box, pitch in ((Box((0, 0), .07, .07, .23, 30), 0.0), (Box((0, 0), .20, .09, .09), 0.0),
                           (Box((0, .1), .20, .09, .09), 30.0), (Box((0, .1), .30, .20, .10, 40), 30.0),
                           (Box((0, 0), .30, .20, .02, 10), 0.0)):
            with self.subTest(box=box, pitch=pitch):
                depth, empty, masks = render(K, pitch_deg=pitch, camera_height_m=1.0, boxes=(box,))
                plane = fit_reference_plane(empty, K)
                result = estimate_volume(depth, empty, K, object_mask=masks[0], geometry_mode="height-map-grid",
                                         reference_plane=plane, min_height_m=0.005)
                truth = box.length * box.width * box.height * 1000.0
                self.assertLess(abs(result.liters - truth) / truth, 0.03)

    def test_cuboid_overhead_rotated_tilted_and_near_square(self) -> None:
        cases = ((Box((0, 0), .20, .09, .09), 0.0, 0.01), (Box((0, 0), .41, .33, .14, 15), 0.0, 0.01),
                 (Box((0, 0), .07, .07, .23, 45), 0.0, 0.05),     # PCA axes are arbitrary here
                 (Box((0, .1), .20, .09, .09), 30.0, 0.04), (Box((0, .1), .30, .20, .10, 40), 30.0, 0.04))
        for box, pitch, tolerance in cases:
            with self.subTest(box=box, pitch=pitch):
                result, *_ = _cuboid(box, pitch)
                truth = box.length * box.width * box.height * 1000.0
                self.assertLess(abs(result.volume_liters - truth) / truth, tolerance)
                self.assertAlmostEqual(result.length_mm, box.length * 1000, delta=max(3.0, 0.03 * box.length * 1000))
                self.assertAlmostEqual(result.width_mm, box.width * 1000, delta=max(3.0, 0.04 * box.width * 1000))

    def test_noisy_depth_does_not_bias_the_top_height_upwards(self) -> None:
        # The median of the upper 10 % of a flat noisy top is its noise tail.
        result, *_ = _cuboid(Box((0, .05), .20, .09, .09, 20), 20.0, noise_m=0.003, dropout=0.05, seed=1)
        self.assertAlmostEqual(result.height_mm, 90.0, delta=2.0)
        truth = 1.62
        self.assertLess(abs(result.volume_liters - truth), 3.0 * result.uncertainty_l)

    def test_thin_object_below_the_height_floor_is_refused_not_forced(self) -> None:
        result, *_ = _cuboid(Box((0, 0), .30, .20, .02, 10))
        self.assertIsNone(result)          # 2 cm < 2.5 cm minimum: no invented cuboid

    def test_frame_edge_clipping_is_flagged_as_a_lower_bound(self) -> None:
        result, *_ = _cuboid(Box((0.48, 0), .20, .09, .09))
        self.assertIn("partial_view_lower_bound", result.flags)
        self.assertLess(result.volume_liters, 1.62)

    def test_partial_mask_is_measured_as_what_is_visible(self) -> None:
        # A mask that lost half the object (occlusion) cannot be detected from
        # geometry alone: the result is the visible part, never "repaired".
        depth, empty, masks = render(K, camera_height_m=1.0, boxes=(Box((0, 0), .20, .09, .09),))
        mask = masks[0].copy()
        columns = np.nonzero(mask.any(axis=0))[0]
        mask[:, : columns[len(columns) // 2]] = False
        result = estimate_box_volume_cuboid(depth, K, mask, fit_reference_plane(empty, K))
        self.assertLess(result.volume_liters, 0.9)

    def test_non_flat_bag_height_map_single_view(self) -> None:
        dome = lambda x, y: np.clip(0.15 * (1 - ((x / 0.15) ** 2 + (y / 0.12) ** 2)), 0, None)  # noqa: E731
        depth, empty, masks = render(K, pitch_deg=20.0, camera_height_m=1.0, heightfield=dome)
        plane = fit_reference_plane(empty, K)
        result = estimate_volume(depth, empty, K, object_mask=masks[0], geometry_mode="height-map-grid",
                                 reference_plane=plane, min_height_m=0.005)
        truth = np.pi * 0.15 * 0.12 * 0.15 / 2 * 1000          # paraboloid cap
        # The slope facing away is partly unobserved from one tilted view.
        self.assertLess(abs(result.liters - truth) / truth, 0.10)


class CuboidProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        from locallife_cloud.pipeline import VisionPipeline

        self.result, *_ = _cuboid(Box((0, 0), .20, .09, .09))
        self.station = VisionPipeline.__new__(VisionPipeline)
        self.station.config = AppConfig()
        self.station.box_templates = [BoxTemplate(
            id="carton_200x90x90", nominal_volume_liters=1.5, length_mm=200, width_mm=90, height_mm=90,
            tolerance_mm=10, measured=True)]

    def _apply(self, cuboid=None) -> Detection:
        detection = Detection("cardboard box", 0.9, (0, 0, 10, 10))
        self.station._apply_box_cuboid(detection, cuboid or self.result, [])
        return detection

    def test_calibration_factor_reaches_the_cuboid_and_dimensions_are_not_scaled(self) -> None:
        self.station.config.volume_calibration_factor = 1.2
        detection = self._apply()
        self.assertAlmostEqual(detection.volume_raw_geometric_l, self.result.volume_liters, places=5)
        self.assertAlmostEqual(detection.realsense_volume_l, self.result.volume_liters * 1.2, places=5)
        self.assertAlmostEqual(detection.box_length_mm, self.result.length_mm, places=1)
        self.assertIn("1.2000", detection.volume_relationship)
        self.assertAlmostEqual(detection.volume_uncertainty_l, self.result.uncertainty_l * 1.2, places=5)
        self.assertTrue(detection.uncertainty_method.startswith("table_relative_cuboid"))

    def test_aggregated_track_result_follows_the_same_policy(self) -> None:
        self.station.config.volume_calibration_factor = 0.9
        aggregate = aggregate_box_measurements([self.result, self.result, self.result])
        detection = self._apply(aggregate)
        self.assertAlmostEqual(detection.realsense_volume_l, aggregate.volume_liters * 0.9, places=5)
        self.assertIsNotNone(aggregate.uncertainty_l)

    def test_template_is_metadata_by_default_and_labelled_when_opted_in(self) -> None:
        detection = self._apply()
        self.assertEqual(detection.box_template_id, "carton_200x90x90")
        self.assertFalse(detection.box_template_volume_used)
        self.assertAlmostEqual(detection.realsense_volume_l, self.result.volume_liters, places=5)
        self.station.config.box_template_volume_override = True
        detection = self._apply()
        self.assertTrue(detection.box_template_volume_used)
        self.assertEqual(detection.realsense_volume_l, 1.5)
        self.assertIsNone(detection.volume_uncertainty_l)
        self.assertIn("TEMPLATE VALUE", detection.volume_relationship)
        self.assertEqual(detection.to_dict()["volume_provenance"]["raw_geometric_volume_l"],
                         round(self.result.volume_liters, 6))

    def test_uncertainty_replaces_the_per_pixel_one(self) -> None:
        detection = Detection("cardboard box", 0.9, (0, 0, 10, 10))
        detection.volume_uncertainty_l, detection.uncertainty_method = 9.99, "height-map-grid: ..."
        self.station._apply_box_cuboid(detection, self.result, [])
        self.assertNotEqual(detection.volume_uncertainty_l, 9.99)
        self.assertTrue(detection.uncertainty_method.startswith("table_relative_cuboid"))


class KnownVolumeCalibrationTests(unittest.TestCase):
    def _station(self, camera: str):
        import threading

        from locallife_cloud.pipeline import VisionPipeline

        class _Store:
            def save_json(self, *_args, **_kwargs) -> None:
                return None

        station = VisionPipeline.__new__(VisionPipeline)
        station.camera_id, station.config, station.lock, station.store = camera, AppConfig(), threading.RLock(), _Store()
        for name in ("_volume_history", "_box_measurement_history", "_box_frames_considered",
                     "_logitech_geometry", "_track_signatures"):
            setattr(station, name, {})
        from locallife_cloud.shape_geometry import GeometryLock
        station._geometry_lock = GeometryLock()
        return station

    def test_factor_is_fitted_against_the_raw_volume_not_the_already_factored_one(self) -> None:
        from types import SimpleNamespace

        station = self._station("realsense")
        station.config.volume_calibration_factor = 1.5          # an older factor
        shown = Detection("cardboard box", 0.9, (0, 0, 10, 10), tracking_status="confirmed",
                          source="yoloe", realsense_volume_l=3.0, volume_raw_geometric_l=2.0)
        station.latest_analysis = SimpleNamespace(detections=[shown])
        record = station.calibrate_known_volume(1.8)
        self.assertAlmostEqual(record["factor"], 0.9)              # 1.8 / 2.0, not 1.5 * 1.8 / 3.0 chained
        self.assertIn("realsense table-relative cuboid", record["applies_to"])

    def test_template_or_held_values_cannot_calibrate(self) -> None:
        from types import SimpleNamespace

        station = self._station("realsense")
        held = Detection("cardboard box", 0.9, (0, 0, 10, 10), tracking_status="predicted",
                         realsense_volume_l=3.0, observation_status="predicted")
        station.latest_analysis = SimpleNamespace(detections=[held])
        with self.assertRaisesRegex(ValueError, "held from an earlier frame"):
            station.calibrate_known_volume(1.8)


if __name__ == "__main__":
    unittest.main()
