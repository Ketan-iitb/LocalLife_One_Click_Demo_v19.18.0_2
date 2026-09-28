"""V37: stable IDs, Logitech heights for flat objects, RealSense dimensions for small ones.

Each failure here was reproduced before it was fixed:

* IDs: the tracker refused to continue a track when the detector renamed the
  object ("handbag" -> "plastic garbage bag"), and because the old track stays
  alive for its grace period the same box alternated ID 1, 2, 1, 2.
* Logitech height: a 30 mm object under a plane fitted 20 mm above the floor
  returned "no_measurable_height_above_plane" -- the dashboard's exact message
  for the slipper.
* RealSense dimensions: a small object whose mask carries a floor halo has
  fewer than the fixed 60 elevated points, and its dimensions came back blank.

Synthetic scenes; not a hardware result.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.logitech_volume import metric_object_volume
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.stable_tracking import LabelTolerantTracker
from locallife_cloud.tracking import ObjectTracker
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import ReferencePlane

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v35_logitech_metric_geometry import (  # noqa: E402
    CAMERA, _box_depth, _floor_plane, _rays, _scene,
)

BOX = (100, 100, 200, 180)


def _ids(tracker, labels, box=BOX):
    ids = []
    for label in labels:
        detection = Detection(label, 0.6, box)
        tracker.update([detection] if label else [])
        ids.append(detection.track_id if label else None)
    return ids


class StableIdTests(unittest.TestCase):
    RELABELLED = ["handbag", "plastic garbage bag", "handbag", "laptop bag",
                  "folded clothing", "textile item"]

    def test_the_old_tracker_churned_ids_on_a_relabel(self) -> None:
        # The bug, kept as evidence: same box, only the name changes.
        self.assertEqual(len(set(_ids(ObjectTracker(), self.RELABELLED))), 2)

    def test_one_object_keeps_one_id_through_relabels(self) -> None:
        self.assertEqual(set(_ids(LabelTolerantTracker(), self.RELABELLED)), {1})

    def test_the_id_survives_a_brief_detector_dropout(self) -> None:
        tracker = LabelTolerantTracker(max_missing_frames=5)
        ids = _ids(tracker, ["handbag", "handbag", None, None, None, "plastic garbage bag"])
        self.assertEqual([item for item in ids if item is not None], [1, 1, 1])

    def test_a_small_movement_keeps_the_id(self) -> None:
        tracker = LabelTolerantTracker()
        first = Detection("slipper", 0.6, (100, 100, 140, 125))
        tracker.update([first])
        moved = Detection("slipper", 0.5, (104, 102, 144, 127))
        tracker.update([moved])
        self.assertEqual(moved.track_id, first.track_id)

    def test_a_genuinely_new_object_elsewhere_gets_a_new_id(self) -> None:
        tracker = LabelTolerantTracker()
        first = Detection("handbag", 0.6, BOX)
        tracker.update([first])
        other = Detection("plastic garbage bag", 0.6, (400, 300, 480, 380))
        tracker.update([first, other])
        self.assertNotEqual(other.track_id, first.track_id)

    def test_a_new_object_after_the_old_one_left_gets_a_new_id(self) -> None:
        tracker = LabelTolerantTracker(max_missing_frames=2)
        ids = _ids(tracker, ["handbag", None, None, None, "cream bottle"])
        self.assertNotEqual(ids[0], ids[-1])

    def test_a_different_object_only_partly_overlapping_is_not_absorbed(self) -> None:
        tracker = LabelTolerantTracker()
        bag = Detection("plastic garbage bag", 0.6, (100, 100, 200, 200))
        tracker.update([bag])
        bottle = Detection("cream bottle", 0.6, (170, 170, 230, 260))
        tracker.update([bag, bottle])
        self.assertNotEqual(bottle.track_id, bag.track_id)

    def test_the_pipeline_uses_the_label_tolerant_tracker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(results_dir=Path(directory), detector_model="local-opencv-background",
                               enable_monocular_depth=False, enable_material_classification=False)
            self.assertIsInstance(VisionPipeline(config).tracker, LabelTolerantTracker)


def _biased(plane, metres: float) -> ReferencePlane:
    a, b, c = plane.coefficients
    norm = math.sqrt(a * a + b * b + 1.0)
    return ReferencePlane(
        tilt_degrees=plane.tilt_degrees, residual_rmse_m=plane.residual_rmse_m,
        inlier_pixels=plane.inlier_pixels, normal=plane.normal,
        coefficients=(a, b, c - metres * norm),
    )


class LogitechHeightTests(unittest.TestCase):
    def setUp(self) -> None:
        origin, direction = _rays()
        self.depth, self.empty, self.mask = _scene(
            _box_depth(origin, direction, (0.0, 0.0), 0.26, 0.10, 0.030))
        self.plane = _floor_plane(self.empty)

    def _measure(self, plane, **kwargs):
        return metric_object_volume(self.depth, CAMERA, self.mask, plane,
                                    min_height_m=0.010, min_pixels=25, **kwargs)

    def test_the_old_path_lost_a_flat_object_under_an_offset_plane(self) -> None:
        result = self._measure(_biased(self.plane, 0.020), local_floor=False)
        self.assertEqual(result.reason, "no_measurable_height_above_plane")

    def test_the_local_floor_recovers_its_height(self) -> None:
        result = self._measure(_biased(self.plane, 0.020))
        self.assertIsNone(result.reason)
        self.assertAlmostEqual(result.diagnostics["height_p90_m"], 0.030, delta=0.005)
        self.assertEqual(result.diagnostics["height_source"], "fitted_support_plane_local_floor")

    def test_a_plane_below_the_floor_is_corrected_downwards(self) -> None:
        result = self._measure(_biased(self.plane, -0.020))
        self.assertIsNone(result.reason)
        self.assertAlmostEqual(result.diagnostics["height_p90_m"], 0.030, delta=0.005)

    def test_a_correct_plane_is_left_alone(self) -> None:
        result = self._measure(self.plane)
        self.assertIsNone(result.diagnostics["local_floor_offset_m"])
        self.assertAlmostEqual(result.diagnostics["height_p90_m"], 0.030, delta=0.005)

    def test_an_empty_mask_is_still_refused(self) -> None:
        result = metric_object_volume(self.depth, CAMERA, np.zeros_like(self.mask), self.plane,
                                      min_height_m=0.010, min_pixels=25)
        self.assertIsNotNone(result.reason)
        self.assertIsNone(result.measurement)


class _FixedDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, detection: Detection) -> None:
        self.detection = detection

    def detect_batch(self, frames):
        return [[Detection(self.detection.label, self.detection.confidence, self.detection.box,
                           mask=self.detection.mask.copy())] for _ in frames]


class RealSenseSmallObjectTests(unittest.TestCase):
    """The field case: geometry-validation mode, a slipper-sized mask 1.8 m away."""

    def _run(self, elevated_rows, elevated_cols):
        size, baseline_m = 160, 1.8
        mask = np.zeros((size, size), dtype=bool)
        mask[60:84, 60:90] = True
        elevated = np.zeros_like(mask)
        elevated[elevated_rows, elevated_cols] = True
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                enable_material_classification=False, roi=(0, 0, 1, 1),
                tracker_confirm_frames=1, operating_mode="geometry_validation",
            )
            pipeline = VisionPipeline(config, detector=_FixedDetector(
                Detection("slipper", 0.9, (60, 60, 90, 84), mask=mask)))
            camera = CameraIntrinsics(fx=600, fy=600, ppx=size / 2, ppy=size / 2,
                                      width=size, height=size)
            empty = np.zeros((size, size, 3), dtype=np.uint8)
            baseline = np.full((size, size), baseline_m, dtype=np.float32)
            pipeline.set_baseline(empty, baseline, camera)
            frame = empty.copy()
            frame[mask] = (0, 0, 255)
            depth = baseline.copy()
            depth[mask] = baseline_m - 0.006          # a soft halo, below 10 mm
            depth[elevated] = baseline_m - 0.03       # the object's top
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=1.0)
        found = [item for item in result.detections if item.confidence > 0]
        self.assertEqual(len(found), 1)
        return found[0]

    def test_a_small_object_gets_its_dimensions(self) -> None:
        found = self._run(slice(66, 72), slice(66, 74))     # 48 points stand up
        self.assertIsNotNone(found.footprint_length_mm)
        self.assertAlmostEqual(found.physical_height_mm, 30.0, delta=6.0)
        self.assertIn("small_object_point_floor", found.dimension_flags)
        self.assertIsNotNone(found.depth_coverage_percent)

    def test_a_mask_with_nothing_above_the_plane_is_still_refused(self) -> None:
        found = self._run(slice(0, 0), slice(0, 0))
        self.assertIsNone(found.footprint_length_mm)
        self.assertIsNotNone(found.depth_coverage_percent)


if __name__ == "__main__":
    unittest.main()
