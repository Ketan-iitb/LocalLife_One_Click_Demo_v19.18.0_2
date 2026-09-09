from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.dashboard import DUAL_DASHBOARD
from locallife_cloud.geometry import dominant_color, fixed_bin_mask
from locallife_cloud.pipeline import VisionPipeline, filter_waste_detections
from locallife_cloud.tracking import ObjectTracker
from locallife_cloud.types import CameraIntrinsics, Detection


class NoDetections:
    runtime = {"device": "cpu"}

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [[] for _ in frames]


class FixedDetection:
    runtime = {"device": "cpu"}

    def __init__(self, detection: Detection) -> None:
        self.detection = detection

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        item = self.detection
        return [[Detection(
            item.label, item.confidence, item.box,
            None if item.mask is None else item.mask.copy(), color=item.color,
        )] for _ in frames]


class RobustPlugAndPlayTests(unittest.TestCase):
    def test_tracker_keeps_id_when_a_deforming_bag_box_changes_scale(self) -> None:
        tracker = ObjectTracker(confirmation_frames=1, max_missing_frames=10)
        first = Detection("garbage bag", .9, (40, 40, 60, 60), color="blue")
        tracker.update([first])
        second = Detection("full garbage bag", .85, (25, 25, 75, 75), color="blue")
        tracker.update([second])
        self.assertEqual(second.track_id, first.track_id)
        self.assertEqual(tracker.total_count, 1)

    def test_confirmed_track_survives_a_short_detector_dropout(self) -> None:
        tracker = ObjectTracker(confirmation_frames=1, max_missing_frames=10)
        item = Detection("garbage bag", .9, (10, 10, 40, 50), color="orange")
        tracker.update([item])
        tracker.update([])
        predicted = tracker.predicted_detections(3)
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].track_id, item.track_id)
        self.assertEqual(predicted[0].source, "tracked-prediction")

    def test_open_bag_uses_material_edge_not_contents_for_color(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        frame[mask] = (0, 130, 255)  # orange contents
        frame[20:28, 20:80] = (0, 180, 0)
        frame[72:80, 20:80] = (0, 180, 0)
        frame[20:80, 20:28] = (0, 180, 0)
        frame[20:80, 72:80] = (0, 180, 0)
        self.assertEqual(dominant_color(frame, mask), "green")

    def test_tiny_and_low_confidence_scene_objects_are_rejected(self) -> None:
        config = AppConfig(
            min_component_pixels=100, bag_only=True, detector_confidence=.25,
            roi=(0, 0, 1, 1), enable_monocular_depth=False,
        )
        shape = (200, 300, 3)
        region = fixed_bin_mask(shape, config.roi, ())
        tiny = Detection("garbage bag", .9, (4, 4, 18, 12))
        uncertain = Detection("garbage bag", .15, (40, 40, 140, 160))
        real = Detection("full garbage bag", .8, (150, 35, 260, 185))
        self.assertEqual(filter_waste_detections([tiny, uncertain, real], shape, region, config), [real])

    def test_track_enters_history_when_volume_becomes_stable_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            mask = np.zeros((40, 40), dtype=bool)
            mask[10:30, 10:30] = True
            detection = Detection("garbage bag", .9, (10, 10, 30, 30), mask, color="blue")
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                roi=(0, 0, 1, 1), min_component_pixels=10,
                restore_saved_baseline=False, automatic_baseline=False,
                tracker_confirm_frames=1, record_only_measured_objects=True,
                volume_stability_frames=2, auto_deposit=False,
            )
            station = VisionPipeline(config, detector=FixedDetection(detection))
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20)
            station.process_frame(frame, intrinsics=intrinsics, persist=False)
            self.assertEqual(station.ledger.summary()["observed_count"], 0)
            reference = np.full((40, 40), 2.0, dtype=np.float32)
            station.set_baseline(frame, reference, intrinsics)
            depth = reference.copy()
            depth[mask] = 1.5
            station.process_frame(frame, depth_m=depth, intrinsics=intrinsics, persist=False)
            result = station.process_frame(frame, depth_m=depth, intrinsics=intrinsics, persist=False)
            self.assertIsNotNone(result.detections[0].realsense_volume_l)
            self.assertEqual(station.ledger.summary()["observed_count"], 1)

    def test_stable_empty_scene_is_saved_without_a_button(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                roi=(0, 0, 1, 1), min_component_pixels=5,
                restore_saved_baseline=False, automatic_baseline=True,
                automatic_baseline_frames=3,
            )
            station = VisionPipeline(config, detector=NoDetections())
            frame = np.full((30, 40, 3), 40, dtype=np.uint8)
            depth = np.full((30, 40), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=15)
            for index in range(3):
                station.process_frame(
                    frame, depth_m=depth, intrinsics=intrinsics,
                    timestamp=float(index + 1), persist=False,
                )
            self.assertIsNotNone(station.baseline_realsense)
            self.assertTrue(station.state()["automatic_setup"]["ready"])
            self.assertTrue((Path(directory) / "baselines" / "metadata.json").is_file())

    def test_peer_camera_cannot_promote_depth_silhouette_during_detector_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                roi=(0, 0, 1, 1), min_component_pixels=20,
                restore_saved_baseline=False, automatic_baseline=False,
                bag_only=True,
            )
            station = VisionPipeline(config, detector=NoDetections())
            empty = np.zeros((60, 80, 3), dtype=np.uint8)
            baseline = np.full((60, 80), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=120, fy=120, ppx=40, ppy=30)
            station.set_baseline(empty, baseline, intrinsics)
            frame = empty.copy()
            frame[15:50, 25:60] = (20, 120, 20)
            depth = baseline.copy()
            depth[15:50, 25:60] = 1.5
            result = station.process_precomputed(
                frame, detections=[], depth_m=depth, intrinsics=intrinsics,
                peer_bag_present=True, persist=False,
            )
            self.assertEqual(result.detections, [])

    def test_dashboard_exposes_a_manual_baseline_escape_hatch_and_known_reference_calibration(self) -> None:
        # REGRESSION GUARD: earlier local builds required the measurement area
        # to become empty on its own before any volume/material could ever be
        # measured. A permanently-occupied scene (e.g. a fixed demo prop) could
        # then never produce a baseline, and there was no way to force one from
        # the dashboard. "Capture baseline now" is the fix: the user removes the
        # object once, clicks it, and puts the object back. This must stay
        # present.
        #
        # An earlier round of this build deliberately left deeper calibration
        # UI (a known-liters calibration form, a Logitech measured-distance
        # entry box) unexposed even though the backend endpoints
        # (`calibrate_known_volume`, the Logitech reference-distance route)
        # already existed. Real-hardware testing showed that gap mattered: an
        # installation's own residual systematic bias (lens distortion,
        # stereo-calibration imperfections, mask-boundary effects) cannot be
        # fully corrected by theory alone, and the one thing that reliably
        # fixes it is measuring a known-volume object once and letting the
        # dashboard solve for the correction factor. Both controls are now
        # deliberately exposed, wired to the existing endpoints.
        lowered = DUAL_DASHBOARD.lower()
        self.assertIn("capture baseline now", lowered)
        self.assertIn("onclick=", lowered)
        self.assertIn("calibrate-volume", lowered)
        self.assertIn("known volume of the object now in view", lowered)
        self.assertIn("set measured distance", lowered)
        self.assertIn("reference-distance", lowered)


if __name__ == "__main__":
    unittest.main()
