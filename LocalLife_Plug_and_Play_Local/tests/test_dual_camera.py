"""Independent camera isolation, paired thesis metrics, and depth precision."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator, infer_camera_id, match_deposits
from locallife_cloud.camera_recovery import run_resilient_camera
from locallife_cloud.config import AppConfig
from locallife_cloud.logitech import bound_logitech_detections, object_color, stabilize_background_depth
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import ReferencePlane, estimate_volume, fit_reference_plane


class SharedDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [
            [Detection(item.label, item.confidence, item.box,
                       None if item.mask is None else item.mask.copy(), color=item.color)
             for item in self.items]
            for _ in frames
        ]


class IndependentMetricDepth:
    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [2.0 - frame.max(axis=2).astype(np.float32) * 0.0015 for frame in frames]


class EdgeCameraRecoveryTests(unittest.TestCase):
    def test_camera_is_reopened_after_transient_usb_failures(self) -> None:
        attempts: list[int] = []
        consumed: list[int] = []

        def factory():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise RuntimeError("simulated camera busy")
            return iter(())

        def consume(frames) -> None:
            consumed.append(sum(1 for _ in frames))

        run_resilient_camera(
            "test-camera",
            factory,
            consume,
            retry_seconds=0,
            max_attempts=3,
        )

        self.assertEqual(attempts, [1, 2, 3])
        self.assertEqual(consumed, [0])


class PrecisionVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((20, 20), 2.0, dtype=np.float32)
        self.depth = np.full((20, 20), 1.5, dtype=np.float32)
        self.camera = CameraIntrinsics(fx=100, fy=100, ppx=10, ppy=10)

    def test_surface_columns_preserve_original_calibrated_geometry(self) -> None:
        result = estimate_volume(self.depth, self.baseline, self.camera)
        self.assertAlmostEqual(result.liters, 45.0, places=5)
        self.assertEqual(result.geometry_mode, "surface-columns")
        self.assertAlmostEqual(result.raw_liters, 45.0, places=5)

    def test_ray_frustum_integrates_exact_pixel_frustum(self) -> None:
        result = estimate_volume(self.depth, self.baseline, self.camera, geometry_mode="ray-frustum")
        expected = 400 * (2.0**3 - 1.5**3) / (3 * 100 * 100) * 1000
        self.assertAlmostEqual(result.liters, expected, places=5)
        self.assertGreater(result.liters, 45.0)

    def test_reference_plane_geometry_uses_backplane_pixel_footprint(self) -> None:
        result = estimate_volume(self.depth, self.baseline, self.camera, geometry_mode="reference-plane")
        self.assertAlmostEqual(result.liters, 80.0, places=5)

    def test_known_volume_factor_changes_value_without_hiding_raw_measurement(self) -> None:
        result = estimate_volume(self.depth, self.baseline, self.camera, calibration_factor=1.2)
        self.assertAlmostEqual(result.raw_liters, 45.0, places=5)
        self.assertAlmostEqual(result.liters, 54.0, places=5)
        self.assertEqual(result.calibration_factor, 1.2)

    def test_systematic_uncertainty_does_not_disappear_with_many_pixels(self) -> None:
        result = estimate_volume(self.depth, self.baseline, self.camera,
                                 depth_noise_m=0.004, systematic_error_fraction=0.03)
        self.assertGreaterEqual(result.uncertainty_l, 45.0 * 0.03)
        self.assertAlmostEqual(result.systematic_uncertainty_l, 1.35, places=5)

    def test_high_baseline_noise_suppresses_false_foreground(self) -> None:
        shallow = np.full((20, 20), 1.97, dtype=np.float32)
        noise = np.full((20, 20), 0.025, dtype=np.float32)
        result = estimate_volume(shallow, self.baseline, self.camera,
                                 min_height_m=0.015, baseline_noise_map=noise,
                                 baseline_noise_m=0.025, noise_sigma=3.0)
        self.assertIsNone(result)

    def test_isolated_stereo_spike_is_repaired_without_changing_true_edges(self) -> None:
        corrupted = self.depth.copy()
        corrupted[10, 10] = 1.0
        unfiltered = estimate_volume(corrupted, self.baseline, self.camera, max_height_m=1.2)
        filtered = estimate_volume(corrupted, self.baseline, self.camera,
                                   max_height_m=1.2, reject_outliers=True)
        self.assertEqual(filtered.rejected_pixels, 1)
        self.assertAlmostEqual(filtered.liters, 45.0, places=5)
        self.assertNotAlmostEqual(unfiltered.liters, filtered.liters, places=3)

    def test_planar_empty_bin_reports_zero_mounting_tilt(self) -> None:
        plane = fit_reference_plane(self.baseline, self.camera)
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane.tilt_degrees, 0.0, places=5)
        self.assertLess(plane.residual_rmse_m, 1e-7)

    def test_tilted_empty_bin_reports_nonzero_mounting_tilt(self) -> None:
        rows, columns = np.indices((80, 80))
        camera = CameraIntrinsics(fx=100, fy=100, ppx=40, ppy=40)
        slope = 0.2
        # Plane z = 2 + slope*x, with x = (u - cx) * z / fx.
        depth = (2.0 / (1.0 - slope * (columns - 40) / 100.0)).astype(np.float32)
        plane = fit_reference_plane(depth, camera)
        self.assertAlmostEqual(plane.tilt_degrees, math.degrees(math.atan(slope)), places=2)

    def test_invalid_geometry_and_calibration_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_volume(self.depth, self.baseline, self.camera, geometry_mode="magic")
        with self.assertRaises(ValueError):
            estimate_volume(self.depth, self.baseline, self.camera, calibration_factor=0)


class IndependentCameraTests(unittest.TestCase):
    def make_station(self, directory: str, *, auto_deposit: bool = False) -> tuple[DualCameraCoordinator, SharedDetector]:
        detector = SharedDetector()
        config = AppConfig(
            results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
            tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
            auto_deposit=auto_deposit, logitech_reference_distance_m=2.0,
        )
        return DualCameraCoordinator(config, detector=detector, depth_estimator=IndependentMetricDepth()), detector

    def establish_baselines(self, manager: DualCameraCoordinator) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        empty = np.zeros((40, 40, 3), dtype=np.uint8)
        depth = np.full((40, 40), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40)
        manager.camera("realsense").process_frame(empty, depth_m=depth, intrinsics=camera, persist=False)
        manager.camera("logitech").process_frame(empty, intrinsics=camera, persist=False)
        manager.camera("realsense").set_baseline()
        manager.camera("logitech").set_baseline()
        return empty, depth, camera

    def object_frame(self, empty: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray, Detection]:
        mask = np.zeros(baseline.shape, dtype=bool)
        mask[10:25, 10:25] = True
        frame = empty.copy()
        frame[mask] = (200, 0, 0)
        depth = baseline.copy()
        depth[mask] = 1.7
        return frame, depth, Detection("blue garbage bag", 0.90, (10, 10, 25, 25), mask, color="blue")

    def test_each_camera_has_its_own_storage_detector_state_and_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, depth, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=10)
            self.assertEqual(manager.camera("realsense").ledger.summary()["observed_bags"], 1)
            self.assertEqual(manager.camera("logitech").ledger.summary()["observed_bags"], 0)
            manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=11)
            self.assertEqual(manager.camera("logitech").ledger.summary()["observed_bags"], 1)
            self.assertNotEqual(manager.camera("realsense").config.results_dir,
                                manager.camera("logitech").config.results_dir)
            self.assertTrue((Path(directory) / "realsense" / "waste_plant_ledger.jsonl").is_file())
            self.assertTrue((Path(directory) / "logitech" / "waste_plant_ledger.jsonl").is_file())

    def test_logitech_volume_comes_only_from_its_own_rgb_depth_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, _, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            result = manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=20)
            self.assertIsNone(result.realsense_total)
            self.assertIsNotNone(result.monocular_total)
            self.assertIsNone(result.detections[0].realsense_volume_l)
            self.assertGreater(result.detections[0].monocular_volume_l, 0)
            self.assertEqual(manager.camera("logitech").ledger.summary()["history"][0]["volume_l"],
                             result.detections[0].monocular_volume_l)

    def test_logitech_measured_reference_distance_sets_independent_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            self.establish_baselines(manager)
            logitech = manager.camera("logitech")
            self.assertEqual(logitech.calibration_mode, "independent-measured-distance")
            self.assertAlmostEqual(logitech.calibration.scale, 1.0, places=5)
            self.assertIsNone(logitech.baseline_realsense)

    def test_unconfigured_logitech_metric_depth_cannot_report_uncalibrated_liters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            manager.camera("logitech").config.logitech_reference_distance_m = 0
            empty, baseline, camera = self.establish_baselines(manager)
            frame, _, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            result = manager.camera("logitech").process_frame(frame, intrinsics=camera)
            self.assertEqual(manager.camera("logitech").calibration_mode, "model-metric-unverified")
            self.assertIsNone(result.monocular_total)
            self.assertIsNone(result.bin_total)
            self.assertIsNone(result.detections[0].monocular_volume_l)
            self.assertIsNone(result.detections[0].height_above_baseline_cm)
            self.assertEqual(manager.camera("logitech").state()["volume_status"]["code"],
                             "missing_reference_distance")
            self.assertTrue(any("liters are blocked" in item for item in result.warnings))

    def test_uncalibrated_logitech_cannot_automatically_deposit_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory, auto_deposit=True)
            manager.camera("logitech").config.logitech_reference_distance_m = 0
            empty, baseline, camera = self.establish_baselines(manager)
            frame, _, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            for timestamp in (1.0, 2.0, 3.0):
                manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=timestamp)
            self.assertEqual(manager.camera("logitech").ledger.summary()["observed_bags"], 1)
            self.assertEqual(manager.camera("logitech").ledger.summary()["deposited_count"], 0)

    def test_moderately_steep_logitech_view_still_reports_volume_with_added_uncertainty(self) -> None:
        # Past the confident-mounting zone (35°) but still well short of the
        # hard ceiling (65°): the ray-swept volume math is geometrically
        # valid at any angle, so this must not be blocked outright -- it
        # should report a real number with a graduated uncertainty penalty
        # and a "reduced confidence" warning instead of "pending forever".
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            station = manager.camera("logitech")
            station.reference_plane = ReferencePlane(
                tilt_degrees=42.7, residual_rmse_m=0.005,
                inlier_pixels=1200, normal=(0.68, 0.0, 0.73),
            )
            frame, _, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            result = station.process_frame(frame, intrinsics=camera)
            self.assertIsNotNone(result.monocular_total)
            self.assertNotEqual(station.state()["volume_status"]["code"], "excessive_camera_tilt")
            self.assertTrue(any("42.7°" in warning and "confident-mounting zone" in warning for warning in result.warnings))
            self.assertGreater(station._logitech_tilt_uncertainty_fraction(), 0.0)

    def test_extremely_steep_logitech_view_is_still_hard_blocked(self) -> None:
        # Past the hard ceiling the bin floor is barely visible at all, so a
        # reported liters figure would be fiction -- this must still block.
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            station = manager.camera("logitech")
            station.reference_plane = ReferencePlane(
                tilt_degrees=71.0, residual_rmse_m=0.005,
                inlier_pixels=1200, normal=(0.95, 0.0, 0.32),
            )
            frame, _, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            result = station.process_frame(frame, intrinsics=camera)
            self.assertIsNone(result.monocular_total)
            self.assertIsNone(result.bin_total)
            self.assertIsNone(result.detections[0].monocular_volume_l)
            self.assertEqual(station.state()["volume_status"]["code"], "excessive_camera_tilt")
            self.assertTrue(any("71.0°" in warning for warning in result.warnings))

    def test_tilt_uncertainty_fraction_is_zero_within_confident_zone_and_ramps_to_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            station = manager.camera("logitech")
            low = station.config.logitech_max_tilt_degrees  # 35.0 default
            high = station.config.logitech_hard_max_tilt_degrees  # 65.0 default
            cap = station.config.logitech_max_tilt_uncertainty_fraction  # 0.50 default

            def fraction_at(tilt: float) -> float:
                station.reference_plane = ReferencePlane(
                    tilt_degrees=tilt, residual_rmse_m=0.005, inlier_pixels=1200, normal=(0.0, 0.0, 1.0),
                )
                return station._logitech_tilt_uncertainty_fraction()

            self.assertEqual(fraction_at(low - 5), 0.0)
            self.assertEqual(fraction_at(low), 0.0)
            midpoint = low + (high - low) / 2
            self.assertAlmostEqual(fraction_at(midpoint), cap / 2, places=6)
            self.assertAlmostEqual(fraction_at(high), cap, places=6)
            # Past the hard ceiling the fraction still caps rather than growing
            # unbounded -- the hard block (_logitech_tilt_invalid) is what
            # actually stops reporting beyond this point, not this fraction.
            self.assertAlmostEqual(fraction_at(high + 20), cap, places=6)

    def test_logitech_occupancy_uses_objects_not_unrelated_background_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            station = manager.camera("logitech")
            station.config.logitech_stabilize_depth = False
            frame, _, bag = self.object_frame(empty, baseline)
            frame[26:38, 26:38] = (175, 0, 0)
            detector.items = [bag]
            result = station.process_frame(frame, intrinsics=camera)
            self.assertIsNotNone(result.bin_total)
            self.assertIsNotNone(result.monocular_total)
            # Occupancy comes from the whole-region integral and the object from
            # the calibrated per-object height map; one object in the bin means
            # they must agree closely, not identically.
            self.assertLess(abs(result.bin_total.liters - result.monocular_total.liters),
                            0.25 * result.bin_total.liters)

    def test_preexisting_impossible_logitech_deposits_are_quarantined_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            real = Detection("cardboard box", .9, (0, 0, 10, 10), track_id=1, color="brown")
            real.realsense_volume_l = 7.937
            manager.camera("realsense").ledger.deposit(real, timestamp=1)
            for track_id, liters in enumerate((189.379, 903.276, 889.067), start=1):
                item = Detection("cardboard box", .9, (0, 0, 10, 10), track_id=track_id, color="grey")
                item.monocular_volume_l = liters
                manager.camera("logitech").ledger.deposit(item, timestamp=track_id)
            restarted, _ = self.make_station(directory)
            summary = restarted.camera("logitech").ledger.summary()
            self.assertEqual(summary["deposited_count"], 0)
            self.assertEqual(summary["cumulative_volume_l"], 0.0)
            self.assertEqual(summary["quarantined_count"], 3)
            self.assertEqual(summary["history"], [])
            self.assertEqual(len(restarted.camera("logitech").ledger.all_records(include_quarantined=True)), 3)
            self.assertEqual(restarted.camera("realsense").ledger.summary()["cumulative_volume_l"], 7.937)

    def test_both_cameras_automatically_deposit_without_merging_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory, auto_deposit=True)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, depth, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            for timestamp in (100.0, 101.0):
                manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera,
                                                          timestamp=timestamp)
                manager.camera("logitech").process_frame(frame, intrinsics=camera,
                                                         timestamp=timestamp + 0.25)
            self.assertEqual(manager.camera("realsense").ledger.summary()["deposited_bags"], 1)
            self.assertEqual(manager.camera("logitech").ledger.summary()["deposited_bags"], 1)
            comparison = manager.comparison()
            self.assertEqual(comparison["paired_count"], 1)
            # The two cameras integrate differently on purpose: RealSense uses
            # its stereo height field, Logitech the calibrated monocular height
            # map (logitech_volume.py). On this synthetic scene they agree to
            # about 13 %; the point of the test is that neither copies the other.
            realsense_volume = manager.camera("realsense").ledger.summary()["history"][0]["volume_l"]
            self.assertLess(comparison["mean_absolute_difference_l"], 0.25 * realsense_volume)

    def test_fused_result_combines_both_cameras_current_object_into_one_number(self) -> None:
        # The dashboard previously showed two independent numbers side by
        # side ("20 L from Logitech, 28 L from Intel") with no combined
        # figure. The fixed installation only ever holds one physical object
        # at a time, so both cameras' current confirmed detection can be
        # fused into a single reported volume/color/material without needing
        # pixel-level cross-camera registration.
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, depth, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=1)
            manager.camera("logitech").process_frame(frame, intrinsics=camera, timestamp=1.1)
            fused = manager.fused_result()
            self.assertTrue(fused["available"])
            self.assertEqual(sorted(fused["sources"]), ["logitech", "realsense"])
            self.assertIsNotNone(fused["volume_l"])
            self.assertGreater(fused["volume_l"], 0)
            self.assertEqual(fused["color"], "blue")
            self.assertIn("realsense", fused["per_camera"])
            self.assertIn("logitech", fused["per_camera"])
            self.assertEqual(fused["volume_source"], "realsense")
            self.assertAlmostEqual(
                fused["volume_l"], fused["per_camera"]["realsense"]["volume_l"], places=3,
            )

    def test_fused_result_falls_back_to_whichever_single_camera_has_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, depth, bag = self.object_frame(empty, baseline)
            detector.items = [bag]
            manager.camera("realsense").process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=1)
            fused = manager.fused_result()
            self.assertTrue(fused["available"])
            self.assertEqual(fused["sources"], ["realsense"])
            self.assertIn("Only realsense", fused["message"])

    def test_fused_result_is_unavailable_before_any_confirmed_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            self.establish_baselines(manager)
            fused = manager.fused_result()
            self.assertFalse(fused["available"])
            self.assertEqual(fused["sources"], [])

    def test_fused_result_never_includes_an_unconfirmed_phantom_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            self.establish_baselines(manager)
            station = manager.camera("realsense")
            phantom = Detection(
                "garbage bag (depth silhouette)", 0.0, (0, 0, 10, 10),
                source="fixed-bin-depth-silhouette", tracking_status="confirmed",
            )
            phantom.realsense_volume_l = 110.0
            station.latest_analysis = station.process_frame(
                np.zeros((40, 40, 3), dtype=np.uint8),
                depth_m=np.full((40, 40), 2.0, dtype=np.float32),
                intrinsics=CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40),
            )
            station.latest_analysis.detections = [phantom]
            fused = manager.fused_result()
            self.assertFalse(fused["available"])

    def test_sparse_realsense_depth_does_not_enter_accurate_deposit_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector = self.make_station(directory, auto_deposit=True)
            empty, baseline, camera = self.establish_baselines(manager)
            frame, depth, bag = self.object_frame(empty, baseline)
            depth[10:18, 10:25] = 0.0
            detector.items = [bag]
            station = manager.camera("realsense")
            for timestamp in (1.0, 2.0, 3.0):
                result = station.process_frame(frame, depth_m=depth, intrinsics=camera,
                                               timestamp=timestamp)
            self.assertIsNotNone(result.realsense_total)
            self.assertLess(result.realsense_total.coverage_ratio, 0.70)
            self.assertEqual(station.state()["volume_status"]["code"], "insufficient_depth_coverage")
            self.assertEqual(station.ledger.summary()["deposited_count"], 0)
            with self.assertRaisesRegex(ValueError, "depth coverage"):
                station.commit_current_bags()

    def test_known_volume_calibration_persists_and_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            result = manager.camera("realsense").calibrate_known_volume(known_liters=12, observed_liters=10)
            self.assertAlmostEqual(result["factor"], 1.2)
            restarted, _ = self.make_station(directory)
            self.assertAlmostEqual(restarted.camera("realsense").config.volume_calibration_factor, 1.2)
            self.assertAlmostEqual(restarted.camera("logitech").config.volume_calibration_factor, 1.0)

    def _single_camera_pipeline(
        self, directory: str, *, allow_unclassified: bool = False, bag_only: bool = True
    ) -> VisionPipeline:
        detector = SharedDetector()
        config = AppConfig(
            results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=25,
            tracker_confirm_frames=1, enable_monocular_depth=False, bag_only=bag_only,
            allow_unclassified_foreground=allow_unclassified,
        )
        pipeline = VisionPipeline(config, detector=detector)
        empty = np.zeros((70, 70, 3), dtype=np.uint8)
        baseline_depth = np.full((70, 70), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100)
        pipeline.set_baseline(empty, baseline_depth, camera)
        return pipeline, detector, empty, baseline_depth, camera

    def test_calibration_default_observed_liters_matches_the_single_confirmed_detections_own_reading(self) -> None:
        # The dashboard's "Calibrate from this object" button never sends
        # `observed_liters` explicitly -- it relies entirely on this default.
        # Before the fix, the default was `realsense_total`/`monocular_total`
        # (a separately-computed camera-wide combined-mask total); it must
        # instead be exactly the single confirmed detection's own displayed
        # `realsense_volume_l`, since that is the number the user is looking
        # at on screen when they place the known object and press Calibrate.
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline_depth, camera = self._single_camera_pipeline(directory)
            mask = np.zeros((70, 70), dtype=bool)
            mask[5:20, 5:20] = True
            detector.items = [Detection("garbage bag", 0.9, (5, 5, 20, 20), mask, color="black")]
            frame = empty.copy()
            frame[mask] = (0, 0, 255)
            depth = baseline_depth.copy()
            depth[mask] = 1.7
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            observed = result.detections[0].realsense_volume_l
            self.assertIsNotNone(observed)

            calibration = pipeline.calibrate_known_volume(known_liters=observed * 1.1)
            self.assertAlmostEqual(calibration["observed_liters"], observed, places=6)
            self.assertAlmostEqual(calibration["factor"], 1.1, places=4)

    def test_calibration_ignores_a_phantom_detection_sharing_the_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline_depth, camera = self._single_camera_pipeline(
                directory, allow_unclassified=True
            )
            mask_real = np.zeros((70, 70), dtype=bool)
            mask_real[5:20, 5:20] = True
            mask_phantom = np.zeros((70, 70), dtype=bool)
            mask_phantom[45:60, 45:60] = True
            detector.items = [Detection("garbage bag", 0.9, (5, 5, 20, 20), mask_real, color="black")]
            frame = empty.copy()
            frame[mask_real] = (0, 0, 255)
            frame[mask_phantom] = (0, 255, 0)
            depth = baseline_depth.copy()
            depth[mask_real] = 1.7
            depth[mask_phantom] = 1.8
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            confirmed = [item for item in result.detections if item.confidence > 0]
            phantom = [item for item in result.detections if item.confidence == 0]
            self.assertEqual(len(confirmed), 1)
            self.assertEqual(len(phantom), 1)

            calibration = pipeline.calibrate_known_volume(
                known_liters=confirmed[0].realsense_volume_l * 1.2
            )
            self.assertAlmostEqual(
                calibration["observed_liters"], confirmed[0].realsense_volume_l, places=6
            )

    def test_calibration_rejects_when_only_a_phantom_is_in_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline_depth, camera = self._single_camera_pipeline(
                directory, allow_unclassified=True
            )
            mask = np.zeros((70, 70), dtype=bool)
            mask[5:20, 5:20] = True
            frame = empty.copy()
            frame[mask] = (0, 0, 255)
            depth = baseline_depth.copy()
            depth[mask] = 1.7
            pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            with self.assertRaisesRegex(ValueError, "No confirmed, measured object"):
                pipeline.calibrate_known_volume(known_liters=1.0)

    def test_calibration_rejects_when_multiple_confirmed_objects_share_the_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline_depth, camera = self._single_camera_pipeline(
                directory, bag_only=False
            )
            mask_a = np.zeros((70, 70), dtype=bool)
            mask_a[5:20, 5:20] = True
            mask_b = np.zeros((70, 70), dtype=bool)
            mask_b[45:60, 45:60] = True
            detector.items = [
                Detection("garbage bag", 0.9, (5, 5, 20, 20), mask_a, color="black"),
                Detection("cardboard box", 0.9, (45, 45, 60, 60), mask_b, color="brown"),
            ]
            frame = empty.copy()
            frame[mask_a] = (0, 0, 255)
            frame[mask_b] = (0, 255, 0)
            depth = baseline_depth.copy()
            depth[mask_a] = 1.7
            depth[mask_b] = 1.8
            pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            with self.assertRaisesRegex(ValueError, "More than one measured object"):
                pipeline.calibrate_known_volume(known_liters=5.0)

    def test_baseline_noise_is_measured_from_multiple_real_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            station = manager.camera("realsense")
            frame = np.zeros((30, 30, 3), dtype=np.uint8)
            camera = CameraIntrinsics(fx=100, fy=100)
            for distance in (1.996, 2.0, 2.004, 1.997, 2.003):
                station.process_frame(frame, depth_m=np.full((30, 30), distance, dtype=np.float32),
                                      intrinsics=camera, persist=False)
            result = station.set_baseline()
            self.assertEqual(result["baseline_frame_count"], 5)
            self.assertGreater(result["baseline_noise_m"], 0)
            self.assertIsNotNone(result["reference_plane"])

    def test_combined_state_preserves_single_camera_fields_without_mixing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _ = self.make_station(directory)
            self.establish_baselines(manager)
            state = manager.state()
            self.assertEqual(state["mode"], "independent-dual-camera-comparison")
            self.assertEqual(set(state["cameras"]), {"realsense", "logitech"})
            self.assertEqual(state["camera_id"], "realsense")
            self.assertEqual(state["cameras"]["logitech"]["plant"]["camera_id"], "logitech")
            json.dumps(state, allow_nan=False)


class ComparisonStatisticsTests(unittest.TestCase):
    def record(self, entry_id: str, kind: str, color: str, liters: float, timestamp: float) -> dict[str, object]:
        return {"entry_id": entry_id, "object_type": kind, "color": color,
                "volume_l": liters, "observed_at": timestamp, "deposited_at": timestamp,
                "status": "deposited"}

    def test_pairs_are_type_aware_unique_and_time_limited(self) -> None:
        hardware = [self.record("r1", "bag", "blue", 10, 100), self.record("r2", "box", "brown", 8, 110)]
        webcam = [self.record("l1", "bag", "blue", 11, 101), self.record("l2", "bag", "brown", 9, 110),
                  self.record("l3", "box", "brown", 7, 111)]
        pairs = match_deposits(hardware, webcam, tolerance_seconds=3)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["logitech_entry_id"], "l1")
        self.assertEqual(pairs[1]["logitech_entry_id"], "l3")
        self.assertEqual(pairs[0]["difference_l"], 1.0)

    def test_outside_time_window_is_never_fabricated_as_a_match(self) -> None:
        pairs = match_deposits([self.record("r", "bag", "blue", 10, 100)],
                               [self.record("l", "bag", "blue", 10, 120)], tolerance_seconds=5)
        self.assertEqual(pairs, [])

    def test_reference_trials_produce_truth_based_mae_rmse_and_bias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = SharedDetector()
            manager = DualCameraCoordinator(AppConfig(results_dir=Path(directory)),
                                            detector=detector, depth_estimator=IndependentMetricDepth())
            hardware = Detection("garbage bag", .9, (0, 0, 1, 1), color="blue", track_id=1)
            hardware.realsense_volume_l = 11
            webcam = Detection("garbage bag", .9, (0, 0, 1, 1), color="blue", track_id=1)
            webcam.monocular_volume_l = 13
            manager.camera("realsense").ledger.deposit(hardware, timestamp=100)
            manager.camera("logitech").ledger.deposit(webcam, timestamp=101)
            manager.record_reference(10)
            metrics = manager.comparison()
            self.assertEqual(metrics["reference_trials"], 1)
            self.assertEqual(metrics["realsense_accuracy"]["mae_l"], 1)
            self.assertEqual(metrics["logitech_accuracy"]["rmse_l"], 3)
            self.assertEqual(metrics["logitech_accuracy"]["mape_percent"], 30)
            with self.assertRaisesRegex(ValueError, "already been recorded"):
                manager.record_reference(10)

    def test_camera_source_routing_is_explicit_and_rejects_unknown_ids(self) -> None:
        self.assertEqual(infer_camera_id("realsense-aligned-rgb-depth"), "realsense")
        self.assertEqual(infer_camera_id("logitech-video:/dev/video4"), "logitech")
        self.assertEqual(infer_camera_id("video:0"), "logitech")
        self.assertEqual(infer_camera_id("anything", "logitech"), "logitech")
        with self.assertRaises(ValueError):
            infer_camera_id("anything", "unknown")


class LogitechIntrinsicsTests(unittest.TestCase):
    @staticmethod
    def intrinsics_function():
        try:
            import requests  # noqa: F401
        except ImportError:
            import pip._vendor.requests as bundled_requests

            sys.modules.setdefault("requests", bundled_requests)
        from locallife_cloud.edge_client import camera_intrinsics_from_fov

        return camera_intrinsics_from_fov

    def test_logitech_focal_length_is_derived_from_horizontal_field_of_view(self) -> None:
        camera = self.intrinsics_function()(640, 480, horizontal_fov_deg=70.42)
        self.assertAlmostEqual(camera["fx"], 453.461, places=2)
        self.assertEqual(camera["fy"], camera["fx"])
        self.assertEqual(camera["ppx"], 319.5)

    def test_checkerboard_calibrated_focal_lengths_override_estimated_values(self) -> None:
        camera = self.intrinsics_function()(640, 480, fx=920.0, fy=918.0)
        self.assertEqual(camera["fx"], 920.0)
        self.assertEqual(camera["fy"], 918.0)

    def test_invalid_lens_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.intrinsics_function()(640, 480, horizontal_fov_deg=180)
        with self.assertRaises(ValueError):
            self.intrinsics_function()(640, 480, fx=-2)


class LogitechTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((80, 100, 3), 125, dtype=np.uint8)
        self.region = np.ones((80, 100), dtype=bool)
        self.object_mask = np.zeros((80, 100), dtype=bool)
        self.object_mask[25:55, 40:70] = True
        self.frame = self.baseline.copy()
        self.frame[self.object_mask] = (40, 85, 145)
        self.detection = Detection("cardboard box", 0.90, (40, 25, 70, 55), self.object_mask.copy())

    def test_background_depth_drift_cannot_expand_detection_across_the_room(self) -> None:
        bounded, warnings = bound_logitech_detections(
            self.frame, self.baseline, [self.detection], self.region, min_pixels=20
        )
        self.assertFalse(warnings)
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0].area_pixels, 900)
        self.assertLess(bounded[0].area_pixels / self.region.size, 0.15)

    def test_brown_cardboard_is_not_labeled_grey_by_surrounding_wall(self) -> None:
        bounded, _ = bound_logitech_detections(
            self.frame, self.baseline, [self.detection], self.region, min_pixels=20
        )
        self.assertEqual(bounded[0].color, "brown")

    def test_loose_whole_frame_detector_is_trimmed_to_actual_changed_object(self) -> None:
        loose = Detection("cardboard box", 0.90, (0, 0, 100, 80), np.ones((80, 100), dtype=bool))
        bounded, warnings = bound_logitech_detections(
            self.frame, self.baseline, [loose], self.region, min_pixels=20
        )
        self.assertFalse(warnings)
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0].area_pixels, 900)
        self.assertEqual(bounded[0].color, "brown")

    def test_whole_frame_detection_without_changed_object_is_rejected(self) -> None:
        loose = Detection("garbage bag", 0.90, (0, 0, 100, 80), np.ones((80, 100), dtype=bool))
        bounded, warnings = bound_logitech_detections(
            self.baseline, self.baseline, [loose], self.region, min_pixels=20
        )
        self.assertEqual(bounded, [])
        self.assertTrue(any("covering most of the scene" in warning for warning in warnings))

    def test_nested_box_detections_count_as_one_physical_object(self) -> None:
        nested_mask = np.zeros((80, 100), dtype=bool)
        nested_mask[32:48, 47:63] = True
        nested = Detection("cardboard shipping box", 0.86, (47, 32, 63, 48), nested_mask)
        bounded, warnings = bound_logitech_detections(
            self.frame, self.baseline, [self.detection, nested], self.region, min_pixels=20
        )
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0].color, "brown")
        self.assertTrue(any("same physical object" in item for item in warnings))

    def test_sparse_wall_and_floor_box_is_rejected_by_its_scene_footprint(self) -> None:
        loose_mask = np.zeros((80, 100), dtype=bool)
        loose_mask[0, :] = True
        loose_mask[-1, :] = True
        loose_mask[:, 0] = True
        loose_mask[:, -1] = True
        loose = Detection("cardboard box", 0.95, (0, 0, 100, 80), loose_mask)
        bounded, warnings = bound_logitech_detections(
            self.frame, self.baseline, [loose], self.region, min_pixels=20
        )
        self.assertEqual(bounded, [])
        self.assertTrue(any("wall or floor" in item for item in warnings))

    def test_room_wide_exposure_change_is_not_measured_as_object_foreground(self) -> None:
        brighter = np.clip(self.baseline.astype(np.int16) + 22, 0, 255).astype(np.uint8)
        brighter[self.object_mask] = (45, 95, 155)
        bounded, warnings = bound_logitech_detections(
            brighter, self.baseline, [self.detection], self.region, min_pixels=20
        )
        self.assertFalse(warnings)
        self.assertEqual(len(bounded), 1)
        self.assertLessEqual(bounded[0].area_pixels, 900 * 2)
        self.assertEqual(bounded[0].color, "brown")

    def test_depth_drift_is_corrected_without_using_realsense_pixels(self) -> None:
        reference = np.full((80, 100), 2.0, dtype=np.float32)
        drifting = np.full((80, 100), 1.8, dtype=np.float32)
        drifting[self.object_mask] = 1.5
        fixed, diagnostics = stabilize_background_depth(
            drifting, reference, self.frame, self.baseline, [self.detection], self.region
        )
        self.assertTrue(diagnostics["applied"])
        self.assertGreater(diagnostics["anchor_pixels"], 5000)
        self.assertAlmostEqual(float(np.median(fixed[~self.object_mask])), 2.0, places=4)
        self.assertAlmostEqual(float(np.median(fixed[self.object_mask])), 1.7, places=3)

    def test_depth_is_not_adjusted_when_no_static_background_exists(self) -> None:
        reference = np.full((80, 100), 2.0, dtype=np.float32)
        drifting = np.full((80, 100), 1.8, dtype=np.float32)
        full = Detection("garbage bag", .9, (0, 0, 100, 80), np.ones((80, 100), dtype=bool))
        fixed, diagnostics = stabilize_background_depth(
            drifting, reference, self.frame, self.baseline, [full], self.region
        )
        self.assertFalse(diagnostics["applied"])
        self.assertTrue(np.array_equal(fixed, drifting))

    def test_black_bag_stays_black_and_blue_bag_stays_blue(self) -> None:
        black = self.baseline.copy()
        black[self.object_mask] = (18, 20, 22)
        self.assertEqual(object_color(black, self.object_mask, baseline=self.baseline,
                                      label="black garbage bag"), "black")
        blue = self.baseline.copy()
        blue[self.object_mask] = (210, 70, 35)
        self.assertEqual(object_color(blue, self.object_mask, baseline=self.baseline,
                                      label="garbage bag"), "blue")

    def test_dark_camera_tinted_bag_remains_black(self) -> None:
        tinted = self.baseline.copy()
        # Representative screenshot sample: RGB [68, 83, 42], converted to BGR.
        tinted[self.object_mask] = (42, 83, 68)
        self.assertEqual(object_color(tinted, self.object_mask, baseline=self.baseline,
                                      label="full garbage bag"), "black")

    def test_medium_grey_object_is_not_forced_to_black(self) -> None:
        grey = self.baseline.copy()
        grey[self.object_mask] = (120, 122, 121)
        self.assertEqual(object_color(grey, self.object_mask, label="garbage bag"), "grey")

    def test_orange_bag_is_reported_as_orange(self) -> None:
        orange = self.baseline.copy()
        orange[self.object_mask] = (20, 125, 240)
        self.assertEqual(
            object_color(orange, self.object_mask, baseline=self.baseline, label="garbage bag"),
            "orange",
        )

    def test_bright_brown_cardboard_remains_brown_instead_of_orange(self) -> None:
        frame = self.baseline.copy()
        frame[self.object_mask] = (48, 105, 165)
        self.assertEqual(object_color(frame, self.object_mask, baseline=self.baseline,
                                      label="cardboard shipping box"), "brown")

    def test_implausible_logitech_liters_never_enter_deposited_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = SharedDetector()
            configuration = AppConfig(
                results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
                tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
                auto_deposit=True, logitech_reference_distance_m=2.0,
                logitech_max_item_volume_l=15.0,
            )
            manager = DualCameraCoordinator(configuration, detector=detector,
                                            depth_estimator=IndependentMetricDepth())
            camera = CameraIntrinsics(fx=20, fy=20)
            station = manager.camera("logitech")
            empty = np.zeros((40, 40, 3), dtype=np.uint8)
            station.process_frame(empty, intrinsics=camera, persist=False)
            station.set_baseline()
            mask = np.zeros((40, 40), dtype=bool)
            mask[10:25, 10:25] = True
            frame = empty.copy()
            frame[mask] = (200, 0, 0)
            detector.items = [Detection("garbage bag", .9, (10, 10, 25, 25), mask)]
            first = station.process_frame(frame, intrinsics=camera)
            station.process_frame(frame, intrinsics=camera)
            self.assertIsNone(first.detections[0].monocular_volume_l)
            self.assertTrue(any("implausible" in warning for warning in first.warnings))
            self.assertEqual(station.ledger.summary()["deposited_count"], 0)
            self.assertIsNone(station.ledger.summary()["history"][0]["volume_l"])

    def test_logitech_does_not_display_height_above_the_physical_maximum(self) -> None:
        class ImplausibleObjectDepth:
            def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
                return [2.0 - frame.max(axis=2).astype(np.float32) * 0.006 for frame in frames]

        with tempfile.TemporaryDirectory() as directory:
            detector = SharedDetector()
            manager = DualCameraCoordinator(
                AppConfig(results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
                          tracker_confirm_frames=1, logitech_reference_distance_m=2.0),
                detector=detector, depth_estimator=ImplausibleObjectDepth(),
            )
            station = manager.camera("logitech")
            camera = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40)
            empty = np.zeros((40, 40, 3), dtype=np.uint8)
            station.process_frame(empty, intrinsics=camera, persist=False)
            station.set_baseline()
            mask = np.zeros((40, 40), dtype=bool)
            mask[10:25, 10:25] = True
            frame = empty.copy()
            frame[mask] = (200, 0, 0)
            detector.items = [Detection("cardboard box", .9, (10, 10, 25, 25), mask)]
            result = station.process_frame(frame, intrinsics=camera)
            self.assertIsNone(result.detections[0].height_above_baseline_cm)
            self.assertIsNone(result.detections[0].monocular_volume_l)
            self.assertTrue(any("height beyond the physical limit" in item for item in result.warnings))

    def test_clearing_logitech_history_keeps_realsense_history_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = SharedDetector()
            manager = DualCameraCoordinator(AppConfig(results_dir=Path(directory)), detector=detector,
                                            depth_estimator=IndependentMetricDepth())
            real = Detection("cardboard box", .9, (0, 0, 10, 10), track_id=1, color="brown")
            real.realsense_volume_l = 7.9
            logi = Detection("cardboard box", .9, (0, 0, 10, 10), track_id=1, color="grey")
            logi.monocular_volume_l = 903.0
            manager.camera("realsense").ledger.deposit(real, timestamp=10)
            manager.camera("logitech").ledger.deposit(logi, timestamp=10)
            reset = manager.camera("logitech").clear_history()
            self.assertEqual(reset["removed_observations"], 1)
            self.assertTrue(reset["backup_created"])
            self.assertEqual(manager.camera("logitech").ledger.summary()["deposited_count"], 0)
            self.assertEqual(manager.camera("realsense").ledger.summary()["cumulative_volume_l"], 7.9)

    def test_logitech_roi_is_independent_persistent_and_requires_new_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = SharedDetector()
            original = AppConfig(results_dir=Path(directory), roi=(0, 0, 1, 1))
            manager = DualCameraCoordinator(original, detector=detector,
                                            depth_estimator=IndependentMetricDepth())
            result = manager.camera("logitech").update_camera_roi((0.25, 0.20, 0.60, 0.70))
            self.assertTrue(result["recapture_baseline"])
            self.assertEqual(manager.camera("realsense").config.roi, (0, 0, 1, 1))
            restarted = DualCameraCoordinator(original, detector=detector,
                                              depth_estimator=IndependentMetricDepth())
            self.assertEqual(restarted.camera("logitech").config.roi, (0.25, 0.20, 0.60, 0.70))
            self.assertEqual(restarted.camera("realsense").config.roi, (0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
