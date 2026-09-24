"""V33: the Logitech in centimetres, calibrated against objects measured with a ruler.

The V32 hardware run showed the remaining fault plainly. One bag read 3.577 L
against the RealSense's 13.437 L, and an object that the RealSense measured
435 x 162 x 389 mm came out 380 x 355 x 117 mm here: a volume that happens to
look close can be three dimensions each wrong in a compensating direction. So
length and width now come from the mat's homography, height from a mapping
fitted on known heights and chosen by held-out error, and the volume from the
per-pixel integral of the two.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.logitech_metric import (
    CALIBRATION_SET,
    EVALUATION_SET,
    INVERSE,
    LINEAR,
    PIECEWISE,
    CameraSetup,
    HeightCalibration,
    HeightSample,
    LogitechMetricStore,
    fit_height_calibration,
    integrate_volume_l,
    robust_height_cm,
    stable_statistics,
    zone_signature,
)
from locallife_cloud.measurement_zone import MeasurementZone
from locallife_cloud.types import CameraIntrinsics, Detection

PROJECT = Path(__file__).resolve().parents[1]
V32_SHA = "be04c8ab5d5ebb8458c6db943768da1ce34e51ff"
SHAPE = (120, 160)
# The mat fills most of the view and is 90 cm across by 60 cm deep.
ZONE_CORNERS = ((14.0, 112.0), (146.0, 112.0), (126.0, 24.0), (34.0, 24.0))


def _sample(name: str, signal_cm: float, height_cm: float, **overrides) -> HeightSample:
    values = dict(name=name, true_length_cm=20.0, true_width_cm=15.0,
                  true_height_cm=height_cm, signal_cm=signal_cm)
    values.update(overrides)
    return HeightSample(**values)


def _mask(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames):
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class MetricDepth:
    """An empty floor at 1.50 m with the object's top 12 cm closer."""

    device = "cpu"

    def estimate_batch(self, frames):
        return [np.where(frame.max(axis=2) > 0, 1.38, 1.5).astype(np.float32) for frame in frames]


class JitteryDepth(MetricDepth):
    def __init__(self) -> None:
        self.frame = 0

    def estimate_batch(self, frames):
        self.frame += 1
        offset = 0.05 if self.frame % 2 else -0.05
        return [np.where(frame.max(axis=2) > 0, 1.33 + offset, 1.5).astype(np.float32)
                for frame in frames]


def _station(directory: str, depth=None, **overrides):
    detector = Detector()
    settings = dict(
        results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
        tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
        operating_mode="geometry_validation", auto_deposit=False,
        logitech_reference_distance_m=0.0, automatic_baseline=False,
    )
    settings.update(overrides)
    manager = DualCameraCoordinator(AppConfig(**settings), detector=detector,
                                    depth_estimator=depth or MetricDepth())
    camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
    return manager, detector, camera


def _object_frame(box=(60, 40, 110, 90)):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    frame = np.zeros((*SHAPE, 3), dtype=np.uint8)
    frame[mask] = (40, 40, 210)
    return frame, Detection("cosmetic bottle", 0.7, box, mask, color="red")


def _calibrated_station(directory: str, depth=None, **overrides):
    """A Logitech with a mat, a floor scale, a baseline and a frozen mapping."""
    manager, detector, camera = _station(directory, depth=depth, **overrides)
    logitech = manager.camera("logitech")
    logitech.set_measurement_zone(ZONE_CORNERS, SHAPE, near_edge_m=0.9, depth_edge_m=0.6)
    logitech.save_camera_setup(camera_floor_distance_cm=150.0, setup_id="bench-1", shape=SHAPE)
    empty = np.zeros((*SHAPE, 3), dtype=np.uint8)
    logitech.process_frame(empty, intrinsics=camera, persist=False)
    logitech.set_baseline()
    return manager, detector, camera, logitech


class HeightMappingTests(unittest.TestCase):
    """The mapping is chosen by held-out error, never by the checkpoint's name."""

    def test_a_linear_signal_recovers_its_scale_and_offset(self) -> None:
        samples = [_sample(f"box{index}", signal, 2.0 * signal + 3.0)
                   for index, signal in enumerate((5.0, 10.0, 18.0, 26.0))]
        calibration, reason = fit_height_calibration(samples, setup_id="bench-1")
        self.assertEqual(reason, "")
        self.assertEqual(calibration.mapping, LINEAR)
        self.assertAlmostEqual(calibration.coefficients[0], 2.0, places=3)
        self.assertAlmostEqual(calibration.coefficients[1], 3.0, places=3)
        self.assertLess(calibration.median_abs_error_cm, 0.01)
        self.assertEqual(calibration.sample_count, 4)
        self.assertEqual(calibration.setup_id, "bench-1")
        self.assertAlmostEqual(calibration.apply(20.0), 43.0, places=3)

    def test_an_inverse_signal_is_not_forced_into_a_straight_line(self) -> None:
        # h = s / (0.2 + 0.01 s): the shape an inverse-depth signal takes.
        samples = [_sample(f"can{index}", signal, signal / (0.2 + 0.01 * signal))
                   for index, signal in enumerate((2.0, 4.0, 8.0, 14.0, 22.0))]
        calibration, _ = fit_height_calibration(samples)
        self.assertEqual(calibration.mapping, INVERSE)
        self.assertLess(calibration.median_abs_error_cm, 0.1)

    def test_a_kinked_response_falls_back_to_the_piecewise_mapping(self) -> None:
        pairs = ((2.0, 3.0), (4.0, 6.0), (6.0, 9.0), (8.0, 30.0), (10.0, 52.0), (12.0, 74.0))
        samples = [_sample(f"step{index}", signal, height)
                   for index, (signal, height) in enumerate(pairs)]
        calibration, _ = fit_height_calibration(samples)
        self.assertEqual(calibration.mapping, PIECEWISE)
        self.assertLess(calibration.median_abs_error_cm, 1.0)

    def test_one_object_gives_a_scale_only_fit_and_says_it_is_provisional(self) -> None:
        calibration, _ = fit_height_calibration([_sample("single", 10.0, 25.0)])
        self.assertEqual(calibration.mapping, LINEAR)
        self.assertAlmostEqual(calibration.apply(10.0), 25.0, places=3)
        self.assertEqual(calibration.coefficients[1], 0.0)
        self.assertIn("in-sample", calibration.selection)
        self.assertEqual(calibration.status, "provisional")

    def test_samples_without_a_measured_signal_are_refused(self) -> None:
        calibration, reason = fit_height_calibration([_sample("ghost", 0.0, 12.0)])
        self.assertIsNone(calibration)
        self.assertEqual(reason, "no_usable_calibration_samples")


class RobustStatisticsTests(unittest.TestCase):
    def test_a_bleeding_edge_pixel_is_not_the_top_of_the_object(self) -> None:
        heights = np.full(400, 18.0)
        heights[:3] = 140.0  # depth bleeding from the floor behind the object
        statistics = robust_height_cm(heights)
        self.assertAlmostEqual(statistics["top_cm"], 18.0, delta=0.5)
        self.assertEqual(statistics["spike_pixels"], 3)

    def test_too_few_valid_pixels_report_nothing_rather_than_a_guess(self) -> None:
        self.assertNotIn("top_cm", robust_height_cm(np.array([12.0, 13.0])))

    def test_volume_is_the_sum_of_per_pixel_prisms(self) -> None:
        heights = np.zeros(SHAPE, dtype=np.float64)
        mask = _mask(40, 60, 40, 60)
        heights[mask] = 10.0
        areas = np.full(SHAPE, 2.0, dtype=np.float64)   # 2 cm^2 per pixel
        litres = integrate_volume_l(heights, areas, mask)
        self.assertAlmostEqual(litres, 400 * 10.0 * 2.0 / 1000.0, places=6)
        self.assertIsNone(integrate_volume_l(heights, None, mask))

    def test_a_settled_track_and_a_jittery_one_are_told_apart(self) -> None:
        settled = stable_statistics([1.20, 1.23, 1.19, 1.22, 1.21], min_frames=5)
        self.assertTrue(settled["stable"])
        self.assertAlmostEqual(settled["median"], 1.21, places=2)
        self.assertLess(settled["coefficient_of_variation"], 0.05)
        jittery = stable_statistics([0.4, 2.9, 0.5, 3.1, 0.6], min_frames=5)
        self.assertFalse(jittery["stable"])
        self.assertIsNotNone(jittery["median"])


class SetupIdentityTests(unittest.TestCase):
    def _setup(self, **overrides) -> CameraSetup:
        values = dict(camera="logitech", width_px=1280, height_px=720,
                      camera_floor_distance_cm=150.0, zone_signature="abc123", setup_id="bench-1")
        values.update(overrides)
        return CameraSetup(**values)

    def test_a_calibration_belongs_to_one_installation(self) -> None:
        saved = self._setup()
        self.assertTrue(self._setup(camera_floor_distance_cm=151.0).matches(saved))
        for changed, expected in (
            (dict(width_px=640, height_px=360), "resolution_changed"),
            (dict(zone_signature="zzz"), "measurement_zone_changed"),
            (dict(setup_id="bench-2"), "camera_setup_id_changed"),
            (dict(camera_floor_distance_cm=162.0), "camera_floor_distance_changed"),
        ):
            with self.subTest(**changed):
                current = self._setup(**changed)
                self.assertFalse(current.matches(saved))
                self.assertEqual(current.difference(saved), expected)

    def test_the_zone_signature_follows_the_mat(self) -> None:
        zone = MeasurementZone(camera="logitech", corners=ZONE_CORNERS, width_px=160, height_px=120,
                               near_edge_m=0.9, depth_edge_m=0.6)
        moved = MeasurementZone(camera="logitech", corners=ZONE_CORNERS, width_px=160, height_px=120,
                                near_edge_m=1.2, depth_edge_m=0.6)
        self.assertNotEqual(zone_signature(zone), zone_signature(moved))
        self.assertEqual(zone_signature(None), "")


class StoreTests(unittest.TestCase):
    def test_calibration_and_evaluation_objects_never_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LogitechMetricStore(Path(directory))
            store.add_sample(_sample("carton", 9.0, 22.0))
            store.add_sample(_sample("bottle", 12.0, 28.0))
            store.add_sample(_sample("unseen backpack", 20.0, 44.0, kind=EVALUATION_SET))
            reopened = LogitechMetricStore(Path(directory))
            self.assertEqual(len(reopened.samples(CALIBRATION_SET)), 2)
            self.assertEqual(len(reopened.samples(EVALUATION_SET)), 1)
            status = reopened.status()
            self.assertEqual(status["samples"], 2)
            self.assertEqual(status["evaluation_samples"], 1)

    def test_a_frozen_calibration_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LogitechMetricStore(Path(directory))
            calibration, _ = fit_height_calibration(
                [_sample("a", 5.0, 10.0), _sample("b", 10.0, 20.0), _sample("c", 15.0, 30.0)],
            )
            store.save_calibration(calibration)
            store.freeze()
            reopened = LogitechMetricStore(Path(directory))
            self.assertTrue(reopened.calibration.frozen)
            self.assertEqual(reopened.calibration.status, "frozen")
            self.assertAlmostEqual(reopened.calibration.apply(12.0), 24.0, places=2)


class PipelineCalibrationTests(unittest.TestCase):
    def test_the_wizard_steps_move_readiness_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, camera = _station(directory)
            logitech = manager.camera("logitech")
            readiness = logitech.state()["measurement_readiness"]
            self.assertEqual(readiness["measurement_zone"], "missing")
            self.assertEqual(readiness["camera_floor_distance"], "missing")
            self.assertEqual(readiness["height_calibration"], "missing")

            logitech.set_measurement_zone(ZONE_CORNERS, SHAPE, near_edge_m=0.9, depth_edge_m=0.6)
            logitech.save_camera_setup(camera_floor_distance_cm=150.0, setup_id="bench-1", shape=SHAPE)
            readiness = logitech.state()["measurement_readiness"]
            self.assertEqual(readiness["measurement_zone"], "ready")
            self.assertEqual(readiness["floor_scale"], "ready")
            self.assertEqual(readiness["camera_floor_distance"], "ready")
            self.assertEqual(readiness["height_calibration"], "missing")
            self.assertEqual(readiness["calibration_samples"], "0/3")
            self.assertEqual(readiness["calibration_status"], "missing")

    def test_a_sample_cannot_be_invented_without_a_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            with self.assertRaises(ValueError):
                manager.camera("logitech").add_height_sample(
                    name="carton", true_length_cm=20, true_width_cm=15, true_height_cm=25,
                )

    def test_fitting_without_samples_states_the_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = manager_result = _station(directory)[0].camera("logitech").fit_height_calibration()
            self.assertFalse(manager_result["ok"])
            self.assertEqual(result["reason"], "no_usable_calibration_samples")

    def test_moving_the_camera_invalidates_the_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _, logitech = _calibrated_station(directory)
            logitech.metric_store.add_sample(_sample("a", 12.0, 24.0, setup_id="bench-1"))
            logitech.metric_store.add_sample(_sample("b", 6.0, 12.0, setup_id="bench-1"))
            logitech.metric_store.add_sample(_sample("c", 18.0, 36.0, setup_id="bench-1"))
            logitech.fit_height_calibration()
            self.assertIsNotNone(logitech.height_calibration)
            # The camera is lifted 30 cm: the mapping no longer describes it.
            logitech.save_camera_setup(camera_floor_distance_cm=180.0, setup_id="bench-1", shape=SHAPE)
            self.assertIsNone(logitech.height_calibration)
            self.assertEqual(logitech.height_calibration_reason,
                             "calibration_invalidated_camera_floor_distance_changed")
            readiness = logitech.state()["measurement_readiness"]
            self.assertEqual(readiness["height_calibration"], "missing")

    def test_an_empty_zone_capture_refuses_an_occupied_mat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _station(directory)
            logitech = manager.camera("logitech")
            frame, detection = _object_frame()
            detector.items = [detection]
            for index in range(3):
                logitech.process_frame(frame, intrinsics=camera, timestamp=10.0 + index)
            refused = logitech.capture_empty_zone()
            self.assertFalse(refused["ok"])
            self.assertEqual(refused["reason"], "object_inside_measurement_zone")

    def test_an_empty_zone_capture_succeeds_on_a_clear_mat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _station(directory)
            logitech = manager.camera("logitech")
            detector.items = []
            logitech.process_frame(np.zeros((*SHAPE, 3), dtype=np.uint8), intrinsics=camera,
                                   persist=False)
            captured = logitech.capture_empty_zone()
            self.assertTrue(captured["ok"], captured)
            self.assertIsNotNone(logitech.reference_monocular)


class CalibratedMeasurementTests(unittest.TestCase):
    def _measure(self, logitech, detector, camera, frames: int = 6):
        frame, detection = _object_frame()
        detector.items = [detection]
        result = None
        for index in range(frames):
            result = logitech.process_frame(frame, intrinsics=camera, timestamp=40.0 + index)
        return result.detections[0]

    def test_length_and_width_come_from_the_mat_not_the_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory)
            measured = self._measure(logitech, detector, camera)
            self.assertEqual(measured.dimension_method.startswith("logitech_"), True)
            self.assertIsNotNone(measured.footprint_length_mm)
            # 50 px of a 132-px-wide mat that spans 90 cm is roughly 30 cm.
            self.assertGreater(measured.footprint_length_mm, 150)
            self.assertLess(measured.footprint_length_mm, 700)
            self.assertIsNotNone(measured.uncalibrated_length_mm)

    def test_a_frozen_mapping_sets_the_reported_height(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory)
            # The scene's signal is 12 cm; these objects say that means 24 cm.
            for name, signal, height in (("a", 6.0, 12.0), ("b", 12.0, 24.0), ("c", 18.0, 36.0)):
                logitech.metric_store.add_sample(_sample(name, signal, height, setup_id="bench-1"))
            logitech.fit_height_calibration()
            logitech.freeze_height_calibration()
            measured = self._measure(logitech, detector, camera)
            self.assertAlmostEqual(measured.height_above_baseline_cm, 24.0, delta=2.0)
            self.assertAlmostEqual(measured.physical_height_mm, 240.0, delta=20.0)
            self.assertTrue(measured.calibration_version.startswith("linear:"))
            self.assertEqual(measured.measurement_quality, "calibrated")
            self.assertEqual(measured.measurement_method, "logitech_calibrated_height_map")
            self.assertGreater(measured.monocular_volume_l, 0)
            # The provisional numbers are kept beside the calibrated ones.
            self.assertIsNotNone(measured.uncalibrated_volume_l)

    def test_volume_is_the_calibrated_per_pixel_integral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory)
            for name, signal, height in (("a", 6.0, 12.0), ("b", 12.0, 24.0), ("c", 18.0, 36.0)):
                logitech.metric_store.add_sample(_sample(name, signal, height, setup_id="bench-1"))
            logitech.fit_height_calibration()
            measured = self._measure(logitech, detector, camera)
            zone = logitech.measurement_zone
            areas = zone.pixel_area_m2(SHAPE) * 10000.0
            mask = logitech.last_metric_context["mask"]
            expected = float(np.sum(areas[mask]) * 24.0 / 1000.0)
            self.assertAlmostEqual(measured.monocular_volume_l, expected, delta=0.35 * expected)

    def test_without_a_calibration_the_estimate_stays_and_is_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory)
            measured = self._measure(logitech, detector, camera)
            self.assertIsNone(measured.calibration_version)
            self.assertIsNotNone(measured.monocular_volume_l)
            self.assertNotEqual(measured.measurement_quality, "pending")

    def test_a_jittery_track_reports_its_median_not_an_empty_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory, depth=JitteryDepth())
            measured = self._measure(logitech, detector, camera, frames=8)
            self.assertIsNotNone(measured.monocular_volume_l)
            self.assertIn(measured.measurement_quality,
                          ("low-confidence-unstable-median", "median-after-stability-timeout",
                           "monocular-calibrated", "calibrated", "provisional-calibration"))

    def test_the_realsense_keeps_its_own_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera, logitech = _calibrated_station(directory)
            realsense = manager.camera("realsense")
            self.assertFalse(realsense.metric_status()["applies"])
            self.assertIsNone(realsense.height_calibration)
            self.assertIsNone(realsense.measurement_zone)
            # Calibrating the Logitech changes nothing on the RealSense side.
            for name, signal, height in (("a", 6.0, 12.0), ("b", 12.0, 24.0), ("c", 18.0, 36.0)):
                logitech.metric_store.add_sample(_sample(name, signal, height, setup_id="bench-1"))
            logitech.fit_height_calibration()
            self.assertIsNone(realsense.height_calibration)
            self.assertEqual(realsense.state()["measurement_readiness"]["height_calibration"], "ready")


class OperatorSurfaceTests(unittest.TestCase):
    def test_the_dashboard_offers_the_wizard_without_an_api_client(self) -> None:
        from locallife_cloud.dashboard import DUAL_DASHBOARD

        for marker in ("Calibration wizard", "pickCorner(event,'logitech')", "Capture Empty Zone",
                       "Add Calibration Sample", "Fit Calibration", "Freeze Calibration",
                       "Reset Calibration", "logitech-calibration-status", "renderCalibration"):
            with self.subTest(marker=marker):
                self.assertIn(marker, DUAL_DASHBOARD)

    def test_every_wizard_step_has_an_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from locallife_cloud.server import create_app

            manager, _, _ = _station(directory)
            app = create_app(manager.config, manager)
            rules = {rule.rule for rule in app.url_map.iter_rules()}
            for route in (
                "/api/cameras/<camera_id>/measurement-zone",
                "/api/cameras/<camera_id>/camera-setup",
                "/api/cameras/<camera_id>/empty-zone",
                "/api/cameras/<camera_id>/height-samples",
                "/api/cameras/<camera_id>/height-calibration/fit",
                "/api/cameras/<camera_id>/height-calibration/freeze",
                "/api/cameras/<camera_id>/height-calibration/reset",
                "/api/cameras/<camera_id>/metric-status",
            ):
                with self.subTest(route=route):
                    self.assertIn(route, rules)


class ProtectedSurfacesTests(unittest.TestCase):
    def test_csv_ledger_detector_and_cloud_are_unchanged_since_v32(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/heightmap_volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/footprint.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/tracking.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/inference.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/vocabulary.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/material.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/sorting_rules.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/stable_identity.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/measurement_mask.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/deposit_state.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(["git", "diff", "--name-only", V32_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V32 commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
