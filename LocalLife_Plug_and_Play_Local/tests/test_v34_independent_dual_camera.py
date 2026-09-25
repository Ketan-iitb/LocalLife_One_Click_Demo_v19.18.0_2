"""V34: the Logitech measures on its own, and says exactly why when it cannot.

The screenshots showed detection working while every new object sat at
"pending - reference distance estimate": a cascade mode printed where a reason
belonged. These tests pin the causes that produced it -- a distance saved by
the wizard but never persisted where the restore path reads it, a mask refused
against a baseline that already contained the object, and a live table that
showed the mode rather than the refusal -- and the independence contract.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.types import CameraIntrinsics, Detection

PROJECT = Path(__file__).resolve().parents[1]
V33_SHA = "77baa6e7f70310daac3187b510d03abdcee46373"
SHAPE = (120, 160)
ZONE_CORNERS = ((14.0, 112.0), (146.0, 112.0), (126.0, 24.0), (34.0, 24.0))


class Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames):
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class MetricDepth:
    device = "cpu"

    def estimate_batch(self, frames):
        return [np.where(frame.max(axis=2) > 0, 1.38, 1.5).astype(np.float32) for frame in frames]


def _object(box=(60, 40, 110, 90), label="cosmetic bottle"):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    frame = np.zeros((*SHAPE, 3), dtype=np.uint8)
    frame[mask] = (40, 40, 210)
    return frame, Detection(label, 0.7, box, mask, color="red")


def _manager(directory: str, **overrides):
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


def _run(station, frame, camera, frames=6, start=10.0):
    result = None
    for index in range(frames):
        result = station.process_frame(frame, intrinsics=camera, timestamp=start + index)
    return result


class CalibrationPersistenceTests(unittest.TestCase):
    def test_a_distance_saved_by_the_wizard_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory)
            manager.camera("logitech").save_camera_setup(
                camera_floor_distance_cm=148.0, setup_id="bench-1", shape=SHAPE)
            restarted, _, _ = _manager(directory)
            logitech = restarted.camera("logitech")
            self.assertAlmostEqual(logitech.config.logitech_reference_distance_m, 1.48, places=3)
            self.assertEqual(logitech.state()["measurement_readiness"]["camera_floor_distance"],
                             "ready")
            self.assertEqual(logitech.metric_store.setup.setup_id, "bench-1")

    def test_centimetres_become_metres_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory)
            logitech = manager.camera("logitech")
            logitech.save_camera_setup(camera_floor_distance_cm=150.0, shape=SHAPE)
            self.assertAlmostEqual(logitech.config.logitech_reference_distance_m, 1.5, places=6)
            self.assertAlmostEqual(logitech.metric_store.setup.camera_floor_distance_cm, 150.0)
            self.assertAlmostEqual(logitech.metric_status()["camera_floor_distance_cm"], 150.0)

    def test_readiness_updates_without_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory)
            logitech = manager.camera("logitech")
            self.assertEqual(logitech.state()["measurement_readiness"]["measurement_zone"], "missing")
            logitech.set_measurement_zone(ZONE_CORNERS, SHAPE, near_edge_m=0.9, depth_edge_m=0.6)
            logitech.save_camera_setup(camera_floor_distance_cm=150.0, shape=SHAPE)
            readiness = logitech.state()["measurement_readiness"]
            self.assertEqual(readiness["measurement_zone"], "ready")
            self.assertEqual(readiness["floor_scale"], "ready")
            self.assertEqual(readiness["camera_floor_distance"], "ready")

    def test_the_calibration_belongs_to_the_logical_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory)
            logitech = manager.camera("logitech")
            logitech.save_camera_setup(camera_floor_distance_cm=150.0, shape=SHAPE)
            self.assertEqual(logitech.metric_store.setup.camera, "logitech")
            self.assertTrue(str(logitech.metric_store.directory).endswith("logitech\\\\calibration")
                            or "logitech" in str(logitech.metric_store.directory))
            self.assertIsNone(manager.camera("realsense").metric_store.setup)


class PendingReasonTests(unittest.TestCase):
    def test_a_refusal_names_itself_instead_of_the_cascade_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _manager(directory)
            logitech = manager.camera("logitech")
            # A person at the edge, outside the measurement zone.
            logitech.set_measurement_zone(ZONE_CORNERS, SHAPE, near_edge_m=0.9, depth_edge_m=0.6)
            frame, detection = _object(box=(0, 0, 18, 18))
            detector.items = [detection]
            result = _run(logitech, frame, camera)
            for measured in result.detections:
                self.assertIsNone(measured.monocular_volume_l)
                self.assertNotIn(measured.measurement_quality,
                                 ("reference-distance-estimate", "uncalibrated-estimate"))
                self.assertEqual(measured.measurement_quality, measured.volume_rejection_reason)

    def test_an_object_inside_the_baseline_is_measured_rather_than_pending_for_ever(self) -> None:
        """A committed scene that already contains the object must not hold it hostage."""
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _manager(directory, baseline_contains_object_frames=4)
            logitech = manager.camera("logitech")
            _, detection = _object()
            detection.track_id = 7
            region = np.ones(SHAPE, dtype=bool)
            # The scene has not changed anywhere: the object was standing here
            # when the baseline was taken.
            unchanged = np.zeros(SHAPE, dtype=bool)
            for _ in range(3):
                refused = logitech._deposit_measurement_mask(
                    detection, detection.mask.copy(), region, unchanged, None)
                self.assertFalse(np.any(refused))
                self.assertEqual(detection.volume_rejection_reason,
                                 "no_new_deposit_under_detection")
            measurable = logitech._deposit_measurement_mask(
                detection, detection.mask.copy(), region, unchanged, None)
            self.assertTrue(np.any(measurable))
            self.assertIsNone(detection.volume_rejection_reason)
            self.assertEqual(detection.measurement_quality, "baseline-contains-object")

    def test_the_live_table_prints_the_refusal(self) -> None:
        from locallife_cloud.dashboard import DUAL_DASHBOARD

        self.assertIn("item.volume_rejection_reason||item.measurement_quality", DUAL_DASHBOARD)


class PlausibilityTests(unittest.TestCase):
    def _station(self, directory: str):
        manager, detector, camera = _manager(directory)
        logitech = manager.camera("logitech")
        logitech.set_measurement_zone(ZONE_CORNERS, SHAPE, near_edge_m=0.9, depth_edge_m=0.6)
        logitech.save_camera_setup(camera_floor_distance_cm=150.0, setup_id="bench-1", shape=SHAPE)
        return manager, detector, camera, logitech

    def test_impossible_numbers_are_refused_with_their_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, logitech = self._station(directory)
            _, detection = _object()
            zone = logitech.measurement_zone
            self.assertEqual(logitech._implausible_measurement(detection, 220.0, zone),
                             "height_exceeds_camera_floor_distance")
            self.assertEqual(logitech._implausible_measurement(detection, 0.0, zone),
                             "no_positive_object_height")
            detection.footprint_length_mm = 4000.0
            self.assertEqual(logitech._implausible_measurement(detection, 20.0, zone),
                             "dimension_larger_than_measurement_zone")
            detection.footprint_length_mm = 200.0
            detection.monocular_volume_l = 900.0
            self.assertEqual(logitech._implausible_measurement(detection, 20.0, zone),
                             "volume_exceeds_measurement_zone_capacity")

    def test_a_plausible_measurement_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, logitech = self._station(directory)
            _, detection = _object()
            detection.footprint_length_mm = 220.0
            detection.footprint_width_mm = 150.0
            detection.monocular_volume_l = 3.4
            self.assertEqual(
                logitech._implausible_measurement(detection, 24.0, logitech.measurement_zone), "")


class IndependenceTests(unittest.TestCase):
    def test_the_logitech_measures_with_no_realsense_frame_at_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _manager(directory)
            logitech = manager.camera("logitech")
            frame, detection = _object()
            detector.items = [detection]
            measured = _run(logitech, frame, camera).detections[0]
            realsense = manager.camera("realsense")
            self.assertEqual(realsense.frames_processed, 0)
            self.assertIsNone(realsense.latest_depth)
            self.assertIsNotNone(measured.monocular_volume_l)
            self.assertIsNone(measured.realsense_volume_l)

    def test_the_realsense_scene_cannot_change_a_logitech_number(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            alone, detector_a, camera = _manager(first)
            frame, detection = _object()
            detector_a.items = [detection]
            solo = _run(alone.camera("logitech"), frame, camera).detections[0].monocular_volume_l

            paired, detector_b, _ = _manager(second)
            detector_b.items = [detection]
            # A completely different RealSense scene, measured at the same time.
            other = np.zeros((*SHAPE, 3), dtype=np.uint8)
            other[10:110, 10:150] = (10, 200, 10)
            depth = np.full(SHAPE, 2.0, dtype=np.float32)
            depth[10:110, 10:150] = 1.2
            for index in range(6):
                paired.camera("realsense").process_frame(
                    other, depth_m=depth, intrinsics=camera, timestamp=10.0 + index)
            together = _run(paired.camera("logitech"), frame, camera).detections[0].monocular_volume_l
            self.assertIsNotNone(solo)
            self.assertAlmostEqual(solo, together, places=6)

    def test_each_camera_keeps_its_own_colour_and_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _manager(directory)
            frame, detection = _object()
            detector.items = [detection]
            measured = _run(manager.camera("logitech"), frame, camera).detections[0]
            self.assertEqual(measured.color, "red")
            self.assertIsNotNone(measured.track_id)
            self.assertEqual(manager.camera("realsense").state()["session_seen"]["total"], 0)

    def test_the_logitech_never_reads_a_realsense_array(self) -> None:
        source = (PROJECT / "locallife_cloud" / "logitech_metric.py").read_text(encoding="utf-8")
        source += (PROJECT / "locallife_cloud" / "logitech_volume.py").read_text(encoding="utf-8")
        source += (PROJECT / "locallife_cloud" / "measurement_zone.py").read_text(encoding="utf-8")
        for forbidden in ("realsense_volume", "reference_realsense", "latest_depth",
                          "depth_m=", "peer_depth"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class ProtectedSurfacesTests(unittest.TestCase):
    def test_realsense_csv_cloud_and_launcher_are_unchanged_since_v33(self) -> None:
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
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/shape_geometry.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(["git", "diff", "--name-only", V33_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V33 commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
