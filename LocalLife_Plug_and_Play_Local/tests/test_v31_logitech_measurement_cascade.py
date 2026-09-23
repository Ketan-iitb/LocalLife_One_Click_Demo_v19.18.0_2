"""V31: a tracked Logitech object always ends with numbers or a stated reason."""

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
V30_SHA = "65ca272a8829e484ff8052d80246f0ce9edf6184"


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

    def __init__(self, invalid: bool = False) -> None:
        self.invalid = invalid

    def estimate_batch(self, frames):
        if self.invalid:
            return [np.full(frame.shape[:2], np.nan, dtype=np.float32) for frame in frames]
        return [np.where(frame.max(axis=2) > 0, 1.38, 1.5).astype(np.float32) for frame in frames]


class JitteryDepth(MetricDepth):
    """A volume that never settles inside the stability tolerance."""

    def __init__(self) -> None:
        super().__init__()
        self.frame = 0

    def estimate_batch(self, frames):
        self.frame += 1
        offset = 0.04 if self.frame % 2 else -0.04
        return [np.where(frame.max(axis=2) > 0, 1.30 + offset, 1.5).astype(np.float32) for frame in frames]


def _scene():
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:90, 60:110] = True
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[mask] = (40, 40, 210)
    camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
    detection = Detection("cosmetic bottle", 0.7, (60, 40, 110, 90), mask, color="red")
    return frame, camera, detection


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
    frame, camera, detection = _scene()
    detector.items = [detection]
    return manager, detector, frame, camera


def _run(station, frame, camera, frames=6, intrinsics=True, start=10.0):
    result = None
    for index in range(frames):
        result = station.process_frame(frame, intrinsics=camera if intrinsics else None,
                                       timestamp=start + index)
    return result


class MeasurementCascadeTests(unittest.TestCase):
    def test_mode_3_measures_without_baseline_or_reference_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory)
            detection = _run(manager.camera("logitech"), frame, camera).detections[0]
            self.assertEqual(detection.calibration_mode, "uncalibrated-estimate")
            for value in (detection.monocular_volume_l, detection.footprint_length_mm,
                          detection.footprint_width_mm, detection.physical_height_mm,
                          detection.height_above_baseline_cm):
                self.assertIsNotNone(value)
                self.assertGreater(value, 0)
                self.assertTrue(np.isfinite(value))

    def test_mode_2_uses_the_measured_camera_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory, logitech_reference_distance_m=1.5)
            logitech = manager.camera("logitech")
            detection = _run(logitech, frame, camera).detections[0]
            self.assertEqual(detection.calibration_mode, "reference-distance-estimate")
            self.assertGreater(detection.monocular_volume_l, 0)
            status = logitech.state()["volume_status"]
            self.assertIn("REFERENCE-DISTANCE ESTIMATE", status["message"])
            self.assertNotIn("pending", status["message"].lower())

    def test_mode_1_is_selected_once_a_baseline_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, frame, camera = _station(directory, logitech_reference_distance_m=1.5)
            logitech = manager.camera("logitech")
            detector.items = []
            logitech.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False)
            logitech.set_baseline()
            detector.items = [_scene()[2]]
            detection = _run(logitech, frame, camera, start=40.0).detections[0]
            self.assertIsNotNone(logitech.calibration)
            self.assertNotIn(detection.calibration_mode,
                             ("uncalibrated-estimate", "reference-distance-estimate"))
            self.assertGreater(detection.monocular_volume_l, 0)

    def test_a_camera_that_sends_no_intrinsics_is_still_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory)
            logitech = manager.camera("logitech")
            detection = _run(logitech, frame, camera, intrinsics=False).detections[0]
            self.assertIsNone(logitech.latest_intrinsics)
            self.assertIsNotNone(detection.monocular_volume_l)
            self.assertGreater(detection.monocular_volume_l, 0)

    def test_invalid_depth_gives_a_named_reason_not_a_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory, depth=MetricDepth(invalid=True))
            detection = _run(manager.camera("logitech"), frame, camera).detections[0]
            self.assertIsNone(detection.monocular_volume_l)
            self.assertIsNotNone(detection.track_id)          # detection and tracking survive
            reason = manager.camera("logitech")._pending_measurement_reason(
                detection, depth_m=None, intrinsics=camera)
            self.assertNotEqual(reason, "pending-empty-baseline")

    def test_a_jittery_volume_finalises_on_the_median_after_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(
                directory, depth=JitteryDepth(), logitech_measurement_timeout_frames=6,
                record_only_measured_objects=True, volume_stability_frames=3,
            )
            logitech = manager.camera("logitech")
            seen = []
            for index in range(12):
                detection = logitech.process_frame(
                    frame, intrinsics=camera, timestamp=10.0 + index * 0.2).detections[0]
                seen.append((detection.track_id, detection.monocular_volume_l,
                             detection.measurement_quality))
            self.assertIsNone(seen[1][1])                     # still inside the window
            self.assertIsNotNone(seen[-1][1])                 # finalised on the median
            self.assertGreater(seen[-1][1], 0)
            self.assertEqual(seen[-1][2], "median-after-stability-timeout")
            self.assertEqual(len({item[0] for item in seen}), 1)   # one object, one id

    def test_detection_classification_and_colour_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory)
            detection = _run(manager.camera("logitech"), frame, camera).detections[0]
            self.assertEqual(detection.label, "cosmetic bottle")
            self.assertEqual(detection.color, "red")
            self.assertEqual(detection.canonical_type, "cosmetic bottle")

    def test_realsense_is_unaffected_by_the_logitech_cascade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory)
            realsense = manager.camera("realsense")
            floor = np.full((120, 160), 1.5, dtype=np.float32)
            depth = floor.copy()
            depth[frame.max(axis=2) > 0] = 1.38
            realsense.process_frame(np.zeros_like(frame), depth_m=floor, intrinsics=camera, persist=False)
            realsense.set_baseline()
            result = _run(realsense, frame, camera, start=60.0)
            self.assertTrue(result.detections)
            self.assertIsNone(result.detections[0].calibration_mode
                              if result.detections[0].calibration_mode in
                              ("uncalibrated-estimate", "reference-distance-estimate") else None)
            self.assertIsNotNone(result.detections[0].track_id)


class RestoredCalibrationWithoutBaselineTests(unittest.TestCase):
    """The reported hardware state: a calibration object exists, nothing else does."""

    def _restored(self, directory: str, **overrides):
        from locallife_cloud.types import DepthCalibration

        manager, _, frame, camera = _station(directory, **overrides)
        logitech = manager.camera("logitech")
        # What a restored profile leaves behind: a calibration, and no empty
        # scene reference, no support plane.
        logitech.calibration = DepthCalibration(
            scale=1.0, offset_m=0.0, rmse_m=0.0, sample_pixels=1000,
            method="model-metric-unverified",
        )
        logitech.calibration_mode = "model-metric-unverified"
        logitech.reference_monocular = None
        logitech.reference_plane = None
        return manager, logitech, frame, camera

    def test_a_restored_calibration_without_a_baseline_still_measures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, logitech, frame, camera = self._restored(directory)
            detection = _run(logitech, frame, camera).detections[0]
            self.assertIsNotNone(logitech.calibration)          # the state that used to block
            self.assertIsNone(logitech.reference_monocular)
            for value in (detection.monocular_volume_l, detection.footprint_length_mm,
                          detection.footprint_width_mm, detection.physical_height_mm,
                          detection.height_above_baseline_cm):
                self.assertIsNotNone(value)
                self.assertGreater(value, 0)
                self.assertTrue(np.isfinite(value))
            self.assertIn(detection.calibration_mode,
                          ("reference-distance-estimate", "uncalibrated-estimate"))
            self.assertIn(detection.measurement_quality,
                          ("reference-distance-estimate", "uncalibrated-estimate",
                           "median-after-stability-timeout"))
            reason = logitech._pending_measurement_reason(detection, depth_m=None, intrinsics=camera)
            self.assertNotIn("pending", reason)
            self.assertNotIn("empty-baseline", reason)
            status = logitech.state()["volume_status"]
            self.assertNotIn("pending", (status["message"] or "").lower())

    def test_a_restored_calibration_with_a_measured_distance_uses_mode_2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, logitech, frame, camera = self._restored(
                directory, logitech_reference_distance_m=1.5)
            detection = _run(logitech, frame, camera).detections[0]
            self.assertEqual(detection.calibration_mode, "reference-distance-estimate")
            self.assertGreater(detection.monocular_volume_l, 0)

    def test_a_calibration_for_another_resolution_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, logitech, frame, camera = self._restored(directory)
            logitech.calibration_rejected_reason = "resolution_changed_recalibrate_empty_scene"
            detection = _run(logitech, frame, camera).detections[0]
            self.assertIsNotNone(detection.monocular_volume_l)
            self.assertIn(detection.calibration_mode,
                          ("reference-distance-estimate", "uncalibrated-estimate"))

    def test_the_provisional_result_survives_tracking_and_serialisation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, logitech, frame, camera = self._restored(directory)
            _run(logitech, frame, camera, frames=8)
            state = logitech.state()
            payload = state["latest"]["detections"][0]
            self.assertIsNotNone(payload["monocular_volume_l"])
            self.assertGreater(payload["monocular_volume_l"], 0)
            self.assertIsNotNone(payload["dimensions_mm"])
            self.assertGreater(payload["dimensions_mm"]["height"], 0)
            self.assertIsNotNone(payload["height_above_baseline_cm"])
            self.assertEqual(payload["track_id"], 1)
            self.assertEqual(payload["label"], "cosmetic bottle")
            self.assertEqual(payload["color"], "red")


class ProtectedFilesTests(unittest.TestCase):
    def test_realsense_geometry_csv_and_cloud_are_unchanged_since_v30(self) -> None:
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
        result = subprocess.run(["git", "diff", "--name-only", V30_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V30 commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
