"""V30 repair: RealSense untouched, Logitech usable without calibration, history sane."""

from __future__ import annotations

import csv
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.event_log import MeasurementEventLog
from locallife_cloud.types import CameraIntrinsics, Detection

PROJECT = Path(__file__).resolve().parents[1]
START_SHA = "8f9d9295465abca36706124c4682b114eb2c3518"


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
        # A metric checkpoint: 1.5 m floor, the object 12 cm closer.
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
    manager = DualCameraCoordinator(AppConfig(**settings), detector=detector, depth_estimator=MetricDepth())
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:90, 60:110] = True
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[mask] = (40, 40, 210)
    depth = np.full((120, 160), 1.5, dtype=np.float32)
    depth[mask] = 1.38
    camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
    detector.items = [Detection("cosmetic bottle", 0.7, (60, 40, 110, 90), mask, color="red")]
    return manager, detector, frame, depth, camera


class LogitechWithoutCalibrationTests(unittest.TestCase):
    def test_an_uncalibrated_logitech_still_reports_numeric_litres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, _, camera = _station(directory)
            logitech = manager.camera("logitech")
            result = None
            for index in range(4):
                result = logitech.process_frame(frame, intrinsics=camera, timestamp=10.0 + index)
            self.assertIsNone(logitech.reference_monocular)          # no baseline captured
            detection = result.detections[0]
            self.assertIsNotNone(detection.track_id)
            self.assertIsNotNone(detection.monocular_volume_l)
            self.assertGreater(detection.monocular_volume_l, 0)
            self.assertEqual(detection.calibration_mode, "uncalibrated-estimate")
            self.assertIsNotNone(detection.physical_height_mm)
            status = logitech.state()["volume_status"]
            self.assertTrue(status["ready"])
            self.assertIn("UNCALIBRATED ESTIMATE", status["message"])

    def test_capturing_the_baseline_switches_to_the_calibrated_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, frame, _, camera = _station(directory, logitech_reference_distance_m=1.5)
            logitech = manager.camera("logitech")
            empty = np.zeros_like(frame)
            detector.items = []
            logitech.process_frame(empty, intrinsics=camera, persist=False)
            logitech.set_baseline()
            self.assertIsNotNone(logitech.calibration)
            self.assertIsNotNone(logitech.reference_monocular)
            detector.items = [Detection("cosmetic bottle", 0.7, (60, 40, 110, 90),
                                        (frame.max(axis=2) > 0), color="red")]
            result = None
            for index in range(4):
                result = logitech.process_frame(frame, intrinsics=camera, timestamp=20.0 + index)
            detection = result.detections[0]
            self.assertNotEqual(detection.calibration_mode, "uncalibrated-estimate")
            self.assertIsNotNone(detection.monocular_volume_l)


class CameraIsolationTests(unittest.TestCase):
    def test_a_logitech_failure_leaves_the_realsense_result_intact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, depth, camera = _station(directory)
            packets = {
                "realsense": {"frame": frame, "depth_m": depth, "intrinsics": camera, "timestamp": time.time()},
                "logitech": {"frame": frame, "intrinsics": camera, "timestamp": time.time()},
            }
            original = type(manager.camera("logitech")).process_precomputed

            def explode(self, *arguments, **keywords):
                if self.camera_id == "logitech":
                    raise RuntimeError("logitech blew up")
                return original(self, *arguments, **keywords)

            with mock.patch.object(type(manager.camera("logitech")), "process_precomputed", explode):
                results = manager.process_packets(packets)
            self.assertIn("realsense", results)
            self.assertNotIn("logitech", results)
            self.assertTrue(results["realsense"].detections)
            self.assertEqual(manager.camera("realsense").tracker.total_count, 1)

    def test_realsense_detects_and_tracks_without_any_logitech_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, depth, camera = _station(directory)
            realsense = manager.camera("realsense")
            realsense.process_frame(np.zeros_like(frame), depth_m=np.full((120, 160), 1.5, np.float32),
                                    intrinsics=camera, persist=False)
            realsense.set_baseline()
            result = None
            for index in range(4):
                result = realsense.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=30.0 + index)
            self.assertTrue(result.detections)
            self.assertIsNotNone(result.detections[0].track_id)
            self.assertIsNotNone(result.detections[0].realsense_volume_l)
            self.assertIsNone(manager.camera("logitech").calibration)

    def test_the_shared_detector_path_is_identical_for_both_cameras(self) -> None:
        from locallife_cloud import inference

        source = Path(inference.__file__).read_text(encoding="utf-8")
        body = source[source.index("def _resize_mask"):source.index("class AdaptiveForegroundSegmenter")]
        self.assertNotIn("from .coordinates import", body)   # Logitech-only, applied in the station


class HistoryDisplayTests(unittest.TestCase):
    def test_an_older_schema_file_is_archived_not_appended_to(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            old_columns = [name for name in MeasurementEventLog.COLUMNS if name != "timestamp_iso"]
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow({"event_id": "old-1", "timestamp": 1_700_000_000, "status": "accepted",
                                 "volume_l": 1.0, "label": "old row"})
            log = MeasurementEventLog(Path(directory))
            self.assertEqual(len(list(Path(directory).glob("measurements.before-*.csv"))), 1)
            log.record({"event_id": "new-1", "timestamp": 1_700_000_100, "volume_l": 2.0,
                        "label": "bottle", "status": "accepted"})
            rows = log.rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "new-1")
            self.assertEqual(rows[0]["label"], "bottle")
            self.assertEqual(float(rows[0]["volume_l"]), 2.0)
            # Every field lands in its own column: no shifted metadata.
            self.assertEqual(rows[0]["processing_mode"], "")
            self.assertTrue(rows[0]["timestamp_iso"].startswith("2023-"))

    def test_the_dashboard_skips_rows_that_are_not_measurements(self) -> None:
        from locallife_cloud.dashboard import DUAL_DASHBOARD

        self.assertIn("Number.isFinite(Number(item.timestamp)))&&item.status)", DUAL_DASHBOARD)
        self.assertIn("Capture Empty Logitech Baseline", DUAL_DASHBOARD)
        self.assertIn("recalibrateLogitech", DUAL_DASHBOARD)


class CsvSchemaTests(unittest.TestCase):
    def test_the_csv_columns_and_writer_are_the_ones_v30_started_with(self) -> None:
        """The repair rotates an outdated file; it does not touch schema or writing."""
        result = subprocess.run(
            ["git", "show", f"{START_SHA}:LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py"],
            capture_output=True, text=True, cwd=PROJECT.parent, timeout=120,
        )
        if result.returncode != 0:
            self.skipTest("git or the V30 starting commit is unavailable here")
        before = result.stdout
        after = (PROJECT / "locallife_cloud" / "event_log.py").read_text(encoding="utf-8")
        for marker, end in (("    COLUMNS = [", "    # British and American spellings"),
                            ("    def _append(", "    @classmethod")):
            with self.subTest(section=marker.strip()):
                self.assertEqual(before[before.index(marker):before.index(end)],
                                 after[after.index(marker):after.index(end)])


class ProtectedFilesTests(unittest.TestCase):
    def test_csv_writer_cloud_and_realsense_geometry_are_unchanged_since_v30_start(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/heightmap_volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(["git", "diff", "--name-only", START_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V30 starting commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
