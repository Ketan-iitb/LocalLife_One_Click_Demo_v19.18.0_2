from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline, reject_prompt_conflicts
from locallife_cloud.streaming import LatestFrameProcessor
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import estimate_volume


class BlockingDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, *, block_first: bool = False) -> None:
        self.items: list[Detection] = []
        self.block_first = block_first
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.batch_sizes: list[int] = []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        self.calls += 1
        self.batch_sizes.append(len(frames))
        if self.block_first and self.calls == 1:
            self.started.set()
            self.release.wait(timeout=5)
        return [[
            Detection(
                item.label, item.confidence, item.box,
                None if item.mask is None else item.mask.copy(), color=item.color,
            )
            for item in self.items
        ] for _ in frames]


class MetricDepth:
    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [2.0 - frame[:, :, 0].astype(np.float32) / 255.0 for frame in frames]


class ContinuousStreamTests(unittest.TestCase):
    def test_two_newest_camera_packets_share_one_detector_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = BlockingDetector()
            manager = DualCameraCoordinator(
                AppConfig(
                    results_dir=Path(directory), roi=(0, 0, 1, 1),
                    min_component_pixels=5,
                ),
                detector=detector,
                depth_estimator=MetricDepth(),
            )
            frame = np.zeros((20, 24, 3), dtype=np.uint8)
            depth = np.full((20, 24), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=90, fy=90, ppx=12, ppy=10, width=24, height=20)
            results = manager.process_packets({
                "realsense": {
                    "frame": frame, "depth_m": depth, "intrinsics": intrinsics,
                    "source": "test-rs", "timestamp": 1.0, "persist": False,
                },
                "logitech": {
                    "frame": frame, "depth_m": None, "intrinsics": intrinsics,
                    "source": "test-logi", "timestamp": 1.0, "persist": False,
                },
            })
            self.assertEqual(set(results), {"realsense", "logitech"})
            self.assertEqual(detector.calls, 1)
            self.assertEqual(detector.batch_sizes, [2])

    def test_async_ingest_publishes_newest_preview_and_drops_stale_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = BlockingDetector(block_first=True)
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                roi=(0, 0, 1, 1), min_component_pixels=5,
            )
            manager = DualCameraCoordinator(config, detector=detector)
            processor = LatestFrameProcessor(manager)
            station = manager.camera("realsense")
            depth = np.full((24, 32), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=16, ppy=12, width=32, height=24)
            try:
                first = np.full((24, 32, 3), 20, dtype=np.uint8)
                station.update_preview(first, depth_m=depth, intrinsics=intrinsics, timestamp=1.0)
                processor.submit("realsense", {
                    "frame": first, "depth_m": depth, "intrinsics": intrinsics,
                    "source": "test", "timestamp": 1.0, "persist": False,
                })
                self.assertTrue(detector.started.wait(timeout=2))
                for timestamp, level in ((2.0, 80), (3.0, 180)):
                    frame = np.full((24, 32, 3), level, dtype=np.uint8)
                    station.update_preview(frame, depth_m=depth, intrinsics=intrinsics, timestamp=timestamp)
                    processor.submit("realsense", {
                        "frame": frame, "depth_m": depth, "intrinsics": intrinsics,
                        "source": "test", "timestamp": timestamp, "persist": False,
                    })
                self.assertGreater(float(np.mean(station.latest_frame)), 170)
                transport = processor.snapshot("realsense")
                self.assertEqual(transport["accepted"], 3)
                self.assertGreaterEqual(transport["dropped"], 1)
                detector.release.set()
                deadline = time.time() + 3
                while time.time() < deadline:
                    if processor.snapshot("realsense")["processed"] >= 2:
                        break
                    time.sleep(0.02)
                self.assertEqual(station.state()["stream"]["received"], 3)
                self.assertGreaterEqual(processor.snapshot("realsense")["processed"], 2)
                self.assertGreaterEqual(processor.snapshot("realsense")["dropped"], 1)
            finally:
                detector.release.set()
                processor.stop()

    def test_preview_lock_is_not_held_during_slow_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = BlockingDetector(block_first=True)
            pipeline = VisionPipeline(
                AppConfig(results_dir=Path(directory), enable_monocular_depth=False),
                detector=detector,
            )
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            worker = threading.Thread(
                target=lambda: pipeline.process_frame(frame, timestamp=1.0, persist=False)
            )
            worker.start()
            self.assertTrue(detector.started.wait(timeout=2))
            started = time.perf_counter()
            pipeline.update_preview(np.full_like(frame, 200), timestamp=2.0)
            elapsed = time.perf_counter() - started
            detector.release.set()
            worker.join(timeout=2)
            self.assertLess(elapsed, 0.2)
            self.assertGreater(float(np.mean(pipeline.latest_frame)), 190)


class AutomaticSetupTests(unittest.TestCase):
    def test_one_action_captures_both_profiles_and_allows_provisional_logitech_liters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = BlockingDetector()
            config = AppConfig(
                results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=10,
                tracker_confirm_frames=1, logitech_allow_provisional_metric=True,
                logitech_require_reference=True, logitech_require_overhead=False,
            )
            manager = DualCameraCoordinator(config, detector=detector, depth_estimator=MetricDepth())
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40)
            empty = np.zeros((40, 40, 3), dtype=np.uint8)
            depth = np.full((40, 40), 2.0, dtype=np.float32)
            manager.camera("realsense").update_preview(empty, depth_m=depth, intrinsics=intrinsics)
            manager.camera("logitech").update_preview(empty, intrinsics=intrinsics)
            setup = manager.automatic_empty_setup()
            self.assertTrue(setup["provisional_logitech"])
            webcam = manager.camera("logitech")
            self.assertEqual(webcam.calibration_mode, "model-metric-unverified")
            self.assertTrue(webcam.state()["monocular_calibrated"])

            mask = np.zeros((40, 40), dtype=bool)
            mask[10:30, 10:30] = True
            frame = empty.copy()
            frame[mask, 0] = 51
            detector.items = [Detection("garbage bag", .9, (10, 10, 30, 30), mask, color="black")]
            result = webcam.process_frame(frame, intrinsics=intrinsics, persist=False)
            self.assertIsNotNone(result.monocular_total)
            self.assertGreater(result.monocular_total.liters, 0)
            self.assertGreaterEqual(
                result.monocular_total.uncertainty_l,
                result.monocular_total.liters * 0.35,
            )
            self.assertEqual(webcam.state()["volume_status"]["code"], "measuring_unverified")


class TriangulatedVolumeTests(unittest.TestCase):
    def test_volpy_style_triangles_integrate_known_flat_height_field(self) -> None:
        baseline = np.full((20, 20), 2.0, dtype=np.float32)
        depth = np.full((20, 20), 1.5, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100, ppx=10, ppy=10)
        result = estimate_volume(
            depth, baseline, camera, geometry_mode="triangulated-surface"
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.liters, 80.0, places=5)
        self.assertAlmostEqual(result.projected_area_m2, 0.16, places=6)

    def test_lotion_bottle_conflict_prevents_false_bag_count(self) -> None:
        bag = Detection("garbage bag", .50, (5, 5, 35, 35))
        bottle = Detection("lotion bottle", .70, (4, 4, 36, 36))
        self.assertEqual(reject_prompt_conflicts([bag, bottle]), [])


if __name__ == "__main__":
    unittest.main()
