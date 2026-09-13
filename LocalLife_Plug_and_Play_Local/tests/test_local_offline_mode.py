from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.inference import AdaptiveForegroundSegmenter, create_segmenter
from locallife_cloud.server import create_app
from locallife_cloud.types import CameraIntrinsics


class LocalOfflineDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            detector_model="local-opencv-background",
            enable_monocular_depth=False,
            roi=(0.0, 0.0, 1.0, 1.0),
            automatic_baseline_frames=5,
            min_component_pixels=80,
            min_detection_area_fraction=0.001,
            max_detection_area_fraction=0.80,
            min_detection_side_fraction=0.02,
            foreground_threshold=18,
        )
        self.empty = np.full((240, 320, 3), 120, dtype=np.uint8)

    def _prime(self, detector: AdaptiveForegroundSegmenter) -> None:
        for _ in range(5):
            result = detector.detect_camera_batch({
                "realsense": self.empty.copy(),
                "logitech": self.empty.copy(),
            })
            self.assertEqual(result, [[], []])

    def test_factory_selects_cpu_detector_without_torch(self) -> None:
        detector = create_segmenter(self.config)
        self.assertIsInstance(detector, AdaptiveForegroundSegmenter)
        self.assertEqual(detector.runtime["cloud_required"], False)

    def test_independent_cameras_detect_new_bag_silhouette(self) -> None:
        detector = AdaptiveForegroundSegmenter(self.config)
        self._prime(detector)
        bag = self.empty.copy()
        cv2.ellipse(bag, (160, 135), (58, 72), 0, 0, 360, (0, 105, 235), -1)
        realsense, logitech = detector.detect_camera_batch({
            "realsense": bag,
            "logitech": self.empty.copy(),
        })
        self.assertEqual(len(realsense), 1)
        self.assertEqual(logitech, [])
        self.assertEqual(realsense[0].label, "garbage bag")
        self.assertGreater(np.count_nonzero(realsense[0].mask), 1_000)

    def test_uniform_exposure_change_is_not_a_bag(self) -> None:
        detector = AdaptiveForegroundSegmenter(self.config)
        self._prime(detector)
        brighter = np.clip(self.empty.astype(np.int16) + 22, 0, 255).astype(np.uint8)
        results = detector.detect_camera_batch({
            "realsense": brighter,
            "logitech": brighter.copy(),
        })
        self.assertEqual(results, [[], []])

    def test_local_server_starts_without_torch_or_cloud(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                detector_model="local-opencv-background",
                enable_monocular_depth=False,
                enable_bucket_sync=False,
                results_dir=Path(directory),
                automatic_baseline_frames=5,
            )
            coordinator = DualCameraCoordinator(config)
            warmup = coordinator.warmup()
            self.assertEqual(warmup["runtime"]["cloud_required"], False)
            app = create_app(config, coordinator)
            response = app.test_client().get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "ok")
            state = app.test_client().get("/api/state")
            self.assertEqual(state.status_code, 200)
            self.assertEqual(state.get_json()["build_version"], "19.19.0-local-ai")

    def test_raw_snapshot_endpoint_serves_an_unannotated_jpeg_once_a_frame_arrives(self) -> None:
        # tools/calibrate_dual_camera.py relies on this endpoint to grab
        # synchronized checkerboard frames from both cameras: it must be a
        # plain single JPEG (not the MJPEG overlay stream) and must fail
        # clearly, not crash, before any frame has arrived.
        with TemporaryDirectory() as directory:
            config = AppConfig(
                detector_model="local-opencv-background",
                enable_monocular_depth=False,
                enable_bucket_sync=False,
                results_dir=Path(directory),
                automatic_baseline_frames=5,
            )
            coordinator = DualCameraCoordinator(config)
            coordinator.warmup()
            app = create_app(config, coordinator)
            client = app.test_client()

            before = client.get("/api/cameras/realsense/raw-snapshot.jpg")
            self.assertEqual(before.status_code, 503)

            coordinator.camera("realsense").process_frame(
                self.empty.copy(), intrinsics=CameraIntrinsics(fx=500, fy=500, ppx=160, ppy=120, width=320, height=240),
            )
            after = client.get("/api/cameras/realsense/raw-snapshot.jpg")
            self.assertEqual(after.status_code, 200)
            self.assertEqual(after.mimetype, "image/jpeg")
            decoded = cv2.imdecode(np.frombuffer(after.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], self.empty.shape[:2])

            missing_camera = client.get("/api/cameras/not-a-real-camera/raw-snapshot.jpg")
            self.assertEqual(missing_camera.status_code, 404)

    def test_both_cameras_complete_automatic_rgb_setup_offline(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                detector_model="local-opencv-background",
                enable_monocular_depth=False,
                enable_bucket_sync=False,
                results_dir=Path(directory),
                roi=(0.0, 0.0, 1.0, 1.0),
                automatic_baseline_frames=5,
                restore_saved_baseline=False,
                min_component_pixels=80,
            )
            coordinator = DualCameraCoordinator(config)
            intrinsics = CameraIntrinsics(
                fx=300.0,
                fy=300.0,
                ppx=160.0,
                ppy=120.0,
                width=320,
                height=240,
            )
            depth = np.full((240, 320), 1.5, dtype=np.float32)
            for index in range(5):
                coordinator.process_packets({
                    "realsense": {
                        "frame": self.empty.copy(),
                        "depth_m": depth.copy(),
                        "intrinsics": intrinsics,
                        "source": "realsense",
                        "timestamp": float(index + 1),
                        "persist": False,
                    },
                    "logitech": {
                        "frame": self.empty.copy(),
                        "depth_m": None,
                        "intrinsics": intrinsics,
                        "source": "logitech",
                        "timestamp": float(index + 1),
                        "persist": False,
                    },
                })
            state = coordinator.state()
            self.assertTrue(state["cameras"]["realsense"]["automatic_setup"]["ready"])
            self.assertTrue(state["cameras"]["logitech"]["automatic_setup"]["ready"])
            self.assertEqual(state["cameras"]["logitech"]["volume_status"]["code"], "local_rgb_only")

    def test_realsense_local_path_reports_tracking_colour_and_litres(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                detector_model="local-opencv-background",
                enable_monocular_depth=False,
                enable_bucket_sync=False,
                results_dir=Path(directory),
                roi=(0.0, 0.0, 1.0, 1.0),
                automatic_baseline_frames=5,
                restore_saved_baseline=False,
                min_component_pixels=80,
                tracker_confirm_frames=2,
                volume_stability_frames=3,
                minimum_depth_coverage=0.50,
            )
            coordinator = DualCameraCoordinator(config)
            intrinsics = CameraIntrinsics(
                fx=300.0, fy=300.0, ppx=160.0, ppy=120.0, width=320, height=240,
            )
            empty_depth = np.full((240, 320), 1.5, dtype=np.float32)
            for index in range(5):
                coordinator.process_packets({
                    "realsense": {
                        "frame": self.empty.copy(), "depth_m": empty_depth.copy(),
                        "intrinsics": intrinsics, "source": "realsense",
                        "timestamp": float(index + 1), "persist": False,
                    },
                    "logitech": {
                        "frame": self.empty.copy(), "depth_m": None,
                        "intrinsics": intrinsics, "source": "logitech",
                        "timestamp": float(index + 1), "persist": False,
                    },
                })

            bag = self.empty.copy()
            cv2.ellipse(bag, (160, 135), (58, 72), 0, 0, 360, (0, 105, 235), -1)
            bag_depth = empty_depth.copy()
            bag_mask = np.zeros((240, 320), dtype=np.uint8)
            cv2.ellipse(bag_mask, (160, 135), (58, 72), 0, 0, 360, 255, -1)
            bag_depth[bag_mask > 0] = 1.30
            for index in range(5, 10):
                coordinator.process_packets({
                    "realsense": {
                        "frame": bag.copy(), "depth_m": bag_depth.copy(),
                        "intrinsics": intrinsics, "source": "realsense",
                        "timestamp": float(index + 1), "persist": False,
                    },
                    "logitech": {
                        "frame": bag.copy(), "depth_m": None,
                        "intrinsics": intrinsics, "source": "logitech",
                        "timestamp": float(index + 1), "persist": False,
                    },
                })
            station = coordinator.state()["cameras"]["realsense"]
            detections = station["latest"]["detections"]
            self.assertTrue(detections)
            self.assertEqual(detections[0]["label"], "garbage bag")
            self.assertNotEqual(detections[0]["color"], "unknown")
            self.assertIsNotNone(detections[0]["realsense_volume_l"])
            self.assertGreater(detections[0]["realsense_volume_l"], 1.0)


if __name__ == "__main__":
    unittest.main()
