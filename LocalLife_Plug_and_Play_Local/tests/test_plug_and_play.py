from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline, reject_prompt_conflicts
from locallife_cloud.types import CameraIntrinsics, Detection


class FakeDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections or []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [[
            Detection(
                label=item.label,
                confidence=item.confidence,
                box=item.box,
                mask=None if item.mask is None else item.mask.copy(),
                source=item.source,
                color=item.color,
            )
            for item in self.detections
        ] for _ in frames]


class PlugAndPlayProfileTests(unittest.TestCase):
    def _config(self, directory: str, **changes: object) -> AppConfig:
        values: dict[str, object] = {
            "results_dir": Path(directory),
            "enable_monocular_depth": False,
            "roi": (0, 0, 1, 1),
            "min_component_pixels": 10,
            "restore_saved_baseline": True,
            "saved_baseline_validation_frames": 2,
        }
        values.update(changes)
        return AppConfig(**values)

    def test_saved_realsense_baseline_is_restored_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.full((30, 40, 3), 60, dtype=np.uint8)
            depth = np.full((30, 40), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=15, width=40, height=30)
            first = VisionPipeline(self._config(directory), detector=FakeDetector())
            first.set_baseline(frame, depth, intrinsics)

            restarted = VisionPipeline(self._config(directory), detector=FakeDetector())
            self.assertTrue(restarted.saved_profile_loaded)
            self.assertEqual(restarted.baseline_restore_state, "validating")
            self.assertTrue(np.array_equal(restarted.baseline_realsense, depth))

            restarted.process_frame(frame, depth_m=depth, intrinsics=intrinsics, persist=False)
            self.assertEqual(restarted.baseline_restore_state, "validating")
            restarted.process_frame(frame, depth_m=depth, intrinsics=intrinsics, persist=False)
            self.assertEqual(restarted.baseline_restore_state, "restored-and-validated")
            self.assertTrue(restarted.state()["baseline_ready"])

    def test_moved_camera_rejects_saved_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((30, 40, 3), dtype=np.uint8)
            depth = np.full((30, 40), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=100, fy=100)
            VisionPipeline(self._config(directory), detector=FakeDetector()).set_baseline(
                frame, depth, intrinsics
            )
            restarted = VisionPipeline(
                self._config(
                    directory,
                    saved_baseline_validation_frames=1,
                    saved_baseline_rgb_threshold=10,
                    saved_baseline_max_changed_fraction=0.20,
                ),
                detector=FakeDetector(),
            )
            changed = np.full_like(frame, 100)
            analysis = restarted.process_frame(
                changed, depth_m=depth, intrinsics=intrinsics, persist=False
            )
            self.assertEqual(restarted.baseline_restore_state, "rejected-scene-changed")
            self.assertIsNone(restarted.baseline_rgb)
            self.assertTrue(any("Saved setup was rejected" in warning for warning in analysis.warnings))

    def test_latest_working_reference_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((20, 24, 3), dtype=np.uint8)
            depth = np.full((20, 24), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=90, fy=90)
            first = VisionPipeline(self._config(directory), detector=FakeDetector())
            first.set_baseline(frame, depth, intrinsics)
            first.latest_frame = np.full_like(frame, 30)
            first.latest_depth = np.full_like(depth, 1.8)
            first._advance_reference()

            restarted = VisionPipeline(self._config(directory), detector=FakeDetector())
            self.assertTrue(np.array_equal(restarted.reference_rgb, first.latest_frame))
            self.assertTrue(np.array_equal(restarted.reference_realsense, first.latest_depth))

    def test_unmeasured_detection_is_visible_but_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[8:32, 8:32] = True
            detector = FakeDetector([
                Detection("garbage bag", 0.9, (8, 8, 32, 32), mask=mask, color="black")
            ])
            station = VisionPipeline(
                self._config(directory, restore_saved_baseline=False, record_only_measured_objects=True),
                detector=detector,
            )
            station.process_frame(frame, persist=False)
            analysis = station.process_frame(frame, persist=False)
            self.assertEqual(len(analysis.detections), 1)
            # LiveFix regression: an unmeasured object must still be tracked and
            # counted as "seen" (it has a real track_id and contributes to
            # automatic_count / BAGS SEEN) even though it is correctly withheld
            # from the durable ledger ("observed_count" stays 0). Before the fix,
            # the tracker only ran on measurement-recordable detections, so this
            # object never received a track_id, never showed a color/ID in the
            # live dashboard table, and BAGS SEEN never incremented for it.
            self.assertIsNotNone(analysis.detections[0].track_id)
            self.assertGreaterEqual(analysis.automatic_count, 1)
            self.assertEqual(station.ledger.summary()["observed_count"], 0)
            self.assertEqual(station.state()["session_seen"]["bags"], 1)
            self.assertEqual(station.state()["session_seen"]["colors"][0]["color"], "black")
            self.assertTrue(any("not added to the database" in warning for warning in analysis.warnings))

    def test_box_only_detection_never_integrates_the_entire_camera_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            depth = np.full((40, 40), 1.5, dtype=np.float32)
            reference = np.full((40, 40), 2.0, dtype=np.float32)
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20, width=40, height=40)
            detector = FakeDetector([
                Detection("garbage bag", 0.9, (10, 10, 20, 20), mask=None, color="blue")
            ])
            station = VisionPipeline(
                self._config(directory, restore_saved_baseline=False), detector=detector,
            )
            # Keep RGB baseline absent so this test isolates box-fallback volume
            # rather than fixed-scene fusion.
            station.reference_realsense = reference
            station.baseline_realsense = reference
            result = station.process_frame(
                frame, depth_m=depth, intrinsics=intrinsics, persist=False
            )
            self.assertEqual(len(result.detections), 1)
            # reference-plane (default since round 16; see volume.py's
            # ray-frustum branch comment and config.py's `volume_geometry`
            # comment for why it replaced ray-frustum as the default -- the
            # short version: ray-frustum's own volume sum never actually
            # used the tilt-corrected height, so it wasn't the fix round 12
            # believed it was). No `reference_plane` is fitted in this test
            # (baseline/reference are set directly, bypassing set_baseline),
            # so this is the plain, uncorrected case: pixel_area_m2 =
            # reference_depth^2 / (fx*fy) = 2.0^2 / 100^2 = 0.0004 m^2;
            # contributions_m3 = height * pixel_area_m2 = 0.5 * 0.0004 =
            # 0.0002 m^3 per pixel; 100 pixels * 0.0002 m^3 = 0.02 m^3 = 20 L.
            self.assertAlmostEqual(result.detections[0].realsense_volume_l, 20.0, places=4)

    def test_box_family_detection_gets_table_relative_cuboid_measurement(self) -> None:
        # Round 16 wiring test: a box-labeled detection processed through
        # the full `VisionPipeline.process_frame()` must receive its volume
        # from `estimate_box_volume_cuboid()` (table-relative L*W*H), not
        # the older per-pixel `estimate_volume()` sum used above for
        # "garbage bag". `estimate_box_volume_cuboid()` itself is unit-
        # tested against exact synthetic ground truth in
        # test_box_cuboid_volume.py, and the plain per-pixel path is
        # covered by the test above, but nothing previously confirmed the
        # two are actually wired together inside the pipeline for a real
        # box-family label -- a genuine coverage gap before this test.
        with tempfile.TemporaryDirectory() as directory:
            image_size = 300
            focal_length = 800.0
            ppx = ppy = image_size / 2.0
            baseline_m = 0.6
            true_height_m = 0.05
            row_half, col_half = 12, 20
            center = image_size // 2
            # Zero tilt: a flat table plane parallel to the image plane, so
            # every floor pixel is exactly `baseline_m` away and the box top
            # is exactly `true_height_m` closer to the camera.
            floor_depth = np.full((image_size, image_size), baseline_m, dtype=np.float32)
            top_depth_value = baseline_m - true_height_m
            mask = np.zeros((image_size, image_size), dtype=bool)
            mask[center - row_half: center + row_half, center - col_half: center + col_half] = True
            depth = np.where(mask, top_depth_value, baseline_m).astype(np.float32)
            intrinsics = CameraIntrinsics(
                fx=focal_length, fy=focal_length, ppx=ppx, ppy=ppy,
                width=image_size, height=image_size,
            )
            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            y1, y2 = center - row_half, center + row_half
            x1, x2 = center - col_half, center + col_half
            detector = FakeDetector([
                Detection("cardboard shipping box", 0.92, (x1, y1, x2, y2), mask=mask, color="brown"),
            ])
            station = VisionPipeline(
                self._config(directory, restore_saved_baseline=False), detector=detector,
            )
            # Set the flat floor directly as the reference depth so
            # `process_frame()`'s lazy `fit_reference_plane()` call
            # (pipeline.py ~line 780) fits a real, usable table plane
            # before the box is measured -- mirrors how a real baseline
            # capture works.
            station.reference_realsense = floor_depth
            station.baseline_realsense = floor_depth
            result = station.process_frame(
                frame, depth_m=depth, intrinsics=intrinsics, persist=False
            )
            self.assertEqual(len(result.detections), 1)
            detection = result.detections[0]
            self.assertIsNotNone(detection.box_length_mm)
            self.assertIsNotNone(detection.box_width_mm)
            self.assertIsNotNone(detection.box_height_mm)
            # Height above the fitted table plane should recover the true
            # 50mm box height closely (flat, noiseless synthetic scene).
            self.assertAlmostEqual(detection.box_height_mm, true_height_m * 1000.0, delta=2.0)
            self.assertIsNotNone(detection.realsense_volume_l)
            # No measured box templates ship by default (box_templates.yaml's
            # 3 entries all have measured: false, per the PDF's "never
            # invent measurements" rule), so this detection's own cuboid
            # volume -- not a template's nominal volume -- must be what
            # reaches `realsense_volume_l`.
            self.assertIsNone(detection.box_template_id)
            expected_liters = (
                detection.box_length_mm * detection.box_width_mm * detection.box_height_mm
            ) / 1_000_000.0
            self.assertAlmostEqual(detection.realsense_volume_l, expected_liters, places=4)
            self.assertEqual(detection.measurement_method, "table_relative_cuboid")

    def test_pillow_prompt_vetoes_overlapping_bag_label(self) -> None:
        bag = Detection("fabric bag", 0.55, (5, 5, 35, 35))
        pillow = Detection("pillow", 0.70, (4, 4, 36, 36))
        self.assertEqual(reject_prompt_conflicts([bag, pillow]), [])
        self.assertEqual(reject_prompt_conflicts([bag]), [bag])


if __name__ == "__main__":
    unittest.main()
