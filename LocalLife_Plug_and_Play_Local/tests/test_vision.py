from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.geometry import (
    connected_components,
    detect_foreground_objects,
    detect_scene_objects,
    dominant_color,
    fuse_scene_detections,
    fixed_bin_mask,
    intersection_over_union,
    roi_pixels,
)
from locallife_cloud.pipeline import (
    VisionPipeline, accepted_object_class, is_bag_detection,
    is_supported_waste_detection, summarize_depth_signal,
)
from locallife_cloud.inference import YoloSegmenter
from locallife_cloud.tracking import ObjectTracker
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import calibrate_monocular_depth, estimate_volume


class FakeDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections or []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        result = []
        for _ in frames:
            copied = [
                Detection(
                    label=item.label,
                    confidence=item.confidence,
                    box=item.box,
                    mask=None if item.mask is None else item.mask.copy(),
                    source=item.source,
                )
                for item in self.detections
            ]
            result.append(copied)
        return result


class FakeDepthEstimator:
    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [2.0 - frame[:, :, 0].astype(np.float32) / 255.0 for frame in frames]


class FakeTensor:
    def __init__(self, values: list[object] | np.ndarray) -> None:
        self.values = np.asarray(values)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class GeometryTests(unittest.TestCase):
    def test_normalized_roi(self) -> None:
        self.assertEqual(roi_pixels((100, 200, 3), (0.1, 0.2, 0.5, 0.4)), (20, 20, 120, 60))

    def test_iou(self) -> None:
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(intersection_over_union((0, 0, 5, 5), (6, 6, 9, 9)), 0)

    def test_multiple_components_are_retained(self) -> None:
        binary = np.zeros((30, 30), dtype=np.uint8)
        binary[2:8, 2:8] = 1
        binary[16:25, 16:25] = 1
        self.assertEqual(len(connected_components(binary, min_area=20)), 2)

    def test_foreground_detects_two_separate_objects(self) -> None:
        baseline = np.zeros((60, 60, 3), dtype=np.uint8)
        frame = baseline.copy()
        frame[5:17, 5:17] = (0, 0, 255)
        frame[30:46, 30:46] = (0, 255, 0)
        detections = detect_foreground_objects(frame, baseline, (0, 0, 1, 1), min_area=50)
        self.assertEqual(len(detections), 2)
        self.assertEqual({item.color for item in detections}, {"red", "green"})

    def test_color_classification(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:] = (255, 0, 0)
        self.assertEqual(dominant_color(image, mask), "blue")


class ContainerSegmentationTests(unittest.TestCase):
    def test_depth_recovers_container_when_rgb_only_reveals_shipping_label(self) -> None:
        baseline = np.full((80, 100, 3), 60, dtype=np.uint8)
        frame = baseline.copy()
        frame[24:34, 30:43] = 230
        baseline_depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth = baseline_depth.copy()
        depth[15:65, 12:87] = 1.70

        scene = detect_scene_objects(
            frame, baseline, depth, baseline_depth, (0, 0, 1, 1), min_area=30
        )

        self.assertEqual(len(scene), 1)
        self.assertEqual(scene[0].box, (12, 15, 87, 65))
        self.assertGreater(scene[0].area_pixels, 3000)

    def test_paper_label_expands_to_whole_cardboard_box(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[15:65, 12:87] = (30, 70, 110)
        whole_mask = np.zeros(frame.shape[:2], dtype=bool)
        whole_mask[15:65, 12:87] = True
        label_mask = np.zeros(frame.shape[:2], dtype=bool)
        label_mask[24:34, 30:43] = True
        component = Detection("storage container", 1.0, (12, 15, 87, 65), mask=whole_mask, color="brown")
        shipping_label = Detection("paper", 0.25, (30, 24, 43, 34), mask=label_mask)

        fused = fuse_scene_detections(frame, [shipping_label], [component])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "cardboard box")
        self.assertEqual(fused[0].box, (12, 15, 87, 65))
        self.assertEqual(fused[0].area_pixels, 3750)

    def test_box_prediction_outranks_nested_higher_confidence_label(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        component_mask = np.zeros(frame.shape[:2], dtype=bool)
        component_mask[5:50, 5:50] = True
        box_mask = np.zeros(frame.shape[:2], dtype=bool)
        box_mask[10:45, 10:45] = True
        label_mask = np.zeros(frame.shape[:2], dtype=bool)
        label_mask[15:22, 15:25] = True
        component = Detection("storage container", 1.0, (5, 5, 50, 50), mask=component_mask)
        detections = [
            Detection("paper", 0.97, (15, 15, 25, 22), mask=label_mask),
            Detection("cardboard shipping box", 0.30, (10, 10, 45, 45), mask=box_mask),
        ]

        fused = fuse_scene_detections(frame, detections, [component])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "cardboard shipping box")
        self.assertEqual(fused[0].box, (5, 5, 50, 50))

    def test_sprawling_scene_object_does_not_inflate_a_tight_correct_neural_box(self) -> None:
        # Reproduces the real-hardware "big rectangle instead of a tight
        # box, wrong volume" report: a real, tightly-boxed neural detection
        # (the actual backpack) whose underlying "changed vs baseline"
        # scene_object was noise-bridged to a second, unrelated changed
        # region far away (background clutter -- a patterned blanket, a
        # doorway edge) by the gap-merging morphological close. The two
        # regions are only connected by a thin 1px corridor, so the whole
        # thing is sprawling (mostly-empty bounding box), not one compact
        # object -- the fused box must stay near the real detection, not
        # balloon out to cover the unrelated region too.
        shape = (100, 220, 3)
        frame = np.zeros(shape, dtype=np.uint8)

        candidate_mask = np.zeros(shape[:2], dtype=bool)
        candidate_mask[10:30, 10:30] = True    # blob A: the real backpack, 20x20=400px
        candidate_mask[60:80, 180:200] = True  # blob B: unrelated background clutter, 20x20=400px
        candidate_mask[20, 30:180] = True      # a 1px-wide bridge connecting them
        candidate_mask[20:60, 179] = True      # ... down to blob B's row
        candidate = Detection(
            "storage container", 1.0,
            box=(10, 10, 200, 80), mask=candidate_mask, source="depth-scene-segmentation",
        )

        neural_mask = np.zeros(shape[:2], dtype=bool)
        neural_mask[10:30, 10:30] = True  # tight, correct box on blob A only
        neural_detection = Detection("garbage bag", 0.85, box=(10, 10, 30, 30), mask=neural_mask)

        fused = fuse_scene_detections(frame, [neural_detection], [candidate])

        self.assertEqual(len(fused), 1)
        # Must stay near the real detection (blob A, x in [10,30)) and not
        # reach anywhere close to the unrelated blob B at x in [180,200).
        self.assertLess(fused[0].box[2], 100)
        self.assertEqual(fused[0].label, "garbage bag")

    def test_compact_scene_object_still_fully_recovers_a_small_label(self) -> None:
        # Companion to the sprawling test above: a genuinely compact
        # scene_object (high fill ratio) must keep the existing, intentional
        # "recover the whole container from a small label" behavior.
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        whole_mask = np.zeros(frame.shape[:2], dtype=bool)
        whole_mask[15:65, 12:87] = True
        label_mask = np.zeros(frame.shape[:2], dtype=bool)
        label_mask[24:34, 30:43] = True
        component = Detection("storage container", 1.0, (12, 15, 87, 65), mask=whole_mask, color="brown")
        shipping_label = Detection("paper", 0.25, (30, 24, 43, 34), mask=label_mask)

        fused = fuse_scene_detections(frame, [shipping_label], [component])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].box, (12, 15, 87, 65))
        self.assertEqual(fused[0].area_pixels, 3750)

    def test_two_separate_containers_remain_separate(self) -> None:
        baseline = np.zeros((70, 90, 3), dtype=np.uint8)
        frame = baseline.copy()
        frame[5:25, 5:30] = 120
        frame[35:60, 45:80] = 200
        baseline_depth = np.full((70, 90), 2.0, dtype=np.float32)
        depth = baseline_depth.copy()
        depth[5:25, 5:30] = 1.8
        depth[35:60, 45:80] = 1.6

        scene = detect_scene_objects(
            frame, baseline, depth, baseline_depth, (0, 0, 1, 1), min_area=30
        )
        detections = [
            Detection("plastic bag", 0.7, (7, 7, 15, 16)),
            Detection("cardboard box", 0.8, (50, 40, 60, 50)),
        ]
        fused = fuse_scene_detections(frame, detections, scene)

        self.assertEqual(len(fused), 2)
        self.assertEqual({item.label for item in fused}, {"plastic bag", "cardboard box"})

    def test_default_prompts_target_bags_and_boxes_only(self) -> None:
        config = AppConfig()
        prompts = config.prompts
        self.assertIn("garbage bag", prompts)
        self.assertIn("garbage sack", prompts)
        self.assertIn("black trash bag", prompts)
        self.assertTrue(all(is_supported_waste_detection(prompt) for prompt in prompts))
        self.assertIn("cardboard shipping box", prompts)
        self.assertNotIn("person", prompts)
        self.assertIn("backpack", config.negative_prompts)
        self.assertIn("shoe", config.negative_prompts)

    def test_local_environment_defaults_skip_expensive_monocular_depth(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(AppConfig.from_env().enable_monocular_depth)

    def test_only_the_three_configured_waste_families_are_accepted(self) -> None:
        self.assertEqual(accepted_object_class("filled polythene bag"), "plastic_bag")
        self.assertEqual(accepted_object_class("kraft paper bag"), "paper_bag")
        self.assertEqual(accepted_object_class("cardboard shipping box"), "cardboard_box")
        for label in ("bag", "backpack", "laptop bag", "shoe", "pillow", "bottle"):
            self.assertIsNone(accepted_object_class(label), label)

    def test_unclassified_foreground_is_ignored_by_default(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[8:50, 10:48] = True
        component = Detection("storage container", 1.0, (10, 8, 48, 50), mask=mask)

        fused = fuse_scene_detections(frame, [], [component])

        self.assertEqual(fused, [])

    def test_opt_in_foreground_is_honestly_marked_unclassified(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[8:50, 10:48] = True
        component = Detection("storage container", 1.0, (10, 8, 48, 50), mask=mask)

        fused = fuse_scene_detections(frame, [], [component], allow_unclassified=True)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "unclassified object")

    def test_bag_labels_are_identified_without_generic_object_classes(self) -> None:
        self.assertTrue(is_bag_detection("black garbage bag"))
        self.assertTrue(is_bag_detection("refuse_sack"))
        self.assertFalse(is_bag_detection("cardboard box"))
        self.assertFalse(is_bag_detection("storage container"))


class VolumeTests(unittest.TestCase):
    def test_known_pinhole_volume(self) -> None:
        baseline = np.full((100, 100), 2.0, dtype=np.float32)
        depth = baseline.copy()
        depth[40:60, 40:60] = 1.5
        mask = np.zeros(depth.shape, dtype=bool)
        mask[40:60, 40:60] = True
        measurement = estimate_volume(depth, baseline, CameraIntrinsics(fx=100, fy=100), object_mask=mask)
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement.liters, 45.0, places=5)
        self.assertEqual(measurement.valid_pixels, 400)

    def test_invalid_depth_is_ignored(self) -> None:
        baseline = np.full((20, 20), 2.0, dtype=np.float32)
        depth = np.full((20, 20), 1.8, dtype=np.float32)
        depth[:10] = np.nan
        result = estimate_volume(depth, baseline, CameraIntrinsics(fx=100, fy=100))
        self.assertEqual(result.valid_pixels, 200)

    def test_volume_requires_baseline_and_intrinsics(self) -> None:
        depth = np.ones((20, 20), dtype=np.float32)
        self.assertIsNone(estimate_volume(depth, None, CameraIntrinsics(fx=100, fy=100)))
        self.assertIsNone(estimate_volume(depth, depth, None))

    def test_calibration_recovers_scale_and_offset(self) -> None:
        prediction = np.linspace(0.5, 2.5, 400, dtype=np.float32).reshape(20, 20)
        reference = prediction * 1.4 + 0.25
        calibration = calibrate_monocular_depth(prediction, reference, minimum_samples=20)
        self.assertIsNotNone(calibration)
        self.assertAlmostEqual(calibration.scale, 1.4, places=4)
        self.assertAlmostEqual(calibration.offset_m, 0.25, places=4)

    def test_invalid_intrinsics_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics(fx=0, fy=100)

    def test_flat_empty_scene_calibrates_with_positive_scale(self) -> None:
        prediction = np.linspace(1.8, 2.2, 400, dtype=np.float32).reshape(20, 20)
        reference = np.full((20, 20), 2.0, dtype=np.float32)
        calibration = calibrate_monocular_depth(prediction, reference, minimum_samples=20)
        self.assertIsNotNone(calibration)
        self.assertGreater(calibration.scale, 0)


class DepthSignalTests(unittest.TestCase):
    def test_missing_depth_signal_is_explicit(self) -> None:
        signal = summarize_depth_signal(None)
        self.assertFalse(signal["available"])
        self.assertEqual(signal["valid_pixels"], 0)
        self.assertIsNone(signal["median_m"])

    def test_depth_signal_ignores_zero_nan_and_out_of_range_values(self) -> None:
        depth = np.asarray([[0.0, 1.0, np.nan], [2.0, 50.0, 0.05]], dtype=np.float32)
        signal = summarize_depth_signal(depth)
        self.assertTrue(signal["available"])
        self.assertEqual(signal["valid_pixels"], 2)
        self.assertEqual(signal["valid_percent"], 33.33)
        self.assertEqual(signal["median_m"], 1.5)

    def test_empty_pipeline_reports_waiting_for_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            pipeline = VisionPipeline(config, detector=FakeDetector())
            self.assertEqual(pipeline.state()["volume_status"]["code"], "waiting_for_camera")

    def test_rgb_only_stream_reports_no_realsense_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            detector = FakeDetector([Detection("garbage bag", 0.8, (5, 5, 20, 20))])
            pipeline = VisionPipeline(config, detector=detector)
            result = pipeline.process_frame(np.zeros((30, 30, 3), dtype=np.uint8), source="video:0")
            self.assertEqual(pipeline.state()["volume_status"]["code"], "no_realsense_depth")
            self.assertTrue(any("--source realsense" in warning for warning in result.warnings))

    def test_valid_depth_without_baseline_reports_exact_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            pipeline = VisionPipeline(config, detector=FakeDetector())
            frame = np.zeros((30, 30, 3), dtype=np.uint8)
            pipeline.process_frame(
                frame,
                depth_m=np.full(frame.shape[:2], 1.5, dtype=np.float32),
                intrinsics=CameraIntrinsics(fx=100, fy=100),
            )
            state = pipeline.state()
            self.assertEqual(state["volume_status"]["code"], "missing_empty_baseline")
            self.assertEqual(state["realsense_depth"]["valid_percent"], 100.0)
            self.assertTrue(state["camera_intrinsics_ready"])

    def test_baseline_without_depth_requests_recapture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            detector = FakeDetector([Detection("garbage bag", 0.8, (5, 5, 20, 20))])
            pipeline = VisionPipeline(config, detector=detector)
            frame = np.zeros((30, 30, 3), dtype=np.uint8)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(frame, intrinsics=camera)
            result = pipeline.process_frame(
                frame, depth_m=np.full(frame.shape[:2], 1.5, dtype=np.float32), intrinsics=camera
            )
            self.assertEqual(pipeline.state()["volume_status"]["code"], "baseline_missing_depth")
            self.assertTrue(any("baseline has no RealSense depth" in item for item in result.warnings))

    def test_zero_depth_frames_are_reported_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            detector = FakeDetector([Detection("garbage bag", 0.8, (5, 5, 20, 20))])
            pipeline = VisionPipeline(config, detector=detector)
            frame = np.zeros((30, 30, 3), dtype=np.uint8)
            result = pipeline.process_frame(
                frame, depth_m=np.zeros(frame.shape[:2], dtype=np.float32), intrinsics=CameraIntrinsics(fx=100, fy=100)
            )
            self.assertEqual(pipeline.state()["volume_status"]["code"], "invalid_realsense_depth")
            self.assertTrue(any("no valid distances" in item for item in result.warnings))

    def test_monocular_depth_signal_is_visible_before_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary))
            pipeline = VisionPipeline(config, detector=FakeDetector(), depth_estimator=FakeDepthEstimator())
            pipeline.process_frame(np.zeros((30, 30, 3), dtype=np.uint8))
            state = pipeline.state()
            self.assertTrue(state["monocular_depth"]["available"])
            self.assertEqual(state["monocular_depth"]["median_m"], 2.0)
            self.assertFalse(state["monocular_calibrated"])


class DetectorParsingTests(unittest.TestCase):
    def test_segmentation_result_preserves_multiple_instances(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[2:12, 2:12] = (0, 0, 255)
        frame[20:32, 20:32] = (0, 255, 0)
        masks = np.zeros((2, 40, 40), dtype=np.float32)
        masks[0, 2:12, 2:12] = 1
        masks[1, 20:32, 20:32] = 1
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[2, 2, 12, 12], [20, 20, 32, 32]]),
            conf=FakeTensor([0.9, 0.8]),
            cls=FakeTensor([0, 1]),
            __len__=lambda _: 2,
        )

        class Boxes(SimpleNamespace):
            def __len__(self) -> int:
                return 2

        result = SimpleNamespace(
            boxes=Boxes(xyxy=boxes.xyxy, conf=boxes.conf, cls=boxes.cls),
            masks=SimpleNamespace(data=FakeTensor(masks)),
            names={0: "plastic bag", 1: "paper bag"},
        )
        detector = object.__new__(YoloSegmenter)
        detector.config = AppConfig(roi=(0, 0, 1, 1), min_component_pixels=20)
        detections = detector._parse_result(frame, result)
        self.assertEqual([item.label for item in detections], ["plastic bag", "paper bag"])
        self.assertEqual({item.color for item in detections}, {"red", "green"})

    def test_roi_filters_objects_outside_measurement_area(self) -> None:
        class Boxes(SimpleNamespace):
            def __len__(self) -> int:
                return 1

        result = SimpleNamespace(
            boxes=Boxes(xyxy=FakeTensor([[0, 0, 8, 8]]), conf=FakeTensor([0.9]), cls=FakeTensor([0])),
            masks=None,
            names={0: "plastic bag"},
        )
        detector = object.__new__(YoloSegmenter)
        detector.config = AppConfig(roi=(0.5, 0.5, 0.5, 0.5), min_component_pixels=5)
        detections = detector._parse_result(np.zeros((40, 40, 3), dtype=np.uint8), result)
        self.assertEqual(detections, [])


class TrackingTests(unittest.TestCase):
    def test_stationary_object_counted_only_once(self) -> None:
        tracker = ObjectTracker(confirmation_frames=2, max_missing_frames=2)
        for _ in range(10):
            tracker.update([Detection("bag", 0.9, (10, 10, 30, 30))])
        self.assertEqual(tracker.total_count, 1)

    def test_reappearing_object_is_new_event(self) -> None:
        tracker = ObjectTracker(confirmation_frames=1, max_missing_frames=1)
        tracker.update([Detection("bag", 0.9, (10, 10, 30, 30))])
        tracker.update([])
        tracker.update([])
        tracker.update([Detection("bag", 0.9, (10, 10, 30, 30))])
        self.assertEqual(tracker.total_count, 2)

    def test_two_distinct_objects_count_independently(self) -> None:
        tracker = ObjectTracker(confirmation_frames=1)
        tracker.update(
            [Detection("bag", 0.9, (0, 0, 10, 10)), Detection("bottle", 0.8, (20, 20, 30, 30))]
        )
        self.assertEqual(tracker.total_count, 2)

    def _phantom(self, box: tuple[int, int, int, int]) -> Detection:
        return Detection(
            "garbage bag (depth silhouette)", 0.0, box, source="fixed-bin-depth-silhouette",
        )

    def test_phantom_track_decays_much_faster_than_a_confirmed_track(self) -> None:
        # A phantom (unconfirmed depth-silhouette) detection carries zero
        # semantic evidence -- letting it live through the same ~24-frame
        # grace window as a real, neural-confirmed track is what let
        # unrelated background noise accumulate into its own long-lived
        # "object". It must expire almost immediately once it stops being
        # re-detected, regardless of the tracker's general missing-frame
        # budget.
        tracker = ObjectTracker(
            confirmation_frames=1, max_missing_frames=24, phantom_max_missing_frames=2,
        )
        tracker.update([self._phantom((10, 10, 60, 60))])
        phantom_id = next(iter(tracker.tracks))
        tracker.update([])
        tracker.update([])
        self.assertIn(phantom_id, tracker.tracks)
        tracker.update([])
        self.assertNotIn(phantom_id, tracker.tracks)

    def test_a_real_confirmed_track_still_uses_the_normal_missing_budget(self) -> None:
        tracker = ObjectTracker(
            confirmation_frames=1, max_missing_frames=24, phantom_max_missing_frames=2,
        )
        tracker.update([Detection("black garbage bag", 0.8, (10, 10, 60, 60), source="yoloe")])
        real_id = next(iter(tracker.tracks))
        for _ in range(10):
            tracker.update([])
        self.assertIn(real_id, tracker.tracks)

    def test_at_most_one_phantom_track_is_ever_alive_at_once(self) -> None:
        # A camera pan / noise flicker can make a *different* region the
        # largest unmatched blob from one frame to the next, each becoming
        # its own track. The fixed measurement bin holds one physical object,
        # so a fresh phantom detection must immediately retire any other
        # still-alive phantom track rather than let both coexist.
        tracker = ObjectTracker(
            confirmation_frames=1, max_missing_frames=24, phantom_max_missing_frames=2,
        )
        tracker.update([self._phantom((0, 0, 20, 20))])
        first_id = next(iter(tracker.tracks))
        tracker.update([])  # first_id now has missing == 1, still alive
        tracker.update([self._phantom((300, 300, 340, 340))])
        self.assertNotIn(first_id, tracker.tracks)
        self.assertEqual(len(tracker.tracks), 1)

    def test_a_confirmed_track_is_never_evicted_by_a_new_phantom(self) -> None:
        tracker = ObjectTracker(
            confirmation_frames=1, max_missing_frames=24, phantom_max_missing_frames=2,
        )
        tracker.update([Detection("black garbage bag", 0.8, (10, 10, 60, 60), source="yoloe")])
        real_id = next(iter(tracker.tracks))
        tracker.update(
            [
                Detection("black garbage bag", 0.8, (10, 10, 60, 60), source="yoloe"),
                self._phantom((300, 300, 340, 340)),
            ]
        )
        tracker.update(
            [
                Detection("black garbage bag", 0.8, (10, 10, 60, 60), source="yoloe"),
                self._phantom((400, 400, 440, 440)),
            ]
        )
        self.assertIn(real_id, tracker.tracks)


class PipelineTests(unittest.TestCase):
    def test_fixed_bin_bag_without_neural_prediction_is_measured_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=25,
                tracker_confirm_frames=1,
                bag_only=True,
                allow_unclassified_foreground=True,
                # This synthetic scene's 0.55 m height over a 1.2 m baseline
                # (a ~46% height/depth ratio, chosen for a clean round-number
                # region, not for real-world plausibility) integrates to
                # ~101 L under the default ray-frustum geometry (round 12:
                # the exact closed-form frustum-volume formula, which this
                # project's own math confirms is more accurate than the
                # previous surface-columns default -- see volume.py's
                # `_plane_perpendicular_height` docstring). That is a
                # materially different, more correct number from the old
                # default's ~81 L, not a regression; it now exceeds this
                # test's implausibility cap, which this test was never
                # actually exercising. Raised here so this test keeps
                # checking what it is named for (mask-extension honesty),
                # not incidentally re-testing the implausibility gate.
                realsense_max_item_volume_l=200.0,
            )
            pipeline = VisionPipeline(config, detector=FakeDetector())
            baseline = np.zeros((80, 80, 3), dtype=np.uint8)
            reference = np.full((80, 80), 1.2, dtype=np.float32)
            camera = CameraIntrinsics(fx=120, fy=120)
            pipeline.set_baseline(baseline, reference, camera)
            frame = baseline.copy()
            frame[10:70, 15:65] = (40, 100, 170)
            depth = reference.copy()
            depth[10:70, 15:65] = 0.65

            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].label, "garbage bag (depth silhouette)")
            self.assertEqual(result.detections[0].confidence, 0.0)
            self.assertEqual(result.automatic_count, 1)
            self.assertIsNotNone(result.realsense_total)

    def test_non_bag_prediction_is_excluded_when_depth_fallback_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mask = np.zeros((60, 60), dtype=bool)
            mask[5:55, 8:50] = True
            detector = FakeDetector([Detection("cardboard box", 0.97, (8, 5, 50, 55), mask=mask)])
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=20,
                tracker_confirm_frames=1,
                allow_unclassified_foreground=False,
                bag_only=True,
            )
            pipeline = VisionPipeline(config, detector=detector)
            baseline = np.zeros((60, 60, 3), dtype=np.uint8)
            reference = np.full((60, 60), 1.3, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(baseline, reference, camera)
            frame = baseline.copy()
            frame[5:55, 8:50] = 120
            depth = reference.copy()
            depth[5:55, 8:50] = 0.8

            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            self.assertEqual(result.detections, [])
            self.assertEqual(result.automatic_count, 0)
            self.assertIsNone(result.realsense_total)

    def test_unrelated_class_is_ignored_while_confirmed_bag_is_measured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            person_mask = np.zeros((80, 100), dtype=bool)
            person_mask[5:35, 5:35] = True
            box_mask = np.zeros((80, 100), dtype=bool)
            box_mask[40:70, 55:90] = True
            detector = FakeDetector([
                Detection("plastic bottle", 0.98, (5, 5, 35, 35), mask=person_mask),
                Detection("black garbage bag", 0.76, (55, 40, 90, 70), mask=box_mask),
            ])
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=20,
                tracker_confirm_frames=1,
                allow_unclassified_foreground=False,
            )
            pipeline = VisionPipeline(config, detector=detector)
            baseline = np.zeros((80, 100, 3), dtype=np.uint8)
            reference = np.full((80, 100), 1.5, dtype=np.float32)
            camera = CameraIntrinsics(fx=140, fy=140)
            pipeline.set_baseline(baseline, reference, camera)
            frame = baseline.copy()
            frame[person_mask] = 100
            frame[box_mask] = 160
            depth = reference.copy()
            depth[person_mask] = 0.9
            depth[box_mask] = 1.2

            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].label, "black garbage bag")
            self.assertEqual(result.automatic_count, 1)
            self.assertIsNotNone(result.detections[0].realsense_volume_l)

    def test_partial_bag_prediction_measures_the_entire_depth_supported_bag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            label_mask = np.zeros((80, 100), dtype=bool)
            label_mask[24:34, 30:43] = True
            detector = FakeDetector(
                [Detection("garbage bag", 0.25, (30, 24, 43, 34), mask=label_mask)]
            )
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=25,
                tracker_confirm_frames=1,
                # Same reasoning as test_fixed_bin_bag_without_neural_prediction
                # above: this scene's 0.3 m height over a 2.0 m baseline totals
                # ~96 L under the (more accurate, round 12) default ray-frustum
                # geometry, versus ~81 L under the old surface-columns default
                # -- a real, intentional improvement in accuracy, not a
                # regression, that happens to cross this test's implausibility
                # cap. Raised so this test keeps checking mask-extension
                # honesty, not the implausibility gate.
                realsense_max_item_volume_l=200.0,
            )
            pipeline = VisionPipeline(config, detector=detector)
            baseline = np.full((80, 100, 3), 60, dtype=np.uint8)
            baseline_depth = np.full((80, 100), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=200, fy=200)
            pipeline.set_baseline(baseline, baseline_depth, camera)
            frame = baseline.copy()
            frame[24:34, 30:43] = 230
            depth = baseline_depth.copy()
            depth[15:65, 12:87] = 1.70

            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].label, "garbage bag")
            self.assertEqual(result.detections[0].box, (12, 15, 87, 65))
            self.assertEqual(result.realsense_total.valid_pixels, 3750)
            self.assertGreater(result.detections[0].realsense_volume_l, 70.0)
            self.assertEqual(result.detections[0].depth_distance_m, 1.7)
            self.assertEqual(result.detections[0].height_above_baseline_cm, 30.0)
            detection_payload = result.detections[0].to_dict()
            self.assertEqual(detection_payload["depth_distance_m"], 1.7)
            self.assertEqual(detection_payload["accepted_class"], "plastic_bag")
            self.assertIsNotNone(detection_payload["dimensions_mm"])
            self.assertGreater(detection_payload["dimensions_mm"]["footprint_length"], 0)
            self.assertGreater(detection_payload["dimensions_mm"]["footprint_width"], 0)
            self.assertAlmostEqual(detection_payload["dimensions_mm"]["height"], 300.0, delta=1.0)
            self.assertEqual(pipeline.state()["volume_status"]["code"], "measuring")

    def test_dome_shaped_object_reports_near_its_peak_height_not_a_footprint_median(self) -> None:
        # Round 8 (real hardware): a ~35 cm bag/pillow was displayed as
        # ~15 cm tall. Root cause was that the dashboard's per-detection
        # height came from a separate raw median computed over the whole
        # detection mask -- for a dome-shaped or tapered object, most of
        # that mask sits near the sloped edges, far below the true peak.
        # This drives the fix end to end through VisionPipeline, the same
        # path the dashboard reads `height_above_baseline_cm` from.
        with tempfile.TemporaryDirectory() as temporary:
            size = 60
            box_mask = np.zeros((size, size), dtype=bool)
            box_mask[10:50, 10:50] = True
            detector = FakeDetector([Detection("garbage bag", 0.9, (10, 10, 50, 50), mask=box_mask)])
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=25,
                tracker_confirm_frames=1,
            )
            pipeline = VisionPipeline(config, detector=detector)
            baseline = np.full((size, size, 3), 60, dtype=np.uint8)
            baseline_depth = np.full((size, size), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=300, fy=300)
            pipeline.set_baseline(baseline, baseline_depth, camera)

            frame = baseline.copy()
            frame[box_mask] = 230
            yy, xx = np.mgrid[0:size, 0:size]
            center = 29.5
            radial = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
            radial /= radial[box_mask].max()
            peak_height_m = 0.35
            height_field = peak_height_m * np.clip(1.0 - radial, 0.0, 1.0)
            depth = baseline_depth.copy()
            depth[box_mask] = (baseline_depth - height_field)[box_mask]

            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            self.assertEqual(len(result.detections), 1)
            reported_cm = result.detections[0].height_above_baseline_cm
            self.assertIsNotNone(reported_cm)
            median_cm = float(np.median(height_field[box_mask])) * 100.0
            self.assertLess(median_cm, 20.0)  # the exact regime the old bug reported
            self.assertGreater(reported_cm, median_cm * 1.5)

    def test_existing_detections_explain_that_an_empty_baseline_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            detector = FakeDetector([Detection("paper", 0.25, (4, 4, 15, 15))])
            config = AppConfig(
                results_dir=Path(temporary), enable_monocular_depth=False, min_component_pixels=10
            )
            pipeline = VisionPipeline(config, detector=detector)

            result = pipeline.process_frame(np.zeros((30, 30, 3), dtype=np.uint8))

            self.assertTrue(any("empty-scene baseline" in warning for warning in result.warnings))

    def test_hardware_volume_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=25,
                tracker_confirm_frames=1,
                allow_unclassified_foreground=True,
                bag_only=True,
            )
            pipeline = VisionPipeline(config, detector=FakeDetector())
            baseline = np.zeros((50, 50, 3), dtype=np.uint8)
            baseline_depth = np.full((50, 50), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(baseline, baseline_depth, camera)

            frame = baseline.copy()
            frame[10:25, 10:25] = (0, 0, 255)
            depth = baseline_depth.copy()
            depth[10:25, 10:25] = 1.7
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].label, "garbage bag (depth silhouette)")
            self.assertEqual(result.detections[0].confidence, 0.0)
            self.assertEqual(result.automatic_count, 1)
            self.assertIsNotNone(result.realsense_total)
            self.assertTrue((Path(temporary) / "events.jsonl").is_file())

    def test_hardware_total_excludes_a_simultaneous_phantom_detection(self) -> None:
        # Round-8+ real hardware routinely shows a phantom "depth silhouette"
        # (a shadow, an unrelated changed-scene fragment) alongside a real
        # confirmed bag in the same frame. `hardware_total`/`monocular_total`
        # -- the per-camera aggregate liters figure rendered as the
        # dashboard's "CURRENT VOLUME" metric -- must reflect only the real,
        # confirmed object's own volume, not the union of both. Before this
        # fix, `combined_mask(detections)` unioned every detection's mask
        # regardless of phantom status, so the aggregate silently absorbed
        # the phantom's own (fictional) volume too.
        with tempfile.TemporaryDirectory() as temporary:
            mask_a = np.zeros((70, 70), dtype=bool)
            mask_a[5:20, 5:20] = True
            mask_b = np.zeros((70, 70), dtype=bool)
            mask_b[45:60, 45:60] = True
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=25,
                tracker_confirm_frames=1,
                allow_unclassified_foreground=True,
                bag_only=True,
            )
            detector = FakeDetector([Detection("garbage bag", 0.9, (5, 5, 20, 20), mask=mask_a.copy())])
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((70, 70, 3), dtype=np.uint8)
            baseline_depth = np.full((70, 70), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, baseline_depth, camera)

            frame = empty.copy()
            frame[mask_a] = (0, 0, 255)
            frame[mask_b] = (0, 255, 0)
            depth = baseline_depth.copy()
            depth[mask_a] = 1.7
            depth[mask_b] = 1.8
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)

            confirmed = [item for item in result.detections if item.confidence > 0]
            phantom = [item for item in result.detections if item.confidence == 0]
            self.assertEqual(len(confirmed), 1)
            self.assertEqual(len(phantom), 1)
            self.assertIsNotNone(confirmed[0].realsense_volume_l)
            self.assertIsNotNone(phantom[0].realsense_volume_l)
            self.assertIsNotNone(result.realsense_total)
            self.assertAlmostEqual(
                result.realsense_total.liters, confirmed[0].realsense_volume_l, places=5
            )
            # The phantom's own volume must not have leaked into the total.
            self.assertLess(
                result.realsense_total.liters,
                confirmed[0].realsense_volume_l + phantom[0].realsense_volume_l - 0.1,
            )

    def test_monocular_volume_is_withheld_without_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mask = np.zeros((30, 30), dtype=bool)
            mask[5:20, 5:20] = True
            detector = FakeDetector([Detection("garbage bag", 0.8, (5, 5, 20, 20), mask=mask)])
            config = AppConfig(results_dir=Path(temporary), roi=(0, 0, 1, 1), min_component_pixels=10)
            pipeline = VisionPipeline(config, detector=detector, depth_estimator=FakeDepthEstimator())
            baseline = np.zeros((30, 30, 3), dtype=np.uint8)
            pipeline.set_baseline(baseline, intrinsics=CameraIntrinsics(fx=100, fy=100))
            frame = baseline.copy()
            frame[5:20, 5:20, 0] = 60
            result = pipeline.process_frame(frame)
            self.assertIsNone(result.monocular_total)
            self.assertTrue(any("withheld" in warning for warning in result.warnings))

    def test_calibrated_monocular_volume_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mask = np.zeros((40, 40), dtype=bool)
            mask[8:28, 8:28] = True
            detector = FakeDetector([Detection("garbage bag", 0.9, (8, 8, 28, 28), mask=mask)])
            config = AppConfig(results_dir=Path(temporary), roi=(0, 0, 1, 1), min_component_pixels=20)
            pipeline = VisionPipeline(config, detector=detector, depth_estimator=FakeDepthEstimator())
            baseline = np.zeros((40, 40, 3), dtype=np.uint8)
            baseline_depth = np.full((40, 40), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(baseline, baseline_depth, camera)
            frame = baseline.copy()
            frame[8:28, 8:28, 0] = 51
            depth = baseline_depth.copy()
            depth[8:28, 8:28] = 1.8
            result = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            self.assertIsNotNone(result.realsense_total)
            self.assertIsNotNone(result.monocular_total)
            self.assertAlmostEqual(result.realsense_total.liters, result.monocular_total.liters, places=4)

    def test_batch_length_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(results_dir=Path(temporary), enable_monocular_depth=False)
            pipeline = VisionPipeline(config, detector=FakeDetector())
            with self.assertRaises(ValueError):
                pipeline.process_batch([np.zeros((10, 10, 3), dtype=np.uint8)], depths=[])


if __name__ == "__main__":
    unittest.main()
