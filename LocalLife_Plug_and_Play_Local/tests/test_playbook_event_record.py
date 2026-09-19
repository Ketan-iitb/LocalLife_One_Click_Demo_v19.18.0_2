"""One deposit produces one complete event record (playbook sections 2, 15, 28).

The playbook's definition of done is that a deposit yields an event carrying an
object class, a colour, a sorting verdict and a non-null volume, with invalid
cases flagged rather than hidden. These drive the whole pipeline to check that
those fields actually arrive together on the same detection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection

SIZE = 70
BASELINE_DEPTH_M = 2.0


class _Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.detections: list[Detection] = []

    def detect_batch(self, frames):
        return [
            [
                Detection(
                    item.label, item.confidence, item.box,
                    None if item.mask is None else item.mask.copy(),
                )
                for item in self.detections
            ]
            for _ in frames
        ]


def _pipeline(directory, **overrides):
    config = AppConfig(
        results_dir=Path(directory),
        enable_monocular_depth=False,
        roi=(0, 0, 1, 1),
        min_component_pixels=20,
        tracker_confirm_frames=1,
        settle_frames=2,
        volume_window_frames=2,
        auto_deposit=True,
        bag_only=False,
        **overrides,
    )
    detector = _Detector()
    pipeline = VisionPipeline(config, detector=detector)
    empty = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    baseline = np.full((SIZE, SIZE), BASELINE_DEPTH_M, dtype=np.float32)
    camera = CameraIntrinsics(fx=100, fy=100, ppx=SIZE / 2, ppy=SIZE / 2)
    pipeline.set_baseline(empty, baseline, camera)
    return pipeline, detector, empty, baseline, camera


def _object(bounds, colour, thickness_m):
    """A mask, its BGR frame and the depth image of an object that thick."""
    top, bottom, left, right = bounds
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top:bottom, left:right] = True
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[mask] = colour
    depth = np.full((SIZE, SIZE), BASELINE_DEPTH_M, dtype=np.float32)
    depth[mask] = BASELINE_DEPTH_M - thickness_m
    return mask, frame, depth, (left, top, right, bottom)


class DepositEventTests(unittest.TestCase):
    def test_one_deposit_carries_class_colour_sorting_and_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline, camera = _pipeline(directory)
            mask, frame, depth, box = _object((25, 45, 25, 45), (40, 190, 40), 0.15)
            # An empty bin first, so the deposit has a real "before" state.
            pipeline.process_frame(
                empty, depth_m=baseline, intrinsics=camera, timestamp=100.0,
            )
            detector.detections = [Detection("filled plastic waste bag", 0.9, box, mask)]
            for timestamp in (101.0, 102.0):
                analysis = pipeline.process_frame(
                    frame, depth_m=depth, intrinsics=camera, timestamp=timestamp,
                )

            deposited = analysis.detections[0]
            self.assertEqual(deposited.accepted_class, "plastic_bag")
            self.assertEqual(deposited.color, "green")
            self.assertGreater(deposited.color_confidence, 0.5)
            self.assertEqual(deposited.sorting_status, "correct")
            self.assertIsNotNone(deposited.realsense_volume_l)
            self.assertGreater(deposited.realsense_volume_l, 0.0)

            payload = deposited.to_dict()
            for field in (
                "accepted_class", "color", "color_confidence",
                "sorting_status", "realsense_volume_l",
            ):
                self.assertIn(field, payload)

    def test_added_volume_records_what_the_bin_gained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline, camera = _pipeline(directory)
            mask, frame, depth, box = _object((25, 45, 25, 45), (40, 190, 40), 0.15)
            pipeline.process_frame(
                empty, depth_m=baseline, intrinsics=camera, timestamp=100.0,
            )
            detector.detections = [Detection("filled plastic waste bag", 0.9, box, mask)]
            for timestamp in (101.0, 102.0):
                analysis = pipeline.process_frame(
                    frame, depth_m=depth, intrinsics=camera, timestamp=timestamp,
                )

            deposited = analysis.detections[0]
            self.assertIsNotNone(deposited.added_volume_l)
            self.assertEqual(deposited.volume_before_l, 0.0)
            self.assertGreater(deposited.added_volume_l, 0.0)
            # Into an empty bin, what the bin gained is what the object
            # measures -- the two paths must agree.
            self.assertAlmostEqual(
                deposited.added_volume_l,
                deposited.realsense_volume_l,
                delta=max(0.5, deposited.realsense_volume_l * 0.25),
            )

    def test_added_volume_is_not_invented_without_a_before_state(self) -> None:
        # No empty frame first: the bag is already there when measurement
        # starts, so the bin's prior occupancy was never observed.
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, _, _, camera = _pipeline(directory)
            mask, frame, depth, box = _object((25, 45, 25, 45), (40, 190, 40), 0.15)
            detector.detections = [Detection("filled plastic waste bag", 0.9, box, mask)]
            for timestamp in (101.0, 102.0):
                analysis = pipeline.process_frame(
                    frame, depth_m=depth, intrinsics=camera, timestamp=timestamp,
                )
            self.assertIsNone(analysis.detections[0].added_volume_l)


class MisSortReportingTests(unittest.TestCase):
    """Playbook section 12: a disallowed object must be reported, not dropped."""

    def _run(self, label):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, baseline, camera = _pipeline(directory)
            mask, frame, depth, box = _object((25, 45, 25, 45), (30, 30, 220), 0.15)
            pipeline.process_frame(
                empty, depth_m=baseline, intrinsics=camera, timestamp=100.0,
            )
            detector.detections = [Detection(label, 0.9, box, mask)]
            return pipeline.process_frame(
                frame, depth_m=depth, intrinsics=camera, timestamp=101.0,
            )

    def test_a_disallowed_object_raises_a_mis_sort_warning(self) -> None:
        for label, family in (
            ("slipper", "footwear"),
            ("cordless drill", "tool"),
            ("vacuum cleaner", "appliance"),
        ):
            with self.subTest(label=label):
                analysis = self._run(label)
                self.assertTrue(
                    any(
                        warning.startswith("MIS-SORT") and family in warning
                        for warning in analysis.warnings
                    ),
                    analysis.warnings,
                )

    def test_a_mis_sorted_object_never_enters_the_ledger(self) -> None:
        # The warning must not come at the cost of letting a slipper be
        # tracked, measured or recorded as deposited waste.
        analysis = self._run("slipper")
        self.assertEqual(analysis.detections, [])

    def test_accepted_waste_raises_no_mis_sort_warning(self) -> None:
        analysis = self._run("filled plastic waste bag")
        self.assertFalse(
            [warning for warning in analysis.warnings if warning.startswith("MIS-SORT")]
        )


if __name__ == "__main__":
    unittest.main()
