"""Sequential deposits that touch, calibration validity, and the CSV record.

The field failure these pin down: one black bag is measured and accepted, a
second black bag is placed touching it, their masks and depth components join,
and the pair is remeasured as a single ~29 L object. Colour cannot separate
them -- both are identical black polythene -- so the separation is geometric:
every cell the first bag occupies already carries its height in the committed
grid, and only cells the second bag raised count towards the new deposit.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.heightmap_volume import (
    align_grids,
    incremental_deposit,
    integrate_height_map,
)
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection
from tests.test_heightmap_volume import (
    INTRINSICS,
    SETTINGS,
    _depth_at_height,
    _frame,
    _plane_coordinates,
)


def scene(boxes, tilt_degrees=0.0, floor_shift_m=0.0):
    """Height grid of a floor carrying `boxes` = (u, v, length, width, height) in metres."""
    coefficients, scale, first, second, slope_x, slope_y = _frame(tilt_degrees)
    coefficients = (coefficients[0], coefficients[1], coefficients[2] + floor_shift_m)
    depth = _depth_at_height(coefficients, scale, slope_x, slope_y, 0.0)
    for _ in range(3):
        u, v = _plane_coordinates(depth, first, second, slope_x, slope_y)
        height = np.zeros_like(u)
        for centre_u, centre_v, length, width, tall in boxes:
            height = np.where(
                (np.abs(u - centre_u) <= length / 2) & (np.abs(v - centre_v) <= width / 2),
                tall,
                height,
            )
        depth = _depth_at_height(coefficients, scale, slope_x, slope_y, height)
    return integrate_height_map(
        depth, INTRINSICS, plane_coefficients=coefficients, settings=SETTINGS
    )


BAG_ONE = (-0.09, 0.0, 0.16, 0.14, 0.10)
BAG_TWO_TOUCHING = (0.075, 0.0, 0.15, 0.14, 0.09)
BAG_TWO_SEPARATE = (0.16, 0.0, 0.15, 0.14, 0.09)
BAG_TWO_LITRES = 0.15 * 0.14 * 0.09 * 1000


class TouchingDepositTests(unittest.TestCase):
    def setUp(self) -> None:
        self.committed = scene([BAG_ONE])

    def test_a_touching_second_bag_is_measured_alone(self) -> None:
        combined = scene([BAG_ONE, BAG_TWO_TOUCHING])
        # The whole scene really does measure as the sum -- that is the figure
        # the merged mask used to publish as one new deposit.
        self.assertGreater(combined.liters, self.committed.liters * 1.7)
        increment = incremental_deposit(self.committed, combined)
        self.assertEqual(increment.quality, "valid")
        self.assertAlmostEqual(
            increment.added_liters, BAG_TWO_LITRES, delta=BAG_TWO_LITRES * 0.20
        )
        self.assertLess(increment.added_liters, self.committed.liters)

    def test_a_separated_second_bag_gives_the_same_answer(self) -> None:
        increment = incremental_deposit(self.committed, scene([BAG_ONE, BAG_TWO_SEPARATE]))
        self.assertEqual(increment.quality, "valid")
        self.assertAlmostEqual(
            increment.added_liters, BAG_TWO_LITRES, delta=BAG_TWO_LITRES * 0.20
        )

    def test_a_partial_footprint_overlap_counts_only_the_extra_height(self) -> None:
        # Second bag stacked partly on the first: only the height it adds above
        # the committed surface is new volume.
        stacked = scene([BAG_ONE, (-0.03, 0.0, 0.12, 0.12, 0.16)])
        increment = incremental_deposit(self.committed, stacked)
        self.assertEqual(increment.quality, "valid")
        self.assertLess(increment.added_liters, 0.12 * 0.12 * 0.16 * 1000)
        self.assertGreater(increment.added_liters, 0.0)

    def test_slight_compression_of_the_first_bag_is_tolerated(self) -> None:
        settled = scene([(-0.09, 0.0, 0.16, 0.14, 0.072), BAG_TWO_TOUCHING])
        increment = incremental_deposit(self.committed, settled)
        self.assertEqual(increment.quality, "valid")
        self.assertIn("existing-contents-settled", increment.flags)
        self.assertGreater(increment.added_liters, 0.0)

    def test_moving_the_first_bag_is_refused_not_counted(self) -> None:
        # Same bag, new place. Summing the positive cells would count it twice.
        increment = incremental_deposit(self.committed, scene([(0.075, 0.0, 0.16, 0.14, 0.10)]))
        self.assertEqual(increment.quality, "rejected")
        self.assertEqual(increment.rejection_reason, "possible_existing_object_movement")
        self.assertEqual(increment.added_liters, 0.0)

    def test_an_unchanged_scene_is_not_a_deposit(self) -> None:
        increment = incremental_deposit(self.committed, self.committed)
        self.assertEqual(increment.quality, "rejected")
        self.assertEqual(increment.rejection_reason, "new_deposit_not_isolatable")
        self.assertEqual(increment.added_liters, 0.0)

    def test_removing_the_bag_is_never_a_deposit(self) -> None:
        # An emptied bin integrates to nothing at all, so there is no "after"
        # grid to difference -- the caller must get None, never a volume.
        self.assertIsNone(scene([]))
        self.assertIsNone(incremental_deposit(self.committed, scene([])))

    def test_grids_with_different_origins_are_aligned_before_differencing(self) -> None:
        # Two captures need not cover the same patch of floor, so differencing
        # must line them up by absolute cell coordinate, never by array index.
        committed = replace(
            self.committed,
            grid=np.array([[0.10, 0.10], [0.10, 0.10]]),
            origin_row=4,
            origin_column=4,
        )
        current = replace(
            self.committed,
            grid=np.array([[0.10, 0.10, 0.05], [0.10, 0.10, 0.05]]),
            origin_row=4,
            origin_column=4,
        )
        before, after = align_grids(committed, current)
        self.assertEqual(before.shape, after.shape)
        # The shared cells are unchanged; only the new column carries height.
        np.testing.assert_allclose(after[:, :2] - before[:, :2], 0.0)
        self.assertTrue(np.all(np.isnan(before[:, 2])))

    def test_a_shifted_origin_does_not_fake_a_deposit(self) -> None:
        shifted = replace(self.committed, origin_row=self.committed.origin_row + 3)
        increment = incremental_deposit(self.committed, shifted)
        self.assertNotEqual(increment.quality, "valid")


class OrganicBagRegressionTests(unittest.TestCase):
    """The confirmed-good isolated cases must not move."""

    def test_apple_and_banana_sized_bags_still_measure_correctly(self) -> None:
        for name, (length, width, height) in (
            ("apple", (0.209, 0.164, 0.076)),
            ("banana", (0.191, 0.123, 0.072)),
        ):
            with self.subTest(bag=name):
                truth = length * width * height * 1000
                measured = scene([(0.0, 0.0, length, width, height)])
                self.assertLess(abs(measured.liters - truth) / truth, 0.10)


SIZE = 80
BASE_DEPTH_M = 2.0


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
        tracker_max_missing_frames=3,
        tracker_live_prediction_frames=2,
        settle_frames=2,
        volume_window_frames=2,
        auto_deposit=True,
        bag_only=False,
        restore_saved_baseline=False,
        **overrides,
    )
    detector = _Detector()
    pipeline = VisionPipeline(config, detector=detector)
    empty = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    floor = np.full((SIZE, SIZE), BASE_DEPTH_M, dtype=np.float32)
    camera = CameraIntrinsics(fx=100, fy=100, ppx=SIZE / 2, ppy=SIZE / 2)
    pipeline.set_baseline(empty, floor, camera)
    return pipeline, detector, empty, floor, camera


def _bag(bounds, thickness_m, colour=(25, 25, 25)):
    top, bottom, left, right = bounds
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top:bottom, left:right] = True
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[mask] = colour
    depth = np.full((SIZE, SIZE), BASE_DEPTH_M, dtype=np.float32)
    depth[mask] = BASE_DEPTH_M - thickness_m
    return mask, frame, depth, (left, top, right, bottom)


class LedgerImmutabilityTests(unittest.TestCase):
    def _deposit_first_bag(self, pipeline, detector, empty, floor, camera):
        mask, frame, depth, box = _bag((25, 40, 20, 35), 0.12)
        pipeline.process_frame(empty, depth_m=floor, intrinsics=camera, timestamp=100.0)
        detector.detections = [Detection("black trash bag", 0.9, box, mask)]
        for stamp in (101.0, 102.0):
            analysis = pipeline.process_frame(
                frame, depth_m=depth, intrinsics=camera, timestamp=stamp,
            )
        return analysis, frame, depth

    def test_a_committed_record_never_changes_afterwards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, floor, camera = _pipeline(directory)
            analysis, frame, depth = self._deposit_first_bag(
                pipeline, detector, empty, floor, camera
            )
            first = analysis.detections[0]
            record = dict(pipeline.ledger.all_records()[0])
            self.assertEqual(record["status"], "deposited")

            # A second bag arrives touching the first; the detector reports the
            # merged pair under the same track.
            merged_mask, merged_frame, merged_depth, merged_box = _bag((25, 40, 20, 60), 0.12)
            detector.detections = [
                Detection("black trash bag", 0.9, merged_box, merged_mask)
            ]
            for stamp in (103.0, 104.0, 105.0):
                pipeline.process_frame(
                    merged_frame, depth_m=merged_depth, intrinsics=camera, timestamp=stamp,
                )
            after = dict(pipeline.ledger.all_records()[0])
            self.assertEqual(after["volume_l"], record["volume_l"])
            self.assertEqual(after["entry_id"], record["entry_id"])
            self.assertEqual(after["deposited_at"], record["deposited_at"])
            self.assertEqual(after["status"], "deposited")
            self.assertIsNotNone(first.realsense_volume_l)

    def test_the_merged_pair_is_not_recorded_as_one_large_new_deposit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, floor, camera = _pipeline(directory)
            self._deposit_first_bag(pipeline, detector, empty, floor, camera)
            committed = [
                record["volume_l"] for record in pipeline.ledger.all_records()
            ]
            merged_mask, merged_frame, merged_depth, merged_box = _bag((25, 40, 20, 60), 0.12)
            detector.detections = [
                Detection("black trash bag", 0.9, merged_box, merged_mask)
            ]
            for stamp in (103.0, 104.0, 105.0):
                pipeline.process_frame(
                    merged_frame, depth_m=merged_depth, intrinsics=camera, timestamp=stamp,
                )
            volumes = [record["volume_l"] for record in pipeline.ledger.all_records()]
            # Whatever else happens, no record may hold the combined figure.
            self.assertLess(max(volumes), sum(committed) * 2.0)


class CalibrationValidityTests(unittest.TestCase):
    def test_moving_the_camera_invalidates_the_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, floor, camera = _pipeline(directory)
            pipeline.process_frame(empty, depth_m=floor, intrinsics=camera, timestamp=100.0)
            self.assertTrue(pipeline._calibration_valid)
            # The camera is moved 25 cm further from the surface.
            moved = np.full((SIZE, SIZE), BASE_DEPTH_M + 0.25, dtype=np.float32)
            analysis = pipeline.process_frame(
                empty, depth_m=moved, intrinsics=camera, timestamp=101.0,
            )
            self.assertFalse(pipeline._calibration_valid)
            self.assertTrue(
                any(
                    "camera_moved_recalibration_required" in warning
                    for warning in analysis.warnings
                ),
                analysis.warnings,
            )

    def test_an_unmoved_camera_keeps_its_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, floor, camera = _pipeline(directory)
            for stamp in (100.0, 101.0, 102.0):
                analysis = pipeline.process_frame(
                    empty, depth_m=floor, intrinsics=camera, timestamp=stamp,
                )
            self.assertTrue(pipeline._calibration_valid)
            self.assertFalse(
                [w for w in analysis.warnings if "camera_moved" in w]
            )


class MeasurementCsvTests(unittest.TestCase):
    def _run(self, directory, **overrides):
        pipeline, detector, empty, floor, camera = _pipeline(directory, **overrides)
        mask, frame, depth, box = _bag((25, 40, 20, 35), 0.12, colour=(40, 190, 40))
        pipeline.process_frame(empty, depth_m=floor, intrinsics=camera, timestamp=100.0)
        detector.detections = [Detection("filled plastic waste bag", 0.9, box, mask)]
        for stamp in (101.0, 102.0):
            pipeline.process_frame(
                frame, depth_m=depth, intrinsics=camera, timestamp=stamp,
            )
        return Path(directory) / "measurements.csv"

    def test_a_row_is_written_that_excel_can_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._run(directory)
            self.assertTrue(path.is_file(), "measurements.csv was never written")
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(float(row["volume_l"]) > 0)
            self.assertEqual(row["color"], "green")
            self.assertEqual(row["sorting_status"], "correct")
            self.assertTrue(row["calibration_id"])

    def test_validation_mode_also_records_rows(self) -> None:
        # The waste ledger is disabled in geometry_validation mode, which is why
        # the dashboard's ledger CSV came back empty for every validation run.
        with tempfile.TemporaryDirectory() as directory:
            path = self._run(directory, operating_mode="geometry_validation")
            self.assertTrue(path.is_file())
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["operating_mode"], "geometry_validation")

    def test_a_row_is_written_once_per_object_not_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, empty, floor, camera = _pipeline(directory)
            mask, frame, depth, box = _bag((25, 40, 20, 35), 0.12)
            pipeline.process_frame(empty, depth_m=floor, intrinsics=camera, timestamp=100.0)
            detector.detections = [Detection("black trash bag", 0.9, box, mask)]
            for stamp in range(101, 110):
                pipeline.process_frame(
                    frame, depth_m=depth, intrinsics=camera, timestamp=float(stamp),
                )
            rows = list(
                csv.DictReader(
                    (Path(directory) / "measurements.csv").read_text(
                        encoding="utf-8"
                    ).splitlines()
                )
            )
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
