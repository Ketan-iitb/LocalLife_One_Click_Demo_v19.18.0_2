"""Waste-plant ledger, color accounting, restart recovery, and auto deposits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.ledger import WastePlantLedger, waste_object_type
from locallife_cloud.pipeline import VisionPipeline, is_supported_waste_detection
from locallife_cloud.storage import ResultStore
from locallife_cloud.types import CameraIntrinsics, Detection


class AdjustableDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.detections: list[Detection] = []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [
            [Detection(item.label, item.confidence, item.box, None if item.mask is None else item.mask.copy())
             for item in self.detections]
            for _ in frames
        ]


class LedgerTests(unittest.TestCase):
    def test_bags_boxes_and_unsupported_objects_are_distinguished(self) -> None:
        self.assertEqual(waste_object_type("black garbage bag"), "bag")
        self.assertEqual(waste_object_type("refuse sack"), "bag")
        self.assertEqual(waste_object_type("cardboard shipping box"), "box")
        self.assertEqual(waste_object_type("parcel"), "box")
        self.assertEqual(waste_object_type("plastic bottle"), "other")
        self.assertTrue(is_supported_waste_detection("white trash bag"))
        self.assertTrue(is_supported_waste_detection("carton box"))
        self.assertFalse(is_supported_waste_detection("plastic bottle"))

    def test_repeated_frames_do_not_duplicate_one_tracked_bag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = WastePlantLedger(ResultStore(Path(temporary)))
            detection = Detection("garbage bag", 0.8, (0, 0, 10, 10), color="blue", track_id=3)
            detection.realsense_volume_l = 12.5
            ledger.observe(detection, timestamp=10)
            ledger.observe(detection, timestamp=11)
            ledger.deposit(detection, timestamp=12)
            ledger.deposit(detection, timestamp=13)
            summary = ledger.summary()
            self.assertEqual(summary["observed_bags"], 1)
            self.assertEqual(summary["deposited_bags"], 1)
            self.assertEqual(summary["cumulative_volume_l"], 12.5)

    def test_seen_and_deposited_items_have_distinct_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = WastePlantLedger(ResultStore(Path(temporary)))
            bag = Detection("red garbage bag", 0.8, (0, 0, 10, 10), color="red", track_id=1)
            bag.realsense_volume_l = 9.0
            box = Detection("cardboard box", 0.7, (20, 20, 30, 30), color="brown", track_id=2)
            box.realsense_volume_l = 15.0
            ledger.observe(bag)
            ledger.observe(box)
            ledger.deposit(bag)
            summary = ledger.summary()
            self.assertEqual(summary["observed_count"], 2)
            self.assertEqual(summary["observed_bags"], 1)
            self.assertEqual(summary["observed_boxes"], 1)
            self.assertEqual(summary["deposited_count"], 1)
            self.assertEqual(summary["cumulative_volume_l"], 9.0)

    def test_plant_configured_colors_control_waste_stream_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = WastePlantLedger(
                ResultStore(Path(temporary)), color_streams={"blue": "plastic", "green": "organics"}
            )
            detection = Detection("garbage bag", 0.8, (0, 0, 10, 10), color="blue", track_id=1)
            detection.realsense_volume_l = 18.25
            ledger.deposit(detection)
            summary = ledger.summary()
            self.assertEqual(summary["history"][0]["waste_stream"], "plastic")
            self.assertEqual(summary["colors"][0]["deposited_count"], 1)
            self.assertEqual(summary["waste_streams"][0]["volume_l"], 18.25)

    def test_no_content_is_invented_for_an_unconfigured_color(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = WastePlantLedger(ResultStore(Path(temporary)))
            detection = Detection("garbage bag", 0.8, (0, 0, 10, 10), color="green", track_id=1)
            detection.realsense_volume_l = 7
            ledger.deposit(detection)
            self.assertIsNone(ledger.summary()["history"][0]["waste_stream"])
            self.assertEqual(ledger.summary()["waste_streams"], [])

    def test_history_and_totals_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(Path(temporary))
            initial = WastePlantLedger(store)
            detection = Detection("cardboard box", 0.9, (0, 0, 10, 10), color="brown", track_id=8)
            detection.realsense_volume_l = 22.75
            initial.deposit(detection, timestamp=100)
            restarted = WastePlantLedger(store)
            summary = restarted.summary()
            self.assertEqual(summary["observed_boxes"], 1)
            self.assertEqual(summary["deposited_boxes"], 1)
            self.assertEqual(summary["cumulative_volume_l"], 22.75)
            self.assertEqual(summary["history"][0]["status"], "deposited")

    def test_dashboard_history_limit_does_not_discard_exportable_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = WastePlantLedger(ResultStore(Path(temporary)), history_limit=2)
            for index in range(5):
                detection = Detection("garbage bag", 0.8, (0, 0, 10, 10), track_id=index + 1)
                detection.realsense_volume_l = float(index + 1)
                ledger.deposit(detection, timestamp=float(index))
            summary = ledger.summary()
            self.assertEqual(summary["observed_count"], 5)
            self.assertEqual(len(summary["history"]), 2)
            self.assertEqual(len(ledger.all_records()), 5)
            self.assertEqual(summary["cumulative_volume_l"], 15.0)


class AutomaticDepositTests(unittest.TestCase):
    def test_geometry_validation_tracks_measures_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            detector = AdjustableDetector()
            config = AppConfig(
                results_dir=Path(temporary), operating_mode="geometry_validation",
                enable_monocular_depth=False, roi=(0, 0, 1, 1),
                min_component_pixels=20, tracker_confirm_frames=1,
                settle_frames=2, volume_window_frames=2, auto_deposit=True,
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((70, 70, 3), dtype=np.uint8)
            baseline = np.full((70, 70), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100, ppx=35, ppy=35)
            pipeline.set_baseline(empty, baseline, camera)
            mask = np.zeros((70, 70), dtype=bool)
            mask[15:55, 18:52] = True
            detector.detections = [Detection("laptop bag", 0.9, (18, 15, 52, 55), mask)]
            frame = empty.copy()
            frame[mask] = (20, 120, 40)
            depth = baseline.copy()
            # The annotated laptop sleeve is 20 mm thick: below waste mode's
            # 25 mm noise gate, but above validation mode's separate 10 mm gate.
            depth[mask] = 1.98

            for timestamp in (100.0, 101.0, 102.0):
                analysis = pipeline.process_frame(
                    frame, depth_m=depth, intrinsics=camera, timestamp=timestamp,
                )

            state = pipeline.state()
            self.assertEqual(state["operating_mode"], "geometry_validation")
            # Recording is no longer tied to the mode. Validation runs used to
            # disable the ledger, which is how ordinary runs ended up with an
            # empty CSV; geometry validation now measures AND records, while
            # still not auto-depositing into the waste plant ledger.
            self.assertTrue(state["waste_ledger_enabled"])
            self.assertFalse(state["auto_deposit"])
            self.assertEqual(state["session_seen"]["total"], 1)
            self.assertEqual(analysis.detections[0].accepted_class, "measurement_object")
            self.assertIsNotNone(analysis.detections[0].footprint_length_mm)
            # The waste-plant ledger stays untouched: depositing into the
            # waste plant is a waste-mode concept and has not changed.
            self.assertEqual(state["plant"]["observed_count"], 0)
            self.assertEqual(state["plant"]["deposited_count"], 0)
            # What DID change: the measurement record no longer depends on the
            # mode, so this validation run has a persisted, downloadable row
            # instead of the empty CSV it used to produce.
            self.assertTrue(state["waste_ledger_enabled"])
            self.assertEqual(state["csv_persistence"]["events_persisted"], 1)
            self.assertTrue(state["csv_persistence"]["healthy"])
            row = pipeline.event_log.rows()[0]
            self.assertEqual(row["operating_mode"], "geometry_validation")
            self.assertTrue(row["event_id"])
            self.assertTrue(float(row["estimated_litres"]) > 0)

    def test_stable_bag_and_box_are_counted_once_and_color_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            detector = AdjustableDetector()
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=20,
                tracker_confirm_frames=2,
                settle_frames=3,
                volume_window_frames=3,
                color_waste_streams={"red": "red plant stream", "blue": "blue plant stream"},
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((70, 70, 3), dtype=np.uint8)
            baseline = np.full((70, 70), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, baseline, camera)

            bag_mask = np.zeros((70, 70), dtype=bool)
            bag_mask[5:25, 5:25] = True
            detector.detections = [Detection("garbage bag", 0.8, (5, 5, 25, 25), bag_mask)]
            bag_frame = empty.copy()
            bag_frame[bag_mask] = (0, 0, 230)
            bag_depth = baseline.copy()
            bag_depth[bag_mask] = 1.7
            for index in range(3):
                pipeline.process_frame(bag_frame, depth_m=bag_depth, intrinsics=camera, timestamp=100 + index)
            first = pipeline.state()["plant"]
            self.assertEqual(first["observed_bags"], 1)
            self.assertEqual(first["deposited_bags"], 1)
            self.assertEqual(first["history"][0]["color"], "red")
            self.assertEqual(first["history"][0]["waste_stream"], "red plant stream")

            box_mask = np.zeros((70, 70), dtype=bool)
            box_mask[40:60, 40:60] = True
            detector.detections = [Detection("cardboard box", 0.85, (40, 40, 60, 60), box_mask)]
            box_frame = bag_frame.copy()
            box_frame[box_mask] = (230, 0, 0)
            box_depth = bag_depth.copy()
            box_depth[box_mask] = 1.6
            for index in range(3):
                pipeline.process_frame(box_frame, depth_m=box_depth, intrinsics=camera, timestamp=200 + index)
            final = pipeline.state()["plant"]
            self.assertEqual(final["observed_count"], 2)
            self.assertEqual(final["deposited_bags"], 1)
            self.assertEqual(final["deposited_boxes"], 1)
            self.assertEqual(len(final["history"]), 2)
            self.assertEqual({item["color"] for item in final["history"]}, {"red", "blue"})
            self.assertGreater(final["cumulative_volume_l"], 0)

    def test_unstable_volume_is_not_prematurely_marked_deposited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            detector = AdjustableDetector()
            config = AppConfig(
                results_dir=Path(temporary), enable_monocular_depth=False, roi=(0, 0, 1, 1),
                min_component_pixels=10, tracker_confirm_frames=1, settle_frames=3,
                settle_volume_tolerance=0.02,
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((40, 40, 3), dtype=np.uint8)
            baseline = np.full((40, 40), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, baseline, camera)
            mask = np.zeros((40, 40), dtype=bool)
            mask[5:25, 5:25] = True
            detector.detections = [Detection("garbage bag", 0.9, (5, 5, 25, 25), mask)]
            frame = empty.copy()
            frame[mask] = 200
            for distance in (1.8, 1.4, 1.65):
                depth = baseline.copy()
                depth[mask] = distance
                pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            summary = pipeline.state()["plant"]
            self.assertEqual(summary["observed_bags"], 1)
            self.assertEqual(summary["deposited_bags"], 0)

    def test_resetting_live_tracks_does_not_erase_plant_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            detector = AdjustableDetector()
            config = AppConfig(
                results_dir=Path(temporary), enable_monocular_depth=False, roi=(0, 0, 1, 1),
                min_component_pixels=10, tracker_confirm_frames=1, auto_deposit=False,
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((30, 30, 3), dtype=np.uint8)
            baseline = np.full((30, 30), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, baseline, camera)
            mask = np.zeros((30, 30), dtype=bool)
            mask[5:20, 5:20] = True
            detector.detections = [Detection("garbage bag", 0.9, (5, 5, 20, 20), mask)]
            frame = empty.copy()
            frame[mask] = 180
            depth = baseline.copy()
            depth[mask] = 1.7
            pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            pipeline.reset_live_tracking()
            self.assertEqual(pipeline.state()["plant"]["observed_bags"], 1)
            pipeline.process_frame(frame, depth_m=depth, intrinsics=camera)
            self.assertEqual(pipeline.state()["plant"]["observed_bags"], 2)

    def test_a_phantom_depth_silhouette_is_tracked_live_but_never_reaches_the_ledger(self) -> None:
        # Real-hardware round-6 screenshots showed a "garbage bag (depth
        # silhouette)" detection -- with zero neural confirmation, from a
        # changed-scene region the detector never actually labeled -- get
        # DEPOSITED into durable history at 87-118 L, alongside a real ~37 L
        # object. A phantom detection may still be tracked and shown live
        # (that mechanism exists to bridge brief detector dropout for an
        # already-confirmed real object), but it must never be written to the
        # durable waste-plant ledger: it carries no evidence it is waste at
        # all, only that *something* in the scene changed.
        with tempfile.TemporaryDirectory() as temporary:
            detector = AdjustableDetector()  # never produces a neural detection
            config = AppConfig(
                results_dir=Path(temporary), enable_monocular_depth=False, roi=(0, 0, 1, 1),
                min_component_pixels=20, tracker_confirm_frames=1, settle_frames=2,
                volume_window_frames=2, allow_unclassified_foreground=True, bag_only=True,
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((60, 60, 3), dtype=np.uint8)
            baseline = np.full((60, 60), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, baseline, camera)

            changed_mask = np.zeros((60, 60), dtype=bool)
            changed_mask[10:40, 10:40] = True
            frame = empty.copy()
            frame[changed_mask] = (60, 140, 210)
            depth = baseline.copy()
            depth[changed_mask] = 1.7  # 0.3 m of "height" -- within the plausible range

            for index in range(4):
                analysis = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=index)

            self.assertTrue(analysis.detections)
            self.assertEqual(analysis.detections[0].source, "fixed-bin-depth-silhouette")
            self.assertEqual(analysis.detections[0].label, "garbage bag (depth silhouette)")

            plant = pipeline.state()["plant"]
            self.assertEqual(plant["observed_count"], 0)
            self.assertEqual(plant["deposited_count"], 0)
            self.assertEqual(plant["cumulative_volume_l"], 0.0)
            # It is still tracked and visible live -- this is not a "nothing
            # was ever detected" gap, only a "never written to the ledger" one.
            self.assertGreaterEqual(pipeline.tracker.total_count, 1)


if __name__ == "__main__":
    unittest.main()
