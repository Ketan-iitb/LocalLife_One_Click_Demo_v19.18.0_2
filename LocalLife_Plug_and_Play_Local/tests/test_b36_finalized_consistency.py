"""B36: one deposit, one answer, wherever it is read.

The field screenshots show one object reported three ways at once -- a live
overlay drawn with one triplet, a table row showing another, a history row
holding a third. All three are honest readings of the same object; they are
readings of *different frames*. The overlay is painted from whatever the
detection holds when the frame is drawn, the history row from whatever it held
when the event was finalised, and the pipeline keeps re-measuring in between.

A thesis cannot cite a number that changes depending on where it was read, so
finalisation now produces a record and the record is the answer.

The second suite here covers the duplicate the screenshots also show: one red
bag wearing "#10 test object (filled plastic waste bag)" and "#11 unclassified
object [red]" at the same time -- two ids and two rows for one bag.

Synthetic. No camera has run against this branch.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from locallife_cloud.duplicate_detections import (
    drop_unclassified_duplicates,
    is_unclassified,
)
from locallife_cloud.finalized_record import FROZEN_FIELDS, FinalizedRegistry
from locallife_cloud.types import Detection

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "3645d286c97e23bd399691db4743c3d34460dab9"


def _bag(track_id: int = 1, **values) -> Detection:
    detection = Detection("filled plastic waste bag", 0.9, (10, 10, 210, 160))
    detection.track_id = track_id
    detection.footprint_length_mm = values.get("length", 329.0)
    detection.footprint_width_mm = values.get("width", 214.0)
    detection.physical_height_mm = values.get("height", 99.0)
    detection.height_above_baseline_cm = values.get("height_cm", 9.9)
    detection.realsense_volume_l = values.get("volume", 5.576)
    detection.measurement_quality = values.get("quality", "measured")
    return detection


class FinalisedValuesAreImmutableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = FinalizedRegistry("realsense")
        self.detection = _bag()
        self.record = self.registry.remember(
            self.detection, event_id="realsense-0001", timestamp=1000.0,
            status="accepted", permanent_id=77,
        )

    def test_the_record_keeps_the_values_it_was_written_with(self) -> None:
        self.assertEqual(self.record.event_id, "realsense-0001")
        self.assertAlmostEqual(self.record.values["footprint_length_mm"], 329.0)
        self.assertAlmostEqual(self.record.values["physical_height_mm"], 99.0)

    def test_a_later_frame_republishes_the_record_not_its_own_reading(self) -> None:
        # The next frame measures the same bag slightly differently, which is
        # exactly what produced 329 x 214 x 99 on the overlay beside
        # 329 x 212 x 91 in the row.
        self.detection.footprint_width_mm = 212.0
        self.detection.physical_height_mm = 91.0
        self.detection.realsense_volume_l = 5.41

        self.registry.republish(self.detection, 77)

        self.assertAlmostEqual(self.detection.footprint_width_mm, 214.0)
        self.assertAlmostEqual(self.detection.physical_height_mm, 99.0)
        self.assertAlmostEqual(self.detection.realsense_volume_l, 5.576)
        self.assertEqual(self.detection.finalized_event_id, "realsense-0001")

    def test_overlay_table_and_export_quote_the_same_record(self) -> None:
        self.detection.physical_height_mm = 91.0
        self.registry.republish(self.detection, 77)
        overlay = self.detection.to_dict()
        table = self.detection.to_dict()
        exported = self.record.to_dict()
        # The overlay and the table are both rendered from the detection; the
        # export is rendered from the record. All three have to agree.
        self.assertEqual(overlay["dimensions_mm"], table["dimensions_mm"])
        for published, recorded in (
            ("footprint_length", "footprint_length_mm"),
            ("footprint_width", "footprint_width_mm"),
            ("height", "physical_height_mm"),
        ):
            with self.subTest(field=recorded):
                self.assertAlmostEqual(
                    overlay["dimensions_mm"][published], exported[recorded],
                )
        self.assertEqual(overlay["finalized_event_id"], exported["event_id"])

    def test_finalising_twice_does_not_rewrite_what_was_published(self) -> None:
        self.detection.physical_height_mm = 537.0
        again = self.registry.remember(
            self.detection, event_id="realsense-0002", timestamp=1010.0,
            status="accepted", permanent_id=77,
        )
        self.assertEqual(again.event_id, "realsense-0001")
        self.assertAlmostEqual(again.values["physical_height_mm"], 99.0)

    def test_an_unfinalised_track_is_left_alone(self) -> None:
        fresh = _bag(track_id=9, height=120.0)
        self.assertIsNone(self.registry.republish(fresh, None))
        self.assertAlmostEqual(fresh.physical_height_mm, 120.0)
        self.assertIsNone(fresh.finalized_event_id)

    def test_every_published_measurement_field_is_covered(self) -> None:
        for name in ("footprint_length_mm", "footprint_width_mm", "physical_height_mm",
                     "monocular_volume_l", "realsense_volume_l"):
            with self.subTest(field=name):
                self.assertIn(name, FROZEN_FIELDS)


class ReusedTrackIdTests(unittest.TestCase):
    def test_a_new_object_does_not_inherit_a_finished_deposit(self) -> None:
        registry = FinalizedRegistry("logitech")
        first = _bag(track_id=3)
        registry.remember(first, event_id="logitech-0001", timestamp=1.0,
                          status="accepted", permanent_id=None)
        registry.release(3)

        second = _bag(track_id=3, length=120.0, width=80.0, height=60.0)
        self.assertIsNone(registry.republish(second, None))
        self.assertAlmostEqual(second.footprint_length_mm, 120.0)

    def test_a_permanent_identity_survives_a_track_number_change(self) -> None:
        registry = FinalizedRegistry("logitech")
        first = _bag(track_id=3)
        registry.remember(first, event_id="logitech-0001", timestamp=1.0,
                          status="accepted", permanent_id=42)
        registry.release(3)
        renumbered = _bag(track_id=8, length=999.0)
        registry.republish(renumbered, 42)
        self.assertAlmostEqual(renumbered.footprint_length_mm, 329.0)


class DuplicateDetectionTests(unittest.TestCase):
    """The red bag that carried two boxes at once."""

    def _unclassified(self, box) -> Detection:
        item = Detection("unclassified object", 0.4, box)
        item.color = "red"
        return item

    def test_an_unclassified_box_over_a_classified_bag_is_dropped(self) -> None:
        bag = Detection("filled plastic waste bag", 0.9, (100, 100, 400, 500))
        ghost = self._unclassified((120, 120, 380, 480))
        kept, dropped = drop_unclassified_duplicates([bag, ghost])
        self.assertEqual(dropped, 1)
        self.assertEqual([item.label for item in kept], ["filled plastic waste bag"])

    def test_the_classified_detection_is_always_the_one_kept(self) -> None:
        bag = Detection("filled plastic waste bag", 0.9, (100, 100, 400, 500))
        ghost = self._unclassified((100, 100, 400, 500))
        kept, _ = drop_unclassified_duplicates([ghost, bag])
        self.assertEqual(len(kept), 1)
        self.assertFalse(is_unclassified(kept[0].label))

    def test_a_separate_unclassified_region_is_untouched(self) -> None:
        bag = Detection("filled plastic waste bag", 0.9, (100, 100, 400, 500))
        elsewhere = self._unclassified((600, 100, 750, 300))
        kept, dropped = drop_unclassified_duplicates([bag, elsewhere])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)

    def test_two_classified_bags_that_touch_are_both_kept(self) -> None:
        first = Detection("filled plastic waste bag", 0.9, (100, 100, 400, 500))
        second = Detection("plastic garbage bag", 0.85, (380, 120, 660, 520))
        kept, dropped = drop_unclassified_duplicates([first, second])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)

    def test_the_bag_count_cannot_fall(self) -> None:
        bags = [
            Detection("filled plastic waste bag", 0.9, (100, 100, 400, 500)),
            Detection("plastic garbage bag", 0.8, (420, 100, 700, 500)),
        ]
        ghosts = [self._unclassified((110, 110, 390, 490)),
                  self._unclassified((430, 110, 690, 490))]
        kept, dropped = drop_unclassified_duplicates(bags + ghosts)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(kept), len(bags))

    def test_nothing_is_dropped_when_no_class_was_recognised(self) -> None:
        ghosts = [self._unclassified((100, 100, 400, 500)),
                  self._unclassified((120, 120, 380, 480))]
        kept, dropped = drop_unclassified_duplicates(ghosts)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)


class ProtectedSurfacesTests(unittest.TestCase):
    def test_export_history_cloud_launcher_and_realsense_core_are_unchanged(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/storage.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/realsense.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/depth.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/heightmap_volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/material.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/sorting_rules.py",
            "Start-LocalLife-Demo.ps1",
        ]
        result = subprocess.run(
            ["git", "diff", "--name-only", SOURCE_SHA, "--", *protected],
            capture_output=True, text=True, cwd=PROJECT.parent, timeout=120,
        )
        if result.returncode != 0:
            self.skipTest("git or the source commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")

    def test_the_new_modules_are_camera_agnostic(self) -> None:
        import ast

        for name in ("finalized_record.py", "duplicate_detections.py"):
            with self.subTest(module=name):
                tree = ast.parse((PROJECT / "locallife_cloud" / name).read_text())
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                    elif isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                self.assertFalse([
                    item for item in imported
                    if any(word in item.lower() for word in ("realsense", "logitech"))
                ])


if __name__ == "__main__":
    unittest.main()
