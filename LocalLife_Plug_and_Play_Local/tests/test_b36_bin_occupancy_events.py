"""B36: what the bin gained, not what the bag is.

The thesis measurement is the change in occupied bin volume across a deposit,
and until now the system reported something else: the litres of whichever bag
the detector was looking at. In a bin that already holds waste those are not
the same quantity. A bag dropped into a gap raises the surface by less than its
own size; one that compresses the pile can lower it. This suite pins the
difference down.

"Occupied volume" here means the volume enclosed below the visible waste
surface relative to the bin floor, from one camera's viewpoint. It cannot see
air pockets beneath the surface, and the tests assert that limitation is stated
rather than hidden.

Everything below is a state machine driven by synthetic readings. No camera has
run against it, and nothing here is evidence about real accuracy.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from locallife_cloud.bin_occupancy import (
    CAMERA_MOVED,
    DEPOSIT_IN_PROGRESS,
    FINALIZED,
    NO_ABSOLUTE,
    NO_OCCUPANCY,
    SCENE_UNSETTLED,
    SETTLING,
    STABLE_POST,
    STABLE_PRE,
    VIEW_OBSTRUCTED,
    BinOccupancyTracker,
    DepositObservation,
    OccupancyReading,
    pair_events,
)
from locallife_cloud.logitech_bin import IMPLAUSIBLE_ASPECT, implausible_aspect_reason
from locallife_cloud.logitech_geometry import GeometryStabiliser

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "c3add1d6a4b6f4e6ccefc64d47ca8cecc3e65706"


def _reading(litres: float | None, *, valid: float = 0.9, absolute: bool = True,
             reason: str | None = None) -> OccupancyReading:
    return OccupancyReading(
        litres=litres, valid_fraction=valid, absolute=absolute, reason=reason,
    )


def _run(tracker: BinOccupancyTracker, frames, **shared):
    """Feed frames as (litres, changed_fraction, tracked) and collect events."""
    events = []
    for index, (litres, changed, tracked) in enumerate(frames):
        event = tracker.observe(DepositObservation(
            reading=_reading(litres, **{k: v for k, v in shared.items()
                                        if k in {"valid", "absolute", "reason"}}),
            changed_fraction=changed, tracked_objects=tracked,
            timestamp=1000.0 + index,
            obstructed=shared.get("obstructed", False),
            label=shared.get("label"), color=shared.get("color"),
            track_id=shared.get("track_id"), envelope_mm=shared.get("envelope_mm"),
        ))
        if event is not None:
            events.append(event)
    return events


def _settle(litres: float, frames: int = 8):
    return [(litres, 0.0, 0)] * frames


def _deposit(litres_after: float, *, arriving: int = 3):
    return [(litres_after, 0.30, 1)] * arriving


class NoDepositTests(unittest.TestCase):
    def test_a_quiet_bin_finalises_nothing(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        events = _run(tracker, _settle(12.0, 20))
        self.assertEqual(events, [])
        self.assertEqual(tracker.state, STABLE_PRE)
        self.assertAlmostEqual(tracker.committed_before.litres, 12.0)

    def test_a_quiet_bin_reports_no_pending_reason(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        _run(tracker, _settle(12.0, 6))
        self.assertIsNone(tracker.pending_reason)


class OneDepositTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = BinOccupancyTracker("realsense", required_stable_frames=3)

    def test_one_deposit_finalises_exactly_one_event(self) -> None:
        events = _run(
            self.tracker,
            _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10),
            label="filled plastic waste bag", color="red", track_id=7,
            envelope_mm=(300.0, 200.0, 150.0),
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event.occupied_before_l, 10.0)
        self.assertAlmostEqual(event.occupied_after_l, 14.5)
        self.assertAlmostEqual(event.delta_occupancy_l, 4.5)
        self.assertEqual(event.status, "finalized")
        self.assertEqual(event.label, "filled plastic waste bag")
        self.assertEqual(event.color, "red")
        self.assertEqual(event.track_id, 7)

    def test_the_bag_envelope_is_reported_separately_from_the_change(self) -> None:
        events = _run(
            self.tracker, _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10),
            label="bag", track_id=1, envelope_mm=(300.0, 200.0, 150.0),
        )
        event = events[0]
        # 300 x 200 x 150 mm bounds 9 L. The bin gained 4.5 L. They are
        # different measurements and the record keeps them apart.
        self.assertEqual(event.envelope_mm, (300.0, 200.0, 150.0))
        self.assertAlmostEqual(event.delta_occupancy_l, 4.5)
        self.assertNotAlmostEqual(event.delta_occupancy_l, 9.0)

    def test_the_before_reading_is_frozen_when_the_deposit_starts(self) -> None:
        """The frames of the bag falling must not become the "before"."""
        frames = _settle(10.0, 6) + [
            (11.0, 0.30, 1), (13.0, 0.35, 1), (14.0, 0.30, 1),   # mid-flight
        ] + _settle(14.5, 10)
        events = _run(self.tracker, frames)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].occupied_before_l, 10.0)

    def test_the_committed_state_advances_to_the_finalised_after(self) -> None:
        _run(self.tracker, _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10))
        self.assertAlmostEqual(self.tracker.committed_before.litres, 14.5)

    def test_the_record_states_what_occupied_volume_means(self) -> None:
        events = _run(self.tracker, _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10))
        text = events[0].to_dict()["occupancy_definition"]
        self.assertIn("below the visible waste surface", text)
        self.assertIn("hidden voids", text)


class SequentialDepositTests(unittest.TestCase):
    def test_two_deposits_give_two_events_and_no_double_count(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        events = _run(
            tracker,
            _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10)
            + _deposit(14.5) + _settle(19.0, 10),
        )
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].delta_occupancy_l, 4.5)
        # The second event measures against the bin as it now is, not against
        # the empty bin: the first bag is not counted twice.
        self.assertAlmostEqual(events[1].occupied_before_l, 14.5)
        self.assertAlmostEqual(events[1].delta_occupancy_l, 4.5)
        self.assertNotEqual(events[0].event_id, events[1].event_id)

    def test_a_second_bag_arriving_mid_event_does_not_split_it(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        frames = (
            _settle(10.0, 6) + _deposit(10.0) + [(14.0, 0.0, 0)]
            + _deposit(14.0) + _settle(19.0, 10)
        )
        events = _run(tracker, frames)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].occupied_before_l, 10.0)
        self.assertAlmostEqual(events[0].occupied_after_l, 19.0)


class ExistingContentsTests(unittest.TestCase):
    def test_a_bin_that_starts_full_is_not_a_deposit(self) -> None:
        tracker = BinOccupancyTracker("logitech", required_stable_frames=3)
        events = _run(tracker, _settle(31.5, 20))
        self.assertEqual(events, [])
        self.assertAlmostEqual(tracker.committed_before.litres, 31.5)

    def test_the_first_deposit_into_a_full_bin_measures_only_the_change(self) -> None:
        tracker = BinOccupancyTracker("logitech", required_stable_frames=3)
        events = _run(tracker, _settle(31.5, 8) + _deposit(31.5) + _settle(34.0, 10))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].occupied_before_l, 31.5)
        self.assertAlmostEqual(events[0].delta_occupancy_l, 2.5)


class CompressionTests(unittest.TestCase):
    def test_a_negative_change_is_kept_as_measured(self) -> None:
        """Waste compressed. That is an observation, not an error to clamp."""
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        events = _run(tracker, _settle(20.0, 6) + _deposit(20.0) + _settle(18.4, 10))
        self.assertEqual(len(events), 1)
        self.assertLess(events[0].delta_occupancy_l, 0.0)
        self.assertEqual(events[0].status, "finalized")

    def test_a_near_zero_change_is_not_promoted_to_a_bag_volume(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        events = _run(
            tracker, _settle(20.0, 6) + _deposit(20.0) + _settle(20.1, 10),
            label="bag", track_id=2, envelope_mm=(300.0, 200.0, 150.0),
        )
        self.assertLess(abs(events[0].delta_occupancy_l), 0.5)


class UnavailableAndPendingTests(unittest.TestCase):
    def test_no_valid_surface_is_named_not_invented(self) -> None:
        tracker = BinOccupancyTracker("logitech", required_stable_frames=3)
        for index in range(8):
            tracker.observe(DepositObservation(
                reading=OccupancyReading(litres=None, valid_fraction=0.0,
                                         reason=NO_OCCUPANCY),
                timestamp=1000.0 + index,
            ))
        self.assertEqual(tracker.pending_reason, NO_OCCUPANCY)
        self.assertIsNone(tracker.committed_before)

    def test_sparse_depth_is_not_used_as_an_occupancy_figure(self) -> None:
        tracker = BinOccupancyTracker("logitech", required_stable_frames=3)
        tracker.observe(DepositObservation(
            reading=OccupancyReading(litres=9.0, valid_fraction=0.10, absolute=True),
            timestamp=1000.0,
        ))
        self.assertIsNone(tracker.committed_before)

    def test_an_obstructed_deposit_stays_pending_with_its_reason(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        _run(tracker, _settle(10.0, 6))
        events = _run(tracker, _deposit(10.0) + _settle(14.5, 10), obstructed=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "pending")
        self.assertEqual(events[0].reason, VIEW_OBSTRUCTED)
        self.assertIsNone(events[0].delta_occupancy_l)

    def test_a_moved_camera_discards_the_comparison(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        _run(tracker, _settle(10.0, 6))
        tracker.observe(DepositObservation(
            reading=_reading(10.0), geometry_moved=True, timestamp=2000.0,
        ))
        self.assertEqual(tracker.state, STABLE_PRE)
        self.assertEqual(tracker.pending_reason, CAMERA_MOVED)

    def test_an_unsettled_scene_says_so(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=4)
        _run(tracker, _settle(10.0, 6))
        _run(tracker, _deposit(10.0) + [(14.0, 0.0, 0), (19.0, 0.0, 0), (11.0, 0.0, 0)])
        self.assertEqual(tracker.pending_reason, SCENE_UNSETTLED)

    def test_without_a_bin_profile_only_the_change_is_claimed(self) -> None:
        tracker = BinOccupancyTracker("logitech", required_stable_frames=3)
        events = _run(
            tracker, _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10),
            absolute=False,
        )
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].absolute)
        self.assertEqual(events[0].reason, NO_ABSOLUTE)
        # The change is still measured; only the absolute figure is withheld.
        self.assertAlmostEqual(events[0].delta_occupancy_l, 4.5)


class StateSequenceTests(unittest.TestCase):
    def test_the_machine_walks_the_documented_sequence(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=2)
        seen = []
        for litres, changed, tracked in (
            _settle(10.0, 3) + _deposit(10.0, arriving=2) + _settle(14.5, 6)
        ):
            tracker.observe(DepositObservation(
                reading=_reading(litres), changed_fraction=changed,
                tracked_objects=tracked, timestamp=1000.0,
            ))
            if not seen or seen[-1] != tracker.state:
                seen.append(tracker.state)
        self.assertEqual(
            seen, [STABLE_PRE, DEPOSIT_IN_PROGRESS, SETTLING, STABLE_POST,
                   FINALIZED, STABLE_PRE],
        )


class PairingTests(unittest.TestCase):
    def test_the_two_cameras_are_paired_without_sharing_values(self) -> None:
        left = BinOccupancyTracker("realsense", required_stable_frames=3)
        right = BinOccupancyTracker("logitech", required_stable_frames=3)
        _run(left, _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10))
        _run(right, _settle(12.0, 6) + _deposit(12.0) + _settle(18.0, 10))
        pairs = pair_events(left.events, right.events, window_s=60.0)
        self.assertEqual(len(pairs), 1)
        first, second = pairs[0]
        # Paired for comparison; each keeps the litres its own camera measured.
        self.assertAlmostEqual(first.delta_occupancy_l, 4.5)
        self.assertAlmostEqual(second.delta_occupancy_l, 6.0)
        self.assertEqual(first.camera, "realsense")
        self.assertEqual(second.camera, "logitech")

    def test_event_ids_are_unique_within_a_camera(self) -> None:
        tracker = BinOccupancyTracker("realsense", required_stable_frames=3)
        _run(
            tracker,
            _settle(10.0, 6) + _deposit(10.0) + _settle(14.5, 10)
            + _deposit(14.5) + _settle(19.0, 10),
        )
        ids = [event.event_id for event in tracker.events]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(item.startswith("realsense-occ-") for item in ids))


class FinalisedValuesAgreeTests(unittest.TestCase):
    """Overlay, table and export must show one answer, not three."""

    def test_a_settled_track_stops_changing_size(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        for _ in range(4):
            first = stabiliser.update(1, length_mm=329.0, width_mm=214.0,
                                      height_mm=99.0, volume_l=5.576)
        self.assertTrue(first.settled)
        # The next frame drifts, as the field screenshots show: 329 x 214 x 99
        # on the overlay beside 329 x 212 x 91 in the row.
        later = stabiliser.update(1, length_mm=329.0, width_mm=212.0,
                                  height_mm=91.0, volume_l=5.4)
        self.assertEqual(
            (later.length_mm, later.width_mm, later.height_mm),
            (first.length_mm, first.width_mm, first.height_mm),
        )

    def test_a_new_object_on_a_reused_id_is_not_given_the_old_size(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        for _ in range(4):
            stabiliser.update(1, length_mm=329.0, width_mm=214.0, height_mm=99.0)
        stabiliser.forget(1)
        fresh = stabiliser.update(1, length_mm=120.0, width_mm=80.0, height_mm=60.0)
        self.assertEqual(fresh.length_mm, 120.0)


class ImplausibleShapeTests(unittest.TestCase):
    def test_the_sliver_from_the_overlay_is_flagged(self) -> None:
        # 163 x 71 x 572 mm: a column three and a half times taller than its
        # own longest ground side. A mask that climbed the bin wall.
        self.assertEqual(
            implausible_aspect_reason(163.0, 71.0, 572.0), IMPLAUSIBLE_ASPECT,
        )

    def test_the_history_row_is_honestly_beyond_this_rule(self) -> None:
        """415 x 225 x 537 mm is wrong, and shape alone cannot prove it.

        Its height is only 1.3 times its longest ground side, which is an
        ordinary upright shape. What makes it wrong is knowing the bag is about
        150 mm high -- outside knowledge this rule does not have. Recorded here
        so the limit of the check is written down rather than implied.
        """
        self.assertIsNone(implausible_aspect_reason(415.0, 225.0, 537.0))

    def test_an_ordinary_bag_is_not_flagged(self) -> None:
        self.assertIsNone(implausible_aspect_reason(300.0, 200.0, 150.0))
        self.assertIsNone(implausible_aspect_reason(329.0, 214.0, 99.0))

    def test_an_upright_bottle_is_not_flagged(self) -> None:
        self.assertIsNone(implausible_aspect_reason(75.0, 75.0, 203.0))

    def test_a_missing_dimension_is_not_a_verdict(self) -> None:
        self.assertIsNone(implausible_aspect_reason(None, 200.0, 150.0))


class ProtectedSurfacesTests(unittest.TestCase):
    def test_csv_history_export_cloud_and_launcher_are_unchanged(self) -> None:
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
            "Start-LocalLife-Demo.ps1",
        ]
        result = subprocess.run(
            ["git", "diff", "--name-only", SOURCE_SHA, "--", *protected],
            capture_output=True, text=True, cwd=PROJECT.parent, timeout=120,
        )
        if result.returncode != 0:
            self.skipTest("git or the source commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")

    def test_the_occupancy_module_reads_no_camera_specific_code(self) -> None:
        import ast

        tree = ast.parse((PROJECT / "locallife_cloud" / "bin_occupancy.py").read_text())
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
