"""50 sequential objects through one pipeline instance, no restart.

Reproduces the reported field failure -- the system detects the first handful of
test objects correctly, then starts tracking background, mislabelling colour and
showing bin occupancy that only ever rises -- and locks in the fix.

The loop is the playbook's own soak procedure (verify empty, present one object,
wait for a verdict, remove it, verify empty again), driven against synthetic
frames so it runs in CI without the rig. What it cannot cover is anything that
needs real optics: sensor noise, auto-white-balance drift, USB bandwidth, GPU
memory. Those stay hardware-verification items.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection

SIZE = 90
BASELINE_DEPTH_M = 2.0
PRESENTATIONS = 50

# label, BGR, thickness (m), side (px). Size and thickness belong to the
# catalogue entry, not the presentation index, so presentation N and N+6 are
# genuinely the same physical object seen at a different position -- which is
# what makes a drifting reference detectable as a changed reading.
CATALOGUE = [
    ("filled plastic waste bag", (40, 190, 40), 0.14, 18),
    ("cardboard shipping box", (60, 110, 165), 0.16, 22),
    ("black trash bag", (25, 25, 25), 0.13, 16),
    ("polythene waste bag", (200, 200, 200), 0.11, 20),
    ("kraft paper bag", (110, 170, 205), 0.15, 17),
    ("large garbage bag", (180, 60, 60), 0.12, 24),
]


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


def _presentation(index: int):
    """Label, frame, depth, mask and box for presentation `index`."""
    label, colour, thickness, side = CATALOGUE[index % len(CATALOGUE)]
    top = 20 + (index * 7) % 30
    left = 20 + (index * 11) % 30
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top : top + side, left : left + side] = True
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[mask] = colour
    depth = np.full((SIZE, SIZE), BASELINE_DEPTH_M, dtype=np.float32)
    depth[mask] = BASELINE_DEPTH_M - thickness
    return label, frame, depth, mask, (left, top, left + side, top + side)


def _build(directory, **overrides):
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
    empty_frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    empty_depth = np.full((SIZE, SIZE), BASELINE_DEPTH_M, dtype=np.float32)
    camera = CameraIntrinsics(fx=100, fy=100, ppx=SIZE / 2, ppy=SIZE / 2)
    pipeline.set_baseline(empty_frame, empty_depth, camera)
    return pipeline, detector, empty_frame, empty_depth, camera


def _soak(pipeline, detector, empty_frame, empty_depth, camera, presentations=PRESENTATIONS):
    """Run the sequential procedure; return one record per presentation."""
    records = []
    clock = 100.0
    for index in range(presentations):
        label, frame, depth, mask, box = _presentation(index)
        detector.detections = [Detection(label, 0.9, box, mask)]
        analysis = None
        for _ in range(3):
            clock += 1.0
            analysis = pipeline.process_frame(
                frame, depth_m=depth, intrinsics=camera, timestamp=clock,
            )
        measured = [
            item for item in analysis.detections if item.realsense_volume_l is not None
        ]
        records.append({
            "index": index,
            "label": label,
            "detections": len(analysis.detections),
            "measured": len(measured),
            "volume_l": measured[0].realsense_volume_l if measured else None,
            "color": measured[0].color if measured else None,
            "track_id": measured[0].track_id if measured else None,
            "bin_occupancy_l": (
                None if analysis.bin_total is None else analysis.bin_total.liters
            ),
        })

        # Remove the object and let the scene settle back to empty.
        detector.detections = []
        for _ in range(6):
            clock += 1.0
            empty_analysis = pipeline.process_frame(
                empty_frame, depth_m=empty_depth, intrinsics=camera, timestamp=clock,
            )
        records[-1]["tracks_after_removal"] = len(pipeline.tracker.tracks)
        records[-1]["idle_bin_occupancy_l"] = (
            None if empty_analysis.bin_total is None else empty_analysis.bin_total.liters
        )
    return records


def _history_sizes(pipeline):
    return {
        "volume": len(pipeline._volume_history),
        "color": len(pipeline._color_history),
        "material": len(pipeline._material_history),
        "box": len(pipeline._box_measurement_history),
        "box_frames": len(pipeline._box_frames_considered),
        "material_frames": len(pipeline._material_frame_counts),
        "bin_before": len(pipeline._bin_total_before_track),
    }


class SequentialSoakTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.pipeline, self.detector, self.frame, self.depth, self.camera = _build(
            self._directory.name
        )
        self.records = _soak(
            self.pipeline, self.detector, self.frame, self.depth, self.camera
        )

    def test_late_objects_are_measured_as_reliably_as_early_ones(self) -> None:
        early = [record["measured"] for record in self.records[:10]]
        late = [record["measured"] for record in self.records[40:]]
        self.assertEqual(sum(early), 10, "early objects were already unreliable")
        self.assertEqual(
            sum(late), 10,
            f"objects 41-50 degraded: {[r['index'] for r in self.records[40:] if not r['measured']]}",
        )

    def test_every_presentation_produced_exactly_one_measured_object(self) -> None:
        missed = [record["index"] for record in self.records if record["measured"] != 1]
        self.assertEqual(missed, [])

    def test_measured_volume_does_not_drift_across_the_session(self) -> None:
        # Presentation N and N+len(CATALOGUE) are the same object at a
        # different position, so a drifting reference shows up as the same
        # object reading differently late in the run.
        by_label: dict[str, list[float]] = {}
        for record in self.records:
            by_label.setdefault(record["label"], []).append(record["volume_l"])
        for label, volumes in by_label.items():
            with self.subTest(label=label):
                early, late = volumes[0], volumes[-1]
                self.assertLess(
                    abs(late - early) / early, 0.35,
                    f"{label} read {early:.2f} L first and {late:.2f} L last",
                )

    def test_tracks_are_released_once_the_scene_is_empty(self) -> None:
        leaked = [
            record["index"] for record in self.records if record["tracks_after_removal"]
        ]
        self.assertEqual(leaked, [])
        self.assertEqual(len(self.pipeline.tracker.tracks), 0)

    def test_per_track_state_is_bounded_by_live_tracks_not_by_objects_seen(self) -> None:
        sizes = _history_sizes(self.pipeline)
        for name, size in sizes.items():
            with self.subTest(buffer=name):
                self.assertLessEqual(
                    size, 2, f"{name} still holds {size} entries after {PRESENTATIONS} objects",
                )

    def test_track_ids_advance_but_never_repeat(self) -> None:
        ids = [record["track_id"] for record in self.records]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, sorted(ids))

    def test_no_object_inherits_the_previous_object_colour(self) -> None:
        # Each catalogue entry is a distinct hue, so the same entry must report
        # the same colour every time it comes round; a change means a new track
        # read a dead track's votes.
        names = [record["color"] for record in self.records]
        for index in range(len(CATALOGUE), len(names)):
            with self.subTest(index=index):
                self.assertEqual(
                    names[index], names[index - len(CATALOGUE)],
                    "the same object reported a different colour later in the run",
                )

    def test_an_emptied_scene_reports_no_occupancy(self) -> None:
        # Bin occupancy must fall back to nothing once the object is gone; a
        # figure that only ever rises is the accumulation bug.
        residual = [
            record["index"] for record in self.records
            if record["idle_bin_occupancy_l"] not in (None, 0.0)
        ]
        self.assertEqual(residual, [])

    def test_occupancy_while_present_does_not_ratchet_upwards(self) -> None:
        occupied = [
            record["bin_occupancy_l"] for record in self.records
            if record["bin_occupancy_l"] is not None
        ]
        self.assertTrue(occupied)
        # Sizes differ across the catalogue, so compare each entry against its
        # own first appearance rather than the run-wide spread.
        for offset in range(len(CATALOGUE)):
            same = occupied[offset :: len(CATALOGUE)]
            with self.subTest(entry=CATALOGUE[offset][0]):
                self.assertLess(max(same) - min(same), max(0.5, same[0] * 0.15))


class ReferenceContaminationRegressionTests(unittest.TestCase):
    """The diagnosis itself: the old behaviour degrades, the new one does not."""

    def test_rewriting_the_reference_on_deposit_degrades_later_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, frame, depth, camera = _build(
                directory, advance_reference_on_deposit=True
            )
            records = _soak(pipeline, detector, frame, depth, camera, presentations=20)
        measured = sum(record["measured"] for record in records)
        self.assertLess(
            measured, 20,
            "expected the superseded reference-rewrite behaviour to lose objects; "
            "if this now passes cleanly the contamination path has changed",
        )

    def test_the_default_keeps_the_reference_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, frame, depth, camera = _build(directory)
            before = pipeline.reference_realsense.copy()
            _soak(pipeline, detector, frame, depth, camera, presentations=8)
            self.assertIsNotNone(pipeline.reference_realsense)
            np.testing.assert_array_equal(pipeline.reference_realsense, before)


if __name__ == "__main__":
    unittest.main()
