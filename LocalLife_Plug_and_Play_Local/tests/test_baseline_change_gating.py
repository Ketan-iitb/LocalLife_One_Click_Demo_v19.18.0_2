"""Tests for the v7 "was this object actually placed here?" gate.

The v6 hardware trial measured a bed. A green bag resting on a duvet was
reported as roughly 1021 x 669 x 394 mm, the RealSense object counter climbed
past 29, and the dashboard raised `dimension_instability`,
`live_fitted_support_plane` and `low_elevated_fraction` on a scene that was
physically standing still.

The chain behind that, in order:

1. `_consider_automatic_baseline()` reset its empty-scene countdown on any
   accepted detection. In `geometry_validation` mode the broad household
   prompt bank detects the room's own furniture in every single frame, so the
   countdown never completed and no empty-scene baseline was ever captured.
2. With no baseline there is no reference depth, so the support plane was
   refitted from each frame's own background -- a background defined by
   excluding that frame's detections, which change constantly. The plane
   moved every frame and the measured object changed shape with it.
3. With no baseline there was also no notion of "new". Elevation above the
   support plane was the only test an object had to pass, and on a bed the
   bag, the duvet folds and a pillow are one *connected* elevated surface, so
   depth-supported mask recovery annexed the lot and the estimator measured
   it faithfully.

These tests lock in the fix: a baseline can be captured in validation mode,
newly-introduced pixels are identified from it, and both detection admission
and mask recovery are constrained by that evidence.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import (
    VisionPipeline,
    reject_unchanged_background_detections,
)
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import (
    fit_reference_plane,
    newly_introduced_mask,
    recover_elevated_object_mask,
)


def _bed_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A raised surface (the bed) with an object placed on top of it.

    Returns the empty baseline depth, the current depth, the bed's own mask
    and the placed object's mask. The object sits directly on the bed, so the
    two are connected in image space and connectivity alone cannot separate
    them -- which is exactly the v6 failure.
    """
    floor_m = 2.40
    baseline = np.full((160, 160), floor_m, dtype=np.float32)
    bed = np.zeros((160, 160), dtype=bool)
    bed[40:150, 20:140] = True
    baseline[bed] = 1.95  # the bed is 45 cm above the floor, and always was

    current = baseline.copy()
    placed = np.zeros((160, 160), dtype=bool)
    placed[70:100, 60:100] = True
    current[placed] = 1.80  # a 15 cm tall object placed on the bed
    return baseline, current, bed, placed


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=200.0, fy=200.0, ppx=80.0, ppy=80.0)


class NewlyIntroducedMaskTests(unittest.TestCase):
    def test_only_the_placed_object_reads_as_new(self) -> None:
        baseline, current, bed, placed = _bed_scene()

        introduced = newly_introduced_mask(current, baseline, min_change_m=0.012)

        self.assertIsNotNone(introduced)
        # Every pixel of the placed object is new.
        self.assertGreater(
            np.count_nonzero(introduced & placed) / np.count_nonzero(placed), 0.95,
        )
        # The bed itself is elevated, but it is not new.
        unchanged_bed = bed & ~placed
        self.assertEqual(int(np.count_nonzero(introduced & unchanged_bed)), 0)

    def test_an_unchanged_scene_produces_no_new_pixels(self) -> None:
        baseline, _current, _bed, _placed = _bed_scene()

        introduced = newly_introduced_mask(baseline, baseline.copy())

        self.assertIsNotNone(introduced)
        self.assertEqual(int(np.count_nonzero(introduced)), 0)

    def test_no_baseline_returns_none_rather_than_an_empty_mask(self) -> None:
        # "No evidence of change" must never be silently treated as "nothing
        # changed": that would blank every measurement on an installation
        # that has not captured a baseline yet.
        _baseline, current, _bed, _placed = _bed_scene()

        self.assertIsNone(newly_introduced_mask(current, None))

    def test_a_noisy_baseline_raises_the_change_threshold(self) -> None:
        baseline, current, _bed, placed = _bed_scene()
        noise = np.full(baseline.shape, 0.060, dtype=np.float32)

        introduced = newly_introduced_mask(
            current, baseline, min_change_m=0.012, noise_map_m=noise,
        )

        # The object stands 15 cm proud, but where the baseline itself was
        # measured to +/- 6 cm that is not a confident enough change.
        self.assertIsNotNone(introduced)
        self.assertEqual(int(np.count_nonzero(introduced & placed)), 0)


class MaskRecoveryChangeConstraintTests(unittest.TestCase):
    def _plane(self, baseline: np.ndarray) -> object:
        floor = np.zeros(baseline.shape, dtype=bool)
        floor[:35, :] = True
        return fit_reference_plane(baseline, _intrinsics(), mask=floor)

    def test_recovery_no_longer_annexes_the_unchanged_bed(self) -> None:
        baseline, current, bed, placed = _bed_scene()
        plane = self._plane(baseline)
        self.assertIsNotNone(plane)
        # A partial semantic seed on the placed object, as a real detector
        # returns for a pale bag.
        seed = np.zeros(current.shape, dtype=bool)
        seed[78:92, 70:90] = True
        introduced = newly_introduced_mask(current, baseline, min_change_m=0.012)

        # `max_expansion` is relaxed here only to isolate what is under test:
        # whether elevation plus connectivity can tell the object from the
        # bed it rests on. That ratio guard is a blunt secondary bound -- it
        # happens to catch this particular seed/bed size combination, but it
        # is size-dependent and cannot be relied on (a bigger seed or a
        # smaller unchanged surface slips straight past it).
        unconstrained = recover_elevated_object_mask(
            current, _intrinsics(), seed, plane, min_height_m=0.010, min_points=20,
            max_expansion=100.0,
        )
        constrained = recover_elevated_object_mask(
            current, _intrinsics(), seed, plane, min_height_m=0.010, min_points=20,
            change_mask=introduced, min_change_fraction=0.55,
        )

        unchanged_bed = bed & ~placed
        # This is the v6 behaviour, reproduced: elevation plus connectivity
        # swallows the whole bed.
        self.assertIsNotNone(unconstrained)
        self.assertGreater(int(np.count_nonzero(unconstrained & unchanged_bed)), 1000)
        # With change evidence, the recovered surface is the object.
        self.assertIsNotNone(constrained)
        self.assertEqual(int(np.count_nonzero(constrained & unchanged_bed)), 0)
        self.assertGreater(
            np.count_nonzero(constrained & placed) / np.count_nonzero(placed), 0.90,
        )

    def test_a_mask_that_is_mostly_unchanged_scenery_is_rejected(self) -> None:
        baseline, current, _bed, _placed = _bed_scene()
        plane = self._plane(baseline)
        # Seed an unchanged part of the bed: nothing was placed here.
        seed = np.zeros(current.shape, dtype=bool)
        seed[120:140, 40:70] = True
        introduced = newly_introduced_mask(current, baseline, min_change_m=0.012)

        recovered = recover_elevated_object_mask(
            current, _intrinsics(), seed, plane, min_height_m=0.010, min_points=20,
            change_mask=introduced, min_change_fraction=0.55,
        )

        self.assertIsNone(recovered)

    def test_a_partial_logo_seed_is_still_recovered_in_full(self) -> None:
        # The v4 behaviour this recovery exists for must survive: a detector
        # that returns only a printed centre panel still yields the whole
        # object, provided the object is genuinely new.
        baseline, current, _bed, placed = _bed_scene()
        plane = self._plane(baseline)
        seed = np.zeros(current.shape, dtype=bool)
        seed[82:88, 76:84] = True
        introduced = newly_introduced_mask(current, baseline, min_change_m=0.012)

        recovered = recover_elevated_object_mask(
            current, _intrinsics(), seed, plane, min_height_m=0.010, min_points=20,
            # The object is ~25x this deliberately tiny seed, so the ratio
            # bound has to be opened for the seed to reach it at all. That
            # bound still applies in production: a seed too small for it
            # falls back to the semantic mask rather than to the bed.
            max_expansion=40.0,
            change_mask=introduced, min_change_fraction=0.55,
        )

        self.assertIsNotNone(recovered)
        self.assertGreater(
            int(np.count_nonzero(recovered)), int(np.count_nonzero(seed)) * 4,
        )
        self.assertGreater(
            np.count_nonzero(recovered & placed) / np.count_nonzero(placed), 0.90,
        )


class UnchangedDetectionRejectionTests(unittest.TestCase):
    def _detection(self, label: str, mask: np.ndarray) -> Detection:
        rows, columns = np.where(mask)
        box = (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))
        return Detection(label, 0.60, box, mask.copy())

    def test_furniture_is_rejected_and_the_placed_object_is_kept(self) -> None:
        baseline, current, bed, placed = _bed_scene()
        introduced = newly_introduced_mask(current, baseline, min_change_m=0.012)
        pillow = np.zeros(current.shape, dtype=bool)
        pillow[45:65, 30:70] = True  # part of the unchanged bed

        kept, rejected = reject_unchanged_background_detections(
            [self._detection("pillow", pillow), self._detection("handbag", placed)],
            introduced,
            current.shape,
            min_change_fraction=0.275,
        )

        self.assertEqual(rejected, 1)
        self.assertEqual([item.label for item in kept], ["handbag"])

    def test_nothing_is_rejected_without_a_baseline(self) -> None:
        # An installation with no captured baseline must keep behaving as it
        # did, rather than showing an empty dashboard.
        _baseline, current, _bed, placed = _bed_scene()

        kept, rejected = reject_unchanged_background_detections(
            [self._detection("handbag", placed)], None, current.shape,
            min_change_fraction=0.55,
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(len(kept), 1)


class ValidationBaselineCaptureTests(unittest.TestCase):
    """Step 1: the empty-scene countdown must not be blocked by the very
    prompt bank that validation mode exists to enable."""

    class _StubDetector:
        device = "cpu"
        runtime = {"device": "cpu"}

        def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
            return [[] for _ in frames]

    def _pipeline(self, mode: str) -> VisionPipeline:
        config = AppConfig(
            results_dir=Path(tempfile.mkdtemp()),
            operating_mode=mode,
            enable_monocular_depth=False,
            roi=(0, 0, 1, 1),
            min_component_pixels=20,
            automatic_baseline=True,
            automatic_baseline_frames=3,
            restore_saved_baseline=False,
            auto_deposit=False,
        )
        return VisionPipeline(config, detector=self._StubDetector())

    def _still_frames(self, pipeline: VisionPipeline, detections: list[Detection]) -> None:
        frame = np.full((60, 60, 3), 120, dtype=np.uint8)
        depth = np.full((60, 60), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100.0, fy=100.0, ppx=30.0, ppy=30.0)
        for _ in range(6):
            pipeline.latest_frame = frame.copy()
            pipeline.latest_depth = depth.copy()
            pipeline._consider_automatic_baseline(
                frame, depth, camera, detections, [],
            )

    def test_validation_mode_captures_a_baseline_despite_furniture_detections(self) -> None:
        pipeline = self._pipeline("geometry_validation")
        furniture = np.zeros((60, 60), dtype=bool)
        furniture[10:40, 10:40] = True

        self._still_frames(pipeline, [Detection("pillow", 0.5, (10, 10, 40, 40), furniture)])

        self.assertIsNotNone(
            pipeline.baseline_rgb,
            "validation mode deadlocked: the household prompt bank blocked its own baseline",
        )

    def test_waste_mode_still_requires_a_genuinely_empty_scene(self) -> None:
        pipeline = self._pipeline("waste")
        bag = np.zeros((60, 60), dtype=bool)
        bag[10:40, 10:40] = True

        self._still_frames(pipeline, [Detection("garbage bag", 0.9, (10, 10, 40, 40), bag)])

        self.assertIsNone(pipeline.baseline_rgb)
        self.assertEqual(pipeline._automatic_baseline_status, "waiting-for-empty-scene")


if __name__ == "__main__":
    unittest.main()
