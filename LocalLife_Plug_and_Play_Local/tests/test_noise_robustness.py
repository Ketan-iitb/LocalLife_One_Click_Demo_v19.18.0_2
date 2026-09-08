"""Regression guards for the real-hardware detection-flood bug (round 2).

On real hardware, a single wrinkled/folded bag produced many small,
disconnected "changed" fragments (folds, shadows, sensor speckle). Combined
with `allow_unclassified_foreground` becoming permissive once any track was
confirmed, `fuse_scene_detections` previously spawned one phantom
"garbage bag (depth silhouette)" detection PER unmatched fragment, flooding
the dashboard with duplicate boxes for one physical object. These tests
guard the fix: fragments are merged before they ever reach fusion, and even
if several unmatched scene objects still exist, fusion recovers at most one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.geometry import detect_scene_objects, fuse_scene_detections
from locallife_cloud.pipeline import VisionPipeline, deduplicate_overlapping_detections, is_bag_detection
from locallife_cloud.types import CameraIntrinsics, Detection


class SceneFragmentMergingTests(unittest.TestCase):
    def test_nearby_fragments_of_one_wrinkled_bag_merge_into_one_component(self) -> None:
        # Three small, spatially separated "changed" patches within a few
        # pixels of each other -- the kind of pattern a folded/wrinkled bag
        # surface or patchy depth dropout produces on real hardware. The old
        # 7x7 closing kernel could not bridge gaps this size and produced
        # three separate detections; the new size-scaled kernel must merge
        # them into one.
        shape = (240, 320, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        baseline = np.zeros(shape, dtype=np.uint8)
        # Three patches roughly 10-14px apart, all inside a ~70x70 region.
        frame[40:60, 40:60] = (10, 120, 10)
        frame[45:65, 68:88] = (10, 120, 10)
        frame[62:82, 50:70] = (10, 120, 10)
        roi = (0.0, 0.0, 1.0, 1.0)
        objects = detect_scene_objects(frame, baseline, None, None, roi, min_area=50)
        self.assertEqual(len(objects), 1)

    def test_scattered_noise_speckle_is_not_recovered_as_its_own_object(self) -> None:
        # A handful of isolated single-pixel-scale "changed" specks far apart
        # from each other and from the real object must not each become a
        # detection: they fall under the 1%-of-region minimum-area floor.
        shape = (240, 320, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        baseline = np.zeros(shape, dtype=np.uint8)
        frame[100:140, 100:140] = (10, 120, 10)  # the real object, ~1600px
        for (y, x) in [(10, 10), (10, 300), (230, 10), (230, 300), (120, 5)]:
            frame[y : y + 2, x : x + 2] = (10, 120, 10)
        roi = (0.0, 0.0, 1.0, 1.0)
        objects = detect_scene_objects(frame, baseline, None, None, roi, min_area=50)
        self.assertEqual(len(objects), 1)
        self.assertGreater(objects[0].area_pixels, 1000)


class ShadowAndBackgroundInflationTests(unittest.TestCase):
    """Round 8: real hardware reported a modest pillow/backpack measured as
    covering most of the frame (~116 L), a shadow being detected as its own
    object, and detection flickering between which camera "saw" anything.
    Root cause: `detect_scene_objects` closed nearby "changed" fragments
    with a fairly large kernel and then blindly solid-filled every resulting
    contour's hull with `cv2.drawContours(..., thickness=FILLED)`. When the
    close bridged a real object to an unrelated nearby patch (most commonly
    the shadow the object itself casts on the floor/wall next to it, but
    also patchy RGB noise), the fill claimed the *entire* enclosed area --
    including all the untouched background between the two -- as part of
    the object. These tests hold the fix in place: only fill a hull solid
    when it is actually densely "changed" inside; otherwise keep just the
    closed silhouette, never the invented gap.
    """

    def test_object_next_to_a_cast_shadow_does_not_absorb_the_shadow(self) -> None:
        # A real 50x50 object with genuine depth change, sitting a few
        # pixels from a much larger darker patch (the shadow it casts) that
        # has no depth change of its own. The two are close enough that the
        # merge-closing kernel bridges them into one contour; the reported
        # object must still be roughly the real object's own footprint, not
        # a box that also swallows the shadow.
        h, w = 200, 300
        baseline = np.full((h, w, 3), 140, dtype=np.uint8)
        frame = baseline.copy()
        baseline_depth = np.full((h, w), 1.20, dtype=np.float32)
        depth = baseline_depth.copy()
        frame[60:110, 60:110] = 40
        depth[60:110, 60:110] = 0.90
        frame[60:110, 114:280] = 95  # shadow: darker, no depth change, close by
        objects = detect_scene_objects(
            frame, baseline, depth, baseline_depth, (0.0, 0.0, 1.0, 1.0),
            threshold=18, min_area=100,
        )
        self.assertEqual(len(objects), 1)
        x1, y1, x2, y2 = objects[0].box
        self.assertLess((x2 - x1) * (y2 - y1), 5000)  # real object is 2500px; shadow spans 166x50=8300 more
        self.assertLess(objects[0].area_pixels / (h * w), 0.30)

    def test_patchy_depth_dark_object_is_still_recovered_in_full(self) -> None:
        # Guards against fixing the shadow-inflation bug by making shadow
        # rejection so aggressive it also rejects real dark objects (a black
        # bag/backpack, documented in earlier rounds) whose own surface is
        # darker than the baseline and whose depth is only patchily sensed.
        # A real object with ~35% scattered depth coverage and no shadow
        # nearby must still be recovered at essentially its full footprint.
        h, w = 200, 300
        baseline = np.full((h, w, 3), 140, dtype=np.uint8)
        frame = baseline.copy()
        baseline_depth = np.full((h, w), 1.20, dtype=np.float32)
        depth = baseline_depth.copy()
        frame[60:110, 60:110] = 40  # a dark object, well below baseline luma
        rng = np.random.default_rng(0)
        patchy = rng.random((50, 50)) < 0.35
        region = depth[60:110, 60:110]
        region[patchy] = 0.90
        depth[60:110, 60:110] = region
        objects = detect_scene_objects(
            frame, baseline, depth, baseline_depth, (0.0, 0.0, 1.0, 1.0),
            threshold=18, min_area=250,
        )
        self.assertEqual(len(objects), 1)
        self.assertGreaterEqual(objects[0].area_pixels, 2400)  # true footprint is 2500px

    def test_shadow_rejection_also_holds_when_depth_signal_is_only_moderate(self) -> None:
        # Depth coverage between the old min_area gate and the new smaller
        # anchor floor must still activate depth-gated (shadow-safe)
        # recovery rather than falling back to unguarded RGB.
        h, w = 200, 300
        baseline = np.full((h, w, 3), 140, dtype=np.uint8)
        frame = baseline.copy()
        baseline_depth = np.full((h, w), 1.20, dtype=np.float32)
        depth = baseline_depth.copy()
        frame[60:110, 60:110] = 40
        # Only a modest strip of direct depth signal (12 rows x 50 cols =
        # 600px): below the min_area=700 used below (the OLD single gate
        # `count >= min_area` would have missed it and fallen back to
        # unguarded RGB, absorbing the shadow), but comfortably above the
        # new, much smaller anchor floor (87 at min_area=700).
        depth[60:72, 60:110] = 0.90
        frame[60:110, 114:280] = 95
        objects = detect_scene_objects(
            frame, baseline, depth, baseline_depth, (0.0, 0.0, 1.0, 1.0),
            threshold=18, min_area=700,
        )
        self.assertEqual(len(objects), 1)
        x1, y1, x2, y2 = objects[0].box
        # The shadow alone spans 166 x 50 = 8300px starting at column 114;
        # if it had been absorbed the box would extend past column 114.
        self.assertLessEqual(x2, 114)


class FuseSceneDetectionsCapTests(unittest.TestCase):
    def test_multiple_unmatched_scene_fragments_yield_at_most_one_phantom(self) -> None:
        shape = (100, 100, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        scene_objects = []
        for (y, x, size) in [(10, 10, 15), (40, 40, 25), (70, 70, 10)]:
            mask = np.zeros(shape[:2], dtype=bool)
            mask[y : y + size, x : x + size] = True
            scene_objects.append(
                Detection(
                    label="storage container", confidence=1.0,
                    box=(x, y, x + size, y + size), mask=mask,
                    source="depth-scene-segmentation",
                )
            )
        fused = fuse_scene_detections(
            frame, [], scene_objects, allow_unclassified=True, bag_only=True,
        )
        self.assertEqual(len(fused), 1)
        # The largest fragment (the 25x25 one) must be the one recovered.
        self.assertEqual(fused[0].area_pixels, 625)

    def test_no_unclassified_recovery_without_the_flag(self) -> None:
        shape = (100, 100, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=bool)
        mask[10:30, 10:30] = True
        scene_objects = [
            Detection(label="storage container", confidence=1.0, box=(10, 10, 30, 30), mask=mask)
        ]
        fused = fuse_scene_detections(frame, [], scene_objects, allow_unclassified=False, bag_only=True)
        self.assertEqual(fused, [])

    def test_whole_frame_change_is_never_promoted_to_a_phantom_bag(self) -> None:
        # A camera pan to the ceiling, or a baseline that no longer matches
        # the current framing, makes almost the entire frame read as
        # "changed" at once. That single giant region used to become "the
        # largest unmatched fragment" and get promoted straight to a
        # 0%-confidence "garbage bag (depth silhouette)" -- exactly the
        # "detecting the ceiling as a garbage bag" report from real hardware.
        shape = (200, 200, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=bool)
        mask[0:198, 40:200] = True  # touches the top, right, and bottom edges
        scene_objects = [
            Detection(
                label="storage container", confidence=1.0,
                box=(40, 0, 200, 198), mask=mask, source="depth-scene-segmentation",
            )
        ]
        fused = fuse_scene_detections(frame, [], scene_objects, allow_unclassified=True, bag_only=True)
        self.assertEqual(fused, [])

    def test_a_smaller_plausible_region_is_still_recovered_over_an_implausible_larger_one(self) -> None:
        shape = (200, 200, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        # A whole-scene-change region: larger, but implausible (3 edges).
        flood_mask = np.zeros(shape[:2], dtype=bool)
        flood_mask[0:198, 100:200] = True
        # A real, compact object sitting away from the frame edges.
        bag_mask = np.zeros(shape[:2], dtype=bool)
        bag_mask[60:120, 20:70] = True
        scene_objects = [
            Detection(
                label="storage container", confidence=1.0,
                box=(100, 0, 200, 198), mask=flood_mask, source="depth-scene-segmentation",
            ),
            Detection(
                label="storage container", confidence=1.0,
                box=(20, 60, 70, 120), mask=bag_mask, source="depth-scene-segmentation",
            ),
        ]
        fused = fuse_scene_detections(frame, [], scene_objects, allow_unclassified=True, bag_only=True)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].area_pixels, 3000)

    def test_large_but_edge_isolated_object_is_still_recovered(self) -> None:
        # A real bag can legitimately be large and touch one frame edge (it
        # extends past the bottom of the crop). That alone must not trip the
        # whole-scene-change rejection.
        shape = (200, 200, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=bool)
        mask[80:200, 40:160] = True  # touches only the bottom edge
        scene_objects = [
            Detection(
                label="storage container", confidence=1.0,
                box=(40, 80, 160, 200), mask=mask, source="depth-scene-segmentation",
            )
        ]
        fused = fuse_scene_detections(frame, [], scene_objects, allow_unclassified=True, bag_only=True)
        self.assertEqual(len(fused), 1)


class DeduplicateOverlappingDetectionsTests(unittest.TestCase):
    def test_heavily_overlapping_same_type_boxes_collapse_to_one(self) -> None:
        shape = (100, 100)
        first = Detection("garbage bag", 0.9, (10, 10, 60, 60))
        second = Detection("garbage bag (depth silhouette)", 0.0, (12, 12, 58, 58))
        result = deduplicate_overlapping_detections([first, second], shape)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)

    def test_distinct_non_overlapping_objects_are_kept(self) -> None:
        shape = (100, 100)
        first = Detection("garbage bag", 0.9, (0, 0, 20, 20))
        second = Detection("cardboard box", 0.8, (60, 60, 90, 90))
        result = deduplicate_overlapping_detections([first, second], shape)
        self.assertEqual(len(result), 2)


class DetectorSequence:
    """Returns a fixed list of detections per call, mimicking an intermittent
    neural detector: confirms the bag once, then goes quiet (the real-world
    "detector momentarily misses it" case that used to flood the dashboard
    with silhouette fragments once a track had been confirmed)."""

    runtime = {"device": "cpu"}

    def __init__(self, sequence: list[list[Detection]]) -> None:
        self._sequence = sequence
        self._calls = 0

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        answer = self._sequence[min(self._calls, len(self._sequence) - 1)]
        self._calls += 1
        return [list(answer) for _ in frames]


class EndToEndFloodRegressionTests(unittest.TestCase):
    def test_wrinkled_bag_with_shadow_never_floods_once_track_is_confirmed(self) -> None:
        # Reproduces the reported real-hardware symptom end to end: after a
        # bag is confirmed once by the neural detector, later frames where
        # the detector misses it (but its wrinkled/shadowed silhouette still
        # shows up as several separate "changed" fragments against the
        # empty-scene baseline) must still yield exactly one live detection,
        # never one box per fragment.
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                results_dir=Path(directory),
                enable_monocular_depth=False,
                enable_material_classification=False,
                roi=(0.0, 0.0, 1.0, 1.0),
                min_component_pixels=20,
                restore_saved_baseline=False,
                automatic_baseline=False,
                bag_only=True,
                tracker_confirm_frames=1,
                tracker_max_missing_frames=10,
            )
            empty = np.zeros((160, 200, 3), dtype=np.uint8)
            confirming = Detection("garbage bag", 0.9, (60, 60, 110, 110))
            detector = DetectorSequence([[confirming], []])
            station = VisionPipeline(config, detector=detector)
            intrinsics = CameraIntrinsics(fx=150, fy=150, ppx=100, ppy=80)
            station.set_baseline(empty, None, intrinsics)

            # Frame 1: the neural detector confirms a bag in a clean region.
            confirm_frame = empty.copy()
            confirm_frame[60:110, 60:110] = (20, 120, 20)
            result = station.process_frame(confirm_frame, intrinsics=intrinsics, persist=False)
            self.assertEqual(len(result.detections), 1)
            self.assertTrue(station.tracker.has_active_counted_track())

            # Frame 2+: detector goes quiet, but the bag's own wrinkles and a
            # cast shadow now show up as three separated "changed" patches.
            noisy_frame = empty.copy()
            noisy_frame[62:80, 62:80] = (20, 120, 20)
            noisy_frame[65:85, 88:108] = (20, 120, 20)
            noisy_frame[82:105, 68:90] = (15, 60, 15)  # shadow-toned patch
            for _ in range(3):
                result = station.process_frame(noisy_frame, intrinsics=intrinsics, persist=False)
                live = [item for item in result.detections if item.tracking_status != "tentative"]
                self.assertLessEqual(len(live), 1)


class BackgroundBleedOnceTrackedRegressionTests(unittest.TestCase):
    """Regression guards for the round-21 real-hardware report: once a real
    bag/box had been confirmed once, `allow_unclassified` used to become
    permissive off a plain `has_active_counted_track()` boolean -- "is
    anything, anywhere, already counted?" -- with no requirement that the
    newly-promoted "unclassified object" region have anything to do with
    that track. On real hardware (a cluttered room whose empty-scene
    baseline predated a couch, a backpack, an office chair now in view) that
    let the single largest unrelated "changed" blob be promoted to its own
    tracked, displayed, drifting box every frame, landing on whichever piece
    of furniture happened to read as most "different from empty" that frame.
    """

    def test_far_away_unmatched_region_is_not_promoted_by_an_unrelated_track(self) -> None:
        shape = (200, 200, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=bool)
        mask[140:190, 140:190] = True  # bottom-right corner
        far_region = Detection(
            label="storage container", confidence=1.0,
            box=(140, 140, 190, 190), mask=mask, source="depth-scene-segmentation",
        )
        # A track already confirmed and counted, but sitting in the opposite
        # (top-left) corner of the frame -- nowhere near the new region.
        unrelated_track_box = (5, 5, 35, 35)

        fused = fuse_scene_detections(
            frame, [], [far_region],
            allow_unclassified=False,
            counted_track_boxes=[unrelated_track_box],
        )
        self.assertEqual(fused, [])

    def test_nearby_unmatched_region_is_still_bridged_by_its_own_track(self) -> None:
        shape = (200, 200, 3)
        frame = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape[:2], dtype=bool)
        mask[60:100, 60:100] = True
        region = Detection(
            label="storage container", confidence=1.0,
            box=(60, 60, 100, 100), mask=mask, source="depth-scene-segmentation",
        )
        # The same track's own last-seen box, close to this new region --
        # this is the legitimate "bridge a brief detector dropout" case.
        own_track_box = (58, 58, 98, 98)

        fused = fuse_scene_detections(
            frame, [], [region],
            allow_unclassified=False,
            counted_track_boxes=[own_track_box],
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "unclassified object")

    def test_end_to_end_confirmed_bag_does_not_unlock_a_phantom_across_the_room(self) -> None:
        # Same overall shape as test_wrinkled_bag_with_shadow_never_floods_
        # once_track_is_confirmed above, but here the neural detector keeps
        # confirming the real bag every frame (never goes quiet), while a
        # *separate*, larger, unrelated "changed" region appears on the far
        # side of the frame -- real-hardware furniture a stale baseline
        # reads as "different from empty". Before the fix, the real bag
        # being counted once was enough (`has_active_counted_track()`) to
        # let that unrelated far region -- being the largest UNMATCHED scene
        # component -- get promoted to its own live "unclassified object"
        # track every frame, regardless of where it was.
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                results_dir=Path(directory),
                enable_monocular_depth=False,
                enable_material_classification=False,
                roi=(0.0, 0.0, 1.0, 1.0),
                min_component_pixels=20,
                restore_saved_baseline=False,
                automatic_baseline=False,
                bag_only=False,
                tracker_confirm_frames=1,
                tracker_max_missing_frames=10,
            )
            empty = np.zeros((200, 260, 3), dtype=np.uint8)
            confirming = Detection("garbage bag", 0.9, (30, 30, 80, 80))
            detector = DetectorSequence([[confirming]])
            station = VisionPipeline(config, detector=detector)
            intrinsics = CameraIntrinsics(fx=150, fy=150, ppx=130, ppy=100)
            station.set_baseline(empty, None, intrinsics)

            confirm_frame = empty.copy()
            confirm_frame[30:80, 30:80] = (20, 120, 20)
            result = station.process_frame(confirm_frame, intrinsics=intrinsics, persist=False)
            self.assertEqual(len(result.detections), 1)
            self.assertTrue(station.tracker.has_active_counted_track())

            # The real bag keeps being confirmed every frame; a separate,
            # larger, unrelated region on the far side of the frame (opposite
            # corner from the confirmed bag) now also reads as "changed"
            # against the stale baseline.
            far_noise_frame = confirm_frame.copy()
            far_noise_frame[130:190, 180:250] = (90, 90, 90)
            for _ in range(3):
                result = station.process_frame(far_noise_frame, intrinsics=intrinsics, persist=False)
                labels = {item.label for item in result.detections if item.tracking_status != "tentative"}
                self.assertNotIn("unclassified object", labels)
            self.assertEqual(len(station.tracker.tracks), 1)


class BagDetectionFurnitureFalsePositiveTests(unittest.TestCase):
    # Reproduces labels actually seen on real hardware once Ultralytics'
    # YOLOE silently fell back to its broad "prompt-free" vocabulary (see
    # CHANGES_local-ai-19.2.0.md): "bean bag chair" and similar furniture
    # items contain the word "bag" and were previously treated as real waste
    # bags by is_bag_detection()'s naive substring check.
    def test_bean_bag_chair_is_not_treated_as_a_waste_bag(self) -> None:
        self.assertFalse(is_bag_detection("bean bag chair"))
        self.assertFalse(is_bag_detection("beanbag chair"))

    def test_real_waste_bags_are_still_recognized(self) -> None:
        for label in ["garbage bag", "trash bag", "black plastic bag", "grocery sack"]:
            self.assertTrue(is_bag_detection(label))


if __name__ == "__main__":
    unittest.main()
