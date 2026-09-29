"""Crowded-bin regressions: touching bags are not one depth island/track."""

import unittest

import numpy as np

from locallife_cloud.geometry import fuse_scene_detections
from locallife_cloud.pipeline import deduplicate_overlapping_detections
from locallife_cloud.types import Detection


class CrowdedBinTests(unittest.TestCase):
    def test_two_confirmed_bags_in_one_depth_component_keep_two_masks(self):
        shape = (100, 120)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        scene = np.zeros(shape, dtype=bool)
        scene[15:80, 10:105] = True  # stereo closing joined adjacent bags
        first = np.zeros(shape, dtype=bool)
        first[22:70, 15:52] = True
        second = np.zeros(shape, dtype=bool)
        second[22:70, 62:100] = True
        fused = fuse_scene_detections(
            frame,
            [Detection("plastic waste bag", .8, (15, 22, 52, 70), mask=first),
             Detection("filled plastic waste bag", .7, (62, 22, 100, 70), mask=second)],
            [Detection("storage container", 1, (10, 15, 105, 80), mask=scene)],
            require_scene_match=True,
        )
        self.assertEqual(len(fused), 2)
        self.assertEqual(len(deduplicate_overlapping_detections(fused, shape)), 2)
        self.assertFalse(np.any(fused[0].mask & fused[1].mask))
        self.assertTrue(all(item.box[2] - item.box[0] < 65 for item in fused))

    def test_two_prompts_for_same_bag_still_fuse_once(self):
        shape = (100, 100)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        scene = np.zeros(shape, dtype=bool)
        scene[20:80, 20:80] = True
        first = np.zeros(shape, dtype=bool)
        first[25:73, 25:73] = True
        second = np.zeros(shape, dtype=bool)
        second[28:76, 28:76] = True
        fused = fuse_scene_detections(
            frame,
            [Detection("garbage bag", .8, (25, 25, 73, 73), mask=first),
             Detection("plastic waste bag", .7, (28, 28, 76, 76), mask=second)],
            [Detection("storage container", 1, (20, 20, 80, 80), mask=scene)],
        )
        self.assertEqual(len(fused), 1)

    def test_nested_boxes_with_distinct_masks_remain_two_tracks(self):
        shape = (100, 100)
        first = np.zeros(shape, dtype=bool)
        first[25:55, 20:45] = True
        second = np.zeros(shape, dtype=bool)
        second[45:75, 55:80] = True
        detections = [
            Detection("garbage bag", .8, (15, 15, 85, 85), mask=first, source="yoloe-scene-fusion"),
            Detection("garbage bag", .7, (20, 20, 80, 80), mask=second, source="yoloe-scene-fusion"),
        ]
        self.assertEqual(len(deduplicate_overlapping_detections(detections, shape)), 2)

    def test_stale_pile_cannot_bridge_one_old_track_as_one_giant_bag(self):
        shape = (100, 120)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        pile = np.zeros(shape, dtype=bool)
        pile[10:90, 15:105] = True
        scene = [Detection("storage container", 1, (15, 10, 105, 90), mask=pile)]
        self.assertEqual(fuse_scene_detections(
            frame, [], scene, counted_track_boxes=[(20, 20, 45, 50)],
        ), [])


class ReviewedPatchTests(unittest.TestCase):
    """Two gaps in the supplied v40 patch, closed here."""

    def _scene(self):
        shape = (100, 120)
        scene = np.zeros(shape, dtype=bool)
        scene[15:80, 10:105] = True
        return shape, np.zeros((*shape, 3), np.uint8), [Detection("storage container", 1, (10, 15, 105, 80), mask=scene)]

    def test_a_bag_labelled_twice_does_not_stop_its_neighbour_being_separate(self):
        shape, frame, scene = self._scene()
        a1 = np.zeros(shape, bool); a1[22:70, 15:52] = True
        a2 = np.zeros(shape, bool); a2[24:72, 17:54] = True
        b = np.zeros(shape, bool); b[22:70, 62:100] = True
        fused = fuse_scene_detections(frame, [
            Detection("garbage bag", .8, (15, 22, 52, 70), mask=a1),
            Detection("plastic waste bag", .7, (17, 24, 54, 72), mask=a2),
            Detection("plastic waste bag", .7, (62, 22, 100, 70), mask=b)], scene, require_scene_match=True)
        self.assertEqual(len(fused), 2)        # the patch as supplied: 1

    def test_close_bags_never_share_measurement_pixels(self):
        shape, frame, scene = self._scene()
        c = np.zeros(shape, bool); c[22:70, 15:55] = True
        d = np.zeros(shape, bool); d[22:70, 60:100] = True
        fused = fuse_scene_detections(frame, [
            Detection("garbage bag", .8, (15, 22, 55, 70), mask=c),
            Detection("garbage bag", .7, (60, 22, 100, 70), mask=d)], scene, require_scene_match=True)
        self.assertEqual(len(fused), 2)
        self.assertFalse(np.any(fused[0].mask & fused[1].mask))   # the patch as supplied: 54 shared

    def test_a_mid_sized_stale_pile_cannot_bridge_an_old_track(self):
        shape = (200, 240)
        pile = np.zeros(shape, bool); pile[60:160, 60:200] = True
        fused = fuse_scene_detections(np.zeros((*shape, 3), np.uint8), [],
                                      [Detection("storage container", 1, (60, 60, 200, 160), mask=pile)],
                                      counted_track_boxes=[(70, 70, 110, 110)])
        self.assertEqual(fused, [])            # v39: one giant "unclassified object"


class LogitechSupportSurfaceTests(unittest.TestCase):
    """Logitech geometry: a bag on earlier waste is measured from that waste."""

    def _measure(self, tilt, surfaces, target):
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        import test_v37_logitech_pose_and_colour as rig
        from locallife_cloud.logitech_volume import metric_object_volume
        from locallife_cloud.volume import fit_reference_plane

        scene = rig.Scene(tilt)
        built = [build(scene) for build in surfaces]
        depth, _ = scene.render(*built)
        own_depth, own = scene.render(built[target])
        mask = own & (np.abs(depth - own_depth) < 1e-9)
        plane = fit_reference_plane(scene.empty, rig.CAMERA, mask=scene.region)
        result = metric_object_volume(depth, rig.CAMERA, mask, plane, reference_depth_m=scene.empty,
                                      min_height_m=0.004, min_pixels=25, cell_size_m=0.005)
        return result.diagnostics["height_p90_m"] * 1000.0, result.diagnostics.get("height_reference")

    def test_a_bag_on_a_pile_is_its_own_height_not_the_piles(self):
        for tilt in (15.0, 25.0, 40.0):
            with self.subTest(tilt=tilt):
                height, reference = self._measure(tilt, [
                    lambda s: s.box((0, 0), 0.60, 0.50, 0.20),
                    lambda s: s.cylinder((0, 0), 0.10, 0.15, base=0.20)], target=1)
                self.assertAlmostEqual(height, 150.0, delta=10.0)      # v39: about 350 mm
                self.assertEqual(reference, "height_above_local_support")

    def test_a_bag_on_the_floor_between_taller_bags_keeps_the_floor(self):
        for tilt in (15.0, 25.0, 40.0):
            with self.subTest(tilt=tilt):
                height, reference = self._measure(tilt, [
                    lambda s: s.box((-0.2, 0), 0.3, 0.6, 0.20),
                    lambda s: s.box((0.2, 0), 0.3, 0.6, 0.45),
                    lambda s: s.cylinder((0, 0), 0.08, 0.15)], target=2)
                self.assertAlmostEqual(height, 150.0, delta=10.0)
                self.assertEqual(reference, "raised_surroundings_are_not_its_support")

    def test_an_object_on_open_floor_is_unchanged(self):
        height, reference = self._measure(25.0, [
            lambda s: s.box((0.25, 0), 0.2, 0.4, 0.4), lambda s: s.box((0, 0), 0.2, 0.2, 0.15)], target=1)
        self.assertAlmostEqual(height, 150.0, delta=10.0)
        self.assertIsNone(reference)


# ---------------------------------------------------------------- pipeline level
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.pipeline import VisionPipeline  # noqa: E402
from locallife_cloud.types import CameraIntrinsics  # noqa: E402,F811

SIZE = (240, 320)
FLOOR = 1.2
CAMERA = CameraIntrinsics(fx=300, fy=300, ppx=160, ppy=120, width=320, height=240)


class _Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, items):
        self.items = list(items)

    def detect_batch(self, frames):
        return [[Detection(d.label, d.confidence, d.box, mask=d.mask.copy()) for d in self.items]
                for _ in frames]


def _mask(r0, r1, c0, c1):
    mask = np.zeros(SIZE, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _bag(mask):
    ys, xs = np.nonzero(mask)
    return Detection("plastic waste bag", 0.8, (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                                int(ys.max()) + 1), mask=mask)


def _paint(frame, depth, mask, height_m, colour):
    frame[mask] = colour
    depth[mask] = FLOOR - height_m


def _station(directory, detector, baseline_frame=None, baseline_depth=None):
    config = AppConfig(results_dir=Path(directory), enable_monocular_depth=False,
                       enable_material_classification=False, roi=(0, 0, 1, 1),
                       tracker_confirm_frames=1)
    station = VisionPipeline(config, detector=detector)
    frame = np.full((*SIZE, 3), 60, np.uint8) if baseline_frame is None else baseline_frame
    depth = np.full(SIZE, FLOOR, np.float32) if baseline_depth is None else baseline_depth
    station.set_baseline(frame.copy(), depth.copy(), CAMERA)
    return station, frame, depth


def _pixels_to_mm(pixels, height_m):
    return pixels * (FLOOR - height_m) / CAMERA.fx * 1000.0


class CrowdedPipelineTests(unittest.TestCase):
    """RealSense path, end to end, on synthetic depth. Not a hardware result."""

    def test_two_touching_bags_are_two_objects_with_their_own_sizes(self):
        a, b = _mask(60, 160, 60, 150), _mask(70, 170, 150, 240)
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, _Detector([_bag(a), _bag(b)]))
            frame, depth = frame.copy(), depth.copy()
            _paint(frame, depth, a, 0.20, (30, 30, 30))
            _paint(frame, depth, b, 0.15, (200, 200, 200))
            for index in range(4):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
        bags = sorted((item for item in result.detections if item.track_id is not None),
                      key=lambda item: item.box[0])
        self.assertEqual(len(bags), 2)          # v39: one object over both bags
        for bag, (rows, cols, height) in zip(bags, ((100, 90, 0.20), (100, 90, 0.15))):
            longest = _pixels_to_mm(max(rows, cols), height)
            self.assertLess(bag.footprint_length_mm, 1.15 * longest)
            self.assertAlmostEqual(bag.physical_height_mm, height * 1000.0, delta=10.0)

    def test_a_pile_already_in_the_bin_plus_one_new_bag_is_one_deposit(self):
        old_a, old_c, new = _mask(60, 160, 40, 120), _mask(60, 160, 120, 200), _mask(90, 190, 200, 280)
        base_frame = np.full((*SIZE, 3), 60, np.uint8)
        base_depth = np.full(SIZE, FLOOR, np.float32)
        _paint(base_frame, base_depth, old_a, 0.20, (30, 30, 30))
        _paint(base_frame, base_depth, old_c, 0.18, (180, 90, 200))
        detector = _Detector([_bag(old_a), _bag(old_c)])
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, detector, base_frame, base_depth)
            clock = 1.0
            for _ in range(6):
                station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=clock)
                clock += 1
            frame, depth = frame.copy(), depth.copy()
            _paint(frame, depth, new, 0.15, (200, 200, 200))
            detector.items.append(_bag(new))
            for _ in range(30):
                station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=clock)
                clock += 1
            deposits = station.ledger.summary()["deposited_count"]
            events = [event.to_dict() for event in station.bin_occupancy.events]
        self.assertEqual(deposits, 1)           # the pile was never counted
        self.assertEqual(len(events), 1)        # v39: never finalised ("No finalised deposit yet")
        event = events[0]
        self.assertGreater(event["occupied_after_l"], event["occupied_before_l"])
        envelope = 0.28 * 0.35 * 0.15 * 1000.0
        self.assertLess(abs(event["delta_occupancy_l"] - envelope) / envelope, 0.25)

    def test_a_dark_bag_with_sparse_depth_does_not_take_its_neighbour(self):
        dark, light = _mask(60, 160, 60, 150), _mask(70, 170, 150, 240)
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, _Detector([_bag(dark), _bag(light)]))
            frame, depth = frame.copy(), depth.copy()
            _paint(frame, depth, dark, 0.20, (15, 15, 15))
            _paint(frame, depth, light, 0.15, (200, 200, 200))
            holes = dark & (np.random.default_rng(1).random(SIZE) < 0.7)
            depth[holes] = 0.0                  # black polythene: stereo dropout
            for index in range(4):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
        by_x = sorted((item for item in result.detections if item.track_id is not None),
                      key=lambda item: item.box[0])
        self.assertEqual(len(by_x), 2)
        dark_bag = by_x[0]
        if dark_bag.footprint_length_mm is not None:
            self.assertLess(dark_bag.footprint_length_mm, 1.15 * _pixels_to_mm(100, 0.20))
        else:
            self.assertTrue(dark_bag.volume_rejection_reason)

    def test_a_brief_dropout_keeps_the_id_and_deposits_once(self):
        bag = _mask(80, 180, 100, 200)
        detector = _Detector([_bag(bag)])
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, detector)
            frame, depth = frame.copy(), depth.copy()
            _paint(frame, depth, bag, 0.15, (200, 200, 200))
            ids, clock = [], 1.0
            for step in range(24):
                detector.items = [] if 3 <= step < 6 else [_bag(bag)]
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=clock)
                clock += 1
                ids += [item.track_id for item in result.detections if item.track_id is not None]
            deposits = station.ledger.summary()["deposited_count"]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(deposits, 1)


class DepthlessBagTests(unittest.TestCase):
    def test_a_black_bag_with_no_stereo_depth_stays_visible_without_a_volume(self):
        dark, light = _mask(60, 160, 60, 150), _mask(70, 170, 150, 240)
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, _Detector([_bag(dark), _bag(light)]))
            frame, depth = frame.copy(), depth.copy()
            _paint(frame, depth, dark, 0.20, (15, 15, 15))
            _paint(frame, depth, light, 0.15, (200, 200, 200))
            depth[dark] = 0.0                    # black polythene: no stereo return at all
            for index in range(4):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
        tracked = [item for item in result.detections if item.track_id is not None]
        self.assertEqual(len(tracked), 2)        # v40 before this fix: the black bag vanished
        black = min(tracked, key=lambda item: item.box[0])
        self.assertIsNone(black.realsense_volume_l)
        self.assertEqual(black.volume_rejection_reason, "no_valid_stereo_depth_dark_or_shiny_surface")


class LogitechDepthGainTests(unittest.TestCase):
    """A model that puts objects 15 % too near; one known-height sample fixes the next object."""

    def test_one_known_object_corrects_a_different_one(self):
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from test_v31_logitech_measurement_cascade import _run, _station as _logitech_station

        class PopDepth:
            device = "cpu"

            def estimate_batch(self, frames):
                out = []
                for f in frames:
                    d = np.full(f.shape[:2], 1.5, np.float32)
                    d[(f[..., 2] == 210) & (f[..., 0] == 40)] = 1.38 * 0.85    # 120 mm object
                    d[(f[..., 0] == 210) & (f[..., 2] == 40)] = 1.30 * 0.85    # 200 mm object
                    out.append(d)
                return out

        def place(colour, box):
            frame = np.zeros((120, 160, 3), np.uint8)
            mask = np.zeros((120, 160), bool)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
            frame[mask] = colour
            return frame, Detection("cosmetic bottle", 0.7, box, mask)

        camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
        empty = np.zeros((120, 160, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, _, _ = _logitech_station(directory, depth=PopDepth(),
                                                        logitech_reference_distance_m=1.5)
            station = manager.camera("logitech")
            detector.items = []
            station.process_frame(empty, intrinsics=camera, persist=False, timestamp=1)
            station.set_baseline()
            station._expect_logitech_pose()
            station.process_frame(empty, intrinsics=camera, persist=False, timestamp=2)
            frame, item = place((40, 40, 210), (60, 40, 110, 90))
            detector.items = [item]
            before = _run(station, frame, camera, start=10).detections[0]
            self.assertGreater(before.physical_height_mm, 300)      # 327 mm for a 120 mm object
            added = station.add_height_sample(name="known", true_length_cm=0, true_width_cm=0,
                                              true_height_cm=12.0)["depth_gain_sample"]
            self.assertAlmostEqual(added["gain"], 1 / 0.85, places=3)
            detector.items = []
            for index in range(40):
                station.process_frame(empty, intrinsics=camera, timestamp=30 + index)
            frame, item = place((210, 40, 40), (5, 5, 45, 45))
            detector.items = [item]
            after = _run(station, frame, camera, start=80).detections[0]
            self.assertAlmostEqual(after.physical_height_mm, 200.0, delta=10.0)
            self.assertIn("object_depth_gain_applied", after.dimension_flags)
            # A moved camera suspends it.
            station.logitech_pose.changed_reason = "camera_pose_changed_camera_tilt_changed"
            self.assertIsNone(station.logitech_depth_gain()[0])


class CalibrationSlipTests(unittest.TestCase):
    """What an operator sees after calibrating: the field reports, reproduced."""

    def _scene(self):
        bag = _mask(80, 180, 100, 200)
        frame = np.full((*SIZE, 3), 60, np.uint8)
        frame[::8], frame[:, ::8] = 90, 90                     # some floor texture
        depth = np.full(SIZE, FLOOR, np.float32)
        _paint(frame, depth, bag, 0.15, (200, 200, 200))
        return bag, frame, depth

    def test_an_object_in_the_captured_empty_scene_stays_visible_and_is_never_deposited(self):
        bag, frame, depth = self._scene()
        with tempfile.TemporaryDirectory() as directory:
            station, _, _ = _station(directory, _Detector([_bag(bag)]), frame, depth)
            for index in range(20):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
            deposits = station.ledger.summary()["deposited_count"]
        self.assertEqual(len(result.detections), 1)       # before: the object vanished
        self.assertIsNone(result.detections[0].realsense_volume_l)
        self.assertEqual(result.detections[0].volume_rejection_reason,
                         "object_was_in_view_when_the_empty_scene_was_captured")
        self.assertEqual(deposits, 0)

    def test_a_baseline_from_another_room_is_discarded_and_the_bag_is_measured(self):
        bag, frame, depth = self._scene()
        other_room = np.random.default_rng(0).integers(0, 255, (*SIZE, 3)).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            station, _, _ = _station(directory, _Detector([_bag(bag)]), other_room,
                                     np.full(SIZE, 0.8, np.float32))
            for index in range(6):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
        self.assertEqual(station.baseline_restore_state, "discarded_camera_view_changed")
        self.assertAlmostEqual(result.detections[0].physical_height_mm, 150.0, delta=10.0)
        self.assertIsNotNone(result.detections[0].realsense_volume_l)


class ImplausibleScaleAndRangeTests(unittest.TestCase):
    def test_a_calibration_that_puts_objects_metres_away_is_refused(self):
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from test_v31_logitech_measurement_cascade import _run, _station as _logitech_station

        with tempfile.TemporaryDirectory() as directory:
            manager, detector, frame, camera = _logitech_station(directory, logitech_reference_distance_m=1.5)
            station = manager.camera("logitech")
            items, detector.items = detector.items, []
            station.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False, timestamp=1)
            station.set_baseline()
            station.calibration.scale *= 6.0          # the field's 9-13 m for objects ~1.5 m away
            detector.items = items
            found = _run(station, frame, camera, start=10).detections[0]
        self.assertLess(found.monocular_distance_m, 2.0)
        self.assertIsNone(found.monocular_volume_l)
        self.assertTrue(found.volume_rejection_reason.startswith("metric_scale_implausible_x"))

    def _realsense(self, floor, mask, height):
        with tempfile.TemporaryDirectory() as directory:
            station, frame, depth = _station(directory, _Detector([_bag(mask)]), None,
                                             np.full(SIZE, floor, np.float32))
            frame, depth = frame.copy(), depth.copy()
            frame[mask] = (200, 200, 200)
            depth[mask] = floor - height
            for index in range(4):
                result = station.process_frame(frame, depth_m=depth, intrinsics=CAMERA, timestamp=1.0 + index)
        return result.detections[0]

    def test_an_object_beyond_the_reliable_range_is_shown_without_dimensions(self):
        far = self._realsense(3.2, _mask(100, 140, 140, 180), 0.2)
        self.assertIsNone(far.footprint_length_mm)
        self.assertIsNone(far.realsense_volume_l)
        self.assertTrue(far.volume_rejection_reason.startswith("beyond_reliable_measuring_range"))
        near = self._realsense(FLOOR, _mask(80, 180, 100, 200), 0.15)
        self.assertIsNotNone(near.realsense_volume_l)

    def test_a_room_sized_footprint_is_refused_not_published(self):
        from locallife_cloud.pipeline import VisionPipeline

        station = VisionPipeline.__new__(VisionPipeline)
        station.camera_id = "realsense"
        station.config = AppConfig()
        pillow = Detection("pillow", 0.8, (0, 0, 10, 10))
        pillow.depth_distance_m = 2.0
        pillow.footprint_length_mm, pillow.footprint_width_mm, pillow.physical_height_mm = 5888.0, 3194.0, 729.0
        pillow.realsense_volume_l = 40.0
        station._withhold_unreliable_geometry(pillow)            # the field reading
        self.assertIsNone(pillow.footprint_length_mm)
        self.assertIsNone(pillow.realsense_volume_l)
        self.assertEqual(pillow.volume_rejection_reason, "implausible_footprint_5.9m_for_one_object")


class MovedCameraFloorScaleTests(unittest.TestCase):
    """The v37 floor rescale trusted a recorded pose after the camera was moved."""

    def test_a_camera_moved_closer_to_the_floor_is_not_rescaled_to_the_old_height(self):
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from test_v31_logitech_measurement_cascade import _run, _station as _logitech_station

        class Floor:
            device = "cpu"
            floor = 1.5

            def estimate_batch(self, frames):
                return [np.where(f.max(axis=2) > 0, self.floor - 0.12, self.floor).astype(np.float32)
                        for f in frames]

        depth = Floor()
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, frame, camera = _logitech_station(directory, depth=depth,
                                                                 logitech_reference_distance_m=1.5)
            station = manager.camera("logitech")
            items, detector.items = detector.items, []
            station.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False, timestamp=1)
            station.set_baseline()
            station._expect_logitech_pose()
            station.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False, timestamp=2)
            depth.floor = 0.5                          # camera now 0.5 m up, same tilt
            detector.items = items
            found = _run(station, frame, camera, start=10, frames=10).detections[0]
        # Before: floor rescaled x3.0 -> 1.14 m away and 360 mm tall for a 120 mm object.
        self.assertAlmostEqual(found.monocular_distance_m, 0.38, delta=0.02)
        self.assertIsNone(found.physical_height_mm)
        self.assertIsNone(found.monocular_volume_l)
        self.assertEqual(found.volume_rejection_reason, "camera_pose_changed_floor_distance")


if __name__ == "__main__":
    unittest.main()
