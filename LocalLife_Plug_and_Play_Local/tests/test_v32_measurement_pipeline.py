"""V32 (second pass): measure the deposit, not the room it was placed in.

The hardware evidence was no longer a shape-formula problem: one detector mask
covered two cartons, the floor, a chair and the wall, and whatever formula ran
on it reported 3.9-5.4 L of background as an object. These tests cover the
repair -- a per-camera measurement zone with a real floor scale, a measurement
mask taken from what changed since the scene was committed, and a deposit that
is measured incrementally.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.deposit_state import (
    FINALIZED,
    MEASURING,
    OBJECT_ENTERING,
    WAITING_FOR_CHANGE,
    WAITING_FOR_STABILITY,
    DepositStateMachine,
    FrameObservation,
)
from locallife_cloud.measurement_mask import (
    DETECTOR_MASK,
    FOREGROUND_COMPONENT,
    NO_NEW_DEPOSIT,
    OUTSIDE_ZONE,
    deposit_component,
    height_change,
    looks_like_background,
    rgb_change,
)
from locallife_cloud.measurement_zone import (
    MeasurementZone,
    MeasurementZoneStore,
    quadrilateral_is_sane,
    zone_from_rectangle,
)
from locallife_cloud.readiness import MeasurementReadiness
from locallife_cloud.types import CameraIntrinsics, Detection

PROJECT = Path(__file__).resolve().parents[1]
V32_SHA = "fc37fa5ebdf6166a24ee9a1c78c84e2e7c58b964"
SHAPE = (120, 160)


def _mask(top: int, bottom: int, left: int, right: int, shape=SHAPE) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _zone_mask(shape=SHAPE) -> np.ndarray:
    """A mat that does not reach the edges of the frame."""
    return _mask(20, 110, 20, 140, shape)


class Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.items: list[Detection] = []

    def detect_batch(self, frames):
        return [[Detection(item.label, item.confidence, item.box, item.mask.copy(), color=item.color)
                 for item in self.items] for _ in frames]


class MetricDepth:
    device = "cpu"

    def estimate_batch(self, frames):
        return [np.where(frame.max(axis=2) > 0, 1.38, 1.5).astype(np.float32) for frame in frames]


def _station(directory: str, **overrides):
    detector = Detector()
    settings = dict(
        results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
        tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
        operating_mode="geometry_validation", auto_deposit=False,
        logitech_reference_distance_m=0.0, automatic_baseline=False,
    )
    settings.update(overrides)
    manager = DualCameraCoordinator(AppConfig(**settings), detector=detector,
                                    depth_estimator=MetricDepth())
    camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=160, height=120)
    return manager, detector, camera


class MeasurementZoneTests(unittest.TestCase):
    """The mat, per camera, at the resolution it was drawn at."""

    def _zone(self, camera: str = "logitech", **overrides) -> MeasurementZone:
        values = dict(
            camera=camera, corners=((40.0, 400.0), (600.0, 400.0), (460.0, 180.0), (180.0, 180.0)),
            width_px=640, height_px=480, near_edge_m=1.0, depth_edge_m=0.8,
        )
        values.update(overrides)
        return MeasurementZone(**values)

    def test_each_camera_keeps_its_own_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MeasurementZoneStore(Path(directory))
            store.save(self._zone("logitech"))
            store.save(self._zone("realsense", corners=((10.0, 470.0), (630.0, 470.0),
                                                        (500.0, 60.0), (140.0, 60.0))))
            reopened = MeasurementZoneStore(Path(directory))
            logitech, realsense = reopened.get("logitech"), reopened.get("realsense")
            self.assertIsNotNone(logitech)
            self.assertNotEqual(logitech.corners, realsense.corners)
            # The same normalised rectangle would have measured a different
            # part of each room, which is why the zones are never shared.
            self.assertFalse(np.array_equal(logitech.mask((480, 640)), realsense.mask((480, 640))))
            self.assertIsNone(reopened.get("thermal"))

    def test_a_zone_scales_to_the_same_framing_and_refuses_another(self) -> None:
        zone = self._zone()
        smaller = zone.for_shape((240, 320))
        self.assertIsNotNone(smaller)
        self.assertAlmostEqual(smaller.corners[0][0], 20.0)
        self.assertAlmostEqual(smaller.corners[0][1], 200.0)
        # 16:9 is a different crop of the room, not the same mat scaled.
        self.assertIsNone(zone.for_shape((720, 1280)))
        self.assertIsNone(zone.mask((720, 1280)))

    def test_the_floor_scale_grows_with_distance(self) -> None:
        zone = self._zone()
        areas = zone.pixel_area_m2((480, 640))
        self.assertIsNotNone(areas)
        mask = zone.mask((480, 640))
        near = float(np.median(areas[380:395][mask[380:395]]))
        far = float(np.median(areas[190:205][mask[190:205]]))
        # A pixel at the back of the mat covers several times the ground of a
        # pixel at the front: one global pixels-per-cm factor cannot be right.
        self.assertGreater(far, 2.0 * near)
        # The whole mat still integrates to its real area.
        self.assertAlmostEqual(float(areas[mask].sum()), 1.0 * 0.8, delta=0.05)

    def test_a_zone_without_its_real_size_has_no_floor_scale(self) -> None:
        zone = self._zone(near_edge_m=0.0, depth_edge_m=0.0)
        self.assertFalse(zone.has_floor_scale)
        self.assertIsNone(zone.homography())
        self.assertIsNone(zone.pixel_area_m2((480, 640)))
        self.assertIsNotNone(zone.mask((480, 640)))

    def test_a_footprint_is_measured_on_the_mat_not_in_pixels(self) -> None:
        zone = self._zone()
        near = _mask(360, 400, 300, 340, (480, 640))
        far = _mask(200, 240, 300, 340, (480, 640))
        near_size = zone.footprint_m(near)
        far_size = zone.footprint_m(far)
        self.assertIsNotNone(near_size)
        # The same pixel patch is a much larger object at the back of the mat.
        self.assertGreater(far_size[2], near_size[2])

    def test_corners_must_make_a_quadrilateral(self) -> None:
        self.assertTrue(quadrilateral_is_sane([(0, 0), (10, 0), (10, 10), (0, 10)]))
        self.assertFalse(quadrilateral_is_sane([(0, 0), (10, 0), (5, 0), (0, 10)]))
        self.assertFalse(quadrilateral_is_sane([(0, 0), (10, 0), (10, 10)]))
        with self.assertRaises(ValueError):
            MeasurementZone(camera="logitech", corners=((0, 0), (1, 0), (1, 1), (0, 1)),
                            width_px=640, height_px=480)

    def test_the_configured_roi_is_usable_as_a_zone(self) -> None:
        zone = zone_from_rectangle("realsense", (0.1, 0.1, 0.8, 0.8), (480, 640))
        mask = zone.mask((480, 640))
        self.assertAlmostEqual(int(np.count_nonzero(mask)) / (480 * 640), 0.64, delta=0.02)


class MeasurementMaskTests(unittest.TestCase):
    """What is measured is what changed, localised by the detector."""

    def test_a_giant_detector_mask_is_cut_back_to_what_arrived(self) -> None:
        zone = _zone_mask()
        # The reported failure: one "beverage carton" mask over both cartons,
        # the floor between them, the chair and the wall.
        detector = _mask(25, 105, 25, 135)
        change = _mask(40, 80, 50, 80)
        choice = deposit_component(detector, change, zone, min_pixels=20)
        self.assertEqual(choice.source, FOREGROUND_COMPONENT)
        self.assertLess(int(np.count_nonzero(choice.mask)), int(np.count_nonzero(detector)) / 3)
        self.assertFalse(choice.mask[95, 130])
        self.assertTrue(choice.mask[60, 65])

    def test_two_cartons_are_measured_separately(self) -> None:
        zone = _zone_mask()
        change = _mask(40, 80, 40, 70) | _mask(40, 80, 95, 125)
        left = deposit_component(_mask(40, 80, 40, 70), change, zone, min_pixels=20)
        right = deposit_component(_mask(40, 80, 95, 125), change, zone, min_pixels=20)
        self.assertEqual(left.diagnostics["changed_components"], 2)
        self.assertFalse(np.any(left.mask & right.mask))
        for choice in (left, right):
            self.assertEqual(int(np.count_nonzero(choice.mask)), 40 * 30)

    def test_the_second_deposit_is_measured_on_its_own(self) -> None:
        """The first carton is in the committed scene, so only the second changed."""
        zone = _zone_mask()
        committed = np.zeros((*SHAPE, 3), dtype=np.uint8)
        committed[_mask(40, 80, 40, 70)] = (30, 120, 60)   # carton one, already counted
        frame = committed.copy()
        frame[_mask(40, 80, 95, 125)] = (30, 120, 60)      # carton two arrives
        change = rgb_change(frame, committed, 18)
        # A detector mask that has merged both cartons:
        merged = _mask(40, 80, 40, 125)
        choice = deposit_component(merged, change, zone, min_pixels=20)
        self.assertEqual(choice.source, FOREGROUND_COMPONENT)
        self.assertEqual(int(np.count_nonzero(choice.mask)), 40 * 30)
        self.assertFalse(np.any(choice.mask[:, :95]))

    def test_furniture_already_in_the_committed_scene_is_not_measured(self) -> None:
        zone = _zone_mask()
        committed = np.zeros((*SHAPE, 3), dtype=np.uint8)
        chair = _mask(30, 100, 30, 90)
        committed[chair] = (90, 90, 90)
        change = rgb_change(committed.copy(), committed, 18)
        choice = deposit_component(chair, change, zone, min_pixels=20)
        self.assertIsNone(choice.mask)
        self.assertEqual(choice.reason, NO_NEW_DEPOSIT)

    def test_a_person_outside_the_zone_is_not_measured(self) -> None:
        zone = _zone_mask()
        person = _mask(0, 18, 0, 18)
        change = person.copy()
        choice = deposit_component(person, change, zone, min_pixels=20)
        self.assertIsNone(choice.mask)
        self.assertEqual(choice.reason, OUTSIDE_ZONE)

    def test_a_shadow_raises_nothing_and_is_refused(self) -> None:
        zone = _zone_mask()
        detector = _mask(40, 80, 50, 90)
        change = detector.copy()
        rise = np.zeros(SHAPE, dtype=np.float32)
        rise[detector] = 0.002
        choice = deposit_component(detector, change, zone, rise_m=rise,
                                   min_pixels=20, min_height_rise_m=0.01)
        self.assertIsNone(choice.mask)
        self.assertEqual(choice.reason, "no_height_rise_above_the_committed_scene")

    def test_an_object_that_stands_above_the_scene_is_measured(self) -> None:
        zone = _zone_mask()
        detector = _mask(40, 80, 50, 90)
        rise = np.zeros(SHAPE, dtype=np.float32)
        rise[detector] = 0.12
        choice = deposit_component(detector, detector.copy(), zone, rise_m=rise,
                                   min_pixels=20, min_height_rise_m=0.01)
        self.assertEqual(choice.source, FOREGROUND_COMPONENT)
        self.assertAlmostEqual(choice.diagnostics["height_rise_m"], 0.12, places=3)

    def test_without_a_committed_scene_the_detector_mask_still_measures(self) -> None:
        """The V31 numeric fallback is preserved when nothing can be differenced."""
        zone = _zone_mask()
        detector = _mask(40, 80, 50, 90)
        choice = deposit_component(detector, None, zone, min_pixels=20)
        self.assertEqual(choice.source, DETECTOR_MASK)
        self.assertTrue(choice.measurable)

    def test_background_sized_masks_are_named_not_measured(self) -> None:
        zone = _zone_mask()
        self.assertIsNotNone(looks_like_background(zone.copy(), zone))
        self.assertIsNone(looks_like_background(_mask(40, 80, 50, 90), zone))
        floor = _mask(60, 110, 20, 140)
        self.assertIsNotNone(looks_like_background(floor, zone))

    def test_height_change_reads_a_rise_not_a_distance(self) -> None:
        committed = np.full(SHAPE, 1.5, dtype=np.float32)
        current = committed.copy()
        current[_mask(40, 80, 50, 90)] = 1.38
        risen = height_change(current, committed, min_rise_m=0.05)
        self.assertTrue(risen[60, 70])
        self.assertFalse(risen[10, 10])
        self.assertIsNone(height_change(current, None))


class DepositStateTests(unittest.TestCase):
    def test_one_deposit_walks_through_every_step(self) -> None:
        machine = DepositStateMachine(settle_frames=2)
        self.assertEqual(machine.state, WAITING_FOR_CHANGE)
        self.assertEqual(machine.observe(FrameObservation(changed_fraction=0.08)), OBJECT_ENTERING)
        machine.observe(FrameObservation(changed_fraction=0.08, tracked_objects=1))
        self.assertEqual(
            machine.observe(FrameObservation(changed_fraction=0.08, tracked_objects=1)),
            WAITING_FOR_STABILITY,
        )
        self.assertEqual(
            machine.observe(FrameObservation(changed_fraction=0.08, tracked_objects=1,
                                             measured_volume_l=1.2)),
            MEASURING,
        )
        self.assertEqual(
            machine.observe(FrameObservation(changed_fraction=0.08, tracked_objects=1,
                                             measured_volume_l=1.2, stable=True)),
            FINALIZED,
        )
        self.assertTrue(machine.describe()["awaiting_commit"])
        machine.committed()
        self.assertEqual(machine.state, WAITING_FOR_CHANGE)

    def test_an_empty_zone_returns_to_waiting(self) -> None:
        machine = DepositStateMachine(settle_frames=2)
        machine.observe(FrameObservation(changed_fraction=0.09))
        self.assertEqual(machine.observe(FrameObservation(changed_fraction=0.0)), WAITING_FOR_CHANGE)


class ReadinessTests(unittest.TestCase):
    def test_every_prerequisite_is_named_with_its_next_step(self) -> None:
        readiness = MeasurementReadiness(camera="logitech", measurement_zone=True,
                                         empty_baseline=True, intrinsics=True)
        payload = readiness.to_dict()
        self.assertEqual(payload["measurement_zone"], "ready")
        self.assertEqual(payload["floor_scale"], "missing")
        self.assertFalse(payload["fully_calibrated"])
        self.assertIn("mat's real width", payload["next_step"])
        self.assertIn("floor scale", payload["summary"])

    def test_a_calibrated_camera_says_so(self) -> None:
        readiness = MeasurementReadiness(
            camera="realsense", measurement_zone=True, empty_baseline=True, intrinsics=True,
            floor_scale=True, metric_depth_mapping=True, method="hardware-depth",
        )
        self.assertTrue(readiness.fully_calibrated)
        self.assertEqual(readiness.missing, ())
        self.assertIn("calibrated", readiness.summary())


class PipelineIntegrationTests(unittest.TestCase):
    def test_the_state_reports_readiness_the_zone_and_the_deposit_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            state = manager.camera("logitech").state()
            self.assertIn("measurement_readiness", state)
            self.assertEqual(state["measurement_readiness"]["camera"], "logitech")
            self.assertEqual(state["measurement_zone"], None)
            self.assertIn(state["measurement_zone_source"], ("configured-roi", "configured-polygon"))
            self.assertEqual(state["deposit_state"]["state"], "WAITING_FOR_CHANGE")

    def test_a_saved_zone_replaces_the_configured_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            logitech = manager.camera("logitech")
            described = logitech.set_measurement_zone(
                [(20.0, 110.0), (140.0, 110.0), (120.0, 30.0), (40.0, 30.0)], SHAPE,
                near_edge_m=0.9, depth_edge_m=0.6,
            )
            self.assertEqual(described["floor_scale"], "ready")
            region = logitech._measurement_region((*SHAPE, 3))
            self.assertLess(int(np.count_nonzero(region)), SHAPE[0] * SHAPE[1])
            self.assertFalse(region[5, 5])
            self.assertTrue(region[100, 80])
            # The other camera is unaffected: it has its own view of the mat.
            self.assertIsNone(manager.camera("realsense").measurement_zone)
            reopened = MeasurementZoneStore(Path(directory) / "logitech" / "calibration")
            self.assertIsNotNone(reopened.get("logitech"))

    def test_relative_depth_is_never_used_as_metres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(
                directory, depth_model="depth-anything/Depth-Anything-V2-Small-hf",
                logitech_reference_distance_m=1.5,
            )
            logitech = manager.camera("logitech")
            region = np.ones(SHAPE, dtype=bool)
            # A relative checkpoint's inverse depth: larger is nearer.
            relative = np.full(SHAPE, 4.0, dtype=np.float32)
            relative[_mask(40, 80, 50, 90)] = 5.0
            metres = logitech._metric_from_relative(relative, region)
            self.assertEqual(logitech.relative_depth_reason,
                             "relative_depth_scaled_to_measured_floor_distance")
            self.assertAlmostEqual(float(np.median(metres)), 1.5, places=2)
            # The nearer object is nearer in metres too, and nothing is 4.0 m.
            self.assertLess(float(metres[60, 70]), 1.5)
            self.assertEqual(logitech.calibration_mode, "relative-depth-estimate")

    def test_relative_depth_without_a_reference_distance_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(
                directory, depth_model="depth-anything/Depth-Anything-V2-Small-hf",
            )
            logitech = manager.camera("logitech")
            relative = np.full(SHAPE, 4.0, dtype=np.float32)
            unchanged = logitech._metric_from_relative(relative, np.ones(SHAPE, dtype=bool))
            self.assertEqual(logitech.relative_depth_reason,
                             "relative_depth_without_reference_distance")
            self.assertTrue(np.array_equal(unchanged, relative))
            self.assertEqual(logitech.state()["measurement_readiness"]["reason"],
                             "relative_depth_without_reference_distance")

    def test_a_metric_checkpoint_is_passed_through_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, _, _ = _station(directory)
            logitech = manager.camera("logitech")
            prediction = np.full(SHAPE, 1.42, dtype=np.float32)
            self.assertTrue(np.array_equal(
                logitech._metric_from_relative(prediction, np.ones(SHAPE, dtype=bool)), prediction,
            ))
            self.assertIsNone(logitech.relative_depth_reason)

    def test_an_object_is_still_measured_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, camera = _station(directory)
            mask = _mask(40, 90, 60, 110)
            frame = np.zeros((*SHAPE, 3), dtype=np.uint8)
            frame[mask] = (40, 40, 210)
            detector.items = [Detection("cosmetic bottle", 0.7, (60, 40, 110, 90), mask, color="red")]
            logitech = manager.camera("logitech")
            for index in range(6):
                result = logitech.process_frame(frame, intrinsics=camera, timestamp=10.0 + index)
            measured = result.detections[0]
            self.assertIsNone(measured.volume_rejection_reason)
            self.assertGreater(measured.monocular_volume_l, 0)
            self.assertIn(logitech.deposit_state.state,
                          ("OBJECT_ENTERING", "WAITING_FOR_STABILITY", "MEASURING", "FINALIZED"))


class ProtectedSurfacesTests(unittest.TestCase):
    def test_csv_ledger_detector_and_cloud_are_unchanged_since_v32(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/heightmap_volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/footprint.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/tracking.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/inference.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/vocabulary.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(["git", "diff", "--name-only", V32_SHA, "--", *protected],
                                capture_output=True, text=True, cwd=PROJECT.parent, timeout=120)
        if result.returncode != 0:
            self.skipTest("git or the V32 commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
