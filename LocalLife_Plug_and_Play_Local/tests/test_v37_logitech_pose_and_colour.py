"""V37: the Logitech at its real, downward installation angle, and one colour definition.

Every scene here is ray-traced: a camera 0.62 m above the floor, tilted 25
degrees from straight down (65 below horizontal, the installation) or 70
degrees (a shallow, front-facing view), with objects of known size on a known
floor. The depth model is modelled as true depth times 1.4, which is roughly
how much further the field Logitech put objects than the RealSense did. These
are synthetic numbers about the code's geometry, not about the C920: they say
nothing about the accuracy of the real camera.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from locallife_cloud.colour_evidence import describe_colour
from locallife_cloud.geometry import classify_color
from locallife_cloud.logitech_pose import (
    PoseGuard, floor_tilt_deg, plane_height_scale, result_record, sanity_flags,
)
from locallife_cloud.logitech_volume import fit_plane_alignment, metric_object_volume
from locallife_cloud.material_evidence import reconcile_material
from locallife_cloud.volume import fit_reference_plane

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v35_logitech_metric_geometry as rig  # noqa: E402

CAMERA = rig.CAMERA
HEIGHT_M = 0.62
MODEL_SCALE_ERROR = 1.4
DOWNWARD, SHALLOW = 25.0, 70.0      # degrees from straight down


def _rays(tilt_deg: float):
    tilt = math.radians(tilt_deg)
    centre = np.array((0.0, -HEIGHT_M * math.tan(tilt), HEIGHT_M))   # axis meets the floor at y = 0
    forward = np.array((0.0, math.sin(tilt), -math.cos(tilt)))
    right = np.array((1.0, 0.0, 0.0))
    down = np.cross(forward, right)
    rows, columns = np.indices((rig.HEIGHT, rig.WIDTH), dtype=np.float64)
    x = (columns - CAMERA.ppx) / CAMERA.fx
    y = (rows - CAMERA.ppy) / CAMERA.fy
    return centre, x[..., None] * right + y[..., None] * down + forward


class Scene:
    def __init__(self, tilt_deg: float) -> None:
        self.origin, self.direction = _rays(tilt_deg)
        floor = rig._floor_depth(self.origin, self.direction)
        self.region = np.isfinite(floor) & (floor < 3.0)
        self.floor = floor
        self.empty = np.where(self.region, floor, 0.0)

    def box(self, centre, length, width, height, base=0.0):
        if base:
            raise ValueError("boxes stand on the floor here")
        return rig._box_depth(self.origin, self.direction, centre, length, width, height)

    def cylinder(self, centre, radius, height, base=0.0):
        return rig._cylinder_depth(self.origin, self.direction, centre, radius, height, base_m=base)

    def render(self, *surfaces):
        nearest = self.floor.copy()
        mask = np.zeros(nearest.shape, dtype=bool)
        for surface in surfaces:
            closer = surface < nearest
            nearest = np.where(closer, surface, nearest)
            mask |= closer
        return np.where(np.isfinite(nearest), nearest, 0.0), mask


def _measure(depth, empty, mask, region, reference=None):
    plane = fit_reference_plane(empty, CAMERA, mask=region)
    result = metric_object_volume(
        depth, CAMERA, mask, plane, reference_depth_m=empty if reference is None else reference,
        min_height_m=0.004, min_pixels=25, cell_size_m=0.005,
    )
    diagnostics = result.diagnostics
    return result, (diagnostics.get("length_mm"), diagnostics.get("width_mm"),
                    None if diagnostics.get("height_p90_m") is None
                    else diagnostics["height_p90_m"] * 1000.0)


def _median_anchor(predicted_empty, region):
    """The old reference-distance anchor: measured height over median depth."""
    return HEIGHT_M / float(np.median(predicted_empty[region]))


class ScaleAnchorTests(unittest.TestCase):
    """The same bag, measured looking down and looking across."""

    BAG = (0.265, 0.142, 0.089)

    def _bag(self, tilt):
        scene = Scene(tilt)
        depth, mask = scene.render(scene.box((0.0, 0.0), *self.BAG))
        return scene, depth * MODEL_SCALE_ERROR, scene.empty * MODEL_SCALE_ERROR, mask

    def _relative_errors(self, dims):
        return [abs(value - truth * 1000.0) / (truth * 1000.0) for value, truth in zip(dims, self.BAG)]

    def test_the_old_median_anchor_depends_on_the_viewing_angle(self) -> None:
        errors = {}
        for tilt in (DOWNWARD, SHALLOW):
            scene, depth, empty, mask = self._bag(tilt)
            scale = _median_anchor(empty, scene.region)
            _, dims = _measure(depth * scale, empty * scale, mask, scene.region)
            errors[tilt] = max(self._relative_errors(dims))
        # Evidence of the bug, kept: shrinks a little looking down, by half across.
        self.assertGreater(errors[DOWNWARD], 0.05)
        self.assertGreater(errors[SHALLOW], 0.40)

    def test_the_plane_anchor_is_right_at_both_angles(self) -> None:
        for tilt in (DOWNWARD, SHALLOW):
            with self.subTest(tilt=tilt):
                scene, depth, empty, mask = self._bag(tilt)
                calibration, diagnostics = plane_height_scale(empty, scene.region, CAMERA, HEIGHT_M)
                self.assertIsNotNone(calibration, diagnostics)
                self.assertAlmostEqual(calibration.scale, 1.0 / MODEL_SCALE_ERROR, delta=0.01)
                _, dims = _measure(calibration.apply(depth), calibration.apply(empty), mask, scene.region)
                for error in self._relative_errors(dims):
                    self.assertLess(error, 0.10, dims)

    def test_without_an_anchor_the_model_scale_error_passes_straight_through(self) -> None:
        # Why an unanchored Logitech is provisional: every dimension is off by
        # the model's own scale error, whatever the angle.
        scene, depth, empty, mask = self._bag(DOWNWARD)
        _, dims = _measure(depth, empty, mask, scene.region)
        self.assertAlmostEqual(dims[2] / 89.0, MODEL_SCALE_ERROR, delta=0.1)

    def test_the_axis_alignment_flattens_a_tilted_floor(self) -> None:
        scene, depth, empty, mask = self._bag(DOWNWARD)
        self.assertGreater(floor_tilt_deg(empty, scene.region, CAMERA), 5.0)
        old, _ = fit_plane_alignment(empty, scene.region, CAMERA, HEIGHT_M)
        result, _ = _measure(old.apply(depth), old.apply(empty), mask, scene.region)
        self.assertEqual(result.reason, "no_measurable_height_above_plane")

    def test_a_missing_floor_gives_no_anchor_rather_than_an_invented_one(self) -> None:
        scene = Scene(DOWNWARD)
        calibration, diagnostics = plane_height_scale(
            np.zeros_like(scene.empty), scene.region, CAMERA, HEIGHT_M)
        self.assertIsNone(calibration)
        self.assertIn("reason", diagnostics)


class ObjectGeometryTests(unittest.TestCase):
    def test_a_bottle_at_the_installation_angle(self) -> None:
        scene = Scene(DOWNWARD)
        depth, mask = scene.render(scene.cylinder((0.0, 0.0), 0.039, 0.203))
        _, (length, width, height) = _measure(depth, scene.empty, mask, scene.region)
        self.assertAlmostEqual(height, 203.0, delta=20.0)
        self.assertLess(max(length, width), 78.0 * 1.25)

    def test_the_floor_homography_must_not_take_elevated_pixels(self) -> None:
        for tilt in (DOWNWARD, SHALLOW):
            with self.subTest(tilt=tilt):
                scene = Scene(tilt)
                depth, mask = scene.render(scene.cylinder((0.0, 0.0), 0.039, 0.203))
                t = -scene.origin[2] / scene.direction[..., 2]
                on_floor = scene.origin + t[..., None] * scene.direction
                y = on_floor[..., 1][mask]
                # Every bottle pixel sent to the floor: the bottle's shadow.
                self.assertGreater((y.max() - y.min()) * 1000.0, 2.0 * 78.0)
                _, (length, width, _) = _measure(depth, scene.empty, mask, scene.region)
                self.assertLess(max(length, width), 78.0 * 1.25)

    def test_a_flat_cable_stays_flat(self) -> None:
        for tilt in (DOWNWARD, SHALLOW):
            with self.subTest(tilt=tilt):
                scene = Scene(tilt)
                depth, mask = scene.render(scene.box((0.0, 0.0), 0.40, 0.008, 0.008))
                _, (_, _, height) = _measure(depth, scene.empty, mask, scene.region)
                self.assertLess(height, 20.0)

    def test_an_irregular_bag_reports_its_tallest_part_and_less_than_its_envelope(self) -> None:
        scene = Scene(DOWNWARD)
        depth, mask = scene.render(scene.box((0.0, 0.0), 0.20, 0.15, 0.12),
                                   scene.box((0.12, 0.0), 0.10, 0.12, 0.06))
        result, (length, width, height) = _measure(depth, scene.empty, mask, scene.region)
        self.assertAlmostEqual(height, 120.0, delta=12.0)
        self.assertLess(result.measurement.liters, length * width * height / 1e6)

    def test_a_bag_on_earlier_waste_is_measured_from_it_and_flagged(self) -> None:
        scene = Scene(DOWNWARD)
        lower = scene.box((0.0, 0.0), 0.30, 0.25, 0.03)
        depth, everything = scene.render(lower, scene.cylinder((0.0, 0.0), 0.06, 0.05, base=0.03))
        before, _ = scene.render(lower)
        upper = everything & (depth < before - 1e-6)
        result, (_, _, height) = _measure(depth, scene.empty, upper, scene.region)
        self.assertAlmostEqual(height, 50.0, delta=8.0)
        flags = sanity_flags("plastic garbage bag", length_mm=120, width_mm=120, height_mm=height,
                             local_floor_offset_m=result.diagnostics.get("local_floor_offset_m"))
        self.assertIn("support_surface_not_floor_uncertain", flags)

    def test_a_cropped_object_is_flagged_not_clamped(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:20, 0:8] = True
        flags = sanity_flags("bag", length_mm=100, width_mm=50, height_mm=40, mask=mask)
        self.assertIn("object_cropped_by_frame_border", flags)

    def test_an_implausible_cable_height_is_flagged(self) -> None:
        flags = sanity_flags("usb cable", length_mm=400, width_mm=10, height_mm=180)
        self.assertIn("thin_object_height_implausible", flags)
        self.assertIn("metric_scale_unanchored",
                      sanity_flags("bag", length_mm=1, width_mm=1, height_mm=1, scale_anchored=False))

    def test_the_result_record_names_every_link(self) -> None:
        scene = Scene(DOWNWARD)
        depth, mask = scene.render(scene.box((0.0, 0.0), 0.265, 0.142, 0.089))
        result, dims = _measure(depth, scene.empty, mask, scene.region)
        record = result_record(
            diagnostics=result.diagnostics, depth=depth, mask=mask, depth_type="metric:test",
            pose={"state": "recorded", "reason": None}, calibration_valid=False,
            raw_volume_l=result.measurement.liters, calibrated_volume_l=None,
            dimensions_mm=dims, rejection=None,
        )
        for key in ("pose_state", "calibration_valid", "mask_area_px", "support_source", "depth_type",
                    "valid_depth_fraction", "raw_depth_range_m", "height_distribution_m",
                    "dimensions_mm", "raw_volume_l", "calibrated_volume_l", "rejection_reason"):
            self.assertIn(key, record)
        self.assertEqual(record["valid_depth_fraction"], 1.0)


class PoseGuardTests(unittest.TestCase):
    def _plane(self, tilt, height=HEIGHT_M):
        scene = Scene(tilt)
        return fit_reference_plane(scene.empty * height / HEIGHT_M, CAMERA, mask=scene.region)

    def test_a_moved_camera_is_detected_after_consecutive_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            guard = PoseGuard(path)
            guard.expect()
            guard.observe(self._plane(DOWNWARD), (640, 480))
            self.assertEqual(guard.state, "recorded")
            moved = self._plane(DOWNWARD + 15.0)
            self.assertIsNone(guard.observe(moved, (640, 480)))       # one frame is not enough
            guard.observe(moved, (640, 480))
            self.assertEqual(guard.observe(moved, (640, 480)), "camera_pose_changed_camera_tilt_changed")
            # Back where it was set up: the suspension lifts.
            self.assertIsNone(guard.observe(self._plane(DOWNWARD), (640, 480)))
            # The recorded pose survives a restart.
            self.assertIsNotNone(PoseGuard(path).recorded)

    def test_a_small_wobble_is_not_a_move(self) -> None:
        guard = PoseGuard(None)
        guard.record(self._plane(DOWNWARD), (640, 480))
        for _ in range(4):
            self.assertIsNone(guard.observe(self._plane(DOWNWARD + 2.0), (640, 480)))

    def test_height_and_resolution_changes_are_moves(self) -> None:
        guard = PoseGuard(None)
        guard.record(self._plane(DOWNWARD), (640, 480))
        self.assertEqual(guard.observe(self._plane(DOWNWARD, 0.80), (640, 480), immediate=True),
                         "camera_pose_changed_camera_height_changed")
        guard.record(self._plane(DOWNWARD), (640, 480))
        self.assertEqual(guard.observe(self._plane(DOWNWARD), (1280, 720), immediate=True),
                         "camera_pose_changed_resolution_changed")

    def test_the_pipeline_suspends_a_height_mapping_from_another_pose(self) -> None:
        from locallife_cloud.config import AppConfig
        from locallife_cloud.pipeline import VisionPipeline

        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(results_dir=Path(directory), detector_model="local-opencv-background",
                               enable_monocular_depth=False, enable_material_classification=False)
            pipeline = VisionPipeline(config, camera_id="logitech")
            fitted = mock.Mock(setup_snapshot=None)
            pipeline.metric_store = mock.Mock(calibration=fitted, setup=None)
            with mock.patch.object(pipeline, "_camera_setup",
                                   return_value=mock.Mock(difference=lambda other: None)):
                pipeline._validate_height_calibration()
                self.assertIs(pipeline.height_calibration, fitted)
                pipeline.logitech_pose.changed_reason = "camera_pose_changed_camera_tilt_changed"
                pipeline._validate_height_calibration()
            self.assertIsNone(pipeline.height_calibration)
            self.assertEqual(pipeline.height_calibration_reason,
                             "calibration_invalidated_camera_pose_changed_camera_tilt_changed")


def _frame(size=(120, 120), colour=(128, 128, 128)):
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (40, 90, 40)          # a green mat
    return frame


def _bottle(cap_at_edge: bool):
    """A grey bottle with a red cap: on its silhouette's edge, or in its middle."""
    frame = _frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[20:100, 40:80] = True
    frame[mask] = (150, 150, 150)
    if cap_at_edge:
        frame[20:32, 40:80] = (30, 30, 210)        # seen from the side: the top rows
    else:
        frame[52:68, 52:68] = (30, 30, 210)        # seen from above: the middle
    return frame, mask


class ColourTests(unittest.TestCase):
    def test_the_old_rule_answered_by_viewpoint(self) -> None:
        side, mask = _bottle(cap_at_edge=True)
        top, _ = _bottle(cap_at_edge=False)
        self.assertNotEqual(classify_color(side, mask)[0], classify_color(top, mask)[0])

    def test_one_definition_gives_grey_with_a_red_accent_from_both_views(self) -> None:
        for edge in (True, False):
            with self.subTest(cap_at_edge=edge):
                frame, mask = _bottle(edge)
                evidence = describe_colour(frame, mask, label="cream bottle")
                self.assertEqual(evidence.colour, "grey")
                self.assertEqual(evidence.accent, "red")

    def test_glare_does_not_make_an_object_white(self) -> None:
        frame = _frame()
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[20:100, 20:100] = True
        frame[mask] = (40, 40, 180)                   # red
        frame[20:60, 20:100] = (252, 252, 252)        # half of it specular glare
        evidence = describe_colour(frame, mask, label="box")
        self.assertEqual(evidence.colour, "red")
        self.assertGreater(evidence.glare_pixels, 0)

    def test_a_deep_shadow_does_not_make_an_object_black(self) -> None:
        frame = _frame()
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[20:100, 20:100] = True
        frame[mask] = (180, 60, 30)                   # blue
        frame[20:60, 20:100] = (12, 10, 8)            # half in deep shadow
        self.assertEqual(describe_colour(frame, mask, label="box").colour, "blue")

    def test_a_transparent_bag_is_called_transparent(self) -> None:
        background = _frame()
        frame = background.copy()
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[20:100, 20:100] = True
        frame[mask] = np.clip(frame[mask].astype(int) + 6, 0, 255).astype(np.uint8)
        evidence = describe_colour(frame, mask, label="plastic garbage bag", background_bgr=background)
        self.assertEqual(evidence.colour, "transparent")
        self.assertEqual(describe_colour(frame, mask, label="shoe", background_bgr=background).colour,
                         "unknown")

    def test_a_patterned_bag_is_mixed_not_a_guess(self) -> None:
        frame = _frame()
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[10:110, 10:110] = True
        stripes = np.array([(30, 30, 210), (200, 60, 30), (30, 200, 230), (160, 40, 160)], dtype=np.uint8)
        for index, row in enumerate(range(10, 110, 5)):
            frame[row:row + 5, 10:110] = stripes[index % 4]
        evidence = describe_colour(frame, mask, label="box")
        self.assertIn(evidence.colour, ("mixed", "unknown"))
        self.assertLess(evidence.confidence, 0.35)

    def test_an_object_that_is_mostly_white_or_black_keeps_its_colour(self) -> None:
        for value, name in ((245, "white"), (0, "black")):
            frame = _frame()
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[20:100, 20:100] = True
            frame[mask] = value
            self.assertEqual(describe_colour(frame, mask, label="bag").colour, name)

    def test_too_little_visible_surface_is_unknown(self) -> None:
        frame = _frame()
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[20:100, 20:100] = True
        frame[mask] = (30, 30, 210)
        frame[20:65, 20:100] = (254, 254, 254)       # glare over half
        frame[65:100, 20:100] = (5, 5, 5)            # deep shadow over the rest
        self.assertEqual(describe_colour(frame, mask).colour, "unknown")


class MaterialTests(unittest.TestCase):
    def test_a_handbag_is_not_a_polythene_bag(self) -> None:
        material, _, evidence = reconcile_material("handbag", ["polythene bag"] * 4)
        self.assertEqual(material, "plastic")
        self.assertIn("identity_is_not_a_waste_bag", evidence["notes"])

    def test_a_split_vote_is_unknown_with_its_confidence(self) -> None:
        material, confidence, evidence = reconcile_material(
            "box", ["paper", "paper", "fabric or textile", "fabric or textile", "plastic"])
        self.assertEqual(material, "unknown")
        self.assertAlmostEqual(confidence, 0.4)
        self.assertIn("material_votes_split", evidence["notes"])

    def test_agreement_is_kept(self) -> None:
        self.assertEqual(reconcile_material("plastic garbage bag", ["polythene bag"] * 3)[0],
                         "polythene bag")

    def test_contents_seen_through_a_bag_are_not_its_material(self) -> None:
        _, _, evidence = reconcile_material("plastic garbage bag", ["food or organic waste"],
                                            colour_state="see_through_or_background_coloured")
        self.assertIn("translucent_contents_visible_not_classified", evidence["notes"])


class RealSenseGeometryPreservedTests(unittest.TestCase):
    """The colour step reads pixels only: RealSense dimensions do not move."""

    def test_realsense_dimensions_are_identical_with_and_without_the_colour_step(self) -> None:
        from test_v37_ids_and_heights import RealSenseSmallObjectTests

        case = RealSenseSmallObjectTests()
        with_colour = case._run(slice(66, 72), slice(66, 74))
        with mock.patch("locallife_cloud.pipeline.describe_colour",
                        side_effect=lambda *a, **k: mock.Mock(state="no_mask")):
            without = case._run(slice(66, 72), slice(66, 74))
        for name in ("footprint_length_mm", "footprint_width_mm", "physical_height_mm",
                     "volume_l", "track_id"):
            self.assertEqual(getattr(with_colour, name, None), getattr(without, name, None), name)



from locallife_cloud.logitech_pose import (  # noqa: E402
    FLOOR_SCALE_MAX_AGE_FRAMES, FloorScaleTracker, PoseFingerprint, expected_floor_depth,
    fingerprint_coefficients,
)


class FloorScaleTests(unittest.TestCase):
    """Each frame re-anchored on the floor it can still see."""

    def setUp(self) -> None:
        self.scene = Scene(DOWNWARD)
        plane = fit_reference_plane(self.scene.empty, CAMERA, mask=self.scene.region)
        self.pose = PoseFingerprint.from_plane(plane, (rig.WIDTH, rig.HEIGHT))
        self.depth, self.mask = self.scene.render(self.scene.box((0.0, 0.0), 0.265, 0.142, 0.089))
        self.floor = self.scene.region & ~self.mask

    def test_the_recorded_floor_predicts_near_and_far_depth(self) -> None:
        expected = expected_floor_depth(self.depth.shape, CAMERA, fingerprint_coefficients(self.pose))
        region = self.scene.region
        np.testing.assert_allclose(expected[region], self.scene.empty[region], rtol=1e-3)
        # A tilted view: the far floor really is much further than the near floor.
        self.assertGreater(np.percentile(expected[region], 90) / np.percentile(expected[region], 10), 1.2)

    def test_a_frame_whose_scale_drifted_is_measured_in_metres_again(self) -> None:
        drifted = self.depth * MODEL_SCALE_ERROR          # the bag made the model rescale the frame
        # Against the empty scene's scale the drifted frame is unmeasurable or wrong.
        refused, before = _measure(drifted, self.scene.empty, self.mask, self.scene.region)
        scale, record = FloorScaleTracker().update(drifted, CAMERA, self.floor, self.pose)
        self.assertEqual(record["state"], "validated_this_frame")
        self.assertAlmostEqual(scale, 1.0 / MODEL_SCALE_ERROR, delta=0.005)
        _, after = _measure(drifted * scale, self.scene.empty, self.mask, self.scene.region)
        for measured, truth in zip(after, (265.0, 142.0, 89.0)):
            self.assertLess(abs(measured - truth) / truth, 0.10, after)
        self.assertTrue(refused.reason is not None or abs(before[2] - 89.0) / 89.0 > 0.3)

    def test_a_prediction_wrong_in_shape_is_refused_not_averaged(self) -> None:
        warped = self.depth + 0.25                         # a shift, not a scale: near and far disagree
        scale, record = FloorScaleTracker().update(warped, CAMERA, self.floor, self.pose)
        self.assertIsNone(scale)
        self.assertEqual(record["reason"], "floor_scale_differs_near_to_far")
        self.assertEqual(record["state"], "unavailable")

    def test_a_covered_floor_reuses_the_last_scale_then_gives_up(self) -> None:
        tracker = FloorScaleTracker(max_age_frames=3)
        tracker.update(self.depth, CAMERA, self.floor, self.pose)
        covered = np.zeros_like(self.floor)                # a full bin: no floor visible
        scale, record = tracker.update(self.depth, CAMERA, covered, self.pose)
        self.assertEqual(record["state"], "reused_previous_scale")
        self.assertEqual(record["reason"], "too_little_visible_floor")
        self.assertAlmostEqual(scale, 1.0, delta=0.01)
        for _ in range(3):
            scale, record = tracker.update(self.depth, CAMERA, covered, self.pose)
        self.assertIsNone(scale)
        self.assertEqual(record["state"], "unavailable")
        self.assertGreater(FLOOR_SCALE_MAX_AGE_FRAMES, 3)

    def test_an_occupied_bin_is_not_used_as_floor(self) -> None:
        # The bag's own pixels would drag the scale: with the mask excluded
        # the scale is exact, with it included it is not.
        drifted = self.depth * MODEL_SCALE_ERROR
        clean, _ = FloorScaleTracker().update(drifted, CAMERA, self.floor, self.pose)
        self.assertAlmostEqual(clean, 1.0 / MODEL_SCALE_ERROR, delta=0.002)


class ElevatedAreaTests(unittest.TestCase):
    """The calibrated volume used each pixel's floor area for a bag's top."""

    def test_a_pixel_seeing_an_elevated_surface_covers_less_ground(self) -> None:
        from locallife_cloud.pipeline import VisionPipeline

        station = VisionPipeline.__new__(VisionPipeline)
        station.config = mock.Mock(logitech_reference_distance_m=0.62)
        station.logitech_derived_distance_m = None
        floor_area = np.full((2, 2), 0.04)                 # cm^2 per pixel on the floor
        heights = np.array([[0.0, 31.0], [15.5, 0.0]])     # cm
        area = station._area_at_height(floor_area, heights)
        self.assertAlmostEqual(area[0, 0], 0.04)
        self.assertAlmostEqual(area[0, 1], 0.04 * 0.25)    # half-way to the camera: a quarter
        self.assertAlmostEqual(area[1, 0], 0.04 * 0.75 ** 2)

    def test_integrating_a_box_top_with_its_own_area_gives_its_volume(self) -> None:
        # A 20 x 20 cm, 31 cm tall box under a camera 62 cm up, straight down:
        # its top pixels see floor patches twice as wide as the top itself.
        top_pixels = 100
        top_area_each = 400.0 / top_pixels
        floor_area_each = top_area_each * 4.0
        heights = np.full(top_pixels, 31.0)
        shrink = (1.0 - heights / 62.0) ** 2
        self.assertAlmostEqual(float(np.sum(heights * floor_area_each * shrink)) / 1000.0, 12.4)
        self.assertAlmostEqual(float(np.sum(heights * floor_area_each)) / 1000.0, 49.6)   # the old answer


class OverlappingObjectsTests(unittest.TestCase):
    def test_one_mask_measures_one_of_two_touching_objects(self) -> None:
        scene = Scene(DOWNWARD)
        box = scene.box((-0.08, 0.0), 0.15, 0.12, 0.10)
        bottle = scene.cylinder((0.06, 0.0), 0.035, 0.20)
        depth, _ = scene.render(box, bottle)
        _, box_only = scene.render(box)
        _, bottle_only = scene.render(bottle)
        mask = box_only & ~bottle_only & (depth >= np.minimum(box, np.inf) - 1e-9)
        _, (length, width, height) = _measure(depth, scene.empty, mask, scene.region)
        self.assertAlmostEqual(height, 100.0, delta=12.0)
        self.assertLess(length, 150.0 * 1.2)


class RelativeCheckpointTests(unittest.TestCase):
    def test_relative_depth_without_a_distance_measures_nothing(self) -> None:
        from test_v31_logitech_measurement_cascade import _run, _station

        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(
                directory, depth_model="depth-anything/Depth-Anything-V2-Small-hf")
            detection = _run(manager.camera("logitech"), frame, camera).detections[0]
        self.assertIsNone(detection.monocular_volume_l)
        self.assertEqual(manager.camera("logitech").calibration_mode,
                         "relative-depth-unscaled-unavailable")

    def test_a_metric_checkpoint_still_measures(self) -> None:
        from test_v31_logitech_measurement_cascade import _run, _station

        with tempfile.TemporaryDirectory() as directory:
            manager, _, frame, camera = _station(directory)
            detection = _run(manager.camera("logitech"), frame, camera).detections[0]
        self.assertGreater(detection.monocular_volume_l, 0)


class PipelineFloorScaleTests(unittest.TestCase):
    """The same object, with and without the model rescaling the whole frame."""

    def _measure(self, drift: float):
        from test_v31_logitech_measurement_cascade import MetricDepth, _run, _scene, _station

        class Drifting(MetricDepth):
            def __init__(self) -> None:
                super().__init__()
                self.factor = 1.0

            def estimate_batch(self, frames):
                return [depth * self.factor for depth in super().estimate_batch(frames)]

        depth = Drifting()
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, frame, camera = _station(
                directory, depth=depth, logitech_reference_distance_m=1.5)
            logitech = manager.camera("logitech")
            detector.items = []
            logitech.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False)
            logitech.set_baseline()
            logitech._expect_logitech_pose()
            logitech.process_frame(np.zeros_like(frame), intrinsics=camera, persist=False)
            self.assertEqual(logitech.logitech_pose.state, "recorded")
            detector.items = [_scene()[2]]
            depth.factor = drift
            detection = _run(logitech, frame, camera, start=40.0).detections[0]
            return detection, dict(logitech.logitech_floor_scale)

    def test_a_rescaled_frame_measures_the_same_object(self) -> None:
        steady, record = self._measure(1.0)
        self.assertEqual(record["state"], "validated_this_frame")
        drifted, record = self._measure(1.3)
        self.assertAlmostEqual(record["applied_scale"], 1 / 1.3, delta=0.01)
        self.assertAlmostEqual(drifted.physical_height_mm, steady.physical_height_mm, delta=5.0)
        self.assertAlmostEqual(drifted.monocular_volume_l, steady.monocular_volume_l,
                               delta=0.05 * steady.monocular_volume_l)


class IdentityConflictTests(unittest.TestCase):
    def test_a_waste_bag_called_handbag_is_reported(self) -> None:
        _, _, evidence = reconcile_material("handbag", ["polythene bag"] * 4)
        self.assertIn("identity_conflict_material_suggests_plastic_waste_bag", evidence["notes"])

if __name__ == "__main__":
    unittest.main()
