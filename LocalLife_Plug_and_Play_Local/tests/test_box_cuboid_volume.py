"""Tests for `estimate_box_volume_cuboid()` (Revised Dual-Camera Volume
Estimation engineering recipe, sections 2/4.3/9) and box-template matching.

Real-hardware testing (round 16) reported a 1.5 L milk box measuring ~62.6 L
on Logitech and being rejected as implausible on RealSense, even after the
round-13 mounting-tilt fix. Reading `estimate_volume()`'s `ray-frustum`
branch found why: its volume sum uses raw camera-Z depth directly and never
reads the tilt-corrected height at all (see volume.py's corrected comment
and config.py's `volume_geometry` comment). `estimate_box_volume_cuboid()`
is the PDF's prescribed replacement for rigid boxes: three robust scalar
dimensions (height above the fitted table plane, and a footprint length/
width from a table-relative point cloud), multiplied together, rather than a
per-pixel sum -- deliberately independent of `estimate_volume()`'s own
formulas so this class validates it against exact synthetic ground truth on
its own terms.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import Detection

from locallife_cloud.box_templates import BoxTemplate, match_box_template
from locallife_cloud.types import BoxVolumeMeasurement, CameraIntrinsics
from locallife_cloud.volume import (
    aggregate_box_measurements,
    estimate_box_volume_cuboid,
    estimate_object_dimensions,
    fit_reference_plane,
    reference_plane_is_usable,
)


def _measurement(length_mm: float, width_mm: float, height_mm: float, **overrides) -> BoxVolumeMeasurement:
    """A single-frame `BoxVolumeMeasurement` with plausible defaults for
    everything `aggregate_box_measurements()` does not itself derive from
    length/width/height -- used to build a synthetic multi-frame track
    history without needing a full backprojected depth scene per frame."""
    fields = dict(
        volume_liters=(length_mm / 1000.0) * (width_mm / 1000.0) * (height_mm / 1000.0) * 1000.0,
        volume_confidence=0.8,
        volume_method="table_relative_cuboid",
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        depth_valid_ratio=0.9,
        object_points=500,
        table_plane_inliers=4000,
        table_plane_rmse_mm=2.0,
        height_p98_mm=height_mm * 1.01,
        height_top_median_mm=height_mm,
        mask_clipped=False,
        flags=("single_view_estimate",),
    )
    fields.update(overrides)
    return BoxVolumeMeasurement(**fields)


def _tilted_box_scene(
    *,
    tilt_degrees: float,
    baseline_distance_m: float,
    true_height_m: float,
    row_half: int,
    col_half: int,
    image_size: int = 300,
    focal_length: float = 800.0,
    center: int | None = None,
):
    """A known-tilt table plane (tilted purely about the camera X axis, so
    real-world X is unaffected by tilt -- see `fit_reference_plane`'s own
    z = a*x + b*y + c parametrisation) with a known-height, known-footprint
    flat-topped box resting on it. `row_half`/`col_half` set the box's pixel
    footprint (independently, so length != width); depth at every pixel is
    the exact closed-form ray/plane intersection for whichever surface
    (floor or box top) that pixel belongs to -- a noiseless, exact synthetic
    scene, matching this project's established testing convention
    (test_bag_station.py's `TiltCorrectedVolumeTests._tilted_scene`).
    """
    theta = np.radians(tilt_degrees)
    b = -np.tan(theta)
    ppx = ppy = image_size / 2.0
    center = image_size // 2 if center is None else center
    rows, columns = np.indices((image_size, image_size), dtype=np.float64)
    y_ratio = (rows - ppy) / focal_length
    floor_depth = baseline_distance_m / (1.0 + b * y_ratio)
    top_c = baseline_distance_m - true_height_m / np.cos(theta)
    top_depth = top_c / (1.0 + b * y_ratio)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[center - row_half: center + row_half, center - col_half: center + col_half] = True
    object_depth = np.where(mask, top_depth, floor_depth).astype(np.float32)
    intrinsics = CameraIntrinsics(
        fx=focal_length, fy=focal_length, ppx=ppx, ppy=ppy, width=image_size, height=image_size,
    )
    return floor_depth.astype(np.float32), object_depth, mask, intrinsics


class BoxCuboidVolumeTests(unittest.TestCase):
    TILT_DEGREES = 25.0
    BASELINE_M = 1.0
    HEIGHT_M = 0.10
    ROW_HALF = 12
    COL_HALF = 20

    def _scene_and_plane(self, **overrides):
        params = dict(
            tilt_degrees=self.TILT_DEGREES, baseline_distance_m=self.BASELINE_M,
            true_height_m=self.HEIGHT_M, row_half=self.ROW_HALF, col_half=self.COL_HALF,
        )
        params.update(overrides)
        floor, depth, mask, intrinsics = _tilted_box_scene(**params)
        # The plane is fit from the floor-only (background/baseline) image,
        # matching how pipeline.py fits `self.reference_plane` at baseline
        # capture time -- before any object is present.
        plane = fit_reference_plane(floor, intrinsics)
        self.assertIsNotNone(plane)
        return depth, mask, intrinsics, plane

    def test_height_recovers_true_height_regardless_of_tilt(self) -> None:
        # Verified numerically: recovered height matches true height to
        # within ~0.02 mm even at a 25 degree tilt, for both the eroded
        # (default) and un-eroded mask -- erosion only affects footprint,
        # never the height computation, since every surviving object pixel
        # sits on the exact same flat top plane.
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.height_mm, self.HEIGHT_M * 1000.0, delta=0.1)

    def test_general_dimensions_recover_height_and_ignore_floor_halo(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane()
        clean = estimate_object_dimensions(depth, intrinsics, mask, plane)
        self.assertIsNotNone(clean)
        self.assertAlmostEqual(clean.height_mm, self.HEIGHT_M * 1000.0, delta=0.1)
        self.assertGreater(clean.length_mm, clean.width_mm)

        # Loose neural masks often include visible floor. Plane-height pixels
        # must not enlarge the physical footprint.
        halo = mask.copy()
        halo[120:180, 110:190] = True
        with_halo = estimate_object_dimensions(depth, intrinsics, halo, plane)
        self.assertIsNotNone(with_halo)
        self.assertAlmostEqual(with_halo.length_mm, clean.length_mm, delta=0.5)
        self.assertAlmostEqual(with_halo.width_mm, clean.width_mm, delta=0.5)
        self.assertAlmostEqual(with_halo.height_mm, clean.height_mm, delta=0.5)

    def test_footprint_recovers_true_dimensions_with_no_erosion(self) -> None:
        # Numerically verified exact ground truth (corner-to-corner 3-D
        # distance on the box's own top plane) for this exact scene:
        # length ~= 43.070 mm, width ~= 28.208 mm.
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane, mask_erosion_px=0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.length_mm, 43.070, delta=0.3)
        self.assertAlmostEqual(result.width_mm, 28.208, delta=0.3)

    def test_default_mask_erosion_shrinks_the_footprint_as_expected(self) -> None:
        # Numerically verified exact ground truth after accounting for the
        # default 2px erosion on each side (PDF section 4.1): the footprint
        # shrinks to the pixel range [row_half-2, col_half-2] on each edge.
        # length ~= 38.697 mm, width ~= 23.302 mm.
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.length_mm, 38.697, delta=0.3)
        self.assertAlmostEqual(result.width_mm, 23.302, delta=0.3)
        # And the un-eroded measurement must be strictly larger in both axes.
        no_erosion = estimate_box_volume_cuboid(depth, intrinsics, mask, plane, mask_erosion_px=0)
        self.assertGreater(no_erosion.length_mm, result.length_mm)
        self.assertGreater(no_erosion.width_mm, result.width_mm)

    def test_volume_liters_equals_length_times_width_times_height(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNotNone(result)
        expected_liters = (
            (result.length_mm / 1000.0) * (result.width_mm / 1000.0) * (result.height_mm / 1000.0) * 1000.0
        )
        self.assertAlmostEqual(result.volume_liters, expected_liters, places=6)

    def test_zero_tilt_still_recovers_correct_dimensions(self) -> None:
        # A degenerate but important case: an overhead, untilted camera. The
        # table-relative method must not depend on any actual tilt existing.
        depth, mask, intrinsics, plane = self._scene_and_plane(tilt_degrees=0.0)
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane, mask_erosion_px=0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.height_mm, self.HEIGHT_M * 1000.0, delta=0.2)

    def test_severe_tilt_does_not_bias_the_recovered_height(self) -> None:
        # This is the core claim the PDF and this project's own round-13/16
        # findings both make: table-relative height stays stable across
        # camera tilt, unlike the raw camera-Z height difference (which
        # would be inflated by 1/cos(tilt) -- e.g. ~26% at 40 degrees).
        for tilt in (0.0, 15.0, 25.0, 40.0):
            with self.subTest(tilt=tilt):
                depth, mask, intrinsics, plane = self._scene_and_plane(tilt_degrees=tilt)
                result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane, mask_erosion_px=0)
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result.height_mm, self.HEIGHT_M * 1000.0, delta=0.3)

    def test_too_few_points_returns_none(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane(row_half=2, col_half=2)
        # A 4x4 pixel box eroded by the default 2px leaves nothing.
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNone(result)

    def test_missing_reference_plane_returns_none(self) -> None:
        depth, mask, intrinsics, _ = self._scene_and_plane()
        self.assertIsNone(estimate_box_volume_cuboid(depth, intrinsics, mask, None))

    def test_mask_touching_the_image_border_is_flagged_clipped(self) -> None:
        # center == row_half puts the mask's top edge exactly at row 0;
        # col_half stays small enough that the column span (center-col_half
        # to center+col_half) remains in-bounds, so only the row edge is
        # actually clipped.
        depth, mask, intrinsics, plane = self._scene_and_plane(
            row_half=12, col_half=8, center=12,
        )
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNotNone(result)
        self.assertIn("mask_clipped", result.flags)
        self.assertLess(result.volume_confidence, 0.80)

    def test_single_view_estimate_flag_is_always_present(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        self.assertIsNotNone(result)
        self.assertIn("single_view_estimate", result.flags)

    def test_high_plane_rmse_lowers_confidence_and_is_flagged(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane()
        from dataclasses import replace
        noisy_plane = replace(plane, residual_rmse_m=0.02)  # 20 mm, well above the 8 mm gate
        clean_result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        noisy_result = estimate_box_volume_cuboid(depth, intrinsics, mask, noisy_plane)
        self.assertIn("high_plane_rmse", noisy_result.flags)
        self.assertNotIn("high_plane_rmse", clean_result.flags)
        self.assertLess(noisy_result.volume_confidence, clean_result.volume_confidence)
        self.assertTrue(reference_plane_is_usable(plane))
        self.assertFalse(reference_plane_is_usable(noisy_plane))

    def test_plane_fit_selects_dominant_floor_instead_of_averaging_floor_and_wall(self) -> None:
        height, width = 100, 120
        intrinsics = CameraIntrinsics(fx=140, fy=140, ppx=60, ppy=50)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        depth[:35, :] = 3.2  # a second, farther wall plane

        plane = fit_reference_plane(depth, intrinsics)

        self.assertIsNotNone(plane)
        self.assertLess(plane.residual_rmse_m, 0.008)
        self.assertAlmostEqual(plane.coefficients[2], 2.0, delta=0.02)

    def test_diagnostics_to_dict_matches_the_recipe_pdf_shape(self) -> None:
        depth, mask, intrinsics, plane = self._scene_and_plane()
        result = estimate_box_volume_cuboid(depth, intrinsics, mask, plane)
        payload = result.to_dict()
        self.assertIn("volume_liters", payload)
        self.assertIn("dimensions_mm", payload)
        self.assertEqual(set(payload["dimensions_mm"]), {"length", "width", "height"})
        self.assertIn("diagnostics", payload)
        for key in (
            "depth_valid_ratio", "object_points", "table_plane_inliers",
            "table_plane_rmse_mm", "height_p98_mm", "height_top_median_mm", "mask_clipped",
        ):
            self.assertIn(key, payload["diagnostics"])
        self.assertIn("flags", payload)


class AggregateBoxMeasurementsTests(unittest.TestCase):
    """`aggregate_box_measurements()` (Revised Dual-Camera Volume Estimation
    recipe, section 13): median L/W/H across accepted frames, one final
    volume, never a per-frame sum or a mean/median of the already-multiplied
    per-frame liters figures."""

    def test_empty_history_returns_none(self) -> None:
        self.assertIsNone(aggregate_box_measurements([]))

    def test_median_dimensions_and_volume_from_once_multiplication(self) -> None:
        # Five near-identical frames with one outlier height (a single noisy
        # depth frame) -- the median must ignore the outlier the way a mean
        # would not.
        measurements = [
            _measurement(100.0, 70.0, 200.0),
            _measurement(101.0, 69.0, 201.0),
            _measurement(99.0, 71.0, 199.0),
            _measurement(100.5, 70.5, 199.5),
            _measurement(100.0, 70.0, 260.0),  # one outlier frame
        ]
        result = aggregate_box_measurements(measurements)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.length_mm, 100.0, delta=0.6)
        self.assertAlmostEqual(result.width_mm, 70.0, delta=0.6)
        self.assertAlmostEqual(result.height_mm, 200.0, delta=0.6)
        # The one-and-only multiplication happens on the aggregated medians,
        # not a mean/median across each frame's own pre-multiplied liters.
        expected_liters = (
            (result.length_mm / 1000.0) * (result.width_mm / 1000.0) * (result.height_mm / 1000.0) * 1000.0
        )
        self.assertAlmostEqual(result.volume_liters, expected_liters, places=9)
        self.assertEqual(result.frames_accepted, 5)
        self.assertEqual(result.frames_considered, 5)

    def test_frames_considered_can_exceed_frames_accepted(self) -> None:
        # A caller passes only the frames that survived its own quality
        # gates into `measurements`, but separately tracks every frame it
        # attempted (including ones estimate_box_volume_cuboid() itself
        # rejected outright) as `frames_considered`.
        measurements = [_measurement(100.0, 70.0, 200.0) for _ in range(3)]
        result = aggregate_box_measurements(measurements, frames_considered=9)
        self.assertIsNotNone(result)
        self.assertEqual(result.frames_accepted, 3)
        self.assertEqual(result.frames_considered, 9)

    def test_identical_frames_report_zero_spread(self) -> None:
        measurements = [_measurement(100.0, 70.0, 200.0) for _ in range(4)]
        result = aggregate_box_measurements(measurements)
        self.assertIsNotNone(result)
        self.assertEqual(result.dimension_std_mm, (0.0, 0.0, 0.0))
        self.assertNotIn("dimension_instability", result.flags)

    def test_high_dimension_spread_is_flagged_and_lowers_confidence(self) -> None:
        stable = [_measurement(100.0, 70.0, 200.0, volume_confidence=0.85) for _ in range(4)]
        unstable = [
            _measurement(100.0, 70.0, 150.0, volume_confidence=0.85),
            _measurement(100.0, 70.0, 300.0, volume_confidence=0.85),
            _measurement(100.0, 70.0, 155.0, volume_confidence=0.85),
            _measurement(100.0, 70.0, 295.0, volume_confidence=0.85),
        ]
        stable_result = aggregate_box_measurements(stable)
        unstable_result = aggregate_box_measurements(unstable)
        self.assertNotIn("dimension_instability", stable_result.flags)
        self.assertIn("dimension_instability", unstable_result.flags)
        self.assertLess(unstable_result.volume_confidence, stable_result.volume_confidence)

    def test_mesh_is_never_used_for_the_aggregated_final_volume(self) -> None:
        measurements = [_measurement(100.0, 70.0, 200.0) for _ in range(3)]
        result = aggregate_box_measurements(measurements)
        self.assertFalse(result.mesh_used_for_final_volume)

    def test_diagnostics_to_dict_includes_multi_frame_fields(self) -> None:
        measurements = [_measurement(100.0, 70.0, 200.0) for _ in range(3)]
        result = aggregate_box_measurements(measurements, frames_considered=5)
        payload = result.to_dict()
        diagnostics = payload["diagnostics"]
        self.assertEqual(diagnostics["frames_considered"], 5)
        self.assertEqual(diagnostics["frames_accepted"], 3)
        self.assertEqual(set(diagnostics["dimension_std_mm"]), {"length", "width", "height"})
        self.assertIn("mesh_used_for_final_volume", diagnostics)
        self.assertFalse(diagnostics["mesh_used_for_final_volume"])


class BoxTemplateMatchTests(unittest.TestCase):
    def test_unmeasured_template_never_matches(self) -> None:
        # Mirrors box_templates.yaml's shipped state: a placeholder template
        # must never silently "match" real measured dimensions.
        placeholder = BoxTemplate(
            id="milk_box_1l", nominal_volume_liters=1.0,
            length_mm=0.0, width_mm=0.0, height_mm=0.0, tolerance_mm=8.0, measured=False,
        )
        match = match_box_template(95.0, 65.0, 200.0, [placeholder])
        self.assertIsNone(match)

    def test_measured_template_matches_within_tolerance(self) -> None:
        real = BoxTemplate(
            id="milk_box_1_5l", nominal_volume_liters=1.5,
            length_mm=95.6, width_mm=95.6, height_mm=200.4, tolerance_mm=8.0, measured=True,
        )
        match = match_box_template(93.0, 98.0, 197.0, [real])
        self.assertIsNotNone(match)
        self.assertEqual(match.template.id, "milk_box_1_5l")

    def test_length_width_swap_is_tried_both_ways(self) -> None:
        real = BoxTemplate(
            id="milk_box_1l", nominal_volume_liters=1.0,
            length_mm=95.6, width_mm=64.4, height_mm=200.4, tolerance_mm=8.0, measured=True,
        )
        # Measured with length/width swapped relative to the template.
        match = match_box_template(64.0, 96.0, 201.0, [real])
        self.assertIsNotNone(match)
        self.assertEqual(match.template.id, "milk_box_1l")

    def test_outside_tolerance_does_not_match(self) -> None:
        real = BoxTemplate(
            id="milk_box_2l", nominal_volume_liters=2.0,
            length_mm=108.0, width_mm=108.0, height_mm=200.4, tolerance_mm=10.0, measured=True,
        )
        match = match_box_template(80.0, 80.0, 150.0, [real])
        self.assertIsNone(match)

    def test_closest_of_several_candidates_wins(self) -> None:
        near = BoxTemplate(
            id="near", nominal_volume_liters=1.0,
            length_mm=95.0, width_mm=65.0, height_mm=200.0, tolerance_mm=10.0, measured=True,
        )
        far = BoxTemplate(
            id="far", nominal_volume_liters=1.0,
            length_mm=90.0, width_mm=70.0, height_mm=195.0, tolerance_mm=10.0, measured=True,
        )
        match = match_box_template(95.0, 65.0, 200.0, [near, far])
        self.assertEqual(match.template.id, "near")

    def test_shipped_yaml_loads_and_starts_unmeasured(self) -> None:
        from locallife_cloud.box_templates import load_box_templates

        templates = load_box_templates()
        self.assertEqual(len(templates), 3)
        ids = {template.id for template in templates}
        self.assertEqual(ids, {"milk_box_1l", "milk_box_1_5l", "milk_box_2l"})
        for template in templates:
            self.assertFalse(template.measured)
            # Every match attempt against the shipped, unmeasured file must
            # fail -- it must never accidentally "validate" a real box.
        self.assertIsNone(match_box_template(95.0, 65.0, 200.0, templates))


class _StubBoxDetector:
    """Same shape as `test_plant_monitor.AdjustableDetector` -- returns a
    fresh copy of whatever `.detections` currently holds for every frame in
    the batch -- kept local to this module so this end-to-end aggregation
    test does not depend on another test module's private helper."""

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


class PipelineBoxAggregationTests(unittest.TestCase):
    """End-to-end: `VisionPipeline` actually reaches
    `aggregate_box_measurements()` for a real tracked box over several
    frames, and the multi-frame median genuinely improves the reported
    result over any single noisy frame -- not just the isolated
    `estimate_box_volume_cuboid()`/`aggregate_box_measurements()` unit tests
    above, but the real per-frame wiring in `pipeline.py`."""

    def _run(self, config: AppConfig, depths_m: list[float]) -> list[Detection]:
        detector = _StubBoxDetector()
        pipeline = VisionPipeline(config, detector=detector)
        empty = np.zeros((70, 70, 3), dtype=np.uint8)
        baseline = np.full((70, 70), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100)
        pipeline.set_baseline(empty, baseline, camera)

        mask = np.zeros((70, 70), dtype=bool)
        mask[10:40, 10:50] = True  # 30 rows x 40 columns -> length != width
        detector.detections = [Detection("cardboard box", 0.9, (10, 10, 50, 40), mask)]
        frame = empty.copy()
        frame[mask] = (40, 90, 170)

        box_detections: list[Detection] = []
        for index, distance in enumerate(depths_m):
            depth = baseline.copy()
            depth[mask] = distance
            analysis = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=100 + index)
            for detection in analysis.detections:
                if detection.box_length_mm is not None:
                    box_detections.append(detection)
        return box_detections

    def test_box_dimensions_stabilize_across_frames_with_zero_spread_on_a_static_scene(self) -> None:
        config = AppConfig(
            results_dir=Path(tempfile.mkdtemp()), enable_monocular_depth=False, roi=(0, 0, 1, 1),
            min_component_pixels=20, tracker_confirm_frames=1, auto_deposit=False,
            box_aggregation_min_frames=3, box_aggregation_window_frames=10,
        )
        # A perfectly static box: the same table-relative geometry every
        # frame -- there should be no real-world reason for its measured
        # dimensions to disagree frame to frame.
        results = self._run(config, depths_m=[1.7] * 7)
        self.assertGreaterEqual(len(results), 6)
        last = results[-1]
        self.assertGreaterEqual(last.box_frames_accepted, config.box_aggregation_min_frames)
        self.assertEqual(last.box_dimension_std_mm, (0.0, 0.0, 0.0))
        self.assertNotIn("dimension_instability", last.box_volume_flags)
        self.assertGreater(last.box_length_mm, 0)
        self.assertGreater(last.box_width_mm, 0)
        self.assertGreater(last.box_height_mm, 0)

    def test_one_noisy_frame_does_not_dominate_the_aggregated_result(self) -> None:
        # PDF section 13: "Never sum per-frame volumes" / never let one
        # noisy frame set the final answer. Five clean frames at 1.7 m
        # object depth (0.3 m table-relative height) plus one depth-spike
        # frame at 1.2 m (0.8 m height, implausible for this ~0.3 m box) --
        # the median-aggregated final height must stay near the five clean
        # frames' true value, not be dragged toward the one outlier.
        config = AppConfig(
            results_dir=Path(tempfile.mkdtemp()), enable_monocular_depth=False, roi=(0, 0, 1, 1),
            min_component_pixels=20, tracker_confirm_frames=1, auto_deposit=False,
            box_aggregation_min_frames=3, box_aggregation_window_frames=10,
        )
        depths = [1.7, 1.7, 1.2, 1.7, 1.7, 1.7]
        results = self._run(config, depths_m=depths)
        last = results[-1]
        # True clean-frame height is 2.0 - 1.7 = 0.3 m = 300 mm; the outlier
        # frame alone would report 2.0 - 1.2 = 0.8 m = 800 mm.
        self.assertAlmostEqual(last.box_height_mm, 300.0, delta=5.0)
        self.assertLess(last.box_height_mm, 500.0)


class PeerLabelledBoxGeometryTests(unittest.TestCase):
    """A peer label is presence evidence, not cross-view object identity.

    The cameras are not pixel-registered. An unlabelled RealSense depth blob
    can therefore be measured only after RealSense itself has an accepted
    semantic detection; otherwise furniture and bedding can become geometry.
    """

    def _config(self) -> AppConfig:
        return AppConfig(
            results_dir=Path(tempfile.mkdtemp()), enable_monocular_depth=False, roi=(0, 0, 1, 1),
            min_component_pixels=20, tracker_confirm_frames=1, auto_deposit=False,
            box_aggregation_min_frames=3, box_aggregation_window_frames=10,
        )

    def _run(
        self, *, own_label: str | None, peer_box_present: bool, peer_bag_present: bool = True
    ) -> list[Detection]:
        """Drive one frame through the real pipeline.

        `own_label=None` reproduces the reported hardware case exactly:
        RealSense's own neural detector returns nothing, so the object only
        reaches the measurement loop as a scene-fusion phantom
        ("unclassified object", zero neural confidence) -- which is what the
        dashboard was showing when its "Box geometry" column stayed empty.
        """
        detector = _StubBoxDetector()
        pipeline = VisionPipeline(self._config(), detector=detector)
        empty = np.zeros((70, 70, 3), dtype=np.uint8)
        baseline = np.full((70, 70), 2.0, dtype=np.float32)
        camera = CameraIntrinsics(fx=100, fy=100)
        pipeline.set_baseline(empty, baseline, camera)

        mask = np.zeros((70, 70), dtype=bool)
        mask[10:40, 10:50] = True
        frame = empty.copy()
        frame[mask] = (40, 90, 170)
        depth = baseline.copy()
        depth[mask] = 1.70

        own = (
            []
            if own_label is None
            else [Detection(own_label, 0.9, (10, 10, 50, 40), mask.copy())]
        )
        analysis = pipeline.process_precomputed(
            frame,
            detections=own,
            depth_m=depth,
            intrinsics=camera,
            timestamp=100.0,
            persist=False,
            peer_bag_present=peer_bag_present,
            peer_box_present=peer_box_present,
        )
        return list(analysis.detections)

    def test_peer_label_cannot_promote_an_unlabelled_realsense_blob(self) -> None:
        # The cameras are not pixel-registered. A simultaneous Logitech box
        # label cannot prove that a RealSense depth component is that object.
        detections = self._run(own_label=None, peer_box_present=True)
        self.assertEqual(detections, [])

    def test_without_a_peer_box_label_an_unlabelled_object_gets_no_cuboid(self) -> None:
        # Guards the other direction: a changed-depth region must not be
        # promoted to a rigid-box measurement on depth evidence alone.
        detections = self._run(own_label=None, peer_box_present=False)
        measured = [item for item in detections if item.box_length_mm is not None]
        self.assertEqual(measured, [])

    def test_realsense_own_box_label_still_measures_without_any_peer_signal(self) -> None:
        # The pre-existing path must be completely unaffected, and must NOT
        # be mislabelled as peer-sourced.
        detections = self._run(
            own_label="cardboard box", peer_box_present=False, peer_bag_present=False
        )
        measured = [item for item in detections if item.box_length_mm is not None]
        self.assertTrue(measured)
        self.assertNotIn("peer_labelled_box", measured[0].box_volume_flags)

    def test_own_box_label_is_not_flagged_peer_sourced_even_when_the_peer_agrees(self) -> None:
        # Both cameras agreeing is the common healthy case -- the flag is
        # only for a measurement whose object type came from the peer.
        detections = self._run(own_label="cardboard box", peer_box_present=True)
        measured = [item for item in detections if item.box_length_mm is not None]
        self.assertTrue(measured)
        self.assertNotIn("peer_labelled_box", measured[0].box_volume_flags)


class NoCapturedBaselineVolumeTests(unittest.TestCase):
    """Regression guards for the round-24 root cause.

    Every liters cell in the reported real-hardware screenshots read
    "pending - pending empty baseline", for both cameras, on every object,
    across many rounds. The reason was structural rather than mathematical:
    both the support plane AND the reference surface each volume method
    measures against could only come from a separately captured EMPTY-SCENE
    baseline. With none captured, `reference_plane` was None (so
    `estimate_box_volume_cuboid()` returned None on its first guard) and
    `reference_realsense` was None (so `estimate_volume()` had nothing to
    subtract) -- so no amount of correct geometry downstream could ever
    produce a number. Capturing that baseline needs a genuinely empty scene,
    which the real room never was.

    These tests drive `VisionPipeline` with `set_baseline()` NEVER called.
    """

    def _pipeline_without_any_baseline(self) -> tuple[VisionPipeline, CameraIntrinsics]:
        config = AppConfig(
            results_dir=Path(tempfile.mkdtemp()), enable_monocular_depth=False, roi=(0, 0, 1, 1),
            min_component_pixels=20, tracker_confirm_frames=1, auto_deposit=False,
            restore_saved_baseline=False, automatic_baseline=False,
        )
        detector = _StubBoxDetector()
        pipeline = VisionPipeline(config, detector=detector)
        # Deliberately NO set_baseline() call anywhere in this test.
        self.assertIsNone(pipeline.reference_realsense)
        self.assertIsNone(pipeline.reference_plane)
        return pipeline, CameraIntrinsics(fx=100, fy=100, ppx=35.0, ppy=35.0)

    def _scene(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A flat floor at 2.0 m with one raised object standing on it."""
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        mask = np.zeros((70, 70), dtype=bool)
        mask[20:45, 20:50] = True
        frame[mask] = (40, 90, 170)
        depth = np.full((70, 70), 2.0, dtype=np.float32)
        depth[mask] = 1.75  # 25 cm tall object
        return frame, depth, mask

    def test_box_volume_is_produced_with_no_baseline_ever_captured(self) -> None:
        pipeline, camera = self._pipeline_without_any_baseline()
        frame, depth, mask = self._scene()

        analysis = pipeline.process_precomputed(
            frame,
            detections=[Detection("cardboard box", 0.9, (20, 20, 50, 45), mask.copy())],
            depth_m=depth,
            intrinsics=camera,
            timestamp=100.0,
            persist=False,
        )

        measured = [item for item in analysis.detections if item.box_length_mm is not None]
        self.assertTrue(measured, "a box must be measurable without any captured baseline")
        box = measured[0]
        self.assertIn("live_fitted_support_plane", box.box_volume_flags)
        # The object stands 25 cm above a flat floor; the plane fitted from
        # the surrounding floor must recover that, not something arbitrary.
        self.assertGreater(box.box_height_mm, 200.0)
        self.assertLess(box.box_height_mm, 300.0)
        self.assertEqual(pipeline.support_plane_source, "live-frame-background")

    def test_bag_liters_are_produced_with_no_baseline_ever_captured(self) -> None:
        # The per-pixel path (bags, and any non-box object) must also work,
        # via the synthesized empty-floor reference.
        pipeline, camera = self._pipeline_without_any_baseline()
        frame, depth, mask = self._scene()

        analysis = pipeline.process_precomputed(
            frame,
            detections=[Detection("garbage bag", 0.9, (20, 20, 50, 45), mask.copy())],
            depth_m=depth,
            intrinsics=camera,
            timestamp=100.0,
            persist=False,
        )

        measured = [
            item for item in analysis.detections if item.realsense_volume_l is not None
        ]
        self.assertTrue(measured, "a bag must get liters without any captured baseline")
        self.assertGreater(measured[0].realsense_volume_l, 0.0)

    def test_pending_reason_is_no_longer_empty_baseline_once_a_plane_is_fitted(self) -> None:
        pipeline, camera = self._pipeline_without_any_baseline()
        frame, depth, mask = self._scene()

        analysis = pipeline.process_precomputed(
            frame,
            detections=[Detection("garbage bag", 0.9, (20, 20, 50, 45), mask.copy())],
            depth_m=depth,
            intrinsics=camera,
            timestamp=100.0,
            persist=False,
        )

        for detection in analysis.detections:
            self.assertNotEqual(detection.measurement_quality, "pending-empty-baseline")

    def test_a_real_captured_baseline_is_never_overwritten(self) -> None:
        # The live-fitted plane is a fallback, not a replacement: a genuine
        # empty-scene capture must still win.
        config = AppConfig(
            results_dir=Path(tempfile.mkdtemp()), enable_monocular_depth=False, roi=(0, 0, 1, 1),
            min_component_pixels=20, tracker_confirm_frames=1, auto_deposit=False,
            restore_saved_baseline=False, automatic_baseline=False,
        )
        pipeline = VisionPipeline(config, detector=_StubBoxDetector())
        camera = CameraIntrinsics(fx=100, fy=100, ppx=35.0, ppy=35.0)
        empty = np.zeros((70, 70, 3), dtype=np.uint8)
        baseline_depth = np.full((70, 70), 2.0, dtype=np.float32)
        pipeline.set_baseline(empty, baseline_depth, camera)
        captured_plane = pipeline.reference_plane
        self.assertIsNotNone(captured_plane)

        frame, depth, mask = self._scene()
        pipeline.process_precomputed(
            frame,
            detections=[Detection("cardboard box", 0.9, (20, 20, 50, 45), mask.copy())],
            depth_m=depth,
            intrinsics=camera,
            timestamp=100.0,
            persist=False,
        )
        self.assertIs(pipeline.reference_plane, captured_plane)
        self.assertEqual(pipeline.support_plane_source, "captured-baseline")


if __name__ == "__main__":
    unittest.main()
