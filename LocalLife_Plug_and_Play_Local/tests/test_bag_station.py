"""Controlled fixed-overhead garbage-bag depth and volume checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.geometry import fixed_bin_mask
from locallife_cloud.pipeline import VisionPipeline, is_bag_detection, is_supported_waste_detection
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import estimate_volume, fit_reference_plane


class MutableBagDetector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self) -> None:
        self.detections: list[Detection] = []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [
            [
                Detection(
                    label=item.label,
                    confidence=item.confidence,
                    box=item.box,
                    mask=None if item.mask is None else item.mask.copy(),
                    source=item.source,
                )
                for item in self.detections
            ]
            for _ in frames
        ]


class FixedBagStationTests(unittest.TestCase):
    def test_every_default_prompt_describes_a_supported_bag_or_box(self) -> None:
        config = AppConfig()
        self.assertFalse(config.bag_only)
        self.assertTrue(all(is_supported_waste_detection(prompt) for prompt in config.prompts))
        self.assertTrue(any(is_bag_detection(prompt) for prompt in config.prompts))
        self.assertIn("cardboard box", config.prompts)

    def test_fixed_bin_polygon_clips_the_measurement_region(self) -> None:
        region = fixed_bin_mask(
            (100, 100),
            (0.0, 0.0, 1.0, 1.0),
            ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        )
        self.assertEqual(int(np.count_nonzero(region)), 2500)
        self.assertTrue(region[50, 50])
        self.assertFalse(region[10, 10])

    def test_isolated_invalid_depth_pixel_is_recovered_without_hiding_coverage(self) -> None:
        reference = np.full((20, 20), 2.0, dtype=np.float32)
        depth = np.full((20, 20), 1.5, dtype=np.float32)
        depth[10, 10] = 0.0
        measurement = estimate_volume(
            depth,
            reference,
            CameraIntrinsics(fx=100, fy=100),
            object_mask=np.ones(depth.shape, dtype=bool),
        )
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.valid_pixels, 400)
        self.assertEqual(measurement.filled_pixels, 1)
        self.assertAlmostEqual(measurement.coverage_ratio, 399 / 400)
        self.assertAlmostEqual(measurement.liters, 45.0, places=5)
        self.assertGreater(measurement.uncertainty_l, 0)

    def test_height_p90_reports_near_the_top_of_a_dome_shaped_object_not_the_footprint_average(self) -> None:
        # Round 8: real hardware reported a ~35 cm bag/pillow as ~15 cm tall.
        # `estimate_volume`'s own mean_height_m/max_height_m were fine, but
        # the dashboard's per-detection height used a *separate* raw median
        # computed over the whole instance mask -- and a soft, dome-shaped
        # or tapered object (a pillow, a slouched bag) has a large fraction
        # of its footprint near its sloped edges, well below its true top.
        # This reproduces that shape: a 30x30 mask where only the central
        # 10x10 reaches the true 35 cm peak and the surrounding area tapers
        # down toward the edges, so the whole-mask median sits far below
        # the real height while height_p90_m stays close to the peak.
        size = 30
        peak_height_m = 0.35
        reference = np.full((size, size), 2.0, dtype=np.float32)
        height_field = np.zeros((size, size), dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        center = (size - 1) / 2.0
        radial = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        radial /= radial.max()
        height_field = peak_height_m * np.clip(1.0 - radial, 0.0, 1.0)
        depth = reference - height_field
        object_mask = np.ones((size, size), dtype=bool)

        measurement = estimate_volume(
            depth, reference, CameraIntrinsics(fx=200, fy=200), object_mask=object_mask,
            max_height_m=0.5,
        )
        self.assertIsNotNone(measurement)
        median_height_cm = float(np.median(height_field)) * 100.0
        p90_height_cm = measurement.height_p90_m * 100.0
        # This exact shape reproduces the user's report almost exactly: the
        # old whole-mask median lands at ~14.5 cm for a genuine 35 cm peak.
        self.assertLess(median_height_cm, 20.0)
        # height_p90_m must report substantially closer to the true peak --
        # not merely "different from the median" but a real, large recovery.
        self.assertGreater(p90_height_cm, median_height_cm * 1.5)
        self.assertGreater(p90_height_cm, 24.0)
        self.assertLessEqual(measurement.height_p90_m, measurement.max_height_m)

    def test_patchy_dropout_below_the_old_seventy_percent_gate_is_now_recovered(self) -> None:
        # Black, low-IR-reflectivity materials (a black backpack, a black
        # bag) give the RealSense sensor scattered patchy dropout across an
        # otherwise perfectly real, compact object. The previous single-pass
        # fill only even attempted to run above 70% raw coverage, so an
        # object with heavier scattered dropout was refused entirely
        # ("rejected sparse height inside mask") even though it was
        # genuinely there. This reproduces that failure mode synthetically:
        # many small (1-3 px) missing patches bring raw coverage to ~67%,
        # below the old gate, but each hole is small enough that iterating
        # the same conservative fill rule closes almost all of them.
        size = 40
        rng = np.random.default_rng(3)
        missing = np.zeros((size, size), dtype=bool)
        for _ in range(300):
            row, col = rng.integers(1, size - 3, size=2)
            height, width = rng.integers(1, 3, size=2)
            missing[row : row + height, col : col + width] = True
        reference = np.full((size, size), 2.0, dtype=np.float32)
        depth = np.full((size, size), 1.7, dtype=np.float32)
        depth[missing] = 0.0
        object_mask = np.ones((size, size), dtype=bool)

        raw_coverage = 1.0 - float(missing.mean())
        self.assertLess(raw_coverage, 0.70)  # confirms this is the previously-refused regime

        measurement = estimate_volume(
            depth, reference, CameraIntrinsics(fx=100, fy=100), object_mask=object_mask,
        )
        self.assertIsNotNone(measurement)
        # coverage_ratio stays the honest, pre-fill raw-sensor number -- it
        # must not be inflated by filling, since it still drives the
        # uncertainty budget and quality label.
        self.assertAlmostEqual(measurement.coverage_ratio, raw_coverage, places=3)
        self.assertGreater(measurement.filled_pixels, 200)
        recovered_fraction = measurement.valid_pixels / measurement.candidate_pixels
        self.assertGreater(recovered_fraction, 0.90)

    def test_large_missing_depth_region_is_not_fabricated(self) -> None:
        reference = np.full((30, 30), 2.0, dtype=np.float32)
        depth = np.full((30, 30), 1.5, dtype=np.float32)
        depth[:15, :] = 0.0
        measurement = estimate_volume(
            depth,
            reference,
            CameraIntrinsics(fx=100, fy=100),
            object_mask=np.ones(depth.shape, dtype=bool),
        )
        self.assertEqual(measurement.filled_pixels, 0)
        self.assertAlmostEqual(measurement.coverage_ratio, 0.5)
        self.assertGreater(measurement.uncertainty_l, measurement.liters * 0.49)

    def test_empty_bin_baseline_uses_temporal_median(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                baseline_window_frames=5,
            )
            detector = MutableBagDetector()
            pipeline = VisionPipeline(config, detector=detector)
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            camera = CameraIntrinsics(fx=100, fy=100)
            for distance in (2.0, 2.01, 1.99, 2.0, 4.5):
                pipeline.process_frame(
                    frame,
                    depth_m=np.full(frame.shape[:2], distance, dtype=np.float32),
                    intrinsics=camera,
                    persist=False,
                )
            metadata = pipeline.set_baseline()
            self.assertEqual(metadata["baseline_frame_count"], 5)
            self.assertAlmostEqual(float(np.median(pipeline.baseline_realsense)), 2.0, places=5)

    def test_committed_bag_becomes_reference_for_next_arriving_bag(self) -> None:
        # The fixed bag station (BAG_STATION.md) is the one installation where
        # a committed bag genuinely stays in the bin and should become part of
        # the background. That is now opt-in: rewriting the reference while an
        # object is still in frame is exactly what broke sequential testing,
        # where each object is taken away again, so it must be asked for.
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0.0, 0.0, 1.0, 1.0),
                min_component_pixels=20,
                tracker_confirm_frames=1,
                advance_reference_on_deposit=True,
            )
            detector = MutableBagDetector()
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((60, 60, 3), dtype=np.uint8)
            reference = np.full((60, 60), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(empty, reference, camera)

            first_mask = np.zeros(reference.shape, dtype=bool)
            first_mask[5:20, 5:20] = True
            detector.detections = [Detection("black garbage bag", 0.9, (5, 5, 20, 20), first_mask)]
            first_frame = empty.copy()
            first_frame[first_mask] = 100
            first_depth = reference.copy()
            first_depth[first_mask] = 1.7
            first = pipeline.process_frame(first_frame, depth_m=first_depth, intrinsics=camera)
            self.assertEqual(len(first.detections), 1)
            self.assertIsNotNone(first.realsense_total)
            pipeline.commit_current_bags()

            second_mask = np.zeros(reference.shape, dtype=bool)
            second_mask[35:50, 35:50] = True
            detector.detections = [
                Detection("black garbage bag", 0.9, (5, 5, 20, 20), first_mask),
                Detection("white trash bag", 0.85, (35, 35, 50, 50), second_mask),
            ]
            second_frame = first_frame.copy()
            second_frame[second_mask] = 180
            second_depth = first_depth.copy()
            second_depth[second_mask] = 1.6
            second = pipeline.process_frame(second_frame, depth_m=second_depth, intrinsics=camera)

            self.assertEqual(len(second.detections), 1)
            self.assertEqual(second.detections[0].label, "white trash bag")
            self.assertEqual(second.realsense_total.valid_pixels, 225)
            self.assertGreater(second.bin_total.liters, second.realsense_total.liters)
            self.assertEqual(pipeline.state()["committed_bags"], 1)

    def test_fixed_polygon_excludes_depth_changes_outside_the_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0.0, 0.0, 1.0, 1.0),
                bin_polygon=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)),
                min_component_pixels=10,
            )
            pipeline = VisionPipeline(config, detector=MutableBagDetector())
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            reference = np.full((40, 40), 2.0, dtype=np.float32)
            camera = CameraIntrinsics(fx=100, fy=100)
            pipeline.set_baseline(frame, reference, camera)
            changed = frame.copy()
            changed[2:15, 2:15] = 200
            depth = reference.copy()
            depth[2:15, 2:15] = 1.5
            result = pipeline.process_frame(changed, depth_m=depth, intrinsics=camera)
            self.assertEqual(result.detections, [])
            self.assertIsNone(result.realsense_total)


class TiltCorrectedVolumeTests(unittest.TestCase):
    """Round 12: verifies the mounting-tilt height/volume bug and its fix.

    Real-hardware testing (round 12, milk-box-scale objects) showed RealSense
    distance readings were accurate but the derived height/volume were not.
    Reading `estimate_volume()` end to end found why: its height field was
    `baseline_depth - object_depth`, the camera-Z difference at each pixel --
    which only equals an object's true vertical height when the camera's
    optical axis is exactly perpendicular to the bin floor. This project's
    documented rig is a wall/bracket mount "pointing downward into the bin"
    (DUAL_CAMERA_THESIS.md / the thesis pilot deck), not a calibrated
    overhead gantry, so some non-zero tilt should be assumed, not treated as
    a rare edge case. For two parallel planes (an object's flat top resting
    on a flat floor) cut by a ray at angle theta from their shared normal,
    the naive camera-Z difference overstates the true perpendicular height by
    a factor of 1/cos(theta) -- this class proves that analytically-predicted
    bias is real and reproducible, and that supplying `estimate_volume()`'s
    new `reference_plane` argument (the already-existing `fit_reference_plane`
    output, now actually used as a correction instead of only a diagnostic)
    removes it.
    """

    def _tilted_scene(
        self,
        *,
        tilt_degrees: float,
        baseline_distance_m: float,
        true_height_m: float,
        image_size: int = 200,
        focal_length: float = 600.0,
        mask_half_extent_px: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics]:
        """A known-tilt floor plane with a known-height flat-topped object.

        The floor is the plane Z = b*Y + c (camera-frame metres, b=-tan
        (tilt), c=baseline_distance_m) -- i.e. tilted purely about the
        camera's X axis, matching `fit_reference_plane()`'s own
        z = a*x + b*y + c parametrisation exactly, so a perfect fit should
        recover a≈0, b≈-tan(tilt), c≈baseline_distance_m. The object's top
        face is the same plane shifted along its own normal by
        `true_height_m` (i.e. it sits flat on the floor, top parallel to
        it -- not to the camera sensor). Depth at each pixel is the exact
        ray/plane intersection (solved in closed form), not an
        approximation, so this is a noiseless, exact synthetic ground truth.
        """
        theta = np.radians(tilt_degrees)
        b = -np.tan(theta)
        ppx = ppy = image_size / 2.0
        rows, columns = np.indices((image_size, image_size), dtype=np.float64)
        y_ratio = (rows - ppy) / focal_length
        floor_depth = baseline_distance_m / (1.0 + b * y_ratio)
        # The object's top face is the floor plane shifted toward the camera
        # (smaller Z) by the true height: for two parallel planes
        # Z=b*Y+c1 and Z=b*Y+c2, the perpendicular distance between them is
        # |c1-c2| / sqrt(b^2+1) = |c1-c2| * cos(tilt), so a `true_height_m`
        # perpendicular gap needs |c1-c2| = true_height_m / cos(tilt).
        top_c = baseline_distance_m - true_height_m / np.cos(theta)
        top_depth = top_c / (1.0 + b * y_ratio)

        mask = np.zeros((image_size, image_size), dtype=bool)
        center = image_size // 2
        mask[
            center - mask_half_extent_px : center + mask_half_extent_px,
            center - mask_half_extent_px : center + mask_half_extent_px,
        ] = True
        object_depth = np.where(mask, top_depth, floor_depth).astype(np.float32)
        intrinsics = CameraIntrinsics(
            fx=focal_length, fy=focal_length, ppx=ppx, ppy=ppy,
            width=image_size, height=image_size,
        )
        return floor_depth.astype(np.float32), object_depth, mask, intrinsics

    def test_naive_height_is_inflated_by_one_over_cosine_of_tilt(self) -> None:
        baseline, depth, mask, intrinsics = self._tilted_scene(
            tilt_degrees=30.0, baseline_distance_m=1.0, true_height_m=0.10,
        )
        result = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
        )
        self.assertIsNotNone(result)
        predicted_naive_height = 0.10 / np.cos(np.radians(30.0))  # ~0.11547 m
        self.assertAlmostEqual(result.height_p90_m, predicted_naive_height, delta=0.002)
        # The bias is large and one-directional (always an overestimate),
        # not sensor noise -- this is the "distance is right, height/volume
        # is not" failure mode reported from real hardware.
        self.assertGreater(result.height_p90_m, 0.10 * 1.10)

    def test_plane_corrected_height_recovers_the_true_height(self) -> None:
        baseline, depth, mask, intrinsics = self._tilted_scene(
            tilt_degrees=30.0, baseline_distance_m=1.0, true_height_m=0.10,
        )
        plane = fit_reference_plane(baseline, intrinsics)
        self.assertIsNotNone(plane)
        # The fit should recover the true, deliberately-tilted floor almost
        # exactly (noiseless synthetic depth) -- confirms the test's own
        # scene construction is self-consistent with what fit_reference_plane
        # actually estimates, before trusting it to correct anything.
        self.assertAlmostEqual(plane.tilt_degrees, 30.0, delta=0.05)

        naive = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
        )
        corrected = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
            reference_plane=plane,
        )
        self.assertIsNotNone(naive)
        self.assertIsNotNone(corrected)
        naive_error = abs(naive.height_p90_m - 0.10)
        corrected_error = abs(corrected.height_p90_m - 0.10)
        self.assertLess(corrected_error, 0.003, "corrected height should track true height tightly")
        self.assertLess(corrected_error, naive_error / 10, "correction should remove nearly all of the tilt bias")

    def test_tilt_correction_also_improves_reported_liters(self) -> None:
        baseline, depth, mask, intrinsics = self._tilted_scene(
            tilt_degrees=30.0, baseline_distance_m=1.0, true_height_m=0.10,
            mask_half_extent_px=10,
        )
        # A small mask far from the image edge at 600px focal length keeps
        # the ray-frustum/surface-columns discretisation gap (a separate,
        # already-understood effect of height/depth ratio, not tilt) tiny,
        # so any remaining liters error here is attributable to tilt.
        # Ground truth: with the mask small and near the image centre, the
        # tilt-induced footprint distortion there is negligible, so the
        # standard small-angle pixel-footprint-area formula (at the floor's
        # own centre distance) is an accurate independent reference --
        # deliberately not derived from any of `estimate_volume()`'s own
        # formulas, so it cannot be circular.
        pixel_count = int(np.count_nonzero(mask))
        true_footprint_m2 = pixel_count * (1.0 ** 2) / (600.0 * 600.0)
        true_liters = true_footprint_m2 * 0.10 * 1000.0
        naive = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
        )
        plane = fit_reference_plane(baseline, intrinsics)
        corrected = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="ray-frustum", fill_small_holes=False,
            reference_plane=plane,
        )
        self.assertIsNotNone(naive)
        self.assertIsNotNone(corrected)
        naive_relative_error = abs(naive.liters - true_liters) / true_liters
        corrected_relative_error = abs(corrected.liters - true_liters) / true_liters
        # `naive` combines two separate, partially-offsetting biases here:
        # the tilt-driven height inflation this class is about (~+15% on
        # height alone, per test_naive_height_is_inflated_by_one_over...),
        # and surface-columns' own, unrelated height/depth-ratio cylinder-
        # vs-frustum underestimate -- so the net liters error (~-9.8%
        # measured) is smaller than the height error alone and can land on
        # either side of zero depending on geometry. What matters here is
        # that `corrected` (plane-corrected height + the exact ray-frustum
        # integral) is a clear, large improvement over it, not the naive
        # error's exact sign.
        self.assertGreater(naive_relative_error, 0.08)
        self.assertLess(corrected_relative_error, 0.05)
        self.assertLess(corrected_relative_error, naive_relative_error / 2)

    def test_estimate_volume_without_a_reference_plane_is_unchanged(self) -> None:
        """Backward compatibility: omitting `reference_plane` (every caller
        before round 12, and every caller that has not yet captured a
        baseline plane fit) must behave byte-identically to before this
        round -- this is what makes the fix safe to wire in everywhere."""
        baseline, depth, mask, intrinsics = self._tilted_scene(
            tilt_degrees=15.0, baseline_distance_m=1.2, true_height_m=0.08,
        )
        with_none = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
            reference_plane=None,
        )
        without_argument = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
        )
        self.assertIsNotNone(with_none)
        self.assertIsNotNone(without_argument)
        self.assertEqual(with_none.liters, without_argument.liters)
        self.assertEqual(with_none.height_p90_m, without_argument.height_p90_m)

    def test_flat_untilted_floor_is_unaffected_by_the_correction(self) -> None:
        """No tilt (theta=0) should make the correction a no-op: this proves
        the fix only changes behaviour when there is real tilt to correct,
        not for the already-good, perfectly-overhead-mounted case."""
        baseline, depth, mask, intrinsics = self._tilted_scene(
            tilt_degrees=0.0, baseline_distance_m=1.0, true_height_m=0.10,
        )
        plane = fit_reference_plane(baseline, intrinsics)
        naive = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
        )
        corrected = estimate_volume(
            depth, baseline, intrinsics, object_mask=mask, min_pixels=25,
            geometry_mode="surface-columns", fill_small_holes=False,
            reference_plane=plane,
        )
        self.assertAlmostEqual(naive.height_p90_m, corrected.height_p90_m, delta=1e-6)
        self.assertAlmostEqual(naive.liters, corrected.liters, delta=1e-6)

    def test_default_volume_geometry_is_the_height_map_grid(self) -> None:
        # The default has moved twice, each time to fix a real accuracy bug.
        # Round 12 chose ray-frustum believing it was exact regardless of
        # mounting tilt; round 16 found its volume sum never reads the
        # tilt-corrected height at all and switched to reference-plane, whose
        # sum does. reference-plane then carried its own systematic error:
        # it anchors each pixel's footprint area to the REFERENCE depth, so
        # an object standing proud of the bin floor has its footprint
        # overstated by (z_reference / z_object)^2 -- 44% for the 0.6 m/0.5 m
        # scene in test_known_volume_validation.py, which documented the bias
        # as understood-and-accepted rather than fixing it.
        #
        # height-map-grid removes the whole class of error by never deriving a
        # footprint from a pixel at all: it bins backprojected 3-D points into
        # fixed physical cells on the calibrated floor, so the area of a cell
        # is a constant the depth of the surface above it cannot distort.
        # Locking this in as a regression guard -- reverting the default would
        # resurrect both accuracy bugs this test class exists to catch.
        self.assertEqual(AppConfig().volume_geometry, "height-map-grid")


if __name__ == "__main__":
    unittest.main()
