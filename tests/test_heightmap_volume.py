"""Height-map grid volume against closed-form synthetic ground truth.

Every scene here is built analytically -- a surface at a known perpendicular
height above a known plane, rendered into the depth image a pinhole camera at a
known tilt would actually produce -- so the expected litres are derived, not
fitted. The rigid box is the playbook's calibration object (section 19), the
dome and wrinkled dome stand in for the crumpled polythene bags that are the
real target (section 3.1), and the noisy variants add the 4 mm depth noise and
patchy dropout a bag's gloss and wrinkles cause on real RealSense hardware
(section 8).
"""

from __future__ import annotations

import unittest

import numpy as np

from locallife_cloud.heightmap_volume import (
    HeightMapSettings,
    added_volume,
    build_reference_depth,
    depth_change_m,
    integrate_height_map,
    median_depth,
    scene_is_stable,
)
from locallife_cloud.types import CameraIntrinsics
from locallife_cloud.volume import ReferencePlane, estimate_volume

WIDTH, HEIGHT = 640, 480
FOCAL = 600.0
FLOOR_DEPTH_M = 1.20
INTRINSICS = CameraIntrinsics(
    fx=FOCAL, fy=FOCAL, ppx=WIDTH / 2, ppy=HEIGHT / 2, width=WIDTH, height=HEIGHT,
)
ROWS, COLUMNS = np.indices((HEIGHT, WIDTH)).astype(float)


def _frame(tilt_degrees: float):
    """Plane coefficients (z = a*x + b*y + c) and per-pixel ray slopes."""
    tilt = np.radians(tilt_degrees)
    coefficients = (0.0, float(np.tan(tilt)), FLOOR_DEPTH_M / float(np.cos(tilt)))
    a, b, c = coefficients
    scale = float(np.sqrt(a * a + b * b + 1.0))
    normal = np.array([a, b, -1.0]) / scale
    seed = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    slope_x = (COLUMNS - INTRINSICS.ppx) / FOCAL
    slope_y = (ROWS - INTRINSICS.ppy) / FOCAL
    return coefficients, scale, first, second, slope_x, slope_y


def _depth_at_height(coefficients, scale, slope_x, slope_y, height):
    a, b, c = coefficients
    return (c - height * scale) / (1.0 - a * slope_x - b * slope_y)


def _plane_coordinates(depth, first, second, slope_x, slope_y):
    points = np.stack([slope_x * depth, slope_y * depth, depth], axis=-1)
    u, v = points @ first, points @ second
    return u - u[HEIGHT // 2, WIDTH // 2], v - v[HEIGHT // 2, WIDTH // 2]


def box_scene(tilt_degrees, length=0.25, width=0.20, height=0.15):
    """A rigid box standing on the floor plane; truth is exactly L*W*H."""
    coefficients, scale, first, second, slope_x, slope_y = _frame(tilt_degrees)
    top = _depth_at_height(coefficients, scale, slope_x, slope_y, height)
    floor = _depth_at_height(coefficients, scale, slope_x, slope_y, 0.0)
    u, v = _plane_coordinates(top, first, second, slope_x, slope_y)
    on_box = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
    return np.where(on_box, top, floor), coefficients, floor, length * width * height * 1000.0


def _dome_height(u, v, radius, peak, wrinkled):
    base = np.clip(peak * (1.0 - (u * u + v * v) / (radius * radius)), 0.0, None)
    if not wrinkled:
        return base
    ripple = 0.012 * np.sin(u / 0.02) * np.cos(v / 0.022)
    return np.where(base > 0, np.clip(base + ripple, 0.0, None), 0.0)


def dome_scene(tilt_degrees, *, wrinkled=False, radius=0.16, peak=0.13):
    """A smooth or wrinkled dome -- a slouched bag rather than a cuboid."""
    coefficients, scale, first, second, slope_x, slope_y = _frame(tilt_degrees)
    floor = _depth_at_height(coefficients, scale, slope_x, slope_y, 0.0)
    depth = floor.copy()
    for _ in range(60):
        u, v = _plane_coordinates(depth, first, second, slope_x, slope_y)
        depth = _depth_at_height(
            coefficients, scale, slope_x, slope_y, _dome_height(u, v, radius, peak, wrinkled)
        )
    step = 0.001
    axis = np.arange(-radius, radius, step) + step / 2
    grid_u, grid_v = np.meshgrid(axis, axis)
    truth = float(np.sum(_dome_height(grid_u, grid_v, radius, peak, wrinkled)) * step * step * 1000)
    return depth, coefficients, floor, truth


def degrade(depth, *, noise_m=0.004, dropout=0.12, seed=11):
    """Add Gaussian depth noise and patchy dropout to a clean scene."""
    generator = np.random.default_rng(seed)
    degraded = depth + generator.normal(0.0, noise_m, depth.shape)
    degraded[generator.random(depth.shape) < dropout] = np.nan
    return degraded


SETTINGS = HeightMapSettings(grid_size_m=0.010, min_valid_depth_fraction=0.0)


def measure(depth, coefficients, settings=SETTINGS):
    return integrate_height_map(
        depth, INTRINSICS, plane_coefficients=coefficients, settings=settings
    )


class HeightMapAccuracyTests(unittest.TestCase):
    """Playbook gates G2/G4: plausible litres for rigid and crumpled objects."""

    def test_rigid_box_is_recovered_within_five_percent_at_every_tilt(self) -> None:
        for tilt in (0.0, 15.0, 30.0):
            with self.subTest(tilt=tilt):
                depth, coefficients, _, truth = box_scene(tilt)
                measured = measure(depth, coefficients)
                self.assertIsNotNone(measured)
                self.assertLess(abs(measured.liters - truth) / truth, 0.05)
                self.assertEqual(measured.quality, "valid")

    def test_smooth_and_wrinkled_domes_are_recovered_within_two_percent(self) -> None:
        for tilt, wrinkled in ((0.0, False), (25.0, False), (0.0, True), (25.0, True)):
            with self.subTest(tilt=tilt, wrinkled=wrinkled):
                depth, coefficients, _, truth = dome_scene(tilt, wrinkled=wrinkled)
                measured = measure(depth, coefficients)
                self.assertLess(abs(measured.liters - truth) / truth, 0.02)

    def test_noise_and_dropout_do_not_move_a_crumpled_bag_reading(self) -> None:
        # The whole point of a robust per-cell height: 4 mm of depth noise and
        # 12% dropout are what a glossy, wrinkled polythene bag actually does to
        # a RealSense, and they must not move the litres by more than a couple
        # of percent (playbook sections 8 and 19).
        depth, coefficients, _, truth = dome_scene(0.0, wrinkled=True)
        measured = measure(degrade(depth), coefficients)
        self.assertLess(abs(measured.liters - truth) / truth, 0.03)

    def test_repeated_noisy_captures_of_one_scene_stay_stable(self) -> None:
        # Playbook gate G3: repeatability on an undisturbed setup.
        depth, coefficients, _, truth = dome_scene(0.0, wrinkled=True)
        readings = [measure(degrade(depth, seed=seed), coefficients).liters for seed in range(6)]
        self.assertLess(float(np.std(readings)) / truth, 0.02)

    def test_it_beats_the_per_pixel_sum_it_replaces(self) -> None:
        # The reason for the change: the per-pixel modes anchor each pixel's
        # footprint to a depth, which biases every object standing proud of the
        # floor. This asserts the improvement, not merely that both run.
        depth, coefficients, floor, truth = dome_scene(0.0, wrinkled=True)
        noisy = degrade(depth)
        grid_error = abs(measure(noisy, coefficients).liters - truth) / truth
        per_pixel = estimate_volume(
            noisy, floor, INTRINSICS,
            min_height_m=0.005, max_height_m=0.8, geometry_mode="reference-plane",
            reference_plane=ReferencePlane(0.0, 0.0, 0, (0, 0, 1), coefficients),
        )
        per_pixel_error = abs(per_pixel.liters - truth) / truth
        self.assertLess(grid_error, per_pixel_error / 4)

    def test_an_empty_bin_measures_nothing(self) -> None:
        # The noise floor must absorb sensor scatter across the whole bin
        # rather than accumulating it into phantom litres.
        _, coefficients, floor, _ = dome_scene(0.0)
        self.assertIsNone(measure(degrade(floor, dropout=0.0), coefficients))


class HeightMapGeometryTests(unittest.TestCase):
    def test_method_b_works_without_a_fitted_plane_and_says_so(self) -> None:
        depth, _, floor, truth = box_scene(0.0)
        measured = integrate_height_map(
            depth, INTRINSICS, reference_depth_m=floor, settings=SETTINGS
        )
        self.assertIn("method-b-depth-difference", measured.flags)
        self.assertLess(abs(measured.liters - truth) / truth, 0.10)

    def test_grid_coarsens_rather_than_returning_nothing_on_coarse_depth(self) -> None:
        # A 10 mm cell asked of a camera whose pixels already span more than
        # that would leave every cell under-sampled; the measurement must
        # degrade to the resolution the sensor supports and flag that it did.
        coarse = CameraIntrinsics(fx=50.0, fy=50.0, ppx=WIDTH / 2, ppy=HEIGHT / 2)
        depth, coefficients, _, _ = box_scene(0.0)
        measured = integrate_height_map(
            depth, coarse, plane_coefficients=coefficients, settings=SETTINGS
        )
        self.assertIsNotNone(measured)
        self.assertIn("grid-coarsened-to-depth-resolution", measured.flags)
        self.assertGreater(measured.grid_size_m, SETTINGS.grid_size_m)

    def test_impossible_heights_are_rejected_not_integrated(self) -> None:
        depth, coefficients, _, truth = dome_scene(0.0)
        spiked = depth.copy()
        spiked[200:210, 300:310] = 0.15  # a reflective spike metres above the bin
        measured = measure(spiked, coefficients)
        self.assertLess(abs(measured.liters - truth) / truth, 0.05)

    def test_a_mask_confines_the_measurement_to_the_object(self) -> None:
        depth, coefficients, _, truth = dome_scene(0.0)
        blocked = np.zeros(depth.shape, dtype=bool)
        blocked[:, : WIDTH // 2] = True
        measured = measure(depth, coefficients)
        half = integrate_height_map(
            depth, INTRINSICS, plane_coefficients=coefficients, mask=blocked, settings=SETTINGS,
        )
        self.assertLess(half.liters, measured.liters)
        self.assertAlmostEqual(half.liters, truth / 2, delta=truth * 0.08)

    def test_large_missing_regions_are_not_invented(self) -> None:
        depth, coefficients, _, _ = dome_scene(0.0)
        complete = measure(depth, coefficients)
        punched = depth.copy()
        punched[210:270, 290:350] = np.nan
        measured = measure(punched, coefficients)
        self.assertLess(measured.liters, complete.liters)


class HeightMapQualityTests(unittest.TestCase):
    """Playbook section 16: never report a confident litre value on bad data."""

    def test_poor_depth_coverage_is_flagged_rather_than_hidden(self) -> None:
        depth, coefficients, _, _ = dome_scene(0.0)
        measured = integrate_height_map(
            degrade(depth, dropout=0.55),
            INTRINSICS,
            plane_coefficients=coefficients,
            settings=HeightMapSettings(grid_size_m=0.010, min_valid_depth_fraction=0.70),
        )
        self.assertEqual(measured.quality, "low_quality")
        self.assertIn("valid depth", measured.rejection_reason)
        self.assertFalse(measured.is_valid)

    def test_nothing_is_returned_without_a_reference_or_a_plane(self) -> None:
        depth, _, _, _ = box_scene(0.0)
        self.assertIsNone(integrate_height_map(depth, INTRINSICS, settings=SETTINGS))

    def test_settings_reject_an_unusable_grid(self) -> None:
        with self.assertRaises(ValueError):
            HeightMapSettings(grid_size_m=0.0)
        with self.assertRaises(ValueError):
            HeightMapSettings(min_height_m=0.5, max_height_m=0.2)


class ReferenceAndStabilityTests(unittest.TestCase):
    def test_reference_median_ignores_per_pixel_dropout(self) -> None:
        _, _, floor, _ = dome_scene(0.0)
        frames = [degrade(floor, noise_m=0.003, dropout=0.3, seed=seed) for seed in range(45)]
        reference, report = build_reference_depth(frames)
        self.assertTrue(report["sufficient_frames"])
        self.assertGreater(report["valid_fraction"], 0.99)
        self.assertLess(float(np.nanmax(np.abs(reference - floor))), 0.01)

    def test_a_thin_reference_stack_is_reported_not_silently_accepted(self) -> None:
        _, _, floor, _ = dome_scene(0.0)
        _, report = build_reference_depth([floor] * 5)
        self.assertFalse(report["sufficient_frames"])

    def test_a_pixel_invalid_in_every_frame_stays_unknown(self) -> None:
        _, _, floor, _ = dome_scene(0.0)
        frames = []
        for _ in range(4):
            frame = floor.copy()
            frame[5:9, 5:9] = np.nan
            frames.append(frame)
        combined = median_depth(frames)
        self.assertTrue(np.all(np.isnan(combined[5:9, 5:9])))
        self.assertTrue(np.all(np.isfinite(combined[100:120, 100:120])))

    def test_stability_trigger_waits_for_a_settled_scene(self) -> None:
        moving, _, floor, _ = dome_scene(0.0)
        settled = degrade(floor, noise_m=0.0005, dropout=0.0)
        # The bin ROI, not the whole image: a bag covers a large share of the
        # bin but only a few percent of the frame, and a change statistic
        # measured over the frame would call a still-falling bag "settled".
        bin_roi = np.zeros(floor.shape, dtype=bool)
        bin_roi[150:330, 230:410] = True
        self.assertLess(depth_change_m(floor, settled, region=bin_roi), 0.002)
        self.assertGreater(depth_change_m(floor, moving, region=bin_roi), 0.002)

    def test_a_small_moving_object_is_not_called_settled(self) -> None:
        # Regression guard for the statistic itself: with a plain median this
        # comparison read exactly 0.0 m of change while the object was plainly
        # there, which would have released the measurement mid-fall.
        moving, _, floor, _ = dome_scene(0.0, radius=0.09)
        bin_roi = np.zeros(floor.shape, dtype=bool)
        bin_roi[120:360, 200:440] = True
        self.assertGreater(depth_change_m(floor, moving, region=bin_roi), 0.002)

    def test_stability_requires_several_consecutive_quiet_frames(self) -> None:
        self.assertFalse(scene_is_stable([0.001] * 3, threshold_m=0.002, required_frames=4))
        self.assertTrue(scene_is_stable([0.001] * 5, threshold_m=0.002, required_frames=4))
        self.assertFalse(
            scene_is_stable([0.001, 0.001, 0.05, 0.001], threshold_m=0.002, required_frames=4)
        )
        self.assertFalse(scene_is_stable([None, 0.001], threshold_m=0.002, required_frames=2))


class DepositDifferenceTests(unittest.TestCase):
    """Playbook sections 5 and 10: the per-bag contribution to a filling bin."""

    def _reading(self, tilt, peak):
        depth, coefficients, _, truth = dome_scene(tilt, peak=peak)
        return measure(depth, coefficients), truth

    def test_added_volume_is_the_change_in_occupied_volume(self) -> None:
        before, truth_before = self._reading(0.0, 0.08)
        after, truth_after = self._reading(0.0, 0.13)
        deposit = added_volume(before, after)
        self.assertEqual(deposit.quality, "valid")
        self.assertAlmostEqual(
            deposit.added_liters, truth_after - truth_before, delta=0.3,
        )

    def test_an_undisturbed_bin_adds_nothing(self) -> None:
        before, _ = self._reading(0.0, 0.10)
        deposit = added_volume(before, before)
        self.assertEqual(deposit.added_liters, 0.0)
        self.assertEqual(deposit.quality, "valid")

    def test_a_bin_that_lost_volume_is_rejected_with_a_reason(self) -> None:
        before, _ = self._reading(0.0, 0.13)
        after, _ = self._reading(0.0, 0.05)
        deposit = added_volume(before, after)
        self.assertEqual(deposit.quality, "rejected")
        self.assertIn("fell by", deposit.rejection_reason)

    def test_an_impossible_deposit_is_rejected(self) -> None:
        before, _ = self._reading(0.0, 0.08)
        after, _ = self._reading(0.0, 0.13)
        deposit = added_volume(before, after, max_added_l=0.5)
        self.assertEqual(deposit.quality, "rejected")
        self.assertIn("plausible deposit bound", deposit.rejection_reason)

    def test_low_quality_depth_propagates_into_the_event(self) -> None:
        before, _ = self._reading(0.0, 0.08)
        after, _ = self._reading(0.0, 0.13)
        after.quality = "low_quality"
        after.rejection_reason = "only 40% of the measured region has valid depth"
        deposit = added_volume(before, after)
        self.assertEqual(deposit.quality, "low_quality")
        self.assertIn("40%", deposit.rejection_reason)


if __name__ == "__main__":
    unittest.main()
