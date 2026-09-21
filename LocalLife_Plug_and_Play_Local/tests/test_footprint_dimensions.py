"""Footprint extents for objects seen from one viewpoint.

Field failure: an upright cylinder 5 cm across and 20 cm tall measured 5 x 2 cm.
The height was right. One camera sees one side, so the cylinder's visible
surface projects to a half-disc -- full diameter across the chord, a couple of
centimetres deep -- and reporting that depth as the width is the bug.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from locallife_cloud.footprint import (
    DimensionSmoother,
    estimate_extents,
    minimum_area_extents,
    small_object_warning,
)


def _visible_arc(radius: float, seen_fraction: float = 0.5, count: int = 400) -> np.ndarray:
    """The half of a circular footprint a single camera can actually see."""
    span = 2.0 * math.pi * seen_fraction
    angles = np.linspace(-span / 2.0, span / 2.0, count)
    return np.column_stack((radius * np.sin(angles), radius * np.cos(angles) - radius))


def _filled_rect(length: float, width: float, count: int = 40) -> np.ndarray:
    xs = np.linspace(-length / 2, length / 2, count)
    ys = np.linspace(-width / 2, width / 2, count)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


class CylinderTests(unittest.TestCase):
    def test_a_half_seen_cylinder_recovers_its_diameter(self) -> None:
        # 5 cm diameter, front half visible: the old estimator reported ~2 cm.
        extents = estimate_extents(_visible_arc(0.025))
        self.assertIsNotNone(extents)
        self.assertTrue(extents.occlusion_corrected)
        self.assertAlmostEqual(extents.length_mm, 50.0, delta=4.0)
        self.assertAlmostEqual(extents.width_mm, 50.0, delta=4.0)
        # The specific wrong answer must not come back.
        self.assertGreater(extents.width_mm, 30.0)

    def test_the_uncorrected_extent_is_the_bug_being_fixed(self) -> None:
        # Same points, correction off: this is what the field saw.
        raw = estimate_extents(_visible_arc(0.025), correct_self_occlusion=False)
        self.assertLess(raw.width_mm, 30.0)

    def test_a_less_than_half_seen_cylinder_still_recovers(self) -> None:
        extents = estimate_extents(_visible_arc(0.025, seen_fraction=0.35))
        self.assertTrue(extents.occlusion_corrected)
        self.assertAlmostEqual(extents.width_mm, 50.0, delta=8.0)

    def test_a_larger_cylinder_scales(self) -> None:
        extents = estimate_extents(_visible_arc(0.06))
        self.assertAlmostEqual(extents.width_mm, 120.0, delta=10.0)


class BoxTests(unittest.TestCase):
    """Large rectangular boxes already measured well and must not move."""

    def test_a_filled_rectangle_is_left_alone(self) -> None:
        extents = estimate_extents(_filled_rect(0.40, 0.30))
        self.assertFalse(extents.occlusion_corrected)
        self.assertAlmostEqual(extents.length_mm, 400.0, delta=5.0)
        self.assertAlmostEqual(extents.width_mm, 300.0, delta=5.0)
        self.assertIn("filled_footprint", extents.flags)

    def test_a_square_footprint_is_left_alone(self) -> None:
        extents = estimate_extents(_filled_rect(0.20, 0.20))
        self.assertFalse(extents.occlusion_corrected)
        self.assertAlmostEqual(extents.length_mm, 200.0, delta=5.0)

    def test_a_diagonally_placed_box_keeps_its_true_size(self) -> None:
        # The case a minimum-area rectangle handles and principal axes do not.
        points = _filled_rect(0.40, 0.20)
        angle = math.radians(37.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        extents = estimate_extents(points @ rotation.T)
        self.assertAlmostEqual(extents.length_mm, 400.0, delta=8.0)
        self.assertAlmostEqual(extents.width_mm, 200.0, delta=8.0)

    def test_a_thin_flat_face_is_not_turned_into_a_circle(self) -> None:
        # A cream box seen edge-on: nearly collinear points must stay flat.
        extents = estimate_extents(_filled_rect(0.10, 0.002))
        self.assertFalse(extents.occlusion_corrected)
        self.assertLess(extents.width_mm, 10.0)

    def test_minimum_area_beats_the_axis_aligned_box_when_rotated(self) -> None:
        points = _filled_rect(0.40, 0.10)
        angle = math.radians(45.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        rotated = points @ rotation.T
        long_side, short_side, _ = minimum_area_extents(rotated)
        axis_aligned = rotated.max(axis=0) - rotated.min(axis=0)
        self.assertAlmostEqual(long_side, 0.40, delta=0.01)
        self.assertAlmostEqual(short_side, 0.10, delta=0.01)
        # The axis-aligned box would have called this ~35 x 35 cm.
        self.assertGreater(axis_aligned.min(), 0.30)


class GuardTests(unittest.TestCase):
    def test_too_few_points_returns_nothing(self) -> None:
        self.assertIsNone(estimate_extents(np.zeros((2, 2))))
        self.assertIsNone(estimate_extents(np.zeros((0, 2))))

    def test_non_finite_points_are_dropped(self) -> None:
        points = _filled_rect(0.2, 0.2)
        points[0] = (np.nan, np.inf)
        self.assertIsNotNone(estimate_extents(points))

    def test_small_objects_are_flagged_not_rejected(self) -> None:
        self.assertEqual(
            small_object_warning(50.0, 50.0, 8.0), "dimension_near_sensor_noise",
        )
        self.assertEqual(
            small_object_warning(50.0, 50.0, 18.0), "small_object_low_precision",
        )
        self.assertIsNone(small_object_warning(400.0, 300.0, 260.0))


class SmoothingTests(unittest.TestCase):
    """One bad frame must not decide a thin object's reported thickness."""

    def test_a_single_bad_frame_is_outvoted(self) -> None:
        smoother = DimensionSmoother(window=9)
        for _ in range(5):
            smoother.update(1, 100.0, 60.0, 20.0)
        # The 4 cm reading the operator saw on a 2 cm box.
        length, width, height = smoother.update(1, 100.0, 60.0, 40.0)
        self.assertAlmostEqual(height, 20.0, places=1)
        self.assertAlmostEqual(width, 60.0, places=1)

    def test_a_real_change_is_followed(self) -> None:
        smoother = DimensionSmoother(window=5)
        for _ in range(5):
            smoother.update(1, 100.0, 60.0, 20.0)
        for _ in range(5):
            result = smoother.update(1, 200.0, 120.0, 80.0)
        self.assertAlmostEqual(result[2], 80.0, places=1)

    def test_tracks_are_kept_apart(self) -> None:
        smoother = DimensionSmoother()
        smoother.update(1, 100.0, 100.0, 100.0)
        self.assertEqual(smoother.update(2, 50.0, 50.0, 50.0), (50.0, 50.0, 50.0))

    def test_an_untracked_detection_passes_through(self) -> None:
        self.assertEqual(
            DimensionSmoother().update(None, 1.0, 2.0, 3.0), (1.0, 2.0, 3.0),
        )

    def test_stability_reports_only_with_enough_history(self) -> None:
        smoother = DimensionSmoother()
        self.assertIsNone(smoother.stability(1))
        for _ in range(4):
            smoother.update(1, 100.0, 60.0, 20.0)
        self.assertEqual(smoother.stability(1), 1.0)
        smoother.update(1, 100.0, 60.0, 40.0)
        self.assertLess(smoother.stability(1), 1.0)

    def test_forgetting_a_track_clears_it(self) -> None:
        smoother = DimensionSmoother()
        smoother.update(1, 100.0, 60.0, 20.0)
        smoother.forget(1)
        self.assertIsNone(smoother.stability(1))


if __name__ == "__main__":
    unittest.main()
