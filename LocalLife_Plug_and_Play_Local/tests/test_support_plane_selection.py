"""Tests for choosing the surface an object RESTS ON as its support plane.

The v8 hardware run reported 392x106x789 mm for a bag roughly 20 cm tall,
847x394x599 mm for an Amazon carton measured at 410x315x140 mm, and
635x159x611 mm for a third object. The heights barely moved as the object
changed, which is the signature of a measurement that is not about the object
at all.

`fit_reference_plane` keeps the RANSAC plane with the most inliers, and
`set_baseline` runs it across the whole measurement region. In the V3 trials
the camera pointed down at a floor, so the largest coherent surface *was* the
surface objects rested on, and reported dimensions came within a few percent
of ruler truth (389x302x123 mm against a 410x315x140 mm box). From v6 onward
the camera looks sideways across a bed with a painted wall behind it. The wall
is large, flat and low-noise; a duvet is soft, wrinkled and returns sparse
speckled depth. RANSAC picks the wall, and every "height above the support
plane" becomes the object's distance in front of the wall.

Tilt does not catch it: `tilt_degrees` measures the angle between the plane
normal and the camera axis, so a wall viewed head-on scores a flattering 0
degrees while the bed it should have picked looks steeply tilted. Both
existing criteria prefer the wall, which is why this survived several rounds
of fixes aimed at masks and baselines.
"""

from __future__ import annotations

import unittest

import numpy as np

from locallife_cloud.types import CameraIntrinsics
from locallife_cloud.volume import (
    estimate_object_dimensions,
    fit_local_support_plane,
    fit_reference_plane,
    fit_support_plane_from_background,
    support_plane_explains_object,
    synthesize_plane_depth,
)

# A support surface receding away from a camera that is tilted down towards
# it, written in the estimator's own z = a*x + b*y + c form.
SUPPORT_COEFFICIENTS = (0.0, -1.20, 1.60)
WALL_DEPTH_M = 2.30
SHAPE = (240, 320)
INTRINSICS = CameraIntrinsics(fx=300.0, fy=300.0, ppx=160.0, ppy=120.0)


def _plane_norm(coefficients: tuple[float, float, float]) -> float:
    a, b, _c = coefficients
    return float(np.sqrt(a * a + b * b + 1.0))


def _scene(object_height_m: float = 0.14):
    """A wall filling most of the view, with an object on a lower surface.

    Returns the empty-scene depth, the depth with the object present, and the
    object's mask. The wall covers appreciably more of the frame than the
    support surface and is perfectly flat, exactly as a painted bedroom wall
    is to a stereo camera.
    """
    support_depth = synthesize_plane_depth(SHAPE, INTRINSICS, SUPPORT_COEFFICIENTS)
    assert support_depth is not None

    empty = np.full(SHAPE, np.nan, dtype=np.float32)
    wall_rows = 150  # the wall occupies the upper ~62% of the frame
    empty[:wall_rows, :] = WALL_DEPTH_M
    empty[wall_rows:, :] = support_depth[wall_rows:, :]

    # The object rests on the support surface: shift each of its pixels
    # towards the camera by exactly `object_height_m` measured perpendicular
    # to that surface.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[165:205, 110:210] = True
    a, b, c = SUPPORT_COEFFICIENTS
    columns, rows = np.meshgrid(
        np.arange(SHAPE[1], dtype=np.float64), np.arange(SHAPE[0], dtype=np.float64),
    )
    denominator = (
        1.0
        - a * (columns - INTRINSICS.ppx) / INTRINSICS.fx
        - b * (rows - INTRINSICS.ppy) / INTRINSICS.fy
    )
    offset = object_height_m * _plane_norm(SUPPORT_COEFFICIENTS) / denominator

    current = empty.copy()
    current[mask] = (empty[mask] - offset[mask]).astype(np.float32)
    return empty, current, mask


class RegionWidePlaneReproducesTheV8FailureTests(unittest.TestCase):
    def test_the_whole_region_fit_locks_onto_the_wall(self) -> None:
        empty, current, mask = _scene()

        plane = fit_reference_plane(
            empty, INTRINSICS, mask=np.ones(SHAPE, dtype=bool),
        )

        self.assertIsNotNone(plane)
        # The wall is fronto-parallel, so a plane fitted to it has a normal
        # pointing straight back at the camera and a flattering 0 deg tilt.
        self.assertLess(plane.tilt_degrees, 5.0)
        # And the object does not stand on it -- it merely faces it.
        self.assertFalse(
            support_plane_explains_object(current, INTRINSICS, mask, plane),
            "the wall must not be accepted as the surface the object rests on",
        )

    def test_measuring_against_the_wall_inflates_height_as_observed(self) -> None:
        empty, current, mask = _scene(object_height_m=0.14)

        wall_plane = fit_reference_plane(empty, INTRINSICS, mask=np.ones(SHAPE, dtype=bool))
        against_wall = estimate_object_dimensions(
            current, INTRINSICS, mask, wall_plane, min_height_m=0.010, min_points=40,
            max_height_m=3.0,
        )

        self.assertIsNotNone(against_wall)
        # This is the v8 symptom: a 14 cm object reported as most of a metre,
        # because the number is really its distance in front of the wall.
        self.assertGreater(against_wall.height_mm, 400.0)


class LocalSupportPlaneTests(unittest.TestCase):
    def test_the_local_fit_finds_the_surface_under_the_object(self) -> None:
        empty, current, mask = _scene(object_height_m=0.14)

        plane = fit_local_support_plane(
            current, INTRINSICS, mask, baseline_depth_m=empty,
        )

        self.assertIsNotNone(plane)
        self.assertTrue(support_plane_explains_object(current, INTRINSICS, mask, plane))
        measured = estimate_object_dimensions(
            current, INTRINSICS, mask, plane, min_height_m=0.010, min_points=40,
        )
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured.height_mm, 140.0, delta=12.0)

    def test_a_taller_object_is_measured_taller(self) -> None:
        # Guards against a fit that simply returns a plausible-looking
        # constant: the reported height has to track the real one.
        heights = []
        for truth in (0.08, 0.14, 0.25):
            empty, current, mask = _scene(object_height_m=truth)
            plane = fit_local_support_plane(
                current, INTRINSICS, mask, baseline_depth_m=empty,
            )
            measured = estimate_object_dimensions(
                current, INTRINSICS, mask, plane, min_height_m=0.010, min_points=40,
            )
            self.assertIsNotNone(measured)
            heights.append(measured.height_mm)
            self.assertAlmostEqual(measured.height_mm, truth * 1000.0, delta=15.0)
        self.assertEqual(heights, sorted(heights))

    def test_the_local_fit_works_without_a_baseline(self) -> None:
        # A live installation that has not captured a baseline must still get
        # the surrounding surface rather than the wall.
        _empty, current, mask = _scene(object_height_m=0.14)

        plane = fit_local_support_plane(current, INTRINSICS, mask)

        self.assertIsNotNone(plane)
        self.assertTrue(support_plane_explains_object(current, INTRINSICS, mask, plane))

    def test_too_little_background_returns_none_rather_than_a_guess(self) -> None:
        _empty, current, _mask = _scene()
        everything = np.ones(SHAPE, dtype=bool)

        self.assertIsNone(
            fit_local_support_plane(current, INTRINSICS, everything),
        )


class DownwardCameraStaysCorrectTests(unittest.TestCase):
    """The V3 geometry -- camera aimed down at the surface -- must not regress.

    There the largest plane and the support plane are the same surface, and
    the reported dimensions were already good. The local fit has to agree with
    the region-wide fit in that case rather than "fixing" something that
    worked.
    """

    def _floor_scene(self, object_height_m: float = 0.14):
        floor_depth = synthesize_plane_depth(SHAPE, INTRINSICS, (0.0, -0.25, 1.30))
        assert floor_depth is not None
        empty = floor_depth.astype(np.float32)
        mask = np.zeros(SHAPE, dtype=bool)
        mask[90:150, 120:220] = True
        columns, rows = np.meshgrid(
            np.arange(SHAPE[1], dtype=np.float64), np.arange(SHAPE[0], dtype=np.float64),
        )
        a, b, _c = (0.0, -0.25, 1.30)
        denominator = (
            1.0
            - a * (columns - INTRINSICS.ppx) / INTRINSICS.fx
            - b * (rows - INTRINSICS.ppy) / INTRINSICS.fy
        )
        offset = object_height_m * _plane_norm((0.0, -0.25, 1.30)) / denominator
        current = empty.copy()
        current[mask] = (empty[mask] - offset[mask]).astype(np.float32)
        return empty, current, mask

    def test_both_fits_agree_when_the_camera_points_at_the_floor(self) -> None:
        empty, current, mask = self._floor_scene(object_height_m=0.14)

        region_plane = fit_support_plane_from_background(
            current, INTRINSICS, object_mask=mask,
        )
        local_plane = fit_local_support_plane(
            current, INTRINSICS, mask, baseline_depth_m=empty,
        )

        for plane in (region_plane, local_plane):
            self.assertIsNotNone(plane)
            measured = estimate_object_dimensions(
                current, INTRINSICS, mask, plane, min_height_m=0.010, min_points=40,
            )
            self.assertIsNotNone(measured)
            self.assertAlmostEqual(measured.height_mm, 140.0, delta=15.0)


if __name__ == "__main__":
    unittest.main()
