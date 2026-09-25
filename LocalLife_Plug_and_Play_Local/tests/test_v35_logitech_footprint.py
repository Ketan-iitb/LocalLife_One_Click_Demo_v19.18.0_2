"""Footprint of a standing object seen by one fixed, tilted camera.

Field evidence this is built from (RealSense truth vs Logitech reported):

    shoe box    333x262  ->  334x294   flat, wide   ~1.0x
    bag         259x237  ->  282x275   tall, wide   ~1.1x
    can          47x 28  ->  115x 85   84 mm tall   ~2.4-3.0x
    bottle       75x 32  ->  227x133   204 mm tall  ~3.0-4.2x

The error tracks height, not detection quality: warping a whole silhouette
through a floor homography smears a standing object into its own shadow.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from locallife_cloud.logitech_footprint import (
    CALIBRATION_MISSING,
    IMPLAUSIBLE_FOOTPRINT,
    INSUFFICIENT_CONTACT,
    MASK_DEPTH_SHAPE_MISMATCH,
    OBJECT_BELOW_NOISE_FLOOR,
    contact_band_mask,
    measure_footprint,
    plausible_height_m,
)


class _Zone:
    """A mat seen by a real pinhole camera, tilted down towards it.

    Exact forward projection and its exact inverse, so a test can place a 3D
    object, render the silhouette a segmenter would see, and check what the
    code recovers. The tilt is what makes the bug reproducible: a straight-down
    camera has no smear at all.
    """

    has_floor_scale = True
    width_m = 1.2
    depth_m = 1.2

    def __init__(self, camera_height_m=1.2, camera_y_m=-0.8, tilt_deg=55.0,
                 focal=600.0, shape=(480, 640)):
        self.camera_height_m = camera_height_m
        self.camera_y_m = camera_y_m
        self.focal = focal
        self.shape = shape
        angle = math.radians(tilt_deg)
        self.sin, self.cos = math.sin(angle), math.cos(angle)

    # -- forward: a 3D point -> the pixel that sees it ----------------------
    def project(self, x, y, z):
        dx, dy, dz = x, y - self.camera_y_m, z - self.camera_height_m
        forward = dy * self.cos - dz * self.sin
        down = dy * self.sin + dz * self.cos
        if forward <= 1e-6:
            return -1.0, -1.0
        row = self.shape[0] / 2.0 + self.focal * down / forward
        column = self.shape[1] / 2.0 + self.focal * dx / forward
        return float(row), float(column)

    # -- inverse: a pixel -> where its ray meets the floor (z = 0) ----------
    def ground_points(self, rows, columns, shape):
        rows = np.asarray(rows, dtype=np.float64)
        columns = np.asarray(columns, dtype=np.float64)
        x_ray = (columns - shape[1] / 2.0) / self.focal
        y_ray = (rows - shape[0] / 2.0) / self.focal
        # Ray direction in world coordinates.
        dir_y = y_ray * self.sin + self.cos
        dir_z = y_ray * self.cos - self.sin
        with np.errstate(divide="ignore", invalid="ignore"):
            steps = np.where(dir_z < -1e-9, -self.camera_height_m / dir_z, np.nan)
        x = steps * x_ray
        y = self.camera_y_m + steps * dir_y
        keep = np.isfinite(x) & np.isfinite(y)
        return np.column_stack((x[keep], y[keep]))

    def pixel_area_m2(self, shape):
        return None

    @property
    def nadir(self):
        return (0.0, self.camera_y_m)


def _render_cylinder(zone, diameter_m, height_m, centre=(0.0, 0.0), samples=90):
    """Silhouette of an upright cylinder standing on the mat."""
    mask = np.zeros(zone.shape, dtype=bool)
    radius = diameter_m / 2.0
    for angle in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        x = centre[0] + radius * math.cos(angle)
        y = centre[1] + radius * math.sin(angle)
        for z in np.linspace(0.0, height_m, samples):
            row, column = zone.project(x, y, z)
            r, c = int(round(row)), int(round(column))
            if 0 <= r < zone.shape[0] and 0 <= c < zone.shape[1]:
                mask[r, c] = True
    # Fill the silhouette column-wise, as a segmenter would produce it.
    filled = np.zeros_like(mask)
    for column in range(mask.shape[1]):
        rows = np.nonzero(mask[:, column])[0]
        if rows.size:
            filled[rows[0]: rows[-1] + 1, column] = True
    return filled


def _render_box(zone, length_m, width_m, height_m, centre=(0.0, 0.0), samples=60):
    mask = np.zeros(zone.shape, dtype=bool)
    xs = np.linspace(-length_m / 2, length_m / 2, samples) + centre[0]
    ys = np.linspace(-width_m / 2, width_m / 2, samples) + centre[1]
    for x in xs:
        for y in ys:
            for z in (0.0, height_m):
                row, column = zone.project(x, y, z)
                r, c = int(round(row)), int(round(column))
                if 0 <= r < zone.shape[0] and 0 <= c < zone.shape[1]:
                    mask[r, c] = True
    filled = np.zeros_like(mask)
    for column in range(mask.shape[1]):
        rows = np.nonzero(mask[:, column])[0]
        if rows.size:
            filled[rows[0]: rows[-1] + 1, column] = True
    return filled


class TallObjectTests(unittest.TestCase):
    """1: a tall object's silhouette must not be projected onto the floor."""

    def setUp(self) -> None:
        self.zone = _Zone()

    def _whole_silhouette_extent(self, mask):
        """What the old code did, for comparison."""
        from locallife_cloud.footprint import minimum_area_extents

        rows, columns = np.nonzero(mask)
        points = self.zone.ground_points(rows, columns, mask.shape)
        long_side, short_side, _ = minimum_area_extents(points)
        return long_side, short_side

    def test_a_can_is_not_smeared_into_its_own_shadow(self) -> None:
        # 47 x 28 x 84 mm is roughly a drink can; use a 50 mm round one.
        mask = _render_cylinder(self.zone, 0.05, 0.084)
        old_long, _ = self._whole_silhouette_extent(mask)
        result = measure_footprint(self.zone, mask, zone_limits_m=(1.2, 1.2))
        self.assertTrue(result.ok, result.reason)
        # The old method inflates well past the true diameter...
        self.assertGreater(old_long, 0.09)
        # ...the new one stays near it.
        self.assertLess(result.length_m, 0.075)
        self.assertGreater(result.length_m, 0.03)

    def test_a_tall_bottle_is_not_reported_as_a_dinner_plate(self) -> None:
        # The cosmetic bottle: 204 mm tall, ~75 mm across, reported 227 x 133.
        mask = _render_cylinder(self.zone, 0.075, 0.204)
        old_long, _ = self._whole_silhouette_extent(mask)
        result = measure_footprint(self.zone, mask, zone_limits_m=(1.2, 1.2))
        self.assertTrue(result.ok, result.reason)
        self.assertGreater(old_long, 0.18)
        self.assertLess(result.length_m, 0.12)

    def test_the_error_grows_with_height_under_the_old_method_only(self) -> None:
        short = _render_cylinder(self.zone, 0.06, 0.03)
        tall = _render_cylinder(self.zone, 0.06, 0.20)
        old_short, _ = self._whole_silhouette_extent(short)
        old_tall, _ = self._whole_silhouette_extent(tall)
        new_short = measure_footprint(self.zone, short, zone_limits_m=(1.2, 1.2))
        new_tall = measure_footprint(self.zone, tall, zone_limits_m=(1.2, 1.2))
        # Old: the taller object reads far wider despite the same diameter.
        self.assertGreater(old_tall, old_short * 1.8)
        # New: both stay within a factor of each other.
        self.assertLess(new_tall.length_m, new_short.length_m * 1.8)


class FlatObjectTests(unittest.TestCase):
    """The cases that already measured well must not move."""

    def setUp(self) -> None:
        self.zone = _Zone()

    def test_a_shoe_box_keeps_its_footprint(self) -> None:
        mask = _render_box(self.zone, 0.333, 0.262, 0.12)
        result = measure_footprint(self.zone, mask, zone_limits_m=(1.2, 1.2))
        self.assertTrue(result.ok, result.reason)
        # Within a quarter of truth from a single view of a box top.
        self.assertGreater(result.length_m, 0.25)
        self.assertLess(result.length_m, 0.45)

    def test_length_is_never_below_width(self) -> None:
        mask = _render_box(self.zone, 0.30, 0.20, 0.10)
        result = measure_footprint(self.zone, mask, zone_limits_m=(1.2, 1.2))
        self.assertGreaterEqual(result.length_m, result.width_m)


class ContactBandTests(unittest.TestCase):
    def test_the_band_is_the_bottom_of_the_silhouette(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 30:60] = True
        band = contact_band_mask(mask)
        rows = np.nonzero(band.any(axis=1))[0]
        self.assertEqual(int(rows[-1]), 79)
        # Proportional to the object, not a fixed pixel count.
        self.assertLessEqual(rows.size, 20)
        self.assertGreaterEqual(rows.size, 3)

    def test_a_short_silhouette_still_yields_usable_rows(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[50:54, 30:60] = True
        self.assertGreaterEqual(int(contact_band_mask(mask).sum()), 30)

    def test_an_empty_mask_gives_an_empty_band(self) -> None:
        self.assertFalse(contact_band_mask(np.zeros((10, 10), dtype=bool)).any())


class HeightCorrectedTests(unittest.TestCase):
    """With per-pixel height and the nadir, the smear is undone exactly."""

    def setUp(self) -> None:
        self.zone = _Zone()

    def test_known_heights_recover_the_true_ground_positions(self) -> None:
        # Build the mask from points whose true height is known exactly, so the
        # height map is truth rather than an assumption about what a filled
        # silhouette contains. This tests the correction itself: every surface
        # point of a 5 cm cylinder must come back onto a 5 cm circle.
        diameter, height = 0.05, 0.084
        radius = diameter / 2.0
        mask = np.zeros(self.zone.shape, dtype=bool)
        heights = np.zeros(self.zone.shape, dtype=np.float64)
        for angle in np.linspace(0, 2 * math.pi, 120, endpoint=False):
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            for z in np.linspace(0.0, height, 40):
                row, column = self.zone.project(x, y, z)
                r, c = int(round(row)), int(round(column))
                if 0 <= r < self.zone.shape[0] and 0 <= c < self.zone.shape[1]:
                    mask[r, c] = True
                    heights[r, c] = z
        result = measure_footprint(
            self.zone, mask, heights_m=heights,
            camera_height_m=self.zone.camera_height_m,
            nadir_xy_m=self.zone.nadir, zone_limits_m=(1.2, 1.2),
        )
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.method, "logitech_height_corrected_footprint")
        # Within a pixel-quantisation margin of the true 50 mm diameter.
        self.assertLess(result.length_m, 0.075)
        self.assertGreater(result.length_m, 0.035)

    def test_the_correction_beats_the_uncorrected_warp_on_the_same_points(self) -> None:
        from locallife_cloud.footprint import minimum_area_extents

        radius, height = 0.025, 0.084
        mask = np.zeros(self.zone.shape, dtype=bool)
        heights = np.zeros(self.zone.shape, dtype=np.float64)
        for angle in np.linspace(0, 2 * math.pi, 120, endpoint=False):
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            for z in np.linspace(0.0, height, 40):
                row, column = self.zone.project(x, y, z)
                r, c = int(round(row)), int(round(column))
                if 0 <= r < self.zone.shape[0] and 0 <= c < self.zone.shape[1]:
                    mask[r, c] = True
                    heights[r, c] = z
        rows, columns = np.nonzero(mask)
        uncorrected = minimum_area_extents(
            self.zone.ground_points(rows, columns, mask.shape)
        )[0]
        corrected = measure_footprint(
            self.zone, mask, heights_m=heights,
            camera_height_m=self.zone.camera_height_m,
            nadir_xy_m=self.zone.nadir, zone_limits_m=(1.2, 1.2),
        )
        self.assertLess(corrected.length_m, uncorrected * 0.7)

    def test_a_height_map_of_the_wrong_shape_is_refused(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        result = measure_footprint(
            self.zone, mask, heights_m=np.zeros((20, 20)),
            camera_height_m=1.2, nadir_xy_m=(0.0, 0.0),
        )
        self.assertEqual(result.reason, MASK_DEPTH_SHAPE_MISMATCH)


class RefusalTests(unittest.TestCase):
    """Every refusal names its cause. None of them is the word "pending"."""

    def test_no_calibration_says_so(self) -> None:
        self.assertEqual(
            measure_footprint(None, np.ones((10, 10), dtype=bool)).reason,
            CALIBRATION_MISSING,
        )

    def test_an_empty_mask_says_so(self) -> None:
        self.assertEqual(
            measure_footprint(_Zone(), np.zeros((10, 10), dtype=bool)).reason,
            INSUFFICIENT_CONTACT,
        )

    def test_a_footprint_larger_than_the_mat_is_refused_not_clamped(self) -> None:
        zone = _Zone()
        mask = _render_box(zone, 0.30, 0.25, 0.05)
        result = measure_footprint(zone, mask, zone_limits_m=(0.05, 0.05))
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, IMPLAUSIBLE_FOOTPRINT)

    def test_height_plausibility_separates_noise_from_impossible(self) -> None:
        self.assertEqual(plausible_height_m(0.002, 1.2)[1], OBJECT_BELOW_NOISE_FLOOR)
        self.assertEqual(plausible_height_m(None, 1.2)[1], "insufficient_object_depth")
        self.assertEqual(plausible_height_m(1.5, 1.2)[1], IMPLAUSIBLE_FOOTPRINT)
        self.assertTrue(plausible_height_m(0.084, 1.2)[0])

    def test_no_reason_is_the_bare_word_pending(self) -> None:
        from locallife_cloud import logitech_footprint as module

        reasons = [
            value for name, value in vars(module).items()
            if name.isupper() and isinstance(value, str)
        ]
        self.assertTrue(reasons)
        for reason in reasons:
            self.assertNotEqual(reason.strip().lower(), "pending")


class IndependenceTests(unittest.TestCase):
    def test_the_module_holds_no_realsense_state(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent
            / "locallife_cloud" / "logitech_footprint.py"
        ).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )
        body = code.split('"""', 2)[-1]
        for forbidden in ("realsense", "RealSense", "depth_frame", "rs."):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
