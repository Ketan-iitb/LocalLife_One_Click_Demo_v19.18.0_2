"""Regression coverage for the "pale yellow labelled grey" bug.

Reported directly on real hardware: a yellow bag/box was being classified
"grey" by the legacy dashboard's colour path. Root cause was
`dominant_color()`'s HSV saturation gate firing before any hue evidence was
consulted -- a pale, washed-out warm surface (cream milk carton, translucent
yellow-tinted bag) has a small max-min channel spread and therefore low HSV
saturation, even though it is unmistakably warm-toned rather than neutral.
"""

from __future__ import annotations

import unittest

import numpy as np

from locallife_cloud.geometry import _lab_b_channel, dominant_color


def _swatch(bgr: tuple[int, int, int], size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    frame = np.tile(np.array(bgr, dtype=np.uint8), (size, size, 1))
    mask = np.ones((size, size), dtype=bool)
    return frame, mask


class PaleYellowColorTests(unittest.TestCase):
    def test_pale_cream_milk_carton_is_yellow_not_grey(self) -> None:
        # Approximates a cream/pale-yellow Tetra Brik-style milk carton:
        # RGB ~ (245, 238, 205) -> low HSV saturation (~0.16), but a real
        # positive LAB b*.
        frame, mask = _swatch((205, 238, 245))
        self.assertEqual(dominant_color(frame, mask), "yellow")

    def test_translucent_yellow_bag_is_yellow_not_grey(self) -> None:
        # A pale, slightly desaturated yellow plastic bag skin.
        frame, mask = _swatch((150, 225, 235))
        self.assertEqual(dominant_color(frame, mask), "yellow")

    def test_genuinely_neutral_light_grey_is_unaffected(self) -> None:
        frame, mask = _swatch((190, 190, 190))
        self.assertEqual(dominant_color(frame, mask), "grey")

    def test_genuinely_neutral_white_is_unaffected(self) -> None:
        frame, mask = _swatch((240, 240, 240))
        self.assertEqual(dominant_color(frame, mask), "white")

    def test_genuinely_neutral_mid_grey_is_unaffected(self) -> None:
        frame, mask = _swatch((120, 120, 120))
        self.assertEqual(dominant_color(frame, mask), "grey")

    def test_fully_saturated_yellow_still_works(self) -> None:
        frame, mask = _swatch((20, 220, 230))
        self.assertEqual(dominant_color(frame, mask), "yellow")

    def test_pale_blue_is_not_misclassified_as_yellow(self) -> None:
        # A cool-toned pale colour must not be swept into "yellow" just
        # because it shares low HSV saturation with the pale-yellow case --
        # the LAB b* sign is what disambiguates warm from cool.
        frame, mask = _swatch((245, 225, 205))
        self.assertEqual(dominant_color(frame, mask), "blue")

    def test_lab_b_channel_sign_matches_warm_cool_intuition(self) -> None:
        # Warm (yellow-leaning) colour: positive b*.
        self.assertGreater(_lab_b_channel(blue=0.80, green=0.93, red=0.96), 0.0)
        # Cool (blue-leaning) colour: negative b*.
        self.assertLess(_lab_b_channel(blue=0.96, green=0.88, red=0.80), 0.0)
        # Neutral grey: b* near zero.
        self.assertAlmostEqual(_lab_b_channel(blue=0.5, green=0.5, red=0.5), 0.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
