"""V52: the two failures seen on the real full-bin screenshots (SYNTHETIC reproductions)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud.colour_evidence import describe_colour  # noqa: E402


class TiltedFullBinTests(unittest.TestCase):
    def test_full_bin_on_a_tilted_mount_without_a_measured_tilt_still_reads_a_fill(self) -> None:
        # Floor hidden under a crumpled pile, camera tilted 30 deg, tilt NOT entered in the profile.
        # V51 assumed "up" = camera axis, called most of the pile a wall and reported
        # "waiting for a clear view of the bin surface" despite 100 % valid depth.
        from test_v49_box_fix import KW, _scene, _uneven_bags

        depth, _ = _scene(_uneven_bags, None, 30.0, 1.10)
        with TemporaryDirectory() as d:
            profile = bf.FillProfile(camera_id="realsense", camera_to_empty_floor_m=1.10,
                                     distance_kind="vertical", tilt_from_vertical_deg=None, usable_height_m=1.00)
            est = bf.FillEstimator("realsense", Path(d), profile)
            reading = est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
        self.assertEqual(reading["status"], "ok", reading.get("reason"))
        self.assertIn("not measured", reading["diagnostics"]["wall_filter"])


class ChangedSurroundColourTests(unittest.TestCase):
    def test_a_changed_surround_is_not_treated_as_a_lighting_change(self) -> None:
        # A white bag whose neighbours are NEW green and dark bags (not the reference surface):
        # chroma gains from them used to tint the white bag.
        rng = np.random.default_rng(0)
        reference = rng.integers(60, 200, (120, 160, 3)).astype(np.uint8)       # empty-bin texture
        frame = np.zeros_like(reference)
        frame[:] = (60, 150, 60)                                                 # pile of green bags
        frame[::7] = (30, 30, 30)
        mask = np.zeros((120, 160), bool)
        mask[30:90, 40:120] = True
        frame[mask] = (225, 228, 230)                                            # white bag
        result = describe_colour(frame, mask, background_bgr=reference)
        self.assertIsNone(result.illumination_gains)
        self.assertEqual(result.colour, "white")


if __name__ == "__main__":
    unittest.main()
