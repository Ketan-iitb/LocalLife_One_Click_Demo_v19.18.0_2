"""Re-find floor on request at a room-floor mounting height (~75 cm): accepted when the operator asks,
still refused unattended (where a flat waste layer could be mistaken for the floor)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import bin_fill as bf  # noqa: E402
from test_v45_bin_fill_events import _profile  # noqa: E402
from test_v49_box_fix import KW, _scene  # noqa: E402


class FloorRefindTests(unittest.TestCase):
    def test_manual_refind_accepts_a_75_cm_floor_and_automatic_does_not(self):
        empty, _ = _scene(lambda x, y: 0 * x, None, 30.0, cam=0.75, far=2.4)
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), _profile(camera_to_empty_floor_m=1.10, usable_height_m=0.5,
                                                                   tilt_from_vertical_deg=30.0))
            automatic = est.recalibrate(empty, KW, None, automatic=True)
            self.assertFalse(automatic["ok"], automatic)
            manual = est.recalibrate(empty, KW, None)
            self.assertTrue(manual["ok"], manual)
            self.assertAlmostEqual(manual["floor_cm"], 75.0, delta=3.0)


if __name__ == "__main__":
    unittest.main()
