"""V51 box rotation: one rigid cuboid, several poses, through the real mask/depth -> measurement path.

Geometry is ray-traced independently of the estimator (test_v49_box_fix._scene / _obb); the estimator
only sees the depth map and the mask. Synthetic: this checks geometry and bookkeeping, not real-camera
accuracy (which needs the physical protocol in the V51 handoff).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud.pipeline import _same_geometry  # noqa: E402
from test_v45_bin_fill_events import _profile  # noqa: E402
from test_v49_box_fix import KW, _obb, _scene  # noqa: E402

BOX = (0.10, 0.125, 0.25)            # the user's carton, external, metres
POSES = {"upright": (0.10, 0.125, 0.25), "side (25 x 12.5 down)": (0.25, 0.125, 0.10),
         "side (25 x 10 down)": (0.25, 0.10, 0.125)}
OTHER = {"flat": (0.20, 0.15, 0.08), "on edge": (0.20, 0.08, 0.15)}


def measure(dims, pitch, cam, noise=0.003, yaw=20.0, usable=None):
    depth, mask = _scene(lambda x, y: 0 * x, _obb(*dims, 0, yaw, 0.0), pitch, cam=cam, far=cam * 3)
    empty, _ = _scene(lambda x, y: 0 * x, None, pitch, cam=cam, far=cam * 3)
    depth = depth + np.random.default_rng(1).normal(0, noise, depth.shape).astype(np.float32)
    with TemporaryDirectory() as d:
        est = bf.FillEstimator("realsense", Path(d), _profile(
            camera_to_empty_floor_m=cam, usable_height_m=cam - 0.35 if usable is None else usable,
            tilt_from_vertical_deg=pitch))
        est.recalibrate(empty, KW, None)
        est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
        r, c = np.nonzero(mask)
        result = est.object_volume((c.min(), r.min(), c.max() + 1, r.max() + 1), mask, 1.5, rigid_hint=True)
        return result, dict(est.last_object)


class CuboidPoseTests(unittest.TestCase):
    def _check(self, dims, pitch, cam):
        truth = dims[0] * dims[1] * dims[2] * 1000.0
        result, obj = measure(dims, pitch, cam)
        self.assertIsNotNone(result, obj)
        litres, height = result
        self.assertTrue(obj["method"].startswith("box "), obj["method"])
        self.assertLess(abs(litres - truth) / truth, 0.05)
        # The SAME estimate: displayed litres = L x W x H (metres -> litres), height = the vertical extent.
        product = obj["length_m"] * obj["width_m"] * obj["height_m"] * 1000.0
        self.assertAlmostEqual(litres, product, delta=0.01)
        self.assertAlmostEqual(height, obj["height_m"], places=6)
        self.assertAlmostEqual(obj["height_m"], dims[2], delta=0.012)      # height above the support
        measured = sorted((obj["length_m"], obj["width_m"], obj["height_m"]))
        for got, want in zip(measured, sorted(dims)):
            self.assertAlmostEqual(got, want, delta=0.012)                  # cm-level, all three axes

    def test_same_box_upright_and_on_two_sides_oblique_views(self) -> None:
        for pitch, cam in ((15.0, 0.9), (45.0, 0.8)):
            for pose, dims in POSES.items():
                with self.subTest(pitch=pitch, pose=pose):
                    self._check(dims, pitch, cam)

    def test_another_cuboid_with_different_proportions(self) -> None:
        for pose, dims in OTHER.items():
            with self.subTest(pose=pose):
                self._check(dims, 45.0, 0.8)

    def test_volume_stays_constant_when_the_box_is_turned(self) -> None:
        volumes = [measure(dims, 45.0, 0.8)[0][0] for dims in POSES.values()]
        self.assertLess((max(volumes) - min(volumes)) / (BOX[0] * BOX[1] * BOX[2] * 1000.0), 0.06)

    def test_noise_free_box_has_no_systematic_error(self) -> None:
        for pose, dims in POSES.items():
            with self.subTest(pose=pose):
                result, _ = measure(dims, 45.0, 0.8, noise=0.0)
                self.assertLess(abs(result[0] - 3.125) / 3.125, 0.03)

    def test_invalid_profile_gives_no_number(self) -> None:
        # Usable height not below the camera: the fill profile is incomplete, nothing is measured.
        result, obj = measure(POSES["upright"], 45.0, 0.8, usable=1.0)
        self.assertIsNone(result)
        self.assertEqual(obj, {})


class StaleGeometryTests(unittest.TestCase):
    UPRIGHT = (0.0, 3.12, 0.25, 0.125, 0.10, "box from 3 visible faces")
    SIDE = (5.0, 3.09, 0.10, 0.25, 0.125, "box cuboid: face L x W x thickness")

    def test_turning_the_box_starts_a_new_geometry(self) -> None:
        self.assertFalse(_same_geometry(self.UPRIGHT, self.SIDE))                 # 25 cm -> 10 cm high
        jitter = (1.0, 3.2, 0.245, 0.127, 0.101, "box from 2 visible faces")
        self.assertTrue(_same_geometry(self.UPRIGHT, jitter))                     # same pose, noise
        surface = (1.0, 3.1, 0.25, 0.125, 0.10, "deformable object: visible surface integrated")
        self.assertFalse(_same_geometry(self.UPRIGHT, surface))                   # another quantity


if __name__ == "__main__":
    unittest.main()
