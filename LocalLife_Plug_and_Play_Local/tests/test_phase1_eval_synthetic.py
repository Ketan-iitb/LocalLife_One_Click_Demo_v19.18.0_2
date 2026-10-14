"""Phase 1 evaluation, check C2 (supervisor review): the default volume method on exact synthetic depth.

Flat floor at 1.00 m seen straight down; boxes rendered independently of the estimator (pixel centres
inside the top face get the top depth). The pixel pitch is chosen so every edge is a whole number of
pixels, so the truth is exact and the tolerance can be the required +-0.5 %. Run before every data session:

    python -m unittest tests.test_phase1_eval_synthetic
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.types import CameraIntrinsics  # noqa: E402
from locallife_cloud.volume import ReferencePlane, estimate_volume  # noqa: E402

F = 7200.0                     # px: 0.125 mm per pixel at 0.90 m, 0.111 mm at 0.80 m -> whole pixels
FLOOR = 1.00
W, H = 2000, 1400
K = CameraIntrinsics(fx=F, fy=F, ppx=W / 2, ppy=H / 2, width=W, height=H)
PLANE = ReferencePlane(0.0, 0.0, W * H, (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
TOLERANCE = 0.005


def render(length_m: float, width_m: float, height_m: float):
    """Box on the floor, centred under the camera: (depth, empty, mask, true litres)."""
    top = FLOOR - height_m
    u = np.arange(W) - K.ppx
    v = np.arange(H) - K.ppy
    inside_u = np.abs(u * top / F) < length_m / 2 - 1e-12
    inside_v = np.abs(v * top / F) < width_m / 2 - 1e-12
    mask = inside_v[:, None] & inside_u[None, :]
    empty = np.full((H, W), FLOOR, np.float32)
    depth = empty.copy()
    depth[mask] = top
    return depth, empty, mask, length_m * width_m * height_m * 1000.0


def measure(depth, empty, mask) -> float:
    r = estimate_volume(depth, empty, K, object_mask=mask, geometry_mode=AppConfig().volume_geometry,
                        fill_small_holes=False, reference_plane=PLANE)
    return r.liters


class SupervisorSyntheticChecks(unittest.TestCase):
    def assertLitres(self, got, truth):
        self.assertLess(abs(got - truth) / truth, TOLERANCE, f"{got:.4f} L vs {truth:.4f} L")

    def test_default_method_is_the_top_surface_area_integral(self):
        self.assertEqual(AppConfig().volume_geometry, "support-plane-jacobian")

    def test_one_litre_box(self):                       # 10 x 10 x 10 cm, top at 0.90 m -> 1.000 L
        depth, empty, mask, truth = render(0.10, 0.10, 0.10)
        self.assertAlmostEqual(truth, 1.0, places=9)
        self.assertLitres(measure(depth, empty, mask), truth)

    def test_taller_box(self):                          # 10 x 10 x 20 cm standing, top at 0.80 m -> 2.000 L
        depth, empty, mask, truth = render(0.10, 0.10, 0.20)
        self.assertLitres(measure(depth, empty, mask), truth)

    def test_same_box_lying_and_standing(self):         # 20 x 10 x 10 cm: lying (top 0.90 m) vs standing (0.80 m)
        lying = measure(*render(0.20, 0.10, 0.10)[:3])
        standing = measure(*render(0.10, 0.10, 0.20)[:3])
        self.assertLitres(lying, 2.0)
        self.assertLitres(standing, 2.0)
        self.assertLess(abs(lying - standing) / 2.0, TOLERANCE)

    def test_floor_depth_area_would_fail(self):
        """The superseded floor-depth pixel area (Z_floor^2 / fx fy) inflates by (Z_floor / Z_top)^2."""
        depth, empty, mask, truth = render(0.10, 0.10, 0.20)
        legacy = float(np.sum((empty - depth)[mask] * (empty[mask] / F) ** 2)) * 1000.0
        self.assertGreater(legacy / truth, 1.5)          # (1.0 / 0.8)^2 = 1.5625


if __name__ == "__main__":
    unittest.main()
