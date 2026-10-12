"""V54.5: one milk carton, fixed camera, moved and turned -- the live per-pixel primary must not depend on
pose, position or which rigid estimator happens to succeed (dashboard: 1.06 / 1.01 / 1.87 L, third edge
5.6 vs 10.4 cm). Scenes are ray-traced independently of the estimators (test_v49_box_fix._scene / _obb).
Synthetic only: geometry and bookkeeping, not physical accuracy.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud import volume_experiment as ve  # noqa: E402
from locallife_cloud.pipeline import _measurable_detection  # noqa: E402
from locallife_cloud.types import Detection  # noqa: E402
from test_v45_bin_fill_events import _profile  # noqa: E402
from test_v49_box_fix import KW, _obb, _scene  # noqa: E402

CARTON = (0.095, 0.07, 0.20)                       # external edges, m (1.33 L)
TRUTH = CARTON[0] * CARTON[1] * CARTON[2] * 1000.0
POSES = {"upright": (0.095, 0.07, 0.20), "lying wide face down": (0.20, 0.095, 0.07),
         "lying narrow face down": (0.20, 0.07, 0.095)}
PLACEMENTS = ((0.0, 0.0, 0.0), (35.0, 0.15, 0.0), (70.0, -0.12, 0.08))     # yaw deg, x m, y m
PITCH, CAM = 45.0, 0.8


def scene(dims, yaw, dx, dy, mode="clean"):
    centre, axes, half = _obb(*dims, 0, yaw, 0.0)
    depth, mask = _scene(lambda x, y: 0 * x, (centre + np.array([dx, dy, 0.0]), axes, half), PITCH, cam=CAM,
                         far=CAM * 3)
    empty, _ = _scene(lambda x, y: 0 * x, None, PITCH, cam=CAM, far=CAM * 3)
    depth = depth + np.random.default_rng(3).normal(0, 0.003, depth.shape).astype(np.float32)
    r, c = np.nonzero(mask)
    box = (c.min(), r.min(), c.max() + 1, r.max() + 1)          # the detector box covers the whole object
    if mode == "partial":                                        # ... but its mask only the lower part
        mask = mask.copy()
        mask[:int(np.percentile(r, 45))] = False
    elif mode == "leak":                                         # ... or bleeds 6 px onto the floor
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    elif mode == "flying":                                       # mixed depths along the object's edge
        band = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & \
            ~cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        mix = np.random.default_rng(4).uniform(0, 1, depth.shape).astype(np.float32)
        depth = np.where(band, mix * depth + (1 - mix) * empty, depth).astype(np.float32)
    return depth, empty, mask, box


def live(depth, empty, mask, box):
    with TemporaryDirectory() as d:
        est = bf.FillEstimator("realsense", Path(d), _profile(camera_to_empty_floor_m=CAM, usable_height_m=CAM - 0.35,
                                                               tilt_from_vertical_deg=PITCH))
        est.recalibrate(empty, KW, None)
        est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
        result = est.object_volume(box, mask, 1.5, rigid_hint=True)
        return result, dict(est.last_object)


class CartonPlacementTests(unittest.TestCase):
    def _all(self, mode):
        out = []
        for pose, dims in POSES.items():
            for yaw, dx, dy in PLACEMENTS:
                result, obj = live(*scene(dims, yaw, dx, dy, mode))
                out.append((pose, yaw, result, obj))
        return out

    def assertConsistent(self, readings, tolerance=0.08):
        for pose, yaw, result, obj in readings:
            with self.subTest(pose=pose, yaw=yaw):
                self.assertIsNotNone(result, obj)
                self.assertTrue(obj["method"].startswith("per-pixel"), obj["method"])   # ONE method, every pose
                self.assertLess(abs(result[0] - TRUTH) / TRUTH, tolerance, (result, obj.get("estimators")))
                rigid = obj["rigid"]
                self.assertIsNotNone(rigid)
                for got, want in zip(rigid["edges_m"], sorted(CARTON, reverse=True)):
                    self.assertLess(abs(got - want), 0.012, rigid)                    # same 3 edges in any pose
        litres = [r[2][0] for r in readings]
        self.assertLess((max(litres) - min(litres)) / TRUTH, 0.10, litres)            # was 11 % clean, 32 % leak

    def test_moved_and_turned_carton_reads_the_same(self):
        self.assertConsistent(self._all("clean"))

    def test_bleeding_mask_does_not_switch_the_method(self):
        self.assertConsistent(self._all("leak"))                 # before: 1.02 ... 1.45 L, method switching

    def test_partial_detector_mask_is_completed_on_a_level_floor(self):
        # before: the uncovered part of the carton counted as "surroundings", the floor looked uneven, the
        # mask was not completed and the carton read -0.01 ... 0.58 L in some poses
        self.assertConsistent(self._all("partial"))

    def test_edge_smeared_depth_is_removed_or_flagged(self):
        for pose, yaw, result, obj in self._all("flying"):
            pp = obj["estimators"]["per_pixel"]
            with self.subTest(pose=pose, yaw=yaw):
                self.assertGreater(pp["flying_pixels_dropped"], 0)
                ok = abs(result[0] - TRUTH) / TRUTH < 0.10
                # either the value holds, or the frame says it cannot be trusted -- never a silent wrong value
                self.assertTrue(ok or pp["quality"].startswith("check"), (result, pp))


class StaleTrackTests(unittest.TestCase):
    def test_predicted_and_held_copies_create_no_geometry(self):
        for source in ("tracked-prediction", "held-through-dropout", "fixed-bin-depth-silhouette"):
            self.assertFalse(_measurable_detection(Detection("packaging", 0.5, (0, 0, 9, 9), source=source)))
        self.assertTrue(_measurable_detection(Detection("packaging", 0.5, (0, 0, 9, 9))))


class SideBySideEstimatorTests(unittest.TestCase):
    def test_experiment_path_reports_primary_and_secondary_separately(self):
        depth, empty, mask, box = scene(POSES["lying wide face down"], 35.0, 0.0, 0.0)
        k = {"fx": KW.fx, "fy": KW.fy, "ppx": KW.ppx, "ppy": KW.ppy, "width": KW.width, "height": KW.height}
        r = ve.measure_frame(depth, k, mask, empty, None, box=box)
        self.assertEqual(r["status"], "ok", r["reasons"])
        self.assertLess(abs(r["volume_l"] - TRUTH) / TRUTH, 0.08)
        self.assertIn("edge_artifact_share", r)
        cuboid = r["secondary_rigid_cuboid"]
        self.assertIsNotNone(cuboid)
        self.assertIn("secondary", cuboid["label"])
        self.assertNotEqual(cuboid["litres"], r["volume_l"])        # computed separately, never substituted


if __name__ == "__main__":
    unittest.main()
