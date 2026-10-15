"""Shape recognition (cuboid / cube / cylinder / sphere) on the live per-pixel path. Objects are ray-traced as
heightfields independently of the estimator (test_v49_box_fix._scene) at the thesis rig: camera 0.8 m above
the floor, 45 deg pitch, 3 mm depth noise. A round object reports L = W = its diameter; the per-pixel volume
stays the primary value. Synthetic only: geometry and bookkeeping, not physical accuracy.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import bin_fill as bf  # noqa: E402
from test_v45_bin_fill_events import _profile  # noqa: E402
from test_v49_box_fix import KW, _obb, _scene  # noqa: E402

PITCH, CAM = 45.0, 0.8


def _standing_cylinder(d, h):
    return lambda x, y: np.where(np.hypot(x, y) <= d / 2, h, 0.0)


def _lying_cylinder(d, length, yaw_deg=30.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def heights(x, y):
        along, across = x * c + y * s, -x * s + y * c
        r = d / 2
        top = r + np.sqrt(np.clip(r * r - across ** 2, 0, None))
        return np.where((np.abs(along) <= length / 2) & (np.abs(across) < r), top, 0.0)
    return heights


def _sphere(d):
    r = d / 2
    return lambda x, y: np.where(np.hypot(x, y) < r, r + np.sqrt(np.clip(r * r - x * x - y * y, 0, None)), 0.0)


def measure(heights=None, box=None):
    depth, mask = _scene(heights or (lambda x, y: 0 * x), box, PITCH, cam=CAM, far=CAM * 3)
    empty, _ = _scene(lambda x, y: 0 * x, None, PITCH, cam=CAM, far=CAM * 3)
    if box is None:
        mask = depth < empty - 0.01
    depth = depth + np.random.default_rng(3).normal(0, 0.003, depth.shape).astype(np.float32)
    r, c = np.nonzero(mask)
    bbox = (c.min(), r.min(), c.max() + 1, r.max() + 1)
    with TemporaryDirectory() as d:
        est = bf.FillEstimator("realsense", Path(d), _profile(camera_to_empty_floor_m=CAM, usable_height_m=CAM - 0.35,
                                                               tilt_from_vertical_deg=PITCH))
        est.recalibrate(empty, KW, None)
        est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
        result = est.object_volume(bbox, mask, 1.5, rigid_hint=box is not None)
        return result, dict(est.last_object)


class ShapeRecognitionTests(unittest.TestCase):
    def assertShape(self, obj, name):
        self.assertIsNotNone(obj.get("shape"), obj)
        self.assertEqual(obj["shape"]["name"], name, obj["shape"])

    def test_cuboid(self):
        _, obj = measure(box=_obb(0.25, 0.126, 0.099, 0, 20, 0.0))
        self.assertShape(obj, "cuboid")

    def test_cube(self):
        _, obj = measure(box=_obb(0.14, 0.14, 0.14, 0, 20, 0.0))
        self.assertShape(obj, "cube")

    def test_standing_cylinder_reports_diameter_by_diameter(self):
        result, obj = measure(_standing_cylinder(0.18, 0.08))
        self.assertShape(obj, "cylinder (standing)")
        self.assertEqual(obj["length_m"], obj["width_m"])                     # L = W = the diameter
        self.assertAlmostEqual(obj["shape"]["diameter_m"], 0.18, delta=0.018)
        self.assertAlmostEqual(obj["shape"]["litres"], math.pi * 0.09 ** 2 * 0.08 * 1000, delta=0.25)
        self.assertEqual(obj["litres"], result[0])                            # per-pixel stays the primary

    def test_lying_cylinder(self):
        _, obj = measure(_lying_cylinder(0.10, 0.26))
        self.assertShape(obj, "cylinder (lying)")
        self.assertAlmostEqual(obj["shape"]["diameter_m"], 0.10, delta=0.015)

    def test_sphere(self):
        _, obj = measure(_sphere(0.16))
        self.assertShape(obj, "sphere")
        self.assertEqual(obj["length_m"], obj["width_m"])

    def test_pillow_shaped_bag_is_not_a_cylinder(self):
        def pillow(x, y):                     # a filled 24 x 14 x 10 cm packet: falls off faster than a circle
            a = np.clip(1 - (x / 0.12) ** 2, 0, None) * np.clip(1 - (y / 0.07) ** 2, 0, None)
            return np.where(a > 0, 0.10 * np.sqrt(a), 0.0)
        _, obj = measure(pillow)
        self.assertNotIn("cylinder", obj["shape"]["name"])

    def test_shape_column_in_both_excel_layouts(self):
        from locallife_cloud.volume_experiment import ExperimentRecorder
        self.assertIn("RealSense shape", ExperimentRecorder.READING_COLUMNS)
        self.assertIn("Logitech shape", ExperimentRecorder.SIMPLE_COLUMNS)

    def test_session_recorded_before_the_shape_column_keeps_its_rows(self):
        import csv
        from locallife_cloud.volume_experiment import _append_csv
        with TemporaryDirectory() as d:
            path = Path(d) / "readings_detailed.csv"
            _append_csv(path, [{"reading_no": 1, "RealSense volume (L)": 1.2}])
            _append_csv(path, [{"reading_no": 2, "RealSense volume (L)": 1.3, "RealSense shape": "cylinder (standing)"}])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([r["RealSense volume (L)"] for r in rows], ["1.2", "1.3"])
        self.assertEqual([r["RealSense shape"] for r in rows], ["", "cylinder (standing)"])


if __name__ == "__main__":
    unittest.main()
