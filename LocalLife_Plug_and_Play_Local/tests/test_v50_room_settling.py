"""V50: deposit counting in a ROOM view (synthetic; logic only, not physical verification).

Reproduces the two live rejections from the V49 screenshots with an explicit noise model:
* RealSense "an existing bag moved: its old spot was vacated" -- per-frame stereo noise that grows
  with range squared (far wall at ~3.5 m) used to read as a vacated spot;
* Logitech "scene did not settle within 30 s (trigger: motion)" -- a person moving far from the
  placed bag kept the whole view "unsettled".
Then the full sequence: baseline -> 3 insertions -> move -> removal, counted exactly once each.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud import session_deposits as sd  # noqa: E402

H, W = 120, 160
ROWS = np.linspace(3.5, 1.2, H, dtype=np.float32)[:, None]        # far wall (top) -> near floor (bottom)


class RoomScene:
    def __init__(self, camera="realsense", depth=True, seed=0, noise_k=0.0025, walker=False):
        rng = np.random.default_rng(seed)
        self.camera, self.depth, self.walker = camera, depth, walker
        self.floor_grey = rng.integers(60, 120, (H, W)).astype(np.int16)
        self.floor_depth = np.repeat(ROWS, W, axis=1)
        self.noise = np.random.default_rng(seed + 1)
        self.noise_k = noise_k
        self.bags: dict[int, dict] = {}

    def add(self, tid, box, height=0.25, shade=30, label="backpack"):
        self.bags[tid] = dict(box=box, height=height, shade=shade, label=label)

    def evidence(self, t):
        grey = self.floor_grey.copy()
        depth = self.floor_depth.copy()
        for bag in self.bags.values():
            x1, y1, x2, y2 = bag["box"]
            grey[y1:y2, x1:x2] = bag["shade"]
            depth[y1:y2, x1:x2] -= bag["height"]
        if self.walker:                                      # a person at the far wall, always moving
            x = int(t * 23) % (W - 20)
            grey[2:30, x:x + 20] = 240 - (int(t * 10) % 3) * 70
            depth[2:30, x:x + 20] = 3.0
        sigma = self.noise_k * depth * depth                   # stereo RMS ~ k z^2
        noisy = depth + self.noise.normal(0.0, 1.0, depth.shape).astype(np.float32) * sigma
        grey = np.clip(grey + self.noise.integers(-4, 5, grey.shape), 0, 255).astype(np.uint8)
        tracks = [sd.TrackInfo(tid, tuple(float(v) for v in b["box"]), label=b["label"], colour="black")
                  for tid, b in self.bags.items()]
        return sd.FrameEvidence(self.camera, t, grey, np.ones((H, W), bool), tracks,
                                depth=noisy.astype(np.float32) if self.depth else None)


def run(counter, scene, start, end, step=0.5):
    t, out = start, []
    while t < end - 1e-9:
        r = counter.observe(scene.evidence(t))
        if r is not None:
            out.append(r)
        t += step
    return out, t


class RoomSettlingTests(unittest.TestCase):
    def _counter(self, d):
        return sd.SessionDeposits(Path(d), clock=lambda: 0.0)

    def test_far_range_depth_noise_is_not_a_vacated_spot(self) -> None:
        with TemporaryDirectory() as d:
            counter, scene = self._counter(d), RoomScene()
            run(counter, scene, 0.0, 6.0)
            scene.add(1, (60, 85, 100, 115))                     # backpack near the camera
            run(counter, scene, 6.0, 12.0)
            self.assertEqual(counter.count_for("realsense"), 1)
            self.assertFalse([r for r in counter.rejected if "vacated" in r.get("reason", "")])

    def test_motion_far_from_the_bag_does_not_block_settling(self) -> None:
        with TemporaryDirectory() as d:
            counter, scene = self._counter(d), RoomScene(camera="logitech", depth=False, walker=True)
            run(counter, scene, 0.0, 22.0)                      # baseline taken at BASELINE_MAX_S
            scene.add(1, (60, 85, 100, 115))
            run(counter, scene, 22.0, 30.0)
            self.assertEqual(counter.count_for("logitech"), 1)
            self.assertFalse([r for r in counter.rejected if "did not settle" in r.get("reason", "")])

    def test_sequence_counts_insertions_once_and_not_moves_or_removals(self) -> None:
        for camera, depth in (("realsense", True), ("logitech", False)):
            with self.subTest(camera=camera), TemporaryDirectory() as d:
                counter, scene = self._counter(d), RoomScene(camera=camera, depth=depth, seed=3)
                scene.add(9, (10, 90, 40, 115), label="waste bag")   # present at start: baseline
                t = run(counter, scene, 0.0, 6.0)[1]
                counts = []
                for tid, box in ((1, (50, 85, 80, 115)), (2, (90, 85, 120, 115)), (3, (125, 80, 155, 110))):
                    scene.add(tid, box, label="plastic bag")
                    t = run(counter, scene, t, t + 6.0)[1]
                    counts.append(counter.count_for(camera))
                    t = run(counter, scene, t, t + 3.0)[1]          # repeated frames, nothing new
                self.assertEqual(counts, [1, 2, 3])
                scene.bags[2]["box"] = (90, 50, 120, 80)             # move an existing bag
                t = run(counter, scene, t, t + 6.0)[1]
                scene.bags.pop(3)                                    # remove one
                t = run(counter, scene, t, t + 6.0)[1]
                self.assertEqual(counter.count_for(camera), 3)
                ids = [e["event_id"] for e in counter.events if e["camera"] == camera]
                self.assertEqual(len(ids), len(set(ids)), "an event was committed twice")


if __name__ == "__main__":
    unittest.main()


class SoftObjectGeometryTests(unittest.TestCase):
    """A backpack-like soft object (flat-ish top, rounded sides, 2.5 cm straps), SYNTHETIC exact depth.
    V49 measured it as 'box cuboid: face L x W x thickness from visible side face' = 2.5-3.2 L for
    11.55 L; the same mechanism produced the live Logitech 14.8 x 13.3 x 6.1 cm = 1.2 L reading."""

    @staticmethod
    def _shape(x, y):
        r = (np.abs(x) / 0.18) ** 4 + (np.abs(y) / 0.14) ** 4
        body = np.where(r < 1, 0.14 * np.clip(1 - r, 0, 1) ** 0.25, 0.0)
        strap = np.where((np.abs(y - 0.05) < 0.02) & (x > 0.18) & (x < 0.30), 0.025, 0.0)
        return np.maximum(body, strap)

    def _measure(self, pitch, **hints):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_v45_bin_fill_events import _profile
        from test_v49_box_fix import KW, _scene

        from locallife_cloud import bin_fill as bf
        depth, _ = _scene(self._shape, None, pitch, 1.0)
        empty, _ = _scene(lambda x, y: 0 * x, None, pitch, 1.0)
        mask = depth < empty - 0.01
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("logitech", Path(d), _profile(camera_to_empty_floor_m=1.0, usable_height_m=0.9,
                                                                   tilt_from_vertical_deg=pitch))
            est.recalibrate(empty, KW, None)
            est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
            r, c = np.nonzero(mask)
            result = est.object_volume((c.min(), r.min(), c.max() + 1, r.max() + 1), mask, 1.5, **hints)
            return result, dict(est.last_object)

    def test_soft_object_is_integrated_not_forced_into_a_box(self) -> None:
        g = np.linspace(-0.4, 0.4, 1601)
        x, y = np.meshgrid(g, g)
        truth = float(self._shape(x, y).sum() * (g[1] - g[0]) ** 2 * 1000)          # 11.55 L
        for pitch in (15.0, 45.0):
            for hints in ({"deformable_hint": True}, {}):                          # backpack class / unresolved
                with self.subTest(pitch=pitch, hints=hints):
                    (litres, _), obj = self._measure(pitch, **hints)
                    self.assertFalse(obj["method"].startswith(("box", "tilted rigid")))
                    self.assertLess(abs(litres - truth) / truth, 0.12)
        _, obj = self._measure(15.0, deformable_hint=True)
        self.assertLess(obj["length_m"], 0.40)              # body (36 cm), not body + straps (47 cm)
        self.assertTrue(obj["method"].startswith("deformable object"))
