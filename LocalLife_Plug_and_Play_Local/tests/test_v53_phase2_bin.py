"""V53: Phase 2 bin geometry (synthetic; logic only, not physical verification).

Reproduces the Phase 2 live failures from the HAR / screenshots:
* the live support plane fitted the bin WALL (82 deg) and objects read 31-55 cm tall;
* a tilted RealSense floor (perpendicular ~84 cm, 110 cm along the view) was never accepted;
* the Logitech scale put the PERPENDICULAR floor distance at the 110 cm reference (heights x 1/cos tilt);
* line-of-sight "rises" of 219-355 cm (lid / hand / person) were counted as bags;
* whole-bin deltas of -542 L / +440 L were stored as bag evidence;
* yesterday's session was resumed and its room tests still counted.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud import session_deposits as sd  # noqa: E402
from locallife_cloud.pipeline import VisionPipeline  # noqa: E402
from locallife_cloud.volume import ReferencePlane  # noqa: E402
from test_v45_bin_fill_events import _profile  # noqa: E402
from test_v49_box_fix import KW, _scene  # noqa: E402
from test_v50_room_settling import RoomScene, run  # noqa: E402


def _plane(tilt_deg: float) -> ReferencePlane:
    a = math.tan(math.radians(tilt_deg))
    norm = math.sqrt(a * a + 1.0)
    return ReferencePlane(tilt_deg, 0.002, 5000, (a / norm, 0.0, -1.0 / norm), (a, 0.0, 1.0))


def _fake_pipeline(tilt=None, floor=None, scale=1.0):
    profile = SimpleNamespace(tilt_from_vertical_deg=tilt, floor_plane=floor, floor_scale=scale)
    return SimpleNamespace(fill=SimpleNamespace(profile=profile))


class SupportPlaneTests(unittest.TestCase):
    def test_wall_plane_is_rejected(self) -> None:
        check = VisionPipeline._implausible_support_plane
        self.assertIsNotNone(check(_fake_pipeline(tilt=43.7), _plane(82.15)))     # live HAR: wall at 82 deg
        self.assertIsNotNone(check(_fake_pipeline(tilt=None), _plane(82.15)))
        self.assertIsNone(check(_fake_pipeline(tilt=43.7), _plane(40.0)))         # the real floor
        self.assertIsNone(check(_fake_pipeline(tilt=None), _plane(40.0)))

    def test_fitted_bin_floor_becomes_the_measurement_plane(self) -> None:
        floor = VisionPipeline._bin_floor_plane(_fake_pipeline(floor=[0.0, 0.84, 0.70]))
        self.assertAlmostEqual(floor.tilt_degrees, math.degrees(math.atan(0.84)), places=4)
        self.assertEqual(floor.coefficients, (0.0, 0.84, 0.70))
        # Logitech model depth (scale != 1) is not metric: never used as a RealSense plane.
        self.assertIsNone(VisionPipeline._bin_floor_plane(_fake_pipeline(floor=[0.0, 0.84, 0.70], scale=0.4)))
        self.assertIsNone(VisionPipeline._bin_floor_plane(_fake_pipeline()))


class FloorCalibrationTests(unittest.TestCase):
    PITCH, AXIS = 40.0, 1.10                              # 110 cm along the view, mount tilted 40 deg

    def _empty(self):
        vertical = self.AXIS * math.cos(math.radians(self.PITCH))
        return _scene(lambda x, y: 0 * x, None, self.PITCH, vertical)[0]

    def test_tilted_realsense_floor_is_accepted_as_an_optical_axis_distance(self) -> None:
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), _profile(camera_to_empty_floor_m=1.10,
                                                                  distance_kind="unknown",
                                                                  tilt_from_vertical_deg=None))
            result = est.recalibrate(self._empty(), KW, None, automatic=True)
            self.assertTrue(result["ok"], result)
            self.assertEqual(est.profile.distance_kind, "optical_axis")
            self.assertAlmostEqual(est.profile.tilt_from_vertical_deg, self.PITCH, delta=2.0)

    def test_logitech_scale_targets_the_perpendicular_distance(self) -> None:
        depth = self._empty() * 2.5                         # model depth in arbitrary units
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("logitech", Path(d), _profile(camera_id="logitech", camera_to_empty_floor_m=1.10,
                                                                 distance_kind="optical_axis",
                                                                 tilt_from_vertical_deg=self.PITCH))
            result = est.recalibrate(depth, KW, None)
            self.assertTrue(result["ok"], result)
            # Perpendicular floor distance = 110 cm x cos 40 = 84 cm, not 110 cm (+31% on every height).
            self.assertAlmostEqual(result["floor_cm"], 110.0 * math.cos(math.radians(self.PITCH)), delta=3.0)

    def test_phase_files_are_separate(self) -> None:
        import os
        old = os.environ.get("LOCALLIFE_SETUP_PHASE")
        try:
            os.environ["LOCALLIFE_SETUP_PHASE"] = "Phase 2"
            self.assertEqual(bf.phase_suffix(), "_phase2")
            os.environ["LOCALLIFE_SETUP_PHASE"] = ""
            self.assertEqual(bf.phase_suffix(), "")
        finally:
            if old is None:
                os.environ.pop("LOCALLIFE_SETUP_PHASE", None)
            else:
                os.environ["LOCALLIFE_SETUP_PHASE"] = old


class SessionGuardTests(unittest.TestCase):
    def test_line_of_sight_occlusion_is_not_a_deposit(self) -> None:
        with TemporaryDirectory() as d:
            counter, scene = sd.SessionDeposits(Path(d), clock=lambda: 0.0), RoomScene(seed=5)
            run(counter, scene, 0.0, 6.0)
            scene.add(1, (40, 5, 120, 40), height=2.8, shade=200, label="person")   # 3.4 m -> 0.6 m
            run(counter, scene, 6.0, 14.0)
            self.assertEqual(counter.count_for("realsense"), 0)
            self.assertTrue([r for r in counter.rejected if "occlusion" in (r.get("reason") or "")])

    def test_real_bag_still_counts_once(self) -> None:
        with TemporaryDirectory() as d:
            counter, scene = sd.SessionDeposits(Path(d), clock=lambda: 0.0), RoomScene(seed=6)
            run(counter, scene, 0.0, 6.0)
            scene.add(1, (60, 85, 100, 115), height=0.15, label="garbage bag")
            run(counter, scene, 6.0, 14.0)
            self.assertEqual(counter.count_for("realsense"), 1)

    def test_whole_bin_delta_is_guarded(self) -> None:
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            for i, delta in enumerate((-542.0, 440.0, -3.0, 5.0)):
                counter.events.append({"camera": "logitech", "delta_occupancy_l": None, "deposit_time": 100.0 * i})
                counter.attach_occupancy("logitech", SimpleNamespace(started_at=100.0 * i, finalized_at=100.0 * i,
                                                                     delta_occupancy_l=delta))
            stored = [e["delta_occupancy_l"] for e in counter.events]
            self.assertEqual(stored, [None, None, -3.0, 5.0])
            self.assertIn("rejected", counter.events[0]["delta_occupancy_note"])
            self.assertIn("not a bag volume", counter.events[2]["delta_occupancy_note"])

    def test_stale_session_is_not_resumed(self) -> None:
        with TemporaryDirectory() as d:
            now = [1000.0]
            first = sd.SessionDeposits(Path(d), clock=lambda: now[0])
            first.events.append({"camera": "realsense", "counted": True, "event_id": 1})
            first._save()
            now[0] += 60.0                                  # a restart: same session
            self.assertEqual(sd.SessionDeposits(Path(d), clock=lambda: now[0]).session_id, first.session_id)
            now[0] += 20 * 3600.0                           # next day: a new session
            later = sd.SessionDeposits(Path(d), clock=lambda: now[0])
            self.assertNotEqual(later.session_id, first.session_id)
            self.assertEqual(later.events, [])
            self.assertTrue(json.loads(first.state_path.read_text(encoding="utf-8")).get("saved_at"))


if __name__ == "__main__":
    unittest.main()
