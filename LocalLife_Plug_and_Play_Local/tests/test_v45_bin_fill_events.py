"""v45: height-based bin fill, per-camera fill profiles and the session new-bag counter.

Synthetic scenes only: they check the logic, not real-world accuracy.
"""

from __future__ import annotations

import csv
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud.bin_occupancy import OccupancyEvent  # noqa: E402
from locallife_cloud.comparison import DualCameraCoordinator  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.server import create_app  # noqa: E402
from locallife_cloud.session_deposits import SessionDeposits  # noqa: E402
from locallife_cloud.types import CameraIntrinsics  # noqa: E402

K = CameraIntrinsics(fx=300.0, fy=300.0, ppx=80.0, ppy=60.0, width=160, height=120)


def _profile(**kw) -> bf.FillProfile:
    base = dict(camera_id="realsense", camera_to_empty_floor_m=1.40, distance_kind="vertical",
                tilt_from_vertical_deg=0.0, usable_height_m=1.00)
    base.update(kw)
    return bf.FillProfile(**base)


def _render(profile: bf.FillProfile, surface) -> np.ndarray:
    """Depth of each pixel ray hitting the horizontal surface h(x, y) (iterated, piecewise flat)."""
    b = math.radians(profile.tilt_from_vertical_deg)
    H = profile.vertical_height_m()
    rows, cols = np.mgrid[0:K.height, 0:K.width]
    xn, yn = (cols - K.ppx) / K.fx, (rows - K.ppy) / K.fy
    down = math.sin(b) * yn + math.cos(b)                  # -(up . ray) per unit depth
    z = H / down
    for _ in range(4):                                      # re-hit the surface under the current point
        x = xn * z
        hy = math.cos(b) * yn * z - math.sin(b) * z
        z = (H - surface(x, hy)) / down
    return z


class FillArithmeticTests(unittest.TestCase):
    def test_30cm_of_100cm_is_30pct_and_rough_198l(self) -> None:
        cells = {(i, j): 0.30 for i in range(10) for j in range(10)}
        reading = bf.fill_reading(cells, 0.9, _profile())
        self.assertEqual(reading["height_fill_pct"], 30.0)
        self.assertEqual(reading["max_fill_height_cm"], 30.0)
        self.assertEqual(reading["remaining_height_cm"], 70.0)
        self.assertEqual(reading["rough_litres"], 198.0)
        self.assertEqual(reading["rough_remaining_litres"], 462.0)
        self.assertIn("assumes roughly uniform filling", reading["rough_litres_label"])
        self.assertIn("unverified", reading["capacity_note"])
        self.assertIsNone(reading["occupied_l"])                       # interior not measured
        self.assertIn("not measured", reading["occupied_reason"])

    def test_occupied_volume_only_with_measured_interior_and_coverage(self) -> None:
        cells = {(i, j): 0.30 for i in range(20) for j in range(12)}   # 1.0 x 0.6 m seen
        full = bf.fill_reading(cells, 0.9, _profile(inner_length_m=1.0, inner_width_m=0.6))
        self.assertAlmostEqual(full["occupied_l"], 180.0)
        few = bf.fill_reading(dict(list(cells.items())[:50]), 0.9, _profile(inner_length_m=1.0, inner_width_m=0.6))
        self.assertIsNone(few["occupied_l"])
        self.assertIn("visible", few["occupied_reason"])

    def test_optical_axis_distance_uses_tilt_and_problems_are_reported(self) -> None:
        tilted = _profile(camera_to_empty_floor_m=1.10, distance_kind="optical_axis", tilt_from_vertical_deg=25.0,
                          usable_height_m=0.80)
        self.assertAlmostEqual(tilted.vertical_height_m(), 1.10 * math.cos(math.radians(25)))
        self.assertEqual(_profile(camera_to_empty_floor_m=1.10, usable_height_m=0.90).vertical_height_m(), 1.10)
        self.assertTrue(any("vertical or along" in p for p in bf.FillProfile("x", 1.1).problems()))
        self.assertTrue(any("inconsistent" in p for p in _profile(camera_above_rim_m=0.10).problems()))
        self.assertEqual(_profile(camera_above_rim_m=0.40).problems(), [])

    def test_heights_above_floor_from_a_tilted_camera(self) -> None:
        for tilt in (0.0, 20.0):
            profile = _profile(tilt_from_vertical_deg=tilt)
            surface = lambda x, y: np.where(np.abs(x) < 0.15, 0.30, 0.0)     # a 30 cm step
            depth = _render(profile, surface)
            h, hx, _ = bf.heights_above_floor(depth, K, profile)
            self.assertLess(abs(np.percentile(h[np.abs(hx) > 0.2], 50)), 0.01)
            self.assertLess(abs(np.percentile(h[np.abs(hx) < 0.1], 50) - 0.30), 0.01)


class EstimatorTests(unittest.TestCase):
    def test_na_reasons_settled_only_and_moved_camera(self) -> None:
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d))
            depth = _render(_profile(), lambda x, y: 0.30 + 0 * x)
            frame = np.zeros((K.height, K.width, 3), np.uint8)
            frame[10:30, 10:60] = 255                                        # a fixed edge outside the bin
            region = np.zeros(depth.shape, bool)
            region[40:110, 40:140] = True
            self.assertIn("incomplete", est.update(frame, depth, K, region, 0.0, 1.0)["reason"])
            est.save_profile(_profile(), frame, region)
            self.assertEqual(bf.FillEstimator("realsense", Path(d)).profile.usable_height_m, 1.00)   # persisted
            self.assertEqual(bf.FillEstimator("logitech", Path(d)).profile.usable_height_m, None)    # per camera
            self.assertIn("no aligned", est.update(frame, None, None, region, 0.0, 2.0,
                                                   depth_reason="no aligned depth frame")["reason"])
            ok = est.update(frame, depth, K, region, 0.0, 3.0)
            self.assertEqual((ok["status"], ok["height_fill_pct"], ok["updated_at"]), ("ok", 30.0, 3.0))
            moving = est.update(frame, depth * 0.5, K, region, 0.5, 4.0)
            self.assertEqual((moving["updated_at"], moving["stale"]), (3.0, True))      # not refreshed
            moved = np.zeros_like(frame)
            moved[80:100, 100:150] = 255
            self.assertIn("moved", est.update(moved, depth, K, region, 0.0, 5.0)["reason"])

    def test_new_bag_height_is_after_top_minus_before_surface_under_it(self) -> None:
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d))
            before = {(i, j): 0.20 for i in range(12) for j in range(12)}
            before[(0, 0)] = 0.70                                            # a tall old pile corner
            after = dict(before)
            for i in range(5, 8):
                for j in range(5, 8):
                    after[(i, j)] = 0.45                                     # bag lands on the 20 cm surface
            est.history.extend([(10.0, before), (20.0, after)])
            height, reason = est.added_height_m(12.0, 19.5)
            self.assertIsNone(reason)
            self.assertAlmostEqual(height, 0.25)              # not 0.45 (top) nor 0.70-0.20 (whole-bin max diff)
            est.history.clear()
            est.history.extend([(10.0, before), (20.0, before)])
            self.assertIsNone(est.added_height_m(12.0, 19.5)[0])            # hidden: no rise, N/A


def _event(n, camera="realsense", track=None, delta=15.0, start=100.0, envelope=(400.0, 300.0, 250.0), status="finalized"):
    return OccupancyEvent(event_id=f"{camera}-occ-{n}", camera=camera, started_at=start, finalized_at=start + 3,
                          occupied_before_l=100.0, occupied_after_l=None if delta is None else 100 + delta,
                          delta_occupancy_l=delta, absolute=True, status=status,
                          reason=None if delta is not None else "surface hidden", label="plastic bag",
                          color="black", track_id=track, envelope_mm=envelope)


class CounterTests(unittest.TestCase):
    def test_scenario_sequence(self) -> None:
        with TemporaryDirectory() as d:
            now = [0.0]
            s = SessionDeposits(Path(d), clock=lambda: now[0])
            s.height_lookup = lambda camera, a, b: (0.24, None) if camera == "realsense" else (None, "no profile")
            s.observe_tracks("realsense", [1, 2, 3], now=1.0)                  # bags already in the bin
            s.observe_tracks("logitech", [7, 8], now=1.0)
            self.assertEqual(s.count, 0)
            s.observe_tracks("realsense", [9], now=50.0)                     # after warm-up: not baseline
            self.assertNotIn(9, s.baseline["realsense"])

            self.assertEqual(s.record("realsense", _event(1, track=10, start=60))["count_after"], 1)
            self.assertEqual(s.count, 1)
            self.assertIsNone(s.record("realsense", _event(1, track=10, start=60)))          # same event again
            s.record("realsense", _event(2, track=11, start=120))
            self.assertEqual(s.count, 2)
            self.assertIsNone(s.record("realsense", _event(3, track=2, start=200)))          # old bag moves
            self.assertIsNone(s.record("realsense", _event(4, track=None, delta=0.4, start=260)))  # occlusion
            self.assertIsNone(s.record("realsense", _event(5, track=11, start=300)))         # re-tracked/dup
            self.assertEqual(s.count, 2)

            merged = s.record("logitech", _event(6, camera="logitech", track=40, start=361))
            first = s.record("realsense", _event(7, track=12, start=360))                    # both cams, one bag
            self.assertEqual(s.count, 3)
            self.assertIs(merged, first)
            self.assertEqual(sorted(first["cameras"]), ["logitech", "realsense"])
            self.assertEqual(first["height_source"], "after top - before surface under the bag")  # RealSense evidence
            self.assertEqual(first["envelope_l"], round(40 * 30 * 24 / 1000, 1))

            hidden = s.record("realsense", _event(8, track=13, delta=None, envelope=None, start=500,
                                                  status="relative_unavailable"))
            self.assertEqual(s.count, 4)
            self.assertEqual((hidden["envelope_l"], hidden["measurement_status"]), (None, "na"))
            self.assertTrue(hidden["reason"])
            self.assertEqual(hidden["material"], "UNKNOWN")

            snap = s.snapshot()
            self.assertEqual(snap["new_bags_this_session"], 4)
            self.assertEqual(snap["rejected_candidates"], 3)
            times = [e["deposit_time"] for e in snap["events"]]
            self.assertEqual(times, sorted(times))

            with s.csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(sum(r["counted"] == "True" for r in rows if "merged" not in (r["reason"] or "")), 4)
            self.assertTrue(any("baseline bag" in r["reason"] for r in rows))

    def test_warmup_event_is_not_a_new_bag(self) -> None:
        with TemporaryDirectory() as d:
            s = SessionDeposits(Path(d), clock=lambda: 0.0)
            self.assertIsNone(s.record("realsense", _event(1, track=5, start=2.0)))
            self.assertEqual(s.count, 0)


class ServerTests(unittest.TestCase):
    def test_endpoints_panel_and_pipeline_hooks(self) -> None:
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            manager = DualCameraCoordinator(config)
            client = create_app(config, manager).test_client()
            for page in ("/", "/research"):
                html = client.get(page).get_data(as_text=True)
                self.assertIn("Bin fill &amp; new deposits", html)
                self.assertIn("NEW bags this session", html)
            data = client.get("/api/bin-fill").get_json()
            self.assertEqual(data["deposits"]["new_bags_this_session"], 0)
            self.assertEqual(data["cameras"]["realsense"]["status"], "na")
            bad = client.post("/api/cameras/realsense/fill-profile", json={"camera_to_empty_floor_cm": "-3"})
            self.assertEqual(bad.status_code, 400)
            saved = client.post("/api/cameras/realsense/fill-profile", json={
                "camera_to_empty_floor_cm": 110, "distance_kind": "vertical", "tilt_from_vertical_deg": 0,
                "usable_height_cm": 80}).get_json()
            self.assertTrue(saved["ok"], saved)
            self.assertIsNone(manager.camera("logitech").fill.profile.usable_height_m)   # other camera untouched

            station = manager.camera("realsense")
            depth = _render(_profile(camera_to_empty_floor_m=1.10, usable_height_m=0.80), lambda x, y: 0.24 + 0 * x)
            station._update_fill(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, None, None, 5.0)
            self.assertEqual(station.fill.reading["height_fill_pct"], 30.0)
            logi = manager.camera("logitech")
            logi._update_fill(np.zeros((K.height, K.width, 3), np.uint8), None, None, depth, K, None, 5.0)
            self.assertEqual(logi.fill.reading["status"], "na")
            self.assertIn("profile incomplete", logi.fill.reading["reason"])

            manager.deposits.session_started_at -= 100
            station.deposit_listener("realsense", _event(1, track=3, start=manager.deposits.session_started_at + 50),
                                     {"material": None})
            self.assertEqual(client.get("/api/bin-fill").get_json()["deposits"]["new_bags_this_session"], 1)
            csv_text = client.get("/api/session-deposits.csv").get_data(as_text=True)
            self.assertIn("event_id", csv_text.splitlines()[0])
            self.assertEqual(len(csv_text.strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
