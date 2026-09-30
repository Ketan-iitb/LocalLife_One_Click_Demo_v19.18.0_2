"""v45: height-based bin fill, per-camera fill profiles and the session new-bag counter.

Synthetic scenes only: they check the logic, not real-world accuracy.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud import bin_fill as bf  # noqa: E402
from locallife_cloud.comparison import DualCameraCoordinator  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.server import create_app  # noqa: E402
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
            held = est.update(moved, depth, K, region, 0.0, 5.0)
            self.assertEqual((held["updated_at"], held["stale"]), (3.0, True))          # last valid kept, marked
            self.assertIn("moved", held["stale_reason"])

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
            self.assertEqual(manager.camera("logitech").fill.profile.camera_to_empty_floor_m, 1.10)  # shared default

            station = manager.camera("realsense")
            depth = _render(_profile(camera_to_empty_floor_m=1.10, usable_height_m=0.80), lambda x, y: 0.24 + 0 * x)
            station._update_fill(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, None, None, 5.0)
            self.assertEqual(station.fill.reading["height_fill_pct"], 30.0)
            logi = manager.camera("logitech")
            logi._update_fill(np.zeros((K.height, K.width, 3), np.uint8), None, None, None, K, None, 5.0)
            self.assertEqual(logi.fill.reading["status"], "na")
            self.assertIn("no Logitech depth-model output", logi.fill.reading["reason"])
            # Its own model depth + the shared 110 cm / 100 cm defaults: an independent, approximate reading.
            logi_depth = _render(_profile(camera_to_empty_floor_m=1.10), lambda x, y: 0.50 + 0 * x)
            logi._update_fill(np.zeros((K.height, K.width, 3), np.uint8), None, None, logi_depth, K, None, 6.0)
            self.assertEqual(logi.fill.reading["height_fill_pct"], 50.0)
            self.assertEqual(logi.fill.reading["rough_litres"], 330.0)
            self.assertIn("monocular model depth, approximate", logi.fill.reading["depth_source"])
            self.assertEqual(station.fill.reading["height_fill_pct"], 30.0)        # RealSense unaffected

            # Every processed frame reaches the counter, with no fill profile or zone needed.
            frame = np.full((K.height, K.width, 3), 90, np.uint8)
            region = np.ones((K.height, K.width), bool)
            logi._emit_deposit_evidence(frame, [], None, None, region)
            snap = client.get("/api/bin-fill").get_json()["deposits"]
            self.assertEqual(snap["cameras"]["logitech"]["state"], "initialising")
            self.assertIsNotNone(snap["updated_at"])
            session = snap["session_id"]
            self.assertEqual(client.get("/api/bin-fill").get_json()["deposits"]["session_id"], session)  # refresh
            fresh = client.post("/api/bin-fill/session/new").get_json()["session"]
            self.assertNotEqual(fresh["session_id"], session)
            self.assertEqual(client.get("/api/session-deposits.csv").status_code, 200)

    def test_defaults_are_provisional_and_never_overwrite_a_saved_profile(self) -> None:
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            default = bf.default_profile("realsense", config)
            self.assertEqual((default.camera_to_empty_floor_m, default.usable_height_m), (1.10, 1.00))
            self.assertEqual(default.status, "approximate")
            self.assertTrue(any("tilt not measured" in a for a in default.assumptions()))
            logi = bf.default_profile("logitech", config)          # shared installation defaults, no form needed
            self.assertEqual((logi.camera_to_empty_floor_m, logi.usable_height_m), (1.10, 1.00))
            self.assertTrue(any("approximate shared-installation assumption" in n for n in logi.notes))
            self.assertEqual(logi.blocking(), [])
            config.logitech_reference_distance_m = 1.12       # the Logitech's own measured distance wins
            self.assertEqual(bf.default_profile("logitech", config).camera_to_empty_floor_m, 1.12)
            self.assertEqual(bf.default_profile("realsense", config).camera_to_empty_floor_m, 1.10)
            config.logitech_reference_distance_m = 0.0

            store = Path(d) / "realsense" / "bin_profile"          # where the RealSense station keeps it
            est = bf.FillEstimator("realsense", store, default)
            self.assertEqual(est.profile.source, "default-provisional")
            self.assertTrue(est.path.exists())                        # defaults persisted once
            est.save_profile(_profile(camera_to_empty_floor_m=1.30, usable_height_m=1.05, tilt_from_vertical_deg=10),
                             None, None)
            again = bf.FillEstimator("realsense", store, default)
            self.assertEqual((again.profile.source, again.profile.camera_to_empty_floor_m), ("saved", 1.30))
            self.assertEqual(again.profile.status, "measured")

            client = create_app(config, DualCameraCoordinator(config)).test_client()
            got = client.post("/api/cameras/realsense/fill-profile", json={"usable_height_cm": 98}).get_json()
            self.assertEqual(got["status"], "measured")
            profile = client.get("/api/bin-fill").get_json()["cameras"]["realsense"]["profile"]
            self.assertEqual((profile["camera_to_empty_floor_m"], profile["usable_height_m"]), (1.30, 0.98))


class FillQualityTests(unittest.TestCase):
    def test_50cm_is_50pct_and_330l(self) -> None:
        cells = {(i, j): 0.50 for i in range(10) for j in range(10)}
        r = bf.fill_reading(cells, 0.9, _profile())
        self.assertEqual((r["height_fill_pct"], r["rough_litres"], r["rough_remaining_litres"]), (50.0, 330.0, 330.0))

    def test_below_floor_depth_warns_and_one_bad_frame_is_not_trusted(self) -> None:
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), _profile())
            frame = np.zeros((K.height, K.width, 3), np.uint8)
            good = _render(_profile(), lambda x, y: 0.40 + 0 * x)
            for t in range(4):
                self.assertEqual(est.update(frame, good, K, None, 0.0, float(t))["height_fill_pct"], 40.0)
            deep = good.copy()
            deep[:, :60] = 1.60                                  # 20 cm-plus below the assumed floor
            warned = est.update(frame, deep, K, None, 0.0, 5.0)
            self.assertTrue(any("below the assumed floor" in w for w in warned["warnings"]))
            self.assertEqual(warned["height_fill_pct"], 40.0)    # excluded, not clamped into the reading
            spike = _render(_profile(), lambda x, y: 0.90 + 0 * x)
            held = est.update(frame, spike, K, None, 0.0, 6.0)
            self.assertTrue(held["stale"])
            self.assertIn("inconsistent depth frame ignored", held["stale_reason"])
            self.assertEqual(held["height_fill_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
