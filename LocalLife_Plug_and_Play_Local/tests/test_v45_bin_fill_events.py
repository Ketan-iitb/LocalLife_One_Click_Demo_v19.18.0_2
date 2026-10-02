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
                self.assertIn("New bags this session", html)
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
            # Model depth reads the floor at 1.6 m (wrong scale): the auto floor fit rescales it to 110 cm.
            logi_depth = _render(_profile(camera_to_empty_floor_m=1.10), lambda x, y: np.where(x < 0, 0.5, 0.0))
            logi._update_fill(np.zeros((K.height, K.width, 3), np.uint8), None, None, logi_depth * (1.6 / 1.1),
                              K, None, 6.0)
            self.assertTrue(8.0 < logi.fill.reading["height_fill_pct"] < 30.0)   # part floor, part 50 cm (by floor area)
            self.assertEqual(logi.fill.reading["geometry"], "auto-detected bin floor")
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
            self.assertTrue(any("below the floor reference" in w for w in warned["warnings"]))
            self.assertEqual(warned["height_fill_pct"], 40.0)    # excluded, not clamped into the reading
            spike = _render(_profile(), lambda x, y: 0.90 + 0 * x)
            held = est.update(frame, spike, K, None, 0.0, 6.0)
            self.assertTrue(held["stale"])
            self.assertIn("inconsistent depth frame ignored", held["stale_reason"])
            self.assertEqual(held["height_fill_pct"], 40.0)


class FloorTests(unittest.TestCase):
    def test_nearly_empty_bin_reads_low_not_the_tallest_bag(self) -> None:
        # Live: one bag in an empty bin showed 44 %. The floor is found and fill is the mean surface.
        with TemporaryDirectory() as d:
            prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00, tilt_from_vertical_deg=None,
                            distance_kind="unknown")
            est = bf.FillEstimator("realsense", Path(d), prof)
            tilt = _profile(camera_to_empty_floor_m=1.10, tilt_from_vertical_deg=20.0)   # real tilt, unknown to it
            bag = lambda x, y: np.where((np.abs(x) < 0.12) & (np.abs(y) < 0.12), 0.30, 0.0)
            depth = _render(tilt, bag)
            r = est.update(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, 0.0, 1.0)
            self.assertEqual(r["geometry"], "auto-detected bin floor")
            self.assertLess(r["height_fill_pct"], 15.0)
            self.assertAlmostEqual(est.profile.tilt_from_vertical_deg, 20.0, delta=2.0)

    def test_reset_refits_the_floor_and_rejects_a_hidden_floor(self) -> None:
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), _profile(camera_to_empty_floor_m=1.10))
            full = _render(_profile(camera_to_empty_floor_m=1.10), lambda x, y: 0.60 + 0 * x)
            self.assertFalse(est.recalibrate(full, K, None)["ok"])           # 50 cm away: waste, not floor
            empty = _render(_profile(camera_to_empty_floor_m=1.10), lambda x, y: 0.0 * x)
            got = est.recalibrate(empty, K, None)
            self.assertTrue(got["ok"])
            self.assertAlmostEqual(got["floor_cm"], 110.0, delta=1.0)



class SurroundingsTests(unittest.TestCase):
    def test_empty_bin_reads_zero_and_fill_does_not_depend_on_detector_boxes(self) -> None:
        with TemporaryDirectory() as d:
            prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
            est = bf.FillEstimator("realsense", Path(d), prof)
            frame = np.zeros((K.height, K.width, 3), np.uint8)
            empty = _render(prof, lambda x, y: 0 * x)
            self.assertEqual(est.update(frame, empty, K, None, 0.0, 1.0)["height_fill_pct"], 0.0)   # valid 0 %
            bag = _render(prof, lambda x, y: np.where((np.abs(x) < 0.12) & (np.abs(y) < 0.12), 0.12, 0.0))
            fills = []
            for boxes in (np.zeros(bag.shape, bool), np.ones(bag.shape, bool)):
                est.fill_history.clear()
                fills.append(est.update(frame, bag, K, None, 0.0, 2.0, objects=boxes)["height_fill_pct"])
            self.assertGreater(fills[0], 0.5)
            self.assertEqual(fills[0], fills[1])          # same surface, same fill, whatever was boxed


class V47ExportTests(unittest.TestCase):
    def test_excel_export_and_occupancy_timeout(self) -> None:
        from locallife_cloud.bin_occupancy import BinOccupancyTracker, DepositObservation, OccupancyReading
        tracker = BinOccupancyTracker("realsense")
        reading = OccupancyReading(litres=100.0, valid_fraction=0.9)
        tracker.observe(DepositObservation(reading=reading, timestamp=0.0))
        tracker.observe(DepositObservation(reading=reading, tracked_objects=1, timestamp=1.0))   # arriving
        event = None
        for t in range(2, 40):                     # never settles (always moving)
            event = event or tracker.observe(DepositObservation(reading=reading, changed_fraction=0.5,
                                                                timestamp=float(t)))
        self.assertIsNotNone(event)                # bounded, not "waste has not settled" for ever
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed here")
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            got = client.get("/api/session-deposits.xlsx")
            self.assertEqual(got.status_code, 200)
            import io
            book = openpyxl.load_workbook(io.BytesIO(got.data))
            self.assertEqual(book.sheetnames, ["Summary", "Deposits"])



class V47VolumeTests(unittest.TestCase):
    def test_bag_on_a_pile_is_measured_from_the_pile_not_the_floor(self) -> None:
        with TemporaryDirectory() as d:
            prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
            est = bf.FillEstimator("realsense", Path(d), prof)
            est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)          # empty-bin floor first
            pile_and_bag = lambda x, y: np.where((np.abs(x) < 0.10) & (np.abs(y) < 0.075), 0.55,  # 20x15, 25 cm
                                                 0.30)                                           # on a 30 cm pile
            depth = _render(prof, pile_and_bag)
            est.update(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, 0.0, 1.0)
            u = lambda x, z: K.ppx + K.fx * x / z
            v = lambda y, z: K.ppy + K.fy * y / z
            z = 1.10 - 0.55
            box = (u(-0.10, z), v(-0.075, z), u(0.10, z), v(0.075, z))
            litres, tall = est.object_volume(box, None, 1.5)
            self.assertAlmostEqual(tall, 0.25, delta=0.03)                     # not 0.55 (pile + bag)
            top = depth < 0.6                                                  # the rendered bag top
            truth = float(np.sum((depth[top] / K.fx) * (depth[top] / K.fy))) * 0.25 * 1000
            self.assertLess(abs(litres - truth) / truth, 0.15, (litres, truth))

    def test_pile_top_never_replaces_a_known_deeper_floor(self) -> None:
        with TemporaryDirectory() as d:
            prof = _profile(camera_to_empty_floor_m=1.10)
            est = bf.FillEstimator("logitech", Path(d), prof)
            self.assertTrue(est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)["ok"])
            pile = _render(prof, lambda x, y: 0.40 + 0 * x)
            self.assertFalse(est.recalibrate(pile, K, None, automatic=True)["ok"])
            self.assertAlmostEqual(est.profile.floor_raw_distance, 1.10, delta=0.02)



class V47LogitechDriftTests(unittest.TestCase):
    def test_frame_to_frame_scale_drift_is_cancelled_and_a_bad_floor_self_heals(self) -> None:
        with TemporaryDirectory() as d:
            prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
            est = bf.FillEstimator("logitech", Path(d), prof)
            region = np.zeros((K.height, K.width), bool)
            region[15:105, 20:140] = True
            frame = np.zeros((K.height, K.width, 3), np.uint8)
            self.assertTrue(est.recalibrate(_render(prof, lambda x, y: 0 * x), K, region)["ok"])
            scene = _render(prof, lambda x, y: np.where(np.abs(x) < 0.12, 0.40, 0.0))
            first = est.update(frame, scene, K, region, 0.0, 1.0)["height_fill_pct"]
            est.fill_history.clear()
            drifted = est.update(frame, scene * 1.3, K, region, 0.0, 2.0)["height_fill_pct"]   # model rescaled
            self.assertGreater(first, 5.0)
            self.assertLess(abs(drifted - first), 1.5, (first, drifted))
            # A wrong stored floor that leaves no valid surface is dropped after 20 s and re-found.
            est.profile.floor_plane = [0.0, 0.0, 9.0]
            est.profile.outside_reference = None
            for t in (10.0, 25.0, 40.0):
                est.update(frame, scene, K, region, 0.0, t)
            self.assertNotEqual(est.profile.floor_plane, [0.0, 0.0, 9.0])



class V48NoiseTests(unittest.TestCase):
    def test_bag_volume_and_fill_survive_stereo_noise_and_dropouts(self) -> None:
        # Before V48, raw per-pixel normals under ~4 mm noise rejected most bag-top pixels (-60..70 %).
        rng = np.random.default_rng(1)
        prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
        scene = lambda x, y: np.where((np.abs(x) < 0.10) & (np.abs(y) < 0.075), 0.55, 0.30)
        depth = _render(prof, scene)
        top = depth < 0.6
        truth = float(np.sum((depth[top] / K.fx) * (depth[top] / K.fy))) * 0.25 * 1000
        z = 0.55
        box = (K.ppx - K.fx * 0.10 / z, K.ppy - K.fy * 0.075 / z, K.ppx + K.fx * 0.10 / z, K.ppy + K.fy * 0.075 / z)
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), prof)
            est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
            noisy = depth + rng.normal(0, 0.004, depth.shape)
            noisy[rng.random(depth.shape) < 0.03] = 0
            reading = est.update(np.zeros((K.height, K.width, 3), np.uint8), noisy, K, None, 0.0, 1.0)
            litres, _ = est.object_volume(box, None, 1.5)
        self.assertLess(abs(litres - truth) / truth, 0.10, (litres, truth))
        self.assertEqual(reading["status"], "ok")
        self.assertIn("valid_depth_pct", reading["diagnostics"])



class V48GapTests(unittest.TestCase):
    def test_narrow_gaps_between_bags_are_filled_but_open_floor_is_not(self) -> None:
        prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
        bags = {(i, j): (0.0 if i % 4 == 0 else 0.40) for i in range(16) for j in range(12)}   # 5 cm cracks
        self.assertGreater(bf.fill_reading(bags, 0.9, prof)["height_fill_pct"], 38.0)          # not 30 %
        half = {(i, j): (0.40 if i < 8 else 0.0) for i in range(16) for j in range(12)}         # 40 cm open floor
        self.assertLess(bf.fill_reading(half, 0.9, prof)["height_fill_pct"], 23.0)
        self.assertEqual(bf.fill_reading({(i, j): 0.0 for i in range(9) for j in range(9)}, 0.9, prof)
                         ["height_fill_pct"], 0.0)



class V48TokenAndNoiseTests(unittest.TestCase):
    def test_reset_with_api_token_enabled(self) -> None:
        # Live bug: on the cloud the API token is on; the panel's reset POST sent no token -> 401,
        # so the session never changed ("3:48:38 PM (resumed)") although the button "worked".
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d), api_token="secret-token")
            manager = DualCameraCoordinator(config)
            client = create_app(config, manager).test_client()
            before = manager.deposits.session_id
            self.assertEqual(client.post("/api/bin-fill/session/new").status_code, 401)
            page = client.get("/").get_data(as_text=True)
            self.assertIn("bfHeaders()", page)                                 # the panel sends the token
            ok = client.post("/api/bin-fill/session/new", headers={"X-API-Token": "secret-token"})
            self.assertEqual(ok.status_code, 200)
            self.assertNotEqual(ok.get_json()["session"]["session_id"], before)
            self.assertEqual(ok.get_json()["session"]["new_bags_this_session"], 0)

    def test_heavy_stereo_noise_still_gives_a_fill_reading(self) -> None:
        rng = np.random.default_rng(2)
        prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), prof)
            est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
            scene = _render(prof, lambda x, y: 0.40 + 0 * x)
            import cv2
            ripple = cv2.GaussianBlur(rng.normal(0, 1, scene.shape), (0, 0), 2.0)
            noisy = scene + ripple / ripple.std() * 0.02     # 2 cm crumpled-plastic waviness: V48 before fix
                                                            # said "waiting for a clear view of the bin surface"
            r = est.update(np.zeros((K.height, K.width, 3), np.uint8), noisy, K, None, 0.0, 1.0)
            self.assertEqual(r["status"], "ok", r.get("reason"))
            self.assertAlmostEqual(r["height_fill_pct"], 40.0, delta=5.0)



class V49LogitechHoldTests(unittest.TestCase):
    def test_a_blinking_logitech_detection_is_held_but_a_removed_bag_is_not(self) -> None:
        from locallife_cloud.types import Detection
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            station = DualCameraCoordinator(config).camera("logitech")
            rng = np.random.default_rng(0)
            frame = rng.integers(30, 80, (240, 320, 3)).astype(np.uint8)
            frame[60:160, 100:220] = (200, 180, 230)                       # a pink bag
            bag = Detection("plastic waste bag", 0.6, (100, 60, 220, 160))
            bag.track_id = 3
            station._remember_tracks(frame, [bag])
            held = station._hold_through_dropout(frame, [])                 # detector blinked
            self.assertEqual(len(held), 1)
            self.assertEqual((held[0].source, held[0].box), ("held-through-dropout", (100, 60, 220, 160)))
            self.assertEqual(station._hold_through_dropout(frame, [bag]), [])   # detector back: nothing added
            gone = frame.copy()
            gone[60:160, 100:220] = rng.integers(30, 80, (100, 120, 3))        # bag taken out
            self.assertEqual(station._hold_through_dropout(gone, []), [])
            seen_at, item, patch = station._held_tracks[3]
            station._held_tracks[3] = (seen_at - 5.0, item, patch)             # missing for > 2 s
            self.assertEqual(station._hold_through_dropout(frame, []), [])



class V49ParcelTests(unittest.TestCase):
    def test_diagonal_parcel_gets_oriented_size_and_its_own_volume_only(self) -> None:
        # Live: a long parcel lying diagonally read 291 x 283 mm (square) -- the axis-aligned box
        # plus the pile inside it. Now: the largest risen region, oriented rectangle.
        import math
        prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
        a = math.radians(35)
        def parcel(x, y):
            u, v = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
            return (np.abs(u) < 0.15) & (np.abs(v) < 0.06)                   # 30 x 12 cm
        neighbour = lambda x, y: (x > 0.10) & (x < 0.16) & (y > 0.07) & (y < 0.13)   # in the box corner
        scene = lambda x, y: np.where(parcel(x, y), 0.37, np.where(neighbour(x, y), 0.38, 0.30))
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), prof)
            est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
            depth = _render(prof, scene)
            est.update(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, 0.0, 1.0)
            top = depth < (1.10 - 0.335)
            rows, cols = np.nonzero(np.abs(depth - (1.10 - 0.37)) < 0.005)
            box = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)   # axis-aligned, no mask
            litres, tall = est.object_volume(box, None, 1.5)
            dims = est.last_object
        self.assertAlmostEqual(tall, 0.07, delta=0.02)
        self.assertGreater(dims["length_m"] / dims["width_m"], 1.8, dims)           # long and thin, not square
        parcel_px = np.abs(depth - (1.10 - 0.37)) < 0.005
        truth = float(np.sum((depth[parcel_px] / K.fx) * (depth[parcel_px] / K.fy))) * 0.07 * 1000
        self.assertLess(abs(litres - truth) / truth, 0.25, (litres, truth))       # the neighbour is excluded


class V49SlabTests(unittest.TestCase):
    def test_tilted_box_volume_excludes_air_under_raised_end(self):
        # synthetic: a 30x20x6 cm box (3.6 L) tilted 30 deg on a 30 cm pile
        prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
        t = math.radians(30); half = 0.15 * math.cos(t)
        foot = lambda x, y: (np.abs(x) < half) & (np.abs(y) < 0.10)
        scene = lambda x, y: np.where(foot(x, y), 0.30 + 0.06 / math.cos(t) + (x + half) * math.tan(t), 0.30)
        with TemporaryDirectory() as d:
            est = bf.FillEstimator("realsense", Path(d), prof)
            est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
            depth = _render(prof, scene)
            est.update(np.zeros((K.height, K.width, 3), np.uint8), depth, K, None, 0.0, 1.0)
            rows, cols = np.nonzero(depth < 1.10 - 0.33)
            litres, tall = est.object_volume((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1), None, 1.5)
        self.assertLess(abs(litres - 3.6) / 3.6, 0.25, litres)     # was 4.86 L (+35 %)
        self.assertLess(tall, 0.10, tall)                           # thickness, not the 19 cm raised corner
        self.assertIn("slab", est.last_object["method"])


if __name__ == "__main__":
    unittest.main()
