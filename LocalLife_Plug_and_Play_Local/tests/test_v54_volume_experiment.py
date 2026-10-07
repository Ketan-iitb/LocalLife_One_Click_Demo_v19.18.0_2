"""V54 volume experiment: per-pixel geometry, baseline, calibration split, recording, replay, evaluation.

Scenes are ray-traced independently of the estimator (test_v49_box_fix._scene / _obb: exact ray-box and
ray-heightfield hits); expected volumes come from the scene definition (box L x W x H, or a numerical
integral of the analytic heightfield). Synthetic: this verifies geometry and bookkeeping, not the
physical accuracy of either camera.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud import volume_experiment as ve  # noqa: E402
from test_v49_box_fix import KW, _obb, _scene  # noqa: E402

FLAT = staticmethod(lambda x, y: 0 * x)


@lru_cache(maxsize=None)
def render(dims=None, pitch=45.0, cam=0.8, yaw=25.0, tilt=0.0, dome=False):
    heights = (lambda x, y: np.maximum(0.0, 0.12 * (1 - (x / 0.15) ** 2 - (y / 0.11) ** 2))) if dome \
        else (lambda x, y: 0 * x)
    box = None if dims is None else _obb(*dims, tilt, yaw, 0.0)
    depth, on_box = _scene(heights, box, pitch, cam=cam, far=cam * 3)
    empty, _ = _scene(lambda x, y: 0 * x, None, pitch, cam=cam, far=cam * 3)
    mask = on_box if not dome else depth < empty - 0.005
    return depth, mask, empty


def noisy(depth, sigma=0.003, seed=1):
    return depth + np.random.default_rng(seed).normal(0, sigma, depth.shape).astype(np.float32)


class PerPixelGeometryTests(unittest.TestCase):
    def assertVolume(self, result, truth, tol):
        self.assertEqual(result["status"], "ok", result["reasons"])
        self.assertLess(abs(result["volume_l"] - truth) / truth, tol, (result["volume_l"], truth))

    def test_empty_scene_is_not_a_zero_measurement(self):
        _, _, empty = render()
        result = ve.measure_frame(noisy(empty), KW, np.zeros(empty.shape, bool), noisy(empty, seed=2))
        self.assertIsNone(result["volume_l"])
        self.assertIn("no object mask", result["reasons"])
        leak = np.zeros(empty.shape, bool)
        leak[100:140, 140:180] = True                       # a mask on bare floor: nothing rises
        result = ve.measure_frame(noisy(empty), KW, leak, noisy(empty, seed=2))
        self.assertIsNone(result["volume_l"])

    def test_one_and_two_litre_cuboids_positions_orientations_and_views(self):
        cases = [((0.10, 0.10, 0.10), 45.0, 0.8, 25.0), ((0.10, 0.10, 0.10), 15.0, 0.9, 60.0),
                 ((0.20, 0.10, 0.10), 45.0, 0.8, 25.0), ((0.20, 0.10, 0.10), 15.0, 0.9, 0.0),
                 ((0.10, 0.10, 0.20), 45.0, 0.8, 40.0),          # the 2 L box standing on end
                 ((0.20, 0.05, 0.20), 30.0, 0.85, 10.0)]         # 2 L on its narrow side
        for dims, pitch, cam, yaw in cases:
            with self.subTest(dims=dims, pitch=pitch, yaw=yaw):
                depth, mask, empty = render(dims, pitch, cam, yaw)
                truth = dims[0] * dims[1] * dims[2] * 1000.0     # m^3 -> L
                self.assertVolume(ve.measure_frame(noisy(depth), KW, mask, noisy(empty, seed=2)), truth, 0.04)
                self.assertVolume(ve.measure_frame(depth, KW, mask, empty), truth, 0.025)   # no noise: no bias

    def test_oblique_view_does_not_double_count_projected_area(self):
        dims = (0.20, 0.10, 0.10)
        depth, mask, empty = render(dims, 60.0, 0.6, 30.0)
        self.assertVolume(ve.measure_frame(depth, KW, mask, empty), 2.0, 0.03)

    def test_domed_bag_against_its_analytic_integral(self):
        g = np.linspace(-0.2, 0.2, 2001)
        x, y = np.meshgrid(g, g)
        truth = float(np.maximum(0.0, 0.12 * (1 - (x / 0.15) ** 2 - (y / 0.11) ** 2)).sum() * (g[1] - g[0]) ** 2 * 1000)
        for pitch in (0.0, 15.0):
            depth, mask, empty = render(None, pitch, 0.8, dome=True)
            self.assertVolume(ve.measure_frame(noisy(depth), KW, mask, noisy(empty, seed=2)), truth, 0.03)
        for pitch in (30.0, 45.0):
            # the far slope behind the crest is out of sight: a documented LOWER bound, never an over-read
            depth, mask, empty = render(None, pitch, 0.8, dome=True)
            result = ve.measure_frame(noisy(depth), KW, mask, noisy(empty, seed=2))
            self.assertLess(result["volume_l_partial"], truth)
            self.assertGreater(result["occluding_boundary_fraction"], 0.0)

    def test_resolution_change_rescales_intrinsics(self):
        depth, mask, empty = render((0.20, 0.10, 0.10))
        full = ve.measure_frame(depth, KW, mask, empty)["volume_l"]
        half = ve.measure_frame(depth[::2, ::2], KW, mask, empty[::2, ::2])   # KW is for 320 x 240
        self.assertEqual(half["status"], "ok", half["reasons"])
        self.assertLess(abs(half["volume_l"] - full) / full, 0.03)
        wrong = dict(fx=KW.fx, fy=KW.fy, ppx=KW.ppx, ppy=KW.ppy, width=160, height=120)   # mislabelled
        self.assertGreater(abs(ve.measure_frame(depth, wrong, mask, empty)["volume_l_partial"] - full) / full, 0.5)

    def test_missing_depth_and_unseen_surface_make_the_frame_invalid(self):
        depth, mask, empty = render((0.20, 0.10, 0.10))
        holed = depth.copy()
        rows, cols = np.nonzero(mask)
        holed[rows.min():rows.max(), (cols.min() + cols.max()) // 2 - 10:(cols.min() + cols.max()) // 2 + 10] = np.nan
        result = ve.measure_frame(holed, KW, mask, empty)
        self.assertIsNone(result["volume_l"])
        self.assertIsNotNone(result["volume_l_partial"])          # kept for diagnosis only
        sparse = depth.copy()
        sparse[mask & (np.random.default_rng(0).random(depth.shape) < 0.6)] = np.nan
        self.assertIn("valid depth", " ".join(ve.measure_frame(sparse, KW, mask, empty)["reasons"]))

    def test_mask_leakage_onto_the_floor_is_not_volume(self):
        import cv2
        depth, mask, empty = render((0.10, 0.10, 0.10))
        leaky = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        tight = ve.measure_frame(depth, KW, mask, empty)
        loose = ve.measure_frame(depth, KW, leaky, empty)
        self.assertGreater(loose["leakage_pixels"], 0)
        self.assertLess(abs(loose["volume_l"] - tight["volume_l"]) / tight["volume_l"], 0.03)

    def test_baseline_shift_is_reported_not_absorbed(self):
        depth, mask, empty = render((0.10, 0.10, 0.10))
        result = ve.measure_frame(depth, KW, mask, empty * 1.03)      # camera moved / depth drift
        self.assertIsNone(result["volume_l"])
        self.assertTrue(any("background differs" in r for r in result["reasons"]))

    def test_overlapping_objects_have_aggregate_but_no_individual_volume(self):
        depth, mask, empty = render((0.20, 0.10, 0.10))
        rows, cols = np.nonzero(mask)
        mid = (cols.min() + cols.max()) // 2
        left, right = mask.copy(), mask.copy()
        left[:, mid:] = False
        right[:, :mid] = False
        result = ve.measure_objects(depth, KW, [left, right], empty)
        self.assertLess(abs(result["volume_l"] - 2.0) / 2.0, 0.03)
        self.assertTrue(all(i["volume_l"] is None for i in result["individual"]))


class BaselineAndCalibrationTests(unittest.TestCase):
    def test_baseline_quality_gates(self):
        _, _, empty = render()
        rec, median = ve.build_baseline([noisy(empty, seed=s) for s in range(6)], KW, None, camera="realsense", meta={})
        self.assertEqual(rec["status"], "ok", rec["reasons"])
        self.assertIsNotNone(median)
        rec, median = ve.build_baseline([noisy(empty, seed=s) for s in range(6)], KW, None, camera="realsense",
                                        meta={}, objects_present=1)
        self.assertEqual(rec["status"], "rejected")
        self.assertIsNone(median)
        rec, _ = ve.build_baseline([empty] * 3, KW, None, camera="realsense", meta={})
        self.assertEqual(rec["status"], "rejected")

    def test_compatibility_detects_mount_resolution_and_model_changes(self):
        ref = {"shape": [240, 320], "depth_model": "m1", "output_kind": "metric", "intrinsics_source": "x",
               "intrinsics": {"fx": 300, "fy": 300, "cx": 160, "cy": 120},
               "plane": {"normal": [0, -0.7071, -0.7071], "d": 0.8}}
        self.assertEqual(ve.compatibility(ref, dict(ref)), [])
        self.assertTrue(ve.compatibility(ref, {**ref, "shape": [480, 640]}))
        self.assertTrue(ve.compatibility(ref, {**ref, "depth_model": "m2"}))
        tilted = {"normal": [0, -0.6, -0.8], "d": 0.8}
        self.assertTrue(any("pose" in r for r in ve.compatibility(ref, {**ref, "plane": tilted})))

    def test_calibration_fits_on_calibration_objects_and_generalises_to_another(self):
        # A monocular-like depth with a wrong uniform scale (0.7 x true): fit on the 1 L box, test on 2 L.
        cal = []
        for yaw in (10.0, 50.0):
            depth, mask, empty = render((0.10, 0.10, 0.10), yaw=yaw)
            raw = ve.measure_frame(depth * 0.7, KW, mask, empty * 0.7)["volume_l"]
            cal.append({"trial_id": f"c{yaw}", "object_id": "box1L", "reference_l": 1.0, "raw_volume_l": raw})
        fit = ve.fit_calibration(cal, camera="logitech", align_background=False, meta={})
        self.assertAlmostEqual(fit["depth_scale"], 1 / 0.7, delta=0.02)
        self.assertEqual(fit["calibration_objects"], ["box1L"])
        depth, mask, empty = render((0.20, 0.10, 0.10))
        test = ve.measure_frame(depth * 0.7, KW, mask, empty * 0.7, scale=fit["depth_scale"])
        self.assertLess(abs(test["volume_l"] - 2.0) / 2.0, 0.04)
        with self.assertRaises(ValueError):
            ve.fit_calibration(cal[:1], camera="logitech", align_background=False, meta={})


def _payload(camera, t, depth, mask, *, rgb=None, model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
    rgb = np.zeros((*depth.shape, 3), np.uint8) if rgb is None else rgb
    meta = {"depth_model": model if camera == "logitech" else "rs", "output_kind": "metric",
            "intrinsics_source": "test", "depth_units": "m"}
    return ve.FramePayload(camera, t, rgb, depth.astype(np.float32), ve.intrinsics_dict(KW),
                           [mask] if mask is not None and mask.any() else [], [7] if mask is not None else [],
                           ["box"], None, meta)


class RecorderReplayEvaluationTests(unittest.TestCase):
    def _session(self, d, *, logitech_scale=0.7):
        rec = ve.ExperimentRecorder(Path(d), frames_per_trial=4, settle_s=0.5, timeout_s=1e9, baseline_frames=5)
        rec.add_object("box1L", reference_volume_l=1.0, reference_method="caliper external 10 x 10 x 10 cm",
                       reference_quantity="external_geometric", reference_status="measured", reference_uncertainty_l=0.02)
        rec.add_object("box2L", reference_volume_l=2.0, reference_method="caliper external 20 x 10 x 10 cm",
                       reference_quantity="external_geometric", reference_status="measured")
        rec.start_session("test")
        _, _, empty = render()
        rec.capture_baseline()
        for i in range(5):
            rec.observe(_payload("realsense", 10 + i, noisy(empty, seed=i), None))
            rec.observe(_payload("logitech", 10 + i, noisy(empty, seed=i) * logitech_scale, None))
        return rec

    def _run_trial(self, rec, dims, t0, designation, scale=0.7, frames=8, still=True, **kw):
        rec.start_trial(object_id="box1L" if dims[2] * dims[0] * dims[1] < 0.0015 else "box2L",
                        designation=designation, placement="centre", **kw)
        rec.trial["started_at"] = t0 - 0.01
        depth, mask, _ = render(dims)
        rec.observe(_payload("realsense", t0 - 5.0, noisy(depth, seed=99), mask))     # captured before the start
        for i in range(frames):
            rgb = np.zeros((*depth.shape, 3), np.uint8) if still else \
                np.random.default_rng(i).integers(0, 255, (*depth.shape, 3), dtype=np.uint8)
            rec.observe(_payload("realsense", t0 + 0.3 * i, noisy(depth, seed=100 + i), mask, rgb=rgb))
            rec.observe(_payload("logitech", t0 + 0.3 * i, noisy(depth, seed=200 + i) * scale, mask, rgb=rgb))
        return rec.stop_trial() or rec.last_trial

    def test_end_to_end_record_replay_calibrate_evaluate(self):
        with TemporaryDirectory() as d:
            rec = self._session(d)
            self.assertEqual(rec.baselines["realsense"]["status"], "ok")
            # a frame from before the trial started is never used
            trial = self._run_trial(rec, (0.10, 0.10, 0.10), 100.0, "calibration")
            rs, lg = trial["cameras"]["realsense"], trial["cameras"]["logitech"]
            self.assertEqual(rs["status"], "ok")
            self.assertLess(abs(rs["volume_l"] - 1.0), 0.05)
            self.assertIsNone(lg["volume_l"])                     # no Logitech calibration yet: explicit failure
            self.assertTrue(any("calibration" in r for f in lg["frames"] for r in (f.get("reasons") or [])))
            trial_dir = rec.session_dir / "trials" / trial["trial_id"]
            replay = ve.replay_trial(trial_dir)
            self.assertAlmostEqual(replay["cameras"]["realsense"]["volume_l"], rs["volume_l"], places=6)
            self.assertTrue(all(f.get("timestamp", 0) >= 100.0 - 0.01 for f in rs["frames"]))
            # fit the Logitech calibration on calibration trials only (as the CLI does), then freeze it
            trial2 = self._run_trial(rec, (0.10, 0.10, 0.10), 200.0, "calibration")
            samples = []
            for tr in (trial, trial2):
                r = ve.replay_trial(rec.session_dir / "trials" / tr["trial_id"],
                                    {"logitech": {"depth_scale": 1.0, "align_background": True, "calibration_id": "fit"}})
                samples.append({"trial_id": tr["trial_id"], "object_id": "box1L", "reference_l": 1.0,
                                "raw_volume_l": r["cameras"]["logitech"]["volume_l"]})
            meta = {k: rec.baselines["logitech"].get(k) for k in ("shape", "intrinsics", "plane", "depth_model",
                                                                  "output_kind", "intrinsics_source", "depth_units")}
            calibration = ve.fit_calibration(samples, camera="logitech", align_background=True, meta=meta)
            rec.calibration_path("logitech").parent.mkdir(parents=True, exist_ok=True)
            rec.calibration_path("logitech").write_text(json.dumps(calibration, default=float))
            with self.assertRaises(ValueError):                    # calibration object as held-out test
                rec.start_trial(object_id="box1L", designation="test")
            test = self._run_trial(rec, (0.20, 0.10, 0.10), 300.0, "test")
            self.assertEqual(test["cameras"]["logitech"]["status"], "ok")
            self.assertLess(abs(test["cameras"]["logitech"]["volume_l"] - 2.0) / 2.0, 0.05)
            # a trial whose object never settles is a timeout, not a settled measurement
            rec.timeout_s = 0.0
            moving = self._run_trial(rec, (0.20, 0.10, 0.10), 400.0, "test", still=False, frames=3)
            self.assertEqual(moving["cameras"]["realsense"]["motion_state"], "timeout_moving")
            self.assertIsNone(moving["cameras"]["realsense"]["volume_l"])
            trials = ve.load_trials([rec.session_dir])
            result = ve.evaluate(trials, calibration_objects={"logitech": {"box1L"}})
            held = [r for r in result["rows"] if r["split"] == "held_out" and r["camera"] == "logitech"
                    and r["motion_state"] == "settled"]
            self.assertEqual(held[0]["valid"], 1)
            self.assertEqual(held[0]["criteria"], "not agreed")
            self.assertGreaterEqual(result["paired"]["n_paired_valid"], 1)
            self.assertTrue((rec.session_dir / "trials.csv").exists())


class EvaluationMathTests(unittest.TestCase):
    def test_errors_are_per_trial_and_failures_count_against_availability(self):
        trials = [{"trial_id": f"t{i}", "object_id": "bag1", "designation": "test", "condition": "isolated",
                   "reference_volume_l": 4.0, "cameras": {"realsense": {"volume_l": v, "motion_state": "settled",
                                                                       "reasons": [] if v else ["coverage"]}}}
                  for i, v in enumerate([3.0, 5.0, 4.0, None])]
        row = ve.evaluate(trials, criteria={"max_mape_pct": 20.0, "min_availability": 0.8})["rows"][0]
        self.assertEqual((row["attempted"], row["valid"]), (4, 3))
        self.assertEqual(row["availability"], 0.75)
        self.assertAlmostEqual(row["mean_signed_error_l"], 0.0)          # bias cancels ...
        self.assertAlmostEqual(row["mae_l"], 2.0 / 3.0, places=5)          # ... absolute error does not
        self.assertAlmostEqual(row["rmse_l"], math.sqrt(2.0 / 3.0), places=5)
        self.assertAlmostEqual(row["mape_pct"], 100 * (0.25 + 0.25 + 0) / 3, places=3)
        self.assertEqual(row["failure_reasons"], {"coverage": 1})
        self.assertEqual(row["criteria"], "fail")                         # availability 0.75 < 0.8
        self.assertEqual(ve.evaluate(trials)["rows"][0]["criteria"], "not agreed")


if __name__ == "__main__":
    unittest.main()


class ServerSmokeTests(unittest.TestCase):
    def test_experiment_endpoints_drive_the_recorder_from_live_frames(self):
        from locallife_cloud.comparison import DualCameraCoordinator
        from locallife_cloud.config import AppConfig
        from locallife_cloud.server import create_app
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            manager = DualCameraCoordinator(config)
            manager.experiment.baseline_frames = 5
            client = create_app(config, manager).test_client()
            self.assertIn("session", client.get("/api/experiment/status").get_json())
            self.assertEqual(client.post("/api/experiment/trial-start", json={"object_id": "x", "designation": "test"})
                             .status_code, 400)                     # no session yet: refused, not crashed
            sid = client.post("/api/experiment/session", json={"note": "smoke"}).get_json()["session_id"]
            self.assertTrue((Path(d) / "experiment" / "sessions" / sid / "session.json").exists())
            client.post("/api/experiment/baseline", json={"cameras": ["realsense"]})
            # real pipeline frames reach the recorder through the pipeline hook
            _, _, empty = render()
            rs = manager.pipelines["realsense"]
            for i in range(6):
                rs.process_frame(np.zeros((*empty.shape, 3), np.uint8), depth_m=noisy(empty, seed=i), intrinsics=KW,
                                 timestamp=1000.0 + i, persist=False)
            status = client.get("/api/experiment/status").get_json()
            self.assertIn("realsense", status["baselines"])
            self.assertEqual(status["baselines"]["realsense"]["frames"], 5)
