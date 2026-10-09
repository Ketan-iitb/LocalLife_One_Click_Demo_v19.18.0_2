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

    def test_calibration_fits_heights_not_volumes_and_generalises_to_another_object(self):
        # A monocular-like metric depth with a wrong scale AND offset: raw = (Z - t) / s, s = 1.3, t = -0.08 m.
        s_true, t_true = 1.3, -0.08

        def raw(z):
            return ((z - t_true) / s_true).astype(np.float32)

        samples = []
        for dims, yaw in (((0.10, 0.10, 0.10), 10.0), ((0.10, 0.10, 0.20), 50.0)):   # tops 10 cm and 20 cm
            depth, mask, empty = render(dims, yaw=yaw)
            samples.append({"trial_id": f"c{yaw}", "object_id": f"cal{dims[2]}", "depth": raw(depth),
                            "baseline": raw(empty), "mask": mask, "roi": None, "intrinsics": KW,
                            "reference_top_height_m": dims[2]})
        # render(): camera 0.8 m above the floor -> the tape-measured PERPENDICULAR floor distance is 0.8 m
        fit = ve.fit_depth_mapping(samples, camera="logitech", output_kind="metric", floor_distance_m=0.8)
        self.assertTrue(fit["offset_identifiable"])
        self.assertAlmostEqual(fit["mapping"]["s"], s_true, delta=0.05)
        self.assertAlmostEqual(fit["mapping"]["t"], t_true, delta=0.03)
        self.assertLess(fit["fit_quality"]["rms_height_residual_m"], 0.004)
        self.assertNotIn("volume", json.dumps(fit["fitted_on"]))            # heights only: no volume enters
        depth, mask, empty = render((0.20, 0.10, 0.10))                       # held-out object
        test = ve.measure_frame(raw(depth), KW, mask, raw(empty), mapping=fit["mapping"])
        self.assertEqual(test["status"], "ok", test["reasons"])
        self.assertLess(abs(test["volume_l"] - 2.0) / 2.0, 0.05)
        # heights only (no floor distance): the offset is not identifiable -> scale only, and said so
        single = ve.fit_depth_mapping(samples, camera="logitech", output_kind="metric")
        self.assertFalse(single["offset_identifiable"])
        self.assertEqual(single["mapping"]["t"], 0.0)
        with self.assertRaises(ValueError):
            ve.fit_depth_mapping([{**samples[0], "reference_top_height_m": None}], camera="logitech",
                                 output_kind="metric")

    def test_measurement_outside_the_calibrated_range_is_not_reported(self):
        depth, mask, empty = render((0.10, 0.10, 0.10))
        mapping = {"kind": "scale", "scale": 1.0, "valid_raw_depth": [0.2, 0.5]}
        result = ve.measure_frame(depth, KW, mask, empty, mapping=mapping)
        self.assertIsNone(result["volume_l"])
        self.assertTrue(any("calibrated range" in r for r in result["reasons"]))


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
                        designation=designation, placement="centre", top_height_m=dims[2], **kw)
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
            # fit the Logitech calibration on calibration placements only (as the CLI does), then freeze it
            trial2 = self._run_trial(rec, (0.10, 0.10, 0.10), 200.0, "calibration")
            samples = []
            for tr in (trial, trial2):
                samples += ve.calibration_samples(rec.session_dir / "trials" / tr["trial_id"], "logitech")
            self.assertEqual(len(samples), 2)
            meta = {k: rec.baselines["logitech"].get(k) for k in ("shape", "intrinsics", "plane", "depth_model",
                                                                  "intrinsics_source", "depth_units")}
            calibration = ve.fit_depth_mapping(samples, camera="logitech", output_kind="metric", meta=meta)
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
                   "reference_volume_l": 4.0, "reference": {"reference_quantity": "displacement"},
                   "cameras": {"realsense": {"volume_l": v, "motion_state": "settled",
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
        # a printed liquid capacity (a 1.5 L milk carton) is not an external volume: not scored as accuracy
        printed = [{**t, "reference": {"reference_quantity": "printed_capacity"}} for t in trials]
        row = ve.evaluate(printed)["rows"][0]
        self.assertNotIn("mape_pct", row)
        self.assertIn("not a measured external volume", row["error_note"])
        self.assertEqual(row["availability"], 0.75)


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


class CameraModelAndValidityTests(unittest.TestCase):
    def test_measured_lens_distortion_is_removed_before_the_geometry(self):
        import cv2
        # Front-facing pinhole scene near the image CORNER (where barrel distortion matters): floor at
        # Z = 1.0 m, a box top at Z = 0.9 m over pixels rows 12:72, cols 12:112 -> footprint
        # (100 x 0.9/300) x (60 x 0.9/300) = 0.30 x 0.18 m, height 0.10 m: 5.4 L exactly.
        empty = np.full((240, 320), 1.0, np.float32)
        depth = empty.copy()
        depth[12:72, 12:112] = 0.9
        mask = np.zeros(depth.shape, bool)
        mask[12:72, 12:112] = True
        dist = [-0.25, 0.08, 0.0, 0.0, 0.0]                  # a webcam-like barrel distortion
        K = np.array([[KW.fx, 0, KW.ppx], [0, KW.fy, KW.ppy], [0, 0, 1]], np.float64)
        rows, cols = np.indices(depth.shape)
        # distorted image(u, v) = pinhole image at undistort(u, v)
        pts = cv2.undistortPoints(np.c_[cols.ravel(), rows.ravel()].astype(np.float64)[:, None, :], K,
                                  np.array(dist), P=K).reshape(depth.shape + (2,)).astype(np.float32)

        def distort(img):
            return cv2.remap(img, pts[..., 0], pts[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=float("nan") if img.dtype == np.float32 else 0)

        d_depth, d_empty, d_mask = distort(depth), distort(empty), distort(mask.astype(np.uint8)).astype(bool)
        ignored = ve.measure_frame(d_depth, KW, d_mask, d_empty)
        modelled = ve.measure_frame(d_depth, KW, d_mask, d_empty, distortion=dist)
        self.assertTrue(modelled["distortion_modelled"])
        self.assertLess(abs(modelled["volume_l_partial"] - 5.4) / 5.4, 0.03)
        self.assertGreater(abs(ignored["volume_l_partial"] - 5.4), 2 * abs(modelled["volume_l_partial"] - 5.4))

    def test_clipping_is_judged_on_the_original_mask(self):
        depth, mask, empty = render((0.20, 0.10, 0.10))
        clipped = mask.copy()
        rows, cols = np.nonzero(mask)
        clipped[rows.min():rows.max() + 1, 0] = True        # one-pixel contact: erosion would remove it
        result = ve.measure_frame(depth, KW, clipped, empty)
        self.assertIsNone(result["volume_l"])
        self.assertTrue(any("image edge" in r for r in result["reasons"]))

    def test_unstable_settled_window_is_not_a_measurement(self):
        frames = [{"phase": "settled", "volume_l": v, "reasons": []} for v in (1.0, 1.4, 0.8, 1.6, 1.0)]
        agg = ve.aggregate_trial(frames, "settled", settled=True)
        self.assertEqual(agg["status"], "unstable")
        self.assertIsNone(agg["volume_l"])
        steady = ve.aggregate_trial([{"phase": "settled", "volume_l": v} for v in (2.0, 2.02, 1.99)], "settled", True)
        self.assertEqual((steady["status"], steady["volume_l"]), ("ok", 2.0))

    def test_checkerboard_intrinsics_recover_a_known_camera(self):
        import cv2
        K = np.array([[610.0, 0, 322.0], [0, 612.0, 241.0], [0, 0, 1]])
        dist = np.array([-0.21, 0.05, 0.001, -0.001, 0.0])
        grid = np.zeros((54, 3), np.float32)
        grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
        obj, img = [], []
        rng = np.random.default_rng(0)
        for i in range(12):
            rvec = rng.normal(0, 0.35, 3)
            tvec = np.array([-0.1 + 0.02 * (i % 4), -0.06 + 0.03 * (i % 3), 0.45 + 0.03 * i])
            p, _ = cv2.projectPoints(grid, rvec, tvec, K, dist)
            obj.append(grid)
            img.append(p.reshape(-1, 2))
        got = ve.intrinsics_from_corners(obj, img, (640, 480), "synthetic")
        self.assertAlmostEqual(got["intrinsics"]["fx"], 610.0, delta=3.0)
        self.assertAlmostEqual(got["distortion"][0], -0.21, delta=0.03)
        self.assertLess(got["reprojection_rms_px"], 0.1)

    def test_relative_inverse_depth_mapping_through_the_same_geometry(self):
        a_true, b_true = 2.5, -0.4                             # d = (1/Z - b) / a  <=>  Z = 1 / (a d + b)
        def inv(z):
            return ((1.0 / z - b_true) / a_true).astype(np.float32)
        samples = []
        for dims, yaw in (((0.10, 0.10, 0.10), 10.0), ((0.10, 0.10, 0.20), 50.0)):
            depth, mask, empty = render(dims, yaw=yaw)
            samples.append({"trial_id": str(yaw), "object_id": "cal", "depth": inv(depth), "baseline": inv(empty),
                            "mask": mask, "roi": None, "intrinsics": KW, "reference_top_height_m": dims[2]})
        with self.assertRaises(ValueError):                    # relative output needs the floor distance
            ve.fit_depth_mapping(samples, camera="logitech", output_kind="relative-inverse")
        fit = ve.fit_depth_mapping(samples, camera="logitech", output_kind="relative-inverse", floor_distance_m=0.8)
        self.assertEqual(fit["mapping"]["kind"], "inverse_affine")
        depth, mask, empty = render((0.20, 0.10, 0.10))
        test = ve.measure_frame(inv(depth), KW, mask, inv(empty), mapping=fit["mapping"])
        self.assertLess(abs(test["volume_l_partial"] - 2.0) / 2.0, 0.06)


class PlacementRobustnessTests(unittest.TestCase):
    """Fixed oblique camera (45 deg from vertical, 0.8 m above the floor, 3 mm depth noise); the same rigid
    boxes at centre/near/far/left/right, upright and lying. Their volume is constant; the estimate's
    error and spread across placements are what is checked (not identical readings)."""

    def test_rigid_boxes_across_positions_and_orientations(self):
        offsets = {"centre": (0.0, 0.0), "near": (0.0, -0.12), "far": (0.0, 0.12),
                   "left": (-0.15, 0.0), "right": (0.15, 0.0)}
        poses = {"1L upright": (0.10, 0.10, 0.10), "2L lying": (0.20, 0.10, 0.10), "2L on end": (0.10, 0.10, 0.20)}
        empty = _scene(lambda x, y: 0 * x, None, 45.0, cam=0.8, far=2.4)[0]
        for pose, dims in poses.items():
            truth = dims[0] * dims[1] * dims[2] * 1000.0
            estimates = []
            for name, (dx, dy) in offsets.items():
                centre, axes, half = _obb(*dims, 0.0, 30.0, 0.0)
                depth, mask = _scene(lambda x, y: 0 * x, (centre + np.array([dx, dy, 0.0]), axes, half), 45.0,
                                     cam=0.8, far=2.4)
                r = ve.measure_frame(noisy(depth, seed=7), KW, mask, noisy(empty, seed=8))
                with self.subTest(pose=pose, placement=name):
                    self.assertEqual(r["status"], "ok", r["reasons"])
                    self.assertLess(abs(r["volume_l"] - truth) / truth, 0.04)
                    self.assertLess(r["low_skirt_area_fraction"], 0.10)
                    estimates.append(r["volume_l"])
            self.assertLess((max(estimates) - min(estimates)) / truth, 0.06, (pose, estimates))

    def test_smeared_monocular_depth_shows_as_a_low_skirt(self):
        import cv2
        depth, mask, empty = render((0.10, 0.10, 0.20))
        rows, cols = np.nonzero(mask)
        loose = np.zeros_like(mask)                              # a box-shaped detector mask
        loose[rows.min() - 12:rows.max() + 12, cols.min() - 12:cols.max() + 12] = True
        smeared = cv2.GaussianBlur(depth, (0, 0), 6)               # depth bled across the object's edges
        clean = ve.measure_frame(depth, KW, loose, empty)
        bad = ve.measure_frame(smeared, KW, loose, empty)
        self.assertLess(clean["low_skirt_area_fraction"], 0.05)
        self.assertGreater(bad["low_skirt_area_fraction"], 0.2)
        self.assertGreater(bad["volume_l_partial"], clean["volume_l_partial"])   # the footprint spreads


class LiveDiagnosticsTests(unittest.TestCase):
    def test_live_integrated_volume_is_not_inflated_by_the_per_cell_maximum(self):
        # A 9 x 7 x 22 cm carton-like box (1.386 L) through the LIVE path with 3 mm noise. The floor raster's
        # per-cell maximum read 1.548 / 1.542 L at 30 / 45 deg; the support-plane Jacobian integration reads 1.39 L.
        from pathlib import Path as _P
        from tempfile import TemporaryDirectory as _T

        from locallife_cloud import bin_fill as bf
        from test_v45_bin_fill_events import _profile
        for pitch in (30.0, 45.0):
            depth, mask = _scene(lambda x, y: 0 * x, _obb(0.09, 0.07, 0.22, 0, 20, 0.0), pitch, cam=0.8, far=2.4)
            empty, _ = _scene(lambda x, y: 0 * x, None, pitch, cam=0.8, far=2.4)
            with self.subTest(pitch=pitch), _T() as d:
                est = bf.FillEstimator("realsense", _P(d), _profile(camera_to_empty_floor_m=0.8, usable_height_m=0.45,
                                                                     tilt_from_vertical_deg=pitch))
                est.recalibrate(empty, KW, None)
                est.update(np.zeros((KW.height, KW.width, 3), np.uint8), noisy(depth), KW, None, 0.0, 1.0)
                r, c = np.nonzero(mask)
                litres, _ = est.object_volume((c.min(), r.min(), c.max() + 1, r.max() + 1), mask, 1.5,
                                              deformable_hint=True)
                self.assertLess(abs(litres - 1.386) / 1.386, 0.03)

    def test_live_support_reading_carries_its_trace(self):
        from test_v51_box_rotation import measure
        result, obj = measure((0.10, 0.10, 0.20), 45.0, 0.8)
        diag = obj["diagnostics"]
        for key in ("object_depth_m_p10_50_90", "support_depth_m_p10_50_90", "valid_coverage", "up_vector",
                    "perpendicular_height_m_p50_90", "footprint_hull_m2", "low_skirt_fraction"):
            self.assertIn(key, diag)
        self.assertAlmostEqual(diag["footprint_hull_m2"], 0.01, delta=0.003)   # a 10 x 10 cm base


class PartialMaskTests(unittest.TestCase):
    """The same rigid 9 x 9 x 20 cm carton-like box (1.62 L) under a fixed 45 deg camera, measured with the
    kind of partial masks a detector gives a printed carton. A truncated mask used to under-read by
    0-100 % depending on placement and pose (the reported 0.5-1.0 L spread)."""
    TRUTH = 0.09 * 0.09 * 0.20 * 1000.0

    @staticmethod
    def _cases():
        import cv2
        empty = _scene(lambda x, y: 0 * x, None, 45.0, cam=0.8, far=2.4)[0]
        for pose, dims in (("upright", (0.09, 0.09, 0.20)), ("lying", (0.20, 0.09, 0.09))):
            for place, dy in (("near", -0.15), ("far", 0.15)):
                c, a, h = _obb(*dims, 0, 15, 0.0)
                depth, mask = _scene(lambda x, y: 0 * x, (c + np.array([0, dy, 0]), a, h), 45.0, cam=0.8, far=2.4)
                depth = noisy(depth, seed=3)
                r, cc = np.nonzero(mask)
                box = (cc.min(), r.min(), cc.max() + 1, r.max() + 1)
                upper = mask.copy()
                upper[r.min() + int(0.6 * (r.max() - r.min())):] = False
                eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                for name, m in (("upper 60%", upper), ("eroded", eroded)):
                    yield f"{pose} {place} {name}", depth, m, box, empty

    def test_experiment_path_completes_partial_masks(self):
        for name, depth, mask, box, empty in self._cases():
            with self.subTest(case=name):
                r = ve.measure_frame(depth, KW, mask, noisy(empty, seed=4), box=box)
                self.assertEqual(r["status"], "ok", r["reasons"])
                self.assertLess(abs(r["volume_l"] - self.TRUTH) / self.TRUTH, 0.04)

    def test_live_rigid_path_completes_partial_masks(self):
        from pathlib import Path as _P
        from tempfile import TemporaryDirectory as _T

        from locallife_cloud import bin_fill as bf
        from test_v45_bin_fill_events import _profile
        for name, depth, mask, box, empty in self._cases():
            with self.subTest(case=name), _T() as d:
                est = bf.FillEstimator("realsense", _P(d), _profile(camera_to_empty_floor_m=0.8, usable_height_m=0.5,
                                                                     tilt_from_vertical_deg=45.0))
                est.recalibrate(empty, KW, None)
                est.update(np.zeros((KW.height, KW.width, 3), np.uint8), depth, KW, None, 0.0, 1.0)
                litres, _ = est.object_volume(box, mask, 1.5, rigid_hint=True)
                self.assertLess(abs(litres - self.TRUTH) / self.TRUTH, 0.10)
                self.assertTrue(est.last_object["diagnostics"]["mask_completion"]["completed"])

    def test_truncated_mask_that_cannot_be_completed_is_not_a_measurement(self):
        name, depth, mask, box, empty = next(iter(self._cases()))
        r = ve.measure_frame(depth, KW, mask, empty)                     # no detector box: zone = the mask
        self.assertIsNone(r["volume_l"])
        self.assertTrue(any("covers only part" in reason for reason in r["reasons"]))

    def test_completion_never_takes_a_neighbouring_detection(self):
        empty = _scene(lambda x, y: 0 * x, None, 45.0, cam=0.8, far=2.4)[0]
        c, a, h = _obb(0.20, 0.09, 0.09, 0, 0, 0.0)
        depth, mask = _scene(lambda x, y: 0 * x, (c, a, h), 45.0, cam=0.8, far=2.4)
        rows, cols = np.nonzero(mask)
        mid = (cols.min() + cols.max()) // 2
        left, right = mask.copy(), mask.copy()
        left[:, mid:] = False
        right[:, :mid] = False
        # "left" is this object; "right" is another detection touching it: excluded, so not grown into
        r = ve.measure_frame(depth, KW, left, empty, exclude=right, box=(cols.min(), rows.min(), mid, rows.max() + 1))
        half = 0.10 * 0.09 * 0.09 * 1000.0                                  # this object: half the 1.62 L box
        self.assertLess(abs(r["volume_l_partial"] - half) / half, 0.15)
