"""Local vs Cloud: Cost and Accuracy -- unit conversions, honest unknowns, pairing, export, endpoints.

All inputs below are test fixtures written in a temporary folder; none of them is a measurement.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from locallife_cloud import cost_accuracy as ca  # noqa: E402


def _cfg(cloud=None, local=None, **top):
    return ca.clean_config({"cloud": cloud or {}, "local": local or {}, **top})


def _write_run(root: Path, run_id: str, mode: str, frames: list[dict], input_id="rec-A", truth=None,
               detector="yoloe", depth="dav2"):
    folder = root / run_id
    folder.mkdir(parents=True)
    meta = {"run_id": run_id, "server_processing_mode": mode, "input_id": input_id, "detector_model": detector,
            "depth_model": depth, "truth_file": truth, "started_wall": 1.0}
    (folder / "run_summary.json").write_text(json.dumps({"metadata": meta}), encoding="utf-8")
    cols = ["run_id", "frame_file", "camera_id", "warmup", "status", "e2e_ms", "detections", "pred_volume_l"]
    with (folder / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for f in frames:
            writer.writerow({"run_id": run_id, "warmup": False, **f})


def _frames(camera, volumes, e2e_ms, status=None):
    out = []
    for i, v in enumerate(volumes):
        out.append({"frame_file": f"{camera}_{i}", "camera_id": camera, "status": (status or {}).get(i, "ok"),
                    "e2e_ms": e2e_ms, "detections": 0 if v is None else 1, "pred_volume_l": "" if v is None else v})
    return out


class CostTests(unittest.TestCase):
    def test_hours_seconds_and_per_thousand(self):
        cost = ca.cloud_cost(_cfg({"machine_rate_per_hour": 2.0, "gpu_included_in_machine_rate": True})["cloud"], 1800)
        self.assertAlmostEqual(cost["hours"], 0.5)
        self.assertAlmostEqual(cost["compute"], 1.0)
        self.assertAlmostEqual(ca.per_thousand(cost["total"], 900), 1.0 / 900 * 1000)
        self.assertIsNone(ca.per_thousand(1.0, 0))                     # zero frames: no figure
        self.assertIsNone(ca.per_thousand(1.0, None))

    def test_gpu_is_never_counted_twice(self):
        included = _cfg({"machine_rate_per_hour": 1.0, "gpu_rate_per_hour": 0.7, "gpu_included_in_machine_rate": True})
        self.assertEqual(ca.cloud_compute_rate(included["cloud"])[0], 1.0)
        separate = _cfg({"machine_rate_per_hour": 0.4, "gpu_rate_per_hour": 0.35, "gpu_count": 2,
                         "gpu_included_in_machine_rate": False})
        self.assertAlmostEqual(ca.cloud_compute_rate(separate["cloud"])[0], 1.1)
        unstated = _cfg({"machine_rate_per_hour": 0.4, "gpu_rate_per_hour": 0.35})
        self.assertIsNone(ca.cloud_compute_rate(unstated["cloud"])[0])   # ambiguous: refused, not guessed

    def test_disk_prorated_and_uptime_override(self):
        cloud = _cfg({"machine_rate_per_hour": 1.0, "gpu_included_in_machine_rate": True, "disk_gb": 200,
                      "disk_rate_per_gb_month": 0.1})["cloud"]
        cost = ca.cloud_cost(cloud, 3600)
        self.assertAlmostEqual(cost["storage"], 200 * 0.1 / 730.0)
        self.assertAlmostEqual(cost["total"], 1.0 + 200 * 0.1 / 730.0)
        cloud["billable_hours_override"] = 2.0                           # VM uptime > evaluated runtime
        self.assertAlmostEqual(ca.cloud_cost(cloud, 3600)["compute"], 2.0)
        cloud["billed_cost"] = 5.0
        billed = ca.cloud_cost(cloud, 3600)
        self.assertEqual((billed["total"], billed["total_kind"]), (5.0, "actual billed cost (entered)"))

    def test_local_energy_watts_to_kw_and_unknowns_stay_unknown(self):
        local = _cfg(local={"power_w": 65, "tariff_per_kwh": 0.25, "hardware_cost": 1000,
                            "hardware_lifetime_hours": 10000})["local"]
        cost = ca.local_cost(local, 7200)
        self.assertAlmostEqual(cost["energy"], 0.065 * 2 * 0.25)
        self.assertAlmostEqual(cost["hardware_allocated"], 1000 * 2 / 10000)
        self.assertAlmostEqual(cost["total"], cost["energy"])          # amortisation shown separately
        unknown = ca.local_cost(_cfg()["local"], 7200)
        self.assertIsNone(unknown["total"])                            # never "free"
        self.assertIsNone(ca.cloud_cost(_cfg()["cloud"], 7200)["total"])

    def test_config_is_whitelisted_and_status_needs_official_source_and_date(self):
        cfg = _cfg({"machine_rate_per_hour": "1.5", "api_key": "x", "status": "Official-rate estimate"}, secret=1)
        self.assertNotIn("secret", cfg)
        self.assertNotIn("api_key", cfg["cloud"])
        self.assertEqual(cfg["cloud"]["status"], "Unverified estimate")
        ok = _cfg({"source_url": "https://cloud.google.com/compute/gpus-pricing", "retrieved_on": "2026-10-02"})
        self.assertEqual(ok["cloud"]["status"], "Official-rate estimate")


class BillingPresetTests(unittest.TestCase):
    def test_billed_setups_count_gpu_once_and_are_labelled_billing_derived(self):
        expected = {"g2-standard-4 + 1x L4, on-demand (Netherlands)": 7.07,
                    "g2-standard-8 + 1x L4, on-demand (Belgium)": 9.07,
                    "g2-standard-4 + 1x L4, Spot (Netherlands)": 4.33,
                    "n1-standard-8 + 1x T4, on-demand (Netherlands)": 7.44}
        for name, per_hour in expected.items():
            cfg = ca.preset_config(name)
            self.assertEqual(cfg["cloud"]["status"], "Billing-derived rates")
            self.assertEqual(cfg["currency"], "SEK")
            self.assertAlmostEqual(ca.cloud_compute_rate(cfg["cloud"])[0], per_hour, places=2)
        cost = ca.cloud_cost(ca.preset_config("g2-standard-4 + 1x L4, on-demand (Netherlands)")["cloud"], 3600)
        self.assertAlmostEqual(cost["storage"], 200 * 1.08 / 730.0)          # disk pro rata for one hour
        self.assertAlmostEqual(sum(v for _, v in ca.BILLING_SUMMARY["breakdown"]), 534, delta=2)   # ~535 SEK bill

    def test_default_config_is_the_main_billed_setup_and_local_stays_unknown(self):
        with TemporaryDirectory() as d:
            cfg = ca.load_config(Path(d) / "missing.json")
        self.assertEqual(cfg["cloud"]["machine_type"], "g2-standard-4")
        self.assertIsNone(ca.local_cost(cfg["local"], 3600)["total"])


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.truth = self.root / "truth.csv"
        with self.truth.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "volume_l", "object_present"])
            for i, v in enumerate([4.0, 2.0, 0.0, 5.0]):
                writer.writerow([f"realsense_{i}", v, 1 if v > 0 else 0])

    def tearDown(self):
        self.tmp.cleanup()

    def _view(self, local_volumes, cloud_volumes, cfg=None, **kw):
        bench = self.root / "bench"
        _write_run(bench, "L1", "local", _frames("realsense", local_volumes, 400) + _frames("logitech", [9.9], 400),
                   truth=str(self.truth), **kw.get("local_kw", {}))
        _write_run(bench, "C1", "cloud", _frames("realsense", cloud_volumes, 100), truth=str(self.truth),
                   **kw.get("cloud_kw", {}))
        entries = ca.load_benchmark_runs([bench])
        return ca.build_view("realsense", entries, cfg or _cfg())

    def test_identical_predictions_give_identical_errors_in_both_modes(self):
        view = self._view([5.0, None, 0.5, 5.0], [5.0, None, 0.5, 5.0])
        local, cloud = view["rows"]
        self.assertEqual(view["accuracy_status"], "evaluated")
        for key in ("mae_l", "mape_pct", "valid_volume", "missing_volume", "eligible"):
            self.assertEqual(local[key], cloud[key], key)
        self.assertAlmostEqual(local["mae_l"], (1.0 + 0.5 + 0.0) / 3)    # missing frame not counted as 0 error
        self.assertEqual((local["valid_volume"], local["missing_volume"]), (3, 1))
        self.assertEqual(local["zero_reference_excluded"], 1)            # 0 L reference: no percentage
        self.assertAlmostEqual(local["mape_pct"], (25.0 + 0.0) / 2)
        self.assertAlmostEqual(local["recall"], 2 / 3)                   # frame 1 present, nothing detected
        self.assertAlmostEqual(local["precision"], 2 / 3)                # frame 2 empty, detected

    def test_unpaired_inputs_are_not_evaluated_but_speed_still_shows(self):
        view = self._view([4.0] * 4, [4.0] * 4, cloud_kw={"input_id": "rec-B"})
        self.assertTrue(view["accuracy_status"].startswith("Accuracy not evaluated"))
        self.assertIn("same recorded input", view["accuracy_status"])
        self.assertIsNone(view["rows"][0]["mae_l"])
        self.assertAlmostEqual(view["rows"][1]["fps"], 10.0)            # 100 ms per frame
        self.assertEqual(view["points"], [])                             # no cost entered: nothing plotted

    def test_no_reference_and_live_runs_are_named_reasons(self):
        bench = self.root / "b2"
        _write_run(bench, "L", "local", _frames("realsense", [1.0], 100))
        _write_run(bench, "C", "cloud", _frames("realsense", [1.0], 100))
        view = ca.build_view("realsense", ca.load_benchmark_runs([bench]), _cfg())
        self.assertIn("no reference (truth) CSV", view["accuracy_status"])
        live = ca.live_runs([{"camera_id": "realsense", "processing_mode": m, "run_id": m, "source_host": "h",
                              "completed_unique": 100, "frames_window_s": 50, "fps": 2.0} for m in ("local", "cloud")])
        view = ca.build_view("realsense", live, _cfg())
        self.assertIn("live runs carry no per-frame predictions", view["accuracy_status"])

    def test_cameras_stay_separate(self):
        view = self._view([4.0] * 4, [4.0] * 4)
        self.assertTrue(all(r["camera"] == "realsense" for r in view["rows"]))
        self.assertTrue(all(k.endswith(":realsense") for k in (view["selected"]["local"], view["selected"]["cloud"])))

    def test_table_graph_and_csv_agree(self):
        cfg = _cfg({"machine_rate_per_hour": 1.0, "gpu_included_in_machine_rate": True},
                   {"power_w": 100, "tariff_per_kwh": 0.3})
        view = self._view([5.0, 2.0, 0.0, 5.0], [4.0, 2.0, 0.0, 5.0], cfg)
        rows = {r["mode"]: r for r in view["rows"]}
        self.assertAlmostEqual(rows["cloud"]["run_cost"], 1.0 * 0.4 / 3600)       # 4 frames x 100 ms
        self.assertAlmostEqual(rows["cloud"]["cost_per_1000"], rows["cloud"]["run_cost"] / 4 * 1000)
        self.assertAlmostEqual(rows["local"]["run_cost"], 0.1 * (1.6 / 3600) * 0.3)
        points = {p["mode"]: p for p in view["points"]}
        exported = {r["mode"]: r for r in csv.DictReader(io.StringIO(ca.view_csv(view)))}
        for mode in ("local", "cloud"):
            self.assertAlmostEqual(points[mode]["x_cost_per_1000"], rows[mode]["cost_per_1000"])
            self.assertAlmostEqual(points[mode]["y_mae_l"], rows[mode]["mae_l"])
            self.assertAlmostEqual(float(exported[mode]["cost_per_1000"]), rows[mode]["cost_per_1000"])
            self.assertAlmostEqual(float(exported[mode]["mae_l"]), rows[mode]["mae_l"])
        self.assertEqual(exported["cloud"]["rate_status"], "Unverified estimate")


class EndpointTests(unittest.TestCase):
    def test_section_endpoints_and_existing_panels_coexist(self):
        from locallife_cloud.comparison import DualCameraCoordinator
        from locallife_cloud.config import AppConfig
        from locallife_cloud.server import create_app
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            for page in ("/", "/research"):
                html = client.get(page).get_data(as_text=True)
                self.assertIn("Local vs Cloud: Cost and Accuracy", html)
                self.assertIn("Bin fill &amp; new deposits", html)             # existing panels unchanged
            view = client.get("/api/cost-accuracy/view?camera=logitech").get_json()
            self.assertEqual(view["camera"], "logitech")
            self.assertTrue(view["accuracy_status"].startswith("Accuracy not evaluated"))
            saved = client.post("/api/cost-accuracy/config",
                                json={"cloud": {"machine_rate_per_hour": "0.9", "gpu_included_in_machine_rate": "true"}})
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(client.get("/api/cost-accuracy/config").get_json()["cloud"]["machine_rate_per_hour"], 0.9)
            text = client.get("/api/cost-accuracy.csv?camera=realsense").get_data(as_text=True)
            self.assertTrue(text.startswith("camera,mode,"))
            self.assertEqual(client.get("/api/local-cloud/view").status_code, 200)   # existing comparison API


if __name__ == "__main__":
    unittest.main()
