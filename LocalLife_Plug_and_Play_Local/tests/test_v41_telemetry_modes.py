"""v41: local/cloud transport telemetry, mode attribution and the replay benchmark."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.edge_client import EdgeStreamState
from locallife_cloud.launcher_service import LaunchController, cost_estimate
from locallife_cloud.server import create_app
from locallife_cloud.telemetry import TelemetryRecorder

_spec = importlib.util.spec_from_file_location(
    "benchmark_local_cloud", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_local_cloud.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _recorder() -> tuple[TelemetryRecorder, FakeClock]:
    clock = FakeClock()
    return TelemetryRecorder("local", clock=clock, wall=lambda: 1.7e9), clock


class TelemetryRecorderTests(unittest.TestCase):
    def _complete(self, rec, clock, frame_id, seq, *, queue=0.1, work=0.2, camera="realsense"):
        key = rec.received(frame_id, camera, run_id="r1", seq=seq, bytes_in=1000)
        clock.now += queue
        rec.started(key)
        clock.now += work
        rec.completed(key, bytes_out=200)
        return key

    def test_timing_boundaries_are_measured_on_one_clock(self) -> None:
        rec, clock = _recorder()
        self._complete(rec, clock, "f1", 1, queue=0.2, work=0.3)
        row = rec.rows()[0]
        self.assertAlmostEqual(row["queue_ms"], 200.0, places=3)
        self.assertAlmostEqual(row["processing_ms"], 300.0, places=3)
        self.assertAlmostEqual(row["server_latency_ms"], 500.0, places=3)
        self.assertIn("server receipt", rec.metrics()["latency"]["boundary"])
        self.assertIn("excludes inference", rec.metrics()["client_upload_rtt"]["boundary"])

    def test_duplicate_frame_ids_are_not_counted_twice(self) -> None:
        rec, clock = _recorder()
        for seq in range(1, 7):
            self._complete(rec, clock, f"f{seq}", seq)
            clock.now += 2.0
        self.assertIsNone(rec.received("f3", "realsense", run_id="r1", seq=3))
        m = rec.metrics()
        self.assertEqual(m["throughput"]["unique_completed"], 6)
        self.assertEqual(m["reliability"]["received"], 6)
        self.assertEqual(m["reliability"]["duplicates_ignored"], 1)
        # Same id from another camera is a different frame.
        self.assertIsNotNone(rec.received("f3", "logitech", run_id="r1", seq=3))

    def test_missing_results_timeouts_and_denominators(self) -> None:
        rec, clock = _recorder()
        for seq in (1, 2, 3, 5, 6, 7, 9):     # 4 and 8 never arrived
            self._complete(rec, clock, f"f{seq}", seq)
            clock.now += 2.0
        rec.failed(rec.received("f10", "realsense", run_id="r1", seq=10), "timeout")
        rec.superseded(rec.received("f11", "realsense", run_id="r1", seq=11))
        rec.received("f12", "realsense", run_id="r1", seq=12)   # still in flight
        r = rec.metrics()["reliability"]
        self.assertEqual(r["lost_in_transit"], 2)
        self.assertEqual(r["received"], 10)
        self.assertEqual(r["sent"], 12)
        self.assertEqual(r["completed"], 7)
        self.assertEqual((r["failed"], r["superseded_by_newer_frame"], r["in_flight"]), (1, 1, 1))
        self.assertAlmostEqual(r["completed_pct"], round(100 * 7 / 12, 2))

    def test_short_window_reports_insufficient_data_not_zero(self) -> None:
        rec, clock = _recorder()
        self._complete(rec, clock, "f1", 1)
        m = rec.metrics()
        self.assertEqual(m["status"], "insufficient data")
        self.assertIsNone(m["throughput"]["fps"])
        self.assertIsNone(m["reliability"]["completed_pct"])
        empty = TelemetryRecorder("cloud").metrics()
        self.assertIsNone(empty["latency"]["p50_ms"])
        self.assertEqual(empty["latency"]["n"], 0)

    def test_window_excludes_old_frames_and_cameras_are_separate(self) -> None:
        rec, clock = _recorder()
        self._complete(rec, clock, "old", 1)
        clock.now += 500
        for seq in range(2, 8):
            self._complete(rec, clock, f"r{seq}", seq)
            self._complete(rec, clock, f"l{seq}", seq, camera="logitech", work=0.9)
            clock.now += 2.0
        self.assertEqual(rec.metrics(60, "realsense")["latency"]["n"], 6)
        self.assertGreater(rec.metrics(60, "logitech")["latency"]["p50_ms"],
                           rec.metrics(60, "realsense")["latency"]["p50_ms"])

    def test_edge_state_numbers_frames_and_counts_reconnects(self) -> None:
        state = EdgeStreamState("run", "logitech")
        first = state.next_metadata()
        state.failed(timeout=True)
        second = state.next_metadata()
        self.assertTrue(second["client"]["reconnected"])
        state.succeeded(12.5)
        third = state.next_metadata()
        self.assertEqual([first["seq"], second["seq"], third["seq"]], [1, 2, 3])
        self.assertEqual(len({first["frame_id"], second["frame_id"], third["frame_id"]}), 3)
        self.assertEqual((third["client"]["timeouts"], third["client"]["reconnects"]), (1, 1))
        self.assertEqual(third["client"]["prev_upload_rtt_ms"], 12.5)


class ServerAttributionTests(unittest.TestCase):
    def _app(self, directory: str, cloud: bool):
        config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                           enable_bucket_sync=False, results_dir=Path(directory),
                           automatic_baseline_frames=2, cloud_enabled=cloud)
        coordinator = DualCameraCoordinator(config)
        coordinator.warmup()
        return create_app(config, coordinator).test_client()

    def _post(self, client, frame_id, sync=True):
        ok, jpg = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))
        meta = {"source": "realsense-aligned-rgb-depth", "camera_id": "realsense", "frame_id": frame_id,
                "run_id": "t", "seq": int(frame_id[-1]),
                "intrinsics": {"fx": 200, "fy": 200, "ppx": 80, "ppy": 60, "width": 160, "height": 120}}
        return client.post("/api/cameras/realsense/ingest" + ("?sync=1" if sync else ""),
                           data={"metadata": json.dumps(meta), "image": (__import__("io").BytesIO(jpg.tobytes()), "f.jpg")},
                           content_type="multipart/form-data")

    def test_every_result_carries_the_servers_actual_mode(self) -> None:
        for cloud, expected in ((False, "local"), (True, "cloud")):
            with TemporaryDirectory() as directory:
                client = self._app(directory, cloud)
                response = self._post(client, "a-1")
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(response.get_json()["processing_mode"], expected)
                self.assertEqual(response.get_json()["frame_id"], "a-1")
                duplicate = self._post(client, "a-1")
                self.assertTrue(duplicate.get_json()["duplicate"])
                summary = client.get("/api/telemetry").get_json()
                self.assertEqual(summary["processing_mode"], expected)
                self.assertEqual(summary["models"]["logitech_depth"], "OFF")   # depth disabled in this config
                self.assertEqual(summary["all"]["reliability"]["completed"], 1)
                self.assertEqual(summary["all"]["reliability"]["duplicates_ignored"], 1)
                csv_text = client.get("/api/telemetry.csv").get_data(as_text=True)
                self.assertIn("a-1", csv_text)
                self.assertTrue(list((Path(directory) / "telemetry").glob(f"telemetry_{expected}_*.csv")))

    def test_unavailable_resources_are_none_not_zero(self) -> None:
        with TemporaryDirectory() as directory:
            summary = self._app(directory, False).get("/api/telemetry").get_json()
            gpu = summary["gpu"]
            if not gpu["available"]:
                self.assertIsNone(gpu["utilization_percent"])
                self.assertIsNone(gpu["memory_used_mb"])


class LauncherMetricsTests(unittest.TestCase):
    def test_snapshot_is_filed_under_the_backend_reported_mode(self) -> None:
        with TemporaryDirectory() as directory:
            control = LaunchController(AppConfig(results_dir=Path(directory)))
            control.state.mode = "cloud"   # launched cloud, but the backend is local
            payload = control.metrics(60, fetch=lambda port, w: {"processing_mode": "local", "all": {}})
            self.assertTrue(payload["mode_mismatch"])
            self.assertIsNotNone(payload["modes"]["local"])
            self.assertIsNone(payload["modes"]["cloud"])
            # Kept across restarts of the control service.
            again = LaunchController(AppConfig(results_dir=Path(directory)))
            self.assertIn("local", again.metric_snapshots)

    def test_unreachable_backend_and_cost_without_rate(self) -> None:
        with TemporaryDirectory() as directory:
            control = LaunchController(AppConfig(results_dir=Path(directory)))

            def down(port, window):
                raise OSError("connection refused")
            payload = control.metrics(60, fetch=down)
            self.assertFalse(payload["backend"]["reachable"])
            self.assertIsNone(payload["cost"]["estimate"])
            self.assertIn("Estimate only", cost_estimate(0.0, 3600.0)["note"])

    def test_vm_status_parses_zone_from_gpu_py(self) -> None:
        with TemporaryDirectory() as directory:
            def runner(command, timeout):
                self.assertEqual(command[-1], "status")
                return subprocess.CompletedProcess(command, 0, "depth-l4: RUNNING in europe-west4-b  (g2)\n", "")
            control = LaunchController(AppConfig(results_dir=Path(directory)), runner=runner)
            control.gpu_script = lambda: Path("gpu.py")
            result = control.vm_status()
            self.assertEqual((result["vm_state"], result["zone"]), ("RUNNING", "europe-west4-b"))


class ReplayQualityTests(unittest.TestCase):
    def _row(self, stem, **values):
        row = {"frame_file": stem, "warmup": False, "status": "ok", "detections": 1, "camera_id": "realsense",
               "e2e_ms": 100.0, "server_inference_ms": 50.0, "bytes_in": 10, "bytes_out": 5,
               "pred_length_mm": 100, "pred_width_mm": 50, "pred_height_mm": 20, "pred_volume_l": 0.1,
               "pred_colour": "red"}
        row.update(values)
        return row

    def test_quality_is_not_evaluated_without_ground_truth(self) -> None:
        self.assertEqual(bench.quality([self._row("a")], None)["status"], "Not evaluated")
        self.assertEqual(bench.quality([self._row("a")], {"zzz": {}})["status"], "Not evaluated")

    def test_quality_denominators_and_missing_predictions(self) -> None:
        truth = {"a": {"length_mm": "100", "width_mm": "50", "height_mm": "25", "volume_l": "0.125", "colour": "Red"},
                 "b": {"length_mm": "10", "width_mm": "10", "height_mm": "10", "volume_l": "1", "colour": "blue"}}
        rows = [self._row("a"), self._row("b", detections=0), self._row("c"), self._row("a", warmup=True)]
        q = bench.quality(rows, truth)
        self.assertEqual(q["trials_with_truth"], 2)
        self.assertEqual(q["detection_success"], "1/2")
        self.assertEqual(q["missing_predictions"], 1)
        self.assertEqual(q["colour_correct"], "1/1")
        self.assertEqual(q["dimension_values"], 3)
        self.assertAlmostEqual(q["volume_pct_error_median"], 20.0)

    def test_summary_excludes_warmup_and_counts_timeouts(self) -> None:
        rows = [self._row("w", warmup=True, e2e_ms=5000.0), self._row("a"),
                self._row("b", status="timeout", e2e_ms=None)]
        s = bench.summarise(rows)
        self.assertEqual((s["sent"], s["completed"], s["timeouts"]), (2, 1, 1))
        self.assertEqual(s["latency_e2e"]["p95_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
