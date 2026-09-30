"""v42: durable Local-vs-Cloud comparison on the localhost:8000 dashboard."""

from __future__ import annotations

import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from locallife_cloud import comparison_store as cs
from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.server import create_app
from locallife_cloud.telemetry import TelemetryRecorder


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _store(directory, mode="local", start=1000.0, wall=1.7e9):
    mono, wall_clock = Clock(start), Clock(wall)
    store = cs.ComparisonStore(Path(directory), processing_mode=mode, clock=mono, wall=wall_clock,
                               run_meta={"detector": "det", "depth_model": None})
    return store, mono, wall_clock


def _feed(store, mono, camera="realsense", n=6, gap=1.0, seqs=None, statuses=None):
    rec = TelemetryRecorder(store.processing_mode, clock=mono)
    rec.sink = store.record_frame
    for i in range(n):
        seq = seqs[i] if seqs else i + 1
        key = rec.received(f"f{i}", camera, run_id="edge1", seq=seq, bytes_in=1000,
                           client={"prev_upload_rtt_ms": 20.0 + i})
        status = (statuses or {}).get(i, "completed")
        if status == "superseded":
            rec.superseded(key)
        else:
            mono.now += 0.1
            rec.started(key)
            mono.now += 0.2
            if status == "failed":
                rec.failed(key, "timeout")
            else:
                rec.completed(key, bytes_out=500)
        mono.now += gap
    return rec


class StoreTests(unittest.TestCase):
    def test_restart_appends_and_reloads_history(self) -> None:
        with TemporaryDirectory() as d:
            first, mono, _ = _store(d)
            _feed(first, mono)
            lines_before = len((Path(d) / "observations.jsonl").read_text().splitlines())
            second, mono2, _ = _store(d, start=5000.0, wall=1.7e9 + 3600)
            _feed(second, mono2, n=3)
            lines_after = len((Path(d) / "observations.jsonl").read_text().splitlines())
            self.assertGreater(lines_after, lines_before)          # appended, not replaced
            runs = second.runs()
            self.assertEqual(len(runs), 2)
            old = [s for s in second.own_summaries() if s["run_id"] == first.run_id]
            self.assertEqual(old[0]["completed_unique"], 6)
            self.assertEqual([r["current"] for r in runs], [False, True])

    def test_duplicate_events_and_late_reply_size_are_counted_once(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            rec = TelemetryRecorder("local", clock=mono)
            rec.sink = store.record_frame
            key = rec.received("f1", "logitech", run_id="e", seq=1)
            rec.started(key)
            rec.completed(key)
            rec.response_bytes(key, 321)          # sync path: size known after completion
            self.assertFalse(store.append({"event": "frame", "camera_id": "logitech", "frame_id": "f1"}))
            frames = [r for r in store.raw_rows() if r["event"] == "frame"]
            self.assertEqual(len(frames), 1)

    def test_metrics_boundaries_denominators_and_unavailable_values(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            _feed(store, mono, n=6, seqs=[1, 2, 3, 5, 6, 7], statuses={1: "superseded", 2: "failed"})
            s = store.current_summaries()[0]
            self.assertEqual((s["received"], s["superseded"], s["failed"], s["lost"]), (6, 1, 1, 1))
            self.assertEqual(s["failure_denominator"], 7)
            self.assertAlmostEqual(s["failure_pct"], round(100 * 2 / 7, 2))
            self.assertAlmostEqual(s["superseded_pct"], round(100 / 6, 2))
            self.assertAlmostEqual(s["latency_p50_ms"], 300.0, places=3)
            self.assertAlmostEqual(s["queue_p50_ms"], 100.0, places=3)
            self.assertAlmostEqual(s["inference_p50_ms"], 200.0, places=3)
            self.assertEqual(s["latency_spread_ms"], round(s["latency_p95_ms"] - s["latency_p50_ms"], 3))
            # Nothing sampled -> N/A (None), never 0.
            for key in ("cpu_avg_pct", "gpu_util_avg_pct", "vram_peak_mb", "recovery_p50_s"):
                self.assertIsNone(s[key], key)
            self.assertIn("Not evaluated", s["quality"])

    def test_old_edge_client_without_sequence_numbers_states_its_denominator(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            rec = TelemetryRecorder("local", clock=mono)
            rec.sink = store.record_frame
            for i in range(3):
                k = rec.received(None, "realsense")
                rec.started(k)
                rec.completed(k)
            s = store.current_summaries()[0]
            self.assertIsNone(s["lost"])
            self.assertIn("lost frames unknown", s["failure_denominator_basis"])

    def test_recovery_time_after_an_outage(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            rec = _feed(store, mono, n=2, gap=1.0)
            mono.now += 20.0                       # outage: no result for 20 s
            k = rec.received("late", "realsense", run_id="edge1", seq=3, client={"reconnected": True})
            rec.started(k)
            rec.completed(k)
            s = store.current_summaries()[0]
            self.assertEqual(s["recovery_n"], 1)
            self.assertGreater(s["recovery_p50_s"], 20.0)

    def test_resources_are_attached_per_machine_and_gpu_absent_stays_na(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            _feed(store, mono, n=2)
            store.record_resources({"cpu_percent": 20.0, "ram_percent": 40.0}, {"available": False}, 1)
            store.record_resources({"cpu_percent": 60.0, "ram_percent": 42.0}, {"available": False}, 2)
            s = store.current_summaries()[0]
            self.assertEqual((s["cpu_avg_pct"], s["cpu_peak_pct"], s["cpu_n"]), (40.0, 60.0, 2))
            self.assertIsNone(s["gpu_util_avg_pct"])
            self.assertEqual(s["host_machine"], store.host)

    def test_pct_difference_rules(self) -> None:
        self.assertIsNone(cs.pct_difference(0.0, 5.0))
        self.assertIsNone(cs.pct_difference(None, 5.0))
        self.assertIsNone(cs.pct_difference(5.0, None))
        self.assertEqual(cs.pct_difference(200.0, 100.0), -50.0)

    def test_corrupt_last_line_is_skipped_on_reload(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d)
            _feed(store, mono, n=2)
            with (Path(d) / "observations.jsonl").open("a") as handle:
                handle.write('{"run_id": "cut-sh')        # interrupted write
            again, _, _ = _store(d, start=9000.0, wall=1.8e9)
            self.assertEqual(again.skipped_lines, 1)
            self.assertEqual([r["run_id"] for r in again.runs()], [store.run_id])   # history intact

    def test_import_other_mode_keeps_source_and_modes_separate(self) -> None:
        with TemporaryDirectory() as local_dir, TemporaryDirectory() as cloud_dir:
            local, lmono, _ = _store(local_dir, "local")
            _feed(local, lmono, n=4, gap=2.0)
            cloud, cmono, _ = _store(cloud_dir, "cloud", wall=1.7e9 + 100)
            _feed(cloud, cmono, n=4, gap=1.0)
            exported = cs.to_csv(local.own_summaries(), cs.SUMMARY_COLUMNS)
            result = cloud.import_summaries(cs.parse_summary_csv(exported))
            self.assertEqual(result, {"accepted": 1, "rejected": 0})
            self.assertEqual(cloud.import_summaries(cs.parse_summary_csv(exported))["accepted"], 0)   # dedup
            bad = cloud.import_summaries([{"run_id": "x", "processing_mode": "mars", "camera_id": "realsense"}])
            self.assertEqual(bad["rejected"], 1)
            view = cloud.comparison()
            self.assertIn("imported", view["other_mode_status"])
            rows = [r for r in view["rows"] if r["camera_id"] == "realsense" and r["metric"] == "fps"]
            self.assertIsNotNone(rows[0]["local"])
            self.assertIsNotNone(rows[0]["cloud"])
            self.assertEqual(rows[0]["pct_difference"], cs.pct_difference(rows[0]["local"], rows[0]["cloud"]))
            logitech = [r for r in view["rows"] if r["camera_id"] == "logitech" and r["metric"] == "fps"]
            self.assertIsNone(logitech[0]["local"])           # no Logitech data: N/A, not zero
            imported = [s for s in view["summaries"] if s["source"] == "imported"]
            self.assertEqual(imported[0]["source_host"], local.host)

    def test_without_other_mode_it_says_not_imported(self) -> None:
        with TemporaryDirectory() as d:
            store, mono, _ = _store(d, "cloud")
            _feed(store, mono)
            view = store.comparison()
            self.assertIn("not imported", view["other_mode_status"])
            self.assertIsNone(view["selected"]["local"])

    def test_historical_run_can_be_selected(self) -> None:
        with TemporaryDirectory() as d:
            first, mono, _ = _store(d)
            _feed(first, mono, n=6, gap=1.0)
            second, mono2, _ = _store(d, start=5000.0, wall=1.7e9 + 3600)
            _feed(second, mono2, n=3, gap=3.0)
            old_key = f"{first.host}|{first.run_id}"
            latest = second.comparison()
            chosen = second.comparison(local_key=old_key)
            fps = lambda v: [r for r in v["rows"] if r["metric"] == "completed_unique" or r["metric"] == "fps"][0]["local"]
            self.assertEqual(chosen["selected"]["local"], old_key)
            self.assertNotEqual(fps(latest), fps(chosen))
            self.assertNotEqual(second.comparison(local_key="nope|nope")["selected"]["local"], "nope|nope")


class DashboardTests(unittest.TestCase):
    def _app(self, directory):
        config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                           enable_bucket_sync=False, results_dir=Path(directory), automatic_baseline_frames=2)
        coordinator = DualCameraCoordinator(config)
        coordinator.warmup()
        return create_app(config, coordinator)

    def _ingest(self, client, frame_id, camera):
        ok, jpg = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))
        meta = {"source": camera, "camera_id": camera, "frame_id": frame_id, "run_id": "e", "seq": 1,
                "intrinsics": {"fx": 200, "fy": 200, "ppx": 80, "ppy": 60, "width": 160, "height": 120}}
        return client.post(f"/api/cameras/{camera}/ingest?sync=1",
                           data={"metadata": json.dumps(meta), "image": (io.BytesIO(jpg.tobytes()), "f.jpg")},
                           content_type="multipart/form-data")

    def test_panel_sits_below_both_streams_on_both_pages(self) -> None:
        with TemporaryDirectory() as d:
            client = self._app(d).test_client()
            for path, logitech_stream in (("/", 'src="/video/cameras/logitech"'),
                                          ("/research", 'src="/video/cameras/logitech"')):
                page = client.get(path).get_data(as_text=True)
                self.assertIn('id="lc-panel"', page)
                self.assertLess(page.index('src="/video/cameras/realsense"'), page.index('id="lc-panel"'))
                self.assertLess(page.index(logitech_stream), page.index('id="lc-panel"'))
                for label in ("Download raw CSV", "Download run summary CSV", "Download comparison CSV"):
                    self.assertIn(label, page)

    def test_exports_match_saved_records_and_survive_restart(self) -> None:
        with TemporaryDirectory() as d:
            client = self._app(d).test_client()
            for i in range(3):
                self.assertEqual(self._ingest(client, f"a{i}", "realsense").status_code, 200)
            self._ingest(client, "a0", "realsense")                  # retransmission
            raw = client.get("/api/local-cloud/raw.csv")
            rows = list(csv.DictReader(io.StringIO(raw.get_data(as_text=True))))
            saved = (Path(d) / "local_cloud_comparison" / "observations.jsonl").read_text().splitlines()
            self.assertEqual(int(raw.headers["X-Row-Count"]), len(rows))
            self.assertEqual(len(rows), len(saved))
            self.assertEqual(sum(r["event"] == "frame" for r in rows), 3)
            # Restart: history is still exported and a second run is added beside it.
            client2 = self._app(d).test_client()
            self._ingest(client2, "b0", "realsense")
            raw2 = list(csv.DictReader(io.StringIO(client2.get("/api/local-cloud/raw.csv").get_data(as_text=True))))
            self.assertEqual(len({r["run_id"] for r in raw2}), 2)
            summaries = client2.get("/api/local-cloud/summaries.csv")
            self.assertEqual(int(summaries.headers["X-Row-Count"]), 2)
            comparison = client2.get("/api/local-cloud/comparison.csv")
            self.assertEqual(int(comparison.headers["X-Row-Count"]), len(cs.METRICS) * 2)
            view = client2.get("/api/local-cloud/view").get_json()
            self.assertEqual(view["processing_mode"], "local")
            self.assertEqual(len(view["runs"]), 2)
            filtered = client2.get("/api/local-cloud/raw.csv?run_id=" + view["runs"][0]["run_id"])
            self.assertLess(int(filtered.headers["X-Row-Count"]), len(raw2))

    def test_import_endpoint_accepts_the_summary_csv(self) -> None:
        with TemporaryDirectory() as d, TemporaryDirectory() as other:
            store, mono, _ = _store(other, "cloud")
            _feed(store, mono, n=3)
            client = self._app(d).test_client()
            text = cs.to_csv(store.own_summaries(), cs.SUMMARY_COLUMNS)
            response = client.post("/api/local-cloud/import", data=text, content_type="text/csv")
            self.assertEqual(response.get_json()["accepted"], 1)
            view = client.get("/api/local-cloud/view").get_json()
            self.assertIn("cloud run(s) available (imported)", view["other_mode_status"])
            self.assertEqual(client.post("/api/local-cloud/import", json={"summaries": "x"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
