"""v42 telemetry: honest labels and boundaries, N/A reasons, matched verdicts, CSV compatibility."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from locallife_cloud import comparison_store as cs
from locallife_cloud import edge_client
from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.server import create_app
from locallife_cloud.telemetry import TelemetryRecorder

_spec = importlib.util.spec_from_file_location(
    "benchmark_local_cloud", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_local_cloud.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

OLD_COMPARISON_COLUMNS = ["camera_id", "metric", "label", "unit", "favourable", "local", "cloud", "abs_difference",
                          "pct_difference", "local_n", "cloud_n", "local_duration_s", "cloud_duration_s",
                          "local_run", "cloud_run", "local_machine", "cloud_machine"]


class Clock:
    def __init__(self, t: float) -> None:
        self.now = t

    def __call__(self) -> float:
        return self.now


def _run(directory, mode, *, input_id=None, n=5, seqs=True, detector="det", wall=1.7e9):
    mono = Clock(100.0)
    store = cs.ComparisonStore(Path(directory), processing_mode=mode, clock=mono, wall=Clock(wall),
                               run_meta={"detector": detector, "depth_model": "da"})
    rec = TelemetryRecorder(mode, clock=mono)
    rec.sink = store.record_frame
    for i in range(n):
        key = rec.received(f"f{i}", "realsense", run_id="edge", seq=(i + 1) if seqs else None,
                           input_id=input_id, client={"prev_upload_rtt_ms": 30.0} if seqs else None)
        mono.now += 0.05
        rec.started(key)
        mono.now += 0.2 if mode == "local" else 0.1
        rec.completed(key, bytes_out=100)
        mono.now += 1.0
    return store


class LabelTests(unittest.TestCase):
    def test_latency_is_labelled_server_side_with_its_boundary(self) -> None:
        labels = {key: label for key, label, *_ in cs.METRICS}
        self.assertTrue(labels["latency_p50_ms"].startswith("2. Server-side result latency"))
        self.assertNotIn("End-to-end", " ".join(labels.values()))
        self.assertIn("NOT end-to-end", cs.BOUNDARIES["latency"])
        self.assertIn("not model forward time alone", cs.BOUNDARIES["inference"])
        self.assertIn("not one-way network latency", cs.BOUNDARIES["upload"])
        self.assertIn("different machines", labels["cpu_avg_pct"])
        self.assertIn("not failures", labels["superseded"])
        for *_, boundary in cs.METRICS:
            self.assertIn(boundary, cs.BOUNDARIES)       # every row has a stated boundary

    def test_dashboard_pages_show_the_new_labels(self) -> None:
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            for path in ("/", "/research"):
                page = client.get(path).get_data(as_text=True)
                self.assertIn("Server-side result latency", page)
                self.assertIn("NOT MATCHED", page)
                self.assertIn("Trend needs at least two", page)


class NaReasonTests(unittest.TestCase):
    def test_old_edge_client_explains_missing_rtt_and_losses(self) -> None:
        with TemporaryDirectory() as d:
            s = _run(d, "local", seqs=False).current_summaries()[0]
            self.assertIsNone(s["upload_p50_ms"])
            self.assertIn("pre-v41 edge client", s["na_reasons"]["upload"])
            self.assertIn("pre-v41 edge client", s["na_reasons"]["lost"])
            self.assertIsNone(s["lost"])
            self.assertIn("no outage", s["na_reasons"]["recovery"])
            self.assertIsNone(s["capture_to_display_ms"])
            self.assertIn("share no clock", s["na_reasons"]["capture_to_display"])

    def test_with_sequence_numbers_rtt_and_losses_are_measured(self) -> None:
        with TemporaryDirectory() as d:
            s = _run(d, "local").current_summaries()[0]
            self.assertEqual(s["upload_p50_ms"], 30.0)
            self.assertEqual(s["lost"], 0)
            self.assertNotIn("upload", s["na_reasons"])
            self.assertAlmostEqual(s["frames_window_s"], 4 * 1.25 + 0.25, places=3)


class MatchingTests(unittest.TestCase):
    def _view(self, local_input, cloud_input, cloud_n=5, cloud_detector="det"):
        with TemporaryDirectory() as ld, TemporaryDirectory() as cd:
            local = _run(ld, "local", input_id=local_input)
            cloud = _run(cd, "cloud", input_id=cloud_input, n=cloud_n, detector=cloud_detector, wall=1.7e9 + 50)
            cloud.import_summaries(json.loads(json.dumps(local.own_summaries(), default=str)))
            return cloud.comparison()

    def test_live_runs_are_not_matched_and_get_no_verdict(self) -> None:
        view = self._view(None, None)
        row = next(r for r in view["rows"] if r["camera_id"] == "realsense" and r["metric"] == "latency_p50_ms")
        self.assertFalse(row["matched"])
        self.assertIn("inputs differ or are unidentified", row["match_notes"])
        self.assertTrue(row["verdict"].startswith("not matched: no verdict"))
        self.assertIsNotNone(row["pct_difference"])        # the numbers are still shown

    def test_same_replay_input_and_models_is_matched_with_a_verdict(self) -> None:
        view = self._view("replay-abc", "replay-abc")
        row = next(r for r in view["rows"] if r["camera_id"] == "realsense" and r["metric"] == "latency_p50_ms")
        self.assertTrue(row["matched"], row["match_notes"])
        self.assertEqual(row["verdict"], "cloud better (lower is better)")

    def test_different_models_or_lengths_are_not_matched(self) -> None:
        self.assertIn("detector model differs", self._view("x", "x", cloud_detector="other")["matches"]["realsense"]["notes"][0])
        self.assertIn("run lengths differ", " ".join(self._view("x", "x", cloud_n=9)["matches"]["realsense"]["notes"]))


class DenominatorAndCsvTests(unittest.TestCase):
    def test_failed_lost_and_superseded_are_separate(self) -> None:
        with TemporaryDirectory() as d:
            mono = Clock(0.0)
            store = cs.ComparisonStore(Path(d), processing_mode="local", clock=mono, wall=Clock(1.7e9))
            rec = TelemetryRecorder("local", clock=mono)
            rec.sink = store.record_frame
            for i, seq in enumerate((1, 2, 3, 5)):                 # seq 4 never arrived
                key = rec.received(f"f{i}", "logitech", run_id="e", seq=seq)
                if i == 1:
                    rec.superseded(key)
                elif i == 2:
                    rec.started(key)
                    rec.failed(key, "boom")
                else:
                    rec.started(key)
                    mono.now += 0.1
                    rec.completed(key)
                mono.now += 1
            s = store.current_summaries()[0]
            self.assertEqual((s["superseded"], s["failed"], s["lost"]), (1, 1, 1))
            self.assertEqual(s["superseded_pct"], 25.0)                # of 4 received
            self.assertEqual(s["failure_denominator"], 5)              # received + gaps
            self.assertEqual(s["failure_pct"], 40.0)                   # (failed + lost) / 5

    def test_csv_keeps_old_columns_first_and_appends_new_ones(self) -> None:
        self.assertEqual(cs.COMPARISON_COLUMNS[:len(OLD_COMPARISON_COLUMNS)], OLD_COMPARISON_COLUMNS)
        self.assertIn("boundary", cs.COMPARISON_COLUMNS)
        self.assertEqual(cs.SUMMARY_COLUMNS.index("latency_p50_ms"), 13)   # unchanged position
        with TemporaryDirectory() as d:
            store = _run(d, "local", seqs=False)
            text = cs.to_csv(store.own_summaries(), cs.SUMMARY_COLUMNS)
            row = next(csv.DictReader(io.StringIO(text)))
            self.assertIn("pre-v41", json.loads(row["na_reasons"])["upload"])
            text = cs.to_csv(store.comparison()["rows"], cs.COMPARISON_COLUMNS)
            rows = list(csv.DictReader(io.StringIO(text)))
            self.assertEqual(len(rows), len(cs.METRICS) * 2)
            self.assertTrue(all(r["boundary"] for r in rows))


class CorrelationTests(unittest.TestCase):
    def test_sync_reply_carries_its_own_frame_id_and_server_timing(self) -> None:
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            ok, jpg = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))
            meta = {"source": "realsense", "camera_id": "realsense", "frame_id": "rep-1", "input_id": "replay-x",
                    "intrinsics": {"fx": 200, "fy": 200, "ppx": 80, "ppy": 60, "width": 160, "height": 120}}
            reply = client.post("/api/cameras/realsense/ingest?sync=1", content_type="multipart/form-data",
                                data={"metadata": json.dumps(meta), "image": (io.BytesIO(jpg.tobytes()), "f.jpg")}).get_json()
            self.assertEqual(reply["frame_id"], "rep-1")
            t = reply["server_timing"]
            self.assertAlmostEqual(t["server_latency_ms"], t["queue_ms"] + t["processing_ms"], places=2)
            raw = client.get("/api/local-cloud/raw.csv").get_data(as_text=True)
            frame = next(r for r in csv.DictReader(io.StringIO(raw)) if r["event"] == "frame")
            self.assertEqual(frame["input_id"], "replay-x")

    def test_edge_upload_rtt_uses_one_monotonic_clock(self) -> None:
        clock = {"mono": 50.0, "wall": 1.7e9}

        def fake_send(*_a, **_k):
            clock["mono"] += 0.040          # 40 ms on the monotonic clock
            clock["wall"] -= 3600.0         # the wall clock jumps back an hour meanwhile
            return {}

        fake_time = types.SimpleNamespace(monotonic=lambda: clock["mono"], time=lambda: clock["wall"],
                                          sleep=lambda _s: None, strftime=__import__("time").strftime)
        original = (edge_client.time, edge_client.send_frame, edge_client.encode_frame)
        edge_client._STREAM_STATES.clear()
        try:
            edge_client.time = fake_time
            edge_client.send_frame = fake_send
            edge_client.encode_frame = lambda *_a, **_k: (b"x", None)
            frame = edge_client.CapturedFrame(image=np.zeros((2, 2, 3), np.uint8), depth_m=None, intrinsics=None,
                                              source="t", camera_id="logitech")
            args = types.SimpleNamespace(cloud="http://x", jpeg_quality=80, request_timeout=5, upload_fps=1000,
                                         run_id="r", record_dir="")
            edge_client._stream_one_frame(None, frame, args, False)
            self.assertAlmostEqual(edge_client._STREAM_STATES["logitech"].prev_upload_rtt_ms, 40.0, places=3)
        finally:
            edge_client.time, edge_client.send_frame, edge_client.encode_frame = original
            edge_client._STREAM_STATES.clear()


class BenchmarkTests(unittest.TestCase):
    def test_input_identity_is_stable_and_content_sensitive(self) -> None:
        frames = [{"stem": "a", "image": b"1", "depth": None}, {"stem": "b", "image": b"2", "depth": b"d"}]
        self.assertEqual(bench.input_identity(frames, 1), bench.input_identity(list(frames), 1))
        self.assertNotEqual(bench.input_identity(frames, 1), bench.input_identity(frames, 2))
        changed = [dict(frames[0], image=b"X"), frames[1]]
        self.assertNotEqual(bench.input_identity(frames, 1), bench.input_identity(changed, 1))

    def test_run_summary_csv_has_na_reasons_and_no_blank_superseded(self) -> None:
        row = {"frame_file": "a", "camera_id": "realsense", "warmup": False, "status": "ok", "e2e_ms": 10.0,
               "server_inference_ms": 5.0, "server_latency_ms": 6.0, "server_queue_ms": 0.1,
               "server_processing_ms": 5.9, "bytes_in": 10, "bytes_out": 5}
        summary = {"metadata": {"run_id": "r", "server_processing_mode": "local", "input_id": "replay-1",
                                "warmup_frames": 0, "started_wall": 1.0, "duration_s": 2.0},
                   "all": bench.summarise([row]), "cameras": {"realsense": bench.summarise([row], "realsense")}}
        with TemporaryDirectory() as d:
            bench.write_run_summary_csv(Path(d) / "s.csv", summary)
            rows = list(csv.DictReader((Path(d) / "s.csv").open()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["superseded"], "0")
            self.assertEqual(rows[0]["queue_p50_ms"], "0.1")
            self.assertIn("edge_upload_rtt", json.loads(rows[0]["na_reasons"]))


if __name__ == "__main__":
    unittest.main()
