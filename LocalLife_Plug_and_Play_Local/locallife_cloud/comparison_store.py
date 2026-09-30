"""Durable Local-versus-Cloud comparison: raw observations, per-run summaries, CSV.

Storage (under <results_dir>/local_cloud_comparison/):

* observations.jsonl        append-only raw events, one JSON object per line:
                            every finished frame (completed / superseded /
                            failed), resource samples and one run_start per
                            server start. Never rewritten or truncated.
* imported_summaries.jsonl  append-only run summaries received from the OTHER
                            processing mode (local and cloud are separate
                            servers with separate disks). Each keeps its
                            source host and run id; the newest version of a
                            (source, run, camera) wins when read.
* run_summaries/<run>.json  the calculated summaries of each own run,
                            rewritten atomically (temp file + os.replace).

A run is one server process in one processing mode. Every event is keyed by
run_id + camera_id + frame_id + event, so a retried write is stored once.
Lines are written with a single O_APPEND write and fsync; a line cut short by
a crash is skipped on reload, never repaired into a number.

Metrics are only calculated from values that were measured. Anything with no
samples is None and shown as N/A -- never zero.
"""

from __future__ import annotations

import csv
import io
import json
import os
import socket
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .benchmark import _percentile

SCHEMA_VERSION = 1
CAMERAS = ("realsense", "logitech")
OUTAGE_GAP_S = 10.0          # no completed result for this long counts as a disconnect

BOUNDARIES = {
    "latency": "server receipt -> analysis result ready (server monotonic clock; excludes upload and "
               "return transport, so it is NOT end-to-end)",
    "inference": "batch start -> result ready (server monotonic clock): detection, depth and measurement "
                 "for the camera batch, not model forward time alone",
    "queue": "server receipt -> inference start (server monotonic clock)",
    "upload": "edge upload request RTT: edge send -> HTTP 202 received (one edge monotonic clock); "
              "includes server enqueue, not one-way network latency",
    "startup": "server process start -> first completed result for the camera (server monotonic clock); "
               "GPU provisioning / stockout wait is NOT included",
    "recovery": "last completed result before an outage (edge reconnect or >10 s without a result) -> "
                "next completed result (server monotonic clock)",
    "capture_to_display": "not measured: Pi capture and browser display share no clock, and displayed "
                          "results are not correlated to frame ids",
    "window": "unique completed frame ids / (last result - first receipt) for this camera, server clock",
    "superseded": "frames replaced in the latest-frame queue by a newer frame before analysis (policy, not "
                  "failure); denominator = frames received by the server",
    "failed": "frames whose analysis raised an error on the server; denominator = failure_denominator",
    "lost": "edge sequence numbers never received by the server; needs the v41 edge client",
    "host": "CPU/RAM % of the machine running the server (laptop in local mode, VM in cloud mode); not "
            "comparable capacity",
    "gpu": "nvidia-smi on the machine running the server; says nothing about whether the pipeline used it",
    "bytes": "HTTP request size received / reply size sent by the server (excludes TCP/TLS overhead)",
}

RAW_COLUMNS = [
    "schema", "recorded_at", "run_id", "source_host", "processing_mode", "camera_id", "frame_id", "event",
    "status", "edge_run_id", "seq", "received_wall", "elapsed_s", "server_latency_ms", "queue_ms",
    "processing_ms", "upload_rtt_ms", "bytes_in", "bytes_out", "reconnected", "error",
    "cpu_percent", "ram_percent", "ram_used_mb", "gpu_util_percent", "vram_used_mb", "gpu_name",
    "detector", "depth_model", "code_commit", "input_id",
]

SUMMARY_COLUMNS = [
    "source", "source_host", "run_id", "processing_mode", "camera_id", "started_wall", "duration_s",
    "host_machine", "gpu_name", "detector", "depth_model", "code_commit",
    "startup_to_first_result_s", "latency_p50_ms", "latency_p95_ms", "latency_n",
    "inference_p50_ms", "inference_p95_ms", "inference_n", "queue_p50_ms", "queue_p95_ms", "queue_n",
    "upload_p50_ms", "upload_p95_ms", "upload_n", "completed_unique", "fps",
    "received", "superseded", "superseded_pct", "failed", "lost", "failure_count", "failure_pct",
    "failure_denominator", "failure_denominator_basis", "recovery_p50_s", "recovery_max_s", "recovery_n",
    "cpu_avg_pct", "cpu_peak_pct", "cpu_n", "ram_avg_pct", "ram_peak_pct", "ram_n",
    "gpu_util_avg_pct", "gpu_util_peak_pct", "vram_avg_mb", "vram_peak_mb", "gpu_n",
    "bytes_in_per_frame", "bytes_out_per_frame", "mb_in_per_min", "mb_out_per_min",
    "latency_spread_ms", "quality", "computed_at",
    # appended in the telemetry-labels revision (earlier columns keep their order)
    "failure_basis_note", "capture_to_display_ms", "frames_window_s", "window_start_wall", "window_end_wall",
    "input_id", "na_reasons",
]

# (key, label, unit, favourable direction, sample-count key, boundary key).
# Keys are unchanged for CSV compatibility; labels say what is actually measured.
METRICS = [
    ("startup_to_first_result_s", "1. Server start to first result", "s", "lower", None, "startup"),
    ("latency_p50_ms", "2. Server-side result latency p50", "ms", "lower", "latency_n", "latency"),
    ("latency_p95_ms", "2. Server-side result latency p95", "ms", "lower", "latency_n", "latency"),
    ("inference_p50_ms", "3. Batch processing time p50 (detection + depth + measurement)", "ms", "lower",
     "inference_n", "inference"),
    ("inference_p95_ms", "3. Batch processing time p95 (detection + depth + measurement)", "ms", "lower",
     "inference_n", "inference"),
    ("queue_p50_ms", "4. Queue waiting time p50", "ms", "lower", "queue_n", "queue"),
    ("queue_p95_ms", "4. Queue waiting time p95", "ms", "lower", "queue_n", "queue"),
    ("upload_p50_ms", "5. Edge upload request RTT p50", "ms", "lower", "upload_n", "upload"),
    ("upload_p95_ms", "5. Edge upload request RTT p95", "ms", "lower", "upload_n", "upload"),
    ("capture_to_display_ms", "5b. Camera capture -> dashboard-visible result", "ms", "lower", None,
     "capture_to_display"),
    ("fps", "6. Completed unique frames per second (observation window)", "frames/s", "higher",
     "completed_unique", "window"),
    ("superseded", "7. Superseded frames (replaced by a newer frame; not failures)", "frames", None,
     "received", "superseded"),
    ("superseded_pct", "7. Superseded frames, % of frames received", "%", None, "received", "superseded"),
    ("failed", "8a. Observed failed/timed-out frames (server side)", "frames", "lower", "received", "failed"),
    ("lost", "8b. Lost frames (edge sequence gaps; unobservable without sequence numbers)", "frames",
     "lower", "failure_denominator", "lost"),
    ("failure_pct", "8. Failed + lost, % of denominator", "%", "lower", "failure_denominator", "failed"),
    ("recovery_p50_s", "9. Recovery time after an outage p50", "s", "lower", "recovery_n", "recovery"),
    ("recovery_max_s", "9. Recovery time after an outage max", "s", "lower", "recovery_n", "recovery"),
    ("cpu_avg_pct", "10. Host CPU average (% of that machine; different machines)", "%", None, "cpu_n", "host"),
    ("cpu_peak_pct", "10. Host CPU peak (% of that machine; different machines)", "%", None, "cpu_n", "host"),
    ("ram_avg_pct", "11. Host RAM average (% of that machine; different machines)", "%", None, "ram_n", "host"),
    ("ram_peak_pct", "11. Host RAM peak (% of that machine; different machines)", "%", None, "ram_n", "host"),
    ("gpu_util_avg_pct", "11. GPU utilisation average", "%", None, "gpu_n", "gpu"),
    ("gpu_util_peak_pct", "11. GPU utilisation peak", "%", None, "gpu_n", "gpu"),
    ("vram_avg_mb", "11. VRAM used average", "MB", None, "gpu_n", "gpu"),
    ("vram_peak_mb", "11. VRAM used peak", "MB", None, "gpu_n", "gpu"),
    ("bytes_in_per_frame", "12. Request bytes received by server per completed frame", "bytes", None,
     "completed_unique", "bytes"),
    ("bytes_out_per_frame", "12. Reply bytes sent by server per completed frame", "bytes", None,
     "completed_unique", "bytes"),
    ("mb_in_per_min", "12. Request data received by server", "MB/min", None, "completed_unique", "bytes"),
    ("mb_out_per_min", "12. Reply data sent by server", "MB/min", None, "completed_unique", "bytes"),
    ("latency_spread_ms", "13. Server-side latency variation (p95 - p50, same samples)", "ms", "lower",
     "latency_n", "latency"),
]

_NUMERIC_SUMMARY = {c for c in SUMMARY_COLUMNS if c not in {
    "source", "source_host", "run_id", "processing_mode", "camera_id", "host_machine", "gpu_name", "detector",
    "depth_model", "code_commit", "failure_denominator_basis", "quality", "failure_basis_note", "input_id",
    "na_reasons"}}


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _stats(values: list[float]) -> tuple[float | None, float | None, int]:
    return _percentile(values, 0.5), _percentile(values, 0.95), len(values)


def _avg_peak(values: list[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    return round(statistics.fmean(values), 2), round(max(values), 2), len(values)


def pct_difference(local: float | None, cloud: float | None) -> float | None:
    """100 x (Cloud - Local) / Local; None when either is missing or Local is zero."""
    if local is None or cloud is None or local == 0:
        return None
    return round(100.0 * (cloud - local) / local, 2)


def summarise_run(records: list[dict[str, Any]], camera_id: str,
                  run_start: dict[str, Any] | None) -> dict[str, Any]:
    """Per-run, per-camera metrics from raw events (own runs only)."""
    frames = [r for r in records if r.get("event") == "frame" and r.get("camera_id") == camera_id]
    resources = [r for r in records if r.get("event") == "resource"]
    start = run_start or {}
    done = [r for r in frames if r.get("status") == "completed"]
    latency = [v for r in done if (v := _num(r.get("server_latency_ms"))) is not None]
    inference = [v for r in done if (v := _num(r.get("processing_ms"))) is not None]
    queue = [v for r in done if (v := _num(r.get("queue_ms"))) is not None]
    upload = [v for r in frames if (v := _num(r.get("upload_rtt_ms"))) is not None]
    received = len(frames)
    superseded = sum(r.get("status") == "superseded" for r in frames)
    failed = sum(r.get("status") == "failed" for r in frames)
    # Sequence gaps per edge run: frames the edge numbered that never arrived.
    seqs: dict[str, list[int]] = {}
    for r in frames:
        if (s := _num(r.get("seq"))) is not None:
            seqs.setdefault(str(r.get("edge_run_id") or ""), []).append(int(s))
    lost = sum((max(v) - min(v) + 1) - len(set(v)) for v in seqs.values())
    denominator = received + lost
    basis = ("frames numbered by the edge client (received + sequence gaps)" if seqs
             else "frames received (edge client sends no sequence numbers: lost frames unknown)")
    # Times on the server's monotonic clock, relative to this server's start.
    finished = sorted(e + (_num(r.get("server_latency_ms")) or 0) / 1000.0
                      for r in done if (e := _num(r.get("elapsed_s"))) is not None)
    arrivals = [e for r in frames if (e := _num(r.get("elapsed_s"))) is not None]
    span = (max(finished) - min(arrivals)) if finished and arrivals else None
    recoveries: list[float] = []
    ordered = sorted(((_num(r.get("elapsed_s")), r) for r in done if _num(r.get("elapsed_s")) is not None),
                     key=lambda item: item[0])
    for (prev_t, prev), (t, r) in zip(ordered, ordered[1:]):
        prev_end = prev_t + (_num(prev.get("server_latency_ms")) or 0) / 1000.0
        end = t + (_num(r.get("server_latency_ms")) or 0) / 1000.0
        if r.get("reconnected") in (True, "True", "true", 1) or end - prev_end > OUTAGE_GAP_S:
            recoveries.append(end - prev_end)
    lat50, lat95, lat_n = _stats(latency)
    inf50, inf95, inf_n = _stats(inference)
    q50, q95, q_n = _stats(queue)
    up50, up95, up_n = _stats(upload)
    cpu = _avg_peak([v for r in resources if (v := _num(r.get("cpu_percent"))) is not None])
    ram = _avg_peak([v for r in resources if (v := _num(r.get("ram_percent"))) is not None])
    gpu = _avg_peak([v for r in resources if (v := _num(r.get("gpu_util_percent"))) is not None])
    vram = _avg_peak([v for r in resources if (v := _num(r.get("vram_used_mb"))) is not None])
    bytes_in = [v for r in done if (v := _num(r.get("bytes_in"))) is not None]
    bytes_out = [v for r in done if (v := _num(r.get("bytes_out"))) is not None]
    minutes = span / 60.0 if span else None
    gpu_names = [r.get("gpu_name") for r in resources if r.get("gpu_name")]
    inputs = {str(r.get("input_id")) for r in frames if r.get("input_id")}
    walls = [w for r in frames if (w := _num(r.get("received_wall"))) is not None]
    old_edge = bool(frames) and not seqs and not upload
    edge_note = ("the Pi's edge client sends no frame ids or upload RTT (pre-v41 edge client: the launcher only "
                 "copies the project to the Pi when it is missing)")
    na: dict[str, str] = {"capture_to_display": BOUNDARIES["capture_to_display"]}
    if not done:
        for group in ("latency", "inference", "queue", "window", "startup", "bytes"):
            na[group] = "no completed frames for this camera in this run"
    if not upload:
        na["upload"] = (edge_note if old_edge else
                        "replay run: frames came from the benchmark client, not the Pi edge sender" if inputs
                        else "no upload RTT samples reported by the edge client")
    if not seqs:
        na["lost"] = edge_note if frames else "no frames received"
    if not recoveries:
        na["recovery"] = "no outage observed in this run"
    if not resources:
        na["host"] = na["gpu"] = "no resource samples in this run"
    else:
        if cpu[2] == 0 and ram[2] == 0:
            na["host"] = "CPU/RAM not readable on this machine (no /proc and no psutil)"
        if gpu[2] == 0 and vram[2] == 0:
            na["gpu"] = "no NVIDIA GPU readable on this machine (nvidia-smi absent or failed)"
    all_times = [e for r in records if (e := _num(r.get("elapsed_s"))) is not None]
    return {
        "source": "own", "source_host": start.get("source_host"), "run_id": start.get("run_id"),
        "processing_mode": start.get("processing_mode"), "camera_id": camera_id,
        "started_wall": start.get("recorded_at"),
        "duration_s": _r(max(all_times), 1) if all_times else None,
        "host_machine": start.get("source_host"),
        "gpu_name": gpu_names[-1] if gpu_names else start.get("gpu_name"),
        "detector": start.get("detector"), "depth_model": start.get("depth_model"),
        "code_commit": start.get("code_commit"),
        "startup_to_first_result_s": _r(min(finished), 2) if finished else None,
        "latency_p50_ms": lat50, "latency_p95_ms": lat95, "latency_n": lat_n,
        "inference_p50_ms": inf50, "inference_p95_ms": inf95, "inference_n": inf_n,
        "queue_p50_ms": q50, "queue_p95_ms": q95, "queue_n": q_n,
        "upload_p50_ms": up50, "upload_p95_ms": up95, "upload_n": up_n,
        "completed_unique": len({r.get("frame_id") for r in done}),
        "fps": _r(len({r.get("frame_id") for r in done}) / span, 3) if span else None,
        "received": received, "superseded": superseded,
        "superseded_pct": _r(100.0 * superseded / received, 2) if received else None,
        "failed": failed, "lost": lost if seqs else None, "failure_count": failed + lost,
        "failure_pct": _r(100.0 * (failed + lost) / denominator, 2) if denominator else None,
        "failure_denominator": denominator, "failure_denominator_basis": basis,
        "recovery_p50_s": _r(statistics.median(recoveries), 2) if recoveries else None,
        "recovery_max_s": _r(max(recoveries), 2) if recoveries else None, "recovery_n": len(recoveries),
        "cpu_avg_pct": cpu[0], "cpu_peak_pct": cpu[1], "cpu_n": cpu[2],
        "ram_avg_pct": ram[0], "ram_peak_pct": ram[1], "ram_n": ram[2],
        "gpu_util_avg_pct": gpu[0], "gpu_util_peak_pct": gpu[1],
        "vram_avg_mb": vram[0], "vram_peak_mb": vram[1], "gpu_n": max(gpu[2], vram[2]),
        "bytes_in_per_frame": _r(statistics.fmean(bytes_in), 1) if bytes_in else None,
        "bytes_out_per_frame": _r(statistics.fmean(bytes_out), 1) if bytes_out else None,
        "mb_in_per_min": _r(sum(bytes_in) / 1e6 / minutes, 3) if bytes_in and minutes else None,
        "mb_out_per_min": _r(sum(bytes_out) / 1e6 / minutes, 3) if bytes_out and minutes else None,
        "latency_spread_ms": _r(lat95 - lat50, 3) if lat50 is not None and lat95 is not None else None,
        "quality": "Not evaluated (no labelled ground truth)",
        "computed_at": time.time(),
        "failure_basis_note": "zero recorded losses is not proof of zero actual losses when lost frames are "
                              "unobservable",
        "capture_to_display_ms": None,
        "frames_window_s": _r(span, 2) if span else None,
        "window_start_wall": _r(min(walls), 3) if walls else None,
        "window_end_wall": _r(max(walls), 3) if walls else None,
        "input_id": next(iter(inputs)) if len(inputs) == 1 else (None if not inputs else "mixed"),
        "na_reasons": na,
    }


def _clean_summary(item: dict[str, Any]) -> dict[str, Any] | None:
    """Only known columns, numbers as numbers, short strings: nothing else is imported."""
    if not isinstance(item, dict):
        return None
    out: dict[str, Any] = {}
    for column in SUMMARY_COLUMNS:
        value = item.get(column)
        if column in _NUMERIC_SUMMARY:
            out[column] = _num(value)
        elif column == "na_reasons":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = {}
            out[column] = {str(k)[:40]: str(v)[:300] for k, v in (value or {}).items()} if isinstance(value, dict) else {}
        else:
            out[column] = None if value in (None, "") else str(value)[:160]
    if out["processing_mode"] not in ("local", "cloud") or out["camera_id"] not in CAMERAS or not out["run_id"]:
        return None
    return out


class ComparisonStore:
    def __init__(self, directory: Path, *, processing_mode: str, run_meta: dict[str, Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "run_summaries").mkdir(exist_ok=True)
        self.observations_path = self.directory / "observations.jsonl"
        self.imports_path = self.directory / "imported_summaries.jsonl"
        self.clock, self.wall = clock, wall
        self.start_mono = clock()
        self.host = socket.gethostname()
        self.processing_mode = processing_mode
        # Unique even for two starts in the same second (a restart must never merge into the last run).
        self.run_id = (f"{processing_mode}-{self.host}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(wall()))}"
                       f"-{uuid.uuid4().hex[:6]}")
        self.meta = dict(run_meta or {})
        self._lock = threading.Lock()
        self._keys: set[tuple[str, str, str, str]] = set()
        self._current: list[dict[str, Any]] = []
        self._run_starts: dict[str, dict[str, Any]] = {}
        self._old_summaries: list[dict[str, Any]] = []
        self.skipped_lines = 0
        self._load()
        self.append({"event": "run_start", "camera_id": "host", "frame_id": "run_start", "status": "started",
                     **{k: self.meta.get(k) for k in ("gpu_name", "detector", "depth_model", "code_commit")}})

    # ------------------------------------------------------------ storage
    def _iter_file(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except ValueError:
                    self.skipped_lines += 1     # a line cut short by a crash
                    continue
                if isinstance(item, dict):
                    yield item

    def _load(self) -> None:
        by_run: dict[str, list[dict[str, Any]]] = {}
        for item in self._iter_file(self.observations_path):
            key = (str(item.get("run_id")), str(item.get("camera_id")), str(item.get("frame_id")), str(item.get("event")))
            if key in self._keys:
                continue
            self._keys.add(key)
            by_run.setdefault(str(item.get("run_id")), []).append(item)
            if item.get("event") == "run_start":
                self._run_starts[str(item.get("run_id"))] = item
        for run_id, records in by_run.items():
            for camera in CAMERAS:
                if any(r.get("camera_id") == camera for r in records):
                    self._old_summaries.append(summarise_run(records, camera, self._run_starts.get(run_id)))

    def _write_line(self, path: Path, item: dict[str, Any]) -> None:
        data = (json.dumps(item, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, data)          # one write per record: a line is never interleaved
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(self, item: dict[str, Any]) -> bool:
        record = {"schema": SCHEMA_VERSION, "recorded_at": round(self.wall(), 3), "run_id": self.run_id,
                  "source_host": self.host, "processing_mode": self.processing_mode,
                  "elapsed_s": round(self.clock() - self.start_mono, 4), **item}
        key = (record["run_id"], str(record.get("camera_id")), str(record.get("frame_id")), str(record.get("event")))
        with self._lock:
            if key in self._keys:
                return False
            try:
                self._write_line(self.observations_path, record)
            except OSError:
                return False
            self._keys.add(key)
            self._current.append(record)
            if record.get("event") == "run_start":
                self._run_starts[self.run_id] = record
        return True

    # ------------------------------------------------------------- feeders
    def record_frame(self, frame: Any) -> bool:
        """A telemetry FrameRecord that reached its final status."""
        row = frame.row()
        return self.append({
            "event": "frame", "camera_id": frame.camera_id, "frame_id": frame.frame_id, "status": frame.status,
            "edge_run_id": frame.run_id, "seq": frame.seq, "received_wall": row["received_wall"],
            "elapsed_s": round(frame.received_mono - self.start_mono, 4),
            "server_latency_ms": row["server_latency_ms"], "queue_ms": row["queue_ms"],
            "processing_ms": row["processing_ms"], "upload_rtt_ms": frame.client_prev_upload_rtt_ms,
            "bytes_in": frame.bytes_in, "bytes_out": frame.bytes_out, "reconnected": frame.reconnected,
            "error": frame.error, "input_id": frame.input_id,
            **{k: self.meta.get(k) for k in ("detector", "depth_model")},
        })

    def record_resources(self, host: dict[str, Any], gpu: dict[str, Any], sequence: int) -> bool:
        return self.append({
            "event": "resource", "camera_id": "host", "frame_id": f"resource-{sequence}", "status": "sampled",
            "cpu_percent": host.get("cpu_percent"), "ram_percent": host.get("ram_percent"),
            "ram_used_mb": host.get("ram_used_mb"),
            "gpu_util_percent": gpu.get("utilization_percent") if gpu.get("available") else None,
            "vram_used_mb": gpu.get("memory_used_mb") if gpu.get("available") else None,
            "gpu_name": gpu.get("name") if gpu.get("available") else None,
        })

    # ----------------------------------------------------------- summaries
    def current_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._current)
        start = self._run_starts.get(self.run_id)
        return [summarise_run(records, camera, start) for camera in CAMERAS
                if any(r.get("camera_id") == camera for r in records)]

    def own_summaries(self) -> list[dict[str, Any]]:
        return list(self._old_summaries) + self.current_summaries()

    def imported_summaries(self) -> list[dict[str, Any]]:
        newest: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in self._iter_file(self.imports_path):
            clean = _clean_summary(item)
            if clean is None:
                continue
            clean["source"] = "imported"
            clean["imported_at"] = _num(item.get("imported_at"))
            key = (str(clean["source_host"]), str(clean["run_id"]), str(clean["camera_id"]))
            if key not in newest or (clean.get("computed_at") or 0) >= (newest[key].get("computed_at") or 0):
                newest[key] = clean
        own = {(s.get("source_host"), s.get("run_id"), s.get("camera_id")) for s in self._old_summaries}
        own.add((self.host, self.run_id, "realsense"))
        own.add((self.host, self.run_id, "logitech"))
        return [s for key, s in newest.items() if key not in own]

    def all_summaries(self) -> list[dict[str, Any]]:
        items = self.own_summaries() + self.imported_summaries()
        return sorted(items, key=lambda s: (_num(s.get("started_wall")) or 0, str(s.get("camera_id"))))

    def import_summaries(self, items: Iterable[Any]) -> dict[str, int]:
        """Store the other mode's run summaries, attributable to their source."""
        accepted = rejected = 0
        known = {(s.get("source_host"), s.get("run_id"), s.get("camera_id"), s.get("computed_at"))
                 for s in self._iter_file(self.imports_path)}
        for item in items:
            clean = _clean_summary(item)
            if clean is None or clean["run_id"] == self.run_id:
                rejected += 1
                continue
            key = (clean["source_host"], clean["run_id"], clean["camera_id"], clean["computed_at"])
            if key in known:
                continue
            known.add(key)
            clean["source"] = "imported"
            clean["imported_at"] = round(self.wall(), 3)
            with self._lock:
                self._write_line(self.imports_path, clean)
            accepted += 1
        return {"accepted": accepted, "rejected": rejected}

    def persist_summaries(self) -> None:
        """Rewrite this run's summary file atomically (temp file, fsync, replace)."""
        summaries = self.current_summaries()
        if not summaries:
            return
        target = self.directory / "run_summaries" / f"{self.run_id}.json"
        temporary = target.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(summaries, handle, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError:
            pass    # the raw observations remain the source of truth

    # ---------------------------------------------------------- comparison
    def runs(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for s in self.all_summaries():
            key = f"{s.get('source_host')}|{s.get('run_id')}"
            entry = seen.setdefault(key, {"key": key, "run_id": s.get("run_id"), "processing_mode": s.get("processing_mode"),
                                          "source": s.get("source"), "source_host": s.get("source_host"),
                                          "started_wall": s.get("started_wall"), "duration_s": s.get("duration_s"),
                                          "current": s.get("run_id") == self.run_id, "cameras": []})
            entry["cameras"].append(s.get("camera_id"))
        return list(seen.values())

    def comparison(self, local_key: str | None = None, cloud_key: str | None = None) -> dict[str, Any]:
        summaries = self.all_summaries()
        runs = self.runs()

        def latest(mode: str) -> str | None:
            keys = [r["key"] for r in runs if r["processing_mode"] == mode]
            return keys[-1] if keys else None
        valid = {mode: {r["key"] for r in runs if r["processing_mode"] == mode} for mode in ("local", "cloud")}
        # A remembered selection that this server does not have falls back to the latest run.
        chosen = {"local": local_key if local_key in valid["local"] else latest("local"),
                  "cloud": cloud_key if cloud_key in valid["cloud"] else latest("cloud")}
        rows = []
        matches = {}
        for camera in CAMERAS:
            picked = {mode: next((s for s in summaries if f"{s.get('source_host')}|{s.get('run_id')}" == key
                                  and s.get("camera_id") == camera), None) for mode, key in chosen.items()}
            matched, notes = match_runs(picked["local"], picked["cloud"])
            matches[camera] = {"matched": matched, "notes": notes}
            for key, label, unit, better, n_key, boundary in METRICS:
                local = _num(picked["local"].get(key)) if picked["local"] else None
                cloud = _num(picked["cloud"].get(key)) if picked["cloud"] else None
                if better and local is not None and cloud is not None and local != cloud and matched:
                    cloud_better = cloud < local if better == "lower" else cloud > local
                    verdict = f"{'cloud' if cloud_better else 'local'} better ({better} is better)"
                elif better and local is not None and cloud is not None and matched:
                    verdict = "equal"
                elif better:
                    verdict = f"not matched: no verdict ({better} is better)" if local is not None and cloud is not None else "—"
                else:
                    verdict = "context only"
                rows.append({
                    "camera_id": camera, "metric": key, "label": label, "unit": unit, "favourable": better,
                    "local": local, "cloud": cloud,
                    "abs_difference": _r(cloud - local, 3) if local is not None and cloud is not None else None,
                    "pct_difference": pct_difference(local, cloud),
                    "local_n": (picked["local"] or {}).get(n_key) if n_key else (1 if local is not None else 0),
                    "cloud_n": (picked["cloud"] or {}).get(n_key) if n_key else (1 if cloud is not None else 0),
                    "local_duration_s": (picked["local"] or {}).get("duration_s"),
                    "cloud_duration_s": (picked["cloud"] or {}).get("duration_s"),
                    "local_run": chosen["local"] if picked["local"] else None,
                    "cloud_run": chosen["cloud"] if picked["cloud"] else None,
                    "local_machine": (picked["local"] or {}).get("host_machine"),
                    "cloud_machine": (picked["cloud"] or {}).get("host_machine"),
                    "boundary": BOUNDARIES.get(boundary, ""),
                    "matched": matched, "match_notes": "; ".join(notes), "verdict": verdict,
                    "local_na_reason": _na_reason(picked["local"], boundary, local),
                    "cloud_na_reason": _na_reason(picked["cloud"], boundary, cloud),
                    "local_window_s": (picked["local"] or {}).get("frames_window_s"),
                    "cloud_window_s": (picked["cloud"] or {}).get("frames_window_s"),
                })
        other = "cloud" if self.processing_mode == "local" else "local"
        other_runs = [r for r in runs if r["processing_mode"] == other]
        return {
            "processing_mode": self.processing_mode, "current_run": f"{self.host}|{self.run_id}",
            "selected": chosen, "runs": runs, "rows": rows, "summaries": summaries, "boundaries": BOUNDARIES,
            "other_mode": other,
            "other_mode_status": (f"{len(other_runs)} {other} run(s) available"
                                  + (" (imported)" if any(r["source"] == "imported" for r in other_runs) else "")
                                  if other_runs else f"Other mode's data not imported ({other})"),
            "quality": "Not evaluated (no labelled ground truth)",
            "matches": matches,
            "counts": self.counts(),
        }

    # ------------------------------------------------------------------ CSV
    def counts(self) -> dict[str, int]:
        with self._lock:
            return {"raw_observations": len(self._keys), "skipped_corrupt_lines": self.skipped_lines}

    def raw_rows(self, run_id: str | None = None, since: float | None = None) -> list[dict[str, Any]]:
        rows, seen = [], set()
        for item in self._iter_file(self.observations_path):
            key = (str(item.get("run_id")), str(item.get("camera_id")), str(item.get("frame_id")), str(item.get("event")))
            if key in seen:
                continue
            seen.add(key)
            if run_id and item.get("run_id") != run_id:
                continue
            if since is not None and (_num(item.get("recorded_at")) or 0) < since:
                continue
            rows.append(item)
        return rows


def to_csv(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: ("" if row.get(c) is None else
                             json.dumps(row.get(c), sort_keys=True) if isinstance(row.get(c), (dict, list))
                             else row.get(c)) for c in columns})
    return output.getvalue()


COMPARISON_COLUMNS = ["camera_id", "metric", "label", "unit", "favourable", "local", "cloud", "abs_difference",
                      "pct_difference", "local_n", "cloud_n", "local_duration_s", "cloud_duration_s",
                      "local_run", "cloud_run", "local_machine", "cloud_machine",
                      # appended: definitions, matching and N/A reasons
                      "boundary", "matched", "match_notes", "verdict", "local_na_reason", "cloud_na_reason",
                      "local_window_s", "cloud_window_s"]


def _na_reason(summary: dict[str, Any] | None, boundary: str, value: float | None) -> str:
    if value is not None:
        return ""
    if summary is None:
        return "no run selected for this mode and camera"
    reasons = summary.get("na_reasons") or {}
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except ValueError:
            reasons = {}
    return str(reasons.get(boundary) or "not measured in this run")


def match_runs(local: dict[str, Any] | None, cloud: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Whether two runs are a like-for-like comparison. Unmatched runs get no better/worse verdict."""
    if local is None or cloud is None:
        return False, ["a run is missing for one mode"]
    notes = []
    if not local.get("input_id") or local.get("input_id") != cloud.get("input_id") or local.get("input_id") == "mixed":
        notes.append("camera inputs differ or are unidentified (live runs); replay the same recorded frames "
                     "with scripts/benchmark_local_cloud.py for a matched comparison")
    for key, name in (("detector", "detector model"), ("depth_model", "depth model")):
        if local.get(key) != cloud.get(key):
            notes.append(f"{name} differs ({local.get(key)} vs {cloud.get(key)})")
    lr, cr = _num(local.get("received")) or 0, _num(cloud.get("received")) or 0
    if max(lr, cr) and abs(lr - cr) > 0.05 * max(lr, cr):
        notes.append(f"run lengths differ ({int(lr)} vs {int(cr)} frames received)")
    return not notes, notes


def parse_summary_csv(text: str) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(text)))


class ResourceSampler:
    """Host/GPU readings every few seconds into the store (never blocks processing)."""

    def __init__(self, store: ComparisonStore, read_host: Callable[[], dict], read_gpu: Callable[[], dict],
                 interval_s: float = 5.0) -> None:
        self.store, self.read_host, self.read_gpu, self.interval_s = store, read_host, read_gpu, interval_s
        self._stop = threading.Event()
        self._sequence = 0
        self.thread = threading.Thread(target=self._run, name="comparison-resources", daemon=True)

    def start(self) -> "ResourceSampler":
        self.thread.start()
        return self

    def sample_once(self) -> None:
        self._sequence += 1
        try:
            self.store.record_resources(self.read_host(), self.read_gpu(), self._sequence)
        except Exception:  # noqa: BLE001 - a failed reading is simply absent (N/A), never zero
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()
            if self._sequence % 6 == 0:
                try:
                    self.store.persist_summaries()
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
