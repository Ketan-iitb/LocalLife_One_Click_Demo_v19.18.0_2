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
    "latency": "server receipt -> analysis result ready (server monotonic clock; excludes network)",
    "inference": "inference start -> result ready (server monotonic clock; shared camera batch)",
    "queue": "server receipt -> inference start (server monotonic clock)",
    "upload": "edge send -> HTTP 202 accepted (edge monotonic clock; upload + enqueue)",
    "startup": "server process start -> first completed result for the camera (server monotonic clock); "
               "GPU provisioning / stockout wait is NOT included",
    "recovery": "last completed result before an outage (edge reconnect or >10 s without a result) -> "
                "next completed result (server monotonic clock)",
}

RAW_COLUMNS = [
    "schema", "recorded_at", "run_id", "source_host", "processing_mode", "camera_id", "frame_id", "event",
    "status", "edge_run_id", "seq", "received_wall", "elapsed_s", "server_latency_ms", "queue_ms",
    "processing_ms", "upload_rtt_ms", "bytes_in", "bytes_out", "reconnected", "error",
    "cpu_percent", "ram_percent", "ram_used_mb", "gpu_util_percent", "vram_used_mb", "gpu_name",
    "detector", "depth_model", "code_commit",
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
]

# (key, label, unit, favourable direction, sample-count key)
METRICS = [
    ("startup_to_first_result_s", "1. Startup to first successful result", "s", "lower", None),
    ("latency_p50_ms", "2. End-to-end result latency p50", "ms", "lower", "latency_n"),
    ("latency_p95_ms", "2. End-to-end result latency p95", "ms", "lower", "latency_n"),
    ("inference_p50_ms", "3. Model inference time p50", "ms", "lower", "inference_n"),
    ("inference_p95_ms", "3. Model inference time p95", "ms", "lower", "inference_n"),
    ("queue_p50_ms", "4. Queue waiting time p50", "ms", "lower", "queue_n"),
    ("queue_p95_ms", "4. Queue waiting time p95", "ms", "lower", "queue_n"),
    ("upload_p50_ms", "5. Frame upload/transport time p50", "ms", "lower", "upload_n"),
    ("upload_p95_ms", "5. Frame upload/transport time p95", "ms", "lower", "upload_n"),
    ("fps", "6. Completed unique frames per second", "frames/s", "higher", "completed_unique"),
    ("superseded", "7. Superseded frames (not errors)", "frames", None, "received"),
    ("superseded_pct", "7. Superseded frames", "%", None, "received"),
    ("failure_count", "8. Failed/timed-out/lost frames", "frames", "lower", "failure_denominator"),
    ("failure_pct", "8. Failed/timed-out/lost frames", "%", "lower", "failure_denominator"),
    ("recovery_p50_s", "9. Recovery time after disconnect p50", "s", "lower", "recovery_n"),
    ("recovery_max_s", "9. Recovery time after disconnect max", "s", "lower", "recovery_n"),
    ("cpu_avg_pct", "10. Host CPU average", "%", None, "cpu_n"),
    ("cpu_peak_pct", "10. Host CPU peak", "%", None, "cpu_n"),
    ("ram_avg_pct", "11. Host RAM average", "%", None, "ram_n"),
    ("ram_peak_pct", "11. Host RAM peak", "%", None, "ram_n"),
    ("gpu_util_avg_pct", "11. GPU utilisation average", "%", None, "gpu_n"),
    ("gpu_util_peak_pct", "11. GPU utilisation peak", "%", None, "gpu_n"),
    ("vram_avg_mb", "11. VRAM average", "MB", None, "gpu_n"),
    ("vram_peak_mb", "11. VRAM peak", "MB", None, "gpu_n"),
    ("bytes_in_per_frame", "12. Data received by server per completed frame", "bytes", None, "completed_unique"),
    ("bytes_out_per_frame", "12. Data sent by server per completed frame", "bytes", None, "completed_unique"),
    ("mb_in_per_min", "12. Data received by server", "MB/min", None, "completed_unique"),
    ("mb_out_per_min", "12. Data sent by server", "MB/min", None, "completed_unique"),
    ("latency_spread_ms", "13. Latency variation (p95 - p50)", "ms", "lower", "latency_n"),
]

_NUMERIC_SUMMARY = {c for c in SUMMARY_COLUMNS if c not in {
    "source", "source_host", "run_id", "processing_mode", "camera_id", "host_machine", "gpu_name", "detector",
    "depth_model", "code_commit", "failure_denominator_basis", "quality"}}


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
            "error": frame.error, **{k: self.meta.get(k) for k in ("detector", "depth_model")},
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
        for camera in CAMERAS:
            picked = {mode: next((s for s in summaries if f"{s.get('source_host')}|{s.get('run_id')}" == key
                                  and s.get("camera_id") == camera), None) for mode, key in chosen.items()}
            for key, label, unit, better, n_key in METRICS:
                local = _num(picked["local"].get(key)) if picked["local"] else None
                cloud = _num(picked["cloud"].get(key)) if picked["cloud"] else None
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
        writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    return output.getvalue()


COMPARISON_COLUMNS = ["camera_id", "metric", "label", "unit", "favourable", "local", "cloud", "abs_difference",
                      "pct_difference", "local_n", "cloud_n", "local_duration_s", "cloud_duration_s",
                      "local_run", "cloud_run", "local_machine", "cloud_machine"]


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
