"""Per-frame transport telemetry for the local-versus-cloud comparison.

Every frame the edge device sends carries a `frame_id`, `camera_id`, `run_id`
and a per-camera sequence number. The processing server records what happened
to each one -- received, analysed, superseded by a newer frame, or failed --
with its own monotonic clock, and keeps the raw rows so any thesis table can
be recomputed from the export rather than from a dashboard average.

Clock rules (so skew between Pi, laptop and VM cannot distort a number):

* `server_latency` is receipt -> result ready, both on the server's monotonic
  clock. It covers queueing and inference, not the network.
* `client_upload_rtt` is measured on the Pi's monotonic clock (send -> HTTP 202)
  and reported in the *next* frame's metadata. It covers upload and enqueue,
  not inference, because live ingest is asynchronous (latest-frame scheduling).
* True send -> result end-to-end time is only measured by the replay benchmark,
  which waits for a synchronous result on one client clock.

Nothing is invented: a metric with no samples is None ("N/A"), and a window
too short to mean anything says "insufficient data".
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .benchmark import _percentile

DEFAULT_WINDOW_S = 60.0
MIN_COMPLETED_FOR_METRICS = 5
MIN_SPAN_S = 10.0

SERVER_LATENCY_BOUNDARY = "server receipt -> analysis result ready (server monotonic clock; excludes network)"
PROCESSING_BOUNDARY = "inference start -> result ready (server monotonic clock)"
CLIENT_RTT_BOUNDARY = "edge send -> HTTP 202 accepted (edge monotonic clock; upload + enqueue, excludes inference)"

TELEMETRY_COLUMNS = [
    "run_id", "frame_id", "camera_id", "seq", "processing_mode", "status",
    "received_wall", "server_latency_ms", "queue_ms", "processing_ms",
    "client_prev_upload_rtt_ms", "bytes_in", "bytes_out", "reconnected", "error",
]


@dataclass
class FrameRecord:
    frame_id: str
    camera_id: str
    run_id: str
    seq: int | None
    processing_mode: str
    received_wall: float
    received_mono: float
    bytes_in: int = 0
    bytes_out: int | None = None
    started_mono: float | None = None
    finished_mono: float | None = None
    status: str = "received"   # received | processing | completed | superseded | failed
    error: str | None = None
    client_prev_upload_rtt_ms: float | None = None
    reconnected: bool = False
    persisted: bool = False

    def _ms(self, start: float | None, end: float | None) -> float | None:
        return None if start is None or end is None else round((end - start) * 1000.0, 3)

    def row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "frame_id": self.frame_id, "camera_id": self.camera_id,
            "seq": self.seq, "processing_mode": self.processing_mode, "status": self.status,
            "received_wall": round(self.received_wall, 3),
            "server_latency_ms": self._ms(self.received_mono, self.finished_mono)
            if self.status == "completed" else None,
            "queue_ms": self._ms(self.received_mono, self.started_mono),
            "processing_ms": self._ms(self.started_mono, self.finished_mono)
            if self.status == "completed" else None,
            "client_prev_upload_rtt_ms": self.client_prev_upload_rtt_ms,
            "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
            "reconnected": self.reconnected, "error": self.error,
        }


def _stats(values: list[float]) -> dict[str, Any]:
    return {"p50_ms": _percentile(values, 0.50), "p95_ms": _percentile(values, 0.95), "n": len(values)}


class TelemetryRecorder:
    """Thread-safe store of per-frame records with rolling-window metrics."""

    def __init__(self, processing_mode: str, *, csv_path: Path | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 max_records: int = 50_000) -> None:
        self.processing_mode = processing_mode
        self.csv_path = csv_path
        self.clock, self.wall = clock, wall
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], FrameRecord] = {}
        self._order: deque[tuple[str, str]] = deque()
        self._max = max_records
        self.duplicates = 0
        self.sink: Callable[[FrameRecord], None] | None = None   # durable store hook
        self.client_counters: dict[str, dict[str, Any]] = {}
        self.edge_resources: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ recording
    def received(self, frame_id: str | None, camera_id: str, *, run_id: str = "", seq: Any = None,
                 bytes_in: int = 0, client: dict[str, Any] | None = None) -> tuple[str, str] | None:
        """Register an arriving frame. Returns its key, or None for a duplicate.

        A frame without an id (an older edge client) gets a server-side one so
        it is still counted once; a repeated id is a retransmission and is
        counted separately, never as a second processed frame.
        """
        client = client or {}
        now = self.clock()
        with self._lock:
            if not frame_id:
                frame_id = f"srv-{camera_id}-{len(self._order) + self.duplicates}-{int(self.wall() * 1000)}"
            key = (camera_id, str(frame_id))
            if key in self._records:
                self.duplicates += 1
                return None
            try:
                seq_value = int(seq) if seq is not None else None
            except (TypeError, ValueError):
                seq_value = None
            rtt = client.get("prev_upload_rtt_ms")
            record = FrameRecord(
                frame_id=str(frame_id), camera_id=camera_id, run_id=str(run_id or ""), seq=seq_value,
                processing_mode=self.processing_mode, received_wall=self.wall(), received_mono=now,
                bytes_in=int(bytes_in or 0),
                client_prev_upload_rtt_ms=float(rtt) if isinstance(rtt, (int, float)) else None,
                reconnected=bool(client.get("reconnected")),
            )
            self._records[key] = record
            self._order.append(key)
            counters = {k: client[k] for k in ("send_failures", "timeouts", "reconnects") if k in client}
            if counters:
                self.client_counters[camera_id] = {**counters, "at": self.wall()}
            resources = client.get("resources")
            if isinstance(resources, dict):
                self.edge_resources[camera_id] = {**resources, "at": self.wall()}
            while len(self._order) > self._max:
                self._records.pop(self._order.popleft(), None)
            return key

    def _set(self, key: tuple[str, str] | None, **values: Any) -> None:
        if key is None:
            return
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return
            for name, value in values.items():
                setattr(record, name, value)
            # A frame is written once, when it first reaches a final status;
            # a later reply-size update must not add a second row.
            final = record.status in {"completed", "superseded", "failed"} and not record.persisted
            if final:
                record.persisted = True
        if final:
            self._append_csv(record)
            if self.sink is not None:
                try:
                    self.sink(record)
                except Exception:  # noqa: BLE001 - storage must never stop processing
                    pass

    def started(self, key: tuple[str, str] | None) -> None:
        self._set(key, started_mono=self.clock(), status="processing")

    def completed(self, key: tuple[str, str] | None, bytes_out: int | None = None) -> None:
        extra = {} if bytes_out is None else {"bytes_out": int(bytes_out)}
        self._set(key, finished_mono=self.clock(), status="completed", **extra)

    def response_bytes(self, key: tuple[str, str] | None, bytes_out: int) -> None:
        """Size of the HTTP reply for this frame (known before async analysis ends)."""
        self._set(key, bytes_out=int(bytes_out))

    def superseded(self, key: tuple[str, str] | None) -> None:
        self._set(key, status="superseded")

    def failed(self, key: tuple[str, str] | None, error: str) -> None:
        self._set(key, finished_mono=self.clock(), status="failed", error=str(error)[:200])

    def _append_csv(self, record: FrameRecord) -> None:
        if self.csv_path is None:
            return
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists()
            with self._lock, self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TELEMETRY_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(record.row())
        except OSError:  # pragma: no cover - disk issues must not stop processing
            pass

    # -------------------------------------------------------------- metrics
    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._records[key].row() for key in self._order if key in self._records]

    def _window(self, window_s: float, camera_id: str | None) -> list[FrameRecord]:
        cutoff = self.clock() - window_s
        with self._lock:
            return [r for key in self._order if (r := self._records.get(key)) is not None
                    and r.received_mono >= cutoff and (camera_id is None or r.camera_id == camera_id)]

    def metrics(self, window_s: float = DEFAULT_WINDOW_S, camera_id: str | None = None) -> dict[str, Any]:
        records = self._window(window_s, camera_id)
        done = [r for r in records if r.status == "completed"]
        span = (max(r.received_mono for r in records) - min(r.received_mono for r in records)) if records else 0.0
        sufficient = len(done) >= MIN_COMPLETED_FOR_METRICS and span >= MIN_SPAN_S
        latency = [r.row()["server_latency_ms"] for r in done]
        processing = [r.row()["processing_ms"] for r in done]
        rtt = [r.client_prev_upload_rtt_ms for r in records if r.client_prev_upload_rtt_ms is not None]
        # Frames the edge numbered but the server never received (sequence gaps).
        lost = 0
        by_stream: dict[tuple[str, str], list[int]] = {}
        for r in records:
            if r.seq is not None:
                by_stream.setdefault((r.run_id, r.camera_id), []).append(r.seq)
        for seqs in by_stream.values():
            lost += (max(seqs) - min(seqs) + 1) - len(set(seqs))
        received = len(records)
        sent = received + lost
        in_flight = sum(r.status in {"received", "processing"} for r in records)
        failed = sum(r.status == "failed" for r in records)
        superseded = sum(r.status == "superseded" for r in records)
        bytes_in = sum(r.bytes_in for r in records)
        bytes_out = [r.bytes_out for r in done if r.bytes_out is not None]
        minutes = max(span, 1e-9) / 60.0
        return {
            "processing_mode": self.processing_mode,
            "camera_id": camera_id or "all",
            "window_s": window_s,
            "observed_span_s": round(span, 2),
            "status": "ok" if sufficient else "insufficient data",
            "latency": {"boundary": SERVER_LATENCY_BOUNDARY, **_stats(latency)},
            "processing": {"boundary": PROCESSING_BOUNDARY, **_stats(processing)},
            "client_upload_rtt": {"boundary": CLIENT_RTT_BOUNDARY, **_stats(rtt)},
            "throughput": {
                "definition": "unique frame ids with a completed analysis / observed span",
                "unique_completed": len({(r.camera_id, r.frame_id) for r in done}),
                "fps": round(len(done) / span, 3) if sufficient and span > 0 else None,
            },
            "reliability": {
                "sent": sent, "received": received, "completed": len(done),
                "superseded_by_newer_frame": superseded, "failed": failed,
                "lost_in_transit": lost, "in_flight": in_flight,
                "duplicates_ignored": self.duplicates,
                "reconnects": sum(r.reconnected for r in records),
                "completed_pct": round(100.0 * len(done) / sent, 2) if sent and sufficient else None,
                "denominator": "frames numbered by the edge client in this window (received + sequence gaps)",
                "client_counters": dict(self.client_counters) if camera_id is None
                else self.client_counters.get(camera_id),
            },
            "transfer": {
                "bytes_in_per_frame": round(bytes_in / received, 1) if received else None,
                "bytes_out_per_completed_frame": round(sum(bytes_out) / len(bytes_out), 1) if bytes_out else None,
                "bytes_in_per_minute": round(bytes_in / minutes, 1) if sufficient else None,
                "bytes_out_per_minute": round(sum(bytes_out) / minutes, 1) if sufficient and bytes_out else None,
            },
        }

    def summary(self, window_s: float = DEFAULT_WINDOW_S) -> dict[str, Any]:
        cameras = sorted({r.camera_id for r in self._window(window_s, None)})
        return {
            "processing_mode": self.processing_mode,
            "all": self.metrics(window_s),
            "cameras": {camera: self.metrics(window_s, camera) for camera in cameras},
            "edge_resources": dict(self.edge_resources),
            "host": host_info(),
            "host_resources": host_resources(),
            "gpu": gpu_resources(),
        }


# ------------------------------------------------------------------ hosts
_commit_cache: dict[str, str | None] = {}


def code_commit() -> str | None:
    """The deployed code's git commit (LOCALLIFE_CODE_COMMIT, else `git rev-parse`), or None."""
    if "value" not in _commit_cache:
        value = os.environ.get("LOCALLIFE_CODE_COMMIT", "").strip() or None
        if value is None and shutil.which("git"):
            try:
                out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                     capture_output=True, text=True, timeout=2, check=False)
                value = out.stdout.strip() or None if out.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                value = None
        _commit_cache["value"] = value
    return _commit_cache["value"]


def host_info() -> dict[str, Any]:
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "code_commit": code_commit()}


def _proc_cpu_times() -> tuple[int, int] | None:
    try:
        values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    except (OSError, ValueError, IndexError):
        return None
    return values[3] + (values[4] if len(values) > 4 else 0), sum(values)


_cpu_last: dict[str, tuple[int, int]] = {}


def host_resources() -> dict[str, Any]:
    """CPU and RAM of this machine; None where it cannot be read."""
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return {"cpu_percent": psutil.cpu_percent(interval=None), "ram_percent": memory.percent,
                "ram_used_mb": round(memory.used / 1e6, 1), "source": "psutil"}
    except ImportError:
        pass
    result: dict[str, Any] = {"cpu_percent": None, "ram_percent": None, "ram_used_mb": None, "source": "/proc"}
    times = _proc_cpu_times()
    if times is not None:
        previous = _cpu_last.get("host")
        _cpu_last["host"] = times
        if previous is not None and times[1] > previous[1]:
            result["cpu_percent"] = round(100.0 * (1 - (times[0] - previous[0]) / (times[1] - previous[1])), 1)
    try:
        info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        total, available = info["MemTotal"], info["MemAvailable"]
        result["ram_percent"] = round(100.0 * (total - available) / total, 1)
        result["ram_used_mb"] = round((total - available) / 1000.0, 1)
    except (OSError, KeyError, ValueError, IndexError, ZeroDivisionError):
        pass
    if result["cpu_percent"] is None and result["ram_percent"] is None:
        result["source"] = "unavailable"
    return result


_gpu_cache: dict[str, Any] = {"at": 0.0, "value": None}


def gpu_resources(max_age_s: float = 5.0) -> dict[str, Any]:
    """GPU utilisation and memory via nvidia-smi (2 s timeout, cached)."""
    now = time.monotonic()
    if _gpu_cache["value"] is not None and now - _gpu_cache["at"] < max_age_s:
        return _gpu_cache["value"]
    value: dict[str, Any] = {"available": False, "name": None, "utilization_percent": None,
                             "memory_used_mb": None, "memory_total_mb": None}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=False,
            ).stdout.strip().splitlines()
            if out:
                name, util, used, total = [part.strip() for part in out[0].split(",")]
                value = {"available": True, "name": name, "utilization_percent": float(util),
                         "memory_used_mb": float(used), "memory_total_mb": float(total)}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    _gpu_cache.update(at=now, value=value)
    return value


def edge_resources() -> dict[str, Any]:
    """The edge device's own CPU/RAM, sent in frame metadata (Linux /proc only)."""
    data = host_resources()
    return {"cpu_percent": data.get("cpu_percent"), "ram_percent": data.get("ram_percent")}


def export_json(recorder: TelemetryRecorder, window_s: float) -> str:
    return json.dumps({"summary": recorder.summary(window_s), "rows": recorder.rows()}, default=str)


__all__ = ["TelemetryRecorder", "FrameRecord", "TELEMETRY_COLUMNS", "host_resources", "gpu_resources",
           "edge_resources", "export_json"]
