"""Replay the SAME recorded frames through local and cloud processing, and compare.

Record frames once on the Pi (the exact uploaded bytes are saved):

    python3 -m locallife_cloud.edge_client --cloud http://<laptop>:8000 --source dual --record-dir ~/phase1_frames

Then, from the Windows laptop, with the backend idle (Pi edge client stopped):

    python scripts/benchmark_local_cloud.py run --frames phase1_frames --url http://127.0.0.1:8000 \
        --expect-mode local --warmup 5 --network "lab wifi" [--truth truth.csv]
    (start cloud mode; the tunnel serves the VM on the same port)
    python scripts/benchmark_local_cloud.py run --frames phase1_frames --url http://127.0.0.1:8000 \
        --expect-mode cloud --warmup 5 --network "lab wifi" [--truth truth.csv]
    python scripts/benchmark_local_cloud.py compare results/benchmark/<local_run> results/benchmark/<cloud_run>

Each frame is posted with ?sync=1, so the timing is send -> analysis result
received, on this client's monotonic clock: no cross-machine clock is used.
The server's own reported processing mode is recorded for every frame, and a
run whose server mode differs from --expect-mode is refused, so a local result
can never be labelled cloud.

truth.csv (optional): frame,length_mm,width_mm,height_mm,volume_l,colour where
frame is the recorded file stem (e.g. realsense_0000012). Without it, quality
is reported as "Not evaluated".

NOTE: sync ingest runs the full pipeline, so replayed frames enter that
server's tracker and event log. Point LOCALLIFE_RESULTS_DIR at a separate
benchmark folder when starting the server for a replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.benchmark import _percentile  # noqa: E402

FRAME_COLUMNS = [
    "run_id", "frame_file", "camera_id", "frame_id", "warmup", "status", "server_processing_mode",
    "e2e_ms", "server_inference_ms", "server_latency_ms", "server_queue_ms", "server_processing_ms",
    "bytes_in", "bytes_out", "detections",
    "pred_length_mm", "pred_width_mm", "pred_height_mm", "pred_volume_l", "pred_colour", "error",
]


def load_frames(directory: Path, camera: str | None) -> list[dict[str, Any]]:
    frames = []
    for meta_path in sorted(directory.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image = meta_path.with_suffix(".jpg")
        if not image.exists() or (camera and meta.get("camera_id") != camera):
            continue
        depth = Path(f"{meta_path.with_suffix('')}_depth.npz")
        frames.append({"stem": meta_path.stem, "meta": meta, "image": image.read_bytes(),
                       "depth": depth.read_bytes() if depth.exists() else None})
    return frames


def input_identity(frames: list[dict[str, Any]], repeat: int) -> str:
    """Same recorded frames (names and bytes) and repeat count -> same id, on any machine."""
    import hashlib

    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame["stem"].encode())
        digest.update(hashlib.sha256(frame["image"]).digest())
        if frame["depth"] is not None:
            digest.update(hashlib.sha256(frame["depth"]).digest())
    digest.update(str(repeat).encode())
    return "replay-" + digest.hexdigest()[:16]


def best_detection(result: dict[str, Any]) -> dict[str, Any] | None:
    """The largest measured object in a result (the one a single-object trial is about)."""
    detections = result.get("detections") or []
    if not detections:
        return None

    def volume(item: dict[str, Any]) -> float:
        for key in ("stable_volume_l", "realsense_volume_l", "monocular_volume_l"):
            if item.get(key) is not None:
                return float(item[key])
        return -1.0
    return max(detections, key=volume)


def prediction(result: dict[str, Any]) -> dict[str, Any]:
    item = best_detection(result)
    if item is None:
        return {}
    dims = item.get("dimensions_mm") or {}
    box = item.get("box_dimensions_mm") or {}
    volume = next((item[k] for k in ("stable_volume_l", "realsense_volume_l", "monocular_volume_l")
                   if item.get(k) is not None), None)
    return {
        "pred_length_mm": dims.get("footprint_length", box.get("length")),
        "pred_width_mm": dims.get("footprint_width", box.get("width")),
        "pred_height_mm": dims.get("height", box.get("height")),
        "pred_volume_l": volume, "pred_colour": item.get("color"),
    }


def _get(session: Any, url: str, timeout: float) -> dict[str, Any] | None:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001 - metadata is best effort, recorded as None
        return None


def _find(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if payload.get(key) is not None:
            return payload[key]
        for value in payload.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=3, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def summarise(rows: list[dict[str, Any]], camera: str | None = None) -> dict[str, Any]:
    measured = [r for r in rows if not r["warmup"] and (camera is None or r["camera_id"] == camera)]
    done = [r for r in measured if r["status"] == "ok"]
    e2e = [r["e2e_ms"] for r in done]
    inference = [r["server_inference_ms"] for r in done if r["server_inference_ms"] is not None]
    server = {key: [r[key] for r in done if r.get(key) is not None]
              for key in ("server_latency_ms", "server_queue_ms", "server_processing_ms")}
    wall = sum(e2e) / 1000.0
    return {
        "camera_id": camera or "all",
        "sent": len(measured), "completed": len(done),
        "completed_pct": round(100.0 * len(done) / len(measured), 2) if measured else None,
        "timeouts": sum(r["status"] == "timeout" for r in measured),
        "failed": sum(r["status"] not in {"ok", "timeout"} for r in measured),
        "latency_e2e": {"boundary": "client send -> analysis result received (one client monotonic clock; "
                                    "includes upload, server processing and reply transport)",
                        "p50_ms": _percentile(e2e, 0.5), "p95_ms": _percentile(e2e, 0.95), "n": len(e2e)},
        "server_inference": {"boundary": "server-reported inference_ms",
                             "p50_ms": _percentile(inference, 0.5), "p95_ms": _percentile(inference, 0.95),
                             "n": len(inference)},
        "server_latency": {"boundary": "server receipt -> result ready (server monotonic clock)",
                           "p50_ms": _percentile(server["server_latency_ms"], 0.5),
                           "p95_ms": _percentile(server["server_latency_ms"], 0.95), "n": len(server["server_latency_ms"])},
        "queue_wait": {"boundary": "server receipt -> processing start (0 on the synchronous replay path)",
                       "p50_ms": _percentile(server["server_queue_ms"], 0.5),
                       "p95_ms": _percentile(server["server_queue_ms"], 0.95), "n": len(server["server_queue_ms"])},
        "processing": {"boundary": "processing start -> result ready (detection + depth + measurement)",
                       "p50_ms": _percentile(server["server_processing_ms"], 0.5),
                       "p95_ms": _percentile(server["server_processing_ms"], 0.95),
                       "n": len(server["server_processing_ms"])},
        "uncorrelated": sum(r["status"] == "uncorrelated" for r in measured),
        "throughput_fps": round(len(done) / wall, 3) if wall > 0 else None,
        "throughput_definition": "completed unique frames / summed e2e time (one request at a time)",
        "bytes_in_per_frame": round(statistics.mean(r["bytes_in"] for r in done), 1) if done else None,
        "bytes_out_per_frame": round(statistics.mean(r["bytes_out"] for r in done), 1) if done else None,
    }


def quality(rows: list[dict[str, Any]], truth: dict[str, dict[str, str]] | None) -> dict[str, Any]:
    if not truth:
        return {"status": "Not evaluated", "reason": "no ground truth supplied"}
    trials = [r for r in rows if not r["warmup"] and r["frame_file"] in truth]
    if not trials:
        return {"status": "Not evaluated", "reason": "no measured frame matches the ground-truth file"}
    detected = [r for r in trials if r["status"] == "ok" and r["detections"]]
    colour = [r for r in detected if truth[r["frame_file"]].get("colour") and r["pred_colour"]]
    dim_abs, dim_pct, vol_abs, vol_pct, dim_missing, vol_missing = [], [], [], [], 0, 0
    for r in detected:
        t = truth[r["frame_file"]]
        try:
            true_dims = sorted(float(t[k]) for k in ("length_mm", "width_mm", "height_mm"))
        except (KeyError, ValueError):
            true_dims = None
        pred_dims = [r[k] for k in ("pred_length_mm", "pred_width_mm", "pred_height_mm")]
        if true_dims and all(v is not None for v in pred_dims):
            for p, a in zip(sorted(float(v) for v in pred_dims), true_dims):
                dim_abs.append(abs(p - a))
                dim_pct.append(100.0 * abs(p - a) / a)
        elif true_dims:
            dim_missing += 1
        if t.get("volume_l"):
            if r["pred_volume_l"] is None:
                vol_missing += 1
            else:
                a = float(t["volume_l"])
                vol_abs.append(abs(float(r["pred_volume_l"]) - a))
                vol_pct.append(100.0 * vol_abs[-1] / a)
    median = lambda v: round(statistics.median(v), 3) if v else None  # noqa: E731
    return {
        "status": "evaluated",
        "trials_with_truth": len(trials),
        "detection_success": f"{len(detected)}/{len(trials)}",
        "missing_predictions": len(trials) - len(detected),
        "colour_correct": f"{sum(r['pred_colour'].lower() == truth[r['frame_file']]['colour'].lower() for r in colour)}/{len(colour)}",
        "dimension_abs_error_mm_median": median(dim_abs), "dimension_pct_error_median": median(dim_pct),
        "dimension_values": len(dim_abs), "dimension_missing_trials": dim_missing,
        "volume_abs_error_l_median": median(vol_abs), "volume_pct_error_median": median(vol_pct),
        "volume_trials": len(vol_abs), "volume_missing_trials": vol_missing,
    }


def run(args: argparse.Namespace) -> int:
    import requests

    frames = load_frames(Path(args.frames), args.camera)
    if not frames:
        print(f"No recorded frames (*.json + *.jpg) in {args.frames}", file=sys.stderr)
        return 2
    session = requests.Session()
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"
    base = args.url.rstrip("/")
    telemetry = _get(session, base + "/api/telemetry", 10)
    if telemetry is None:
        print(f"Backend not reachable at {base}/api/telemetry (stage: connection). "
              "Start the pipeline in that mode first.", file=sys.stderr)
        return 2
    server_mode = telemetry.get("processing_mode")
    if server_mode != args.expect_mode:
        print(f"Server reports processing mode '{server_mode}', not '{args.expect_mode}'. "
              "Refusing to record it under the wrong mode.", file=sys.stderr)
        return 3
    state = _get(session, base + "/api/state", 10) or {}
    health = _get(session, base + "/health", 10) or {}
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{server_mode}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    truth = None
    if args.truth:
        with open(args.truth, newline="", encoding="utf-8") as handle:
            truth = {row["frame"]: row for row in csv.DictReader(handle)}

    input_id = input_identity(frames, args.repeat)
    rows: list[dict[str, Any]] = []
    started_wall = time.time()
    for index, frame in enumerate(frames * args.repeat):
        meta = frame["meta"]
        camera = meta.get("camera_id", "realsense")
        frame_id = f"{run_id}-{camera}-{index:06d}"
        metadata = {"source": meta.get("source", "replay"), "intrinsics": meta.get("intrinsics"),
                    "camera_id": camera, "intrinsics_origin": meta.get("intrinsics_origin", ""),
                    "frame_id": frame_id, "run_id": run_id, "seq": index + 1, "timestamp": time.time(),
                    "input_id": input_id}
        files = {"image": ("frame.jpg", frame["image"], "image/jpeg")}
        if frame["depth"] is not None:
            files["depth"] = ("depth.npz", frame["depth"], "application/octet-stream")
        row: dict[str, Any] = {"run_id": run_id, "frame_file": frame["stem"], "camera_id": camera,
                               "frame_id": frame_id, "warmup": index < args.warmup, "detections": 0,
                               "server_processing_mode": None, "server_inference_ms": None,
                               "bytes_in": 0, "bytes_out": 0, "e2e_ms": None, "error": ""}
        t0 = time.perf_counter()
        try:
            response = session.post(f"{base}/api/cameras/{camera}/ingest?sync=1", files=files,
                                    data={"metadata": json.dumps(metadata)}, timeout=args.timeout)
            elapsed = (time.perf_counter() - t0) * 1000.0
            body = response.request.body
            row["bytes_in"] = len(body) if isinstance(body, (bytes, bytearray)) else 0
            row["bytes_out"] = len(response.content)
            result = response.json()
            if response.status_code != 200:
                row.update(status=f"http_{response.status_code}", error=str(result.get("error", ""))[:200])
            else:
                row.update(status="ok", e2e_ms=round(elapsed, 3),
                           server_processing_mode=result.get("processing_mode"),
                           server_inference_ms=result.get("inference_ms"),
                           detections=len(result.get("detections") or []), **prediction(result))
                timing = result.get("server_timing") or {}
                row.update(server_latency_ms=timing.get("server_latency_ms"), server_queue_ms=timing.get("queue_ms"),
                           server_processing_ms=timing.get("processing_ms"))
                if result.get("frame_id") != frame_id:
                    # A reply that does not carry the frame id it answers cannot be correlated.
                    row.update(status="uncorrelated", error=f"reply frame_id {result.get('frame_id')!r}")
                if result.get("processing_mode") != args.expect_mode:
                    row.update(status="mode_mismatch", error=f"server said {result.get('processing_mode')}")
        except requests.Timeout:
            row.update(status="timeout", error=f"no result within {args.timeout}s")
        except (requests.RequestException, ValueError) as exc:
            row.update(status="error", error=str(exc)[:200])
        rows.append(row)
        print(f"{index + 1}/{len(frames) * args.repeat} {camera} {row['status']} "
              f"{row['e2e_ms'] if row['e2e_ms'] is not None else '-'} ms", flush=True)

    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    host = telemetry.get("host") or {}
    cameras = sorted({r["camera_id"] for r in rows})
    summary = {
        "metadata": {
            "run_id": run_id, "requested_mode": args.expect_mode, "server_processing_mode": server_mode,
            "started_wall": started_wall, "duration_s": round(time.time() - started_wall, 2),
            "frames_dir": str(Path(args.frames).resolve()), "unique_frames": len(frames),
            "repeat": args.repeat, "warmup_frames": args.warmup,
            "measured_frames": sum(not r["warmup"] for r in rows),
            "resolutions": sorted({f"{(f['meta'].get('intrinsics') or {}).get('width')}x"
                                   f"{(f['meta'].get('intrinsics') or {}).get('height')}" for f in frames}),
            "client_code_commit": _git_commit(), "server_code_commit": host.get("code_commit"),
            "same_commit": (None if not host.get("code_commit") or not _git_commit()
                            else host.get("code_commit") == _git_commit()),
            "server_host": host, "server_gpu": telemetry.get("gpu"),
            "server_version": health.get("version"),
            "detector_model": _find(state, "detector_model"), "depth_model": _find(state, "depth_model"),
            "runtime": _find(state, "runtime"),
            "client_host": {"hostname": socket.gethostname(), "platform": platform.platform()},
            "network": args.network, "url": base, "timeout_s": args.timeout, "input_id": input_id,
            "superseded_note": "the replay posts synchronously (?sync=1), so no frame is superseded",
        },
        "all": summarise(rows),
        "cameras": {camera: summarise(rows, camera) for camera in cameras},
        "quality": quality(rows, truth),
        "quality_by_camera": {camera: quality([r for r in rows if r["camera_id"] == camera], truth)
                              for camera in cameras},
    }
    summary.update(latency_e2e=summary["all"]["latency_e2e"], throughput_fps=summary["all"]["throughput_fps"])
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_run_summary_csv(out / "run_summary.csv", summary)
    print(f"Wrote {out / 'frames.csv'} and {out / 'run_summary.json'}")
    return 0 if any(r["status"] == "ok" for r in rows) else 1


RUN_SUMMARY_COLUMNS = [
    "run_id", "processing_mode", "camera_id", "input_id", "server_host", "server_gpu", "detector_model",
    "depth_model", "server_code_commit", "measured_frames", "warmup_frames", "run_start_wall", "duration_s",
    "sent", "completed", "timeouts", "failed", "uncorrelated", "superseded",
    "e2e_p50_ms", "e2e_p95_ms", "e2e_n", "server_latency_p50_ms", "server_latency_p95_ms",
    "queue_p50_ms", "queue_p95_ms", "processing_p50_ms", "processing_p95_ms", "server_n",
    "edge_upload_rtt_p50_ms", "throughput_fps", "na_reasons",
]


def write_run_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    """One row per camera and one for all cameras, with N/A reasons instead of blanks."""
    meta = summary["metadata"]
    rows = []
    for camera, s in [("all", summary["all"]), *summary["cameras"].items()]:
        rows.append({
            "run_id": meta["run_id"], "processing_mode": meta["server_processing_mode"], "camera_id": camera,
            "input_id": meta.get("input_id"), "server_host": (meta.get("server_host") or {}).get("hostname"),
            "server_gpu": (meta.get("server_gpu") or {}).get("name"), "detector_model": meta.get("detector_model"),
            "depth_model": meta.get("depth_model"), "server_code_commit": meta.get("server_code_commit"),
            "measured_frames": s["sent"], "warmup_frames": meta["warmup_frames"],
            "run_start_wall": meta["started_wall"], "duration_s": meta["duration_s"],
            "sent": s["sent"], "completed": s["completed"], "timeouts": s["timeouts"], "failed": s["failed"],
            "uncorrelated": s["uncorrelated"], "superseded": 0,
            "e2e_p50_ms": s["latency_e2e"]["p50_ms"], "e2e_p95_ms": s["latency_e2e"]["p95_ms"],
            "e2e_n": s["latency_e2e"]["n"], "server_latency_p50_ms": s["server_latency"]["p50_ms"],
            "server_latency_p95_ms": s["server_latency"]["p95_ms"], "queue_p50_ms": s["queue_wait"]["p50_ms"],
            "queue_p95_ms": s["queue_wait"]["p95_ms"], "processing_p50_ms": s["processing"]["p50_ms"],
            "processing_p95_ms": s["processing"]["p95_ms"], "server_n": s["server_latency"]["n"],
            "edge_upload_rtt_p50_ms": None, "throughput_fps": s["throughput_fps"],
            "na_reasons": json.dumps({
                "edge_upload_rtt": "replay runs from this client, not the Pi edge sender",
                "superseded": "0 by design: synchronous replay does not supersede frames",
                "capture_to_display": "not measured: no shared clock or frame correlation with the browser",
                **({"server_timing": "server returned no per-frame timing (older server)"}
                   if not s["server_latency"]["n"] and s["completed"] else {}),
            }),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _pct(fraction: Any) -> float | None:
    """'7/10' -> 70.0; anything else -> None."""
    try:
        done, total = (float(v) for v in str(fraction).split("/"))
        return round(100.0 * done / total, 1) if total else None
    except ValueError:
        return None


def cost_vs_accuracy(local: dict[str, Any], cloud: dict[str, Any], local_rate: float | None,
                     cloud_rate: float | None, currency: str) -> list[dict[str, Any]]:
    """Cost and accuracy side by side, from the SAME replay. Cost = hourly rate / measured
    throughput; accuracy only where ground truth was supplied. Missing inputs stay N/A."""
    rows = []

    def per_1000(rate, summary):
        fps = summary["all"].get("throughput_fps")
        return None if rate is None or not fps else round(rate / (float(fps) * 3600.0) * 1000.0, 4)

    def acc(summary, key):
        q = summary["quality"]
        if q.get("status") != "evaluated":
            return None
        return _pct(q.get(key)) if key in ("detection_success", "colour_correct") else q.get(key)

    lc, cc = per_1000(local_rate, local), per_1000(cloud_rate, cloud)
    for metric, a, b in [
        (f"rate per running hour ({currency})", local_rate, cloud_rate),
        (f"cost per 1000 frames ({currency})", lc, cc),
        ("detection success %", acc(local, "detection_success"), acc(cloud, "detection_success")),
        ("colour correct %", acc(local, "colour_correct"), acc(cloud, "colour_correct")),
        ("volume error % (median, lower is better)", acc(local, "volume_pct_error_median"),
         acc(cloud, "volume_pct_error_median")),
        ("dimension error % (median, lower is better)", acc(local, "dimension_pct_error_median"),
         acc(cloud, "dimension_pct_error_median")),
    ]:
        rows.append({"metric": metric, "local": "N/A" if a is None else a, "cloud": "N/A" if b is None else b})
    vl, vc = acc(local, "volume_pct_error_median"), acc(cloud, "volume_pct_error_median")
    if None not in (lc, cc, vl, vc) and vl != vc:
        gain = vl - vc                                    # percentage points of volume error removed by cloud
        rows.append({"metric": f"extra cost per 1000 frames per volume-error point gained ({currency})",
                     "local": "-", "cloud": round((cc - lc) / gain, 4) if gain > 0 else "cloud is not more accurate"})
    else:
        rows.append({"metric": f"extra cost per 1000 frames per volume-error point gained ({currency})",
                     "local": "-", "cloud": "N/A (needs both rates and --truth on both runs)"})
    return rows


def load_cloud_rate(path: str | None, shape: str | None, spot: bool) -> tuple[float | None, str, str]:
    """(rate per hour, currency, provenance) from a filled cloud_rates.json; never a built-in price."""
    if not path:
        return None, "USD", "no rates file (--rates); cost N/A"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = (data.get("shapes") or {}).get(shape or "", {})
    rate = entry.get("spot_per_hour" if spot else "on_demand_per_hour")
    if rate is None:
        return None, data.get("currency", "USD"), f"no {'spot' if spot else 'on-demand'} rate for {shape!r} in {path}"
    verified = bool(data.get("source")) and bool(data.get("as_of"))
    note = (f"{shape}, {'spot' if spot else 'on-demand'}, zone {data.get('zone') or '?'}, "
            f"source {data.get('source') or '-'}, as of {data.get('as_of') or '-'}")
    return float(rate), data.get("currency", "USD"), ("" if verified else "UNVERIFIED: ") + note


def cost_accuracy_svg(rows: list[dict[str, Any]], currency: str) -> str:
    """Cost per 1000 frames (x) vs median volume error % (y), one point per mode; no point is drawn
    for a mode without BOTH a measured cost and a ground-truth error (nothing is invented)."""
    get = {r["metric"]: r for r in rows}
    cost = get.get(f"cost per 1000 frames ({currency})", {})
    err = get.get("volume error % (median, lower is better)", {})
    pts = [(mode, cost.get(mode), err.get(mode)) for mode in ("local", "cloud")]
    pts = [(m, float(c), float(e)) for m, c, e in pts if isinstance(c, (int, float)) and isinstance(e, (int, float))]
    w, h, l, r_, t, b = 560, 380, 70, 30, 40, 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           'font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{l}" y="22" font-size="14" font-weight="bold">Cost vs accuracy (same replay)</text>',
           f'<line x1="{l}" y1="{h - b}" x2="{w - r_}" y2="{h - b}" stroke="#333"/>',
           f'<line x1="{l}" y1="{t}" x2="{l}" y2="{h - b}" stroke="#333"/>',
           f'<text x="{(l + w - r_) / 2}" y="{h - 16}" text-anchor="middle">cost per 1000 frames ({currency})</text>',
           f'<text x="18" y="{(t + h - b) / 2}" text-anchor="middle" transform="rotate(-90 18 {(t + h - b) / 2})">'
           'median volume error % (lower is better)</text>']
    if not pts:
        out.append(f'<text x="{(l + w - r_) / 2}" y="{(t + h - b) / 2}" text-anchor="middle" fill="#a33">'
                   'Not plotted: needs a rate (--rates) and ground truth (--truth) for a mode</text>')
    else:
        xmax = max(max(c for _, c, _ in pts) * 1.25, 1e-6)
        ymax = max(max(e for _, _, e in pts) * 1.25, 1.0)
        for i in range(5):                                       # gridlines with values
            yv, xv = ymax * i / 4, xmax * i / 4
            y = h - b - (h - t - b) * i / 4
            x = l + (w - l - r_) * i / 4
            out.append(f'<line x1="{l}" y1="{y:.1f}" x2="{w - r_}" y2="{y:.1f}" stroke="#ddd"/>'
                       f'<text x="{l - 6}" y="{y + 4:.1f}" text-anchor="end">{yv:.1f}</text>'
                       f'<text x="{x:.1f}" y="{h - b + 18}" text-anchor="middle">{xv:.3f}</text>')
        for mode, c, e in pts:
            x = l + (w - l - r_) * c / xmax
            y = h - b - (h - t - b) * e / ymax
            colour = "#1f6feb" if mode == "local" else "#d1242f"
            right = x > l + 0.55 * (w - l - r_)                   # keep the label inside the plot
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}"/>'
                       f'<text x="{x - 9 if right else x + 9:.1f}" y="{y - 10:.1f}" '
                       f'text-anchor="{"end" if right else "start"}">{mode}: {c:.4f} {currency}, {e:.1f} %</text>')
    out.append("</svg>")
    return "\n".join(out)


def compare(args: argparse.Namespace) -> int:
    runs = [json.loads((Path(p) / "run_summary.json").read_text(encoding="utf-8")) for p in args.runs]
    by_mode = {r["metadata"]["server_processing_mode"]: r for r in runs}
    if set(by_mode) != {"local", "cloud"}:
        print("compare needs one local and one cloud run", file=sys.stderr)
        return 2
    local, cloud = by_mode["local"], by_mode["cloud"]
    warnings = []
    for key in ("input_id", "unique_frames", "warmup_frames", "repeat", "detector_model", "depth_model"):
        if local["metadata"].get(key) != cloud["metadata"].get(key):
            warnings.append(f"{key} differs: local={local['metadata'].get(key)} cloud={cloud['metadata'].get(key)}")
    if local["metadata"].get("server_code_commit") != cloud["metadata"].get("server_code_commit"):
        warnings.append("server code commit differs or is unknown on one side")
    lines, table = [], []
    for scope in ["all", *sorted(set(local["cameras"]) | set(cloud["cameras"]))]:
        a = local["all"] if scope == "all" else local["cameras"].get(scope, {})
        b = cloud["all"] if scope == "all" else cloud["cameras"].get(scope, {})
        for label, fn in [
            ("latency e2e p50 ms", lambda s: (s.get("latency_e2e") or {}).get("p50_ms")),
            ("latency e2e p95 ms", lambda s: (s.get("latency_e2e") or {}).get("p95_ms")),
            ("server inference p50 ms", lambda s: (s.get("server_inference") or {}).get("p50_ms")),
            ("server-side result latency p50 ms", lambda s: (s.get("server_latency") or {}).get("p50_ms")),
            ("queue wait p50 ms", lambda s: (s.get("queue_wait") or {}).get("p50_ms")),
            ("processing p50 ms", lambda s: (s.get("processing") or {}).get("p50_ms")),
            ("throughput fps", lambda s: s.get("throughput_fps")),
            ("completed/sent %", lambda s: s.get("completed_pct")),
            ("completed / sent", lambda s: f"{s.get('completed')}/{s.get('sent')}" if s else None),
            ("timeouts", lambda s: s.get("timeouts")),
            ("bytes in per frame", lambda s: s.get("bytes_in_per_frame")),
            ("bytes out per frame", lambda s: s.get("bytes_out_per_frame")),
        ]:
            table.append({"scope": scope, "metric": label, "local": fn(a), "cloud": fn(b),
                          "matched": not warnings, "match_notes": "; ".join(warnings)})
    for key in ("detection_success", "colour_correct", "dimension_pct_error_median", "volume_pct_error_median"):
        table.append({"scope": "quality", "metric": key, "local": local["quality"].get(key, local["quality"]["status"]),
                      "cloud": cloud["quality"].get(key, cloud["quality"]["status"]),
                      "matched": not warnings, "match_notes": "; ".join(warnings)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "local_vs_cloud_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "local", "cloud", "matched", "match_notes"])
        writer.writeheader()
        writer.writerows(table)
    lines.append("**MATCHED comparison**: same recorded input, models, warm-up and repeat." if not warnings else
                 "**NOT MATCHED**: differences below; do not draw a better/worse conclusion.")
    lines.append("| Scope | Metric | Local | Cloud |\n|---|---|---|---|")
    lines += [f"| {r['scope']} | {r['metric']} | {r['local']} | {r['cloud']} |" for r in table]
    lines.append("\nLatency boundary: client send -> analysis result received (client monotonic clock).")
    lines.append(f"Local run {local['metadata']['run_id']}, cloud run {cloud['metadata']['run_id']}; "
                 f"network: {cloud['metadata'].get('network')}; cloud GPU: {(cloud['metadata'].get('server_gpu') or {}).get('name')}.")
    lines += [f"WARNING: {w}" for w in warnings]
    cloud_rate, currency, provenance = load_cloud_rate(args.rates, args.shape, args.spot)
    if cloud_rate is None and args.cloud_cost_per_hour is not None:
        cloud_rate, currency, provenance = args.cloud_cost_per_hour, args.currency, "UNVERIFIED: --cloud-cost-per-hour"
    if cloud_rate is None and os.environ.get("LOCALLIFE_CLOUD_COST_PER_HOUR", "").strip():
        cloud_rate, currency = float(os.environ["LOCALLIFE_CLOUD_COST_PER_HOUR"]), args.currency
        provenance = "UNVERIFIED: LOCALLIFE_CLOUD_COST_PER_HOUR"
    cva = cost_vs_accuracy(local, cloud, args.local_cost_per_hour, cloud_rate, currency)
    # Separate section for the professor's cost-vs-accuracy question: its own file and chart.
    (out / "cost_vs_accuracy.svg").write_text(cost_accuracy_svg(cva, currency), encoding="utf-8")
    section = ["# Cost vs accuracy: local laptop vs cloud GPU", "",
               f"Same recorded frames replayed in both modes. Cloud rate: {provenance}.",
               "Cost = rate per running hour / measured throughput; excludes the stopped-VM disk, images, "
               "egress and laptop power unless --local-cost-per-hour is given. Accuracy only from --truth.", "",
               "| Metric | Local | Cloud |", "|---|---|---|"]
    section += [f"| {r['metric']} | {r['local']} | {r['cloud']} |" for r in cva]
    section += ["", "![cost vs accuracy](cost_vs_accuracy.svg)"]
    if warnings:
        section += ["", "NOT MATCHED runs -- no better/worse conclusion: " + "; ".join(warnings)]
    (out / "cost_vs_accuracy.md").write_text("\n".join(section) + "\n", encoding="utf-8")
    with (out / "cost_vs_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "local", "cloud"])
        writer.writeheader()
        writer.writerows(cva)
    lines.append("\n**Cost vs accuracy** (same replay; cost = rate / measured throughput, running time only -- "
                 "excludes stopped-VM disk, images, egress; accuracy only with --truth)")
    lines.append("| Metric | Local | Cloud |\n|---|---|---|")
    lines += [f"| {r['metric']} | {r['local']} | {r['cloud']} |" for r in cva]
    (out / "local_vs_cloud_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--frames", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--expect-mode", required=True, choices=["local", "cloud"])
    r.add_argument("--camera", choices=["realsense", "logitech"])
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--truth")
    r.add_argument("--network", default="unspecified")
    r.add_argument("--token", default="")
    r.add_argument("--out", default="artifacts/benchmark")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs=2)
    c.add_argument("--out", default="artifacts/benchmark")
    c.add_argument("--cloud-cost-per-hour", type=float, default=None,
                   help="VM running rate (default: LOCALLIFE_CLOUD_COST_PER_HOUR); N/A when unset")
    c.add_argument("--local-cost-per-hour", type=float, default=None,
                   help="optional laptop running cost (power); N/A when unset")
    c.add_argument("--currency", default="USD")
    c.add_argument("--rates", help="filled cloud_rates.json (see scripts/cloud_rates.example.json)")
    c.add_argument("--shape", default="g2-standard-4+nvidia-l4",
                   help="key in the rates file for the VM the cloud run used (gpu.py status)")
    c.add_argument("--spot", action="store_true", help="the cloud run used a Spot VM")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
