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
    "e2e_ms", "server_inference_ms", "bytes_in", "bytes_out", "detections",
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
    wall = sum(e2e) / 1000.0
    return {
        "camera_id": camera or "all",
        "sent": len(measured), "completed": len(done),
        "completed_pct": round(100.0 * len(done) / len(measured), 2) if measured else None,
        "timeouts": sum(r["status"] == "timeout" for r in measured),
        "failed": sum(r["status"] not in {"ok", "timeout"} for r in measured),
        "latency_e2e": {"boundary": "client send -> analysis result received (client monotonic clock)",
                        "p50_ms": _percentile(e2e, 0.5), "p95_ms": _percentile(e2e, 0.95), "n": len(e2e)},
        "server_inference": {"boundary": "server-reported inference_ms",
                             "p50_ms": _percentile(inference, 0.5), "p95_ms": _percentile(inference, 0.95),
                             "n": len(inference)},
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

    rows: list[dict[str, Any]] = []
    started_wall = time.time()
    for index, frame in enumerate(frames * args.repeat):
        meta = frame["meta"]
        camera = meta.get("camera_id", "realsense")
        frame_id = f"{run_id}-{camera}-{index:06d}"
        metadata = {"source": meta.get("source", "replay"), "intrinsics": meta.get("intrinsics"),
                    "camera_id": camera, "intrinsics_origin": meta.get("intrinsics_origin", ""),
                    "frame_id": frame_id, "run_id": run_id, "seq": index + 1, "timestamp": time.time()}
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
            "network": args.network, "url": base, "timeout_s": args.timeout,
        },
        "all": summarise(rows),
        "cameras": {camera: summarise(rows, camera) for camera in cameras},
        "quality": quality(rows, truth),
        "quality_by_camera": {camera: quality([r for r in rows if r["camera_id"] == camera], truth)
                              for camera in cameras},
    }
    summary.update(latency_e2e=summary["all"]["latency_e2e"], throughput_fps=summary["all"]["throughput_fps"])
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out / 'frames.csv'} and {out / 'run_summary.json'}")
    return 0 if any(r["status"] == "ok" for r in rows) else 1


def compare(args: argparse.Namespace) -> int:
    runs = [json.loads((Path(p) / "run_summary.json").read_text(encoding="utf-8")) for p in args.runs]
    by_mode = {r["metadata"]["server_processing_mode"]: r for r in runs}
    if set(by_mode) != {"local", "cloud"}:
        print("compare needs one local and one cloud run", file=sys.stderr)
        return 2
    local, cloud = by_mode["local"], by_mode["cloud"]
    warnings = []
    for key in ("frames_dir", "unique_frames", "warmup_frames", "repeat", "detector_model", "depth_model"):
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
            ("throughput fps", lambda s: s.get("throughput_fps")),
            ("completed/sent %", lambda s: s.get("completed_pct")),
            ("completed / sent", lambda s: f"{s.get('completed')}/{s.get('sent')}" if s else None),
            ("timeouts", lambda s: s.get("timeouts")),
            ("bytes in per frame", lambda s: s.get("bytes_in_per_frame")),
            ("bytes out per frame", lambda s: s.get("bytes_out_per_frame")),
        ]:
            table.append({"scope": scope, "metric": label, "local": fn(a), "cloud": fn(b)})
    for key in ("detection_success", "colour_correct", "dimension_pct_error_median", "volume_pct_error_median"):
        table.append({"scope": "quality", "metric": key, "local": local["quality"].get(key, local["quality"]["status"]),
                      "cloud": cloud["quality"].get(key, cloud["quality"]["status"])})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "local_vs_cloud_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "local", "cloud"])
        writer.writeheader()
        writer.writerows(table)
    lines.append("| Scope | Metric | Local | Cloud |\n|---|---|---|---|")
    lines += [f"| {r['scope']} | {r['metric']} | {r['local']} | {r['cloud']} |" for r in table]
    lines.append("\nLatency boundary: client send -> analysis result received (client monotonic clock).")
    lines.append(f"Local run {local['metadata']['run_id']}, cloud run {cloud['metadata']['run_id']}; "
                 f"network: {cloud['metadata'].get('network')}; cloud GPU: {(cloud['metadata'].get('server_gpu') or {}).get('name')}.")
    lines += [f"WARNING: {w}" for w in warnings]
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
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
