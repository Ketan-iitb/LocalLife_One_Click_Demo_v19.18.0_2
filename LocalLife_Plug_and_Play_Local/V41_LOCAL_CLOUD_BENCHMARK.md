# v41 — Local vs cloud telemetry and replay benchmark

Branched from V39 `46c81d4893334d68a0e175ea053d170c16f2463a`. Measurement code (detection, tracking, IDs,
geometry, volume, ledger, CSV) is unchanged.

## Frame path and what is measured

Pi `edge_client` → HTTP multipart `/api/ingest` → `server.py` (laptop in local mode; GPU VM behind the
launcher's SSH tunnel in cloud mode) → `LatestFrameProcessor` (newest frame per camera, older ones superseded)
→ `DualCameraCoordinator.process_packets` → dashboard.

| Metric | Boundary / definition |
|---|---|
| Live latency p50/p95 | server receipt → result ready, server monotonic clock (excludes network) |
| Edge upload RTT | Pi send → HTTP 202, Pi monotonic clock (upload + enqueue, excludes inference) |
| Replay e2e latency | client send → result received (`?sync=1`), one client monotonic clock |
| Throughput | unique completed frame ids / observed span (duplicates and polls not counted) |
| Reliability | completed / sent; sent = received + sequence gaps; superseded, failed, reconnects shown separately |
| Quality | replay only, only with `--truth`; otherwise "Not evaluated" |
| Resources | processing host CPU/RAM (/proc or psutil), GPU via nvidia-smi, Pi CPU/RAM from frame metadata; N/A when unreadable |
| Transfer | request bytes in, reply bytes out, per frame and per minute |
| Cost | estimate only, from `LOCALLIFE_CLOUD_COST_PER_HOUR`; N/A otherwise |

Capture-to-display latency is **not** reported (no shared clock between Pi and browser).
Every reply and telemetry row carries the server's own `processing_mode` (`CLOUD_ENABLED`, set by the launcher on the VM only).

## Commands (Windows laptop, repository root)

```
START_LOCAL_LIFE.cmd                     control page: http://127.0.0.1:8765/  (mode selector + comparison panel)
python gpu.py status                     VM state and zone (free; also the "Check VM" button)
python gpu.py up                         start/restart the VM (charges while running)
python gpu.py down                       sync home and stop the VM (disk ~7 kr/day)
```
Dashboard (either mode): http://127.0.0.1:8000/ — telemetry: `/api/telemetry`, `/api/telemetry.csv`, `/api/telemetry.json`.
Raw per-frame telemetry is also appended to `<LOCALLIFE_RESULTS_DIR>/telemetry/telemetry_<mode>_<start>.csv`
(on the VM in cloud mode; use the export link to copy it to the laptop).

Record frames for replay (on the Pi; the normal launcher command plus `--record-dir`):
```
python3 -m locallife_cloud.edge_client --cloud http://127.0.0.1:18000 --source dual --record-dir ~/phase1_frames
```
Replay (laptop, with the Pi edge client stopped; start the backend with a separate `LOCALLIFE_RESULTS_DIR`
because replayed frames enter that server's event log):
```
cd LocalLife_Plug_and_Play_Local
python scripts\benchmark_local_cloud.py run --frames phase1_frames --expect-mode local --warmup 5 --network "lab wifi" --out artifacts\benchmark
python scripts\benchmark_local_cloud.py run --frames phase1_frames --expect-mode cloud --warmup 5 --network "lab wifi" --out artifacts\benchmark
python scripts\benchmark_local_cloud.py compare artifacts\benchmark\<local_run> artifacts\benchmark\<cloud_run> --out artifacts\benchmark
```
Outputs: `frames.csv` + `run_summary.json` per run; `local_vs_cloud_summary.csv/.md` for the thesis.
