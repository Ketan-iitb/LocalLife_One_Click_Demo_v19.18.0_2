# v42 — Local vs Cloud comparison on http://127.0.0.1:8000

Branched from `Working_branch_v41_cloud_local_benchmark` @ `d3c92c6`. No change to processing,
cameras, transport, models, deployment, gpu.py, tunnel or mode selection.

**Where:** directly below the RealSense and Logitech live streams on `/` (operator) and `/research`
(Research Mode). Both render the same `/api/local-cloud/view`.

**Storage** (`<LOCALLIFE_RESULTS_DIR>/local_cloud_comparison/`, default `artifacts/` next to where
the server starts; on the VM `/home/HP/LocalLife_Plug_and_Play_Local/artifacts/...`):
- `observations.jsonl` — append-only raw events (every finished frame, resource samples every 5 s,
  one `run_start` per server start); key run_id+camera_id+frame_id+event, stored once; fsync per line;
  a line cut by a crash is skipped on reload.
- `imported_summaries.jsonl` — the other mode's run summaries, with source host and run id.
- `run_summaries/<run_id>.json` — this run's calculated summaries, replaced atomically every 30 s and on exit.

**Local ↔ cloud sync:** both servers are reached at the same browser origin (127.0.0.1:8000), so the
panel keeps each mode's run summaries in the browser and imports the other mode's into the server it
is showing (automatic, attributed). In a different browser: "Download run summary CSV" in one mode,
then "Import other mode's run summary CSV" in the other. Until then the panel says
"Other mode's data not imported".

**Downloads:** `/api/local-cloud/raw.csv` (all runs; `?run_id=`, `?since=<unix s>`),
`/api/local-cloud/summaries.csv` (`?run_id=`), `/api/local-cloud/comparison.csv?local_run=&cloud_run=`.
Each response carries `X-Row-Count`.

**Boundaries:** latency = server receipt → result ready; inference = inference start → result;
queue = receipt → inference start (all server monotonic clock); upload = edge send → HTTP 202 on the
edge clock (needs the v41 edge client on the Pi); startup = server start → first result per camera
(GPU provisioning not included); recovery = last result before an outage (edge reconnect or >10 s
gap) → next result. Accuracy metrics: "Not evaluated" without labelled ground truth.

## Measurement definitions (telemetry revision)

| Row | What is measured | Clock |
|---|---|---|
| Server-side result latency (column keys `latency_p50_ms/p95`) | server receipt → result ready. **Not end-to-end**: excludes upload and reply transport | server monotonic |
| Batch processing time (`inference_*`) | batch start → result ready: detection + depth + measurement for the camera batch, not model forward time alone | server monotonic |
| Queue waiting time | server receipt → batch start | server monotonic |
| Edge upload request RTT (`upload_*`) | edge send → HTTP 202 received; includes server enqueue; not one-way latency. N/A when the Pi runs a pre-v41 edge client (the launcher copies the project to the Pi only when it is missing) | one edge monotonic clock |
| Camera capture → dashboard-visible result | **N/A**: Pi and browser share no clock and displayed results are not correlated to frame ids | — |
| Superseded | frames replaced in the latest-frame queue by a newer frame (policy, not failure); % of frames received | — |
| Failed (8a) / Lost (8b) | server-side errors / edge sequence gaps; zero recorded losses is not proof of zero losses when sequence numbers are absent | — |
| Host CPU/RAM | % of the machine running the server (laptop vs VM): different machines, not comparable capacity | — |

Old CSV columns keep their names and order; new columns are appended (`boundary`, `matched`, `match_notes`,
`verdict`, `*_na_reason`, `*_window_s`; summaries: `frames_window_s`, `input_id`, `na_reasons` …).

**Matched comparisons only.** A better/worse verdict is shown only when both runs processed the same recorded
input (`input_id`, set by `scripts/benchmark_local_cloud.py`), with the same detector and depth model and a
similar frame count. Live camera runs are always "NOT MATCHED" and show numbers without a verdict. Trend
charts appear only when one mode has at least two runs. The benchmark writes `run_summary.csv` per run
(N/A reasons included) and `compare` marks the pair MATCHED / NOT MATCHED.
