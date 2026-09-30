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
