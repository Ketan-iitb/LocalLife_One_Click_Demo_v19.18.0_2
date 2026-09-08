# LocalLife Plug-and-Play Volume Experiment

> This is the isolated experimental cloud copy. It uses port 8100 and
> `~/LocalLife_Plug_and_Play_Data`; it does not replace the stable
> `~/LocalLife_Dual_Camera_Thesis` installation. Start with
> [PLUG_AND_PLAY_EXPERIMENT.md](PLUG_AND_PLAY_EXPERIMENT.md).

This edition compares **two genuinely separate physical cameras**: Intel
RealSense D435 measured hardware depth versus Logitech C920 RGB processed by
Depth Anything V2. Each camera owns a separate baseline, tracker, event ledger,
color totals, cumulative volume, calibration factor, and exportable database.
The dashboard places their live images, depth maps, detections, and measurements
side by side, and computes matched-object agreement and ground-truth accuracy.

Read [DUAL_CAMERA_THESIS.md](DUAL_CAMERA_THESIS.md) for deployment, dual-camera
Raspberry Pi streaming, independent calibration, RealSense precision controls,
known-volume trials, and thesis-ready exports.

The Logitech station also uses tightly bounded object-instance masks,
object-interior color recognition, independent background-depth stabilization,
a separate configurable measurement region, implausible-volume rejection, and
a backup-protected Logitech-only history reset.

## Existing waste-plant capabilities

This edition can monitor **garbage bags and cardboard boxes**, maintains persistent
observed/deposited counts, records cumulative volume and object color, and
automatically logs settled deposits. Read [PLANT_MONITOR.md](PLANT_MONITOR.md)
for operating instructions, color-to-waste-stream configuration, history export,
and the distinction between observed and confirmed deposited objects.

As of LiveFix 6, the shipped `cloud.env.example` defaults to
`LOCALLIFE_BAG_ONLY=false` with both bag and box/carton/parcel prompts, so
rigid cartons (the current thesis priority -- see
[LIVEFIX_CHANGELOG.md](LIVEFIX_CHANGELOG.md)) are detected and measured out of
the box. Set `LOCALLIFE_BAG_ONLY=true` yourself only if you deliberately want
the specialized fixed-bin bag-only station described in
[BAG_STATION.md](BAG_STATION.md). See
[MEASUREMENT_FIX_REPORT.md](MEASUREMENT_FIX_REPORT.md) for the repaired
live-count, volume-mask, frame-synchronisation, latency, color, and
VolPy-inspired triangulation details, and
[LIVEFIX_CHANGELOG.md](LIVEFIX_CHANGELOG.md) (LiveFix 6) for the table-relative
box-cuboid volume math audit, the multi-frame L/W/H aggregation, and the
dashboard box-geometry diagnostics.

Cloud-first replacement for the reconstructed Local Life V14 Raspberry Pi prototype. The Raspberry Pi or another computer remains responsible for the cameras; the Google Cloud NVIDIA L4 VM performs object detection, instance segmentation, monocular depth estimation, GPU training, batch evaluation, and experiment logging.

The original V14 files remain unchanged in [`legacy/`](legacy/).

## Why V14 failed

The original application does not load an object-detection model. Its "detection" is an empty-background subtraction that requires a previously captured baseline, discards every connected component except the largest, and is sensitive to lighting, camera motion, and its restrictive region of interest.

Other problems include a hard-coded `/home/locallife/LocalLife` output directory, an assumed USB camera index, suppressed RealSense exceptions, manual-only counting, and a separate Windows depth worker that always posts `volume_l: null`. A Google Cloud VM cannot access a USB camera physically attached to a Raspberry Pi.

V15 corrects these architectural problems instead of attempting to run the Pi-only application unchanged on the VM.

## Architecture

The Raspberry Pi streams **Intel RealSense RGB plus its aligned hardware depth**
and a **separate Logitech C920 RGB feed** through the cloud tunnel. The NVIDIA
L4 VM runs one shared YOLOE detector, applies Depth Anything V2 to Logitech
images only, and maintains independent trackers, baselines, measurements, event
ledgers, exports, and side-by-side dashboard panels.

The default detection model is `yoloe-11l-seg.pt`, prompted with garbage bags,
trash bags, refuse sacks, cardboard boxes, and related waste-container variants.
People, bottles, and unrelated generic classes are absent. For maximum
detection reliability, replace the generic prompted model with a `best.pt`
model fine-tuned on annotated overhead images from the actual installation.

After an empty-bin baseline is captured, aligned RealSense depth and RGB change detection recover the complete physical outline of each deposited bag. Partial neural bag detections expand to the entire depth-supported silhouette. Previously committed bags become part of the rolling reference so the next arriving bag can be measured separately.

If the installed YOLOE text encoder cannot initialize on the VM's Python version, the service automatically tries `yoloe-11l-seg-pf.pt`, the prompt-free open-vocabulary segmentation variant, and records the reason in its runtime status.

The default depth model is `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`, not the smaller laptop model used by V14. Both inference models use the NVIDIA L4 and FP16 where available.

## 1. Upload the project to the VM

Download and extract `LocalLife_Cloud_V15.zip` on your Windows laptop. In PowerShell:

```powershell
gcloud config set project locallife-thesis-depth

gcloud compute scp --recurse `
  "C:\path\to\LocalLife_Cloud_V15" `
  depth-l4:~/ `
  --zone=europe-west1-b

gcloud compute ssh depth-l4 --zone=europe-west1-b
```

Alternatively, upload the ZIP itself and run `unzip LocalLife_Cloud_V15.zip` after connecting to the VM.

## 2. Install cloud dependencies without replacing PyTorch/CUDA

On the VM:

```bash
cd ~/LocalLife_Cloud_V15
nvidia-smi
bash scripts/setup_cloud.sh
source .venv/bin/activate
python scripts/check_environment.py --require-cuda
```

The setup script creates a virtual environment with `--system-site-packages`. This preserves the VM's existing CUDA-enabled PyTorch 2.9 instead of accidentally installing a CPU-only replacement.

Download both models ahead of the first experiment if desired:

```bash
python scripts/check_environment.py --download-models --require-cuda
```

Optional local configuration:

```bash
cp cloud.env.example cloud.env
```

## 3. Start the cloud dashboard

On the VM:

```bash
cd ~/LocalLife_Cloud_V15
bash scripts/start_cloud.sh
```

The service deliberately binds to `127.0.0.1:8000`; do not expose an unauthenticated GPU service to the public internet.

In a separate PowerShell window on your laptop, create an SSH tunnel:

```powershell
gcloud compute ssh depth-l4 `
  --zone=europe-west1-b `
  -- -N -L 8000:127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000
```

## 4. Connect the physical RealSense camera

The VM cannot see USB devices connected to the Raspberry Pi. Run the edge bridge on whichever physical computer has the Intel RealSense D435/D435i attached.

Copy this project to that device. On a Raspberry Pi, preserve system-installed `pyrealsense2`:

```bash
cd ~/LocalLife_Cloud_V15
python3 -m venv --system-site-packages .venv-edge
source .venv-edge/bin/activate
pip install -r requirements-edge.txt
python -c "import pyrealsense2; print('RealSense available')"
```

If the Google Cloud CLI is configured on the Pi, establish a secure tunnel in one Pi terminal:

```bash
gcloud compute ssh depth-l4 \
  --project=locallife-thesis-depth \
  --zone=europe-west1-b \
  -- -N -L 8000:127.0.0.1:8000
```

In another Pi terminal:

```bash
cd ~/LocalLife_Cloud_V15
source .venv-edge/bin/activate

python -m locallife_cloud.edge_client \
  --cloud http://127.0.0.1:8000 \
  --source realsense \
  --upload-fps 4
```

The bridge aligns depth to the RealSense RGB frame before transmission and sends the correct color-camera intrinsics. The old V14 application must not own the same camera simultaneously.

For a regular USB webcam:

```bash
python -m locallife_cloud.edge_client \
  --cloud http://127.0.0.1:8000 \
  --source 0
```

A plain USB camera still provides detection, segmentation, tracking, and monocular depth, but calibrated liters require an aligned RealSense reference and camera intrinsics.

For an existing video:

```bash
python -m locallife_cloud.edge_client \
  --cloud http://127.0.0.1:8000 \
  --source ~/videos/waste-drop.mp4
```

If `gcloud` is unavailable on the Pi, use any SSH connection capable of forwarding local port 8000 to `127.0.0.1:8000` on the VM. A laptop on the same trusted local network can also forward a LAN-bound port for the Pi, but take care not to expose that port outside the trusted network.

## 5. Capture a valid measurement

1. Point the RealSense at the empty measurement area and allow exposure to stabilize.
2. Start the edge bridge and confirm the dashboard shows live frames.
3. Keep the scene empty and still until automatic setup reports ready.
4. Place an object completely within the region of interest.
5. Inspect its prompted class, confidence, segmentation mask, and track ID.
6. Compare RealSense liters and calibrated monocular liters.
7. Let the stable, validated measurement enter history automatically.

For accurate bag measurements, keep the overhead camera rigidly fixed and approximately perpendicular to the bin opening. Automatic setup captures the reference after at least nine empty, stable frames while the garbage receptacle is present but contains no bags. The live overlay displays only confirmed tracks and their estimated liters when valid depth is available.

The dashboard displays both the RealSense hardware depth map and the Depth Anything estimated depth map, together with valid-depth coverage, median distance, camera-intrinsics status, baseline readiness, and an explicit volume-readiness message. Each detected bag reports its camera distance, height above the rolling reference, estimated liters, and approximate depth-related uncertainty. The overall bin occupancy remains referenced to the original empty receptacle. Check the complete diagnostic response with:

```bash
curl http://127.0.0.1:8000/api/state
```

If `realsense_depth.available` is false, the Pi is sending RGB/video only; start the bridge with `--source realsense`. If depth is available but `volume_status.code` is `missing_empty_baseline`, clear and steady the scene while automatic setup runs. If `baseline_missing_depth` appears, keep the scene empty after the RealSense depth signal returns. Monocular results remain explicitly identified as AI estimates.

The automatic count increases only when an object persists long enough to become a confirmed track. A stationary object is not counted repeatedly, and a confirmed track remains visible through a short detector dropout.

Because this fixed installation is restricted to garbage bags, a new depth-supported silhouette inside the calibrated bin can still be measured if the neural detector misses it. Such detections are explicitly marked `garbage bag (depth silhouette)` with zero semantic confidence; they must not be described as neural recognition. Set `LOCALLIFE_ALLOW_UNCLASSIFIED=false` if confirmed neural bag recognition is mandatory.

Results are written to:

```text
artifacts/
  frames.jsonl
  events.jsonl
  validated_measurements.jsonl
  baselines/
  training/
  batch/
```

The service synchronizes them to `gs://locallife-thesis-depth-data/cloud-v15` approximately every three minutes.

## 6. Audit your existing YOLO dataset

Before training, upload your dataset to Cloud Storage or copy it onto the VM:

```bash
gcloud storage rsync -r \
  gs://locallife-thesis-depth-data/datasets/my-waste-data \
  ./datasets/my-waste-data

python -m locallife_cloud.dataset \
  ./datasets/my-waste-data/dataset.yaml \
  --output artifacts/dataset_audit.json
```

The audit catches missing train/validation images, missing label files, invalid normalized coordinates, unknown class IDs, empty annotations, mixed detection/segmentation label formats, and severe class imbalance.

Your previous `mAP50 ≈ 0.52`, precision `≈ 0.57`, and recall `≈ 0.46` should be treated as a baseline, not as production-ready detection. Inspect missed objects, annotation quality, class balance, lighting, and whether validation images are genuinely independent.

## 7. Train a stronger waste detector or segmenter

```bash
cd ~/LocalLife_Cloud_V15
source .venv/bin/activate

python -m locallife_cloud.train \
  --data ./datasets/my-waste-data/dataset.yaml \
  --name waste-v15-960 \
  --epochs 120 \
  --imgsz 960 \
  --batch -1 \
  --workers 2
```

Bounding-box annotations automatically select `yolo11m.pt`; polygon annotations select `yolo11m-seg.pt`. Segmentation labels are preferable for accurate volume boundaries. `--batch -1` lets Ultralytics choose a GPU-safe batch size, and two workers suit this VM's four vCPUs and 16 GB RAM.

Every epoch saves a checkpoint. An `on_model_save` callback uploads `last.pt`, `best.pt`, and training metrics to the Cloud Storage bucket so a Spot interruption does not erase completed epochs.

Resume interrupted training:

```bash
bash scripts/sync_bucket.sh pull

python -m locallife_cloud.train \
  --data ./datasets/my-waste-data/dataset.yaml \
  --name waste-v15-960 \
  --resume ./artifacts/training/waste-v15-960/weights/last.pt
```

Use your trained model for the live service:

```bash
export LOCALLIFE_DETECTOR_MODEL="$PWD/artifacts/training/waste-v15-960/weights/best.pt"
bash scripts/start_cloud.sh
```

## 8. Run batched thesis experiments

```bash
python -m locallife_cloud.batch \
  --input ./datasets/experiment/rgb \
  --depth-dir ./datasets/experiment/depth \
  --baseline-image ./datasets/experiment/empty.jpg \
  --baseline-depth ./datasets/experiment/empty-depth.npy \
  --intrinsics ./datasets/experiment/intrinsics.json \
  --batch-size 6 \
  --output ./artifacts/batch/measurements.csv
```

Depth files are matched by image filename stem and must contain meters. An intrinsics file looks like:

```json
{
  "fx": 615.0,
  "fy": 615.0,
  "ppx": 320.0,
  "ppy": 240.0,
  "width": 640,
  "height": 480
}
```

Add a `ground_truth_l` column to the generated CSV using measured reference volumes, then calculate thesis metrics:

```bash
python -m locallife_cloud.evaluate \
  ./artifacts/batch/measurements.csv \
  --output ./artifacts/batch/evaluation.json
```

The evaluator reports MAE, RMSE, bias, median absolute error, MAPE, Pearson correlation, and Bland–Altman limits for both hardware depth and calibrated monocular depth.

Measure actual GPU throughput and choose a safe batch size from your own frames:

```bash
python -m locallife_cloud.benchmark \
  --images ./datasets/experiment/rgb \
  --batch-sizes 1,2,4,6 \
  --repeats 3 \
  --output ./artifacts/benchmark.json
```

The report includes frames per second, per-frame latency, visible-object counts, and peak CUDA memory for each batch size.

## Measurement limitations

The volume calculation integrates the visible height field above an empty-scene baseline using camera intrinsics. It does not reconstruct hidden surfaces, and bags can deform, self-occlude, or contain air. Report this explicitly in the thesis.

Monocular liters are intentionally withheld until the model has been calibrated against aligned RealSense depth. An uncalibrated relative-depth map is not a valid volume measurement.

Color or visual object labels alone do not establish material composition. Reliable paper/plastic classification requires representative labeled examples, controlled validation, and an honest discussion of ambiguous or mixed-material packaging.

A load cell measures mass, not volume. Do not convert kilograms into liters without an independently justified material-density model.

## Troubleshooting

`CUDA available: false`: activate the VM's original PyTorch environment and rerun setup; do not install a generic CPU-only `torch` wheel.

No objects detected: ensure the object lies inside `LOCALLIFE_ROI`, wait for automatic setup to become ready, and collect labelled raw camera images for a detector fine-tuned on the actual installation. Avoid lowering confidence until false positives have been evaluated.

Only part of a garbage bag is detected: clear all bags and let automatic setup rebuild the empty depth reference without moving the camera. Fixed-bin depth fusion expands a partial bag prediction to the full supported silhouette. Keep `LOCALLIFE_PROMPTS` limited to waste-bag and sack variants.

No RealSense volume: verify the bridge uses `--source realsense`, the empty baseline contains aligned depth, the RealSense RGB intrinsics are transmitted, and the object raises the surface by more than `LOCALLIFE_MIN_HEIGHT_M`.

No monocular volume: confirm the dashboard reports **Depth calibration: ready**. A USB webcam without aligned RealSense reference cannot provide defensible calibrated liters.

GPU memory pressure: lower `LOCALLIFE_BATCH_SIZE` to `2`, reduce `LOCALLIFE_IMAGE_SIZE` to `640`, or use `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf`.

Spot interruption: restart the VM, restore bucket artifacts with `bash scripts/sync_bucket.sh pull`, and resume from `weights/last.pt`.

Run checks:

```bash
bash scripts/run_tests.sh
```

## Primary references

- Ultralytics YOLOE: https://docs.ultralytics.com/models/yoloe/
- Ultralytics training and resumption: https://docs.ultralytics.com/modes/train/
- Ultralytics training callbacks: https://docs.ultralytics.com/usage/callbacks/
- Depth Anything V2 Metric Indoor Large: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
