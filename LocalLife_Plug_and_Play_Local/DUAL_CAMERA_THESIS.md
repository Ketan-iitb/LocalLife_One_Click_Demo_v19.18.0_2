# Independent dual-camera depth comparison

This project compares two separate physical cameras viewing the same calibrated
waste bin. It does **not** present the RealSense RGB image as Logitech footage.

| Camera | Physical input | Depth method | Persistent records |
| --- | --- | --- | --- |
| Intel RealSense D435 | Its own RGB and factory-aligned depth | Hardware active-stereo depth | `artifacts/realsense/waste_plant_ledger.jsonl` |
| Logitech C920 | Its own independent USB RGB stream | Depth Anything V2 Metric Indoor Large | `artifacts/logitech/waste_plant_ledger.jsonl` |
| Comparison | One-to-one matched deposited events | Timestamp and object type matching | `artifacts/comparison/reference_trials.jsonl` |

The two streams share the cloud GPU's loaded detection model. They do not share
RGB images, tracking IDs, baselines, object histories, measurement ledgers, or
volume calibration factors. GPU access is serialized to protect shared models.

## Deploy as a separate project

Download `LocalLife_Dual_Camera_Thesis.zip` on Windows and upload it to both the
VM and Raspberry Pi from Windows PowerShell:

```powershell
gcloud.cmd compute scp "$env:USERPROFILE\Downloads\LocalLife_Dual_Camera_Thesis.zip" depth-l4:LocalLife_Dual_Camera_Thesis.zip --zone=europe-west1-b
scp "$env:USERPROFILE\Downloads\LocalLife_Dual_Camera_Thesis.zip" locallife@locallife.local:LocalLife_Dual_Camera_Thesis.zip
gcloud.cmd compute ssh depth-l4 --zone=europe-west1-b
```

Inside the cloud VM, stop the previous server, extract the new project, and
reuse the existing working virtual environment and downloaded models:

```bash
pkill -f '^python -m locallife_cloud.server' || true
python3 -m zipfile -e ~/LocalLife_Dual_Camera_Thesis.zip ~
cd ~/LocalLife_Dual_Camera_Thesis
ln -s ~/LocalLife_Waste_Plant_Monitor/.venv .venv
source .venv/bin/activate
cp cloud.env.example cloud.env
test -f ~/LocalLife_Waste_Plant_Monitor/yoloe-11l-seg.pt && ln -s ~/LocalLife_Waste_Plant_Monitor/yoloe-11l-seg.pt yoloe-11l-seg.pt
test -f ~/LocalLife_Waste_Plant_Monitor/mobileclip_blt.ts && ln -s ~/LocalLife_Waste_Plant_Monitor/mobileclip_blt.ts mobileclip_blt.ts
python -m unittest discover -s tests -q
bash scripts/start_cloud.sh --disable-sync
```

If your active working environment is inside `~/LocalLife_Bag_Volume_Candidate`
or `~/LocalLife_Cloud_V15` instead, use that existing `.venv` path in the
`ln -s` command. Do not create a second cloud environment unnecessarily.

Keep the existing Windows SSH tunnel, or open it in another PowerShell window:

```powershell
gcloud.cmd compute ssh depth-l4 --zone=europe-west1-b -- -N -L 192.168.0.189:8000:127.0.0.1:8000
```

Replace `192.168.0.189` if `ipconfig` reports a different Wi-Fi address.

On the Raspberry Pi, extract the same new project:

```bash
python3 -m zipfile -e ~/LocalLife_Dual_Camera_Thesis.zip ~
cd ~/LocalLife_Dual_Camera_Thesis
python3 -m locallife_cloud.edge_client --cloud http://192.168.0.189:8000 --source dual --upload-fps 3
```

The Pi command automatically searches for a named Logitech/C920 V4L device and
runs both physical cameras in parallel. If automatic discovery chooses the
wrong device, inspect the actual cameras and provide its path explicitly:

```bash
v4l2-ctl --list-devices
python3 -m locallife_cloud.edge_client --cloud http://192.168.0.189:8000 --source dual --logitech-source /dev/video4 --upload-fps 3
```

`/dev/video4` above is only an example; use the actual Logitech device.

You can also run the two sources separately in two Raspberry Pi terminals:

```bash
python3 -m locallife_cloud.edge_client --cloud http://192.168.0.189:8000 --source realsense --upload-fps 3
python3 -m locallife_cloud.edge_client --cloud http://192.168.0.189:8000 --source logitech --upload-fps 3
```

## Camera calibration before comparison

1. Fix both cameras above the same bin. They must not move after calibration.
2. Measure the physical Logitech-lens-to-empty-bin distance with a tape measure.
3. Ensure no bags, boxes, or hands appear inside either camera's bin region.
4. Wait until both video and depth panels have stable images.
5. Click **Capture RealSense baseline**.
6. If the Logitech image contains a wall or surrounding floor, use **Set Logitech
   region** so its measurement rectangle contains only the actual bin opening.
7. Click **Set measured distance (required)** in the Logitech panel and enter the
   physically measured empty-bin distance in meters while the bin is empty. This
   also captures its baseline. It uses an independent physical reference, not
   RealSense pixels projected into a different camera viewpoint.
8. Check the reported Logitech camera tilt. Liters are reported at any angle up
   to 65° by default (the per-pixel volume integral is geometrically valid at
   any mounting angle -- it never assumed overhead), but above 35° the reported
   uncertainty grows with the angle and a warning says so; above 65° the bin
   floor is barely visible at all and measurement is withheld outright. Mount
   as close to overhead as practical for the tightest numbers regardless.
9. Put a reference box or bag with independently measured volume inside both
   views. To correct consistent scale bias, click **Calibrate known liters** in
   the corresponding camera panel. Each factor is stored independently.
10. Evaluate accuracy on additional objects that were **not** used for fitting.

Without the measured Logitech reference distance, the RGB and AI depth preview
remain visible and recognized objects can be observed, but their heights,
liters, bin occupancy, and automatic deposits are deliberately withheld. The
metric depth model's native scale is not evidence of a physically calibrated
measurement.

## Dual-camera fusion calibration (System 2, phase 1)

Everything above measures each camera independently, on purpose -- point 7
explicitly avoids projecting RealSense pixels into the Logitech's different
viewpoint. A proper fusion (RealSense as the trusted metric source,
Logitech reprojected into that same frame to fill whatever the RealSense
alone could not see) needs to know the two cameras' actual relative
position and orientation first, which a checkerboard stereo calibration
solves for directly instead of guessing.

`locallife_cloud/calibration.py` and `locallife_cloud/fusion.py` implement
that: fitting each camera's own intrinsics and their rigid relative pose
from several synchronized checkerboard views (`calibration.py`), and
reprojecting one camera's depth into the other's pixel grid plus a
primary-wins/secondary-fills merge (`fusion.py`). Both are pure geometry/
array code with no camera or network dependency, and are covered by
`tests/test_calibration_fusion.py`, which synthesizes checkerboard views
from a known ground-truth pose and confirms the real OpenCV solver
(`cv2.stereoCalibrate`) recovers it to within about 1° and 1 cm.

To actually produce a calibration file from your rig:

1. Start the local dashboard as usual (`RUN_EVERYTHING_ONE_CLICK.cmd`, or
   `Start-PlugAndPlay-Experiment.ps1 -Role Local`).
2. Print a checkerboard pattern at 100% scale (no "fit to page") and
   measure one square with a ruler -- that measurement is what gives the
   whole calibration real-world units, so get it right. Mount it flat
   (tape to cardboard) so it does not flex.
3. Run, from the `LocalLife_Plug_and_Play_Local` folder:
   ```
   python scripts/calibrate_dual_camera.py --columns 9 --rows 6 --square-size-m 0.024 --pairs 18
   ```
   (columns/rows are *internal* corners -- a 10x7-square board has 9x6.
   Adjust `--square-size-m` to your printed board's actual measured
   square size.)
4. Slowly move the board so both cameras can see it at once, holding it
   still for a moment in each position; the script auto-captures a pair
   whenever it sees the board in both views and pauses briefly so you can
   reposition. Vary distance, tilt, and position across the shared field
   of view -- a dozen near-identical fronto-parallel shots calibrate far
   worse than a dozen genuinely different poses.
5. The script reports each camera's own reprojection error and the
   combined stereo error in pixels (under about 1 px is a good fit; well
   above that means recapture with more pose variety) and writes
   `data/dual_camera_calibration.json` (override with `--output`).

**What phase 1 does not yet do**: this calibration file is not read by the
live pipeline yet. Producing it correctly (this phase) and consuming it to
actually fuse a frame's two depth maps into one combined volume measurement
(the next phase) are being kept as separate, separately-tested changes on
purpose -- wiring two independently-tracked, independently-baselined camera
streams together touches tracking, the ledger, and the dashboard's
per-camera assumptions in ways worth reviewing on their own, not bundled
into the calibration math itself.

## Logitech tuning: object mask, color, and impossible liters

The Logitech monocular prediction can shift globally when an object enters the
image. Treating the whole changed AI depth map as an object incorrectly includes
walls and floor, produces grey instead of the true object color, and can report
hundreds of fabricated liters. The tuned Logitech pipeline therefore:

- Uses the actual YOLO instance mask as the object's semantic anchor.
- Limits any RGB foreground expansion to the local object; full-room components
  cannot replace the object mask.
- Rejects masks whose bounding footprint spans a wall or the surrounding floor,
  even when the mask itself is sparse.
- Suppresses nested prompts so one physical cardboard box is not counted twice.
- Compensates room-wide exposure changes before extracting RGB foreground.
- Samples color from the changed object interior and prioritizes chromatic
  object pixels over neutral wall glare.
- Recognizes ordinary cardboard hues as brown and stabilizes colors over
  successive frames.
- Corrects monocular depth drift using static background from the same Logitech
  camera; it never borrows RealSense pixels.
- Rejects implausible object volumes instead of silently clipping or depositing
  them into the Logitech ledger.
- Requires an independently measured reference distance and a near-overhead
  view before reporting liters, object heights, or depositing records.
- Rejects object heights beyond the configured physical maximum rather than
  showing contradictory 120 cm heights with a small partial volume.
- Computes bin occupancy only inside confirmed current and previously deposited
  Logitech object masks; the wall and floor cannot contribute liters.
- Automatically quarantines previously deposited impossible Logitech entries,
  preserving the original JSONL audit events while excluding them from history,
  color totals, deposited counts, and cumulative liters.
- Supports a separate Logitech region of interest and history reset.

Relevant controls are:

```bash
# Optional, independently positioned Logitech measurement rectangle.
LOCALLIFE_LOGITECH_ROI=0.30,0.15,0.65,0.75
LOCALLIFE_LOGITECH_MAX_SCENE_FRACTION=0.45
LOCALLIFE_LOGITECH_MAX_MASK_EXPANSION=2.0
LOCALLIFE_LOGITECH_MAX_ITEM_VOLUME_L=120
LOCALLIFE_LOGITECH_STABILIZE_DEPTH=true
LOCALLIFE_LOGITECH_REQUIRE_REFERENCE=true
LOCALLIFE_LOGITECH_REQUIRE_OVERHEAD=true
LOCALLIFE_LOGITECH_MAX_TILT_DEG=35
LOCALLIFE_LOGITECH_DUPLICATE_OVERLAP=0.55
LOCALLIFE_LOGITECH_MIN_VALID_HEIGHT_FRACTION=0.35
```

The example ROI must be adjusted for your actual camera view. It should contain
the bin opening and exclude surrounding walls, floor, equipment, and unrelated
objects. The dashboard's **Set Logitech region** button configures it without
changing the RealSense ROI; capture a fresh Logitech baseline afterward.

The default 120 L plausibility ceiling is an operational safeguard, not a claim
that every waste plant uses this bag size. Raise or lower it to match the actual
largest plausible incoming object. Rejected objects remain visible as observed
but cannot be deposited with a fabricated volume.

If earlier Logitech experiments already produced incorrect 189–903 L entries,
entries above the configured maximum are quarantined automatically when the
updated application starts. Their original observations and quarantine events
remain in the Logitech JSONL audit trail, but they do not appear in active totals.
Use **Clear incorrect history** only if you also want to remove earlier incorrect
observations below the configured maximum. The old Logitech ledger is backed up
first; RealSense history and its measurements remain unchanged.

### Apply the Logitech tuning update to an existing deployment

The tuning archive overlays the existing project but deliberately excludes your
`cloud.env`, virtual environment, models, baselines, and stored camera history.
Only the cloud VM needs this update; the existing Raspberry Pi dual-camera
client and SSH tunnel remain compatible.

In Windows PowerShell:

```powershell
gcloud.cmd compute scp "$env:USERPROFILE\Downloads\LocalLife_Dual_Camera_Logitech_Calibrated.zip" depth-l4:LocalLife_Dual_Camera_Logitech_Calibrated.zip --zone=europe-west1-b
gcloud.cmd compute ssh depth-l4 --zone=europe-west1-b
```

Inside the cloud VM:

```bash
pkill -f '^python -m locallife_cloud.server' || true
python3 -m zipfile -e ~/LocalLife_Dual_Camera_Logitech_Calibrated.zip ~
cd ~/LocalLife_Dual_Camera_Thesis
source .venv/bin/activate
python -m unittest discover -s tests -q
bash scripts/start_cloud.sh --disable-sync
```

After reloading the dashboard, previous impossible Logitech deposits are
quarantined automatically. Mount the Logitech above the bin, set a region that
excludes the wall and surrounding floor, remove all objects, and click
**Set measured distance (required)** with the actual tape-measured distance in
meters. Check that the displayed tilt is no more than 35°, then test one brown
box and one black bag separately. Clear the remaining Logitech history only if
you also want to remove earlier incorrect observed-only records.

The Logitech focal length initially comes from an assumed horizontal field of
view. For stronger thesis evidence, perform an OpenCV checkerboard/ChArUco lens
calibration and supply its measured focal lengths to the edge bridge:

```bash
python3 -m locallife_cloud.edge_client \
  --cloud http://192.168.0.189:8000 \
  --source dual \
  --logitech-fx 920.0 \
  --logitech-fy 918.0
```

The focal-length values above are examples, not measured properties of your
camera. Different capture resolutions require corresponding intrinsics.

## RealSense accuracy controls

The improved RealSense estimator keeps the proven original default geometry:

```text
pixel area = measured_depth² / (fx × fy)
volume = Σ(object height × pixel area)
```

It adds temporal-median empty-bin baselines; per-pixel baseline noise estimated
with median absolute deviation; noise-adaptive foreground thresholds; isolated
stereo-spike rejection that preserves object edges; conservative isolated-hole
repair; robust empty-floor plane/tilt diagnostics; temporal-median volume
smoothing; and persistent calibration against measured objects. Objects with
RealSense depth coverage below the configured minimum remain visible with their
uncertainty, but neither automatic settlement nor manual confirmation can add
their incomplete and downward-biased volume to the deposited thesis history.

Reported uncertainty combines independent depth noise, measured baseline noise,
missing-depth coverage, and an explicit correlated systematic-error floor. The
default 2.5% systematic floor is a configurable assumption, **not** a validated
accuracy guarantee. Measure its suitability on representative calibration and
held-out validation objects.

Optional geometry models support documented thesis sensitivity experiments:

```bash
LOCALLIFE_VOLUME_GEOMETRY=surface-columns
LOCALLIFE_VOLUME_GEOMETRY=ray-frustum
LOCALLIFE_VOLUME_GEOMETRY=reference-plane
```

`surface-columns` preserves the existing object's visible-surface footprint;
`ray-frustum` integrates each diverging camera-ray frustum exactly; and
`reference-plane` uses the empty-bin backplane footprint. These assumptions
produce different volumes for the same silhouette. Select one using physical
ground-truth validation instead of assuming the largest value is most accurate.

Other controls in `cloud.env` include:

```bash
LOCALLIFE_BASELINE_FRAMES=9
LOCALLIFE_VOLUME_FRAMES=5
LOCALLIFE_DEPTH_NOISE_M=0.004
LOCALLIFE_DEPTH_NOISE_SIGMA=3.0
LOCALLIFE_SYSTEMATIC_ERROR_FRACTION=0.025
LOCALLIFE_REJECT_DEPTH_OUTLIERS=true
LOCALLIFE_MIN_DEPTH_COVERAGE=0.70
LOCALLIFE_COMPARISON_MATCH_SECONDS=8.0
```

## Thesis outputs and reproducibility

Each camera independently reports observed and deposited bag/box counts,
dominant colors, configured waste streams, current item volume, bin occupancy,
cumulative liters, and chronological detection history. Matching uses object
type plus deposit timing, and never reuses an object in a second pair.

Download separate histories and a paired comparison dataset from:

```text
/api/cameras/realsense/history.csv
/api/cameras/logitech/history.csv
/api/comparison.csv
```

The comparison panel reports paired counts, camera-to-camera signed bias, mean
absolute disagreement, RMSE between cameras, color agreement, and unmatched
objects. These are **agreement** metrics; neither camera is automatically
physical ground truth.

After a paired object settles, select **Record known-volume ground truth** and
enter a separately measured reference. Once reference trials exist, both camera
panels report MAE, RMSE, signed bias, and MAPE against physical ground truth.

Current externally visible volume, cumulative incoming volume, and remaining
bin occupancy differ when bags compress, overlap, or hide other surfaces. No
single overhead camera can directly observe hidden back surfaces or contents.
