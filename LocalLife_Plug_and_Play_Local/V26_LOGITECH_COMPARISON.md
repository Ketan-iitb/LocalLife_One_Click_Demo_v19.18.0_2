# V26: shape-aware geometry, Logitech + Depth Anything V2, paired comparison

Branch `Working_branch_v26_cloud_logitech_comparison`. It was created from
`Working_branch_v21_cloud_local_launcher` at
`555c9eee36964679cce907db5dfac253526db14f`, then fast-forwarded through the
v22–v25 fix commits. Those commits already sat linearly on top of V21. They
bring canonical CSV persistence, verified cloud SSH, stable measurement IDs,
cylinder footprints and cloud startup status. V21 and every older branch are
unchanged.

## Cloud path

The v3.2 deployment (`LocalLife_Windows_Deployment_v3_2/app`) was checked
against this branch:

* `gpu.py` is byte-identical.
* The cloud port itself (`34f8a63 Port the v3.2 cloud deployment`) and its
  later fixes are already part of this branch.
* The v3.2 copies of `pipeline.py`, `server.py`, `config.py` and the other
  modules are *older* than the ones here. Copying them would have rolled back
  later fixes, so they were not copied.

The launcher flow is unchanged: `gpu.py up` → final zone → authenticated,
verified `gcloud compute ssh/scp` → backend → tunnel → Pi → dashboard. No
credentials are stored in the repository. Machine-specific values live in
`.env` and `cloud.env`, using the templates.

## RealSense (unchanged baseline)

The capture, alignment, intrinsics, empty support-plane baseline, ROI, depth
filtering, segmentation, tracking, event IDs and the existing cuboid path are
untouched. The Phase 1 PDF results are **RealSense-only** evidence.

## Shape-aware geometry (`shape_geometry.py`)

After segmentation, the object's points above the support plane
(`volume.object_plane_points`) are routed to one method. The choice comes from
measured geometry, never from the label alone.

| Method | Evidence | Selected volume |
|---|---|---|
| `cuboid` | footprint fills its minimum-area (oriented) rectangle, flat top, flat cross-profile | L × W × H |
| `cylinder` (upright) | disc footprint (fill ≈ π/4, circle residual ≤ 0.07), or a visible shell whose circle fit restores the hidden half | π r² h |
| `cylinder` (lying) | filled rectangle, height ≈ width, curved cross-profile | π r² L |
| `irregular_rigid` / `flexible_or_unknown` | anything else with a valid height map | Σ h(u,v)·A(u,v) over the cleaned mask |
| `uncertain` | < 50 points, implausible dimensions, or no height-map volume | none |

Every result reports three volumes, which are different quantities:
`bounding_box_volume_litres`, `mesh_volume_litres` (the height-map integral)
and `selected_volume_litres`. It also reports `geometry_method` and
`geometry_confidence`, plus `cylinder_diameter_mm`, `cylinder_height_mm`,
`cylinder_fit_residual` and `cylinder_volume_litres` when the method is
cylinder.

For flexible bags the selected volume is the **current external occupied
volume**. It is not the bag's capacity. Dimensions stay descriptive; their
product is not an irregular object's volume.

`GeometryLock` accepts a method only when it wins `max(3, settle_frames)`
frames. It then freezes the method, the median dimensions and the litres for
the life of the track.

## Logitech + Depth Anything V2

Pipeline: RGB → shared detector/segmentation → Depth Anything V2 → metric
calibration → support plane → the same shape router. The router works on the
mask eroded by 2 px (monocular depth bleeds across edges) → dimensions and
volume → colour/material/sorting → stable final event.

**Metric calibration (`logitech_calibration.py`).** The raw model output is
never reported as millimetres or litres.

* One tape-measured flat reference (the empty bin floor) sets the scale of the
  metric checkpoint: `Z = a·Z_pred`.
* Two or more references at different distances (for example a board raised in
  the bin) fit `Z = a·Z_pred + b` by least squares.
* Relative (non-metric) checkpoints predict inverse depth, so they are fitted
  on `1/Z = a·d + b`. This needs at least two distances.

The calibration is stored in `results/logitech/calibration/logitech_depth.json`
with its `calibration_id`, scale, shift, method, date, reference distance and
resolution. A fit made at another resolution is not reused. Calibration samples
are calibration data only: never evaluation objects, and never RealSense
readings. Without a valid calibration the dashboard shows **"Relative depth only -
metric volume unavailable"** and no litres are reported. In the template,
`LOCALLIFE_LOGITECH_ALLOW_PROVISIONAL_METRIC` is now `false`.

**Lens calibration.** Calibrate the C920 with a checkerboard using
`scripts/calibrate_dual_camera.py`, then place the output in one of these two
locations:

* `results/dual_camera_calibration.json` (the file as written), or
* the Logitech `MonoCalibration` alone, as
  `results/logitech/calibration/logitech_lens.json`.

Every Logitech frame is then lens-undistorted (`LogitechLens`) before
detection, depth inference and geometry, and the calibrated intrinsics replace
the ones estimated from the field of view. A profile applies at its own
resolution, or at a same-aspect resolution with scaled intrinsics. Any other
crop is left untouched and reported as `lens_profile_resolution_mismatch`.
Keep resolution and crop fixed for every comparison session.

## Paired comparison (`paired_events.py`)

Research modes are set with `LOCALLIFE_RESEARCH_MODE`: `paired`,
`realsense_only` or `logitech_only`. Fusion stays a separate, optional reading.

Each camera finalises independently. The paired log groups the two finalised
measurements that arrive within `LOCALLIFE_COMPARISON_PAIR_WINDOW_S`
(default 20 s) and have compatible object types under one
`comparison_event_id`. A camera with no result in that window gets one explicit
`missing` row. The other camera's numbers are never substituted.

**Canonical CSV:** `results/comparison/comparison_measurements.csv`, one row per
finalised camera measurement. The columns are:

```
session_id, comparison_event_id, measurement_id, camera_source, timestamp,
processing_mode, calibration_id, calibration_method, object_type,
geometry_method, geometry_confidence, length_mm, width_mm, height_mm,
cylinder_diameter_mm, cylinder_height_mm, cylinder_fit_residual,
bounding_box_volume_litres, mesh_or_shape_volume_litres, selected_volume_litres,
volume_meaning, ground_truth_volume_litres, ground_truth_method, dataset_split,
absolute_error_litres, percentage_error, colour, colour_confidence, material,
material_confidence, sorting_result, overall_confidence, processing_time_ms,
status, reason, model_version, pipeline_version
```

How the CSV behaves:

* **Idempotent.** `measurement_id` is the idempotency key, so repeated frames,
  changing detector IDs, refreshes, retries and restarts add no rows.
* **Safe on failure.** A failed write is queued (`POST /api/comparison/retry`).
  It never stops tracking.
* **Excel-friendly.** The file is UTF-8 with a BOM and is fsynced after every
  row.
* **One source of truth.** The writer and `GET /api/comparison/measurements.csv`
  use the same file. A downloaded CSV is a **snapshot**, so download again to
  get newer rows.
* **History kept.** Rows from earlier sessions are kept.

The per-camera `measurements.csv` keeps its established columns.

**Ground truth:** `POST /api/comparison/ground-truth`. You can also use the form
on `/research`. It records the actual object, material, colour, L/W/H,
reference litres, method and `dataset_split` (`evaluation` or `calibration`).
The method is one of `cuboid_dimensions`, `cylinder_dimensions`,
`manufacturer_capacity`, `displacement_reference_container` or
`bounding_box_reference`. `manufacturer_capacity` is labelled as nominal
capacity, not external occupied volume. Ground truth never reaches inference.

**Dashboard:** `/research` has a *Paired camera comparison* section. It shows
each camera's geometry, dimensions, litres, colour, material, sorting,
confidence, time and status, then the Logitech − RealSense difference, the
ground-truth error and the CSV status. The Logitech calibration coefficients and
the sample form are under *Research / advanced*. The operator page `/` is
unchanged.

**Performance:** `benchmark.py` already records `capture_fps`, `inference_fps`,
`end_to_end_latency_ms`, `frames_dropped`, and the upload, inference and return
latencies separately. Each paired row also carries that camera's
`processing_time_ms`.

## Experimental protocol

1. Fix the camera mounts, resolution and crop. Calibrate the Logitech lens
   (checkerboard).
2. Capture the empty baselines. Add Logitech calibration samples: the empty
   floor at a tape-measured distance, plus a flat board at a second height.
   Capture the baseline again.
3. For each evaluation object (never one used in step 2): place it, wait until
   both cameras finalise, then enter ground truth for that `comparison_event_id`.
4. Download the latest CSV and compute per-camera error from
   `selected_volume_litres` against `ground_truth_volume_litres`, using only
   `dataset_split=evaluation`.

## Limitations

* The shape thresholds were tuned on synthetic geometry. They need checking on
  real RealSense and Logitech captures.
* A lying cylinder is inferred from its top profile, and a tilted cylinder that
  is neither upright nor lying falls back to the height map.
* Monocular metric accuracy is bounded by the calibration references. Object
  edges remain less reliable than with stereo depth.
* Pairing is by time window and object family. Two different objects finalised
  within the window by different cameras could be grouped together, so present
  one object at a time.
* No real-hardware acceptance run is part of this change. All tests use
  synthetic data and stand-in models, and no experimental data was generated or
  fabricated.

## Real-hardware acceptance

`scripts/check_v26_acceptance.py --url http://127.0.0.1:8000` reads a live
system and checks these steps:

1. The backend is healthy.
2. Both camera streams are arriving.
3. The Logitech metric calibration and lens profile are active.
4. The latest comparison event has both cameras.
5. The downloaded CSV contains both complete rows.

Run it after one controlled object has been measured. Repeat with a box, a can
(which should report the cylinder method and diameter) and an irregular object
(which should report the height-map method). The script creates no data, so a
PASS requires real cameras.
