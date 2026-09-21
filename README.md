# LocalLife — measurement core (review copy)

A RealSense D435 + Logitech C920 system that measures the volume of waste
deposited into a bin, and classifies each deposit by colour, material and
sorting correctness.

This branch is a **reading copy for review**. It contains the measurement
science and its tests, and deliberately leaves out the deployment plumbing
(web dashboard, HTTP server, Windows launcher, cloud orchestration, camera
drivers, training and dataset scripts), which carry no algorithmic content.
Nothing here holds credentials, keys or configuration values.

## Where to start

The volume method is the heart of the project; the rest supports it.

| Step | Module | What it does |
| --- | --- | --- |
| 1 | `locallife_cloud/heightmap_volume.py` | The volume method: 2.5D height-map integration over a calibrated support plane |
| 2 | `locallife_cloud/volume.py` | Support-plane fitting, per-detection measurement, the geometry modes |
| 3 | `locallife_cloud/geometry.py` | Colour classification (HSV relative saturation + an absolute chroma floor, CIELAB b\*) |
| 4 | `locallife_cloud/sorting_rules.py` | Correct vs mis-sorted deposit families |
| 5 | `locallife_cloud/pipeline.py` | Per-frame orchestration: detection, tracking, measurement, deposit acceptance |
| 6 | `locallife_cloud/event_log.py` | What counts as a finalised measurement, and how it is recorded |

Supporting modules: `tracking.py` (object identity across frames),
`ledger.py` (the deposit record), `fusion.py` and `accuracy.py` (combining the
two cameras), `calibration.py`, `box_templates.py`, `logitech.py`,
`material.py`, `types.py`, `config.py`, `comparison.py`, `benchmark.py`,
`storage.py`, `pointcloud_volume.py` (an alternative point-cloud estimator,
kept for comparison).

## The three design decisions most worth questioning

1. **Height-map integration over a fitted plane, not a point-cloud hull.**
   `heightmap_volume.py` integrates per-cell heights above a RANSAC-fitted
   support plane. Occlusion shadows are filled at the 25th percentile rather
   than the median: median filling fabricated a full-height ring around a
   reference box and inflated its volume by 12%, which p25 reduced to 4%. The
   grid also coarsens itself when the requested cell is finer than the depth
   sensor's own footprint at that distance.

2. **Deposits are differenced against a committed scene, not measured alone.**
   Two touching black bags merge into one mask and one depth component --
   nothing in colour or class can separate identical polythene. So each new
   deposit is measured cell-by-cell against the scene as it stood when the
   previous deposit was accepted (`incremental_deposit`). A pair that cannot be
   isolated is *withheld with a reason* rather than recorded as one large
   object. See `tests/test_deposit_isolation.py`.

3. **A finalised event, not a per-frame estimate, is the unit of record.**
   `event_log.py` writes one row per accepted deposit, keyed by a durable
   event id, so repeated frames, restarts and reconnections cannot duplicate or
   suppress a measurement.

## Running the tests

```bash
pip install -r requirements-local.txt      # numpy, opencv, scipy
PYTHONPATH=. python -m pytest tests -q
```

143 tests, no camera or GPU needed. The measurement tests build synthetic depth
frames with known geometry, so the reported errors are against ground truth
rather than against another estimate.

Worth reading as specifications in their own right:

- `tests/test_heightmap_volume.py` — the volume method against known boxes,
  including tilted planes and partial occlusion
- `tests/test_deposit_isolation.py` — touching objects, and what gets withheld
- `tests/test_sequential_soak.py` — behaviour over a long run of deposits
- `tests/test_noise_robustness.py` — sensor noise and depth dropout
- `tests/test_color_classification.py` — including dark objects, where relative
  saturation alone misreads black as blue

## Current accuracy

On the reference-object set, mean absolute percentage error in volume improved
from 15.4% to 1.1% after the height-map method replaced the earlier estimator.
That figure comes from controlled reference objects; it is not a claim about
arbitrary deformable bags, whose true volume is harder to establish.

## Known limitations

- Deformable bags have no single "true" volume, so accuracy there is bounded by
  how the ground truth is defined.
- The support plane must be re-calibrated if the camera moves; the pipeline
  detects a changed pose and marks measurements accordingly rather than
  silently applying a stale plane.
- Colour naming is bounded by illumination; the absolute chroma floor prevents
  the worst failure (dark objects read as saturated colours) but does not make
  colour lighting-invariant.
