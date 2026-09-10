# LocalLife Waste Estimation Changelog

This file records user-facing behavior changes to the current LocalLife waste
estimation build. It uses a simple project-change version (`v1`, `v2`, ...),
separate from `PACKAGE_VERSION.txt` and the older LiveFix round numbers.

When adding a future version, record the observed problem, affected files,
configuration changes, validation performed, hardware status, and any known
limitations. This should make it possible to identify the version that
introduced a regression without guessing from file modification dates.

## v3 — 2026-09-09 — Phase 1B validation framework

### Status

- Implemented in the local workspace; not deployed to the friend's laptop.
- No camera, RealSense firmware, USB, baseline, geometry, detector, or volume
  calibration value was changed.
- The logging/reporting framework is ready, but its records remain empty until
  trials are performed with ruler-measured dimensions and independently known
  volumes.

### Reason for this version

Phase 1A made RealSense footprint L×W×H visible and restricted accepted waste
to plastic bags, paper bags, and cardboard boxes. Phase 1B supplies the
controlled evidence needed to determine whether those physical measurements
are accurate before changing the geometry or fitting a volume correction.

### Validation and calibration changes

- Extended `scripts/validate_known_volume.py` to record a unique trial ID and
  explicitly separate `validation` trials from factor-fitting `calibration`
  trials.
- A RealSense trial is accepted only when exactly one confirmed/predicted,
  non-phantom object belonging to an accepted waste class is present.
- The object detection's own liters value is recorded instead of trusting a
  scene-wide aggregate that could hide ambiguity.
- Added ground-truth inputs for waste type, colour, filled-object footprint
  length/width, physical height, placement, and setup notes.
- Added captured outputs for predicted class/colour, RealSense L×W×H,
  dimension confidence/flags/method, depth coverage, detection confidence,
  measurement quality, inference time, volume uncertainty, and active volume
  calibration factor.
- Added volume absolute/percentage error, dimension signed errors, sample
  range, population standard deviation, and the raw repeated sample values.
- Added an automatically regenerated CSV beside the detailed JSONL log. Older
  JSONL trials remain readable and are included where fields are available.
- Restricted `--calibrate` to `--camera realsense --role calibration`.
  Logitech and fused results can still be observed for comparison but cannot
  set the final metric-volume correction through this tool.
- No correction coefficient is supplied or guessed in this version. A factor
  may be fitted only when a tester deliberately provides `--calibrate` with a
  physically known reference; independent objects must then be used for the
  validation set.

### Example physical validation command

```text
python scripts/validate_known_volume.py --camera realsense --role validation \
  --known-liters 5 --label "filled-polythene-5L-01" \
  --object-type plastic_bag --actual-color black \
  --actual-length-mm 320 --actual-width-mm 210 --actual-height-mm 180 \
  --placement centre --samples 10
```

The example numbers above demonstrate command syntax only; they are not bag
calibration values and must be replaced with the tester's actual measurements.

### Files changed

| Area | Files | Purpose |
|---|---|---|
| Trial capture/reporting | `scripts/validate_known_volume.py` | Safe RealSense object selection, physical truth fields, repeatability metrics, JSONL and CSV output |
| Regression tests | `tests/test_known_volume_validation.py` | Reject ambiguous two-object scenes and preserve the accepted object's volume/dimensions |

### Validation completed

- Python compilation of the Phase 1B script — passed.
- Known-volume test module: 10 tests — all passed.
- Hardware accuracy remains pending on the friend's RealSense installation.

## v2 — 2026-09-09

### Status

- Implemented in the local workspace.
- Not deployed to `locallife@192.168.0.123`.
- No camera firmware, RealSense settings, USB settings, or other hardware
  configuration was changed.
- Synthetic/local behavior is verified; accuracy on the friend's physical rig
  is still unverified.

### Reason for this version

Real-world result images showed non-waste objects—including a backpack,
pillows, bottles, bedding, shoes, and background regions—being labelled as
garbage bags. Distance and colour were generally working, but height was
inaccurate, footprint dimensions were not visible, volumes were inflated, and
the dashboard could take approximately two to three minutes to show a result.

### Detection and classification changes

- Added the application-level accepted classes `plastic_bag`, `paper_bag`, and
  `cardboard_box`.
- A broad label such as `bag` is no longer sufficient for acceptance. A bag
  must have a plastic/waste/polythene or paper/kraft qualifier.
- Added explicit lookalike prompts for backpacks, rucksacks, laptop bags,
  briefcases, duffel bags, handbags, shoes, pillows, cushions, bedding,
  clothing, bottles, furniture, people, hands, and feet.
- Lookalike prompts are kept in `LOCALLIFE_NEGATIVE_PROMPTS`, separate from
  accepted prompts. They help the open-vocabulary detector name and reject an
  object instead of forcing it into the closest waste class.
- Added an overlap veto: when a similarly confident negative detection owns
  the same pixels as a waste label, the waste detection is rejected.
- An unlabelled RealSense depth region can no longer become a new waste object
  solely because Logitech reported a bag or box in the same frame. The camera
  views are not pixel-registered, so simultaneous presence is not proof of
  object identity.
- Phantom/depth-silhouette detections are excluded from final volume,
  automatic deposit, and durable ledger records.

### Dimension and height changes

- Added a general RealSense-only three-dimensional measurement contract:
  `footprint_length`, `footprint_width`, and `height`, reported in millimetres.
- Added `estimate_object_dimensions()` using aligned RealSense depth, factory
  intrinsics, and a fitted support plane.
- Only valid points measurably above the support plane enter footprint PCA.
  Floor pixels inside a loose segmentation mask therefore cannot inflate
  length or width.
- Disconnected elevated noise is removed by keeping the largest coherent
  component.
- Length and width use trimmed percentiles in support-plane coordinates;
  height uses a robust upper percentile rather than a raw maximum or the
  median of an entire loose mask.
- Measurements carry confidence and diagnostic flags, including low depth
  coverage, low elevated-point fraction, high plane error, mask clipping,
  single-view geometry, and live support-plane fitting.
- The same support plane is now used consistently for height, footprint, and
  synthetic reference depth within a frame.
- Cardboard boxes retain their rigid table-relative cuboid measurement and
  multi-frame dimension aggregation. Plastic and paper bags are explicitly
  described as a visible filled-object footprint, not as a rigid cuboid or
  the manufactured flat-bag size.

### Volume-authority changes

- RealSense is now the sole authority for final physical dimensions and final
  volume.
- Logitech may still provide colour, material, presence, and an optional
  diagnostic monocular estimate, but its liters are never averaged into the
  final result.
- The fused result now exposes `volume_source: "realsense"` whenever a final
  volume is available.
- Cardboard cuboid masks are restricted to the configured measurement region,
  reject floor-height pixels, retain only coherent elevated geometry, and
  detect clipping before mask erosion hides it.
- The example maximum plausible volume for a single item was reduced from
  120 L to 90 L.

### Dashboard, overlay, API, and persistence changes

- Added `accepted_class` and general `dimensions_mm` to serialized detections.
- Added dimension confidence, method, and flags to API output.
- Added accepted type and RealSense footprint `L×W×H` to the live dashboard.
- Added `LxWxH ... mm` directly to the annotated RealSense camera image, so
  dimensions are visible without scrolling to the table.
- Propagated accepted class and dimensions into scene-fused detections and
  ledger observations.
- Preserved the existing box-specific fields for backward compatibility.

### Latency changes

- Changed monocular Depth Anything from enabled-by-default to opt-in because
  it is expensive on a local CPU and is not permitted to determine final
  metric geometry.
- `cloud.env.example` now uses `LOCALLIFE_ENABLE_DEPTH=false`.
- Existing installations may still have `LOCALLIFE_ENABLE_DEPTH=true` in
  their own `cloud.env`; that explicit setting overrides the new default.
- The primary YOLO detector resolution/model and camera firmware were not
  changed in v2. Additional speed tuning must be benchmarked on the actual
  laptop so detection accuracy is not silently reduced.

### Files changed

| Area | Files | Purpose |
|---|---|---|
| Configuration | `locallife_cloud/config.py`, `cloud.env.example` | Accepted/negative prompts, RealSense authority defaults, latency default, plausible-volume limit |
| Detector setup | `locallife_cloud/inference.py` | Load accepted and negative prompt banks into YOLOE |
| Classification and orchestration | `locallife_cloud/pipeline.py` | Strict class mapping, phantom gates, support-plane consistency, dimension wiring, ledger/deposit safety |
| Metric geometry | `locallife_cloud/volume.py` | RealSense footprint L/W/H, floor/noise rejection, safer box cuboid masks |
| Data contracts | `locallife_cloud/types.py` | Accepted class, general dimensions, confidence, method, and flags |
| Scene fusion | `locallife_cloud/geometry.py` | Preserve accepted class through mask fusion |
| Final camera fusion | `locallife_cloud/comparison.py` | RealSense-only final volume and explicit volume source |
| UI and overlay | `locallife_cloud/dashboard.py`, `locallife_cloud/server.py` | Accepted type and visible L×W×H |
| Persistence | `locallife_cloud/ledger.py` | Store accepted class and dimensions |
| Regression tests | `tests/test_box_cuboid_volume.py`, `tests/test_dual_camera.py`, `tests/test_plug_and_play.py`, `tests/test_robust_plug_and_play.py`, `tests/test_vision.py` | Lock in v2 classification, geometry, fusion, latency, and serialization behavior |

### Validation completed

- Python compilation: `python3 -m compileall -q locallife_cloud` — passed.
- Whitespace/error check: `git diff --check` — passed for the v2 changes.
- Focused regression suite: 183 tests — all passed.
- Full repository discovery: 272 tests executed; 8 import/dependency errors
  occurred because this validation environment lacks OpenCV/Torch modules,
  and 3 pre-existing untouched colour/noise tests failed. Do not describe the
  complete repository suite as green until dependencies are installed and
  those existing failures are resolved.

### New regression guarantees

- Backpack, laptop bag, shoe, pillow, and bottle labels are not accepted as
  waste objects.
- A generic `bag` label is not accepted.
- A Logitech peer label cannot promote an unidentified RealSense silhouette.
- Final fused liters exactly match the RealSense reading, not a weighted
  RealSense/Logitech average.
- A synthetic tilted object recovers plane-relative height correctly.
- Adding a floor halo to an object mask does not expand its measured physical
  footprint.
- A plastic bag detection reaches the API as `plastic_bag` with populated
  RealSense `dimensions_mm`.
- Monocular depth is disabled when no environment override is supplied.

### Known limitations and pending work

- Physical RealSense accuracy has not yet been validated on the friend's
  laptop/camera installation.
- Filled-bag footprint dimensions represent the visible 3-D object. They are
  not the flat manufacturer's polythene dimensions.
- Plastic/paper bag volume still uses the existing RealSense height-field
  integration. It has not yet been calibrated using the user's standard
  polythene sizes and known/approximate volumes.
- `box_templates.yaml` remains deliberately unmeasured. Do not enable a box
  template until its physical L/W/H values have been measured.
- A captured empty-bin baseline remains preferable. Live support-plane fitting
  is a fallback and is explicitly flagged.
- Actual latency improvement depends on the friend's `cloud.env`, CPU/GPU,
  installed models, and whether monocular depth is explicitly re-enabled.

### Hardware validation required before declaring v2 accurate

1. Deploy only after reviewing this diff and the friend's active `cloud.env`.
2. Confirm `LOCALLIFE_ALLOW_UNCLASSIFIED=false` and, for the low-latency path,
   `LOCALLIFE_ENABLE_DEPTH=false`.
3. Capture a genuinely empty measurement-area baseline without moving either
   camera afterward.
4. Test negative objects: backpack, shoe, pillow, and bottle. The accepted
   detection count must remain zero.
5. Test one plastic bag, one paper bag, and one cardboard box individually.
6. Record ruler-measured L/W/H beside reported L/W/H, depth coverage,
   confidence, flags, and processing delay.
7. Use separate known-volume objects for calibration and validation; never
   validate using the same object that set the calibration factor.

### Regression diagnosis and rollback clues

- Wrong objects appear again: inspect `DEFAULT_NEGATIVE_PROMPTS`,
  `accepted_object_class()`, `reject_prompt_conflicts()`, and the active
  `LOCALLIFE_PROMPTS`/`LOCALLIFE_NEGATIVE_PROMPTS` environment values.
- Dimensions disappear: inspect `estimate_object_dimensions()`, the RealSense
  intrinsics, support-plane availability, object mask, and minimum point gate.
- Dimensions are inflated: inspect measurement ROI/polygon, `mask_clipped`,
  floor-height filtering, and whether the baseline/camera moved.
- Final volume changes when Logitech changes: inspect
  `DualCameraCoordinator.fused_result()`; `volume_source` must remain
  `realsense`.
- Phantom deposits return: inspect `_is_phantom_detection()`, accepted-class
  checks around ledger observation/auto-deposit, and
  `LOCALLIFE_ALLOW_UNCLASSIFIED`.
- Two-to-three-minute delay returns: inspect the active `cloud.env` for
  `LOCALLIFE_ENABLE_DEPTH=true`, then benchmark detector inference separately
  before reducing resolution or changing the model.

## v1 — baseline before 2026-09-09

This label represents the local project behavior before the v2 changes above.
Its important characteristics were:

- Detector prompts were dominated by positive bag/box terms, which could force
  lookalikes into a waste class.
- Generic bag labels and peer-camera-assisted silhouettes could enter paths
  that were too permissive for the user's required three classes.
- General bag L/W/H was not exposed on the dashboard or image overlay.
- Fused final liters could blend RealSense and Logitech measurements.
- Monocular Depth Anything was enabled by default and could contribute major
  local processing latency.

For older LiveFix implementation history, see `LIVEFIX_CHANGELOG.md`.
