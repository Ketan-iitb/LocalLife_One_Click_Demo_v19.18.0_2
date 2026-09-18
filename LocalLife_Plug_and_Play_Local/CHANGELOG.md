# LocalLife Waste Estimation Changelog

This file records user-facing behavior changes to the current LocalLife waste
estimation build. It uses a simple project-change version (`v1`, `v2`, ...),
separate from `PACKAGE_VERSION.txt` and the older LiveFix round numbers.

When adding a future version, record the observed problem, affected files,
configuration changes, validation performed, hardware status, and any known
limitations. This should make it possible to identify the version that
introduced a regression without guessing from file modification dates.

## v7.1 — 2026-09-18 — Measure only the new surface; stop publishing flagged dimensions

### Status

- Implemented locally, following the first `images/v7` hardware run of v7.
  Not yet retested on the RealSense computer. No camera setting, firmware,
  intrinsics, or calibration factor was changed, and no dimension
  mathematics was changed.

### Evidence from the v7 run

v7's baseline fix worked: no result in `images/v7` carries
`live_fitted_support_plane`, so an empty-scene baseline was captured for the
first time, and RealSense live tracks fell from the v6 range of 4-9 to 1-2.

The reported dimensions were still wrong. A black bag on a bed reported
`711 x 325 x 606 mm`, then `635 x 159 x 611 mm`, then `521 x 158 x 611 mm`
across three scenes -- a 611 mm height that barely moved while the object
changed. The overlays show why: YOLOE returned one mask covering the bag and
a large blotch of the wall behind it. Depth-supported recovery correctly
refused that surface -- `support_plane_mask_recovered` appears on none of
those results -- but refusing fell back to the raw semantic mask, the
contaminated one, so the wall was measured anyway. Every such result already
carried `mask_clipped` and `low_elevated_fraction` and was published all the
same.

### Changes

- Dimensions and rigid cuboids are now computed from a separate
  `dimension_mask`: the detection's mask intersected with the
  newly-introduced region. No path reaches the estimator without passing the
  novelty test, so refusing a recovered surface can no longer fall back to a
  contaminated one. Results carry `mask_constrained_to_new_surface` when this
  narrowed the mask.
- `instance_mask` itself is deliberately left unconstrained for volume and
  depth coverage. Coverage means "how much of this object has valid depth";
  pixels the camera failed to measure are not newly-introduced, so
  intersecting them away would erase exactly the holes coverage exists to
  detect. A regression test caught this: constraining it let a sparse-depth
  object be auto-deposited as a good measurement.
- Dimensions flagged `mask_clipped`, `low_elevated_fraction`,
  `high_plane_rmse` or `no_newly_introduced_surface` are now withheld rather
  than published with a lowered confidence score. The flags remain visible as
  the stated reason. Reversible with
  `LOCALLIFE_REJECT_FLAGGED_DIMENSIONS=false`.
- The detection admission gate was lowered from half the measurement fraction
  to `LOCALLIFE_DETECTION_CHANGE_MIN_FRACTION` (0.10). At the higher value a
  real object whose detector mask was only ~26% new was discarded outright
  and vanished from the dashboard. Admission answers "is any of this newly
  placed?"; accuracy is handled by measuring only the new part.

### Validation completed

- Focused suite: 297 tests passed, including 15 in
  `tests/test_baseline_change_gating.py`. The contamination test is paired
  with a second test proving the same scene really would mismeasure if the
  whole semantic mask were used, so it cannot pass vacuously.
- The same 9 pre-existing OpenCV/Torch-dependent failures remain, confirmed
  failing on the unmodified tree.
- No hardware accuracy claim. The 120 mm recovered in the synthetic scene is
  synthetic; ruler validation on the RealSense rig is still outstanding.

### Known limitations

- Steps 5 and 6 of the agreed correction remain open: deformable label drift
  can still fork a track, and there is no single-dominant-object gate. The
  v7 run still shows occasional `unclassified object` phantom tracks.
- `fuse_scene_detections` discards a neural detection overlapping the changed
  region by less than 30%. That rule predates this work but only became
  active now that baselines exist, and it can drop a real object with a very
  sloppy mask. It was left alone rather than retuned blind.

## v7 — 2026-09-18 — Baseline deadlock and measured-object novelty

### Status

- Implemented locally; not yet deployed to or retested on the friend's
  RealSense computer. No camera firmware, laser power, exposure, USB, depth
  preset, ROI, factory intrinsics, or volume calibration factor was changed.
- No dimension mathematics was changed. This version corrects which pixels
  reach the estimator, not how the estimator turns pixels into millimetres.

### Evidence and reason

The v6 hardware trial measured the bed rather than the bag placed on it: one
green bag produced simultaneous handbag/duffel/backpack/pillow detections,
the RealSense object counter climbed to 29-51 with 4-9 live tracks in a
mostly unchanged scene, and reported dimensions included approximately
1021 x 669 x 394 mm and 1074 x 355 x 359 mm. The dashboard flagged
`dimension_instability`, `mask_clipped`, `low_elevated_fraction`,
`low_valid_depth`, `live_fitted_support_plane` and
`support_plane_mask_recovered` throughout.

Reading the mask/baseline/dedup/tracking path found a deadlock rather than a
geometry error. `_consider_automatic_baseline()` reset its empty-scene
countdown on any accepted detection; `geometry_validation` mode's broad
household prompt bank detects the room's own furniture in every frame, so the
countdown never completed and no empty-scene baseline was ever captured. With
no baseline:

- the support plane was refitted from every frame's own background, and that
  background is defined by excluding the frame's own detections, which change
  constantly, so the plane moved every frame and the measured object changed
  shape while standing still;
- there was no notion of "newly introduced" at all. Elevation above the
  support plane was the only test a surface had to pass, and on a bed the
  bag, the duvet folds and a pillow form one *connected* elevated component,
  so depth-supported mask recovery annexed the lot and the estimator measured
  it faithfully.

### Changes

- The automatic empty-scene countdown no longer resets on detections while
  `geometry_validation` mode is active; it gates on camera and scene
  stillness alone, and reports `verifying-still-scene-N-of-M` with an
  explicit warning to keep the reference object out of view until setup
  completes. Production `waste` mode still requires a genuinely empty scene,
  unchanged. Reversible through
  `LOCALLIFE_VALIDATION_BASELINE_IGNORES_DETECTIONS=false`.
- Added `newly_introduced_mask()`: the RealSense pixels now reading
  measurably closer than the captured empty baseline. Where the baseline's
  own noise map is larger, the change threshold rises with it. With no
  baseline it returns None, so an uncalibrated installation keeps its
  previous behaviour rather than showing an empty dashboard.
- Detections whose masks contain almost no newly-introduced pixels are
  rejected before tracking or measurement. An unchanged pillow, duvet or
  headboard is therefore rejected on physical grounds whatever the
  open-vocabulary detector chooses to call it, which no amount of prompt-bank
  tuning can achieve: a closed prompt bank always assigns every salient
  region its nearest accepted label.
- Depth-supported mask recovery intersects the elevated surface with that
  newly-introduced region, and rejects a recovered mask that is still mostly
  unchanged scenery. The constraint is skipped when the changed region no
  longer explains the semantic seed, so a stale baseline degrades to the
  previous behaviour instead of silently measuring nothing.
- Added `newly_introduced_pixels` to the API/state diagnostics. Zero with an
  object in view means the baseline is stale or was captured with the object
  already present.

### Validation completed

- Focused suite: 293 tests passed, including 11 new regression tests in
  `tests/test_baseline_change_gating.py` covering novelty detection, the bed
  contamination case, rejection of unchanged furniture detections, and the
  validation-mode baseline capture that waste mode must not inherit.
- The 9 remaining failures in this dependency-limited WSL interpreter
  (`test_calibration_fusion`, `test_material`, `test_noise_robustness`) are
  pre-existing and require OpenCV/Torch; they were confirmed failing on the
  unmodified tree before these changes. Three further modules
  (`test_local_offline_mode`, `test_material_transformers_compat`,
  `test_recipe_pipeline`) cannot import without `cv2`/`torch` and must be run
  in the friend's installed environment.
- No hardware accuracy claim is made. This version must be retested on the
  RealSense rig with a ruler-measured object.

### Known limitations and pending work

- Steps 4-6 of the agreed correction are not in this version: dimensions are
  still published when the estimator has already flagged them untrustworthy
  (`mask_clipped`, `high_plane_rmse`, live-fitted plane); label drift between
  deformable labels still forks a track, so one bag can still be counted
  several times; and no single-dominant-object gate exists for validation
  trials yet.
- `max_expansion` still bounds recovery by a ratio to the semantic seed. A
  very small seed on a much larger object therefore falls back to the
  semantic mask -- undersized, but no longer the furniture.
- If the baseline is captured with the reference object already in view, that
  object becomes part of the scene and will be rejected as unchanged.
  `newly_introduced_pixels` reading zero is the signal to recapture.

## v6 — 2026-09-17 — V3 dimension stability and rigid-object correction

### Status

- Implemented locally as build `19.21.0-local-ai`; not yet deployed to or
  retested on the friend's RealSense computer.
- No camera firmware, laser power, exposure, USB, depth preset, ROI, factory
  intrinsics, baseline files, or volume calibration factor was changed.

### Evidence and reason

- V3 image 8 reported approximately `348 x 227 x 102 mm` for a ruler value of
  `360 x 245 x 130 mm`; image 9 reported `389 x 302 x 123 mm` for
  `410 x 315 x 140 mm`.
- The same stationary household objects also varied materially between
  frames. A single global scale was rejected because it would worsen other
  cases, including the laptop bag, and because only two V3 rigid objects have
  complete independent ruler truth.
- Rigid objects labelled `book` or `storage container` were incorrectly using
  the general visible-footprint path instead of the rigid cuboid path.

### Changes

- Rigid validation labels including book, storage container, package, parcel,
  carton, shoebox, and box now use the RealSense table-relative cuboid
  estimator. Logitech remains non-authoritative for metric geometry.
- Removed the cuboid estimator's default two-pixel mask erosion. The existing
  elevation and percentile filters remain, while explicit erosion is still
  available for a deliberately noisy-mask experiment.
- Rigid height now uses the bounded 98th percentile already calculated by the
  estimator instead of the upper-decile median (approximately p95), avoiding
  systematic shortening from visible side-wall and bevel pixels without using
  a noise-sensitive raw maximum.
- Added per-track median L/W/H aggregation for every RealSense validation
  object, not only cardboard boxes. The API and dashboard now expose accepted
  versus considered frames and dimension spread; high spread adds the
  `dimension_instability` flag and lowers confidence.
- Geometry-validation tracking now keeps one track when the open-vocabulary
  label changes between book, storage container, and cardboard box on adjacent
  frames, preserving the dimension history.
- Corrected the Amazon reference box manifest/template from the historical
  filename value `410 x 330 x 140 mm` to the V3 ruler remeasurement
  `410 x 315 x 140 mm`; its external cuboid reference is now `18.081 L`.

### Validation completed

- Focused rigid/general geometry, tracking, reference validation, camera
  recovery, streaming, classification, and integration suite: 252 tests
  passed.
- Full 294-test discovery is not clean in this reduced WSL interpreter:
  missing OpenCV/Torch caused eight import/runtime errors, and four unrelated
  material/noise assertions also failed in that dependency-limited run. The
  selected 252-test suite above is the valid local regression result; the
  dependency-specific modules must run in the friend's installed environment.
- Live V3 accuracy remains pending until build `19.21.0-local-ai` is deployed
  and the two ruler objects are repeated with one still object and an empty,
  stable RealSense baseline.

### Known limitations

- No ruler-derived multiplier is hard-coded. Using the same two objects both
  to fit and claim accuracy would be circular; they remain validation cases.
- Deformable bags, backpacks, pillows, and clothing expose their current
  filled visible footprint. Their flat manufactured size is not a valid
  single-view 3-D ground truth.

## v5 — 2026-09-16 — Reversible household-object geometry validation

### Status

- Implemented locally as build `19.20.0-local-ai`; not deployed to the
  friend's laptop and not yet tested with live RealSense hardware.
- No camera firmware, exposure, laser, USB, depth preset, ROI, baseline, or
  calibration factor was changed.

### Reason for this version

The measured bags and cartons are not always available at the friend's home.
Phase 1B therefore needs ordinary household objects to exercise RealSense
footprint length, footprint width, and height without permanently weakening
the final waste-only classifier.

### Changes

- Added the explicit `LOCALLIFE_OPERATING_MODE` switch. `waste` preserves the
  existing strict plastic-bag, paper-bag, and cardboard-box rules;
  `geometry_validation` accepts a separate bank of common household reference
  objects as `measurement_object` tracks.
- Former waste negatives including backpacks, laptop bags, shoes, bottles,
  cans, cushions, and laundry hampers may be measured only in validation mode.
  People, hands, feet, unclassified silhouettes, and fixed scene/background
  classes remain rejected.
- Hard-disabled waste-ledger writes, ledger refreshes, and auto-deposit while
  validation mode is active. This guard is in the pipeline, so an environment
  setting of `LOCALLIFE_AUTO_DEPOSIT=true` cannot bypass it.
- Kept both cameras active. RealSense remains the only authority for physical
  dimensions and final metric volume; Logitech remains a secondary RGB,
  colour, material, and optional comparison source.
- Extended the rigid cuboid path to box/carton/parcel/package labels in
  validation mode. Other household objects continue to use the general
  support-plane footprint/height estimator.
- Added a validation-only 10 mm minimum-height gate so the annotated 20 mm
  laptop sleeve can be measured. Production waste mode keeps its existing
  25 mm depth-noise gate.
- Added a prominent dashboard mode banner, `TEST OBJECTS SEEN`, explicit
  `WASTE LEDGER DISABLED` status, and raw detector labels beside the generic
  `measurement_object` class.
- Added `scripts/validate_reference_dimensions.py`. It reads all nine rows in
  `parameterised_objects/reference_objects.csv`, including the three objects
  that are production waste negatives, and records median RealSense L×W×H plus
  signed/percentage dimension errors. Missing deformable-object liters remain
  missing; the tool makes no volume claim for them.
- The shipped `cloud.env.example` selects validation mode for this temporary
  phase. Returning one line to `LOCALLIFE_OPERATING_MODE=waste` restores the
  production classifier without reverting code.
- The Windows one-click launcher now carries a validated `-OperatingMode`
  parameter through its child windows and remote start command. Its temporary
  default is `geometry_validation`, so the normal double-click workflow
  actually runs this phase even though that launcher does not source
  `cloud.env`; pass `-OperatingMode waste` to restore production behavior.

### Validation completed

- Focused configuration, classification, filtering, RealSense geometry,
  dual-camera, tracking, ledger, reference-manifest, and color suite: 219
  tests passed.
- The additional OpenCV-only local-offline test module could not import in
  this WSL interpreter because `cv2` is not installed. No assertion in that
  module ran; the friend's installed local environment must run it.
- Live hardware accuracy is still pending. A valid trial requires a current
  build badge, an empty/stable RealSense baseline, and exactly one still
  reference object in view.

### Known limitations

- The prompt bank covers common household test items rather than every object
  category that could exist in a home. Extra categories can be supplied with
  `LOCALLIFE_VALIDATION_PROMPTS` without changing production waste prompts.
- Single-view RealSense dimensions still depend on a clean support plane,
  aligned depth, and a complete instance mask. This version creates a safer,
  broader test protocol; it does not claim hardware accuracy before the logged
  physical trials are run.

## v4 — 2026-09-13 — Parameterised-object correction pass

### Status

- Implemented in the local workspace as build `19.19.0-local-ai`; not deployed
  to the friend's laptop and not yet verified with a physical RealSense frame.
- No camera firmware, exposure, laser, USB, or other hardware setting changed.

### Evidence reviewed

- Reviewed all nine ruler-annotated photos in `parameterised_objects/` and all
  seven September 12 result screenshots in `images/`.
- The screenshots show an older runtime: they render raw labels such as
  `white garbage bag` and `unclassified object`, whereas the current source
  displays only canonical `plastic bag`, `paper bag`, or `cardboard box`
  labels and suppresses unclassified foreground.
- The live status in the screenshots reports zero baseline frames. In the
  white-bag case, the semantic mask covers only a central patch/logo, which
  explains why distance can remain plausible while height and footprint are
  severely undersized.

### Changes

- Added support-plane-anchored measurement-mask recovery. RealSense may expand
  an accepted semantic seed to the connected physical surface elevated above
  the fitted floor, but it cannot create a waste object without an accepted
  plastic/paper/cardboard detection. Expansion is bounded and cannot jump to
  an unrelated raised component elsewhere in the scene.
- Recomputes colour from the recovered physical surface, reducing decisions
  based on a logo, shaded centre patch, or carpet halo.
- Added `paper shopping bag`, `milk carton`, and `drink carton` positive
  prompts; added soda/aluminium/tin can negatives for the Pepsi-can failure.
- Converted the four measured rigid boxes/cartons into enabled geometry
  templates. Their reference liters are external L×W×H cuboid volumes, not an
  assumption about printed liquid capacity.
- Added `parameterised_objects/reference_objects.csv`; backpacks, laptop bags,
  and the fabric laundry hamper are explicit negative controls. Deformable bag
  liters remain blank until an independently known filled volume is supplied.
- Added a visible `Build 19.19.0-local-ai` badge and `build_version` API field so
  a stale deployed copy can be identified directly from the screen/state.

### Known limitations

- Static RGB screenshots do not contain raw aligned RealSense depth, camera
  intrinsics, or per-frame masks, so they cannot prove physical dimension
  accuracy. A hardware retest with one still object at a time is required.
- The dimensions encoded in the filenames are now treated as ruler ground
  truth. No correction factor was fitted to those same objects.
- Polythene volume calibration remains intentionally pending until known
  filled volumes are supplied; bounding dimensions alone are insufficient.

### Validation completed

- Reference manifest audit: 9 objects, 6 accepted references, 3 negative
  controls, and all 4 rigid L×W×H volume calculations verified.
- Focused detection, colour, geometry, dual-camera, tracking, and Phase 1B
  validation suite: 201 tests passed.
- A broader 219-test run passed 217 tests. The two failures are existing
  patchy-depth morphology tests in the environment without OpenCV; the live
  camera installation uses the OpenCV path. They were not introduced by v4.
- Real-camera numerical accuracy remains pending on the friend's laptop.

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
