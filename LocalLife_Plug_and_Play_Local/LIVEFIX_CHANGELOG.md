# LiveFix 6 — box-volume math audit, config fix, and multi-frame aggregation

## Request

A detailed "LLM-Ready Dual-Camera Volume Fix Build Specification" PDF, requesting
a full audit and rebuild of the RealSense box-volume math: table-plane RANSAC,
table-relative coordinates, robust top-height, XY footprint, L*W*H cuboid volume,
template-assisted known cartons, multi-frame track aggregation, a mesh-not-used-
for-final-volume policy, and a Yellow-vs-Grey color fix -- with the immediate
target being reliable volume for rigid 1 L / 1.5 L / 2 L test cartons.

## Audit result: most of the spec was already built and already verified

Reading `volume.py`, `box_templates.py`, `geometry.py`, and `pipeline.py` against
the PDF line by line found that an earlier round ("Revised Dual-Camera Volume
Estimation" recipe) had already implemented almost the entire spec, independently
of this exact PDF, and had already proven it correct with exact-ground-truth
synthetic tests (`tests/test_box_cuboid_volume.py`):

- `fit_reference_plane()` fits the table/floor plane from the **empty-scene
  baseline** depth capture (not RANSAC on an occupied frame) -- stronger than
  the PDF's own §9.1 suggestion, since it can never mistake the object's own
  face for the table.
- `_plane_perpendicular_height()` / `estimate_box_volume_cuboid()` measure true
  table-relative height (never raw camera-Z), confirmed numerically stable
  across 0°, 15°, 25°, and 40° mounting tilts.
- `estimate_box_volume_cuboid()` computes robust top-height (90th-percentile
  band median, 98th-percentile logged for comparison -- never a raw `max`), a
  PCA-based XY footprint (a closed-form analogue of the PDF's rasterize +
  minimum-area-rectangle), and `volume_l = length * width * height * 1000`
  exactly per the PDF's own formula -- not a per-pixel integral.
- `box_templates.py` / `box_templates.yaml` already ship exactly the PDF's
  1 L / 1.5 L / 2 L template contract, unmeasured (`measured: false`) by
  design so nothing is ever matched against an invented dimension.
- `geometry.py`'s `dominant_color()` already fixes the Yellow-vs-Grey bug the
  PDF describes (§16): OpenCV's raw LAB `a`/`b` channels need a `-128`
  recentring the naive code was skipping, confirmed by an explicit comment
  and test coverage already in place ("this was silently returning grey for
  exactly that case (reported directly: a yellow bag/box labelled grey)").
- Mesh is never used for final volume anywhere in this pipeline -- there is no
  mesh-volume code path at all in the live measurement flow, so the PDF's §14
  policy already holds trivially.

None of this needed rebuilding. Rebuilding correct, already-tested geometry
from scratch on the strength of a new PDF alone -- without first reading what
was already there -- would have been a real risk of introducing a regression
into code more rigorously verified than most of this project.

## What was actually broken: a config gap, not the math

`cloud.env.example` (the shipped example configuration) had
`LOCALLIFE_BAG_ONLY=true` and a `LOCALLIFE_PROMPTS` list containing only
bag/sack vocabulary -- **zero** box/carton/parcel prompt terms, and
`LOCALLIFE_BAG_ONLY=true` additionally hard-filters out any non-bag label
regardless of prompts (`is_bag_detection()` in `pipeline.py`). Under that
config, a rigid box or carton could never be detected at all, however correct
`estimate_box_volume_cuboid()`'s own geometry is -- no detection, no mask, no
volume, ever. This matches "volume ana chahiye bhai sach mai" (the volume
genuinely never showing up) far better than a math bug would: a math bug
would produce a *wrong* number, not *no* number. Fixed by defaulting
`cloud.env.example` to `LOCALLIFE_BAG_ONLY=false` with both bag and
box/carton/milk-carton prompt terms; `BAG_STATION.md`'s specialized bag-only
station remains available by setting `LOCALLIFE_BAG_ONLY=true` deliberately.

Separately, the computed box diagnostics (length/width/height/method/
confidence/template match) were already serialized in the `/api/state` JSON
(`Detection.to_dict()`) but were **never rendered on the dashboard** --
the live-detections table only ever showed a bare "Liters" number, so a user
watching the actual dashboard had no way to see *why* a box's volume was
pending, low-confidence, or missing its plane/mask diagnostics, directly
contradicting the PDF's own "diagnostics are mandatory" principle. Fixed by
adding a "Box geometry" column to both live-detection tables in
`dashboard.py`, showing L×W×H, confidence, accepted/considered frame counts,
matched template (or method), and any flags, whenever a RealSense detection
has a cuboid measurement.

## New: multi-frame track aggregation (PDF §13)

The existing cuboid measurement was single-frame: each frame's own L/W/H was
computed independently, and only the final, already-multiplied liters number
was smoothed across frames (a rolling median). The PDF is explicit that boxes
should instead aggregate **dimensions**, not the pre-multiplied volume:
"Estimate L/W/H per accepted frame, aggregate dimensions using median, and
calculate final volume once. Never sum per-frame volumes." Implemented as
`aggregate_box_measurements()` in `volume.py`, fed by a new per-track history
of accepted single-frame `estimate_box_volume_cuboid()` results in
`pipeline.py` (`_box_measurement_history`, `_box_frames_considered`,
`box_aggregation_min_frames`/`box_aggregation_window_frames` in `config.py`).
Once a track has enough accepted frames, its reported length/width/height and
volume come from the median across the window, with a `dimension_instability`
flag and reduced confidence when the per-frame spread is too wide -- and a
`frames_considered` vs `frames_accepted` distinction, plus `dimension_std_mm`,
now flow through to `Detection.to_dict()` and the dashboard.

Wiring this correctly needed one real fix along the way: `detection.track_id`
is not assigned until `ObjectTracker.update()` runs, which happens *after*
every detection's own single-frame box measurement in `pipeline.py`'s
`_assemble()` -- so the aggregation bookkeeping cannot live inline in the
original per-detection measurement loop (confirmed directly: it silently
never accumulated any history there, `frames_accepted` stuck at 1 forever).
Fixed by stashing each frame's raw single-frame cuboid result by object
identity (`id(detection)`), then folding it into the now-track-known
aggregation history in a second pass immediately after `tracker.update()`.

## Testing

Nine new tests. `tests/test_box_cuboid_volume.py`'s `AggregateBoxMeasurementsTests`
(unit-level, synthetic `BoxVolumeMeasurement` histories): median-of-dimensions-
then-single-multiplication (not mean/median of per-frame volumes), a real
outlier-rejection case (one noisy-height frame among five clean ones), the
`dimension_instability` flag and confidence penalty, `frames_considered` vs
`frames_accepted`, and the diagnostics `to_dict()` shape. `PipelineBoxAggregationTests`
(full `VisionPipeline`, real multi-frame processing): a static box's dimensions
converge to exactly zero cross-frame spread, and -- the core practical claim --
one single noisy depth frame (implying an implausible 800 mm height) does not
drag the final reported height away from the five clean frames' true 300 mm
value once aggregation is active. Full suite: 318/318 passing (was 309/309).

**Not yet verified on the user's actual cameras and real cartons.** Please
measure your exact 1 L / 1.5 L / 2 L test cartons with calipers, fill in
`box_templates.yaml`, set `measured: true`, run with the corrected
`cloud.env.example` (or your own `cloud.env` with `LOCALLIFE_BAG_ONLY=false`
and box/carton prompts included), and check the dashboard's new "Box
geometry" column for the actual L×W×H/confidence/frame-count numbers.

# LiveFix 5 — measurement-synchronised final repair

LiveFix 4 correctly allowed unmeasured detections into the tracker, but the web
metrics still read `plant.observed_bags` and `plant.colors` from the
measurement-gated durable ledger. Therefore **BAGS SEEN** and the color table
could still show zero even though the bag had a track ID. LiveFix 5 adds an
independent session-seen summary and makes the dashboard consume it, while the
ledger remains thesis-safe.

This pass also fixes bounding-box-only detections integrating the full ROI,
ties overlays to their exact analyzed frame, batches the shared detector across
both cameras, vectorizes isolated depth-hole repair, applies color smoothing to
both cameras, and exposes exact pending measurement reasons. The complete suite
now contains 127 passing tests. See `MEASUREMENT_FIX_REPORT.md`.

# LiveFix 4 — earlier changes

## Root cause of "no volume, no colors" on the dashboard

The pipeline (`locallife_cloud/pipeline.py`) only ever ran the object
tracker on detections that had *already* passed
`_measurement_is_recordable()` — i.e. objects whose liters figure was
already trustworthy (RealSense baseline captured + enough depth
coverage, or Logitech reference distance/tilt verified). Any object
waiting on one of those calibration steps never received a `track_id`.

Without a `track_id` it was invisible to:
- color smoothing (`_color_history`) → color column stayed blank
- volume smoothing (`_volume_history`) → liters stayed blank
- `BAGS SEEN` / `BOXES SEEN` (`tracker.total_count`)
- the ledger's "observed" record

...even though the video overlay (`_annotate_frame` in `server.py`)
draws straight from the very same `latest.detections` list, so the
bounding box, distance, and even the color label were already being
drawn on screen (see the "plastic shopping bag [grey] | AI 0.99 m"
overlay with `#?` in place of an ID in the screenshots). The comment
above `LOCALLIFE_RECORD_ONLY_MEASURED` promises "Show uncertain
detections live, but never add them to experiment databases" — the
code did not actually do the "show live" half.

### Fix
`pipeline.py`: the tracker now runs on **every** confirmed bag/box
detection, so an object is seen / counted / colored the instant it is
recognized. Only the ledger write (`ledger.observe`, which is what
starts a permanent history/CSV record) stays gated behind
`_measurement_is_recordable()`, matching the documented intent.
Added a regression test
(`tests/test_plug_and_play.py::test_unmeasured_detection_is_visible_but_not_recorded`)
that asserts an unmeasured object still gets a `track_id` and
increments `automatic_count`, while still being excluded from the
ledger. All 124 existing tests still pass unmodified.

### What this does NOT change
None of the physical-measurement safety gates were touched: RealSense
still requires an empty baseline + intrinsics + depth coverage;
Logitech still requires (depending on config) a measured reference
distance and a mounting tilt under `LOCALLIFE_LOGITECH_MAX_TILT_DEG`
before it will *trust* a liters number. Those are correct, thesis-grade
checks — this fix only stops them from also hiding the object entirely.

## Two setup issues in your screenshots (not code bugs)

1. **Logitech camera tilt (44.8°) exceeds the 35° ceiling.** The
   dashboard is telling you this outright: "Logitech camera tilt is
   44.8°; mount it above the bin or restrict its region to the bin
   floor (maximum 35.0°)." Until the camera is mounted closer to
   overhead (or you set a Logitech ROI that only covers the bin
   floor), Logitech liters are intentionally withheld — this is the
   `logitech_max_tilt_degrees` gate in `config.py`, not a bug. You can
   raise it for a handheld demo via `LOCALLIFE_LOGITECH_MAX_TILT_DEG`,
   but understand the volume math gets less trustworthy the more
   oblique the camera angle is.
2. **The test object was a pillow, and pillows are deliberately
   blacklisted.** `pipeline.py` keeps a `NEGATIVE_WASTE_LABELS` set
   (`pillow, cushion, blanket, bedding, chair, ...`) specifically so
   the model doesn't count furniture as waste. Your tasseled pillow is
   correctly recognized as "pillow" by the open-vocabulary model and
   rejected — that's the system working as designed, not failing.
   Test with an actual bag or cardboard box to see real numbers.

## Smaller changes
- `dashboard.py` / `server.py`: dashboard polling tightened from
  1100ms/900ms to 550ms so live numbers update roughly 2x faster —
  this is the main source of the "website is a little delayed"
  feeling (the two cameras also share one GPU inference lock by
  design, so expect real inference latency of a few hundred ms per
  frame per camera; that part is not "lag", it's the YOLOE-seg +
  Depth-Anything-V2 model actually running).
- `dashboard.py`: the live-detections table now writes
  `pending — <reason>` in the Liters cell instead of a bare `—` when
  an object is tracked but not yet measurable, so it's obvious the
  object *was* seen.

## Confirms your architecture already follows volpy
`LOCALLIFE_VOLUME_GEOMETRY=triangulated-surface` in
`cloud.env.example`, implemented in `geometry.py`/`volume.py`, is
already a Delaunay-triangulation-based surface integration — the same
plane-equation double-integral approach as agu3rra/volpy, applied to
calibrated depth points instead of a static terrain survey. No
architectural change against that reference was needed; the gap was
purely the tracking bug above.
