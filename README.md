# LocalLife One-Click Demonstration — v19.21.0

**v19.21.0 — built against the Final Implementation Playbook; the volume number is
finally right.** The playbook freezes one method (RealSense metric depth + a one-time
empty-bin reference + 2.5D height-map grid integration + before/after deposit difference)
and forbids rebuilding the rest of the project. That is exactly what this round did: the
detector, segmentation, dashboard and launcher are untouched.

**The volume bug, and it was two bugs compounding.** Every volume mode this project has
ever shipped integrated once per *pixel*, weighting each pixel by `z²/(fx·fy)`. That is
the footprint of a ray meeting a surface square-on — and a crumpled polythene bag, the
thing this system exists to measure, is mostly oblique micro-facets where the true
footprint is larger by `1/cos θ`. On top of that, the `reference-plane` default (chosen
in round 16) anchored that footprint to the **reference** depth rather than the depth the
object was actually seen at, so anything standing proud of the bin floor had its footprint
inflated by `(z_reference / z_object)²` — **+78% for a 0.5 m object in a 2.0 m bin**. This
was not hidden: `test_known_volume_validation.py` had measured the bias at 25.44% for its
own scene and recorded it as understood-and-accepted rather than fixing it. Between the two,
every liters figure this rig produced was biased high, by an amount that changed with how
tall the object was — which is why no amount of recalibration ever made it stick.

**The fix, per playbook §3–§7.** New `locallife_cloud/heightmap_volume.py` backprojects
depth to 3-D, takes each point's perpendicular height above the calibrated floor plane,
bins those points into fixed **10 mm cells laid out on that plane**, and integrates a robust
**median height per cell × the constant cell area**. Cell area no longer depends on where
the surface is, camera tilt cannot distort the grid, and tens of thousands of noisy samples
collapse into a few hundred medians — which is precisely why it survives a wrinkled bag.

**Measured against closed-form synthetic ground truth** (a rigid box at 0°/15°/30° of
mounting tilt, a smooth dome, and a wrinkled dome with 4 mm depth noise and 12% dropout):

| | old per-pixel | new height-map grid |
|---|---|---|
| mean absolute percentage error | **15.4%** | **1.1%** |
| crumpled bag + noise, level | +23.4% | −0.2% |
| crumpled bag + noise, 25° tilt | +13.4% | −1.3% |
| rigid box, 0° | +31.6% | +4.0% |

Three more real problems were found while building it, each fixed and regression-tested:
hole cells are now filled from the **lower quartile** of their neighbours rather than the
median (most holes beside an object are its own occlusion shadow, and a median resolved
them to the object side, fabricating a full-height ring around a reference box — +12% → +4%);
the grid **coarsens itself** when the depth image cannot support the configured cell size,
instead of silently returning nothing; and the settle trigger uses a high percentile rather
than the median, because a bag still falling covers well under half the bin ROI and the
median read exactly **zero** change while it was visibly moving.

**Colour (§11).** Classification already ran on masked pixels only; what was missing was
honesty about support. It now reports the share of the object's own pixels that agree with
the answer as `color_confidence`, and returns **UNKNOWN** rather than forcing a colour when
no class holds 35% of them. The per-pixel classifier was vectorised in the process, so this
costs no extra time per frame.

**Mis-sorting (§12).** New `locallife_cloud/sorting_rules.py` — a deterministic table over
the labels the existing detector already produces, no new model. Slippers, tools, appliances,
furniture and textiles are named as MIS-SORT (a disallowed family beats the word "bag", so
"vacuum cleaner bag" is not waved through); anything unrecognised or low-confidence is
**UNKNOWN / MANUAL CHECK**, never guessed. Disallowed objects still never enter tracking,
measurement or the ledger — they now raise a visible mis-sort warning instead of vanishing.

**Per-deposit incremental volume (§5, §10).** The bin no longer has to be emptied between
bags: each deposit records `volume_before_l`, `volume_after_l` and `added_volume_l` from the
change in total bin occupancy across the object's arrival. Purely additive — it changes no
deposit decision and no per-object measurement.

Tunables live in one place (`config.py` / `cloud.env.example`, playbook §27):
`LOCALLIFE_VOLUME_GRID_SIZE_M`, `LOCALLIFE_VOLUME_MIN_POINTS_PER_CELL`,
`LOCALLIFE_VOLUME_CELL_PERCENTILE`. The old per-pixel modes remain available via
`LOCALLIFE_VOLUME_GEOMETRY` for the thesis sensitivity comparison.

**46 new tests** (25 height-map, 15 colour/sorting, 6 end-to-end event record), full suite
**338 passing**. **Not yet verified on real RealSense hardware** — every number above comes
from scenes with exact derived ground truth, which is the strongest check available without
the rig. Per playbook §17 the next step is Day 1's: capture the empty-bin baseline, put one
measured rigid box in front of the camera, and confirm the terminal's litres before trusting
the dashboard.

# LocalLife One-Click Demonstration — v19.18.0 (previous)

**v19.18.0 — the root cause of "pending hamesha aa raha hai", finally found.** Your screenshots were
telling us the answer in plain text the whole time and I kept looking past it. Every single liters cell,
both cameras, every object, read:

> `pending — pending empty baseline`

That string comes from one place in the code: `reference_realsense is None` — **no empty-scene baseline
had ever been captured**. And everything about volume was gated behind it:

- the support plane was fitted *only* from a captured empty-scene depth frame → `reference_plane` stayed
  `None` → `estimate_box_volume_cuboid()` returned `None` on its very first guard, so `Box geometry`
  showed `—`;
- `estimate_volume()` (the per-pixel path used for bags) had no reference surface to subtract from → no
  liters at all.

So the cuboid model, the multi-frame aggregation, the tracking fix, the peer-label fix — all of it was
downstream of a gate that never opened on your rig. That is why round after round changed nothing you
could see. Capturing that baseline requires the measurement area to look genuinely *empty* for several
seconds, and your real room — bag already in shot, chair, blanket, clutter — is never empty. The system
was, in effect, waiting forever for a condition that would never happen.

**The fix (build spec §9.1, which prescribes exactly this):** the support plane no longer needs a
captured empty scene. New `fit_support_plane_from_background()` fits it from the **live frame's own
background** — the floor visible *around* the object, with the object masks dilated by ~12% and
excluded, so the noisy segmentation fringe never contaminates the fit. From that plane, new
`synthesize_plane_depth()` then *computes* the empty-floor reference in closed form instead of requiring
it to be captured: with the plane `z = a·x + b·y + c` and the pinhole ray, the reference depth at every
pixel is exactly `z = c / (1 − a·(u−ppx)/fx − b·(v−ppy)/fy)`. No iteration, no approximation, no empty
scene. That single change unblocks **both** paths — the cuboid box model and the per-pixel bag integral.

A genuine captured baseline is still strictly preferred and is **never** overwritten; this only fills in
when none exists. Anything measured this way is flagged `live_fitted_support_plane`, and a warning says
so, so the diagnostics stay honest about where the reference came from.

4 new regression tests drive the pipeline with `set_baseline()` **never called** — the exact reported
situation — and assert a box gets real L×W×H (25 cm object recovered within 20–30 cm), a bag gets real
liters, and no detection reports `pending-empty-baseline`. Reverting the fix makes them fail with
literally `AssertionError: 'pending-empty-baseline' == 'pending-empty-baseline'` — the same string from
your screenshots. Full suite: **336/336**.

**On the Logitech distance being ~2× RealSense's (0.47 m vs 1.06 m):** that is expected, not a new bug.
The C920 has no depth sensor; its distance comes from a monocular AI model whose absolute scale is
mathematically ambiguous until it is given one real measured distance — which is why the panel itself
says "monocular AI depth requires a measured reference distance". It is also why both your spec and this
project's own design use RealSense as the *only* metric sensor and Logitech purely for colour, material
and object type. Logitech's distance is not used for volume anywhere, so it cannot be the cause of the
pending liters.

**On training a model on 500+ images:** worth saying plainly — your detector is already doing its job
correctly in every screenshot ("filled plastic waste bag", "black trash bag", "large garbage bag",
"cardboard shipping box", material "polythene bag (100%)"). Detection was never what was failing, so
training data would not have moved the pending-liters problem at all. If accuracy still needs work after
this build produces real numbers, that is the point at which extra training data becomes the right
lever — with measured ground truth from your own 1 L / 2 L / 5 L references, not scraped images.

# LocalLife One-Click Demonstration — v19.17.0 (previous)

**v19.17.0 (the real "volume estimate hi nhi ho rahi" cause, found and fixed):** You sent the v19.12.2
launcher files back with "isme vhi volume estimation ka problem aa raha hai — volume estimate hi nhi ho
rahi." Two separate things were found by reading the code against your own screenshots.

**1. The actual volume blocker — RealSense's own label was gating the geometry.** Your dashboard
screenshots show the whole failure precisely: RealSense reported the carton as `#2 unclassified object`
(a depth silhouette its own detector never labelled) with `Box geometry (RealSense only)` showing `—`,
while **Logitech, in the very same frame, correctly called it `parcel box` / `cardboard shipping box`**.
The table-relative cuboid measurement — the entire L×W×H model built for exactly this object — was gated
on `_is_box_label(detection.label)`, i.e. on **RealSense's own** label. With that label missing, the
geometry path was skipped silently, the "Box geometry" column stayed empty, and the reported liters fell
back to the generic per-pixel height×area integral over a mask that can bleed into background — which is
how a roughly 1 L carton came out as **7.833 L**. That is not "volume is inaccurate", it is "the accurate
volume model never ran at all", which is exactly what you were describing.

The fix follows this rig's own design instead of working around it: Logitech is the designated
appearance/classification camera (it has no metric depth and never contributes geometry), so its
object-type call now opens RealSense's cuboid geometry path — while **every millimetre still comes
exclusively from RealSense depth**. `comparison.py` computes a `peer_box_present` signal from the peer
camera's own confirmed box-family detections (with the same 2.5 s grace window `peer_bag_present`
already uses, since either detector can miss a frame) and threads it through `process_precomputed` →
`_assemble`. Any measurement whose object type came from the peer is flagged `peer_labelled_box` in its
own diagnostics, so a reader can always tell. Phantom/silhouette detections are deliberately **not**
excluded from this path: a phantom is "a region no neural label vouched for", but when the peer camera
has independently confirmed a box in that frame it is no longer unvouched-for — that is the same
reasoning `allow_unclassified` already applies to `peer_bag_present`. This opens the *geometry* path
only; phantom exclusion from the durable ledger and from aggregate volume totals is enforced separately
and is untouched.

**2. A real diagnostic gap that made every previous round harder than it needed to be.** The version
banner Window 1 prints is a hardcoded string inside `Start-LocalLife-Demo.ps1` — it says which
**launcher** is running and can say nothing about the Python code doing the actual work. Those two can
genuinely diverge (`pip install -e .` records one specific directory; several extracted copies of this
package can sit side by side in Downloads; and `pip show` succeeding makes the install step skip
entirely). Window 1 now prints `Package in use: <version> from <path>` — the package Python *actually*
imports — and, if that path is not the copy the launcher was started next to, warns loudly and
re-points the editable install at the correct one. From now on any pasted log answers "which build is
really running?" directly instead of it having to be guessed. **This is a robustness and diagnostics
fix, not the cause of your volume problem** — because the server is started with its working directory
set to the project root and run via `python -m`, the correct copy was in fact winning at import time.

4 new regression tests (`PeerLabelledBoxGeometryTests`) reproduce the exact reported case end to end
through the real pipeline — RealSense's detector returning nothing, the object surviving only as a
scene-fusion phantom, the peer camera confirming a box — and were confirmed to **fail against the old
gate and pass with the fix**, plus guards for the other direction (no peer box label → no cuboid on
depth evidence alone) and that the pre-existing own-label path is unaffected and never mislabelled as
peer-sourced. The launcher's new probe helper was exercised in a real installed PowerShell 7.4.6 against
the actual function extracted from the launcher file (a good probe, a failing import, and a missing
interpreter — all three return cleanly without ever stopping the launcher). Full suite: **332/332
passing** (328 previous + 4 new).

**Also worth knowing, since you sent v19.12.2:** the files you uploaded are the **v19.12.2** package —
four rounds of volume work older than what had already been delivered. v19.12.2 predates the entire
table-relative cuboid model (v19.13.0), the box templates and the yellow-vs-grey colour fix (v19.13.0),
multi-frame aggregation (v19.14.0), the background-bleed tracking fix (v19.15.0) and everything above.
If a folder named like `LocalLife_One_Click_Demo_v19.12.2` is still what you are double-clicking, none
of that work is running at all — check Window 1's new `Package in use:` line to confirm which build you
are actually on.

**Note on the Raspberry Pi:** the Pi runs its own copy at `~/LocalLife_Plug_and_Play_Local` and the
launcher only starts it (`python3 -m locallife_cloud.edge_client` from inside that folder) — it does not
refresh it. So the RealSense depth-scale guard and post-processing filter chain added in v19.16.0 only
reach the Pi once that folder is updated there.

# LocalLife One-Click Demonstration — v19.16.0 (previous)

**v19.16.0 (audit against a new "LLM-Ready Proven Volume System" build spec v5; RealSense depth-scale
guard + expanded post-processing filter chain):** You sent a new, very detailed 8-page spec PDF (build
specification v5) plus "same problem aa raha hai bhai" and asked to rebuild against it, prioritizing its
P0-P5 (disable raw mesh/OBB volume, table plane, table-relative height, footprint, cuboid formula,
multi-frame aggregation) before templates/color/two-view/bags. Rather than rebuild blindly into the PDF's
suggested from-scratch `project/app/...` layout, `volume.py`/`geometry.py`/`pipeline.py`/`types.py` were
read directly and checked against each P0-P9 item first — this project's own established practice (see
rounds 12/16/20 below), since several earlier rounds already built most of this:

- **P0-P5 (raw mesh/OBB disabled as final volume, table plane, table-relative height, footprint, cuboid
  formula, multi-frame median aggregation): already fully built**, across round 13 (tilt-corrected
  perpendicular height), round 16 (`estimate_box_volume_cuboid()` — table-relative L×W×H, robust
  top-percentile height, PCA footprint), and round 20 (`aggregate_box_measurements()` — per-track median
  L/W/H over accepted frames, one final `L*W*H` multiplication, `dimension_instability` flagging). None of
  this needed rebuilding — re-verified correct and unchanged. This confirms your "same problem" report
  was the tracking/background-bleed bug fixed in v19.15.0, not a volume-math gap.
- **P6 (known 1/2/5 L templates) and P7 (yellow-vs-grey color fix): already built**, round 16 —
  `box_templates.py`/`box_templates.yaml` (measured-dimensions-only, never invented) and
  `_lab_b_channel()`'s corrected OpenCV LAB 128-offset fix.
- **The spec's separate-measurands principle (never call external geometric volume and nominal capacity
  the same "volume"): already satisfied structurally** — `BoxVolumeMeasurement.to_dict()` reports
  `volume_liters` (measured cuboid geometry) and `template_nominal_volume_liters` (the matched template's
  label capacity) as two distinct fields, not one.
- **Genuinely new gap found and fixed: RealSense depth scale had no runtime sanity check.** The spec
  explicitly requires "verify the RealSense depth scale at runtime... a factor-of-1000 error must fail
  loudly." `edge_client.py` read `depth_sensor.get_depth_scale()` from the real device and used it
  directly with no validation. New `_validate_depth_scale()` rejects a non-finite/non-positive scale or
  one outside the plausible ~1e-5-1e-2 m/unit band for a real D400-series sensor, raising clearly instead
  of silently producing wrong-by-orders-of-magnitude depth/height/volume numbers downstream.
- **Genuinely new gap found and fixed: the post-processing filter chain was missing stages the spec
  (citing Intel's own documentation) calls for.** Only spatial + temporal filtering were applied, and
  directly in the depth domain. Checked Intel's own post-processing-filters documentation directly (it
  does **not** actually specify where alignment belongs relative to filtering, contrary to what the
  spec's pseudocode implies) before changing anything: alignment stays where it already was (matching
  Intel's own official `align-depth2color.py` reference example) since moving it has no documented basis
  and risks a regression, not a confirmed fix. Spatial and temporal filtering were moved into the
  disparity domain (the SDK's own recommendation for those two filters specifically) and hole-filling was
  added — both previously entirely missing. Decimation was deliberately **not** added: it changes the
  depth frame's resolution, and this codebase's pixel-correspondence architecture (mask indexing, ROI
  pixels, pinhole backprojection) assumes depth and the color-aligned frame share one pixel grid — adding
  it without re-deriving intrinsics and resizing every consumer would silently misalign depth and color,
  a much larger, separately-scoped change not undertaken here.
- **Deliberately not built this round, per your own stated priority ordering** ("Uske baad known 1/2/5 L
  templates, color correction, two-view mode aur garbage bags" — already-built items aside): two-view
  fusion (rotate object ~90° on the tray, register/merge two RealSense views), the separate garbage-bag
  heightmap branch's own median+MAD multi-frame reporting (boxes already have this via
  `aggregate_box_measurements`; bags still use the existing per-pixel `estimate_volume()` path), mesh
  watertight/manifold validation (moot in practice since `mesh_used_for_final_volume` already defaults
  `False` everywhere and nothing currently sets it `True`, so mesh is already never used for final volume
  — just not backed by an explicit validator function), and the spec's exact output-JSON key names
  (`nominal_capacity_liters`, `volume_tolerance_liters`, etc. — the existing field names carry the same
  separated meanings under this project's own established naming).

7 new tests (`tests/test_edge_client.py`): depth-scale validation (real scale passes; factor-of-1000-high
and -low both rejected; zero/negative/NaN/infinite rejected; a full `iter_realsense()` end-to-end
reproduction confirming a bad device scale stops iteration before any frame is yielded) and the expanded
filter chain (confirms the exact call order — disparity transform, spatial, temporal, back to depth, hole
filling — and that `filter_depth=False` still skips the whole chain). Full suite: **328/328 passing**
(321 previous + 7 new), independently re-verified from a fresh zip extraction. **Not verified on real
RealSense hardware** — the depth-scale guard and filter-chain change are both evidence-based (the SDK's
own official documentation, checked directly) but need confirmation that a real device's actual depth
scale and filtered output still behave as expected on your next run.

**v19.15.0 (backend round: background-bleed tracking bug fixed, model-size performance guidance):** After
re-downloading and running v19.14.0, you sent real dashboard screenshots and reported it running
"extremely slow" and "more inaccurate than before" — RealSense showing `#N unclassified object` boxes
that visibly bled onto unrelated background clutter (a couch cushion, a backpack, a rolling office chair)
instead of the cardboard box actually being tested, while Logitech mostly labeled it correctly. Traced to
a real, reproducible bug in `tracking.py`/`geometry.py`: once ANY bag/box had been confirmed once,
`fuse_scene_detections`'s `allow_unclassified` gate used to fire off a plain
`tracker.has_active_counted_track()` boolean — "is anything, anywhere, already counted?" — with no
requirement that the newly-promoted "unclassified object" region have anything to do with that counted
track. On a cluttered real room (a baseline captured before the couch/backpack/chair were in frame), that
let the single largest unrelated "changed" blob be promoted to its own live, tracked, displayed box every
frame — a different piece of furniture winning "largest" from frame to frame, which is exactly the
drifting boxes reported. It also meant two objects (the real one plus the phantom) were being tracked,
measured, and drawn every frame instead of one, a real contributor to the reported slowness. Fixed by
scoping that gate: `ObjectTracker.counted_track_boxes()` now returns the boxes of currently-counted
tracks, and `fuse_scene_detections` only promotes the largest unmatched region when it sits near one of
those boxes (overlapping it, or within one track-box-diagonal of its center) — bridging that *specific*
already-confirmed object through a brief detector dropout, exactly as originally intended, never licensing
an unrelated new object elsewhere in the frame. 3 new regression tests (including an end-to-end pipeline
test that reproduces the exact real-hardware scenario and is confirmed to fail against the old logic and
pass against the fix), full suite **321/321 passing** (318 previous + 3 new).

Separately, on the "extremely slow" report: `config.py` configures both AI models to their heaviest
variant — `yoloe-11l-seg.pt` (the **large** YOLOE-11 segmentation model) and
`Depth-Anything-V2-Metric-Indoor-**Large**-hf` for monocular depth. Both are sized for a real GPU; on a
CPU-only laptop (the dashboard's own "GPU initializes on first frame" badge staying blank suggests this is
the case here) they are inherently slow, and this was not changed or newly introduced by any recent round.
If Window 1's own timing/FPS output still looks slow after this fix, the single biggest lever available
without any code change is dropping to smaller model variants via environment variables before launching,
e.g. `LOCALLIFE_DETECTOR_MODEL=yoloe-11s-seg.pt` and
`LOCALLIFE_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (or the `-Base-` size for a
middle ground) — this trades some detection/depth precision for meaningfully faster CPU inference. This
has **not** been benchmarked on real hardware from here (no GPU/timing telemetry was shared), so treat it
as a lever to try, not a guaranteed fix.

**Also recommended (not a code change): recapture the empty-bin baseline for the current camera framing**
before testing again ("Capture empty-bin baseline" on the dashboard). The background-bleed bug above is
worst when the baseline was captured before today's clutter (couch, backpack, chair) was in view, since
depth/RGB "changed since baseline" is the only signal `detect_scene_objects` has — a stale baseline makes
far more of the frame register as "changed" every frame, which is both slower (more candidate regions to
evaluate) and more error-prone even with the scoping fix above.

**v19.14.0 (Python/backend round, launcher unchanged except its version banner):** You asked to pause
cloud/launcher work and focus on rebuilding and re-verifying the backend volume math so it actually
produces correct liters ("iske hisab sai build karna... vapas se reconsider mathematics ko kar ki piche
backend par sach mai hum kese volume ko calculate karenge... iss baar volume ana chahiye bhai sach mai").
`volume.py` was audited against a new, very detailed spec PDF first, before touching any code: the
table-plane fit, robust top-percentile perpendicular height, PCA footprint, and `L*W*H` cuboid formula
(all built in v19.13.0) were already correct and already verified against exact synthetic ground truth,
so none of that was rebuilt. Two real, evidence-based gaps were found and fixed instead: (1)
`cloud.env.example` shipped with `LOCALLIFE_BAG_ONLY=true` and zero box/carton prompt terms, so a rigid
1/1.5/2 L test carton could never be detected at all, no matter how correct the volume math was — now
`LOCALLIFE_BAG_ONLY=false` with cardboard/carton/parcel prompts added; (2) the box diagnostics the
backend already computed (L×W×H, confidence, template match) were serialized in the API JSON but never
shown on the dashboard — a new "Box geometry" column now renders them. The one genuinely missing spec
feature, track-based multi-frame median aggregation of L/W/H across several accepted frames into a single
final volume (never summing per-frame volumes), was built as `aggregate_box_measurements()`; wiring it in
surfaced and fixed a real bug where the aggregation silently never ran because it originally checked
`detection.track_id` before the tracker had assigned one. 9 new tests, full suite **318/318 passing**
(309 previous + 9 new), independently re-verified twice from a fresh zip extraction. **Not yet verified
on real RealSense/Logitech hardware or real cartons** — needs your next real test run, ideally after
measuring your actual 1/1.5/2 L boxes with calipers and filling in `box_templates.yaml`
(`measured: true`) for template-matched high-confidence results. See
`LocalLife_Plug_and_Play_Local/LIVEFIX_CHANGELOG.md`'s "LiveFix 6" section for the full writeup. **Note
on this delivery:** the previous v19.14.0 zip you received was accidentally just the inner
`LocalLife_Plug_and_Play_Local` project folder, missing this one-click launcher wrapper entirely — that
was the actual cause of "one click run everything wala system kaha gaya"; this zip restores it, unchanged
except its version banner, wrapped around the same updated project folder. **Update, same delivery:** the
very next real run of that restored launcher hit a genuine bug —
`python.exe: can't open file '...\LocalLife_Plug_and_Play_Local\pip': No such file or directory` during
the one-time package install. Root cause: `Invoke-NativeTolerantly`'s `$StdinLines` parameter (added in
v19.13.3 for the PuTTY host-key fix) had no explicit `Position`, so PowerShell's default positional
binder silently absorbed the pip call's leading `-m` argument into it instead of `$Arguments` — the
process that actually ran was `python.exe pip install -e . -q` (no `-m`), which Python reads as "run a
script literally named `pip`". Reproduced directly in a real installed pwsh, then fixed with
`[CmdletBinding(PositionalBinding = $false)]` on the function plus explicit `Position` only on
`$Executable` and `$Arguments`, so `$StdinLines` can now only ever bind by its `-StdinLines` name — exactly
how every real call site already uses it. Re-verified against the actual edited function extracted from
the launcher (not a rewritten copy): the pip call now correctly runs with `-m` intact, and the
gcloud/PuTTY `-StdinLines` call shape still works unchanged. Full suite still 318/318; no Python touched.
This zip includes the fix — no further action needed on your end beyond re-downloading.

**v19.13.3 (launcher-only patch):** v19.13.2's fix (switching gcloud's own `--strict-host-key-checking`
flag from `accept-new` to `no`) cleared the "Invalid choice" error, but your very next real run showed
the actual root problem was still there: Windows' bundled PuTTY (`plink.exe`/`pscp.exe`), which is what
`gcloud compute ssh`/`gcloud compute scp` shell out to on Windows, still popped its own interactive
"Store key in cache?" prompt on first connection — a prompt that flag never controlled in the first
place, confirmed this round: gcloud's own `ssh.py` hardcodes the PuTTY backend on Windows with no
supported way to switch to OpenSSH instead. With nobody able to answer that prompt, Window 1's SSH
project-check limped through after a delay and the very next `gcloud compute scp` upload failed outright
with `pscp: unable to open ~/: failure` repeated once per file. Fixed with the standard, verified
technique for automating this exact PuTTY prompt: all 5 real `gcloud compute ssh`/`gcloud compute scp`
call sites now pipe `"y"` into the process's own stdin (answering the prompt exactly as a person typing
it and pressing Enter would, caching the key so later calls to the same VM don't re-prompt), added
carefully to avoid a real PowerShell parameter-binding ambiguity found while building this (mixing a
pipeline-bound parameter with the existing `Invoke-NativeTolerantly` helper's remaining-arguments
parameter can silently misdirect a real argument, and — worse — under this script's own
`$ErrorActionPreference = 'Stop'`, a failed pipeline bind becomes a *crash*, not a warning; both were
confirmed directly in a real installed pwsh before relying on the final, safe pattern). Verified
end-to-end against a fake `gcloud`-like stub reproducing the exact prompt-and-failure sequence from your
log (extracted the real, edited functions straight out of the launcher, not a rewritten copy) — the SSH
check and scp upload calls both now succeed automatically, while a deliberately-unmodified call with no
answer supplied still correctly fails, confirming the fix is doing real work and not just papering over
the symptom. Full Python suite unaffected (309/309, no Python touched this round). **Not yet verified
against your actual Windows machine and real Google Cloud VM** — this matches the documented, standard
way to automate PuTTY's host-key prompt and was verified against a faithful stand-in for `plink`/`pscp`'s
exact behavior, but the real confirmation is your next run getting past the upload step.

**v19.13.2 (launcher-only patch):** The v19.13.1 zone-timeout fix worked exactly as intended on your
next real Cloud-mode run — the VM was found already running in `europe-west4-c` within 26 seconds, no
premature timeout — but it uncovered the very next step failing: `gcloud compute ssh`/`gcloud compute
scp` rejected the `--strict-host-key-checking=accept-new` flag added back in round 8.7 with
`ERROR: (gcloud.compute.ssh) argument --strict-host-key-checking: Invalid choice: 'accept-new'. Valid
choices are [ask, no, yes]`, blocking both Window 1's SSH project-check/upload/bootstrap and Window 2's
tunnel SSH. That flag was added specifically to skip an interactive "Store key in cache?" prompt from
Windows' bundled `plink.exe`, and was checked against Google's own reference docs at the time — but
those docs evidently don't match every gcloud CLI version in the field, and your real installed gcloud
only accepts `ask`, `no`, or `yes` for its own `--strict-host-key-checking` flag. All 5 real call sites
(`Start-AppRole`'s SSH check, its `scp` upload, its remote-bootstrap SSH; `Start-CloudTunnelRole`'s
tunnel SSH; `Initialize-PiCloudTunnel`'s VM-setup SSH) now pass `--strict-host-key-checking=no` instead —
the closest of your gcloud's own confirmed-valid choices to the original intent, since `ask` would just
reintroduce an interactive prompt in an unattended window. Tradeoff, documented in a code comment at the
fix site: `no` never prompts but also never verifies the VM's host key on any connection (vs.
`accept-new`'s pin-on-first-use); this is judged acceptable here because `gpu.py` can hand you a brand
new VM (and thus a brand new host key) on any zone move, which would otherwise mean manually clearing a
pinned key on every such move, and because the connection is already gated by your own gcloud/IAM auth,
not by host-key trust. This does **not** touch the separate, unrelated OpenSSH-native
`-o StrictHostKeyChecking=accept-new` calls used elsewhere in the same script for the Raspberry Pi's own
direct `ssh`/`scp` commands — that is a different program's flag, `accept-new` is a genuinely valid
OpenSSH value, and the user's error was specific to gcloud's own flag only. Verified: real `pwsh` 7.4.6
parser confirms the launcher clean; full Python suite unaffected (309/309, no Python touched this round).
**Not yet verified against a real end-to-end SSH/tunnel round-trip on your machine** — this fixes the
exact "Invalid choice" error from your log, but your next real run is what confirms the tunnel actually
comes up.

**v19.13.1 (launcher-only patch):** First real Cloud-mode run hit a genuine capacity problem —
`europe-west1-b` had no L4 GPU available, so `gpu.py` correctly did what it's designed to do: capture
the VM's disk as an image ("3-8 min") and hunt for a fresh zone with capacity. But Window 2 (the secure
tunnel) gave up after only 2 minutes with `DEMONSTRATION ERROR: The cloud VM zone was never recorded by
Window 1`, while Window 1 was still legitimately mid-image-capture, minutes away from succeeding. Fixed
properly, not just with a bigger blind timeout: the wait (now a shared `Wait-ForCloudZoneFile` helper,
used by both Window 2 and the Pi's tunnel setup) is raised to a realistic 40-minute ceiling matching
`gpu.py`'s own documented worst case for a forced zone move, prints periodic reassuring progress
messages so a long wait doesn't look frozen, and — the more important half of the fix — checks whether
Window 1's own process is still alive: if it already crashed or was closed without ever finishing, this
now fails immediately with a clear message instead of always waiting out the full timeout. The main
orchestrator's own health-check wait (`Wait-ForAppHealth`, Cloud mode) was raised from 8 to ~50 minutes
to match, since it can't succeed until both windows finish. Verified: control-flow tests for all three
scenarios (zone appears mid-wait, Window 1 already dead, Window 1 alive but genuinely timed out) pass
against a scaled-down reproduction of the real logic; the real `pwsh` 7.4.6 parser confirmed the
launcher clean; full Python suite unaffected (309/309, no Python touched this round). Also fixed: the
two `.cmd` launchers' header comments were stale version numbers left over from round 8 (`v19.7.0`,
`v19.8.3`) that never matched the real installed version — cosmetic only, but confusing; now say so
explicitly instead of showing a wrong number. **Not verified against a real zone-move completing
end-to-end** — this fixes the exact premature-timeout bug from your log, but the actual multi-minute
`gpu.py` recovery itself needs confirmation on your next real run.

**v19.13.0:** Both cameras are now confirmed streaming and detecting real objects on your hardware —
but real-hardware screenshots (milk boxes, bags, a UN3091 shipping box) showed volume was still wildly
wrong (a 1.5 L milk box reading ~62.6 L), and pale/washed-out yellow objects were showing as grey. Two
separate, real bugs, found by reading the code directly rather than guessing, plus a substantial new
capability from your uploaded `revised_dual_camera_volume_engineering_recipe.pdf`:
**(1) The volume bug — and a correction to what v19.12.0 actually fixed.** v19.12.0 claimed `ray-frustum`
(the default integration mode since then) was "mathematically exact regardless of tilt." That was wrong —
I hadn't checked whether its volume *formula* (not just its height field) actually used the tilt-corrected
height. It didn't: `ray-frustum`'s volume sum is computed directly from raw camera-Z depth
(`(reference_depth**3 - object_depth**3) / (3*focal_product)`) and never reads the perpendicular-height
correction at all, so the round-13 tilt fix, though real and tested, never reached the production liters
figure for RealSense's default mode. The default `volume_geometry` is now `reference-plane`, whose formula
genuinely multiplies the corrected height by a footprint area — `config.py` and `cloud.env.example` both
carry a corrected comment explaining this so the mistake isn't repeated.
**(2) A new table-relative box measurement, per the PDF's core recipe.** For any box-family detection
(label containing "box", "cardboard", "carton", or "parcel") on RealSense, a new
`estimate_box_volume_cuboid()` now measures three robust scalar dimensions instead of summing per-pixel
contributions: height above the fitted table plane (median of the top 10th-percentile band, matching the
PDF's exact prescription), and footprint length/width via a PCA-derived principal-axis projection with
percentile-trimmed extents (2nd-98th percentile, avoiding raw min/max noise). `volume = L * W * H`. This is
the PDF's prescribed fix for exactly your symptom — a good mesh/mask with a badly wrong number — and is
wired into `pipeline.py` as a targeted override for box labels only (bags and Logitech are untouched).
Also new: a box-template system (`box_templates.py`/`box_templates.yaml`) that can report a known box's
*nominal* volume once you've measured it — it ships with your 1 L / 1.5 L / 2 L milk-box entries as
`measured: false` placeholders with zeroed dimensions, deliberately **not** filled in with plausible-looking
numbers (per the PDF's explicit "never let code or an LLM invent real measurements" instruction). Measure
your actual boxes with calipers/a ruler, fill in `box_templates.yaml`, and this system will report exact
known volumes for template-matched boxes instead of a single-view estimate.
**(3) The yellow-showing-as-grey bug.** `dominant_color()`'s low-saturation gate returned grey/white/black
*before* ever checking hue — a pale, washed-out warm color (a cream milk carton, a translucent yellow bag)
naturally has low HSV saturation despite being visibly warm-toned, so it was swallowed into "grey" before
the yellow branch could see it. Fixed with a corrected LAB colorspace check (OpenCV encodes LAB's a/b
channels 0-255 around a 128 midpoint, not the conventional signed range — reading raw bytes without
subtracting 128 silently understates warm/cool tint) fused with the existing HSV path.
27 new tests (8 color, 18 box-cuboid/template, 1 pipeline-wiring integration test) all pass individually
and against exact synthetic ground truth; 3 pre-existing tests whose hardcoded expected values were tied to
the old `ray-frustum` default were updated with explanatory comments (never reverted the fix). Full suite:
309/309. **Not yet verified against real hardware for this specific round** — please run your milk-box
tests again and paste the new dashboard numbers (and, if you can, `box_dimensions_mm`/`box_volume_flags`
from the detection JSON) so this can be confirmed or iterated on further.
Deliberately **not** rebuilt this round, disclosed honestly: full per-frame RANSAC table-plane refitting
with explicit object-mask exclusion (still reuses the baseline-derived plane, which should be close enough
since the object was never in that baseline frame, but this wasn't independently re-verified this round);
two-view box fusion; a dedicated bag heightmap mode (existing bag volume paths are untouched, working as
before); the PDF's literal `project/app/capture/...` directory reorganization (kept the existing, tested
package layout to avoid destabilizing a working system); a FastAPI endpoint update to emit the new
`BoxVolumeMeasurement` JSON contract (`recipe_api.py` wasn't touched this round). Also unresolved: an
"8.45 m" RealSense distance reading visible in one of your screenshots for a milk box sitting right in
front of the camera — this is a separate, unexplained symptom I have not root-caused yet.

**v19.12.2:** The v19.12.1 `pkill` fix didn't fully clear the same "Device or resource busy" error on
the next real-hardware run. Root cause is more specific than "a stray process": killing an old
`edge_client` process (however it dies — SIGTERM, SIGKILL, a closed window, a dropped SSH connection)
does not run its own `pipeline.stop()`, so librealsense's USB-level session can stay stuck even after
the process is gone — killing the process alone was never guaranteed to fix this. `edge_client.py`'s
`iter_realsense()` now catches a `pipeline.start()` failure directly, issues a real
`device.hardware_reset()` (librealsense's own documented recovery for this exact error, not just
closing a handle) to every enumerated RealSense device, waits for it to re-enumerate on the USB bus,
and retries once before giving up to the outer retry loop. The launcher's `pkill` was also strengthened
to `-9` (SIGKILL, since SIGTERM can be ignored or blocked mid-syscall) with a longer wait. 8 new tests
(`tests/test_edge_client.py`) using a fake `pyrealsense2` module (the real SDK needs hardware and isn't
installable here) verify the reset-and-retry path, the no-device-found case, a still-failing-after-reset
case, and that a healthy start never pays the reset cost. Full suite: 282/282. **Not verified against
real hardware yet** — if this still doesn't resolve it, please paste the Pi window's own text (not just
the Windows summary), since that's the only place the actual camera error appears.

**v19.12.1:** Fixed a real-hardware launcher bug: "xioctl(VIDIOC_S_FMT) failed, errno=16, Device or
resource busy" on RealSense, and "Unable to open camera/video source" on Logitech, right after
opening the Raspberry Pi window. Root cause: `Start-LocalLife-Demo.ps1`'s Pi bootstrap already
kills a leftover SSH tunnel process before reopening it (`pkill -f "id_locallife_cloud"`), but never
did the equivalent for a leftover `edge_client` camera process — so if a previous demo run's window
was closed or errored out before the remote SSH command exited cleanly, that process kept running on
the Pi and kept both camera device nodes open, making the next run's camera open fail even though the
cameras were physically fine. Both Pi-launch code paths (Local and Cloud mode) now `pkill` any stray
`locallife_cloud.edge_client` process and give the kernel a second to release the USB device nodes
before starting a fresh one, matching the existing tunnel-cleanup pattern. If this still happens after
updating, it means something *other* than this launcher (e.g. `realsense-viewer` left open, or another
SSH session) is holding the camera — check with `fuser /dev/video*` on the Pi and close whatever
else has it open.

**v19.12.0:** Real-hardware testing showed RealSense distance was accurate but height/volume was not. Reading `volume.py` directly found why: its height field was the raw camera-Z difference between the baseline and current depth, which only equals an object's true vertical height when the camera is mounted exactly perpendicular to the bin floor — this project's actual rig (a wall/bracket mount, confirmed by your own uploaded thesis pilot deck) is essentially never perfectly overhead, so height (and every derived liters figure) was systematically inflated by `1/cos(mounting tilt)` — 10-40% for a plausible 25-30° tilt. `estimate_volume()` now optionally takes the already-existing fitted floor plane (previously only used for the Logitech tilt warning) and computes each pixel's true perpendicular height above it instead of the raw depth difference; the default volume geometry also changed to `ray-frustum`, the only one of the four integration modes whose volume is mathematically exact regardless of tilt. Fully backward compatible (a no-op for an already-level camera). Verified against a closed-form-exact synthetic 30°-tilted scene: naive height inflated ~15.5% as predicted, corrected height within 0.3% of true. Your Logitech-as-supporting-camera idea is close to what the existing "Fused Result" panel already does (it inherits this fix automatically); volpy was reviewed again with the same conclusion as before. See DEMONSTRATION_INSTRUCTIONS.md for the full writeup, including what is/isn't verified (real hardware confirmation is still pending your next test run).

**v19.11.0:** The recipe pipeline's internals rebuilt to match a new, far more detailed spec you pasted (`dual_camera_estimation_recipe_v3.pdf`, "Zero-Flaw Implementation Blueprint") — still the same additive, off-by-default second result alongside your real dashboard, but the six recipe modules underneath now follow v3's exact algorithms: pose normalization (rotate the point cloud so the table plane becomes world Z-up before any measurement, not after), a new plane-fit box volume method (top-face + front-face RANSAC, two-view accuracy target vs. a flagged single-view fallback) alongside the old OBB method (kept specifically as v3's own requested ablation baseline), a bag heightmap with genuine interior-hole convex-fill and size-class discretization/tolerance, a LAB (not HSV) color classifier, and a two-level CLIP-then-fallback-CNN material cascade. A new `recipe_config.yaml` holds every v3 tuning knob in one place. See DEMONSTRATION_INSTRUCTIONS.md for the full writeup, including three real bugs found and fixed while building this and what is/isn't verified (cloud API contract update and real-hardware validation are both explicitly deferred, per your own request to build local-only first).

**v19.10.0:** A separate, additive dual-camera "recipe" pipeline (RealSense-only Open3D volume via oriented-bounding-box/heightmap, Logitech-preferred 8-class color and 7-class CLIP material, generic YOLOv8n/11n-seg detection) built alongside the existing dashboard, exactly to the spec you pasted — never replacing the existing tracked/ledgered/color-mapped measurement, and **off by default**. New: `locallife_cloud/pointcloud_volume.py`, `recipe_color.py`, `recipe_material.py`, `recipe_detect.py`, `recipe_pipeline.py`, `recipe_api.py` (a standalone FastAPI endpoint returning your exact JSON schema — `volume_liters`, `volume_confidence`, `color`, `color_confidence`, `material`, `material_confidence`, `object_type`, `timestamp`). The existing dashboard also gained a "Recipe Result" panel showing this second estimate side by side with the Fused Result, but it stays blank/"disabled" unless you set `LOCALLIFE_RECIPE_ENABLED=1` — turning it on is a deliberate opt-in because the recipe's generic YOLO detector has none of the curated bag/box vocabulary's false-positive rejection, and it loads its own separate CLIP instance for material even when the dashboard's own material classifier is already loaded. A new `-Role RecipeApi` launcher role starts the FastAPI service standalone (`Start-LocalLife-Demo.ps1 -Role RecipeApi`, port 8100 by default) without touching the tested App/Pi/CloudTunnel flow. See DEMONSTRATION_INSTRUCTIONS.md for the full writeup, including two real bugs found and fixed while building this (a RANSAC-plane-vs-object-mask misidentification, and a degenerate-point-cloud qhull crash) and what is/isn't verified.

**v19.9.0:** Two changes, both about Cloud mode.

1. **Redesigned the cloud tunnel architecture to match the simple 3-tunnel shape you asked for:** laptop↔Pi (login only), laptop↔cloud (dashboard viewing only), and a new direct Pi↔cloud tunnel that carries the actual camera traffic straight from the Raspberry Pi to the GPU VM — it no longer relays through the laptop. The launcher provisions this itself: a dedicated keypair generated on the Pi, and a restricted, no-shell "locallife-tunnel" user on the VM that can only forward ports (nothing else). No extra credentials needed from you.
2. **Fixed a real bug this redesign also happens to close:** the previous Cloud mode bound the VM's server to `0.0.0.0` (the public interface) but never actually set the `LOCALLIFE_API_TOKEN` the server requires once it does that — so the server would have refused to start the moment Cloud mode was run against real infrastructure. (The v19.8.6 note below saying Cloud mode was "confirmed already compatible" was based on reading the code, not a live GPU run — this round is the first time that code path was actually traced end-to-end.) The VM now binds `127.0.0.1` only, reachable exclusively through the two SSH tunnels above, so no token is needed at all.

Also this round: Logitech's phantom full-frame bounding-box (visible in the pillow/backpack/banana-bag screenshots) is diagnosed as Depth-Anything-V2's monocular depth jittering across the background enough to read as "changed" — a jitter level RealSense's real stereo depth doesn't have. The scene-change detector now uses a higher, Logitech-specific height threshold (5 cm vs the shared 2.5 cm) before it will call a region "changed", cutting the false-positive background regions in a synthetic reproduction of the jitter pattern. This is a mitigation based on the diagnosed cause, not a live-hardware-confirmed fix — please re-run the same pillow/backpack/banana-bag tests and share new screenshots so it can be tuned further if the box is still too big.

**v19.8.8:** Cloud mode is now genuinely one-click — `gpu.py` (the collaborator's EU zone-hunting version) is bundled in this zip, and both Cloud mode and the Raspberry Pi now install the project themselves on a fresh machine instead of erroring "not installed". See DEMONSTRATION_INSTRUCTIONS.md.

**v19.8.7 (launcher-only patch):** fixed `DEMONSTRATION ERROR: The application did not become reachable` when Window 1 was actually fine, just slow to start (cold model download from Hugging Face took longer than the ~2 min the launcher was willing to wait). Local mode now waits up to ~6 minutes.

**v19.8.6:** real hardware confirmed working end-to-end. Fixed a false "116 L" reading (the implausible-volume safety cap was 120 L, sized for a wheelie bin, not a single tracked bag — lowered to 90 L). Height inaccuracy and RealSense/Logitech readings alternating both trace to the Logitech webcam's monocular depth needing a measured reference-distance calibration (dashboard field) — see DEMONSTRATION_INSTRUCTIONS.md. Cloud mode confirmed already compatible with the collaborator's new zone-hunting gpu.py, no code change needed.

**v19.8.5 (launcher-only patch):** fixed `DEMONSTRATION ERROR: The term 'if' is not recognized as the name of a cmdlet` in Window 2 — a PowerShell 5.1 parsing quirk in how the v19.8.2 token fix was written, only surfaced now that Window 2 actually opens (v19.8.4).

Fixes a volume-calibration bug: calibrating against a known-volume object could silently pick up a stray unconfirmed "depth silhouette" sharing the frame instead of your object's own reading, poisoning every later measurement on that camera. Adds `scripts/validate_known_volume.py`, a CLI you run against your own hardware with a known object (a 1 L cube, a 2 L box) to get real accuracy numbers (percent error, MAPE, median/90th-percentile error). Cloud mode is untouched this round — deferred, per request, while volume accuracy gets fixed.

**v19.8.1 (launcher-only patch):** fixed `does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found` — the project never shipped one; added `setup.py`.

**v19.8.2 (launcher-only patch):** fixed `Binding outside localhost requires LOCALLIFE_API_TOKEN` — the launcher now auto-generates a token each run and threads it to the server, the Pi uploader, and the dashboard automatically.

**v19.8.3 (launcher-only patch):** fixed a harmless-but-confusing recurring warning, `Cloud Storage synchronization failed: [WinError 2]` — Local mode now starts the server with `--disable-sync` since it doesn't need the cloud-sync thread at all.

**v19.8.4 (launcher-only patch):** fixed Window 2 (Raspberry Pi + cameras) never visibly appearing — on Windows 11 with Windows Terminal as the default terminal app, the second window was actually opening as a new TAB inside Window 1's terminal instead of its own window. Every window the launcher opens now goes through `conhost.exe` to force a real, separate, visible window every time.

**v19.7.1 (launcher-only patch):** fixed a startup crash on Windows PowerShell 5.1 (the built-in `powershell.exe` most laptops actually run) — `A positional parameter cannot be found that accepts argument 'LocalLife_Plug_and_Play_Local'`.

**v19.7.2 (launcher-only patch):** fixed a second startup crash — `Python was not found; run without arguments to install from the Microsoft Store...`.

**v19.7.3 (launcher-only patch):** fixed a third startup crash — `WARNING: Package(s) not found: locallife-cloud` surfacing as a fatal error during the completely normal first-run install step — plus, proactively, the same class of issue in the Cloud mode SSH/tunnel calls and an interactive host-key prompt those calls could hang on. See "Launcher fixes" below.

**Start here:** `DEMONSTRATION_INSTRUCTIONS.md` — full walkthrough for both modes, calibration steps, and troubleshooting.

## Quick start

- **Free, laptop-only (default):** double-click `START_LOCAL_LIFE_DEMO.cmd`
- **Cloud GPU (needs a Google Cloud account):** double-click `START_LOCAL_LIFE_CLOUD.cmd`
- **Stop either one:** double-click `STOP_LOCAL_LIFE_DEMO.cmd`
- **Check your setup first:** double-click `CHECK_LOCAL_LIFE_SETUP.cmd`

## What's in this folder

| File | Purpose |
|------|---------|
| `START_LOCAL_LIFE_DEMO.cmd` | One-click start, local laptop mode (free) |
| `START_LOCAL_LIFE_CLOUD.cmd` | One-click start, cloud GPU mode (via gpu.py) |
| `STOP_LOCAL_LIFE_DEMO.cmd` | Stop whichever mode is running |
| `CHECK_LOCAL_LIFE_SETUP.cmd` | Readiness check |
| `Start-LocalLife-Demo.ps1` | The launcher (both modes; `-Mode Local` / `-Mode Cloud`) |
| `gpu.py` | The multi-zone cloud GPU VM manager (from your collaborator's email) |
| `LocalLife_Plug_and_Play_Local/` | The fixed project source — install this on the Raspberry Pi (and, for Cloud mode, on the VM's disk image once) |
| `DEMONSTRATION_INSTRUCTIONS.md` | Full instructions |

## This round's fixes (v19.8.0)

Real-hardware feedback: distance is good, color/material are good, volume and height are not. The volume/calibration code was read end to end rather than guessed at.

**The bug:** the "Calibrate from this object" button never tells the backend what it measured for your object — it relies on the backend to look that up. That lookup was a separate camera-wide "current volume" total, not necessarily your object's own number; if an unconfirmed "depth silhouette" (a shadow, background noise — already excluded from counting/deposit in earlier rounds) shared the frame, its volume got folded in too. The resulting calibration factor then multiplies into every future reading on that camera, so one contaminated calibration click degraded accuracy going forward, not just once.

**The fix:** calibration now always uses the single confirmed object's own displayed reading, and refuses clearly if zero or more than one confirmed object is in view (or only a phantom is). The "CURRENT VOLUME" dashboard figure got the same phantom-exclusion fix. Everything else in the volume math (calibration factor reaching every camera and the fused result, the displayed height matching the same pixels the liters figure is integrated from, the liters formula's own unit conversion) was checked and confirmed already correct — no other bug found there.

**New tool:** `scripts/validate_known_volume.py` — run it against your own cameras with a known-volume object (a 1 L cube, a 2 L box, per your thesis proposal's own reference objects) to get real percent-error and MAPE/median/90th-percentile numbers instead of a guess. See `DEMONSTRATION_INSTRUCTIONS.md` for usage.

Verified via a hand-derived synthetic 1 L/2 L cuboid matched exactly against the pipeline's own pinhole-projection formula (within ±15% uncalibrated, ~1% once calibrated against that same object) plus the full test suite (208/208 passing, 11 new tests). Not yet confirmed against your real cameras or a real 1 L/2 L object — that's the next step, ideally via `validate_known_volume.py` itself.

Cloud mode untouched this round (deferred).

## This round's fixes (v19.7.0)

1. A modest object no longer gets reported as covering ~116 L of background — the scene-change mask no longer solid-fills the empty gap when it bridges to an unrelated nearby patch (most often a cast shadow).
2. Shadows are no longer detected as their own object — same fix as above.
3. Height is no longer wildly underestimated for dome-shaped/tapered objects (a 35 cm bag was reporting ~15 cm) — the dashboard now reports the 90th-percentile height from the same accepted volume measurement, not a raw whole-mask median.
4. Detection flicker between cameras is substantially reduced as a side effect of #1/#2 (a stable mask no longer flips detection thresholds frame to frame); the **Fused Result** panel remains the right place to read one combined number.

Verified against synthetic reproductions of each exact failure shape plus the full test suite (197/197 passing, 5 new tests this round). Not yet confirmed against your actual cameras — that's the next step.

## Launcher fixes

**v19.7.1:** The launcher crashed immediately at startup on Windows PowerShell 5.1 (the built-in `powershell.exe`, not PowerShell 7) with `A positional parameter cannot be found that accepts argument 'LocalLife_Plug_and_Play_Local'`. Cause: `Find-ProjectRoot`/`Find-GpuScript` called `Join-Path` with three path segments at once (`Join-Path $PSScriptRoot '..' $ProjectDirectory`); PowerShell 7 supports that, but Windows PowerShell 5.1's `Join-Path` only accepts two (`-Path`/`-ChildPath`), so the third segment had nowhere to bind and PowerShell rejected the whole call. Fixed by nesting two-argument `Join-Path` calls instead, which works on both versions.

**v19.7.2:** After the fix above, the launcher located the project correctly but then failed with `Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.` This is not a real error from our script — it's the literal text Windows itself prints when you run the fake `python.exe`/`python3.exe` "stub" that Windows always puts on `PATH` under `...\AppData\Local\Microsoft\WindowsApps\`, even when real Python was never installed. `Assert-PythonAvailable` was trusting the first thing named `python`/`python3` that `Get-Command` found, which is that stub whenever it comes first in `PATH`. Fixed: the launcher now rejects any candidate whose path contains `WindowsApps`, tries the official `py` launcher first (installed system-wide by python.org, not shadowed by the stub) to discover the real interpreter, and otherwise checks every `python`/`python3` match on `PATH` (not just the first) by actually running `--version` and requiring real `Python 3.x` output before trusting it. If no real interpreter is found anywhere, the error now clearly says so and points to python.org instead of surfacing Windows' own confusing Store-alias message.

**v19.7.3:** With a genuine Python install now correctly detected, the very next line failed with `DEMONSTRATION ERROR: WARNING: Package(s) not found: locallife-cloud` — for a completely normal first run where the package simply had not been installed yet. That "WARNING" line is pip's own, ordinary stderr output from the launcher's own `pip show` check (used specifically to decide whether to run the one-time `pip install`); under Windows PowerShell's `$ErrorActionPreference = 'Stop'`, that stderr text was being promoted into a script-terminating error before the launcher's own exit-code check ever ran, so the intended "not installed yet → install it" recovery path never got a chance to execute. New `Invoke-NativeTolerantly` helper wraps every native command whose stderr shouldn't be treated as fatal (pip show/install, the server process, `gpu.py`, and every `gcloud`/`ssh` call) — it relaxes error handling for just that one call, checks the real exit code afterward exactly as before, and falls back safely even if something still throws. While in there, also added `--strict-host-key-checking=accept-new` (gcloud) and `-o StrictHostKeyChecking=accept-new` (ssh) to every cloud/Pi SSH connection, since a fresh VM or a first-time Pi connection would otherwise show an interactive host-key prompt with nothing able to answer it in an unattended launcher window — this hadn't caused a crash yet by itself but was on track to (it was visible mid-transcript in the Cloud-mode test as a "Store key in cache?" prompt).

All three fixes confirmed via the PowerShell parser (no syntax errors) and by running the Doctor role end to end, which now correctly locates the project and a real Python interpreter again; the Python-stub fix was additionally verified with a standalone test simulating both "only the Store stub on PATH" (now throws our own clear message instead of surfacing Windows' raw text) and "a real interpreter present" (correctly preferred over the stub) cases; the stderr/`Invoke-NativeTolerantly` fix was verified end to end against a stand-in Python interpreter that reproduces the exact reported sequence (pip show fails with a stderr warning → pip install succeeds → server starts and even prints a benign warning of its own during startup) and confirmed it no longer aborts anywhere along that path.

## Cloud mode, in short

The old cloud launcher pointed at one fixed VM zone, which failed outright whenever that zone had no GPU stock (a real problem per the email — L4s were sold out nearly everywhere in Europe on a recent check). `gpu.py` hunts across 11 EU zones automatically (T4 fallback, optional US zones), and keeps the VM's disk — installed packages, cached models — intact across restarts and zone moves. `Start-LocalLife-Demo.ps1 -Mode Cloud` drives it and tunnels the dashboard back to your laptop exactly like Local mode, just with the model inference running on a cloud GPU instead.
