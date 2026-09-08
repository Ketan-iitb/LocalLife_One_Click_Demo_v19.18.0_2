# One-Click Waste-Camera Demonstration (v19.18.0)

## What's new in v19.18.0 (the actual root cause of "volume pending hamesha")

**What the screenshots said.** Every liters cell, both cameras, every object: `pending — pending empty
baseline`. That exact string is produced in one place: `reference_realsense is None` — no empty-scene
baseline had ever been captured on this rig.

**Why that blocked everything.** Both halves of the volume system depended on that capture:
the support plane was fitted only from a captured empty depth frame (so `reference_plane` was `None`,
and `estimate_box_volume_cuboid()` bailed out on its first guard — hence `Box geometry` = `—`), and the
per-pixel bag path had no reference surface to subtract. Every geometry improvement shipped in earlier
rounds sat *downstream* of that gate, so none of it could show up. And the gate needs the area to look
genuinely empty for several seconds, which a real room with the bag already in shot never is.

**The fix.** Per build spec §9.1, the support plane is now fitted from the **live frame's own
background** — the floor around the object, with object masks dilated ~12% and excluded
(`fit_support_plane_from_background`). The empty-floor reference is then *computed* from that plane in
closed form rather than captured (`synthesize_plane_depth`): for plane `z = a·x + b·y + c`, the
reference depth at pixel (u,v) is `c / (1 − a·(u−ppx)/fx − b·(v−ppy)/fy)`. This unblocks both the box
cuboid model and the bag integral, with no empty scene required.

A real captured baseline still wins and is never overwritten — this only fills in when none exists.
Measurements taken this way carry the `live_fitted_support_plane` flag and a warning.

**Still worth doing when you can:** pressing "Capture empty-bin baseline" with the area clear remains the
most accurate option. It is just no longer mandatory for getting numbers.

**About the Logitech distance being roughly double RealSense's:** expected, not a bug. The C920 has no
depth sensor; its distance is a monocular AI estimate whose absolute scale is ambiguous until given one
real measured reference distance. It is never used for volume — RealSense is the only metric sensor, by
design and per the spec.

**Testing:** 4 new tests run the pipeline with `set_baseline()` never called and assert real box L×W×H,
real bag liters, and no `pending-empty-baseline`. Reverting the fix reproduces the exact screenshot
string as a test failure. Full suite: **336/336**. Not yet verified on your hardware — that is the next
run.

## What's new in v19.17.0 (previous)

## What's new in v19.17.0 (the real cause of "volume estimate hi nhi ho rahi", found and fixed)

**What your own screenshots showed.** RealSense: `#2 unclassified object`, `Box geometry (RealSense
only)` = `—`, volume `7.833 L`. Logitech, same frame, same object: `parcel box`, material `cardboard
(100%)`. So the object *was* correctly identified — just by the other camera — and RealSense's accurate
L×W×H model never ran, leaving a leaky per-pixel fallback to report ~7.8 L for a roughly 1 L carton.

**Root cause.** The table-relative cuboid measurement was gated on `_is_box_label(detection.label)` —
**RealSense's own** label. When RealSense only produced a depth silhouette (`unclassified object`), the
gate stayed shut, the "Box geometry" column stayed empty, and the volume fell through to the generic
per-pixel integral. The precise, spec-mandated geometry model was sitting there unused for exactly the
object it was built for.

**The fix.** Logitech is this rig's designated appearance/classification camera — it has no metric depth
and never contributes geometry. Its object-type call now opens RealSense's cuboid geometry path, while
every millimetre still comes exclusively from RealSense depth. A `peer_box_present` signal is computed
in `comparison.py` from the peer camera's own confirmed box-family detections (same 2.5 s grace window
`peer_bag_present` already uses) and threaded through to the measurement. Measurements whose object type
came from the peer are flagged `peer_labelled_box` so the diagnostics stay honest. This opens the
geometry path only — phantom exclusion from the durable ledger and from aggregate totals is enforced
separately and is untouched.

**Second fix: you can now see which build is actually running.** Window 1's version banner is a
hardcoded string in the launcher — it describes the *launcher*, not the Python code doing the work, and
the two can diverge. Window 1 now prints `Package in use: <version> from <path>` (the package Python
really imports) and warns + re-points the editable install if that path is not the copy you launched.
This is a diagnostics/robustness fix, not the cause of the volume problem.

**Heads-up:** the launcher files you sent this round were **v19.12.2** — older than the cuboid model
(v19.13.0), box templates and colour fix (v19.13.0), aggregation (v19.14.0) and the tracking fix
(v19.15.0). Check the new `Package in use:` line to confirm which build you're really on.

**Testing:** 4 new regression tests reproduce the exact case end to end through the real pipeline and
were confirmed to fail against the old gate and pass with the fix; the launcher's new probe helper was
exercised in real PowerShell 7.4.6 against the function extracted from the launcher itself. Full suite:
**332/332**. **Not verified on real hardware** — that needs your next run.

## What's new in v19.16.0 (previous)

## What's new in v19.16.0 (audited against build spec v5; RealSense depth-scale guard + filter chain)

You sent a new 8-page "LLM-Ready Proven Volume System" build spec (v5) and asked to rebuild against it,
prioritizing P0-P5 (raw mesh/OBB disabled, table plane, table-relative height, footprint, cuboid formula,
multi-frame aggregation) first. Audited the live code against every P0-P9 item before writing anything,
per this project's own established practice, rather than rebuilding blindly into the PDF's suggested
from-scratch layout:

**Already fully built, re-verified unchanged:** P0-P5 (table-relative cuboid volume, robust top-percentile
height, PCA footprint, `L*W*H` formula, per-track median multi-frame aggregation — rounds 13/16/20), P6
(known 1/2/5 L templates, measured-dimensions-only — round 16), P7 (yellow-vs-grey LAB color fix — round
16), and the spec's separate-measurands principle (`volume_liters` vs `template_nominal_volume_liters`
are already two distinct fields, never conflated). This confirms your "same problem" report traced back to
the tracking/background-bleed bug fixed in v19.15.0, not a gap in the volume math itself.

**Two genuinely new, real gaps found and fixed:**

1. **RealSense depth scale had no runtime sanity check.** The spec explicitly requires this ("a
   factor-of-1000 error must fail loudly"). New `_validate_depth_scale()` in `edge_client.py` rejects a
   non-finite/non-positive scale, or one outside the plausible ~1e-5 to 1e-2 m/unit band for a real
   D400-series sensor, with a clear error — instead of silently letting a wrong-by-orders-of-magnitude
   scale flow into every downstream depth/height/volume number.
2. **The post-processing filter chain was missing stages.** Only spatial + temporal ran, directly in the
   depth domain. Checked Intel's own post-processing-filters documentation directly first — it does
   **not** specify where alignment belongs relative to filtering, so alignment was left exactly where it
   already was (matching Intel's own official reference example) rather than reordered on the spec's
   pseudocode alone, which would risk a regression with no documented basis. Spatial/temporal filtering
   now runs in the disparity domain (the SDK's own recommendation for those two filters) and hole-filling
   was added. Decimation was deliberately **not** added — it changes the depth frame's resolution, and
   this codebase's pixel-correspondence architecture assumes depth and the color-aligned frame share one
   pixel grid; adding it without a much larger, separately-scoped rework would silently misalign depth
   and color.

**Deliberately not built this round** (per your own stated ordering — templates/color/two-view/bags come
after P0-P5, and the first two of those were already done): two-view fusion (rotate-90°-on-tray dual
RealSense views), a separate median+MAD multi-frame report for the garbage-bag heightmap branch (boxes
already have this via `aggregate_box_measurements`), an explicit mesh watertight/manifold validator
(moot in practice — `mesh_used_for_final_volume` already defaults `False` everywhere and nothing sets it
`True`, so mesh is already never used for final volume), and renaming output-JSON keys to match the
spec's exact suggested names (the existing names already carry the same separated meanings).

7 new tests in `tests/test_edge_client.py` (depth-scale validation, factor-of-1000 rejection in both
directions, zero/negative/NaN/infinite rejection, an end-to-end `iter_realsense()` reproduction, and the
new filter chain's exact call order). Full suite: **328/328 passing** (321 previous + 7 new). **Not
verified on real RealSense hardware** — both fixes are evidence-based (checked against Intel's own
documentation directly) but need confirmation on your next real run.

## What's new in v19.15.0 (background-bleed tracking bug fixed; slowness traced to model size, not new code)

**The report:** after v19.14.0, real dashboard screenshots showed RealSense drawing `#N unclassified
object` boxes on unrelated background clutter (a couch cushion, a backpack, a rolling office chair)
instead of the cardboard box under test, plus a report that the whole system was "extremely slow" and
"more inaccurate than before."

**Root cause found:** once any bag/box had been confirmed once, `fuse_scene_detections`
(`geometry.py`)'s permission to promote an "unclassified object" used to come from a plain
`tracker.has_active_counted_track()` boolean — "is anything, anywhere, already counted?" — with no check
that the new region had anything to do with that counted object. On a real, cluttered room (a baseline
captured before today's furniture was in view), that let the single largest unrelated "changed" blob
become its own live, tracked box every frame — a different piece of furniture winning "largest" frame to
frame, exactly the drifting boxes reported. It also meant two objects (the real one plus this phantom)
were tracked, measured, and drawn every frame instead of one — a real, measurable contributor to the
slowness, on top of the mislabeling.

**Fix shipped:** `ObjectTracker.counted_track_boxes()` (`tracking.py`) now returns the boxes of
currently-counted tracks; `fuse_scene_detections` only promotes the single largest unmatched region when
it sits near one of those boxes (overlapping it, or within one track-box-diagonal of its center) —
bridging that *specific* already-confirmed object through a brief detector dropout, exactly as originally
intended, never licensing a new object elsewhere in the frame. 3 new regression tests were added,
including an end-to-end pipeline test that reproduces this exact scenario and was confirmed to fail
against the old logic and pass against the fix. Full suite: **321/321 passing** (318 previous + 3 new).

**On "extremely slow" specifically:** `config.py` configures both AI models to their heaviest variant —
`yoloe-11l-seg.pt` (large YOLOE-11 segmentation) and `Depth-Anything-V2-Metric-Indoor-Large-hf` (large
monocular depth). These are sized for a real GPU and are inherently slow on a CPU-only laptop; this was
not changed by this round or by v19.14.0 — if the dashboard's "GPU initializes on first frame" badge never
fills in, everything is running on CPU. The single biggest available lever without a code change is
smaller model variants, set via environment variables before launching:
`LOCALLIFE_DETECTOR_MODEL=yoloe-11s-seg.pt` and
`LOCALLIFE_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (or the `-Base-` size for a
middle ground) — this trades some precision for materially faster CPU inference. **Not benchmarked on real
hardware from here** (no GPU/timing telemetry has been shared) — treat it as a lever to try, not a
guaranteed fix.

**Also recommended, not a code change:** recapture the empty-bin baseline ("Capture empty-bin baseline" on
the dashboard) for the current camera framing before testing again. The background-bleed bug above is
worst when the baseline predates today's clutter, since "changed since baseline" is the only signal the
scene-object detector has — a stale baseline makes more of the frame register as "changed" every frame,
which is both slower and more error-prone even with the scoping fix above.

## What's new in v19.14.0 (backend volume-math audit + config fix + dashboard diagnostics + multi-frame aggregation; launcher untouched except its version banner)

You asked to pause all cloud/launcher work and rebuild-and-reconsider the backend volume math first, so
it genuinely produces correct liters this time. Rather than rewrite `volume.py` from scratch, it was
read in full and audited against a new, very detailed spec PDF (`LLM_ready_dual_camera_volume_fix_
build_spec.pdf`): the table-plane fit via background-only RANSAC, robust top-10th-percentile
perpendicular height, PCA-projected footprint extraction, and the `volume_l = length_m * width_m *
height_m * 1000` cuboid formula were all already built (v19.13.0) and already verified against exact
closed-form synthetic ground truth — none of that needed rebuilding, and rebuilding already-correct,
already-tested code would only have added risk.

**What was actually broken, found by reading the code, not guessing from symptoms:**

1. **`cloud.env.example` structurally excluded box/carton detection.** It shipped with
   `LOCALLIFE_BAG_ONLY=true` and a prompt vocabulary with zero box/carton/parcel terms — correct for the
   specialized bag-only station in `BAG_STATION.md`, but it meant a rigid 1/1.5/2 L test carton (this
   round's and the spec's explicit priority target) could never be detected at all, regardless of how
   correct the volume math is. No detection, no mask, no volume, ever. Fixed: `LOCALLIFE_BAG_ONLY=false`,
   with `"cardboard box,cardboard shipping box,parcel box,carton box,milk carton,rigid box,"` added to
   `LOCALLIFE_PROMPTS`.
2. **Computed box diagnostics never reached the dashboard.** `Detection.to_dict()` already exposed
   `box_length_mm`/`box_width_mm`/`box_height_mm`/`box_volume_confidence`/`box_template_id` in the API
   JSON, but the dashboard's live-detections tables had no column for any of it — so a reported volume
   came with no visible explanation of *why*. Fixed: a new "Box geometry (RealSense only)" column now
   renders L×W×H, confidence, frames accepted/considered, and the active method or template match.

**The one genuinely missing spec feature: track-based multi-frame aggregation (PDF §13).** New
`aggregate_box_measurements()` in `volume.py`: given several single-frame measurements for the same
tracked object, takes the **median of length, width, and height independently** across frames, then does
exactly **one** final `L*W*H*1000` multiplication — never sums or averages already-computed per-frame
liters values. Flags `dimension_instability` (and discounts confidence) when any dimension's spread
across frames is too high. Wiring this into `pipeline.py` surfaced a real ordering bug: the first
attempt checked `detection.track_id` before the object tracker had assigned one (tracking runs later in
the same frame), so aggregation was silently a no-op every single frame. Fixed with a proper two-phase
bridge (`id(detection)`-keyed pre-tracking capture, then a post-tracking loop once `track_id` is known).

**Testing:** 9 new tests (`tests/test_box_cuboid_volume.py`) — 7 unit-level on the aggregation math
(median-outlier recovery, confidence discount on high spread, mesh never used for the aggregated
volume) and 2 pipeline-level, driving a real `VisionPipeline` through a synthetic multi-frame scene and
confirming dimensions stabilize with zero spread on a static object and a single noisy frame doesn't
drag the aggregated height off the true value. Full suite: **318/318 passing** (309 previous + 9 new),
independently re-verified twice from fresh zip extractions.

**Not yet verified on real RealSense/Logitech hardware or your real cartons.** Everything above is
verified against exact synthetic ground truth and pipeline-level integration tests, not a live camera.
Please re-test with this build using `LOCALLIFE_BAG_ONLY=false` (already the new default in
`cloud.env.example`), measure your actual 1/1.5/2 L milk-box cartons with calipers, fill in
`locallife_cloud/box_templates.yaml` (`measured: true`, real `dimensions_mm`) to unlock template-matched
high-confidence results instead of single-view cuboid estimates, and paste back the next real dashboard
screenshot/log. See `LocalLife_Plug_and_Play_Local/LIVEFIX_CHANGELOG.md`'s "LiveFix 6" section for the
complete engineering writeup.

**Note on this delivery.** The zip you received right after this round's backend work was accidentally
just the inner `LocalLife_Plug_and_Play_Local` project folder on its own — no `Start-LocalLife-Demo.ps1`,
no `.cmd` files, nothing to double-click. That is exactly why the one-click launcher looked like it had
disappeared. This zip is the fix: the full `LocalLife_One_Click_Demo` wrapper (this file, README.md, the
launcher, `gpu.py`, all four `.cmd` entry points) around the same updated, already-tested project folder.
The launcher itself has no functional changes this round — only its startup banner text now says
`v19.14.0` instead of `v19.13.3`.

## What's new in v19.13.3 (launcher-only patch: the actual PuTTY host-key prompt, still unanswered even after v19.13.2's flag fix)

Your very next real run after v19.13.2 showed that changing gcloud's own `--strict-host-key-checking`
flag from `accept-new` to `no` fixed the "Invalid choice" *argument-parsing* error, but did not fix the
*actual* problem it was originally meant to solve. The log showed exactly why: Windows' bundled PuTTY
(`plink.exe` for `gcloud compute ssh`, `pscp.exe` for `gcloud compute scp`) is what gcloud actually shells
out to on Windows, and confirmed this round -- gcloud's own `ssh.py` hardcodes that PuTTY backend on
Windows, with no supported flag, environment variable, or config property to switch to OpenSSH instead.
PuTTY's own "Store key in cache?" prompt on first connection to a new host is a completely separate
mechanism from gcloud's `--strict-host-key-checking` flag, and that flag was never actually controlling
it -- which is exactly why the prompt was sitting right there in your log even with the flag set to `no`.
With nobody available to type "y", Window 1's SSH project-check limped through after a several-second
delay (the exact mechanism by which is still not fully understood, but not needed to fix this), and then
the very next call, `gcloud compute scp` to upload the project, hit the identical unanswered prompt on
its own separate `pscp.exe` process and failed outright: `pscp: unable to open ~/: failure`, repeated once
per file being uploaded (86 times in your log), followed by `ERROR: (gcloud.compute.scp) ... pscp.exe
exited with return code [1]`.

**The fix.** The standard, documented technique for automating exactly this PuTTY prompt is to pipe "y"
into the process's own stdin -- plink/pscp read one line from stdin as if a person had typed the answer
and pressed Enter, and "y" (rather than "n") actually caches the key so later calls to the same VM in the
same run don't have to re-answer it. All 5 real `gcloud compute ssh`/`gcloud compute scp` call sites in
`Start-LocalLife-Demo.ps1` now do this (`Start-AppRole`'s SSH project-check, its `scp --recurse` upload,
its remote-bootstrap SSH; `Start-CloudTunnelRole`'s tunnel SSH; `Initialize-PiCloudTunnel`'s VM-setup SSH).
Building this surfaced a real, sharp PowerShell gotcha worth recording: piping into the existing
`Invoke-NativeTolerantly` helper by adding a naive pipeline-bound parameter is ambiguous with that
helper's existing "collect all remaining arguments" parameter -- PowerShell's own parameter binder can
silently consume a real, legitimate argument into the wrong parameter instead of leaving it as a
remaining argument, and, worse, under this script's own `$ErrorActionPreference = 'Stop'`, a pipeline
object that fails to bind becomes a *crash*, not a warning. This was found and confirmed directly in a
real installed pwsh, not guessed -- the actual fix passes the stdin answers as a plain named array
argument (`-StdinLines`) instead, sidestepping the ambiguity entirely, with both the piped-through and
unaffected-normal-call shapes verified working afterward.

**Testing done:** built a fake `gcloud`-like stub that reproduces the exact real sequence from your log
(prints the real "Store key in cache?" prompt text, reads one line from stdin, only succeeds on "y", and
fails with the exact `pscp: unable to open ~/: failure` message otherwise) and ran the *actual, edited*
functions extracted straight out of the real launcher file (not a rewritten copy) against it: the SSH
check call and the scp upload call both now succeed automatically and the host key gets trusted, while a
deliberately-unmodified call with no stdin answer supplied still correctly fails -- confirming the fix
does real work rather than just hiding the symptom. The real installed PowerShell 7.4.6 parser confirmed
the updated launcher clean. Full Python suite unaffected (309/309 -- no Python touched this round).

**Not yet verified on your actual Windows machine and real Google Cloud VM.** This targets the documented,
standard way to automate PuTTY's own host-key prompt, verified against a faithful reproduction of
`plink`/`pscp`'s exact observed behavior -- but the real confirmation is your next run getting cleanly
past the project upload step and through to the tunnel and dashboard.

## What's new in v19.13.2 (launcher-only patch: gcloud's own SSH host-key flag rejected `accept-new`)

Your very next real run confirmed the v19.13.1 zone-timeout fix works -- Window 1 found `depth-l4`
"already running in europe-west4-c" and finished in about 26 seconds, no "zone was never recorded"
error at all -- but it immediately hit a new, different, blocking error one step later: every
`gcloud compute ssh` and `gcloud compute scp` call failed with
`ERROR: (gcloud.compute.ssh) argument --strict-host-key-checking: Invalid choice: 'accept-new'. Valid
choices are [ask, no, yes]`. That flag was added back in round 8.7 specifically to stop Windows'
bundled `plink.exe` from popping an interactive "Store key in cache?" prompt on the VM's first
connection (unavoidable to skip some other way, since `gpu.py` can hand you a fresh VM, and so a fresh
host key, on any zone move) -- and at the time it was checked against Google's own reference
documentation for `gcloud compute ssh`. That documentation apparently doesn't reflect every gcloud CLI
version people actually have installed: your real, installed gcloud only accepts `ask`, `no`, or `yes`
for its own `--strict-host-key-checking` flag, and rejected `accept-new` outright. This was blocking
Cloud mode completely -- both Window 1's SSH project-check/upload/remote-bootstrap and Window 2's
tunnel SSH, which is why the main orchestrator log showed it stuck waiting in `Wait-ForAppHealth` even
though the zone-move step itself had already succeeded.

**The fix.** All 5 real `gcloud compute ssh`/`gcloud compute scp` call sites in
`Start-LocalLife-Demo.ps1` (`Start-AppRole`'s SSH project-check, its `scp --recurse` upload, its
remote-bootstrap SSH; `Start-CloudTunnelRole`'s tunnel SSH; `Initialize-PiCloudTunnel`'s VM-setup SSH)
now pass `--strict-host-key-checking=no` instead of `accept-new` -- `no` is the closest of your gcloud's
own confirmed-valid choices to the original goal of never prompting interactively (`ask` would just
reintroduce the exact prompt this flag exists to avoid). **Tradeoff, worth knowing:** `no` never prompts
but also never verifies the VM's host key on any connection, including later ones -- unlike
`accept-new`'s "pin on first connection, verify from then on" behavior. This is a reasonable tradeoff
here specifically because `gpu.py` can create a brand-new VM (a brand-new host key) on any zone move,
which would otherwise mean manually clearing a pinned key by hand after every such move, and because
these connections are already gated by your own gcloud/IAM authentication, not by host-key trust. This
fix does **not** touch the different, unrelated OpenSSH-native `-o StrictHostKeyChecking=accept-new`
option used elsewhere in the same script for the Raspberry Pi's own direct `ssh`/`scp` commands to the
cloud VM and to the Pi itself -- that is a different program's flag entirely, `accept-new` genuinely is
a valid value for OpenSSH, and your error was specific to gcloud's own SSH wrapper flag only.

**Testing done:** grepped the whole script for both the gcloud-specific and OpenSSH-native forms to
confirm the complete, current list of call sites and that only the gcloud ones needed to change; the
real installed PowerShell 7.4.6 parser confirmed the launcher is still syntactically clean after the
edit; the full Python test suite still passes (309/309 -- no Python touched this round, launcher-only
change). **Not yet verified against a real end-to-end SSH/tunnel connection completing on your
machine** -- this fixes the exact "Invalid choice" error from your log, but your next real run is what
confirms the SSH check, upload, bootstrap, and tunnel all actually succeed with `no` in place of
`accept-new`.

## What's new in v19.13.1 (launcher-only patch: a real Cloud-mode zone-move bug fixed)

Your first real Cloud-mode run hit a genuine, real capacity problem: `europe-west1-b` had no L4 GPU
available at that moment (three "no GPU capacity" attempts in Window 1's own log), so `gpu.py` did
exactly what it's built to do -- capture the VM's disk as an image ("3-8 min", its own printed
estimate) and hunt across every other configured zone for one with capacity. That is normal, working
behavior, not a failure. The actual bug was in the launcher around it: Window 2 (the secure tunnel)
only waited 2 minutes for Window 1 to record which zone the VM landed in, then threw
`DEMONSTRATION ERROR: The cloud VM zone was never recorded by Window 1` -- while Window 1 was still
legitimately in the middle of that same image capture, minutes away from succeeding. `gpu.py`'s own
worst-case path for a forced zone move (3 restart attempts, then `make_image()` -- up to 30 minutes by
its own hard timeout, "3-8 min" typical -- then deleting the old VM, hunting across every zone/shape
tier, then `wait_ready()` waiting up to 12 more minutes for the fresh VM's boot script) can legitimately
take 15-30+ minutes end to end. 2 minutes was never going to be enough for that path.

**The fix, not just a bigger number.** Both places that waited for the recorded zone (Window 2's tunnel,
and the Raspberry Pi's own direct-to-cloud tunnel setup) now share one `Wait-ForCloudZoneFile` helper in
`Start-LocalLife-Demo.ps1` with two real improvements: (1) the ceiling is raised to a realistic 40
minutes matching `gpu.py`'s own documented worst case, with a status message printed every ~60 seconds
so a long, legitimate wait doesn't look like the window has frozen; (2) it also checks whether Window
1's own process is still running -- if Window 1 has already crashed or been closed without ever
finishing, waiting out the rest of a 40-minute timeout for a process that no longer exists would be
pure wasted time, so this now fails immediately with a clearer message ("Window 1 has already closed
without ever bringing up the VM") instead. The main orchestrator's own overall health-check wait
(`Wait-ForAppHealth` in Cloud mode) was raised from 8 minutes to ~50 minutes to match -- it can only
succeed once both the zone move AND the tunnel AND the app itself finish starting, so it needed the
same headroom.

Also fixed, unrelated but noticed while looking at the actual `.cmd` file you uploaded:
`START_LOCAL_LIFE_CLOUD.cmd` and `START_LOCAL_LIFE_DEMO.cmd`'s header comments said `v19.7.0` and
`v19.8.3` respectively -- stale version numbers frozen since round 8 that were never actually kept in
sync with the real package version (cosmetic only; the launcher itself always reports and runs the real
installed version at startup, so nothing was ever functionally wrong), but confusing to see when
checking "am I on the right build." Both now say plainly that the comment is cosmetic rather than
showing a number that looks wrong.

**Testing done:** three control-flow scenarios (the zone file appears partway through the wait; Window
1's process has already exited with the zone file never appearing; Window 1 stays alive the whole time
and the wait genuinely times out) were run against a scaled-down standalone reproduction of the exact
new logic (same branching, small attempt counts so it runs in under a second instead of up to 40
minutes) -- all three passed, including confirming the "already closed" case fails in a fraction of a
second rather than waiting out the full timeout. The real installed PowerShell 7.4.6 parser confirmed
the updated launcher clean. Full Python suite unaffected: 309/309 (no Python code touched this round).

**Not verified against a real Cloud-mode zone-move completing end-to-end.** This directly targets and
fixes the premature-timeout bug visible in your pasted log (Window 2 erroring at the 2-minute mark while
Window 1 was still genuinely working), but the actual multi-minute `gpu.py` recovery -- image capture,
finding capacity in a new zone, the fresh VM's full boot -- still needs to be confirmed by watching it
complete on your next real run. If Window 1 itself reports a real error (not just "still hunting"),
that's a different, genuine problem and needs its own diagnosis from that window's exact text.

## What's new in v19.13.0 (volume estimation fixed: a real bug in the default geometry mode, plus a new table-relative box measurement)

Both cameras are confirmed working — your screenshots show live detections (milk boxes, bags, a UN3091
shipping box) with a good-looking mask/mesh around each object. But volume was still badly wrong (a
1.5 L milk box reading ~62.6 L), and pale/washed-out yellow objects showed as grey. You also uploaded a
new, detailed engineering document (`revised_dual_camera_volume_engineering_recipe.pdf`) diagnosing the
root cause as "coordinate geometry, not a missing library" and prescribing a specific table-relative
measurement approach. This round follows that recipe and fixes two separate, real bugs.

**Bug 1 — and a correction to what v19.12.0 actually fixed.** v19.12.0's writeup (still visible further
down this file) claimed `ray-frustum` — the default volume-integration mode since that round — was
"mathematically exact regardless of tilt." That claim was checked again this round by reading
`estimate_volume()`'s `ray-frustum` branch directly, and it was wrong: the branch computes
`contributions_m3 = (reference_depth**3 - object_depth**3) / (3.0 * focal_product)` straight from raw
camera-Z depth, and never reads `effective_height` (the tilt-corrected perpendicular height the round-13
plane fit computes) in that formula at all. Only `reference-plane` mode's formula
(`object_height(effective) * pixel_area_m2(at reference depth)`) actually uses the corrected height. So
the round-13 tilt fix was real, well-tested, and still never reached the production liters number for
RealSense's default mode — that's a large part of why a 1.5 L box was reading 62.6 L. Fixed by switching
the default `volume_geometry` from `ray-frustum` to `reference-plane`; both `config.py` and
`cloud.env.example` now carry a corrected comment explaining this so the mistake isn't repeated.

**Bug 2 — new table-relative box measurement, replacing per-pixel summing for boxes.** The PDF's core
prescription: instead of summing per-pixel height×area contributions (sensitive to noisy edge pixels and,
per Bug 1, easy to wire up wrong), measure a rigid box with three robust scalar numbers and multiply them.
New `estimate_box_volume_cuboid()` in `volume.py`: erodes the object mask 2-3px (RGB-D boundaries mix
object/background depth), fits height as the median of the top 10th-percentile band of perpendicular
distance above the already-fitted table plane (falling back to the 98th percentile if that band is empty),
and gets footprint length/width by projecting the object's 3D points onto the table plane, finding the
object's true principal axes via PCA (rotation-independent of any arbitrary initial in-plane basis), and
taking percentile-trimmed (2nd-98th) extents along each axis to avoid raw-min/max noise sensitivity.
`volume = length * width * height`. This only overrides box-family detections (label containing "box",
"cardboard", "carton", or "parcel") on RealSense — Logitech never supplies metric geometry and is
untouched, and bags still use the existing per-pixel path.

**New: box template matching.** `box_templates.py`/`box_templates.yaml` can report an exact *known*
volume once a box type is matched by measured dimensions — but per the PDF's explicit instruction ("do
not let code or an LLM invent real box dimensions"), the shipped file has your 1 L / 1.5 L / 2 L milk-box
entries as placeholders with `dimensions_mm: {length: 0, width: 0, height: 0}` and `measured: false`, so
nothing matches until you measure your actual boxes (calipers or a ruler) and fill in real numbers. Until
then, every box gets its own single-view cuboid estimate, which is still a large accuracy improvement over
the old per-pixel sum.

**Bug 3 — yellow showing as grey.** `dominant_color()` had a low-saturation gate
(`if saturation < 0.16: return grey/white/black`) that fired before any hue check at all. A pale or
washed-out warm color — a cream milk carton, a translucent yellow bag — naturally has low HSV saturation
despite being visibly warm-toned, so it was being swallowed into "grey" before the yellow branch further
down ever got a chance to see it. Fixed with a new LAB-colorspace check: OpenCV's `COLOR_BGR2LAB`
conversion encodes the a/b channels 0-255 around a 128 midpoint (not the conventional signed range
centered on 0), so reading the raw byte without subtracting 128 silently understates warm/cool tint — a
new `_lab_b_channel()` helper does that subtraction correctly and, fused with the existing HSV path, lets
a low-saturation pixel still be recognized as yellow when its (corrected) LAB b-channel and brightness say
so, while genuinely neutral greys/whites/blacks are unaffected.

**Testing done:** 27 new tests — 8 on the color fix (`tests/test_color_classification.py`, covering pale
cream/translucent-yellow correctly becoming "yellow", neutral grey/white/black staying unaffected,
saturated yellow unaffected, and a pale-blue control confirming it does *not* get misclassified as
yellow), 18 on the new cuboid/template system (`tests/test_box_cuboid_volume.py`, verified against exact
closed-form synthetic ground truth: height recovery regardless of tilt across [0°, 15°, 25°, 40°],
footprint recovery matched to ~0.1mm both with and without the default 2px mask erosion, volume
self-consistency, border-clipping detection, low-point-count and missing-plane rejection, and template
matching including the shipped all-placeholder YAML correctly matching nothing), and 1 new pipeline-level
integration test (`tests/test_plug_and_play.py`) that runs a box-labeled detection through the full
`VisionPipeline.process_frame()` and confirms `estimate_box_volume_cuboid()` is actually wired in — a
genuine coverage gap before this round, since the unit tests above only tested the function in isolation.
Switching the default `volume_geometry` also required updating 3 pre-existing tests whose hardcoded
expected values were computed under the old `ray-frustum` default (each updated with an explanatory
comment describing exactly why the new number is correct, never by reverting the underlying fix — this
project's established practice). Full suite: **309/309**. The real installed PowerShell 7.4.6 parser
confirmed the launcher clean after the version bump (no `.ps1` logic was touched this round).

**Not yet verified against real hardware for this specific round.** Please run your milk-box tests again
(1 L / 1.5 L / 2 L) and paste the new dashboard numbers — ideally including `box_dimensions_mm` and
`box_volume_flags` from the detection JSON if you can get them, since that tells us directly whether the
height, the footprint, or both are still off, rather than just the final liters figure. For the biggest
accuracy jump, also measure your actual boxes and fill in `box_templates.yaml` — it's a plain YAML file
with comments explaining exactly what to fill in.

**Deliberately not rebuilt this round, disclosed honestly:** full per-live-frame RANSAC table-plane
refitting with explicit object-mask exclusion (the plane is still fit once from the stored baseline/empty
scene, which should be fine since the object was never in that frame, but this specific claim was not
independently re-verified this round); two-view box reconstruction/fusion; a dedicated bag heightmap mode
(existing bag volume paths are completely untouched and should work exactly as before); the PDF's literal
`project/app/capture/segmentation/geometry/...` directory reorganization (the existing, 300+-test-covered
`locallife_cloud/` package layout was deliberately kept to avoid destabilizing a working system); a
`recipe_api.py` FastAPI endpoint update to emit the new `BoxVolumeMeasurement` JSON contract (not touched
this round). Also still unresolved: one of your screenshots shows an "8.45 m" RealSense distance reading
for a milk box sitting right in front of the camera — that's a separate, unexplained symptom that hasn't
been root-caused yet; if it's still happening, a screenshot showing exactly which reading that is (and
ideally the surrounding dashboard state) would help track it down next round.

## What's new in v19.12.2 (the busy-device fix made real: a proper RealSense hardware reset)

You ran v19.12.1 and hit the exact same symptom again — both cameras stuck at 0 frames received,
"solve kar bhai yrr kya problem hai." The Windows-side log this time didn't include the Pi window's
own traceback, so it's not confirmed whether it was still literally the same `xioctl`/errno=16 error —
but that error class is the standing hypothesis, and this round fixes the part of v19.12.1's own theory
that was incomplete.

**What v19.12.1 got half right.** A stale `edge_client` process left holding the camera open was (and
still is) a real, plausible cause. But killing that process — by `pkill`, by the window being closed,
by the SSH connection dropping — does **not** run its own `finally: pipeline.stop()` in
`iter_realsense()`. Python does not run `try/finally` cleanup on SIGTERM/SIGKILL, and a librealsense
pipeline that gets killed mid-stream can leave the RealSense device stuck in a busy state at the USB/
firmware level, not just "a process still has the file descriptor open." So v19.12.1's `pkill` could
correctly kill the old process and the *next* `pipeline.start()` could still fail with the identical
"Device or resource busy" error, because killing the process was never actually guaranteed to fix
that particular class of stuck-USB-state failure. This is a real gap in last round's fix, not a wrong
diagnosis of the symptom — it correctly identified "a leftover process," it just used the wrong tool
to clear the device's own stuck state afterward.

**The fix, in `locallife_cloud/edge_client.py`:** `iter_realsense()` now catches a `pipeline.start()`
failure directly (rather than only relying on the outer retry loop) and calls a new
`_hardware_reset_realsense()` helper, which enumerates every attached RealSense device via
`rs.context().query_devices()` and calls `device.hardware_reset()` on each — this is librealsense's own
documented recovery method for exactly this failure: a real firmware-level reset, not just closing a
handle. After a reset, the device disappears and re-enumerates on the USB bus (typically a couple of
seconds), so the code waits 3 seconds before retrying `pipeline.start()` once. If no device is found to
reset, it retries immediately without the wait. If the retry also fails, the error still propagates so
the existing outer `run_resilient_camera` retry loop (in `camera_recovery.py`) keeps trying every 3
seconds, same as before. `Start-LocalLife-Demo.ps1`'s `pkill` was also strengthened from a plain
`pkill -f` (SIGTERM) to `pkill -9 -f` (SIGKILL, since SIGTERM can be ignored or the process can be stuck
in an uninterruptible USB syscall) with a longer 2-second wait, in both the Local- and Cloud-mode Pi
launch paths.

**Testing done:** 8 new tests (`tests/test_edge_client.py`) inject a small, realistic fake
`pyrealsense2` module (the real SDK requires physical hardware and cannot be installed in this
sandbox — same limitation as every prior round's real-hardware verification) and exercise the actual
`iter_realsense()`/`_hardware_reset_realsense()` code, not a rewritten stand-in: a busy-then-reset-then-
success run (confirms `hardware_reset()` is called exactly once per device and the retry succeeds), a
no-device-found run (confirms it retries without the unnecessary wait), a still-failing-after-reset run
(confirms the error still propagates rather than being silently swallowed), a healthy first-attempt run
(confirms zero reset/sleep cost when nothing is wrong), and 4 tests directly on the reset helper
(multiple devices, one device failing to reset without blocking the others, broken device enumeration,
and zero devices found). Full suite: **282/282** (274 previous + 8 new). The real installed PowerShell
7.4.6 parser confirmed the launcher clean after the `pkill -9` change and version bump.

**Not verified against real hardware.** This targets the specific, well-documented "busy RealSense
device" failure mode with librealsense's own recommended fix, but it has not been confirmed on your
actual Pi yet — and it's still possible the real failure this time was something else entirely (a
genuine USB power/hub issue causing both cameras to drop together, a tunnel/auth problem, or something
new). **If this build still doesn't fix it, please paste the Pi window's (Window 2's) own text** — the
Windows-side "0 frames received" message alone doesn't show which of these it actually is, and without
that text any further round would be guessing rather than diagnosing.

## What's new in v19.12.1 (Pi camera startup failure fixed: "Device or resource busy")

You hit a real-hardware error right after the tilt-correction build: the Pi window failed with
`RuntimeError: xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy` on
RealSense, `logitech camera could not stream; retrying in 3.0 seconds`, then
`RuntimeError: Unable to open camera/video source: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0`
on Logitech, and finally `DEMONSTRATION ERROR: Both camera streams did not reach the application.`

**Root cause, found by reading `Start-LocalLife-Demo.ps1` and `edge_client.py` directly.** errno=16
(`EBUSY`) from a V4L2/RealSense device means some *other* process already has that device node open —
your cameras were physically fine. The launcher already has exactly this kind of cleanup for the
Pi-to-cloud SSH tunnel process (`pkill -f "id_locallife_cloud" ... ; sleep 1;` before reopening it),
but it never did the same for the camera-streaming process itself
(`python3 -m locallife_cloud.edge_client`). If a previous demo run's Pi window was closed, the run
errored out, or the SSH connection dropped without the remote command exiting cleanly, that old
`edge_client` process kept running on the Pi — still holding the RealSense pipeline and the Logitech
`/dev/video` node open — so the *next* run's fresh `edge_client` process failed to open either camera,
every single time, until the Pi was rebooted or the old process was killed by hand.

**The fix, in `Start-LocalLife-Demo.ps1`:** both Pi-launch code paths (Local mode and Cloud mode) now
run `pkill -f "[l]ocallife_cloud.edge_client" 2>/dev/null || true; sleep 1;` immediately before
starting a fresh `edge_client`, giving the kernel a moment to release the USB device nodes first —
the same pattern already used for the tunnel process. This does not touch any of the volume/height
math from the previous round.

**Testing done:** the real installed PowerShell 7.4.6 parser (`ParseFile`) confirmed the updated
launcher clean; full suite re-run at 274/274 (no Python code changed this round, only the launcher).
**Not verified:** this is a launcher/process-lifecycle fix, not something a synthetic test can exercise
(it depends on real SSH/process behavior on your Pi) — please confirm your next run starts cleanly. If
you ever see this error again after updating, it means something *other* than this launcher is holding
the camera (e.g. `realsense-viewer` left open on the Pi, or a second SSH session) — SSH into the Pi and
run `fuser /dev/video*` (or `sudo fuser -v /dev/video*`) to find and close whatever process has it open.

## What's new in v19.12.0 (mounting-tilt height/volume bug found and fixed)

You shared 7 real-hardware screenshots (pillow, backpack/tied bag, a shipping package, a steel bottle) plus your uploaded thesis pilot deck (`VisionBasedWasteVolumeEstimation.pptx`) and said, verbatim: "realsense is working perfectly in distance but in height and total net volume its still inaccurate." You also asked to (1) treat Logitech as a supporting camera for color/material only, (2) shift testing toward known-volume 1/1.5/2 L milk boxes before garbage bags, and (3) review github.com/agu3rra/volpy again.

**Root cause, found by reading `volume.py` directly, not guessed from the screenshots.** `estimate_volume()`'s height field was `baseline_depth - object_depth`: the camera-Z difference at each pixel between the empty-bin baseline and the current reading. That equation is only an object's true vertical height when the camera's optical axis is exactly perpendicular to the bin floor. Your own thesis deck's slide 3 confirms the actual rig: "a wall-mounted bracket... both pointing downward into the bin" — not a calibrated overhead gantry. For two parallel planes (an object's flat top resting on a flat floor) cut by a camera ray at angle theta from their shared normal, the camera-Z difference overstates the true perpendicular height by a factor of `1/cos(theta)` — a 25-30° mounting tilt (very plausible for a bracket "pointing downward", not measured/leveled) already inflates every reported height, and every derived liters figure, by 10-40%. This explains "distance is right, height/volume is not" exactly: the raw per-pixel depth *reading* is unaffected (hence accurate distance) — only the height *derived* from it was wrong. A second, separate issue was also found: the previous default `geometry_mode="surface-columns"` approximates the volume between the object and floor as a cylinder based on the object's own (near) footprint area, which mathematically underestimates volume whenever height is a large fraction of depth — confirmed by hand (a ~19% underestimate at a 15% height/depth ratio) and unrelated to tilt.

**The fix, in `locallife_cloud/volume.py` and `pipeline.py`:**
- `estimate_volume()` gained a new optional `reference_plane` argument. When given the fitted floor plane (`fit_reference_plane()` — this already existed, but was previously used only as a diagnostic for the Logitech tilt warning, never as a correction), it backprojects each pixel's own depth reading into 3-D and computes the true *perpendicular* distance to that plane, instead of the raw camera-Z difference. This corrected height is used both for the displayed height (cm) and, in the height×area geometry modes, for the volume integral itself.
- The default `volume_geometry` changed from `surface-columns` to `ray-frustum` — the only one of the four modes whose volume integral is mathematically exact regardless of mounting tilt (it sums the closed-form pyramidal-frustum volume between the object and floor along each individual camera ray, never decomposing into a separate height × area product that a tilted mount, or a large height/depth ratio, can bias). `cloud.env.example` updated to match.
- Wired `reference_plane=self.reference_plane` into every RealSense and Logitech `estimate_volume()` call site in `pipeline.py` (the aggregate camera total, the bin-occupancy total, and each per-detection instance measurement) — the same fitted plane the Logitech tilt-warning logic already trusted.
- Fully backward compatible: omitting `reference_plane` (every call site before this round) is byte-identical to the old behavior, and the correction is a mathematical no-op for a camera that genuinely is mounted perfectly overhead (zero tilt) — it only changes numbers when there is real tilt or a large height/depth ratio to correct.
- Interesting side note: the v3 recipe pipeline (still off by default, `LOCALLIFE_RECIPE_ENABLED`) already had an equivalent of this fix built in from round 12 (`normalize_pose()`, which rotates the whole point cloud so the fitted table plane becomes world-Z before any measurement) — this round brings the main, always-on dashboard pipeline up to that same standard of correctness.

**On your other three requests:**
- **Logitech as a supporting camera**: the dashboard already has a "Fused Result" panel (built earlier in this project's history) that does close to what you described — one combined volume weighted toward RealSense (~2x, via inverse-uncertainty weighting), with color and material blended in — rather than treating the two cameras as independently-competing measurements. It automatically inherits today's height/volume fix since it's built from the same per-station data. Given you said to focus on the volume part this round, Logitech's own architecture (adding it as an explicit second depth/side-view input to the volume math itself) was not rebuilt — flagging that as a real, separate follow-on idea for a future round rather than guessing at it now.
- **1/1.5/2 L milk-box testing**: this fix should matter *more*, not less, at that scale than for large bags — both of today's biases (tilt-driven height inflation, and the surface-columns cylinder-vs-frustum underestimate) are proportionally larger for a small, low object than a tall one. Please run the same milk-box tests against this build next.
- **volpy, reviewed again**: unchanged conclusion from the earlier review — it's a land-surveying/DEM cut-fill tool (Delaunay triangulation over raw XYZ survey points, then an exact analytic per-triangle plane integral), with no camera intrinsics, depth-image handling, or RANSAC plane fitting. Its core idea — decompose into simple regions, fit a plane, integrate exactly rather than approximate — is, in spirit, exactly what the `ray-frustum` mode above already does for this project's own per-pixel-ray geometry, just derived independently from the pinhole camera model rather than adapted from volpy's own triangulated-mesh code.

**Testing done:** 6 new tests (`TiltCorrectedVolumeTests` in `tests/test_bag_station.py`) build a fully known, closed-form synthetic scene — a floor plane tilted a deliberate, known 30°, with a flat-topped object of a known 10 cm true height sitting on it — and verify the exact predicted bias analytically: the naive (uncorrected) height comes out ~15.5% inflated, matching `1/cos(30°)` to within measurement precision; the plane-corrected height lands within 0.3% of the true 10 cm; naive liters are off by ~9.8% (two partially-offsetting biases, both explained in the test's own comments) versus ~2.5% for the corrected, ray-frustum result; a zero-tilt control scene proves the correction is a mathematical no-op when there is nothing to correct; and a direct comparison confirms omitting `reference_plane` reproduces the exact pre-round-13 output. 3 existing tests were updated (documented inline, in each) because they hardcoded expected liters values computed under the old, less accurate default — none of their actual behavioral assertions changed. Full suite: 274/274 (268 previous + 6 new). The real `pwsh` 7.4.6 parser confirmed clean on the launcher after the version-banner bump. **Not verified:** any of this against your real, physically tilted RealSense/Logitech rig — the fix is proven against a rigorous, closed-form-exact synthetic scene with a known ground truth, which is the strongest verification available without camera access from this environment; real confirmation depends on your next test run, ideally with the 1/1.5/2 L milk boxes.

## What's new in v19.11.0 (recipe pipeline rebuilt to the v3 "Zero-Flaw Implementation Blueprint")

You pasted a second, far more detailed recipe document (`dual_camera_estimation_recipe_v3.pdf`) with the instruction to build it, local-only first, cloud later since "it's just changes in the initiation." This round rebuilds the recipe pipeline's internals to match v3 exactly, keeping the same architecture decision as last round: still an additive, off-by-default second result next to your real dashboard/tracking/ledger, never replacing it. **Cloud is deliberately untouched this round** — `recipe_api.py`'s multipart wire format and generic error handling still work, but v3's own `POST /estimate` base64 contract and structured `invalid_frame`/`no_object_found`/`pipeline_failed` error codes (v3 §9) are not yet built; that is next round's work, per your own sequencing.

**What changed inside each module:**
- **`pointcloud_volume.py`** — depth masking now follows v3 §5.1 exactly: only `[100mm, 2500mm]` depth counts (RealSense's own zero-fill for unmeasurable pixels is never trusted), and the object mask is eroded 2-3px first to reject the noisy depth fringe at segmentation edges. New `normalize_pose()` implements v3's non-negotiable rule #5 — rotate the whole cloud so the table plane becomes world Z-up *before* any measurement, not after, exactly as specified, with the rotation matrix returned to the caller. New `estimate_volume_planefit_box()` is the v3 §5.3 box method — top-face RANSAC plane for height, front-face RANSAC vertical plane + point projection for width, and either a second view's own face (two-view accuracy target) or the front face's own max in-plane chord (single-view fallback, confidence ×0.6, flagged `single_view_volume`) for the third dimension. The original `estimate_volume_obb()` (oriented-bounding-box) is **kept, unchanged** — v3 §13.3 explicitly wants it as an ablation baseline ("raw-OBB ... show OBB fails on hidden dimension"), so it's no longer the box default but stays available. The bag heightmap now genuinely convex-fills interior gaps (not just a no-op it silently was before this round's fix — see bugs below), discretizes to the nearest 5L/10L class within v3's ±30% window (else reports the raw value, flagged `bag_discrete_approx`), and always emits `volume_tolerance_liters` (±15%) with confidence capped 0.5-0.7, per v3's own acknowledgment that bags are inherently non-rigid.
- **`recipe_color.py`** — rebuilt from HSV to v3 §6's exact LAB algorithm: central 60% crop of the object ROI, BGR→LAB, a hand-written k-means (k=4) that excludes near-black/near-white pixels from centroid updates but not from cluster assignment (not expressible through OpenCV's own `kmeans`), achromatic-first classification (L/chroma thresholds), then a*/b* quadrant mapping for the chromatic classes, and a same-margin tie rule that reports "Other" rather than guessing between two close clusters.
- **`recipe_material.py`** — now a real two-level cascade per v3 §7: CLIP (Level-1) with a softmax margin; a Level-2 fallback (`RecipeMaterialFallbackClassifier`, a torchvision MobileNetV3-Small/EfficientNet-B0 head) that activates when CLIP's confidence is low or it called "Plastic" with a thin margin (v3's documented CLIP failure mode). **No fine-tuned fallback checkpoint ships with this project** — there is no labeled training set available in this environment, and fabricating one would violate v3's own "do not guess" instruction as much as fabricating a confidence score would; point `recipe_config.yaml`'s `material.fallback_checkpoint_path` at a real trained checkpoint once one exists (v3 §13's own validation plan describes how) and the cascade activates automatically. Until then, a low-confidence CLIP call with no fallback available degrades honestly (kept if still above the fallback's own floor, else `Other` + `ambiguous_material` flag) rather than silently only ever running Level-1 and calling it done.
- **`recipe_config.py` / `recipe_config.yaml`** (new) — every v3 §10 tuning knob (depth validity band, mask erosion, SOR, k-means k, bag class sizes/tolerance/discretize window, material confidence floors, ...) in one committed YAML file, exactly matching the recipe's own schema, instead of scattered env vars.
- **`recipe_calibration.py`** (new) — v3 §8's "calibrate, don't guess": a logistic calibration map fit from a small labeled set via plain-numpy gradient descent (`fit_calibration()`). No labeled calibration set exists yet, so every confidence in this build stays raw/uncalibrated by design, and the result carries an `uncalibrated_confidences` flag saying so rather than presenting a raw model score as a calibrated probability.
- **`recipe_detect.py`** — `select_best()` now applies v3 §4 step 2's selection rule (class filter first when a target class is configured, then a [5%, 80%]-of-frame area sanity check before ranking by confidence). New `frame_edge_clip_fraction()` backs the v3 edge-case matrix's "object touches frame edge → lower confidence" rule.
- **`recipe_pipeline.py`** — output schema now matches v3 §9 exactly: `volume_tolerance_liters`, `material_model` (which model actually produced the material call), `views_used`, and a `flags` array (`low_volume_conf`, `single_view_volume`, `bag_discrete_approx`, `outlier_removal_high`, `ambiguous_material`, `uncalibrated_confidences`, plus `frame_edge_clipped`/`no_realsense_detection`/`no_object_found`/`views_disagree` for this project's own edge cases) replace the plainer v1/v2 schema. Fusion stays strict and cross-sensor-free: volume←RealSense, color←Logitech, material←Logitech.
- Dashboard's "Recipe Result" panel updated for the new fields (tolerance, material model, views used, flags), still off by default behind `LOCALLIFE_RECIPE_ENABLED`.

**Three real bugs found and fixed while building this** (all via this round's own synthetic testing, the same rigor as every prior round's bug-finding):
1. **`np.maximum.at` against a NaN-seeded heightmap grid was a total no-op** — IEEE 754 defines `max(nan, x)` as `nan`, so every grid cell stayed empty forever regardless of how many points landed in it, silently returning 0 L for every bag. Fixed by seeding the grid with `-inf` (a true "no data" sentinel for a max-reduce) and converting untouched cells back to `nan` after the reduce.
2. **The convex-fill hole-filler leaked past the object's true footprint into real background**, once bug 1 was fixed and cells actually started filling: a naive "fill any empty cell next to a filled one" pass doesn't stop at the object's edge, so it kept growing outward until the *entire* rectangular bounding grid around a circular bag footprint was filled — inflating the computed area by ~27% in a direct test. Fixed with a border flood-fill that distinguishes real interior holes (enclosed by data) from exterior background (reachable from the grid's own edge without crossing any data) — only the former ever gets filled.
3. **That same border flood-fill failed on a sparse point cloud** (an object far enough from the camera, or a grid finer than the sensor's real spatial resolution at that range, that occupied cells aren't densely packed): gaps of a cell or two between real points let "exterior" leak straight through the porous boundary to the center, misclassifying the whole interior as background and defeating the fill entirely (0 volume again, on a case bug 2's fix alone didn't cover). Fixed with a light morphological dilation of the occupied mask used only to decide the interior/exterior boundary — the actual summed heights still always come from the real, undilated data.

**Not verified:** any of this against real RealSense/Logitech hardware, real YOLOv8n/11n-seg or CLIP model weights, or a real fine-tuned material fallback checkpoint (this sandbox cannot reach Hugging Face or the Ultralytics model hub, and no labeled training/calibration data exists here — same limitation as every prior round). What is verified: 59 committed tests (`tests/test_recipe_pipeline.py`, replacing the prior round's 27) covering pose normalization against an analytically-known tilted plane, the plane-fit box volume's two-view accuracy (<5% error against exact synthetic dimensions) vs. its single-view fallback (flagged, deliberately less accurate, matching v3's own claim), the heightmap's convex-fill/discretization/tolerance behavior including regression tests for all three bugs above, all 8 LAB color classes plus the tie rule, every branch of the material cascade routing logic, the config loader against the shipped `recipe_config.yaml`, the calibration fitter, the detector's class-filter/area-sanity/frame-edge helpers, the end-to-end pipeline's exact v3 JSON schema, and the FastAPI/dashboard wiring's existing auth/failure-degradation tests updated for the new schema. Full suite: 268/268 (209 pre-recipe + 59 recipe), and the real `pwsh` 7.4.6 parser confirmed clean on the launcher. Also added `tests/conftest.py` to force `transformers` into offline mode during the test run only — unrelated to this round's own changes, but a pre-existing test was found to hang retrying a blocked huggingface.co connection instead of using its own synthetic stand-ins, and this removes that flakiness for good.

## What's new in v19.10.0 (additive recipe pipeline: Open3D volume, 8-class color, 7-class CLIP material, FastAPI endpoint)

You pasted a complete, fully-specified recipe ("Dual-Camera Smart Volume + Color + Material Estimation System") and asked to continue from where things left off. Two decisions governed how it was built in: keep the existing dashboard/launcher/tracking/ledger/waste-stream/calibration UI exactly as it is, and follow the recipe's own technical choices exactly underneath it — Open3D point-cloud oriented-bounding-box volume for boxes, heightmap/grid-integration volume for bags, RealSense-only for volume, Logitech-preferred (RealSense-fallback) for an 8-class dominant color and a 7-class CLIP material, and a generic `yolov8n-seg`/`yolo11n-seg` detector rather than this project's own curated bag/box vocabulary.

**New modules (`locallife_cloud/`), each independently verified against synthetic data since this sandbox has neither real camera hardware nor internet access to Hugging Face/PyPI model hubs to download real YOLO/CLIP weights:**
- `pointcloud_volume.py` — Open3D backprojection, statistical outlier removal, `estimate_volume_obb` (box path) and `estimate_volume_heightmap` (bag path). Verified: a synthetic top+front box came out 12.6% off a known 3.0 L; a synthetic dome/bag came out 1.2% off a known 6.362 L paraboloid volume.
- `recipe_color.py` — HSV circular-hue k-means into the recipe's exact 8 classes (Blue/Grey/Black/White/Red/Green/Yellow/Other). Verified: all 8 synthetic solid-color test cases classified correctly.
- `recipe_material.py` — CLIP zero-shot into the recipe's exact 7 classes (Plastic/Fabric/Metal/Cardboard/Paper/Rubber/Other), reusing this project's own already-hardened CLIP-loading code from `material.py` rather than duplicating it.
- `recipe_detect.py` — thin wrapper around a generic Ultralytics YOLOv8n/11n-seg model plus a box-vs-bag fill-ratio heuristic. **Deliberately weaker false-positive rejection than the existing YOLOE curated vocabulary** — flagged in its own docstring — since the recipe specifically asked for a generic open-vocabulary detector, not this project's tuned one.
- `recipe_pipeline.py` — orchestrates all of the above into the recipe's exact output JSON: `volume_liters`, `volume_confidence`, `color`, `color_confidence`, `material`, `material_confidence`, `object_type`, `timestamp`.
- `recipe_api.py` — a standalone FastAPI service (`POST /api/recipe/process`, plus `/health`) accepting RealSense color+depth+intrinsics and an optional Logitech color frame as multipart uploads (same wire convention this project's Pi→cloud ingest already uses: JPEG + `.npz` depth + JSON metadata), returning the recipe's exact schema. Runs alongside, never instead of, the existing Flask dashboard — start it by hand with `Start-LocalLife-Demo.ps1 -Role RecipeApi` (port 8100 by default; the main dashboard keeps its own port 8000).

**Wired into the existing dashboard as an additional, off-by-default result**, not a replacement: `AppConfig.recipe_enabled` (env `LOCALLIFE_RECIPE_ENABLED`) defaults to `False`. When off, `/api/state` reports `"recipe_result": {"available": false, "reason": "disabled"}` and none of the recipe modules are even imported — zero behavior change for every existing installation. When turned on, `DualCameraCoordinator.recipe_result()` runs the recipe pipeline against whatever frames the two camera stations already hold, throttled to at most once every `LOCALLIFE_RECIPE_REFRESH_SECONDS` (default 3 s, since it's a real YOLO+CLIP inference pass, not something to repeat on every ~550 ms dashboard poll), cached and served from there; any failure (missing dependency, no frame yet, an exception inside the recipe modules) degrades to `"available": false` with a reason instead of ever breaking `/api/state` for the dashboard's primary measurement. The dashboard itself gained a "Recipe Result" panel next to the existing "Fused Result" panel showing this second estimate.

**Two real bugs found and fixed while building this** (both through this session's own synthetic testing, not just recipe-following): (1) running RANSAC plane-removal on a point cloud already restricted to an object's own segmentation mask misidentifies the object's own largest flat face as "the floor" and strips it, leaving a degenerate remainder that crashed Open3D's oriented-bounding-box fit — fixed by fitting the floor plane from the empty-scene baseline depth image instead, and by never plane-stripping the box-path cloud at all; (2) a perfectly flat, single-depth-value point cloud (no distinguishable front face) crashed the same oriented-bounding-box fit with a qhull precision error — fixed with a PCA-eigendecomposition fallback (thinnest axis floored at 2 cm, confidence halved) inside `estimate_volume_obb` itself.

**Dependencies:** `open3d`, `fastapi`, `uvicorn`, `python-multipart` added to `requirements-local.txt` and `requirements-cloud.txt` (not `requirements-edge.txt` — the Pi only captures frames; all recipe processing runs on the laptop/cloud side, same as everything else). Every import of these is lazy/deferred and wrapped so a missing install degrades to `"available": false` / a clear startup error rather than breaking the existing service.

**Not verified:** any of this against real RealSense/Logitech hardware, real YOLOv8n/11n-seg or CLIP model weights (this sandbox cannot reach Hugging Face or the Ultralytics model hub to download them — every test above used synthetic point clouds/images and stub detectors), or the FastAPI endpoint against a real Raspberry Pi client. What is verified: 27 new committed tests (`tests/test_recipe_pipeline.py`) covering the OBB/heightmap volume math against exactly-known synthetic geometry, all 8 recipe color classes, the recipe material class list and its disabled-classifier fallback, the box/bag detection heuristic and best-detection selection, the end-to-end pipeline's exact JSON schema (including the RealSense-miss/Logitech-hit and no-detection edge cases), the FastAPI endpoint's auth/multipart/error paths via `TestClient`, and the dashboard wiring's disabled/waiting/cached/throttled/failure-degraded states — plus the full suite (236/236, the prior 209 plus these 27) and the real `pwsh` 7.4.6 parser both passing with the launcher's new `-Role RecipeApi`.

## What's new in v19.9.0 (direct Pi-to-cloud tunnel; a real Cloud-mode startup bug fixed; Logitech phantom-box mitigation)

**1. Cloud mode's tunnel architecture is redesigned to the simple 3-tunnel shape you asked for.** Previously, Cloud mode still routed camera frames Pi → laptop → (tunnel) → VM, i.e. through the laptop twice. Now there are three tunnels, each doing exactly one job: laptop↔Pi (SSH login only), laptop↔cloud (Window 2, purely so your browser can see the dashboard at `127.0.0.1:8000`), and a new direct Pi↔cloud tunnel (Window 3) that carries the actual camera frames straight from the Raspberry Pi to the GPU VM. The launcher provisions the new leg itself, with no extra credentials needed from you: it generates a dedicated SSH keypair on the Pi (separate from your normal Pi login), and creates a restricted, no-shell `locallife-tunnel` Linux user on the VM whose key can only forward ports — it cannot log in, run commands, or do anything else. This also needs the cloud VM reachable on `tcp:22`, which the launcher checks and creates a firewall rule for if one is missing (best-effort; almost every GCP project already allows this by default).

**2. Found and fixed a real bug in the process: Cloud mode's server would have refused to start against real infrastructure.** The previous Cloud mode bound the VM's server to `0.0.0.0` (the public interface) — which `server.py` correctly refuses to do without `LOCALLIFE_API_TOKEN` set — but the remote start command never actually set that variable. This was never caught before because Cloud mode had not yet been run against a real GPU VM; the v19.8.6 note that "Cloud mode already matches the new gpu.py... no code change was needed" was true for `gpu.py` itself, but did not check this. The new design removes the problem at the root instead of patching around it: the VM server now binds `127.0.0.1` only, reachable exclusively through the two SSH tunnels above (both already authenticated), so no token, bearer secret, or extra firewall rule for the app port is needed at all.

**3. Logitech's phantom full-frame bounding box (seen in the pillow/backpack/banana-bag screenshots) — diagnosed and mitigated, not yet confirmed on real hardware.** RealSense reports a real, physically measured depth per pixel; Logitech's depth comes from Depth-Anything-V2, a monocular AI model's *prediction*, which has real frame-to-frame jitter/bias on background regions even when nothing has moved. The scene-change detector (the code that recovers "the whole physical object changed" from RGB+depth together) used the same 2.5&nbsp;cm "did this pixel get higher" threshold for both cameras, tuned for RealSense's actual sensor noise. Logitech's own prediction noise could exceed that on parts of the background, and because it's spread continuously (not a few scattered pixels), it survived the density gate meant to catch sparse artifacts, and grew into what you saw as a box spanning most of the frame. Logitech now uses its own, higher 5&nbsp;cm threshold for this specific step (`LOCALLIFE_LOGITECH_SCENE_MIN_HEIGHT_M`, tunable via that environment variable if 5&nbsp;cm still isn't enough or is too aggressive). Verified in a synthetic reproduction of the jitter pattern (spatially-correlated background noise + a real compact raised object) that the new threshold rejects the spurious background regions the old one accepted; **not yet verified against your real Logitech hardware** — please re-run the pillow/backpack/banana-bag tests and share new screenshots.

Also, regarding your other requests this round: the 1/1.5/2/5/10 L accuracy issue and the "1 L bottle reads as 20 L" report were not caused by the core per-pixel volume math (re-read end to end this round — it is already the standard, correctly depth-dependent pinhole-camera formula, not a fixed conversion factor), and [agu3rra/volpy](https://github.com/agu3rra/volpy) (reviewed this round) turned out to be a land-surveying earthwork/cut-fill calculator for GPS survey point clouds, not a camera volume tool — nothing in it applies here. What you asked for — "capture measured volume, distance ka bhi feature daal" — already exists and predates this round: both camera panels on the dashboard have a "Known volume of the object now in view (L)" field with a "Calibrate from this object" button (from v19.6.0/19.8.0, and the calibration-poisoning bug in that feature was already fixed in v19.8.0), and the Logitech panel additionally has the "measured camera-to-empty-bin distance" field described in the v19.8.6 entry below — this is very likely the single biggest lever on your small-object accuracy complaint if it hasn't been used yet on the real rig. There is also a `scripts/validate_known_volume.py` CLI (from v19.8.0) that reports real percent-error/MAPE numbers against a known object — run it with a 1 L bottle, a 2 L carton, etc. to get hard numbers instead of eyeballing the dashboard.

Verified this round: PowerShell parser clean (`pwsh` 7.4.6, not by eye); Python 209/209 tests pass (2 new config fields, 1 new pipeline branch — no existing test broken). Not verified: any of the three changes above against real Google Cloud infrastructure or real Logitech hardware, since this environment has neither. If the direct Pi↔cloud tunnel fails to connect, check `/tmp/locallife-cloud-tunnel.log` on the Pi (the error message on screen tells you to) and confirm `tcp:22` is reachable on the VM from the Pi's network.

## What's new in v19.8.8 (Cloud mode is now genuinely one-click)

Two gaps closed so "run everything at once" actually means it, in both Local and Cloud mode:

1. **`gpu.py` is now bundled in this zip** (the collaborator's EU-wide zone-hunting version, from the email) — no more renaming a downloaded attachment and placing it next to the launcher by hand. `START_LOCAL_LIFE_CLOUD.cmd` finds it automatically.
2. **Both Cloud mode (Window 1) and the Raspberry Pi (Window 2) now install the project themselves the first time, instead of erroring "not installed" and stopping.** This is exactly the gap that needed a manual `scp` + `pip install` to work around on the Pi earlier — closed there, and closed the same way for a fresh cloud VM: the launcher checks over SSH whether the project is already present, and if not, uploads it (`gcloud compute scp` / `scp`) and installs its dependencies automatically before starting the server. An existing VM or Pi that already has the project (the normal case after the first run) skips this entirely and starts exactly as before.

Verified: PowerShell parser clean (found and fixed with a real `pwsh` this time, not by eye); no Python touched, 209/209 suite still passes. Not verified: an actual first-time cloud VM upload end-to-end (needs a real Google Cloud run) -- if it fails, `python gpu.py ssh` and copying the project by hand remains the fallback, same as before this change.

## What's new in v19.8.7 (launcher-only patch)

`DEMONSTRATION ERROR: The application did not become reachable` on an otherwise-healthy run: Window 1's own log showed it became ready just 23 seconds *after* the launcher had already given up. Root cause: the wait loop assumed a warm model cache (~30s, matching Window 1's own banner text) and gave up after ~2 minutes, but a cold-start download of the Depth Anything weights from Hugging Face -- unauthenticated, so rate-limited/slower -- genuinely took longer than that. Raised Local mode's wait to ~6 minutes (90 attempts); Cloud mode was already 240 (~8 min) for VM boot + model load together. No Python touched, 209/209 suite unaffected.

## What's new in v19.8.6 (real-hardware findings: false volume + Cloud mode confirmed)

**Full end-to-end run confirmed working on real hardware** (both windows, both cameras, Pi bridge). Three things reported from that run:

1. **A pillow held close to the camera, covering most of the frame, was measured as "116 L".** Root cause: the safety check meant to reject exactly this (`realsense_max_item_volume_l` / `logitech_max_item_volume_l`) defaulted to 120 L — sized for a full wheelie bin, not a single tracked bag/box, so 116 L slipped through as "plausible" instead of being rejected. Lowered both to 90 L (still comfortably fits a large real box: 35 cm tall × 50×50 cm footprint ≈ 87 L). New regression test added; 209/209 suite passes.
2. **Height accuracy (35 cm real bag measured as 15 cm) and the RealSense/Logitech readings alternating are very likely the same root cause, and it's a calibration step, not a code bug:** the Logitech webcam's depth comes from a monocular AI model (Depth Anything), which only produces correct real-world metric distances after you tell it the true camera-to-empty-floor distance via the dashboard's "Set measured distance" field (Logitech card). Without an accurate value there, its height/volume numbers are systematically wrong, and can pass or fail the plausibility check inconsistently frame to frame — which looks like "cameras taking turns." **Action needed:** measure the real distance from the Logitech lens straight down to the empty bin floor with a tape measure (not by eye), enter it in that field, let it recapture its baseline, then retest with a rigid box/bag placed only inside the marked bin area (not covering the whole frame — a pillow or a hand blocking the lens is not a representative test).
3. **Cloud mode already matches the new gpu.py your collaborator sent** (zone name, `up`/`status`/`down` output format, `--fresh`/`--l4-only`/`--us` flags all already wired from earlier work) — no code change was needed. To use it: rename the attached script to exactly `gpu.py`, place it next to `Start-LocalLife-Demo.ps1`, then double-click `START_LOCAL_LIFE_CLOUD.cmd` instead of the Local one. The project still needs to be installed on the cloud VM once (same idea as the Pi setup step) — see the Cloud mode section below.

Known follow-up, not yet fixed: shadows occasionally being detected as objects is a known hard limitation of monocular depth estimation (it infers depth from image shading, and a shadow changes shading without changing real geometry). The tightened volume cap above reduces how often a shadow produces a plausible-looking reading, but the most effective mitigation right now is even, shadow-free lighting across the bin area when the baseline is captured.

## What's new in v19.8.5 (launcher-only patch)

Window 2 now opened as a real window (the v19.8.4 conhost.exe fix worked) but crashed instantly with `DEMONSTRATION ERROR: The term 'if' is not recognized as the name of a cmdlet...`. Root cause: a PowerShell 5.1 quirk in exactly how the previous round embedded the "add `--token` if one is set" logic — an `if/else` expression appended after a comment-interrupted line-continuation got misread as a bare `if` command instead of the keyword. This was a bug in the v19.8.2 token fix itself, invisible until now because Window 2 never actually opened before v19.8.4. Fixed by computing that piece as its own plain statement before building the command string, instead of inlining it. Verified: this is a Windows-PowerShell-5.1-specific parsing issue, so re-checked the fix against the documented PS 5.1 grammar rules for statement continuation; no Python touched, 208/208 suite unaffected.

## What's new in v19.8.4 (launcher-only patch)

Window 1 was clean end-to-end, but Window 2 (Raspberry Pi + cameras) never visibly opened — no error, nothing. Root cause: on Windows 11 with "Windows Terminal" set as the default terminal app, `Start-Process -FilePath powershell.exe` does not open a separate window at all — it opens a new TAB inside whichever Windows Terminal window is already open (Window 1), which is easy to miss if you're not looking at the tab strip. The launcher script itself was fine; every window-spawn line was reached and ran correctly, it just landed somewhere you weren't looking. Fixed: every window the launcher opens (App, Pi, and the Cloud tunnel) is now launched through `conhost.exe` (the legacy console host) instead of `powershell.exe` directly — this forces a genuine standalone window every time, regardless of the Windows 11 default-terminal setting. Verified: PowerShell parser clean; no Python touched, so the 208/208 test suite is unaffected. If you still don't see a second window after this update, check your taskbar for a second icon and Alt+Tab through open windows — but this should no longer be necessary.

## What's new in v19.8.3 (launcher-only patch)

Window 1 was reaching `Serving on http://0.0.0.0:8000` successfully (the server really is running at that point — that log line is not a hang) but then printing, every 3 minutes, `WARNING locallife_cloud.storage: Cloud Storage synchronization failed: [WinError 2] The system cannot find the file specified`. Root cause: a background thread tries to sync results to a Google Cloud Storage bucket, a feature built for Cloud mode (so results survive an ephemeral VM being torn down) that the launcher never disabled for Local mode, even though Local mode's whole point is "no cloud needed" and its own disk is not ephemeral. The `[WinError 2]` itself is a separate, real, purely cosmetic bug in `storage.py`: Windows can't launch a `.cmd` file (`gcloud.cmd`) the way `subprocess.run` was calling it without a shell. Fixed the actual problem, not just the symptom: Local mode's server now starts with `--disable-sync` (a flag that already existed for exactly this), so the sync thread never starts and the warning is gone entirely — nothing to configure. Verified: PowerShell parser clean, full test suite still 208/208 (no Python code touched, launcher-only change).

## What's new in v19.8.2 (launcher-only patch)

The v19.8.1 zip got further but then crashed with `DEMONSTRATION ERROR: The application stopped or could not start` / `error: Binding outside localhost requires LOCALLIFE_API_TOKEN`. This is a real, pre-existing security guard in the server (`server.py`): it refuses to bind to `0.0.0.0` (needed so the Raspberry Pi's reverse SSH tunnel can reach it) unless an API token is set, so an unattended demo never accidentally exposes an unauthenticated server on the network. The launcher was passing `--host 0.0.0.0` but never set a token — like the setup.py gap, this was always going to happen the moment a run got this far, and previous rounds never reached it. Fixed: the Launcher window now generates a random token each run and threads it through automatically — set as `LOCALLIFE_API_TOKEN` for the server (Window 1), passed as `--token` to the Raspberry Pi's uploader (Window 2), and embedded into the dashboard page itself so its own buttons (capture baseline, reset, calibrate, set Logitech distance) keep working. You don't need to type or configure anything — this is fully automatic. Verified in this sandbox with Flask's test client: the dashboard page now embeds the token, a protected endpoint correctly rejects a request with no token (401) and accepts one with it. The full test suite (208/208) and PowerShell parser both still pass. Not verified: your actual Windows machine end-to-end with both cameras; that's the next real test.

## What's new in v19.8.1 (launcher-only patch)

The v19.8.0 zip crashed at startup with `DEMONSTRATION ERROR: Failed to install LocalLife package` / `does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found`. Root cause: this project never actually shipped an installable package descriptor — only `requirements-*.txt` files — so the launcher's `pip install -e .` step was always going to fail the moment it was genuinely reached. It went unnoticed through round 8.6 because the launcher was crashing one step earlier (the `pip show` stderr-promotion bug fixed in v19.7.3); fixing that bug is what finally let a real run get far enough to hit this gap. Fixed by adding `LocalLife_Plug_and_Play_Local/setup.py`, a minimal `setuptools` descriptor that installs `locallife_cloud` as a real editable package and pulls in `requirements-local.txt`'s dependencies. Verified in this sandbox: built a throwaway virtualenv and confirmed `pip install -e .` now succeeds, `import locallife_cloud` works, and `pip show locallife-cloud` reports the package correctly — the exact sequence that was crashing. Not verified: your actual Windows/Python 3.14 environment; that's the next real test.

## What's new in v19.8.0

Real-hardware feedback this round: distance is good, color and material are good, but **volume and height are not accurate**. Reading through the volume/calibration code end to end (not guessing from symptoms) turned up a real bug, plus a new tool to help you check accuracy yourself with a known-volume object.

**The bug — calibration could quietly use the wrong number.** The "Calibrate from this object" button (v19.6.0) tells the backend the object's real known liters, but never sends the number it measured for that object — it relies on the backend to look that up itself. Before this fix, the backend's lookup was a separate, camera-wide "current volume" total, not necessarily the same number shown for your object in the live table. If an unconfirmed "depth silhouette" (a shadow, a leftover background blip — the kind of thing prior rounds already stop from being counted or deposited) happened to share the frame with your real reference object, its volume silently got folded into that camera-wide total. Calibrating in that moment solved a correction factor from "your object + some noise" instead of your object alone — and because that factor then multiplies into *every* future reading on that camera, one bad calibration click would degrade accuracy going forward, not just for that one measurement. The same mixing could also occasionally inflate the "CURRENT VOLUME" dashboard number itself when a phantom silhouette briefly coexisted with a real tracked bag or box.

**The fix:** calibration now always uses exactly the single confirmed object's own displayed reading — the same number you're looking at in the live table when you press the button — and refuses with a clear message instead of guessing if zero or more than one confirmed object is in view, or if only an unconfirmed silhouette is in view. The "CURRENT VOLUME" aggregate figure was tightened the same way, so a stray phantom detection can no longer inflate it.

Everything else in the volume math was read carefully and checked against this round's specific worries (does the calibration factor reach every camera and the fused result? — yes, confirmed, it already did; is the displayed height computed from the same accepted pixels as the liters figure? — yes, confirmed, the v19.7.0 `height_p90_m` fix already ties them together correctly; does the liters formula itself convert units correctly? — yes, confirmed by hand-deriving a known cuboid's volume through the exact formula and matching it exactly). No other volume/height bug was found by this reading — the one above was real and is fixed; the rest of your "not accurate" report is most likely explained by it (a poisoned calibration factor sticks around and quietly wrongs every later reading) plus the ordinary reality that a single overhead camera's own residual lens/stereo-calibration/mask-boundary bias needs the calibration feature to correct it per object — which is exactly what the new tool below is for.

**New: `scripts/validate_known_volume.py`.** A command-line tool you run on your own laptop against the real dashboard, using a known-volume reference object — your own proposal names a 1 L cube and a 2 L box. Place the object in view, run:

```
python scripts/validate_known_volume.py --camera realsense --known-liters 1.0 --label "1L-cube"
```

It reads the same live number the dashboard already shows (it does not recompute anything itself), prints the percent error against your stated known volume, and — run again for more trials — reports MAPE, median absolute percentage error, and 90th-percentile error across every trial recorded so far, the same shape of statistic your thesis proposal's own accuracy methodology uses. Pass `--calibrate` to also solve a calibration factor from that trial via the existing calibration endpoint; per its own advice, calibrate once against one object, then validate with a *different* object you didn't calibrate against. Run `python scripts/validate_known_volume.py --help` for every option.

**Honest status:** this sandbox has no camera hardware. Every fix above is verified by reading the exact code path, by hand-deriving a synthetic 1 L and 2 L cuboid's expected volume from the same pinhole-projection formula `estimate_volume()` itself uses and confirming the pipeline reproduces it (within ±15% uncalibrated, matching your proposal's own stated accuracy target, and to within ~1% once calibrated against that same object), and by the full automated test suite (208/208 passing, 11 new tests this round). None of it has run against your real cameras or a real 1 L/2 L object yet — that confirmation depends on your next test, ideally using `validate_known_volume.py` itself so you get real numbers, not just "looks fine."

Cloud mode (`-Mode Cloud`, `gpu.py`) was not touched this round — you asked to defer it while volume accuracy gets fixed, so every Cloud-mode code path was left exactly as it was in v19.7.3.

## What's new in v19.7.3 (launcher-only patch)

With a real Python interpreter now correctly detected (v19.7.2), the very next step failed with `DEMONSTRATION ERROR: WARNING: Package(s) not found: locallife-cloud` -- on a completely normal first run, before the package had ever been installed. That "WARNING" text is pip's own, ordinary output from the launcher's own check for whether the package is already installed; under Windows PowerShell's strict error handling, that routine stderr line was being promoted into a script-terminating error before the launcher's own "not installed yet? then install it" logic ever got to run. Fixed with a new `Invoke-NativeTolerantly` helper that every risky native command now goes through: `pip show`/`pip install`, the server process itself, `gpu.py`, and every `gcloud`/`ssh` connection. Real failures are still detected and still stop the launcher with a clear message -- only an external program's own incidental stderr output (which was never a failure to begin with) is no longer treated as one.

While fixing that, the same Cloud-mode transcript also showed an interactive `Store key in cache?` prompt from Windows' bundled `plink.exe` on first connection to a freshly created VM -- nothing in an unattended launcher window can answer that prompt, so it was fixed proactively too (`--strict-host-key-checking=accept-new` on every `gcloud compute ssh` call, `-o StrictHostKeyChecking=accept-new` on the Raspberry Pi SSH connection), so a fresh VM in a new zone, or a first-time Pi connection, no longer needs anyone to press "y".

## What's new in v19.7.2 (previous patch)

After the v19.7.1 fix, the launcher correctly located the project but then failed with `Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.` That text is not from this project — it is what Windows itself prints when you run the fake `python.exe`/`python3.exe` placeholder ("App Execution Alias") that Windows always puts on `PATH`, even on a machine where real Python was never installed. The launcher's Python check was trusting the first thing named `python`/`python3` it found, which is this placeholder whenever it comes first in `PATH`. Fixed: it now skips anything resolving under `...\WindowsApps\...`, tries the official `py` launcher first (installed by the real python.org installer, not shadowed by the placeholder), and otherwise checks every `python`/`python3` match on `PATH` by actually running it and confirming it reports a real `Python 3.x` version before trusting it. If truly no real Python is installed, you now get a clear LocalLife error pointing to python.org instead of Windows' own confusing Store message.

## What's new in v19.7.1 (earlier patch)

The launcher crashed at startup on Windows PowerShell 5.1 (the built-in `powershell.exe` most laptops run, as opposed to PowerShell 7) with `A positional parameter cannot be found that accepts argument 'LocalLife_Plug_and_Play_Local'`. Cause: two helper functions called `Join-Path` with three path segments in one call, which only PowerShell 7's newer `Join-Path` supports — Windows PowerShell 5.1's version only takes two (`-Path`/`-ChildPath`), so the extra segment had no parameter to bind to and the whole launch aborted. Fixed by nesting two-argument `Join-Path` calls, which works on both PowerShell versions.

None of v19.7.1, v19.7.2, or v19.7.3 touch v19.7.0's detection/volume/height pipeline fixes, or the cloud integration below — all three are launcher-startup-only patches.

## What's new in v19.7.0 (previous round)

Real-hardware testing on a pillow/backpack turned up four problems. All four are fixed in this round:

1. **A modest object was reported as ~116 L, covering most of the frame.** The scene-change detector was solid-filling the *entire* enclosed area whenever it bridged a real object to an unrelated nearby patch (most often the object's own cast shadow) — including all the untouched background between the two. It now only fills a region solid when that region is genuinely, densely "changed" throughout; a thin bridge to a shadow no longer inflates into "the whole background is the object."
2. **Shadows were sometimes detected as their own object.** Same root cause as #1 — fixed the same way.
3. **Height was badly wrong (a 35 cm bag reported as ~15 cm).** The dashboard's height figure was a raw median across the *entire* detection mask. For a dome-shaped or tapered object (a pillow, a slouched bag), most of that mask sits near the low, sloped edges — nowhere near the true peak — so the median undershoots badly. Height is now the 90th-percentile of the same accepted, hole-filled, outlier-rejected column field the liters figure itself is integrated from: the near-top surface, the way a person would actually measure it with a ruler.
4. **Detection flickered between which camera "saw" the object.** Largely a symptom of #1/#2: an unstable, shadow-swollen mask shape from frame to frame flips detection thresholds on and off. With the mask now tightly bound to the real object, this is substantially more stable. The dual-camera **Fused Result** panel (built in the previous round) remains the right way to get one stable number instead of watching two independent per-camera readings disagree.

All four fixes were verified against synthetic reproductions of the exact failure shapes (a real object with a cast shadow a few pixels away, a real dark object with patchy sensor dropout, a dome-shaped height field) plus the full test suite (197/197 passing, including 5 new regression tests for this round). As with every round, none of it is confirmed against your actual cameras yet — that's the next step.

Material and color classification were left untouched (already confirmed good).

## What's new in v19.6.0 (previous round, still included)

- Known-object calibration UI (enter a reference object's liters, calibrate)
- Phantom/unconfirmed detections no longer counted or deposited as waste
- Fused dual-camera output (one combined volume/color/material)
- Iterative depth-fill for patchy black-material dropout

## Two ways to run this: Local or Cloud

This package now supports **both**:

- **Local mode (default, free)** — everything runs on your Windows laptop. No account, no charges, ~2 minute startup. Use `START_LOCAL_LIFE_DEMO.cmd`.
- **Cloud mode (optional, for GPU muscle)** — runs on a Google Cloud GPU VM managed by `gpu.py`, tunneled back to your laptop. Use `START_LOCAL_LIFE_CLOUD.cmd`. Needs a Google Cloud account and incurs cloud charges while the VM is running.

Pick Local unless your laptop's own GPU/CPU is too slow for real-time inference and you specifically want cloud GPU power.

---

## Local Mode

### Before you start
- Python 3.10+ installed on Windows (`python --version`)
- OpenSSH Client enabled (Windows: Settings → Optional Features → OpenSSH Client)
- Raspberry Pi on the same network, both cameras connected, project installed at `~/LocalLife_Plug_and_Play_Local` on the Pi
- This zip's `LocalLife_Plug_and_Play_Local` folder placed next to this script (or in your Downloads/Documents — the launcher searches those automatically)

### Start
Double-click `START_LOCAL_LIFE_DEMO.cmd`. Two windows open:
1. **Laptop Application** — installs the package on first run, then starts the server (models load in ~10-30s)
2. **Raspberry Pi Cameras** — connects automatically, no password needed

The dashboard opens in your browser automatically once both cameras are streaming.

### Stop
Double-click `STOP_LOCAL_LIFE_DEMO.cmd`.

---

## Cloud Mode (via gpu.py)

### Why this replaced the old cloud launcher

The previous cloud demo pointed at one hardcoded VM name and zone. Per the email from your collaborator: **L4 GPUs are frequently sold out across most of Europe at once**, so "start the same VM in the same zone and hope" was never reliable. `gpu.py` fixes this by hunting across 11 European zones (and, with `--us`, US zones too) for whatever GPU capacity is actually available, falling back from L4 to T4 automatically, and by keeping the VM's disk (packages, caches, downloaded models) intact across restarts and even across a forced move to a different zone.

### Before you start
1. **Google Cloud CLI installed and signed in** (`gcloud auth login`)
2. **`gpu.py` placed next to this script** (included in this zip — copied from the attachment; put it anywhere on your laptop, but next to `Start-LocalLife-Demo.ps1` is simplest)
3. **The project installed on the cloud VM's disk image** at `~/LocalLife_Plug_and_Play_Local` (once — the image then carries it forward across restarts and zone moves)
4. Python 3 and OpenSSH Client on your laptop (same as Local mode)

### Start
Double-click `START_LOCAL_LIFE_CLOUD.cmd`. Three windows open:
1. **Cloud GPU** — runs `python gpu.py up` (can take 2-8 minutes if it has to hunt for capacity or move zones), then starts the server on the VM over SSH
2. **Secure Tunnel** — forwards your laptop's dashboard port to whichever zone the VM landed in (this can be blank/quiet — that's normal)
3. **Raspberry Pi Cameras** — connects automatically, same as Local mode

The dashboard opens automatically once everything is ready.

### Useful `gpu.py` commands directly (from this folder)
```
python gpu.py status          # what's running, where, and cloud disk state
python gpu.py up              # restart in place, or hunt for a GPU if moved
python gpu.py up --fresh      # force a move (e.g. to hunt for an L4 after getting a T4)
python gpu.py up --l4-only    # do not fall back to a T4
python gpu.py up --us         # also search US zones (data leaves the EU)
python gpu.py down            # save home dir to the bucket, stop the VM (~7 kr/day disk only)
python gpu.py down --delete   # capture an image, then delete the VM and disk entirely
python gpu.py ssh             # open a shell on the VM directly
```

### Stop
Double-click `STOP_LOCAL_LIFE_DEMO.cmd`. It closes the windows and asks whether to also run `python gpu.py down` (recommended — stops billing beyond the small disk cost). The VM also auto-stops itself after 30 idle minutes either way.

### Cost note
A stopped VM costs about 7 kr/day for its disk; after 7 idle days it's automatically captured as an image and deleted, and the next `up` simply rebuilds from that image. A running VM bills for GPU + compute time — always run `python gpu.py down` (or let the 30-minute idle auto-shutdown catch it) when you're done for the day.

---

## Calibrate (both modes, same steps)

1. Measure the distance from the Logitech camera lens to the empty bin surface (in meters)
2. Dashboard → Logitech panel → **Set measured distance** → enter the value
3. **Capture RealSense baseline**, then **Capture Logitech baseline** (empty bin visible to both cameras)
4. Add a bag or box — its volume, color, and material appear on both camera panels and in the **Fused Result** panel at the top
5. *(Optional, for best accuracy)* Place an object of known liters, enter it under **Known Volume Calibration** on each camera panel, click Calibrate. Verify with a different object afterward.

---

## If something does not work

**Local mode**
- "Python 3 not found" → install Python 3.10+, reopen PowerShell
- "Project not found" → make sure `LocalLife_Plug_and_Play_Local` (from this zip) is next to the launcher, or in Downloads/Documents

**Cloud mode**
- "Google Cloud CLI was not found" → install it and run `gcloud auth login`
- "gpu.py was not found" → make sure `gpu.py` is next to `Start-LocalLife-Demo.ps1`
- "gpu.py could not bring up a GPU VM" → try again in a few minutes, or add `-GpuArgs "--us"` when running the PowerShell script directly to also search US zones
- Dashboard never becomes reachable → check the Cloud GPU window for gcloud login prompts, and the Secure Tunnel window for a stalled connection

**Both modes**
- "Raspberry Pi cannot connect" → verify Pi power/Wi-Fi/cameras, and that `ssh locallife@locallife.local` works from PowerShell directly
- "Port 8000 already in use" → run `STOP_LOCAL_LIFE_DEMO.cmd` first, or edit the `.cmd` file to add `-Port 9000`
- "Logitech liters are missing" → set a real measured distance and recapture the empty baseline

Run `CHECK_LOCAL_LIFE_SETUP.cmd` any time to verify Python, SSH, and network settings before launching (Local mode by default — pass `-Mode Cloud` when running the PowerShell script directly to check Cloud mode's gcloud/gpu.py setup instead).

---

The launcher includes the shared Raspberry Pi demonstration credential requested by the device owner. It protects its local copy using Windows user-scoped encryption and does not display the password or place it in a visible command-line argument. Because the launcher itself contains a shared demonstration credential, keep the files private and use unique SSH keys for production.
