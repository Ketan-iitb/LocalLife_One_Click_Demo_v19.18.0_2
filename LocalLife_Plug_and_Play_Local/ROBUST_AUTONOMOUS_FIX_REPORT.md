# Robust Autonomous Measurement Build 17.0

## What the live screenshots proved

The previous dashboard could draw a neural box while leaving the tables empty.
The tracker was created only once, before depth became valid, so a later valid
measurement never entered history. Generic prompts also matched a laptop,
chair/shelf objects, and the contents of an open bag. IoU-only tracking lost a
bag when its mask changed size, and one-frame detector gaps removed the live row.

## Implemented reliability chain

1. Automatic empty-scene setup waits for nine stable, object-free frames, saves
   a temporal depth profile, restores it after restart, and rebuilds it after a
   moved-camera rejection. A bag detected by either camera blocks setup.
2. Waste-specific prompts replace generic shopping- and paper-bag prompts.
   Confidence, minimum footprint, side length, maximum scene fraction, ROI, and
   negative lookalike gates run before tracking or volume integration.
3. Nested prompt boxes are collapsed. Tentative one-frame hits are not drawn as
   confirmed objects.
4. Tracking combines mask-box overlap, centroid motion, scale change, and object
   family. Confirmed tracks remain visible through six missed inference frames.
5. If one camera temporarily misses a confirmed bag, the other camera can
   authorize depth/RGB silhouette recovery; depth alone never invents a new
   semantic bag without that support or an existing track.
6. RealSense and Logitech volumes require sufficient depth coverage, sufficient
   foreground height inside the mask, plausible maximum volume, and three
   mutually stable readings. Unstable values remain pending and cannot enter
   cumulative totals or history.
7. A confirmed track is checked for database eligibility on every frame. This
   fixes the delayed-measurement history bug while retaining one record per ID.
8. Bag colour is sampled from the material contour instead of an open bag's
   contents, then smoothed across the track.
9. The web page has no manual setup, ROI, commit, distance, or calibration
   buttons. It displays automatic setup state, confirmed live tracks, pending
   reasons, stable history, and CSV exports.

## Verification

The hermetic suite contains 135 tests and covers automatic setup, saved-profile
validation, depth geometry, volume uncertainty, false prompt conflicts,
dropouts, deforming masks, delayed history insertion, peer-camera recovery,
open-bag colour, impossible volumes, duplicate deposits, data persistence,
stream backpressure, and multi-camera isolation.

## Accuracy boundary

RealSense volume integrates the visible height field above an empty reference;
it is not the manufacturer's nominal bag capacity and cannot reconstruct hidden
surfaces or trapped air. Logitech is an RGB-only monocular estimate and is shown
with a large systematic uncertainty. A numerical accuracy claim requires a
labelled trial set of physically measured bags; screenshots are diagnostic
evidence, not metric ground truth.
