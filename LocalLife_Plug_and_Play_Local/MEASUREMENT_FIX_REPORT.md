# Measurement and dashboard repair report

## Outcome

This build reports each confirmed garbage bag immediately with a track ID and
color, then reports liters as soon as the camera-specific measurement gates are
satisfied. A missing liters value is never silent: the live table displays the
specific pending state, such as empty baseline, RealSense depth, camera
intrinsics, calibration, depth coverage, or measurable height.

## Root causes repaired

1. The dashboard read **BAGS SEEN**, **BOXES SEEN**, and color totals from the
   durable ledger. `LOCALLIFE_RECORD_ONLY_MEASURED=true` correctly keeps
   uncalibrated observations out of that ledger, but it also made live counts
   appear blank. Live session totals are now independent from durable history.
2. Per-object volume passed `detection.mask` directly. If a detector supplied a
   valid bounding box but no mask, `None` caused the estimator to integrate the
   entire camera ROI. The resolved instance mask now always uses segmentation
   when present and a clipped bounding-box fallback otherwise.
3. The browser annotated the newest raw preview with an older inference result.
   The measurement view now uses the exact processed frame associated with its
   boxes, color, height, and liters.
4. The same large YOLOE model was called separately for RealSense and Logitech.
   The newest packet from each camera is now sent through one shared detector
   batch; only Logitech runs monocular depth. Stale frames remain deliberately
   replaceable so the queue cannot grow without bound.
5. Small RealSense depth-hole repair looped over invalid pixels in Python several
   times per frame. It is now vectorized while retaining the conservative rule
   that a hole needs at least five valid 3x3 neighbours.
6. Color smoothing previously applied only to Logitech. Both cameras now smooth
   the tracked bag color across recent frames. The existing object-interior
   classifier explicitly covers black, white, grey, red, orange, yellow, brown,
   green, cyan, blue, purple, and pink.

## Volume method and VolPy reference

VolPy turns a 3-D point survey into a Delaunay triangular mesh and integrates
the plane over each triangle. That concept is useful, but the package is for
terrain surveys rather than RGB-D cameras. LocalLife therefore uses the same
mathematical pattern directly on the calibrated image grid:

1. align RealSense depth to its RGB pixels;
2. subtract the current surface from the empty or previously accepted reference;
3. back-project pixels using factory camera intrinsics;
4. split each surface cell into two triangles;
5. integrate positive object height over the triangular footprint;
6. convert cubic metres to liters and propagate sensor, missing-depth, and
   configured systematic uncertainty.

This is an apparent occupied-volume measurement of the visible height field,
not a claim that one overhead camera observes the bag's hidden underside. Exact
accuracy must be established with independent known-volume bags not used for
calibration.

Reference reviewed: `agu3rra/volpy`, branch `master`, commit
`0602f03dcaac25e147fb490099f32d60b05b7305`.

## Verification

- Python compilation: passed.
- Automated suite: **127 tests passed**.
- Added regression coverage for live unmeasured bag/color totals, bounding-box
  volume isolation, dual-camera detector batching, and orange-bag color.
- The stable cloud folder, port 8000, Pi project, and stable data are not changed
  by this experimental package.
