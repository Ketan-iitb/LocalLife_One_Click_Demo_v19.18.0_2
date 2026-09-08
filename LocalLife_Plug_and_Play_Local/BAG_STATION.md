# Fixed overhead garbage-bag volume station

This candidate is specifically intended for a stationary Intel RealSense D435
mounted above one garbage receptacle. Its operational assumption is that objects
deposited inside the calibrated bin opening are garbage bags.

## What changes

- YOLOE receives garbage-bag and sack prompts only.
- The fixed bin opening can be restricted to a rectangular ROI and an optional
  precisely traced normalized polygon.
- A depth-supported silhouette recovers the complete physical bag even if the
  detector recognizes only part of it. Depth-only silhouettes are explicitly
  identified and never assigned fake 100% recognition confidence.
- Empty-bin depth is aggregated from up to nine recent frames using the temporal
  median, rejecting isolated unstable RealSense measurements.
- The edge bridge requests the RealSense high-accuracy preset when supported and
  applies spatial and temporal depth filters by default.
- Isolated missing depth pixels can be repaired from nearby measurements;
  substantial missing regions are never invented. Original depth coverage and
  an approximate measurement uncertainty remain visible.
- Per-track bag volume is stabilized with a five-frame rolling median.
- The original empty-bin reference is preserved for total occupied volume.
- **Commit settled bag** records the current bag and uses the occupied bin as
  the reference for measuring the next arriving bag.

## Operational workflow

1. Fix the camera rigidly above the receptacle, approximately perpendicular to
   the opening.
2. Adjust `LOCALLIFE_ROI` to exclude the rim and surrounding area.
3. Optionally trace the actual bin opening using normalized image coordinates:

   ```bash
   LOCALLIFE_BIN_POLYGON=0.12:0.10,0.88:0.10,0.91:0.91,0.09:0.91
   ```

   Replace these illustrative coordinates with the real installation geometry.

4. Start the Pi bridge and let at least nine frames arrive with the bin empty.
5. Click **Capture empty-bin baseline** and verify RealSense depth and camera
   intrinsics are present.
6. Deposit a garbage bag and wait approximately five uploaded frames for the
   volume estimate to settle.
7. Read **New bag volume**, its coverage, and its approximate uncertainty.
   **Total bin volume** separately estimates all visible occupied space.
8. Click **Commit settled bag** before introducing the next bag.

## Measurement limitations

The result estimates externally visible occupied volume above the reference
surface. It is not a bag's advertised capacity, the volume of its contents
excluding trapped air, or a closed-surface 3-D reconstruction. A single overhead
camera cannot directly observe the underside, occlusions, or material displaced
beneath overlapping bags.

Displayed uncertainty accounts for configured depth noise and missing depth
coverage. It does not capture every systematic error, including camera tilt,
intrinsics error, misalignment, reflective plastic, bag deformation, baseline
drift, or an inaccurate bin polygon.

Actual accuracy must be established using representative bags with independently
known external volumes. Compare predicted versus reference liters using MAE,
RMSE, percentage error, and bias. Without physical reference measurements, a
real-world accuracy percentage cannot be established honestly.

## Deployment status

This candidate is separate from the existing `LocalLife_Cloud_V15` installation.
Replacing an active VM deployment requires an explicit decision and a backup of
the existing working version.
