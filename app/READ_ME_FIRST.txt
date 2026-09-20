LOCAL LIFE ACCURACY DEPLOYMENT v3.0
===================================

Normal use:
1. Extract this whole ZIP.
2. Power the Raspberry Pi and connect BOTH cameras.
3. Double-click START_LOCAL_LIFE_DEMO.cmd.
4. START automatically updates the cloud accuracy code and Raspberry Pi camera client when required.
5. The browser must say: Accuracy Deployment v3.0.

What v3 adds
------------
PHASE 1
- Multi-frame median depth stabilization.
- Existing RealSense spatial/temporal filtering retained.
- Noise-aware depth thresholding retained and integrated with acceptance.
- Conservative Logitech object-mask cleanup.
- Interior multi-pixel HSV colour voting with temporal colour confidence.
- Safer deposit acceptance using depth coverage, uncertainty, quality and colour confidence.
- Logitech exposure/white-balance lock after camera warm-up (best effort).
- Rolling reference is preserved and temporal buffers are reset after each accepted drop.

PHASE 2
- Persistent multi-point physical Logitech depth calibration.
- Affine calibration with 2 points; quadratic calibration when 3+ useful points exist.
- Calibration rejects non-monotonic/pathological fits.
- Recalibration invalidates the old Logitech baseline and requires recapture.
- No invalid pixel-to-pixel RealSense/Logitech calibration is performed without extrinsics.

PHASE 3
- Matched physical drops receive a confidence/uncertainty-weighted operational fused volume.
- Research values from both cameras remain independent and unchanged.
- Operational totals count one physical matched drop once.
- Per-colour bag counts and fused volume totals are shown on the operator dashboard.
- Colour -> content meanings can be edited ad hoc without changing raw detected colour.
- Training-sample endpoint stores real images, masks and metadata for later site-specific fine-tuning.

Important
---------
A site-specific fine-tuned segmentation model is NOT fabricated in this package. Real dumpster images must first be collected and human-reviewed. v3 includes the data-capture hook and the runtime already supports changing LOCALLIFE_DETECTOR_MODEL to a fine-tuned weights file.

Research dashboard: /research
Operator dashboard: /
