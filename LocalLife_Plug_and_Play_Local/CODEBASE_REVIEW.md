# Technical review: reconstructed V14 versus cloud-ready V15

## Confirmed defects in the supplied archive

1. **There is no object-detection model.** `logitech_foreground()` performs LAB background subtraction. It cannot recognize waste classes, and without `SET EMPTY SCENE`, it returns an all-zero mask.
2. **Multiple objects are discarded.** `keep_largest_component()` intentionally deletes every foreground region except the largest. It cannot correctly count or measure multiple visible items.
3. **The active ROI is restrictive.** The original normalized ROI starts at `(0.20, 0.18)` and spans `(0.60, 0.62)`. Objects outside that central area are ignored.
4. **Counting is manual.** `/api/accept` increments the counter after a button press; no persistent object tracking exists.
5. **Monocular volume never appears.** The Windows worker always posts `{"volume_l": null}`. The dashboard can therefore never display a Depth Anything volume.
6. **Cloud execution fails on device assumptions.** The result path is hard-coded to `/home/locallife/LocalLife/v14_results`; `VideoCapture(0)` assumes a local USB camera; and RealSense initialization requires physically attached hardware.
7. **Hardware failures are suppressed.** The RealSense capture loop uses `except Exception: pass`, hiding alignment, timeout, camera, or stream errors.
8. **Cross-camera fusion is not geometrically valid.** The Logitech and RealSense cameras have different optical centers. Their RGB and depth frames cannot be combined pixel-for-pixel without extrinsic calibration and reprojection.
9. **There is no training, validation, checkpoint, or dataset-quality workflow.** The archive cannot diagnose poor precision/recall or survive a Spot interruption cleanly.

## Changes in V15

- The edge bridge streams RealSense-aligned RGB/depth and matching intrinsics to the GPU VM.
- Prompted YOLOE instance segmentation detects waste-related object categories, with a multi-component foreground fallback.
- The L4 executes a larger metric Depth Anything model in mixed precision.
- Persistent tracking confirms and counts each object once per appearance.
- Baseline-subtracted volume is computed per object and per frame using the camera's pinhole geometry.
- Monocular volume is emitted only after calibration against aligned RealSense measurements.
- A YOLO dataset auditor checks labels, splits, normalized coordinates, class IDs, missing annotations, and imbalance.
- Training auto-selects a medium detection or segmentation model, uses GPU-safe batch sizing, saves every epoch, and syncs checkpoints to Cloud Storage.
- Batch processing produces per-frame CSV outputs and reference-volume evaluation metrics suitable for the thesis.
- A GPU benchmark compares batch sizes using measured throughput, latency, and peak CUDA memory.
- The cloud dashboard is localhost-only by default and is accessed over an SSH tunnel.
- The unchanged reconstructed archive is preserved in `legacy/`.

## Remaining empirical work

Detection performance, volume accuracy, and GPU throughput cannot be established from source code alone. Evaluate the system with your actual camera, lighting, calibration geometry, annotated waste images, and independently measured reference volumes. A custom-trained model will generally be more trustworthy than zero-shot labels for domain-specific waste categories.
