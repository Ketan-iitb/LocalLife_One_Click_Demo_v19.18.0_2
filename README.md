# LocalLife waste-bin measurement: core code for review

This branch holds only the main Python modules of the measurement system. The full project also has launchers, cloud deployment, a dashboard, tests and documents; those are not included here. The code comes from the development branch `Working_branch_v47`.

## What the system does
Two cameras look into a nominal 660 L waste bin. Each camera works **independently**: nothing is copied between them and they are never forced to agree.

- **Intel RealSense D435:** colour plus aligned hardware stereo depth.
- **Logitech C920:** colour only. Depth comes from a monocular depth model (Depth Anything V2), so its metric scale is approximate.

For each camera the system:
1. detects and tracks bags (colour, material, type);
2. measures each bag's size and volume from depth;
3. estimates how full the bin is;
4. counts **new bags dropped this session**.

## Processing flow (per camera)
`frame + depth → inference.py (detect/segment) → tracking.py (track IDs) → pipeline.py (measurement chain) → bin_fill.py (fill level) + session_deposits.py (new-bag events)`

`comparison.py` runs the two cameras side by side, each with its own `VisionPipeline`.

## Files
| File | Role |
|---|---|
| `pipeline.py` | Main per-camera pipeline: detection results, masks, depth, measurement, events |
| `comparison.py` | Runs both cameras independently; holds the shared session counter |
| `config.py`, `types.py` | Settings and data types (`Detection`, intrinsics, …) |
| `inference.py`, `tracking.py` | Object detection/segmentation and track IDs |
| `geometry.py` | Masks, region of interest, colour classification |
| `footprint.py`, `mask_leak.py` | Bag length × width; removal of mask leakage (a few leaked pixels had stretched 40 cm to 77–90 cm) |
| `volume.py`, `heightmap_volume.py` | RealSense volume from the 3D point cloud and height map |
| `logitech_volume.py` | Logitech volume from monocular depth, with physical bounds |
| `bin_profile.py` | Nominal 660 L bin profile (unverified) and physical plausibility checks |
| `bin_fill.py` | **Fill level**: finds the empty-bin floor automatically, excludes walls, fill = average waste height ÷ 100 cm usable height, and gives the per-bag volume above the surrounding surface |
| `bin_occupancy.py` | Before/after change in whole-bin occupancy per deposit (bounded, closes after at most 30 s) |
| `session_deposits.py` | **New-bag counter**: per camera, initialising → watching → candidate → settling → confirmed/rejected |

## Main ideas in the latest changes
- **Fill level:**
  - Each camera finds the bin floor in its own depth: a RANSAC plane through the deepest points.
  - For RealSense the floor is checked against the ~110 cm camera-to-floor reference.
  - For Logitech the scale is set from that reference and corrected every frame for the model's scale drift, using the unchanging walls/rim outside the bin.
  - Steep wall surfaces are ignored.
  - Height counts only where a bag is detected or a flat pile stands at least 15 cm high.
  - Litres = 660 × fill fraction. This is an approximation, not a measured volume.
- **New-bag counting:**
  - Bags present at start are the baseline: they count toward fill, not as new.
  - A deposit needs motion followed by a persistent change after settling (timed, so a low frame rate still works).
  - Rejected and logged with a reason: an existing bag moving (its old spot is vacated), a bag being removed (the surface dropped), lighting changes, detector ID switches, and candidates that never settle (30 s timeout).
- **Bag size:** measured from the deposit's own change region: height = surface after minus surface before under the bag; volume = that rise summed over the bag's footprint area.
- **No clamping:** implausible numbers are reported as unavailable, with a reason.

## Limitations
- The Logitech depth scale is approximate (monocular).
- Bag volumes have not yet been checked against reference objects of known size; no accuracy figure is claimed.
- The 660 L capacity and 100 cm usable height are nominal values.
