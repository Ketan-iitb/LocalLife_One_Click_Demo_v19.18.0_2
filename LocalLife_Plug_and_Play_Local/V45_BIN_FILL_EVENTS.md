# V45 – bin fill level and new-bag counter

Dashboard panel **"Bin fill & new deposits"** (below the streams on `/` and `/research`).

## Fill level (per camera, own profile)
- Saved per camera in `results/<camera>/bin_profile/fill_<camera>.json` via the panel form or
  `POST /api/cameras/<id>/fill-profile` (cm): `camera_to_empty_floor_cm`, `distance_kind`
  (`vertical` | `optical_axis`), `tilt_from_vertical_deg`, `usable_height_cm` (empty floor → rim),
  optional `camera_above_rim_cm`, `inner_length_cm`, `inner_width_cm`, `capacity_l` (default 660,
  unverified), `capacity_verified`.
- **Save with the bin EMPTY and the tripod fixed**: an edge snapshot outside the bin is stored; if the
  camera/bin moves, the reading becomes N/A ("camera or bin moved … re-measure").
- Height of every depth pixel above the empty floor: `h = H + up·P` (H = vertical camera height,
  optical-axis distance × cos(tilt) when measured along the axis). Pooled into 5 cm cells.
- Shown: maximum reliable fill height (p95 of cell tops), **height-based fill %** = that / usable
  height, remaining height, and **rough litres = capacity × fraction** ("rough height-based
  equivalent; assumes roughly uniform filling"). "Estimated occupied volume" only when the inner
  length/width are measured and ≥ 60 % of that floor area is seen.
- Refreshed only from settled frames (motion ≤ 2 %); otherwise the last reading is kept and marked stale.
- N/A with the reason: incomplete profile, no aligned depth, camera moved, < 30 % valid depth.
- RealSense: aligned hardware depth. Logitech: only when its own depth is calibrated from its own
  measured distance (`independent-measured-distance`); a RealSense-borrowed or unverified model
  scale → N/A. The 660 L figure is never used as a scale.

## New bags this session
- Counted only from finalised before/after occupancy events (enter → settle).
- Tracks visible during the first 5 s are the baseline (initial fill, not new).
- Not counted (logged as rejected candidates): baseline track, already-counted track, event id
  seen before, measured occupancy change < 2 L (old bag moved / occlusion).
- Both cameras within 12 s → one bag, both cameras listed as evidence; the better-measured
  camera's size is kept (RealSense on a tie).
- Hidden new bag (no measurable size) still counts; volume N/A with the reason.
- Per event: id, deposit time, cameras, track, colour, type, material (UNKNOWN unless the
  classifier is ≥ 0.5 confident), L×W (detector footprint), height = AFTER top − BEFORE surface
  under the bag's footprint (p90 of the risen cells), **new-bag outer envelope L** and
  **estimated change in bin occupancy L** as separate columns.
- `results/session/session_deposits.csv` (append-only, session id per row);
  `GET /api/session-deposits.csv`, `GET /api/bin-fill`.

## Still needed on site (per camera)
1. Camera-to-empty-floor distance and whether it is vertical or along the optical axis
   (the known ~110 cm belongs to one camera only).
2. Tilt from vertical.
3. Usable floor-to-rim height; camera-above-rim offset (cross-check).
4. Inner length and width (for occupied volume).
5. Capacity from the bin label.
6. A drawn measurement zone (excludes walls, rim and neighbours from bag L×W).
7. Logitech: its own metric calibration.
