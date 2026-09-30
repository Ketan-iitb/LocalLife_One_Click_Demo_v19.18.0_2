# v44 — 660 L bin: calibration profile and crowded-bin geometry

Branched from `Working_branch_v42_local_cloud_telemetry` @ `48f2583` (old branches untouched).

## What changed
- v40 crowded-bin work brought in (cherry-picked, originals untouched): touching bags kept apart,
  Logitech height from the local support surface, one-time Logitech depth gain, implausible-scale guard,
  range/footprint gates, moved-camera handling.
- Logitech no longer clips heights at the 0.80 m cap: a mask mostly above the physical bound is
  `height_exceeds_physical_bound` (V42 reported ~0.79 m for every bag in the deep bin).
- A bag on an uneven pile (support unknown) is `support_unknown_on_uneven_pile` instead of pile + bag height.
- Per-camera bin profile (`<results>/<camera>/bin_profile/<camera>.json`, API
  `GET/POST /api/cameras/<camera>/bin-profile`): nominal EN 840 660 L bounds marked UNVERIFIED until
  measured; bounds reject (never clamp) impossible L/W/H; moved-camera check on empty-bin frames only.
- Overlay: short `#id type` tags that do not overlap; full geometry stays in the tables/CSV.

## On-site checklist (per camera, empty bin)
1. Confirm the bin label/product (660 L?) — capacity is only a sanity bound.
2. Mark the optical centre (RealSense: front glass centre line; C920: lens centre).
3. Tape: optical centre → empty floor (m), optical centre above rim (m), inner depth, inner length and
   width at rim and at floor; note tilt (phone inclinometer) and ±uncertainty in mm.
4. Place ≥3 rigid objects of known size (one box) at ≥5 positions (centre, near/far, left/right);
   record the dimensions this camera measures.
5. POST the measurements + references to `/api/cameras/<camera>/bin-profile`; it is accepted only
   if median error ≤5 % and worst ≤12 %. Re-measure after moving a tripod.

Synthetic tests only — no real-bin accuracy has been measured yet.
