# V45 – bin fill level and NEW-bag counter

Dashboard panel **"Bin fill & new deposits"** (below the streams on `/` and `/research`).

## NEW bags this session
Every processed frame of each camera feeds a time-based watcher (`session_deposits.py`):
`initialising → watching → candidate → settling → confirmed | rejected → watching`.

- **Baseline:** once per session, after ≥ 4 s and 1.5 s of stillness (or 20 s at most), the
  current view, depth and tracks become the baseline. Bags already in the bin count toward the
  fill, but NEW stays 0.
- **Candidate:** motion in the event region, or a track not already known (by id, or by box overlap
  so an ID switch is not new).
- **Settling:** 1.5 s of elapsed stillness, so a low frame rate still settles. After 30 s without
  settling the candidate is rejected with that reason.
- **Confirmed** when the settled view differs from the last committed one by:
  - a new bag-like track over a persistent change;
  - an existing mask that grew into a touching new bag; or
  - a compact, persistent new region (with a local depth rise where depth exists).

  None of this needs a fill profile, litres or a rise of the bin's maximum height.
- **Rejected, with the reason stored:**
  - no persistent change (hand, occlusion);
  - change explained by a known bag moving;
  - a vacated old spot (a bag that was never tracked has moved);
  - change without a surface rise (RealSense);
  - scattered change.

  Detector ID churn is counted, not logged per row.
- **Pillow/textile labels** count as bag-like (they are misread bags); person/hand never.
- **Both cameras within 12 s** count once, with evidence kept per camera. If more than one pairing
  is possible the event is marked "ambiguous" and merged with the nearest.
- **Persistence:** the session and events are saved to `results/session/session_state.json` and
  resumed after a restart; a browser refresh changes nothing. The CSV
  `results/session/session_deposits.csv` gets one row per event (written after the merge window)
  and one per rejected candidate.
- **New session:** the "Start new deposit session" button, `POST /api/bin-fill/session/new`, or the
  operator's START NEW SESSION.
- **Colour and change detection:** motion and change use the per-channel maximum difference. A red
  bag on a dark pile can match its grey level.
- **Per event:**
  - colour and material by temporal consensus (UNKNOWN unless ≥ 3 confident votes);
  - L×W from the detector footprint;
  - height = after top − before surface under the bag, where depth allows;
  - envelope L; occupancy Δ (whole-bin tracker, evidence only).

  A count with volume N/A is allowed.
- **Event region:** the drawn measurement zone if present, otherwise the configured ROI / bin
  polygon. A missing polygon never disables counting.

## Fill level (per camera)
- **Provisional defaults**, shown at start-up and never written to disk:
  - floor distance 110 cm for the camera named by `LOCALLIFE_FLOOR_DISTANCE_CAMERA`; otherwise
    the Logitech's own measured reference distance if one is set; otherwise the RealSense, shown as
    an assumption;
  - usable height 100 cm (approximate: 110 cm − an estimated 10 cm above the rim);
  - capacity 660 L, nominal.

  The other camera gets no invented distance.
- **Installation settings** (collapsed): only changed fields are saved
  (`POST /api/cameras/<id>/fill-profile`, cm). Saved values survive restarts and always take
  precedence over the defaults.
- **Profile status:** "approximate" while the tilt or distance kind is not measured (the camera is
  assumed to look straight down); "measured" only when both are given.
- **Shown:**
  - max reliable waste height (p95 of 5 cm cell tops);
  - HEIGHT fill % = that / usable height;
  - rough litres = capacity × fraction ("height-based approximation; assumes roughly uniform
    filling");
  - occupied volume only with measured inner length/width and ≥ 60 % of the floor seen;
  - remaining height, last processed frame, last valid measurement, reason when unavailable.

  Example: 30 cm of 100 cm = 30 % HEIGHT fill ≈ 198 L of 660 L.
- **Sources:** RealSense uses aligned depth. Logitech is unavailable until its own depth is
  calibrated from its own measured distance.
- **Warning:** shown if depth reaches well below the assumed floor, which means the distance may
  belong to the other camera.

## Dimensions
The footprint is the minimum-area rectangle over every mask point. About 1 % of pixels leaked onto
a rim or wall, or a thin spill into a neighbour, turned a 40 cm bag into 77–90 cm
(`tests/test_v45_deposits_live.py`). Masks are now trimmed of disconnected islands and thin strips
before measurement (`mask_leak.py`, flag `leaked_mask_pixels_trimmed`). A thick merge with a
neighbour is still only flagged. RealSense hardware depth is unchanged, and no scale factor was
added.

## On-site check
1. Start and wait for **"Watching for deposits"** (about 5 s with a still view).
2. Drop bag A, wait about 3 s: NEW = 1 and "Last confirmed deposit" updates.
3. Drop bag B: NEW = 2.
4. Push an old bag: NEW stays 2, and the rejection reason is shown.
5. If the 110 cm was measured from the Logitech, correct it in Installation settings.
