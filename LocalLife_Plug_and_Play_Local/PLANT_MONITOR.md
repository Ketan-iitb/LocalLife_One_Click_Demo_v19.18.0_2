# Waste plant monitoring station

This version expands the fixed overhead RealSense station from bag-only
measurement into a running waste-plant register for bags and cardboard boxes.

## Live operational metrics

- Total garbage bags observed.
- Total cardboard boxes observed.
- Total bags deposited after a stable depth measurement.
- Total boxes deposited after a stable depth measurement.
- Cumulative externally visible volume of deposited objects.
- Live volume of the current object and occupied volume of the complete bin.
- Dominant color of every tracked object.
- Observed count, deposited count, and deposited liters for each color.
- Optional waste-stream totals when the plant supplies an actual color mapping.
- Chronological history with time, tracking ID, object type, class, color,
  configured stream, liters, and observed/deposited status.

One stationary bag or box is recorded only once. A newly tracked item first
appears as **observed**. After the configured number of stable measured frames,
it becomes **deposited** and the occupied scene automatically becomes the
reference for the next item. Manual **Commit current item** remains available.

Automatic deposit means a stable newly arrived object has been observed inside
the calibrated bin. It does not prove its complete physical falling trajectory;
trajectory verification would require additional motion-event instrumentation.

## Color and contents

Detected bag color can be reported directly, but a camera cannot know what a
particular color means at a plant unless its conventions are configured. Add a
quoted mapping to `cloud.env` only after confirming the actual facility rules:

```bash
LOCALLIFE_COLOR_MAP="blue:plastic,green:organic,black:general waste"
```

The values above are examples, not claims about the site's real conventions.
Without configuration the dashboard correctly reports color while labeling the
waste stream as **not configured**.

## Settings

```bash
LOCALLIFE_BAG_ONLY=false
LOCALLIFE_ALLOW_UNCLASSIFIED=false
LOCALLIFE_CONF=0.12
LOCALLIFE_MIN_PIXELS=700
LOCALLIFE_MIN_HEIGHT_M=0.025
LOCALLIFE_TRACK_CONFIRM=4
LOCALLIFE_AUTO_DEPOSIT=true
LOCALLIFE_SETTLE_FRAMES=5
LOCALLIFE_SETTLE_TOLERANCE=0.12
LOCALLIFE_HISTORY_LIMIT=100
```

Keep `LOCALLIFE_ALLOW_UNCLASSIFIED=false` to avoid counting arbitrary depth
noise as bags or boxes. Increasing `LOCALLIFE_SETTLE_FRAMES` delays automatic
deposit but requires a more stable physical measurement. The history limit
affects the dashboard only; the CSV export includes the complete ledger.

## History and restart recovery

The persistent append-only ledger is stored at:

```text
artifacts/waste_plant_ledger.jsonl
```

Dashboard totals and history survive restarting the cloud service when the same
`LOCALLIFE_RESULTS_DIR` is retained. Machine-readable summaries are available at:

```text
/api/history
/api/history.csv
```

The CSV includes the complete history and is also downloadable from the
dashboard for thesis evaluation, plant reporting, or spreadsheet analysis.

## Measurement boundaries

Volume is the estimated externally visible occupied volume relative to the
previous settled scene. Cumulative deposited volume and current bin occupancy
are different quantities: accumulated objects can compress, overlap, and hide
surfaces. Real-world accuracy must still be evaluated against representative
bags and boxes with independently measured reference volumes.
