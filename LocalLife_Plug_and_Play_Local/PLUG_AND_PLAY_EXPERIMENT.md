# LocalLife Plug-and-Play Volume Experiment

This is an isolated A/B copy of `LocalLife_Dual_Camera_Thesis`. It exists so
volume-estimation changes can be tested without changing the stable cloud
folder, stable launcher, or stable databases.

## Isolation contract

| Resource | Stable system | Experimental system |
| --- | --- | --- |
| Cloud code | `~/LocalLife_Dual_Camera_Thesis` | `~/LocalLife_Plug_and_Play_Experiment` |
| Dashboard port | `8000` | `8100` |
| Windows-to-Pi bridge | `18000` | `18100` |
| Cloud data | existing stable artifacts | `~/LocalLife_Plug_and_Play_Data` |
| Windows session | `LocalLifeDemo` | `LocalLifePlugAndPlayDemo` |
| Raspberry Pi code | `~/LocalLife_Dual_Camera_Thesis` | same read-only camera streamer |

Only one inference server and one Pi camera client should run at a time. The
NVIDIA L4 and the physical cameras should not be shared by two simultaneous
demonstrations. Switching launchers changes the running A/B system; it does not
delete either system's files.

## Why color can work while liters are blank

Color requires an RGB image and an object mask. Volume additionally requires:

1. aligned depth or a calibrated monocular depth estimate;
2. camera intrinsics;
3. an empty-bin depth baseline;
4. sufficient valid depth coverage;
5. physically plausible height and volume;
6. for thesis-grade Logitech results, a measured lens-to-empty-bin distance
   and acceptable angle. The experiment can show provisional values before
   that distance is entered, with at least 35% systematic uncertainty.

The stable application saved baseline arrays but did not reload them when the
cloud process restarted. This experiment restores the saved installation
profile, validates it against the first live frames, and reuses it only when
the camera view still matches. A moved camera or substantially changed bin
scene invalidates the profile and requests a new empty baseline.

## First installation setup

Perform this once after the experimental cloud folder is installed:

1. Fix both cameras in their final positions.
2. Remove every object from the measurement area.
3. Wait until both panels say `LIVE VIDEO`.
4. Keep the views empty and still while automatic setup verifies nine frames.
5. Wait for both automatic setup states to say `ready`.
6. Test one actual garbage bag in the shared measurement area.

After a normal cloud restart, the saved setup is restored and validated
automatically. If a camera or scene moved, the profile is rejected and rebuilt
after another stable, empty run.

## Data integrity

Unmeasured detections remain visible for diagnosis but are not written into
the experimental object ledger. Negative prompts for pillows, cushions,
blankets, bedding, chairs, furniture, bottles, and lotion bottles help prevent
common non-waste objects from entering the bag/box database.

## Live transport and volume integration

Camera upload is decoupled from GPU inference. The web preview publishes each
new frame immediately, while a fair worker analyzes the newest pending frame
from each camera and discards stale queued frames. This keeps both views live
when YOLOE or Depth Anything is slower than camera capture.

The default experimental geometry is `triangulated-surface`, adapted from
VolPy's point-to-triangle-to-plane integration idea. It backprojects the empty
reference with calibrated intrinsics, splits each image cell into two planar
triangles, and integrates the measured height field over those triangles. The
other three integration modes remain selectable for controlled comparisons.

## Rollback

Close the experimental windows and run the original stable
`START_LOCAL_LIFE_DEMO.cmd`. The original launcher uses port 8000 and starts
`~/LocalLife_Dual_Camera_Thesis`; no experimental calibration or history is
read by the stable application.
