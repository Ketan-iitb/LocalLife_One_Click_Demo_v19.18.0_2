"""Trim mask leakage before an object's footprint is measured.

The footprint is the minimum-area rectangle over EVERY mask point
(footprint.estimate_extents), so a few pixels leaked onto the rim, a wall or a
neighbouring bag stretch it: ~1 % of points 45 cm away turned a 40 cm bag into
a 90 cm one (tests/test_v45_deposits_live.py). Two leak shapes are removed:

* islands not connected to the main body (after bridging gaps of a few pixels);
* thin strips (spill into a touching neighbour), only when removing them keeps
  >= 85 % of the mask -- a thin object keeps its whole mask.

A thick, contiguous merge with a neighbour cannot be told apart here; that is
still flagged by crowded_scene / merged_neighbour_reason.
"""

from __future__ import annotations

import numpy as np

KEEP_FRACTION = 0.85


def trim_mask_leak(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return mask, False
    area = int(np.count_nonzero(mask))
    if area < 200:
        return mask, False
    binary = mask.astype(np.uint8)
    bridge = cv2.dilate(binary, np.ones((5, 5), np.uint8))
    count, labels = cv2.connectedComponents(bridge, connectivity=8)
    keep = mask.copy()
    if count > 2:
        owners = labels[mask]
        main = np.bincount(owners).argmax()
        keep = mask & (labels == main)
    side = max(3, int(round(0.08 * np.sqrt(area))) | 1)            # ~8 % of the object's size
    kernel = np.ones((side, side), np.uint8)
    solid = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    body = cv2.dilate(cv2.morphologyEx(solid, cv2.MORPH_OPEN, kernel), kernel)
    count, labels = cv2.connectedComponents(body, connectivity=8)
    if count > 1:
        owners = labels[keep]
        owners = owners[owners > 0]
        if owners.size:
            opened = keep & (labels == np.bincount(owners).argmax())
            if np.count_nonzero(opened) >= KEEP_FRACTION * np.count_nonzero(keep):
                keep = opened
    trimmed = int(np.count_nonzero(keep)) < area
    return (keep, True) if trimmed and np.count_nonzero(keep) >= 3 else (mask, False)
