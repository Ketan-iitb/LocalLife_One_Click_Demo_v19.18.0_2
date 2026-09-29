"""Keep one bag's measurement to that bag when bags touch.

RealSense recovers a bag's full surface from a partial detector mask by taking
the connected region standing above the support plane (volume.py,
`recover_elevated_object_mask`, protected). On an open floor that region is the
bag. In a bin every bag stands above the bin floor and they touch, so the
connected region is the pile: one bag's seed could claim its neighbours, the
bin wall and old waste, up to six times the seed's own size -- one plausible
way a single bag becomes a 791 mm object or one giant box over several bags.

On an open floor the old behaviour stands -- a logo-sized seed on a pale bag
still recovers the whole bag. Only when the recovered region runs into another
detection or out of the picture (the pile, the bin wall) is it bounded: it may
not enter another detection, it may not
reach far beyond the seed's own extent, and it keeps only the part connected
to the seed. If what remains is still far larger than the seed, recovery is
refused and the detector's own mask is measured, with a flag saying why.
"""

from __future__ import annotations

import numpy as np

# Recovery may extend the seed's box by this fraction of its size per side.
MAX_BOX_GROWTH = 0.35
# ... and may not end up more than this many times the seed's area.
MAX_AREA_GROWTH = 3.0


def bound_recovered_mask(
    recovered: np.ndarray, seed: np.ndarray, others: list[np.ndarray],
) -> tuple[np.ndarray | None, str | None]:
    """(bounded recovered mask or None, flag)."""
    import cv2

    seed = seed.astype(bool)
    if not np.any(seed):
        return None, None
    recovered = recovered.astype(bool)
    near_others = np.zeros_like(recovered)
    for other in others:
        if other is not None and other.shape == recovered.shape:
            near_others |= cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    near_others &= ~seed
    edge = np.zeros_like(recovered)
    edge[:2], edge[-2:], edge[:, :2], edge[:, -2:] = True, True, True, True
    if not np.any(recovered & near_others) and not np.any(recovered & edge & ~seed):
        return recovered, None
    bounded = recovered.copy()
    for other in others:
        if other is not None and other.shape == bounded.shape:
            # Another detection's pixels, and a thin band around them.
            bounded &= ~(cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) | seed
    ys, xs = np.nonzero(seed)
    height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    grow_y, grow_x = int(MAX_BOX_GROWTH * height), int(MAX_BOX_GROWTH * width)
    window = np.zeros_like(bounded)
    window[max(0, ys.min() - grow_y):ys.max() + grow_y + 1,
           max(0, xs.min() - grow_x):xs.max() + grow_x + 1] = True
    trimmed = bounded & window
    count, labels = cv2.connectedComponents(trimmed.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[trimmed & seed])
    keep = keep[keep > 0]
    trimmed = np.isin(labels, keep) & trimmed
    flag = None
    if int(np.count_nonzero(trimmed)) < int(np.count_nonzero(recovered)):
        flag = "recovery_bounded_to_this_object"
    if int(np.count_nonzero(trimmed)) > MAX_AREA_GROWTH * int(np.count_nonzero(seed)):
        return None, "recovery_refused_spreads_beyond_object"
    return trimmed, flag

