"""Drop mask pixels whose depth belongs to neither the object nor what is behind it.

Where an object's silhouette meets the floor, stereo matching interpolates
between the two surfaces (RealSense "flying pixels"), and a monocular model
blurs the step over several pixels (Depth Anything V2 edge bleed). Those pixels
back-project to points strung out along their rays, between the object's top
and the floor. Projected onto the floor they smear the footprint away from the
camera: in a ray-traced scene a 2-pixel ring of them turned a 70 mm can into a
281 x 96 mm footprint, while its height stayed right. That is the pattern the
field numbers show -- heights plausible, footprints two to four times too long.

A pixel is dropped when, within a small window, the depth spans a real step
(more than `jump_m`) and this pixel sits strictly inside that step, away from
both the near and the far surface. A pixel on the object or on the floor sits
at one end of the step and is kept, and a smooth surface without a step is
never touched. If the rule would remove too much of a mask, the mask is left
alone and the caller is told -- a filter that eats the object is worse than
the artefact.
"""

from __future__ import annotations

import numpy as np

JUMP_M = 0.03
TOLERANCE = 0.08
MAX_REMOVED_FRACTION = 0.30


def drop_depth_edge_pixels(
    depth_m: np.ndarray | None, mask: np.ndarray | None, *, radius: int = 2,
    jump_m: float = JUMP_M, tolerance: float = TOLERANCE,
    max_removed_fraction: float = MAX_REMOVED_FRACTION,
) -> tuple[np.ndarray | None, dict]:
    """(filtered mask, diagnostics). The input mask is returned unchanged when refused."""
    if depth_m is None or mask is None or depth_m.shape != mask.shape or not np.any(mask):
        return mask, {"applied": False, "reason": "no_depth_or_mask"}
    import cv2

    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth = depth_m.astype(np.float32)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    low = cv2.erode(np.where(valid, depth, np.float32(1e3)), kernel)
    high = cv2.dilate(np.where(valid, depth, np.float32(0.0)), kernel)
    span = high - low
    inside_step = (
        valid & (span > jump_m)
        & (depth - low > tolerance * span) & (high - depth > tolerance * span)
    )
    edge = mask.astype(bool) & inside_step
    total = int(np.count_nonzero(mask))
    removed = int(np.count_nonzero(edge))
    fraction = removed / max(total, 1)
    diagnostics = {"removed_pixels": removed, "removed_fraction": round(fraction, 4), "radius_px": radius}
    if fraction > max_removed_fraction:
        return mask, {**diagnostics, "applied": False, "reason": "would_remove_too_much_of_the_mask"}
    return mask.astype(bool) & ~edge, {**diagnostics, "applied": removed > 0}
