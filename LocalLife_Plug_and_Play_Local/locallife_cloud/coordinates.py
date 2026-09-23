"""One coordinate system per camera frame, and the checks that prove it.

Every Logitech measurement is made in the *original* C920 pixel grid: the mask,
the depth map and the intrinsics must all describe that same grid. Two things
break it silently and both inflate dimensions:

* a segmentation mask produced at the detector's square input is stretched back
  instead of having its letterbox padding removed first, which widens the
  object along the padded axis;
* a scale is applied twice (once by the model wrapper, once by us).

`restore_mask` undoes a letterbox properly, and `frame_consistency` returns the
problems rather than guessing, so a caller can refuse to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """How a frame was fitted into a square/padded model input."""

    scale: float
    pad_x: int
    pad_y: int
    source_shape: tuple[int, int]   # (height, width) of the original frame
    target_shape: tuple[int, int]   # (height, width) of the model input

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": round(self.scale, 6), "pad_x": self.pad_x, "pad_y": self.pad_y,
            "source_shape": list(self.source_shape), "target_shape": list(self.target_shape),
        }


def letterbox_params(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Letterbox:
    """Aspect-preserving fit of `source_shape` into `target_shape`, centred."""
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width, scaled_height = round(source_width * scale), round(source_height * scale)
    return Letterbox(
        scale=scale,
        pad_x=max(0, (target_width - scaled_width) // 2),
        pad_y=max(0, (target_height - scaled_height) // 2),
        source_shape=(source_height, source_width),
        target_shape=(target_height, target_width),
    )


def restore_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """A model-space mask back in original frame pixels, padding removed first."""
    import cv2

    frame_height, frame_width = frame_shape[:2]
    if mask.shape[:2] == (frame_height, frame_width):
        return mask.astype(bool)
    box = letterbox_params((frame_height, frame_width), mask.shape[:2])
    scaled_height = round(frame_height * box.scale)
    scaled_width = round(frame_width * box.scale)
    cropped = mask[box.pad_y:box.pad_y + scaled_height, box.pad_x:box.pad_x + scaled_width]
    if cropped.size == 0:
        cropped = mask
    return cv2.resize(
        cropped.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def frame_consistency(
    frame: np.ndarray,
    mask: np.ndarray | None,
    depth: np.ndarray | None,
    intrinsics: Any,
    *,
    region: np.ndarray | None = None,
    foreign_intrinsics: Any = None,
) -> list[str]:
    """Everything that would make a metric measurement meaningless, named.

    An empty list means mask, depth and intrinsics all describe this frame.
    """
    problems: list[str] = []
    shape = frame.shape[:2]
    if mask is not None and mask.shape[:2] != shape:
        problems.append(f"mask_shape_{mask.shape[:2]}_not_frame_{shape}")
    if depth is not None and depth.shape[:2] != shape:
        problems.append(f"depth_shape_{depth.shape[:2]}_not_frame_{shape}")
    if intrinsics is None:
        problems.append("missing_intrinsics")
    else:
        if getattr(intrinsics, "width", 0) and (int(intrinsics.width), int(intrinsics.height)) != (shape[1], shape[0]):
            problems.append(
                f"intrinsics_for_{int(intrinsics.width)}x{int(intrinsics.height)}_not_{shape[1]}x{shape[0]}"
            )
        if foreign_intrinsics is not None and _same_camera(intrinsics, foreign_intrinsics):
            # The other camera's matrix would silently produce the other
            # camera's geometry from this camera's pixels.
            problems.append("intrinsics_belong_to_the_other_camera")
    if mask is not None and region is not None and np.any(mask.astype(bool) & ~region.astype(bool)):
        problems.append("mask_outside_calibrated_roi")
    return problems


def _same_camera(left: Any, right: Any) -> bool:
    return all(
        abs(float(getattr(left, name)) - float(getattr(right, name))) < 1e-6
        for name in ("fx", "fy", "ppx", "ppy")
    )


def clip_to_region(mask: np.ndarray, region: np.ndarray | None) -> np.ndarray:
    """A measurement mask never extends past the calibrated bin region."""
    return mask.astype(bool) if region is None else (mask.astype(bool) & region.astype(bool))
