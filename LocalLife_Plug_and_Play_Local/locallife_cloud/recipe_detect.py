"""Object detection/segmentation, per the dual-camera recipe's Step 2.

The recipe asks for a generic YOLOv8n-seg or YOLOv11n-seg model on both
camera color images -- deliberately NOT this project's existing YOLOE
detector with its curated waste-bag/box text-prompt vocabulary
(`config.py`'s `DEFAULT_PROMPTS`, `pipeline.py`'s detector setup). That
existing curated vocabulary was built over several rounds specifically to
suppress false detections (see the project changelog's rounds 2-3), and
switching to a generic open-vocabulary segmentation model removes that
safety net -- this module implements the recipe exactly as specified, but
callers should be aware detection false positives are more likely here than
with the existing YOLOE-based `pipeline.py` path this project already ships.

Per the recipe: run segmentation on both camera color images, then pick one
best detection -- the highest confidence, and among close-confidence
candidates the largest mask -- rather than trying to track/fuse multiple
objects (the existing project's tracker/ledger already does that; this
module deliberately stays single-object-per-call, matching the recipe's own
"select the best detection" step).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_RECIPE_DETECTOR_MODEL = "yolov8n-seg.pt"

# A box's mask fills most of its own bounding rectangle (it has straight
# edges); a bag/sack is lumpier and typically fills noticeably less of its
# bounding box even when fully upright. This is a simple, explainable
# heuristic, not a learned classifier -- consistent with the recipe's own
# "keep it simple" instruction -- and only decides which volume method
# `pointcloud_volume.py` uses (oriented-bounding-box vs. heightmap), not
# anything reported to the user directly.
_BOX_FILL_RATIO_THRESHOLD = 0.75


@dataclass(slots=True)
class RecipeDetection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray
    object_type: str  # "box" or "bag"
    fill_ratio: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box_xyxy": list(self.box),
            "object_type": self.object_type,
            "fill_ratio": round(float(self.fill_ratio), 4),
        }


def classify_object_type(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, float]:
    """Box-vs-bag shape heuristic: mask fill ratio within its own bounding box."""
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    fill_ratio = float(np.count_nonzero(mask)) / box_area
    object_type = "box" if fill_ratio >= _BOX_FILL_RATIO_THRESHOLD else "bag"
    return object_type, fill_ratio


class RecipeDetector:
    """Thin wrapper around a generic Ultralytics YOLOv8n/11n-seg model."""

    def __init__(self, model_name: str = DEFAULT_RECIPE_DETECTOR_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model: Any | None = None
        self.enabled = True
        self.load_error: str | None = None

    def load(self) -> None:
        if self.model is not None or not self.enabled:
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
        except Exception as exc:  # pragma: no cover - defensive, e.g. offline first run
            LOGGER.warning("Recipe detector failed to load %s (%s); detection disabled", self.model_name, exc)
            self.enabled = False
            self.load_error = str(exc)

    def detect(self, image_bgr: np.ndarray, *, confidence_threshold: float = 0.25) -> list[RecipeDetection]:
        """Run segmentation on one color image. Returns every detection found
        (already carrying object_type/fill_ratio); `select_best` picks one."""
        if not self.enabled:
            return []
        if self.model is None:
            self.load()
        if self.model is None:
            return []

        try:
            results = self.model.predict(source=image_bgr, conf=confidence_threshold, verbose=False)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Recipe detector inference failed (%s)", exc)
            return []

        detections: list[RecipeDetection] = []
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            mask_data = result.masks.data.cpu().numpy() if hasattr(result.masks.data, "cpu") else np.asarray(result.masks.data)
            boxes = result.boxes
            names = result.names if hasattr(result, "names") else {}
            for index in range(len(boxes)):
                confidence = float(boxes.conf[index])
                xyxy = boxes.xyxy[index]
                box = tuple(int(round(float(value))) for value in xyxy)
                mask_small = mask_data[index] > 0.5
                mask = _resize_mask_to_image(mask_small, image_bgr.shape[:2])
                if np.count_nonzero(mask) < 20:
                    continue
                class_id = int(boxes.cls[index]) if boxes.cls is not None else -1
                label = str(names.get(class_id, "object")) if isinstance(names, dict) else "object"
                object_type, fill_ratio = classify_object_type(mask, box)
                detections.append(
                    RecipeDetection(
                        label=label,
                        confidence=confidence,
                        box=box,
                        mask=mask,
                        object_type=object_type,
                        fill_ratio=fill_ratio,
                    )
                )
        return detections


def _resize_mask_to_image(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    import cv2

    resized = cv2.resize(
        mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def select_best(
    *detection_lists: list[RecipeDetection],
    target_class: str | None = None,
    area_range: tuple[float, float] = (0.05, 0.80),
    frame_area: int | None = None,
) -> RecipeDetection | None:
    """v3 §4 step 2's selection rule: "if a target class is known (box/
    container/bag), filter by class first. Else pick highest-confidence
    mask, then sanity-check mask area in [5%, 80%] of frame. Reject
    obviously wrong 'largest mask' picks."

    Pools every detection across however many camera detection lists are
    given. When `target_class` is set, only candidates whose `object_type`
    (box/bag -- this module's own shape classification, the closest
    equivalent this generic detector has to a semantic class name) matches
    are considered at all. Otherwise, candidates are ranked by (confidence,
    mask area) descending, but any candidate whose mask covers a fraction of
    `frame_area` outside `area_range` is rejected before ranking -- this is
    what stops an obviously-wrong "whole background" or "tiny speck" mask
    from ever being picked as "the object" in the first place. Returns None
    if nothing usable was detected anywhere.
    """
    pooled: list[RecipeDetection] = [item for detections in detection_lists for item in detections]
    if not pooled:
        return None

    if target_class is not None:
        filtered = [item for item in pooled if item.object_type == target_class]
        if filtered:
            pooled = filtered
        # If nothing matches the requested class, fall through to the
        # generic confidence/area-sanity rule below rather than returning
        # nothing -- a caller that mis-set target_class shouldn't lose a
        # real, visible detection entirely.

    if frame_area:
        sane = [
            item for item in pooled
            if area_range[0] <= (int(np.count_nonzero(item.mask)) / frame_area) <= area_range[1]
        ]
        if sane:
            pooled = sane

    return max(pooled, key=lambda item: (round(item.confidence, 2), int(np.count_nonzero(item.mask))))


def frame_edge_clip_fraction(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> float:
    """v3 edge-case matrix: "Object touches frame edge -> pad ROI; if mask
    clipped >30%, lower confidence." A single segmentation mask alone can't
    reveal how much of an object extends *beyond* the frame (those pixels
    were never captured) -- this approximates "how clipped" a detection is
    from what the mask's own bounding box already shows: the fraction of the
    box's 4 sides that sit flush against the frame boundary (0.0 = box is
    fully interior, 1.0 = box touches all 4 sides). A caller dampens
    `volume_confidence`/flags when this exceeds a threshold (recipe_pipeline.py).
    """
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape
    touches = 0
    if x1 <= 1:
        touches += 1
    if y1 <= 1:
        touches += 1
    if x2 >= frame_w - 1:
        touches += 1
    if y2 >= frame_h - 1:
        touches += 1
    return touches / 4.0
