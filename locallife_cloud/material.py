"""Zero-shot material classification for detected waste objects.

No labeled photos of plastic/paper/cardboard/etc. bags exist for this project,
so training a conventional classifier is not possible yet. Instead this module
uses a CLIP-style vision-language model in zero-shot mode: it compares the
detected object's crop against a bank of short text prompts describing common
waste materials and picks the closest match. This needs no training data, runs
entirely locally (no cloud call), and its accuracy can be improved later either
by tuning the prompts below or, once labeled photos exist, by swapping this
class for a small trained classifier without touching the rest of the
pipeline (only `classify()`'s return contract matters to callers).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import AppConfig


LOGGER = logging.getLogger(__name__)

# Every label below must also appear as a key in DEFAULT_MATERIAL_PROMPTS.
# "polythene bag" and "plastic" are kept separate on purpose: a thin,
# flexible polythene/plastic carrier bag looks visually very different from
# a rigid plastic item (a bottle, a crate, hard packaging), and waste-sorting
# in practice treats them differently even though both are "plastic" by
# chemistry. "food or organic waste" is likewise its own category rather than
# folded into "mixed or general waste" because distinguishing wet/organic
# waste from dry recyclables is one of the explicit goals of this project.
DEFAULT_MATERIAL_LABELS: tuple[str, ...] = (
    "plastic",
    "polythene bag",
    "paper",
    "cardboard",
    "fabric or textile",
    "metal",
    "food or organic waste",
    "mixed or general waste",
)

# Several phrasings per label improve zero-shot robustness: CLIP's similarity
# score for a single prompt is noisy, but averaging several related prompts
# for the same material behaves like a small ensemble.
DEFAULT_MATERIAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "plastic": (
        "a photo of a rigid plastic container",
        "a hard plastic waste item",
        "a plastic bottle or crate",
        "molded plastic packaging waste",
    ),
    "polythene bag": (
        "a photo of a thin polythene bag",
        "a lightweight plastic shopping bag",
        "a clear plastic wrapping bag",
        "a thin plastic carrier bag",
        "a black plastic bin liner",
        "a shiny plastic waste sack",
    ),
    "paper": (
        "a photo of a paper bag",
        "a brown paper sack",
        "crumpled paper waste",
        "a torn paper shopping bag",
    ),
    "cardboard": (
        "a photo of a cardboard box",
        "a flattened cardboard carton",
        "corrugated cardboard packaging",
        "a brown cardboard shipping box",
    ),
    "fabric or textile": (
        "a photo of a cloth laundry bag",
        "a fabric tote bag",
        "a woven textile sack",
        "a striped cotton laundry basket bag",
    ),
    "metal": (
        "a photo of a metal can",
        "a crushed aluminum can",
        "a metal container",
        "shiny metal scrap",
    ),
    "food or organic waste": (
        "a photo of food waste",
        "organic kitchen waste in a bag",
        "vegetable and fruit peels waste",
        "wet food scraps in a bag",
        "spoiled food and leftovers",
    ),
    "mixed or general waste": (
        "a photo of a mixed general-waste bag",
        "an opaque black garbage bag with unknown contents",
        "an overstuffed bag of assorted trash",
    ),
}


def _feature_tensor(output: Any) -> Any:
    """Unwrap the embedding tensor from a CLIPModel feature call.

    `CLIPModel.get_text_features()`/`get_image_features()` returned a plain
    tensor in older `transformers` releases. Newer releases (the
    `@can_return_tuple` API, seen from transformers 5.x onward) return a
    `BaseModelOutputWithPooling` object instead, whose `.pooler_output` holds
    the same projected embedding a plain-tensor return used to be. Handling
    both shapes here keeps this classifier working across the
    `transformers>=4.45,<6` range this project installs, instead of crashing
    with an AttributeError the moment `.norm()` is called on the wrong type.
    """
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _crop_object(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), frame_bgr.shape[1] - 1))
    y1 = max(0, min(int(y1), frame_bgr.shape[0] - 1))
    x2 = max(x1 + 1, min(int(x2), frame_bgr.shape[1]))
    y2 = max(y1 + 1, min(int(y2), frame_bgr.shape[0]))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    if mask is not None and mask.shape == frame_bgr.shape[:2]:
        local_mask = mask[y1:y2, x1:x2]
        if local_mask.shape == crop.shape[:2] and np.any(local_mask):
            # Neutral mid-grey outside the object mask keeps background walls,
            # floors, and furniture from biasing the material guess.
            background = np.full_like(crop, 128)
            crop = np.where(local_mask[:, :, None], crop, background)
    return crop


class MaterialClassifier:
    """CLIP zero-shot crop -> material label. No training data required."""

    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or "cpu"
        self.enabled = config.enable_material_classification
        self.model: Any | None = None
        self.processor: Any | None = None
        self._text_features: Any | None = None
        self._prompt_labels: list[str] = []
        self.labels: list[str] = list(config.material_labels) or list(DEFAULT_MATERIAL_LABELS)
        self.runtime: dict[str, Any] = {
            "enabled": self.enabled,
            "model": config.material_model if self.enabled else None,
            "labels": self.labels,
        }

    def load(self) -> None:
        if not self.enabled or self.model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            LOGGER.warning(
                "Material classification disabled: transformers/torch is missing (%s). "
                "Install requirements-local.txt to enable it.",
                exc,
            )
            self.enabled = False
            self.runtime["enabled"] = False
            self.runtime["error"] = str(exc)
            return

        try:
            self.model = CLIPModel.from_pretrained(self.config.material_model)
            self.processor = CLIPProcessor.from_pretrained(self.config.material_model)
            self.model.to(self.device).eval()

            prompts: list[str] = []
            prompt_labels: list[str] = []
            for label in self.labels:
                phrases = DEFAULT_MATERIAL_PROMPTS.get(label, (f"a photo of {label} waste",))
                for phrase in phrases:
                    prompts.append(phrase)
                    prompt_labels.append(label)
            self._prompt_labels = prompt_labels

            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_text_features(**inputs))
            self._text_features = features / features.norm(dim=-1, keepdim=True)
            LOGGER.info(
                "Material classifier ready: %s prompts across %s labels on %s",
                len(prompts), len(self.labels), self.device,
            )
        except Exception as exc:  # pragma: no cover - defensive, e.g. offline first run
            LOGGER.warning("Material classifier failed to load (%s); material detection disabled", exc)
            self.enabled = False
            self.model = None
            self.runtime["enabled"] = False
            self.runtime["error"] = str(exc)

    def classify(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        box: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        """Return (material_label, confidence). ("unknown", 0.0) if unavailable."""
        if not self.enabled:
            return "unknown", 0.0
        if self.model is None:
            self.load()
        if self.model is None or self._text_features is None:
            return "unknown", 0.0

        crop = _crop_object(frame_bgr, mask, box)
        if crop is None:
            return "unknown", 0.0

        try:
            import torch
            from PIL import Image

            image = Image.fromarray(crop[:, :, ::-1])
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_features = _feature_tensor(self.model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ self._text_features.T).squeeze(0)
            probabilities = torch.softmax(similarity * 100.0, dim=0)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Material classification failed on one crop (%s)", exc)
            return "unknown", 0.0

        label_scores: dict[str, float] = {}
        for index, prompt_label in enumerate(self._prompt_labels):
            label_scores[prompt_label] = label_scores.get(prompt_label, 0.0) + float(probabilities[index])
        best_label = max(label_scores, key=label_scores.get)
        return best_label, float(label_scores[best_label])
