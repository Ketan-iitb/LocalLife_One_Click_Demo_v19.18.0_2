"""Material classification, per the dual-camera recipe v3 §7's routing
logic (a two-level cascade, superseding the CLIP-only version this module
originally shipped with for the v1/v2-era text recipe):

- **Level-1 (default)**: CLIP zero-shot on the cropped Logitech image over
  the fixed seven-class taxonomy (Plastic, Fabric, Metal, Cardboard, Paper,
  Rubber, Other), scored by top logit plus softmax margin over the runner-up.
- **Level-2 fallback**: if CLIP's confidence is below
  `material.clip_confidence_floor` (0.6 default), or CLIP called "Plastic"
  with a thin margin (v3's own documented CLIP failure mode: thin/
  translucent plastic reads as ambiguous with Paper/Other), classification
  switches to a fine-tuned CNN (MobileNetV3-Small / EfficientNet-B0, same
  seven classes). If the fallback *also* reports below
  `material.fallback_confidence_floor` (0.55), the result is
  `material="Other"` with `ambiguous_material=True` -- v3's own "do not
  guess" instruction -- rather than presenting either model's low-confidence
  guess as an answer.
- Every result records which model actually produced it (`material_model`:
  `"clip"` or the configured fallback name) in the recipe's output schema.

`RecipeMaterialClassifier` (Level-1, CLIP) reuses the project's existing
CLIP-loading machinery (`material.py`'s `_feature_tensor()` unwrap helper and
`_crop_object()` background-neutralizing crop, both already hardened against
`transformers` API-shape differences documented there) rather than
duplicating it, but keeps its own class list and prompts -- the existing
`MaterialClassifier` in `material.py` uses a different taxonomy tuned for
this project's waste-stream dashboard and continues to power that dashboard
unchanged.

`RecipeMaterialFallbackClassifier` (Level-2) is a torchvision
MobileNetV3-Small backbone with a small linear head over the same seven
classes. **No fine-tuned checkpoint ships with this project** -- there is no
labeled training set available in this build environment, and v3's own
instruction is "do not guess," which applies as much to fabricating a
fine-tuned model's weights as to fabricating a confidence score. Point
`material.fallback_checkpoint_path` (recipe_config.yaml) at a real
checkpoint once one exists (trained per v3 §13's validation plan) and the
cascade activates automatically; until then, `classify_material_cascade()`
degrades honestly (see its own docstring) rather than silently only ever
running Level-1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import AppConfig
from .material import _crop_object, _feature_tensor

LOGGER = logging.getLogger(__name__)

RECIPE_MATERIAL_CLASSES: tuple[str, ...] = (
    "Plastic",
    "Fabric",
    "Metal",
    "Cardboard",
    "Paper",
    "Rubber",
    "Other",
)

RECIPE_MATERIAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "Plastic": (
        "a photo of a rigid plastic container",
        "a plastic bottle",
        "a thin polythene plastic bag",
        "molded plastic packaging",
    ),
    "Fabric": (
        "a photo of a cloth fabric bag",
        "a woven textile sack",
        "a fabric tote bag",
        "a cotton or polyester cloth item",
    ),
    "Metal": (
        "a photo of a metal can",
        "a metal container",
        "shiny aluminum or steel scrap",
        "a crushed metal can",
    ),
    "Cardboard": (
        "a photo of a cardboard box",
        "corrugated cardboard packaging",
        "a brown cardboard carton",
        "a flattened cardboard box",
    ),
    "Paper": (
        "a photo of a paper bag",
        "crumpled paper waste",
        "a paper sheet or wrapping",
        "a torn paper shopping bag",
    ),
    "Rubber": (
        "a photo of a rubber item",
        "a rubber tire or tube",
        "a black rubber mat or seal",
        "molded rubber material",
    ),
    "Other": (
        "a photo of an unidentifiable waste item",
        "mixed or unknown material waste",
        "an object of unclear material",
    ),
}


class RecipeMaterialClassifier:
    """CLIP zero-shot crop -> one of the recipe's seven material classes."""

    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or "cpu"
        self.enabled = config.enable_material_classification
        self.model: Any | None = None
        self.processor: Any | None = None
        self._text_features: Any | None = None
        self._prompt_labels: list[str] = []
        self.labels: list[str] = list(RECIPE_MATERIAL_CLASSES)
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
                "Recipe material classification disabled: transformers/torch is missing (%s).",
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
                for phrase in RECIPE_MATERIAL_PROMPTS.get(label, (f"a photo of {label.lower()} waste",)):
                    prompts.append(phrase)
                    prompt_labels.append(label)
            self._prompt_labels = prompt_labels

            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_text_features(**inputs))
            self._text_features = features / features.norm(dim=-1, keepdim=True)
            LOGGER.info(
                "Recipe material classifier ready: %s prompts across %s classes on %s",
                len(prompts), len(self.labels), self.device,
            )
        except Exception as exc:  # pragma: no cover - defensive, e.g. offline first run
            LOGGER.warning("Recipe material classifier failed to load (%s); disabled", exc)
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
        """Return (material_label, confidence). ("Other", 0.0) if unavailable.
        Kept for backward compatibility; `classify_with_margin` is what the
        v3 cascade (`classify_material_cascade`) actually calls."""
        label, confidence, _margin = self.classify_with_margin(frame_bgr, mask, box)
        return label, confidence

    def classify_with_margin(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        box: tuple[int, int, int, int],
    ) -> tuple[str, float, float]:
        """Return (material_label, confidence, margin). `margin` is the
        softmax gap between the top-1 and runner-up class scores (v3 §7
        step 1: "get top logits + softmax margin") -- 0.0 when unavailable
        or only one class was ever scoreable."""
        if not self.enabled:
            return "Other", 0.0, 0.0
        if self.model is None:
            self.load()
        if self.model is None or self._text_features is None:
            return "Other", 0.0, 0.0

        crop = _crop_object(frame_bgr, mask, box)
        if crop is None:
            return "Other", 0.0, 0.0

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
            LOGGER.warning("Recipe material classification failed on one crop (%s)", exc)
            return "Other", 0.0, 0.0

        label_scores: dict[str, float] = {}
        for index, prompt_label in enumerate(self._prompt_labels):
            label_scores[prompt_label] = label_scores.get(prompt_label, 0.0) + float(probabilities[index])
        ranked = sorted(label_scores.values(), reverse=True)
        best_label = max(label_scores, key=label_scores.get)
        margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])
        return best_label, float(label_scores[best_label]), margin


class RecipeMaterialFallbackClassifier:
    """v3 §7 Level-2: a fine-tuned MobileNetV3-Small (or EfficientNet-B0)
    over the same seven classes, torchvision-backed. See this module's own
    top-level docstring for why no checkpoint ships with this project --
    `enabled` stays False, and every `classify_with_margin` call returns a
    clearly-unavailable result, until `checkpoint_path` points at a real
    trained checkpoint (a plain `state_dict` for the head-replaced backbone
    this class builds in `load()`).
    """

    def __init__(self, checkpoint_path: str | None, backbone: str = "mobilenetv3_small", device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.backbone_name = backbone
        self.device = device or "cpu"
        self.model: Any | None = None
        self.enabled = bool(checkpoint_path)
        self.labels: list[str] = list(RECIPE_MATERIAL_CLASSES)
        self.load_error: str | None = None

    def load(self) -> None:
        if not self.enabled or self.model is not None:
            return
        import os

        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            LOGGER.info(
                "Recipe material fallback (%s) disabled: no checkpoint at %s -- "
                "Level-2 routing degrades to Level-1's own result instead (see recipe_material.py docstring)",
                self.backbone_name, self.checkpoint_path,
            )
            self.enabled = False
            self.load_error = "checkpoint not found"
            return
        try:
            import torch
            import torchvision

            if self.backbone_name == "efficientnet_b0":
                backbone = torchvision.models.efficientnet_b0(weights=None)
                backbone.classifier[-1] = torch.nn.Linear(backbone.classifier[-1].in_features, len(self.labels))
            else:
                backbone = torchvision.models.mobilenet_v3_small(weights=None)
                backbone.classifier[-1] = torch.nn.Linear(backbone.classifier[-1].in_features, len(self.labels))
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            backbone.load_state_dict(state_dict)
            backbone.to(self.device).eval()
            self.model = backbone
            LOGGER.info("Recipe material fallback classifier ready: %s from %s", self.backbone_name, self.checkpoint_path)
        except Exception as exc:  # pragma: no cover - defensive, needs a real checkpoint to exercise
            LOGGER.warning("Recipe material fallback classifier failed to load (%s); disabled", exc)
            self.enabled = False
            self.model = None
            self.load_error = str(exc)

    def classify_with_margin(
        self, frame_bgr: np.ndarray, mask: np.ndarray | None, box: tuple[int, int, int, int]
    ) -> tuple[str, float, float]:
        if not self.enabled:
            return "Other", 0.0, 0.0
        if self.model is None:
            self.load()
        if self.model is None:
            return "Other", 0.0, 0.0

        crop = _crop_object(frame_bgr, mask, box)
        if crop is None:
            return "Other", 0.0, 0.0
        try:
            import torch
            import torchvision.transforms.functional as tvf
            from PIL import Image

            image = Image.fromarray(crop[:, :, ::-1]).resize((224, 224))
            tensor = tvf.to_tensor(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(tensor).squeeze(0)
                probabilities = torch.softmax(logits, dim=0)
            ranked, _indices = torch.sort(probabilities, descending=True)
            best_index = int(torch.argmax(probabilities))
            best_label = self.labels[best_index]
            confidence = float(probabilities[best_index])
            margin = float(ranked[0] - ranked[1]) if ranked.shape[0] > 1 else confidence
            return best_label, confidence, margin
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Recipe material fallback classification failed on one crop (%s)", exc)
            return "Other", 0.0, 0.0


def classify_material_cascade(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    *,
    clip_classifier: "RecipeMaterialClassifier",
    fallback_classifier: "RecipeMaterialFallbackClassifier | None" = None,
    clip_confidence_floor: float = 0.6,
    fallback_confidence_floor: float = 0.55,
    plastic_thin_margin: float = 0.10,
) -> tuple[str, float, str, bool]:
    """v3 §7's full two-level routing logic. Returns
    `(material_label, material_confidence, material_model, ambiguous_material)`.

    Level-1 (CLIP) always runs first. It is accepted as-is unless its own
    confidence is below `clip_confidence_floor`, or it called "Plastic"
    with a margin below `plastic_thin_margin` (v3's documented CLIP failure
    mode for thin/translucent plastic). In either case, Level-2 (the
    fallback classifier) is tried:

    - If the fallback is available and scores >= `fallback_confidence_floor`,
      its own result is used (`material_model` names the fallback backbone).
    - If the fallback is available but *also* scores too low, the result is
      `("Other", <its own confidence>, <fallback name>, True)` -- v3's own
      "do not guess" instruction.
    - If no fallback classifier is configured/available at all (the common
      case in this build -- see this module's own docstring), there is no
      second opinion to consult. Rather than silently keeping a
      known-too-low CLIP score, this degrades honestly: if CLIP's own
      confidence is already below `fallback_confidence_floor` (i.e. it would
      have failed Level-2 as well), the result is Other + ambiguous;
      otherwise CLIP's label is kept with its own (still low) confidence and
      `material_model="clip"`, so a real-but-uncertain CLIP call is not
      thrown away purely because no fallback model is installed.
    """
    label, confidence, margin = clip_classifier.classify_with_margin(frame_bgr, mask, box)
    thin_plastic = label == "Plastic" and margin < plastic_thin_margin
    needs_fallback = confidence < clip_confidence_floor or thin_plastic

    if not needs_fallback:
        return label, confidence, "clip", False

    if fallback_classifier is not None and fallback_classifier.enabled:
        fallback_label, fallback_confidence, _fallback_margin = fallback_classifier.classify_with_margin(
            frame_bgr, mask, box
        )
        if fallback_confidence < fallback_confidence_floor:
            return "Other", fallback_confidence, fallback_classifier.backbone_name, True
        return fallback_label, fallback_confidence, fallback_classifier.backbone_name, False

    if confidence < fallback_confidence_floor:
        return "Other", confidence, "clip", True
    return label, confidence, "clip", False
