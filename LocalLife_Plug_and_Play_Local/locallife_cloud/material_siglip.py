"""SigLIP 2 as the material classifier, beside the CLIP one it may replace.

The material classifier has been `openai/clip-vit-base-patch32`: the smallest
CLIP, from 2021. SigLIP 2 is the same kind of model -- an image encoder and a
text encoder in one embedding space, used zero-shot against the same prompts --
trained later, with a sigmoid loss, on more and better data. It is a like-for-
like swap: nothing downstream of `classify()` changes.

It is not the default, and that is deliberate. It has not been run against the
real bin: this environment has neither PyTorch nor access to the model hub, and
the transformers issue tracker carries a report of SigLIP 2 producing
unexpectedly low scores in some versions (huggingface/transformers#43994). The
current CLIP classifier is producing plausible readings on the field
screenshots. Replacing a working component with an unmeasured one on the
strength of a newer release date would be the wrong way round for a thesis.
`scripts/benchmark_classifiers.py` runs both over the same labelled crops from
the real bin; the one that wins there should be the default.

Selecting it is one setting:

    LOCALLIFE_MATERIAL_MODEL=google/siglip2-base-patch16-224

If it cannot load -- no network on first run, a transformers release without
SigLIP 2, a checkpoint name the hub does not know -- the classifier falls back
to CLIP, logs why, and records the fallback in its runtime state. Material
classification never silently disappears because of the upgrade.

Scoring
-------
SigLIP is trained with a sigmoid, so each image-text score is an independent
probability rather than one share of a softmax. Those raw sigmoid values are
routinely small for every prompt, which is a known source of confusion when
they are read as confidences. The classifier here instead takes the model's own
calibrated logits (similarity x learned scale + learned bias) and normalises
them across the candidate prompts, exactly as the CLIP path does, so the
confidence thresholds used elsewhere in the pipeline keep the same meaning
whichever model is loaded.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from .config import AppConfig
from .material import (
    DEFAULT_MATERIAL_PROMPTS,
    MaterialClassifier,
    _crop_object,
    _feature_tensor,
)

LOGGER = logging.getLogger(__name__)

CLIP_FALLBACK_MODEL = "openai/clip-vit-base-patch32"
SIGLIP_DEFAULT_MODEL = "google/siglip2-base-patch16-224"
# SigLIP's text tower was trained on fixed-length, padded sequences, and gives
# poor embeddings for text padded any other way.
SIGLIP_TEXT_LENGTH = 64


def is_siglip(model_name: str | None) -> bool:
    return "siglip" in str(model_name or "").lower()


def aggregate_label_scores(
    logits: np.ndarray, prompt_labels: list[str],
) -> tuple[str, float, dict[str, float]]:
    """Softmax over prompts, summed per label: the same shape the CLIP path uses.

    Kept free of torch so it can be tested without a model, and so the two
    classifiers provably turn scores into labels the same way.
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size != len(prompt_labels):
        return "unknown", 0.0, {}
    shifted = np.exp(values - values.max())
    probabilities = shifted / shifted.sum()
    scores: dict[str, float] = {}
    for label, probability in zip(prompt_labels, probabilities):
        scores[label] = scores.get(label, 0.0) + float(probability)
    best = max(scores, key=scores.get)
    return best, float(scores[best]), scores


class SiglipMaterialClassifier(MaterialClassifier):
    """SigLIP 2 zero-shot crop -> material label, falling back to CLIP."""

    def __init__(self, config: AppConfig, device: str | None = None) -> None:
        super().__init__(config, device)
        self._fallback: MaterialClassifier | None = None
        self._logit_scale = 1.0
        self._logit_bias = 0.0
        self.runtime["family"] = "siglip"

    def _use_fallback(self, reason: str) -> None:
        LOGGER.warning(
            "SigLIP material classifier unavailable (%s); falling back to %s",
            reason, CLIP_FALLBACK_MODEL,
        )
        self.model = None
        self._fallback = MaterialClassifier(
            replace(self.config, material_model=CLIP_FALLBACK_MODEL), self.device,
        )
        self.runtime.update({
            "fallback_model": CLIP_FALLBACK_MODEL,
            "fallback_reason": reason,
            "model": CLIP_FALLBACK_MODEL,
            "family": "clip-fallback",
        })

    def load(self) -> None:
        if not self.enabled or self.model is not None or self._fallback is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            # Without torch there is no CLIP either; the base class reports it.
            super().load()
            self.runtime["error"] = str(exc)
            return
        try:
            self.model = AutoModel.from_pretrained(self.config.material_model)
            self.processor = AutoProcessor.from_pretrained(self.config.material_model)
            self.model.to(self.device).eval()

            prompts: list[str] = []
            prompt_labels: list[str] = []
            for label in self.labels:
                phrases = DEFAULT_MATERIAL_PROMPTS.get(label, (f"a photo of {label} waste",))
                for phrase in phrases:
                    # SigLIP was trained on lower-cased text.
                    prompts.append(phrase.lower())
                    prompt_labels.append(label)
            self._prompt_labels = prompt_labels

            inputs = self.processor(
                text=prompts, return_tensors="pt",
                padding="max_length", max_length=SIGLIP_TEXT_LENGTH, truncation=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_text_features(**inputs))
            self._text_features = features / features.norm(dim=-1, keepdim=True)
            scale = getattr(self.model, "logit_scale", None)
            bias = getattr(self.model, "logit_bias", None)
            self._logit_scale = float(scale.exp().item()) if scale is not None else 10.0
            self._logit_bias = float(bias.item()) if bias is not None else 0.0
            self.runtime.update({"logit_scale": self._logit_scale, "logit_bias": self._logit_bias})
            LOGGER.info(
                "SigLIP material classifier ready: %s prompts across %s labels on %s",
                len(prompts), len(self.labels), self.device,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure falls back
            self._use_fallback(str(exc))

    def classify(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        box: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        if not self.enabled:
            return "unknown", 0.0
        if self.model is None and self._fallback is None:
            self.load()
        if self._fallback is not None:
            return self._fallback.classify(frame_bgr, mask, box)
        if self.model is None or self._text_features is None:
            return "unknown", 0.0
        crop = _crop_object(frame_bgr, mask, box)
        if crop is None:
            return "unknown", 0.0
        try:
            import torch
            from PIL import Image

            inputs = self.processor(images=Image.fromarray(crop[:, :, ::-1]), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_features = _feature_tensor(self.model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ self._text_features.T).squeeze(0)
            logits = (similarity * self._logit_scale + self._logit_bias).detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SigLIP material classification failed on one crop (%s)", exc)
            return "unknown", 0.0
        label, confidence, _ = aggregate_label_scores(logits, self._prompt_labels)
        return label, confidence


def create_material_classifier(config: AppConfig, device: str | None = None) -> MaterialClassifier:
    """The material classifier the configuration asks for."""
    if is_siglip(config.material_model):
        return SiglipMaterialClassifier(config, device)
    return MaterialClassifier(config, device)
