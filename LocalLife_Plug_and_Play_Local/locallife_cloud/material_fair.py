"""Material scores that do not depend on how many prompts a label has.

Both zero-shot classifiers took a softmax over every *prompt* and summed it per
label. The prompt lists differ in length -- "polythene bag" has six phrasings,
"mixed or general waste" three, the rest four or five -- so before looking at
the image at all, "polythene bag" held 6/35 of the probability mass and
"mixed" 3/35. A label won partly for having more sentences written about it.

Here each label's prompts are first averaged into one logit (the mean
similarity to that label's phrasings), and the softmax runs over labels, so
every label starts equal. The result is still a zero-shot *ranking* score --
a relative preference among the listed labels, not a calibrated probability
that the object is made of that material -- and it is reported as such. When
the best two labels are close the answer is "unknown" rather than a coin toss.

material.py is protected and unchanged; this subclasses it.
"""

from __future__ import annotations

import logging

import numpy as np

from . import material as _material
from .material import MaterialClassifier, _crop_object, _feature_tensor

# V51: the visible-material vocabulary needs GLASS (a bottle may be plastic or glass). Registered
# here, through material.py's own label/prompt tables, so material.py itself is unchanged; every
# classifier built afterwards (CLIP and SigLIP, which falls back to this one) scores it.
_material.DEFAULT_MATERIAL_PROMPTS.setdefault("glass", (
    "a photo of a glass bottle", "a clear glass jar", "a green or brown glass bottle",
    "transparent glass container",
))
if "glass" not in _material.DEFAULT_MATERIAL_LABELS:
    _material.DEFAULT_MATERIAL_LABELS = (*_material.DEFAULT_MATERIAL_LABELS, "glass")

LOGGER = logging.getLogger(__name__)

UNKNOWN = "unknown"
# Best label must lead the runner-up by this much of the label-level softmax.
ABSTAIN_MARGIN = 0.10
SCORE_MEANING = "zero-shot ranking among listed labels; not a calibrated probability"


def balanced_label_scores(
    logits: np.ndarray, prompt_labels: list[str],
) -> tuple[str, float, float, dict[str, float]]:
    """(label, score, margin over the runner-up, all label scores), one vote per label."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size != len(prompt_labels):
        return UNKNOWN, 0.0, 0.0, {}
    grouped: dict[str, list[float]] = {}
    for label, value in zip(prompt_labels, values):
        grouped.setdefault(label, []).append(float(value))
    labels = list(grouped)
    means = np.array([np.mean(grouped[label]) for label in labels])
    shifted = np.exp(means - means.max())
    probabilities = shifted / shifted.sum()
    scores = {label: float(p) for label, p in zip(labels, probabilities)}
    order = np.argsort(probabilities)[::-1]
    best = labels[int(order[0])]
    margin = float(probabilities[order[0]] - (probabilities[order[1]] if len(order) > 1 else 0.0))
    if margin < ABSTAIN_MARGIN:
        return UNKNOWN, float(probabilities[order[0]]), margin, scores
    return best, float(probabilities[order[0]]), margin, scores


class BalancedClipMaterialClassifier(MaterialClassifier):
    """The CLIP classifier with per-label (not per-prompt) scoring and abstention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.runtime["scoring"] = "per-label-mean-logit-softmax"
        self.runtime["score_meaning"] = SCORE_MEANING
        self.last_scores: dict[str, float] = {}
        self.last_margin = 0.0

    def classify(self, frame_bgr, mask, box):
        if not self.enabled:
            return UNKNOWN, 0.0
        if self.model is None:
            self.load()
        if self.model is None or self._text_features is None:
            return UNKNOWN, 0.0
        crop = _crop_object(frame_bgr, mask, box)
        if crop is None:
            return UNKNOWN, 0.0
        try:
            import torch
            from PIL import Image

            inputs = self.processor(images=Image.fromarray(crop[:, :, ::-1]), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _feature_tensor(self.model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            logits = ((features @ self._text_features.T).squeeze(0) * 100.0).float().cpu().numpy()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Material classification failed on one crop (%s)", exc)
            return UNKNOWN, 0.0
        label, score, self.last_margin, self.last_scores = balanced_label_scores(
            logits, self._prompt_labels)
        return label, score
