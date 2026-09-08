"""Regression guard for a real crash reported on the user's machine:

    Material classifier failed to load ('BaseModelOutputWithPooling' object
    has no attribute 'norm'); material detection disabled

Root cause: in `transformers` 5.x, `CLIPModel.get_text_features()` and
`get_image_features()` stopped returning a plain tensor and instead return a
`BaseModelOutputWithPooling` object whose `.pooler_output` holds the actual
projected embedding (confirmed by reading the installed 5.16.1 source
directly). `material.py`'s `_feature_tensor()` helper unwraps both the old
(plain tensor) and new (`.pooler_output`) shapes. These tests simulate both
shapes without needing real network access or model weights, by mocking
`transformers.CLIPModel`/`CLIPProcessor` directly.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from locallife_cloud.config import AppConfig
from locallife_cloud.material import MaterialClassifier, _feature_tensor


class FakePoolingOutput:
    """Stands in for transformers 5.x's BaseModelOutputWithPooling: it is
    NOT a tensor and deliberately has no .norm(), so a caller that forgets
    to unwrap .pooler_output crashes exactly like the reported bug."""

    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


class FeatureTensorUnwrapTests(unittest.TestCase):
    def test_unwraps_new_transformers_pooling_output(self) -> None:
        tensor = torch.randn(2, 8)
        wrapped = FakePoolingOutput(tensor)
        result = _feature_tensor(wrapped)
        self.assertTrue(torch.equal(result, tensor))

    def test_passes_through_plain_tensor_from_older_transformers(self) -> None:
        tensor = torch.randn(2, 8)
        result = _feature_tensor(tensor)
        self.assertTrue(torch.equal(result, tensor))

    def test_unwraps_a_tuple_return(self) -> None:
        tensor = torch.randn(2, 8)
        result = _feature_tensor((tensor, "unused"))
        self.assertTrue(torch.equal(result, tensor))


def _make_fake_clip_model(embed_dim: int = 8):
    """A minimal stand-in for CLIPModel that reproduces the transformers 5.x
    contract: get_text_features/get_image_features return an object with
    .pooler_output, not a plain tensor."""

    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    def fake_get_text_features(**inputs):
        batch = inputs["input_ids"].shape[0]
        torch.manual_seed(0)
        return FakePoolingOutput(torch.randn(batch, embed_dim))

    def fake_get_image_features(**inputs):
        batch = inputs["pixel_values"].shape[0]
        torch.manual_seed(1)
        return FakePoolingOutput(torch.randn(batch, embed_dim))

    model.get_text_features.side_effect = fake_get_text_features
    model.get_image_features.side_effect = fake_get_image_features
    return model


def _make_fake_processor():
    processor = MagicMock()

    def fake_call(*, text=None, images=None, return_tensors=None, padding=None):
        if text is not None:
            return {"input_ids": torch.zeros((len(text), 4), dtype=torch.long)}
        return {"pixel_values": torch.zeros((1, 3, 4, 4))}

    processor.side_effect = fake_call
    return processor


class MaterialClassifierAgainstNewTransformersTests(unittest.TestCase):
    """`transformers` ships as a `_LazyModule`, whose custom attribute
    resolution ignores plain `unittest.mock.patch("transformers.CLIPModel")`
    (verified directly against the installed 5.16.1 package: patching the
    module attribute does not change what `from transformers import
    CLIPModel` resolves to). Swapping in a throwaway fake module via
    `sys.modules` sidesteps that and reliably exercises `material.py`'s real
    `load()`/`classify()` code paths end to end against the exact response
    shape (`.pooler_output`, no `.norm()`) transformers 5.x actually returns.
    """

    def test_load_and_classify_do_not_crash_on_the_pooling_output_shape(self) -> None:
        config = AppConfig(enable_material_classification=True, material_model="fake/does-not-download")
        classifier = MaterialClassifier(config)

        fake_model = _make_fake_clip_model()
        fake_processor = _make_fake_processor()
        clip_model_cls = MagicMock(from_pretrained=MagicMock(return_value=fake_model))
        clip_processor_cls = MagicMock(from_pretrained=MagicMock(return_value=fake_processor))
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.CLIPModel = clip_model_cls
        fake_transformers.CLIPProcessor = clip_processor_cls

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            classifier.load()

        self.assertTrue(classifier.enabled)
        self.assertIsNotNone(classifier._text_features)

        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            label, confidence = classifier.classify(frame, None, (0, 0, 10, 10))
        self.assertIn(label, classifier.labels)
        self.assertIsInstance(confidence, float)


if __name__ == "__main__":
    unittest.main()
