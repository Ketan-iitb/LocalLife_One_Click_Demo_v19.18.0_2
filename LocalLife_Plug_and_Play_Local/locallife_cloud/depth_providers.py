"""One interface for every candidate metric-depth model.

The pipeline runs Depth Anything V2. Whether that is the right choice for this
fixed indoor rig is an empirical question, so every candidate is wrapped the
same way and `scripts/benchmark_depth_models.py` scores them on the same saved
frames. A provider that cannot be imported reports exactly why instead of being
silently skipped -- an unavailable model is a result, not an omission.

Models are loaded once per provider instance, never per frame.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .types import CameraIntrinsics


class DepthProvider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def predict(self, frame_bgr: np.ndarray, intrinsics: CameraIntrinsics | None) -> np.ndarray: ...


@dataclass
class TransformersDepthProvider:
    """Any Depth Anything V2 checkpoint served through transformers."""

    checkpoint: str
    name: str = ""
    device: str = "auto"
    metric: bool = True
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or self.checkpoint.split("/")[-1]

    def available(self) -> tuple[bool, str]:
        for module in ("torch", "transformers"):
            if importlib.util.find_spec(module) is None:
                return False, f"{module} is not installed"
        return True, "ok"

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(device).eval()

    def predict(self, frame_bgr: np.ndarray, intrinsics: CameraIntrinsics | None = None) -> np.ndarray:
        import torch
        from PIL import Image

        if self._model is None:
            self.load()
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self._processor(images=[image], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            prediction = self._model(**inputs).predicted_depth
        resized = torch.nn.functional.interpolate(
            prediction[0][None, None].float(), size=frame_bgr.shape[:2], mode="bicubic", align_corners=False,
        )
        return resized[0, 0].cpu().numpy().astype(np.float32)


@dataclass
class TorchHubDepthProvider:
    """Metric3D-v2 / UniDepth-v2 / Depth Pro, each behind its own optional import.

    These take the calibrated C920 intrinsic matrix, which is the reason to
    benchmark them at all on a fixed rig: they predict metric depth directly
    rather than up to an unknown affine transform.
    """

    name: str
    module: str
    factory: str
    metric: bool = True
    _model: Any = field(default=None, repr=False)

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("torch") is None:
            return False, "torch is not installed"
        if importlib.util.find_spec(self.module) is None:
            return False, f"{self.module} is not installed (pip install it to include this model)"
        return True, "ok"

    def load(self) -> None:
        module = importlib.import_module(self.module)
        self._model = getattr(module, self.factory)()

    def predict(self, frame_bgr: np.ndarray, intrinsics: CameraIntrinsics | None = None) -> np.ndarray:
        if self._model is None:
            self.load()
        matrix = None
        if intrinsics is not None:
            matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                               [0, intrinsics.fy, intrinsics.ppy],
                               [0, 0, 1]], dtype=np.float32)
        return np.asarray(self._model.infer(frame_bgr[:, :, ::-1], matrix), dtype=np.float32)


def candidate_providers(device: str = "auto") -> list[DepthProvider]:
    """Every model the bake-off considers, in the order it reports them."""
    return [
        TransformersDepthProvider("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=device),
        TransformersDepthProvider("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", device=device),
        TransformersDepthProvider("depth-anything/Depth-Anything-V2-Small-hf", device=device, metric=False),
        TorchHubDepthProvider("Metric3D-v2", "metric3d", "load_metric3d"),
        TorchHubDepthProvider("UniDepth-v2", "unidepth", "load_unidepth"),
        TorchHubDepthProvider("DepthPro", "depth_pro", "load_depth_pro"),
    ]


def timed_predict(
    provider: DepthProvider, frame: np.ndarray, intrinsics: CameraIntrinsics | None,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    depth = provider.predict(frame, intrinsics)
    return depth, (time.perf_counter() - started) * 1000.0
