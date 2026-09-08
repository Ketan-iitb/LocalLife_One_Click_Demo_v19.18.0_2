"""Single-source configuration for the dual-camera recipe pipeline, per the
v3 blueprint's §10 "Config Schema (single config.yaml)".

This is deliberately a *separate* config surface from `config.py`'s
`AppConfig` (env-var driven, controls the existing dashboard/tracking/
ledger service): the v3 recipe is explicit that its own tuning knobs -- mask
erosion, depth-validity band, SOR/plane thresholds, k-means k, bag class
sizes and tolerance, etc. -- live in one `config.yaml`, not scattered env
vars, so a thesis run can be reproduced from one committed file. Only the
two flags that gate the recipe pipeline's existence in the shared dashboard
(`recipe_enabled`, `recipe_refresh_seconds`) stay on `AppConfig`, because
they are about *this project's* integration choice (off by default,
throttled), not something the recipe blueprint itself specifies.

Every field below defaults to exactly the value the recipe document's own
`config.yaml` sample shows (§10); a project can still override any subset
via its own YAML file (see `from_yaml`) without needing to specify every key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

_DEFAULT_YAML_FILENAME = "recipe_config.yaml"


@dataclass(slots=True)
class RealsenseCameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    preset: str = "HighAccuracy"
    align_depth_to_color: bool = True


@dataclass(slots=True)
class LogitechCameraConfig:
    width: int = 640
    height: int = 480


@dataclass(slots=True)
class SegmentationConfig:
    model: str = "yolov8n-seg.pt"  # or "yolov11n-seg.pt"
    target_class: str | None = None  # null = auto; or "box"/"bag"
    conf_threshold: float = 0.45
    area_range: tuple[float, float] = (0.05, 0.80)


@dataclass(slots=True)
class SorConfig:
    nb_neighbors: int = 20
    std_ratio: float = 2.0


@dataclass(slots=True)
class DepthConfig:
    valid_mm: tuple[float, float] = (100.0, 2500.0)
    mask_erode_px: int = 3
    sor: SorConfig = field(default_factory=SorConfig)
    plane_threshold_m: float = 0.01

    @property
    def valid_range_m(self) -> tuple[float, float]:
        low, high = self.valid_mm
        return (float(low) / 1000.0, float(high) / 1000.0)


@dataclass(slots=True)
class ColorConfig:
    use_camera: str = "logitech"
    kmeans_k: int = 4
    center_crop_frac: float = 0.6


@dataclass(slots=True)
class MaterialConfig:
    primary: str = "clip"
    fallback_model: str = "mobilenetv3_small"
    # v3 §7: "Level-2 fallback: if CLIP confidence < 0.6 ... switch to
    # fallback." "If the fine-tuned model also reports < 0.55, return
    # material=Other + ambiguous_material=true."
    clip_confidence_floor: float = 0.6
    fallback_confidence_floor: float = 0.55
    # v3 §7 point 5: "thin/translucent plastic ~ Paper/Other -- always route
    # these to the fine-tuned model." A CLIP "Plastic" call whose softmax
    # margin over the runner-up class is below this is treated as "thin".
    plastic_thin_margin: float = 0.10
    fallback_checkpoint_path: str | None = None


@dataclass(slots=True)
class VolumeConfig:
    box_method: str = "plane_fit"
    bag_class_sizes_l: tuple[float, ...] = (5.0, 10.0)
    bag_tolerance_frac: float = 0.15
    bag_discretize_window: float = 0.30


@dataclass(slots=True)
class RecipeConfig:
    realsense: RealsenseCameraConfig = field(default_factory=RealsenseCameraConfig)
    logitech: LogitechCameraConfig = field(default_factory=LogitechCameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)

    @classmethod
    def default(cls) -> "RecipeConfig":
        return cls()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RecipeConfig":
        """Build a `RecipeConfig` from a nested dict (typically parsed YAML),
        tolerant of missing keys/sections -- anything not given keeps its
        dataclass default, so a caller's config.yaml only needs to state
        what it wants to override."""
        payload = payload or {}
        config = cls.default()

        def _apply(target: Any, updates: dict[str, Any] | None) -> Any:
            if not updates or not is_dataclass(target):
                return target
            kwargs = {f.name: getattr(target, f.name) for f in fields(target)}
            for key, value in updates.items():
                if key not in kwargs:
                    continue
                current = kwargs[key]
                if is_dataclass(current) and isinstance(value, dict):
                    kwargs[key] = _apply(current, value)
                elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
                    kwargs[key] = tuple(value)
                else:
                    kwargs[key] = value
            return type(target)(**kwargs)

        return _apply(config, payload)

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "RecipeConfig":
        """Load from a YAML file. `path` defaults to `recipe_config.yaml`
        next to this module. Missing file or missing PyYAML both fall back
        to `default()` rather than raising -- the recipe pipeline is
        additive and off by default (see `config.py`'s own
        `recipe_enabled`), so a config-loading problem should degrade to
        sane defaults, not break the caller.
        """
        resolved_path = path or os.path.join(os.path.dirname(__file__), _DEFAULT_YAML_FILENAME)
        if not os.path.isfile(resolved_path):
            return cls.default()
        try:
            import yaml
        except ImportError:
            return cls.default()
        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:
            return cls.default()
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        def _dump(value: Any) -> Any:
            if is_dataclass(value):
                return {f.name: _dump(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, tuple):
                return list(value)
            return value

        return _dump(self)
