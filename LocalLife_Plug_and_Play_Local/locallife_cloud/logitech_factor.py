"""Frozen empirical correction for Logitech volumes.

The geometry is corrected first (support-plane height map, calibrated
intrinsics); what remains is a systematic monocular bias. It is measured once,
from calibration objects whose real volume is known, and then frozen:

    f_i = V_reference / V_logitech_raw          per calibration sample
    f   = median of the samples (robust to one bad observation)
    V_corrected = V_raw * f

Rules that keep the camera comparison honest, enforced here rather than
documented:

* the factor is only applied once it is frozen, so a run cannot be corrected
  by its own observations;
* an object already used as a calibration sample is recorded, so the same
  observation is never also an evaluation point;
* a per-group factor needs `MIN_GROUP_SAMPLES` independent samples, otherwise
  the global factor is used;
* the raw value is always kept beside the corrected one.

RealSense is a benchmark, never ground truth: a sample carries whatever
reference volume the operator measured, and its source.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

GROUPS = ("rigid_box", "cylinder", "flexible_bag", "flat_object", "electronics", "irregular")
MIN_GROUP_SAMPLES = 4
MIN_GLOBAL_SAMPLES = 3
FACTOR_LIMITS = (0.05, 20.0)
VERSION = 1


@dataclass
class FactorSample:
    object_name: str
    group: str
    reference_litres: float
    raw_litres: float
    reference_source: str = "manual"
    captured_at: float = field(default_factory=time.time)

    @property
    def factor(self) -> float:
        return self.reference_litres / self.raw_litres

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_name": self.object_name, "group": self.group,
            "reference_litres": round(self.reference_litres, 6),
            "raw_litres": round(self.raw_litres, 6), "factor": round(self.factor, 6),
            "reference_source": self.reference_source, "captured_at": self.captured_at,
        }


def _robust_factor(values: list[float]) -> tuple[float, float]:
    """Median factor and its relative dispersion (MAD / median)."""
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = 1.4826 * float(np.median(np.abs(array - median)))
    return median, (mad / median if median else float("inf"))


class LogitechVolumeFactors:
    """Versioned store of calibration samples and the frozen factors."""

    def __init__(self, path: Path, *, camera_setup: str = "") -> None:
        self.path = Path(path)
        self.camera_setup = camera_setup
        self.samples: list[FactorSample] = []
        self.global_factor: float | None = None
        self.group_factors: dict[str, float] = {}
        self.frozen_at: float | None = None
        self.version = VERSION
        self._load()

    # ----------------------------------------------------------------- state
    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.version = int(payload.get("version", VERSION))
        self.camera_setup = payload.get("camera_setup") or self.camera_setup
        self.global_factor = payload.get("global_factor")
        self.group_factors = {key: float(value) for key, value in (payload.get("group_factors") or {}).items()}
        self.frozen_at = payload.get("frozen_at")
        self.samples = [
            FactorSample(
                object_name=str(item.get("object_name", "")), group=str(item.get("group", "irregular")),
                reference_litres=float(item["reference_litres"]), raw_litres=float(item["raw_litres"]),
                reference_source=str(item.get("reference_source", "manual")),
                captured_at=float(item.get("captured_at", 0.0)),
            )
            for item in payload.get("samples", [])
            if item.get("raw_litres")
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "camera_setup": self.camera_setup,
            "global_factor": self.global_factor,
            "group_factors": self.group_factors,
            "frozen_at": self.frozen_at,
            "samples": [sample.to_dict() for sample in self.samples],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    # ------------------------------------------------------------ collection
    def add_sample(
        self, object_name: str, group: str, reference_litres: float, raw_litres: float,
        *, reference_source: str = "manual",
    ) -> dict[str, Any]:
        if group not in GROUPS:
            raise ValueError(f"group must be one of {', '.join(GROUPS)}")
        if not (reference_litres > 0 and raw_litres > 0):
            raise ValueError("Reference and raw volumes must both be positive litres")
        if self.frozen_at is not None:
            raise ValueError("The factors are frozen; unfreeze before adding calibration samples")
        sample = FactorSample(object_name.strip() or "unnamed", group, float(reference_litres),
                              float(raw_litres), reference_source)
        if not FACTOR_LIMITS[0] <= sample.factor <= FACTOR_LIMITS[1]:
            raise ValueError(f"Implausible factor {sample.factor:.2f}; check the reference volume")
        self.samples.append(sample)
        self._save()
        return self.status()

    def freeze(self) -> dict[str, Any]:
        """Fit the factors from the samples collected so far and lock them."""
        if len(self.samples) < MIN_GLOBAL_SAMPLES:
            raise ValueError(f"At least {MIN_GLOBAL_SAMPLES} calibration samples are needed before freezing")
        self.global_factor, _ = _robust_factor([sample.factor for sample in self.samples])
        self.group_factors = {}
        for group in GROUPS:
            factors = [sample.factor for sample in self.samples if sample.group == group]
            if len(factors) >= MIN_GROUP_SAMPLES:
                self.group_factors[group], _ = _robust_factor(factors)
        self.frozen_at = time.time()
        self._save()
        LOGGER.info("Logitech volume factors frozen: global %.3f, groups %s",
                    self.global_factor, self.group_factors)
        return self.status()

    def unfreeze(self) -> dict[str, Any]:
        self.frozen_at = None
        self._save()
        return self.status()

    # ----------------------------------------------------------- application
    def factor_for(self, group: str | None) -> tuple[float, str]:
        """The factor to apply and where it came from."""
        if self.frozen_at is None or not self.global_factor:
            return 1.0, "uncalibrated_raw"
        if group and group in self.group_factors:
            return self.group_factors[group], f"group:{group}"
        return self.global_factor, "global"

    def correct(self, raw_litres: float | None, group: str | None) -> tuple[float | None, dict[str, Any]]:
        if raw_litres is None:
            return None, {"factor": None, "source": "no_raw_volume"}
        factor, source = self.factor_for(group)
        return round(raw_litres * factor, 6), {"factor": factor, "source": source,
                                               "raw_litres": round(raw_litres, 6)}

    def calibration_objects(self) -> set[str]:
        """Objects already used for calibration; they are not evaluation points."""
        return {sample.object_name.strip().lower() for sample in self.samples}

    def status(self) -> dict[str, Any]:
        dispersion = None
        if self.samples:
            _, dispersion = _robust_factor([sample.factor for sample in self.samples])
        return {
            "version": self.version,
            "camera_setup": self.camera_setup,
            "frozen": self.frozen_at is not None,
            "frozen_at": self.frozen_at,
            "global_factor": self.global_factor,
            "group_factors": self.group_factors,
            "sample_count": len(self.samples),
            "samples_per_group": {group: sum(1 for item in self.samples if item.group == group)
                                  for group in GROUPS if any(item.group == group for item in self.samples)},
            "dispersion": None if dispersion is None else round(dispersion, 4),
            "samples": [sample.to_dict() for sample in self.samples],
            "note": ("Applied to every Logitech volume." if self.frozen_at is not None else
                     "Collecting calibration samples; raw volumes are reported until the factors are frozen."),
        }


def geometry_group(geometry_method: str | None, label: str = "") -> str:
    """The calibration group an object belongs to, from its geometry and name."""
    method = (geometry_method or "").lower()
    words = set(str(label or "").lower().replace("_", " ").split())
    if method == "cylinder":
        return "cylinder"
    if method == "cuboid":
        return "rigid_box"
    if words & {"bag", "sack", "pillow", "cushion", "cloth", "textile", "clothing"}:
        return "flexible_bag"
    if words & {"headphones", "charger", "cable", "phone", "laptop", "mouse", "keyboard",
                "bulb", "lamp", "drill", "battery", "electronic", "electrical"}:
        return "electronics"
    if words & {"packet", "wrapper", "book", "painting", "frame", "envelope"}:
        return "flat_object"
    return "irregular"
