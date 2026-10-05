"""V53 Phase 2: the Logitech's monocular readings referenced to the RealSense on the SAME bin.

Depth Anything depth is relative (affine): one floor-distance scale cannot fix its relief, so in the
bin a 3-5 L bag read 0.5 L (6 cm tall) and a ~20 % bin read 76 % while the RealSense, with metric
stereo depth, read 5.8 L. When both cameras watch one bin, every deposit both cameras see and every
settled frame both cameras measure is a free calibration sample: the Logitech number times the median
RealSense/Logitech ratio. Raw monocular values are always kept beside the referenced ones, and the
referenced values are labelled -- they are NOT an independent Logitech measurement.

Enabled only for Phase 2 (LOCALLIFE_SETUP_PHASE=phase2) or LOCALLIFE_LOGITECH_CROSS_REFERENCE=1;
Phase 1's independent camera comparison is unchanged. NumPy only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bin_fill import phase_suffix, setup_phase

PAIR_WINDOW_S = 15.0           # the same deposit is committed by both cameras within this
MIN_SAMPLES = 1                # volume: one paired deposit already beats an unreferenced monocular value
MIN_FILL_SAMPLES = 3
FILL_SAMPLE_EVERY_S = 10.0
RATIO_RANGE = (0.05, 20.0)
KEEP = 30
FILL_FIELDS_CM = ("max_fill_height_cm", "tallest_cm")


def enabled() -> bool:
    flag = os.environ.get("LOCALLIFE_LOGITECH_CROSS_REFERENCE", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return flag in ("1", "true", "yes", "on") or setup_phase() == "phase2"


class CrossCameraReference:
    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / f"logitech_cross_reference{phase_suffix()}.json"
        self.volume: list[float] = []
        self.fill: list[float] = []
        self._last_fill_at = 0.0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.volume, self.fill = list(data.get("volume", [])), list(data.get("fill", []))
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"volume": self.volume[-KEEP:], "fill": self.fill[-KEEP:],
                                             "saved_at": time.time()}), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _ratio(samples: list[float], minimum: int) -> float | None:
        return float(np.median(samples[-KEEP:])) if len(samples) >= minimum else None

    @staticmethod
    def _add(samples: list[float], reference: float | None, monocular: float | None) -> bool:
        if not reference or not monocular or reference <= 0 or monocular <= 0:
            return False
        ratio = float(reference) / float(monocular)
        if not RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]:
            return False
        samples.append(ratio)
        del samples[:-KEEP]
        return True

    # -------------------------------------------------------------- per-deposit volume
    def on_logitech_event(self, record: dict[str, Any], events: list[dict[str, Any]]) -> None:
        """Called once, when a Logitech row is finalised: learn from a paired RealSense deposit, then
        reference this row's volume."""
        raw = record.get("logitech_monocular_envelope_l", record.get("envelope_l"))
        if raw is None:
            return
        pair = [e for e in events if e.get("camera") == "realsense" and e.get("envelope_l")
                and abs(float(e["deposit_time"]) - float(record["deposit_time"])) <= PAIR_WINDOW_S]
        if pair:
            best = min(pair, key=lambda e: abs(float(e["deposit_time"]) - float(record["deposit_time"])))
            if self._add(self.volume, best["envelope_l"], raw):
                self._save()
        ratio = self._ratio(self.volume, MIN_SAMPLES)
        if ratio is None:
            return
        record["logitech_monocular_envelope_l"] = raw
        record["envelope_l"] = round(float(raw) * ratio, 2)
        record["volume_method"] = (f"{record.get('volume_method') or 'monocular'}; x{ratio:.2f} referenced to "
                                   f"RealSense ({len(self.volume)} paired deposits, not independent)")

    # -------------------------------------------------------------- bin fill
    def reference_fill(self, realsense: dict[str, Any], logitech: dict[str, Any], now: float | None = None
                       ) -> dict[str, Any]:
        now = time.time() if now is None else now
        both_ok = realsense.get("status") == "ok" and logitech.get("status") == "ok"
        if both_ok and now - self._last_fill_at >= FILL_SAMPLE_EVERY_S:
            self._last_fill_at = now
            if self._add(self.fill, realsense.get("max_fill_height_cm"), logitech.get("max_fill_height_cm")):
                self._save()
        ratio = self._ratio(self.fill, MIN_FILL_SAMPLES)
        if ratio is None or logitech.get("status") != "ok":
            return logitech
        out = dict(logitech)
        out["monocular_raw"] = {k: logitech.get(k) for k in ("height_fill_pct", "rough_litres", *FILL_FIELDS_CM)}
        usable = float(logitech.get("usable_height_cm") or 100.0)
        fill_cm = min(usable, float(logitech.get("max_fill_height_cm") or 0.0) * ratio)
        fraction = fill_cm / usable if usable > 0 else 0.0
        capacity = float(logitech.get("capacity_l") or 660.0)
        out.update(max_fill_height_cm=round(fill_cm, 1), height_fill_pct=round(100 * fraction, 1),
                   remaining_height_cm=round(usable - fill_cm, 1),
                   rough_litres=round(capacity * fraction, 1), rough_remaining_litres=round(capacity * (1 - fraction), 1),
                   tallest_cm=None if logitech.get("tallest_cm") is None
                   else round(min(usable, float(logitech["tallest_cm"]) * ratio), 1),
                   cross_reference=f"Logitech heights x{ratio:.2f}, referenced to the RealSense on the same bin "
                                   f"({len(self.fill)} samples); raw monocular values in monocular_raw")
        return out
