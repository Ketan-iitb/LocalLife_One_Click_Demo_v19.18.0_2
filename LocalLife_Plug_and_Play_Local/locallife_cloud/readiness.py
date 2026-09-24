"""What the station still needs before it can measure, in plain fields.

"PENDING — measurable height" told an operator nothing: it named neither the
missing piece nor the action that would supply it. Readiness is therefore
reported one prerequisite at a time, each either ready or missing, with the
single next step spelled out. A station that is missing a prerequisite still
measures -- the provisional numeric estimate stays visible and labelled -- so
this panel explains the quality of a number rather than the absence of one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

READY = "ready"
MISSING = "missing"

# In the order an operator would set them up.
STEPS = ("measurement_zone", "empty_baseline", "intrinsics", "floor_scale", "metric_depth_mapping",
         "camera_floor_distance", "height_calibration")

INSTRUCTIONS = {
    "measurement_zone": "Draw the four corners of the deposit mat for this camera "
                        "(Calibrate measurement zone).",
    "empty_baseline": "Clear the measurement zone and press Capture Empty Measurement Zone.",
    "intrinsics": "Connect the camera stream that reports its intrinsics, or set the camera's "
                  "field of view in the configuration.",
    "floor_scale": "Enter the deposit mat's real width and depth in metres with its corners.",
    "metric_depth_mapping": "Capture a few Logitech calibration samples at known distances, "
                            "or set the measured camera-to-floor distance.",
    "camera_floor_distance": "Measure the camera's optical centre to the empty floor and enter it "
                             "in centimetres (Camera setup).",
    "height_calibration": "Measure two or three rigid objects with a ruler, add each as a "
                          "calibration sample, then fit and freeze the height calibration.",
}


@dataclass(frozen=True)
class MeasurementReadiness:
    """One camera's prerequisites, and what to do about the first gap."""

    camera: str
    measurement_zone: bool = False
    empty_baseline: bool = False
    intrinsics: bool = False
    floor_scale: bool = False
    metric_depth_mapping: bool = False
    # Both default to ready: a camera with hardware depth needs neither, and
    # the Logitech pipeline sets them explicitly from its own metric store.
    camera_floor_distance: bool = True
    height_calibration: bool = True
    method: str = ""
    reason: str = ""
    depth_output: str = ""
    calibration_samples: int = 0
    calibration_samples_required: int = 0
    calibration_status: str = ""

    @property
    def fully_calibrated(self) -> bool:
        return all(getattr(self, step) for step in STEPS)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(step for step in STEPS if not getattr(self, step))

    @property
    def next_step(self) -> str:
        gaps = self.missing
        if not gaps:
            return "Calibrated. Measurements are metric."
        return INSTRUCTIONS[gaps[0]]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            step: (READY if getattr(self, step) else MISSING) for step in STEPS
        }
        payload.update({
            "camera": self.camera,
            "calibration_samples": f"{self.calibration_samples}/{self.calibration_samples_required}"
            if self.calibration_samples_required else "",
            "calibration_status": self.calibration_status,
            "fully_calibrated": self.fully_calibrated,
            "measurement_method": self.method,
            "reason": self.reason,
            "depth_output": self.depth_output,
            "next_step": self.next_step,
            "summary": self.summary(),
        })
        return payload

    def summary(self) -> str:
        """One line for the dashboard."""
        gaps = self.missing
        if not gaps:
            return f"{self.camera}: calibrated ({self.method or 'metric'})"
        readable = ", ".join(step.replace("_", " ") for step in gaps)
        method = self.method or "uncalibrated estimate"
        return f"{self.camera}: measuring as {method}; missing {readable}"
