"""Confidence calibration, per the dual-camera recipe v3 §8: "Confidence
calibration (calibrate, don't guess): sample a small labeled set (<=50
images), fit a logistic mapping from raw model score -> empirical accuracy.
Only then are scores 'calibrated.' Until calibrated, treat raw softmax
margins as uncalibrated and note it in flags."

No labeled calibration set exists for this deployment yet -- that requires
real captures scored by a human against ground truth, which this build
environment cannot produce on its own. Rather than inventing calibration
numbers, every confidence value this pipeline reports passes through the
identity `CalibrationMap()` below (a no-op, `fitted=False`) until a caller
runs `fit_calibration()` against a real labeled set of its own; `flags`
carries `uncalibrated_confidences` in that state so nothing downstream
mistakes a raw model score for a calibrated probability -- exactly what v3's
own instruction asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CalibrationMap:
    """A 1-D logistic calibration curve: calibrated = sigmoid(a * raw + b).
    `fitted=False` (the default) means `apply()` is the identity clip --
    i.e. "uncalibrated", per v3 §8."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def apply(self, raw_score: float) -> float:
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        if not self.fitted:
            return raw_score
        x = self.a * raw_score + self.b
        return float(1.0 / (1.0 + np.exp(-x)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "fitted": self.fitted}


def fit_calibration(
    raw_scores: list[float],
    correct: list[bool],
    *,
    epochs: int = 500,
    learning_rate: float = 0.1,
    min_samples: int = 4,
) -> CalibrationMap:
    """v3 §8: fit a 1-D logistic mapping from raw model score to empirical
    accuracy on a small labeled set. Plain-numpy gradient descent -- a
    single scalar-in/scalar-out logistic fit needs no ML framework or
    scikit-learn dependency.

    Returns an unfitted (identity) `CalibrationMap` if there are too few
    samples or only one class present in `correct` -- fitting a logistic
    curve to that would overfit noise, not calibrate anything; an honestly
    unfitted map is preferable to a confidently wrong one.
    """
    x = np.asarray(raw_scores, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if x.size < min_samples or np.unique(y).size < 2:
        return CalibrationMap()

    a, b = 1.0, 0.0
    for _epoch in range(epochs):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad_a = float(np.mean((p - y) * x))
        grad_b = float(np.mean(p - y))
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return CalibrationMap(a=a, b=b, fitted=True)
