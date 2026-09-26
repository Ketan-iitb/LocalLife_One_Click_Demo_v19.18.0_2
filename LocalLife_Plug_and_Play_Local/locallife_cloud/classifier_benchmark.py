"""Which classifier is actually better on *this* bin: measured, not assumed.

A newer model is not automatically a better one for a dark bin full of
overlapping bags. The honest way to choose is to label a few hundred crops from
the real installation and score every candidate on the same crops. This module
is the scoring half of that; `scripts/benchmark_classifiers.py` is the driver
that runs the models.

Two kinds of classifier are compared:

* zero-shot -- the model and its text prompts, untouched, exactly as the
  pipeline runs it;
* a linear probe on frozen image features -- the "train a small head on your
  own data" approach. Here it is a nearest-class-mean classifier on
  L2-normalised embeddings, evaluated by stratified k-fold so no crop is ever
  scored by a model that saw it. It needs no training library, and it is the
  simplest probe that is still a fair test of whether a backbone's features
  separate the classes of this bin.

Everything here is numpy only, so it is testable without a model and reusable
for any classifier that can be expressed as a function from crop to label.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def load_labelled_crops(root: str | Path) -> list[Sample]:
    """Folder-per-class crops: `root/<class name>/<image>`.

    Hidden files and non-image files are ignored, and the order is sorted so a
    benchmark is reproducible run to run.
    """
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"Crop folder not found: {base}")
    samples: list[Sample] = []
    for folder in sorted(item for item in base.iterdir() if item.is_dir()):
        if folder.name.startswith("."):
            continue
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES and not image.name.startswith("."):
                samples.append(Sample(image, folder.name))
    if not samples:
        raise ValueError(f"No labelled images under {base}; expected {base}/<class>/<image>")
    return samples


@dataclass
class BenchmarkResult:
    name: str
    labels: list[str]
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    predictions: list[tuple[str, str, str]] = field(default_factory=list)
    unknown: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "classifier": self.name,
            "samples": len(self.predictions),
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "unknown_predictions": self.unknown,
            "per_class": {
                label: {key: round(value, 4) for key, value in stats.items()}
                for label, stats in self.per_class.items()
            },
        }


def score(
    name: str, truths: list[str], predictions: list[str], *, paths: list[str] | None = None,
) -> BenchmarkResult:
    """Accuracy, macro-F1, per-class precision/recall and the confusion matrix.

    Macro-F1 is reported beside accuracy because a bin is dominated by one or
    two kinds of object: a classifier that calls everything "polythene bag" can
    score a high accuracy and still be useless for sorting.
    """
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions differ in length")
    labels = sorted(set(truths) | {item for item in predictions if item != "unknown"})
    index = {label: position for position, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    unknown = 0
    for truth, predicted in zip(truths, predictions):
        if predicted not in index:
            unknown += 1
            continue
        confusion[index[truth]][index[predicted]] += 1
    correct = sum(1 for truth, predicted in zip(truths, predictions) if truth == predicted)
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for label in sorted(set(truths)):
        position = index[label]
        true_positive = confusion[position][position]
        predicted_as = sum(row[position] for row in confusion)
        actually = sum(1 for truth in truths if truth == label)
        precision = true_positive / predicted_as if predicted_as else 0.0
        recall = true_positive / actually if actually else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actually}
        f1_scores.append(f1)
    rows = list(zip(
        paths if paths is not None else [""] * len(truths), truths, predictions,
    ))
    return BenchmarkResult(
        name=name,
        labels=labels,
        accuracy=correct / len(truths) if truths else 0.0,
        macro_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
        per_class=per_class,
        confusion=confusion,
        predictions=rows,
        unknown=unknown,
    )


def evaluate_zero_shot(
    name: str, samples: list[Sample], predict: Callable[[Path], str],
) -> BenchmarkResult:
    """Score a classifier as the pipeline runs it: one crop in, one label out."""
    predictions = [predict(sample.path) for sample in samples]
    return score(
        name, [sample.label for sample in samples], predictions,
        paths=[str(sample.path) for sample in samples],
    )


def _normalise(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


def stratified_folds(labels: list[str], folds: int, seed: int = 0) -> list[np.ndarray]:
    """Fold assignment that keeps every class spread across every fold."""
    rng = np.random.default_rng(seed)
    assignment = np.zeros(len(labels), dtype=np.int64)
    for label in sorted(set(labels)):
        positions = np.array([i for i, item in enumerate(labels) if item == label])
        rng.shuffle(positions)
        assignment[positions] = np.arange(positions.size) % folds
    return [np.flatnonzero(assignment == fold) for fold in range(folds)]


def nearest_centroid_probe(
    name: str, features: np.ndarray, labels: list[str], *, folds: int = 5,
    paths: list[str] | None = None, seed: int = 0,
) -> BenchmarkResult:
    """Linear probe by nearest class mean, cross-validated.

    Every crop is predicted by centroids computed without it, so the score is a
    held-out score. A class with fewer examples than folds still contributes;
    it simply appears in fewer training folds.
    """
    embedded = _normalise(features)
    if embedded.shape[0] != len(labels):
        raise ValueError("features and labels differ in length")
    folds = max(2, min(int(folds), len(labels)))
    predictions = ["unknown"] * len(labels)
    for held_out in stratified_folds(labels, folds, seed):
        if held_out.size == 0:
            continue
        training = np.setdiff1d(np.arange(len(labels)), held_out)
        classes = sorted({labels[i] for i in training})
        if not classes:
            continue
        centroids = _normalise(np.stack([
            embedded[[i for i in training if labels[i] == label]].mean(axis=0)
            for label in classes
        ]))
        similarity = embedded[held_out] @ centroids.T
        for row, position in enumerate(held_out):
            predictions[int(position)] = classes[int(np.argmax(similarity[row]))]
    return score(name, list(labels), predictions, paths=paths)


def write_report(results: Iterable[BenchmarkResult], out_dir: str | Path) -> Path:
    """One summary JSON, plus per-classifier predictions and confusion CSVs."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    collected = list(results)
    summary = {
        "classifiers": [result.summary() for result in collected],
        "ranking_by_macro_f1": [
            result.name for result in sorted(collected, key=lambda item: item.macro_f1, reverse=True)
        ],
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    for result in collected:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in result.name)
        with (target / f"{safe}_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "true_label", "predicted_label", "correct"])
            for path, truth, predicted in result.predictions:
                writer.writerow([path, truth, predicted, int(truth == predicted)])
        with (target / f"{safe}_confusion.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true \\ predicted", *result.labels])
            for label, row in zip(result.labels, result.confusion):
                writer.writerow([label, *row])
    return target / "summary.json"
