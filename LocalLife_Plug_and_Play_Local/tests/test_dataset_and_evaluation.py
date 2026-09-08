from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.dataset import audit_dataset
from locallife_cloud.evaluate import evaluate_csv, metric_summary


class DatasetAuditTests(unittest.TestCase):
    def _dataset(self, root: Path, label: str) -> Path:
        for split in ("train", "val"):
            images = root / "images" / split
            labels = root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "example.jpg").write_bytes(b"placeholder")
            (labels / "example.txt").write_text(label + "\n", encoding="utf-8")
        yaml = root / "dataset.yaml"
        yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [plastic, paper]\n", encoding="utf-8")
        return yaml

    def test_detection_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_dataset(self._dataset(Path(temporary), "0 0.5 0.5 0.2 0.3"))
            self.assertTrue(report["ready"])
            self.assertEqual(report["task"], "detect")
            self.assertEqual(report["total_instances"], 2)

    def test_segmentation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_dataset(self._dataset(Path(temporary), "0 0.1 0.1 0.4 0.1 0.4 0.4"))
            self.assertTrue(report["ready"])
            self.assertEqual(report["task"], "segment")

    def test_invalid_coordinate_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_dataset(self._dataset(Path(temporary), "0 1.5 0.5 0.2 0.3"))
            self.assertFalse(report["ready"])

    def test_unknown_class_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_dataset(self._dataset(Path(temporary), "4 0.5 0.5 0.2 0.3"))
            self.assertFalse(report["ready"])

    def test_inconsistent_class_count_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            yaml = self._dataset(Path(temporary), "0 0.5 0.5 0.2 0.3")
            yaml.write_text(yaml.read_text(encoding="utf-8") + "nc: 5\n", encoding="utf-8")
            report = audit_dataset(yaml)
            self.assertFalse(report["ready"])


class EvaluationTests(unittest.TestCase):
    def test_known_error_metrics(self) -> None:
        result = metric_summary(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.1, 1.9, 3.2]))
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["mae_l"], 0.4 / 3, places=7)

    def test_missing_predictions_are_excluded(self) -> None:
        result = metric_summary(np.asarray([1.0, 2.0]), np.asarray([1.0, np.nan]))
        self.assertEqual(result["samples"], 1)

    def test_csv_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "measurements.csv"
            with destination.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["ground_truth_l", "realsense_volume_l", "monocular_volume_l"],
                )
                writer.writeheader()
                writer.writerow({"ground_truth_l": 2, "realsense_volume_l": 2.2, "monocular_volume_l": 1.8})
            report = evaluate_csv(destination)
            self.assertEqual(report["realsense"]["samples"], 1)
            self.assertAlmostEqual(report["monocular_calibrated"]["mae_l"], 0.2)


if __name__ == "__main__":
    unittest.main()
