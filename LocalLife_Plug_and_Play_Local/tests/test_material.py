"""Material classification: standalone behavior and pipeline stabilization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.material import MaterialClassifier
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection


class FixedDetection:
    runtime = {"device": "cpu"}

    def __init__(self, detection: Detection) -> None:
        self.detection = detection

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        item = self.detection
        return [[Detection(
            item.label, item.confidence, item.box,
            None if item.mask is None else item.mask.copy(), color=item.color,
        )] for _ in frames]


class SequencedMaterialClassifier:
    """Deterministic stand-in so tests never need real model weights."""

    enabled = True

    def __init__(self, labels_and_scores: list[tuple[str, float]]) -> None:
        self._answers = list(labels_and_scores)
        self._calls = 0

    def load(self) -> None:
        return None

    def classify(self, frame_bgr, mask, box) -> tuple[str, float]:
        answer = self._answers[min(self._calls, len(self._answers) - 1)]
        self._calls += 1
        return answer


class MaterialClassifierStandaloneTests(unittest.TestCase):
    def test_disabled_classifier_returns_unknown_without_importing_torch(self) -> None:
        config = AppConfig(enable_material_classification=False)
        classifier = MaterialClassifier(config)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        label, confidence = classifier.classify(frame, None, (0, 0, 10, 10))
        self.assertEqual(label, "unknown")
        self.assertEqual(confidence, 0.0)

    def test_missing_dependency_disables_itself_instead_of_raising(self) -> None:
        config = AppConfig(enable_material_classification=True, material_model="does-not-matter")
        classifier = MaterialClassifier(config)
        # transformers/torch may or may not be installed in the test environment;
        # either way load() must never raise, and enabled must reflect reality.
        classifier.load()
        self.assertIsInstance(classifier.enabled, bool)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        label, confidence = classifier.classify(frame, None, (0, 0, 10, 10))
        if not classifier.enabled:
            self.assertEqual(label, "unknown")
            self.assertEqual(confidence, 0.0)


class MaterialPipelineIntegrationTests(unittest.TestCase):
    def test_material_is_stabilized_by_majority_vote_across_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            mask = np.zeros((40, 40), dtype=bool)
            mask[10:30, 10:30] = True
            detection = Detection("garbage bag", 0.9, (10, 10, 30, 30), mask, color="blue")
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                enable_material_classification=True, material_reclassify_frames=1,
                roi=(0, 0, 1, 1), min_component_pixels=10,
                restore_saved_baseline=False, automatic_baseline=False,
                tracker_confirm_frames=1, record_only_measured_objects=False,
            )
            classifier = SequencedMaterialClassifier([
                ("plastic", 0.9), ("plastic", 0.9), ("cardboard", 0.6), ("plastic", 0.9),
            ])
            station = VisionPipeline(
                config, detector=FixedDetection(detection), material_classifier=classifier,
            )
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20)
            result = None
            for _ in range(4):
                result = station.process_frame(frame, intrinsics=intrinsics, persist=False)
            self.assertIsNotNone(result)
            live = [item for item in result.detections if item.tracking_status != "tentative"]
            self.assertEqual(len(live), 1)
            self.assertEqual(live[0].material, "plastic")
            self.assertGreaterEqual(live[0].material_confidence, 0.5)

    def test_low_confidence_material_guesses_are_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            mask = np.zeros((40, 40), dtype=bool)
            mask[10:30, 10:30] = True
            detection = Detection("garbage bag", 0.9, (10, 10, 30, 30), mask, color="blue")
            config = AppConfig(
                results_dir=Path(directory), enable_monocular_depth=False,
                enable_material_classification=True, material_reclassify_frames=1,
                material_confidence_threshold=0.5,
                roi=(0, 0, 1, 1), min_component_pixels=10,
                restore_saved_baseline=False, automatic_baseline=False,
                tracker_confirm_frames=1, record_only_measured_objects=False,
            )
            classifier = SequencedMaterialClassifier([("plastic", 0.2)])
            station = VisionPipeline(
                config, detector=FixedDetection(detection), material_classifier=classifier,
            )
            intrinsics = CameraIntrinsics(fx=100, fy=100, ppx=20, ppy=20)
            result = station.process_frame(frame, intrinsics=intrinsics, persist=False)
            live = [item for item in result.detections if item.tracking_status != "tentative"]
            self.assertEqual(len(live), 1)
            self.assertEqual(live[0].material, "unknown")


if __name__ == "__main__":
    unittest.main()
