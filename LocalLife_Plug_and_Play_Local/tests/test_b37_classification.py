"""B37: classification improvements that can be measured, not just claimed.

Three things:

* One canonical name per waste bag. The real-bin screenshots show identical
  bags captioned "filled plastic waste bag", "plastic garbage bag" and
  "plastic trash bag" in one frame. Only the material survives the mapping,
  because sorting depends on it.
* SigLIP 2 as an optional material classifier, with an automatic fallback to
  CLIP. Not the default: it has not been run on the real bin.
* A benchmark that scores classifiers on labelled crops from the real bin, so
  the default is chosen by measurement.

No model runs in these tests; this environment has no PyTorch.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from locallife_cloud.classifier_benchmark import (
    Sample,
    evaluate_zero_shot,
    load_labelled_crops,
    nearest_centroid_probe,
    score,
    stratified_folds,
    write_report,
)
from locallife_cloud.config import AppConfig
from locallife_cloud.material import MaterialClassifier
from locallife_cloud.material_siglip import (
    CLIP_FALLBACK_MODEL,
    SiglipMaterialClassifier,
    aggregate_label_scores,
    create_material_classifier,
    is_siglip,
)
from locallife_cloud.vocabulary import canonical_name
from locallife_cloud.waste_bag_names import (
    PAPER_WASTE_BAG,
    PLASTIC_WASTE_BAG,
    object_type,
    waste_bag_name,
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "40126b2f0d7137832dc82f9098140dd455b4d737"


class WasteBagNameTests(unittest.TestCase):
    def test_the_screenshot_phrasings_collapse_to_one_name(self) -> None:
        for label in (
            "filled plastic waste bag", "plastic garbage bag", "plastic trash bag",
            "transparent plastic waste bag", "black garbage bag", "garbage bag",
            "trash bag", "bin bag", "refuse sack", "Plastic-Garbage_Bag",
        ):
            with self.subTest(label=label):
                self.assertEqual(waste_bag_name(label), PLASTIC_WASTE_BAG)
                self.assertEqual(object_type(label, 0.9), PLASTIC_WASTE_BAG)

    def test_paper_stays_paper_because_sorting_depends_on_it(self) -> None:
        self.assertEqual(waste_bag_name("paper waste bag"), PAPER_WASTE_BAG)
        self.assertEqual(waste_bag_name("brown paper bag"), PAPER_WASTE_BAG)

    def test_bags_that_are_not_waste_bags_are_left_alone(self) -> None:
        for label in ("handbag", "laptop bag", "backpack", "tote bag", "school bag"):
            with self.subTest(label=label):
                self.assertIsNone(waste_bag_name(label))
                self.assertNotIn(object_type(label, 0.9), {PLASTIC_WASTE_BAG, PAPER_WASTE_BAG})

    def test_an_unsure_bag_is_still_reported_as_a_bag(self) -> None:
        self.assertEqual(object_type("plastic garbage bag", 0.20), "waste bag")
        self.assertEqual(object_type("plastic garbage bag", 0.80), PLASTIC_WASTE_BAG)

    def test_everything_else_passes_through_to_the_frozen_vocabulary(self) -> None:
        self.assertEqual(object_type("headset", 0.8), "headphones")
        self.assertEqual(object_type("folded clothing", 0.8), "folded textile")
        self.assertEqual(object_type("cream bottle", 0.8), "cosmetic bottle")
        self.assertEqual(object_type("something odd", 0.1), "unknown deposited object")
        self.assertEqual(canonical_name("headset"), "headphones")

    def test_the_volume_calibration_group_of_a_bag_is_unchanged(self) -> None:
        from locallife_cloud.logitech_factor import geometry_group

        self.assertEqual(geometry_group(None, object_type("plastic trash bag", 0.9)), "flexible_bag")
        self.assertEqual(geometry_group(None, object_type("plastic trash bag", 0.1)), "flexible_bag")


class SiglipSelectionTests(unittest.TestCase):
    def test_clip_is_still_the_default(self) -> None:
        config = AppConfig()
        self.assertFalse(is_siglip(config.material_model))
        self.assertIs(type(create_material_classifier(config)), MaterialClassifier)

    def test_a_siglip_model_name_selects_the_siglip_classifier(self) -> None:
        config = replace(AppConfig(), material_model="google/siglip2-base-patch16-224")
        self.assertIsInstance(create_material_classifier(config), SiglipMaterialClassifier)

    def test_a_siglip_load_failure_falls_back_to_clip_and_says_so(self) -> None:
        config = replace(AppConfig(), material_model="google/siglip2-base-patch16-224")
        classifier = SiglipMaterialClassifier(config)
        classifier._use_fallback("simulated: model hub unreachable")
        self.assertIsInstance(classifier._fallback, MaterialClassifier)
        self.assertEqual(classifier.runtime["model"], CLIP_FALLBACK_MODEL)
        self.assertEqual(classifier.runtime["family"], "clip-fallback")
        self.assertIn("unreachable", classifier.runtime["fallback_reason"])

    def test_without_torch_nothing_crashes(self) -> None:
        config = replace(AppConfig(), material_model="google/siglip2-base-patch16-224")
        classifier = SiglipMaterialClassifier(config)
        image = np.zeros((32, 32, 3), np.uint8)
        label, confidence = classifier.classify(image, None, (0, 0, 32, 32))
        # Here torch is missing, so the answer is "unknown", never an exception.
        self.assertIsInstance(label, str)
        self.assertGreaterEqual(confidence, 0.0)

    def test_scores_become_labels_the_same_way_as_the_clip_path(self) -> None:
        logits = np.array([4.0, 3.5, 0.0, -1.0])
        labels = ["polythene bag", "polythene bag", "paper", "metal"]
        best, confidence, scores = aggregate_label_scores(logits, labels)
        self.assertEqual(best, "polythene bag")
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)
        self.assertGreater(confidence, 0.9)

    def test_mismatched_scores_are_unknown(self) -> None:
        self.assertEqual(aggregate_label_scores(np.array([1.0]), ["a", "b"])[0], "unknown")


class BenchmarkScoringTests(unittest.TestCase):
    def test_macro_f1_exposes_a_classifier_that_names_everything_one_class(self) -> None:
        truths = ["polythene bag"] * 8 + ["paper"] * 2
        lazy = score("lazy", truths, ["polythene bag"] * 10)
        self.assertAlmostEqual(lazy.accuracy, 0.8)
        self.assertLess(lazy.macro_f1, 0.5)

    def test_a_perfect_classifier_scores_one(self) -> None:
        truths = ["paper", "metal", "paper"]
        result = score("perfect", truths, list(truths))
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)

    def test_unknown_predictions_are_counted_not_hidden(self) -> None:
        result = score("shy", ["paper", "metal"], ["unknown", "metal"])
        self.assertEqual(result.unknown, 1)
        self.assertAlmostEqual(result.accuracy, 0.5)

    def test_folds_keep_every_class_spread(self) -> None:
        labels = ["a"] * 10 + ["b"] * 5
        folds = stratified_folds(labels, 5)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(15)))
        for fold in folds:
            self.assertEqual(sum(1 for i in fold if labels[i] == "b"), 1)

    def test_the_probe_learns_separable_classes_held_out(self) -> None:
        rng = np.random.default_rng(0)
        features = np.vstack([
            rng.normal([5, 0, 0], 0.3, (20, 3)),
            rng.normal([0, 5, 0], 0.3, (20, 3)),
        ])
        labels = ["plastic"] * 20 + ["paper"] * 20
        result = nearest_centroid_probe("probe", features, labels, folds=5)
        self.assertGreater(result.accuracy, 0.95)

    def test_the_probe_is_at_chance_on_noise(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(0, 1, (60, 16))
        labels = ["a", "b", "c"] * 20
        result = nearest_centroid_probe("noise", features, labels, folds=5)
        # Held out, so noise cannot score well. A probe that leaked the
        # held-out crop into its own centroid would.
        self.assertLess(result.accuracy, 0.6)

    def test_zero_shot_evaluation_and_report(self) -> None:
        samples = [Sample(Path(f"x/{i}.jpg"), label)
                   for i, label in enumerate(["paper", "metal", "paper"])]
        result = evaluate_zero_shot("stub", samples, lambda path: "paper")
        with tempfile.TemporaryDirectory() as directory:
            summary = json.loads(write_report([result], directory).read_text())
            self.assertEqual(summary["ranking_by_macro_f1"], ["stub"])
            self.assertTrue((Path(directory) / "stub_predictions.csv").exists())
            self.assertTrue((Path(directory) / "stub_confusion.csv").exists())

    def test_crops_load_from_one_folder_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("paper", "polythene bag"):
                (root / label).mkdir()
                (root / label / "a.jpg").write_bytes(b"x")
                (root / label / "notes.txt").write_text("ignored")
            (root / ".hidden").mkdir()
            samples = load_labelled_crops(root)
            self.assertEqual(sorted(s.label for s in samples), ["paper", "polythene bag"])

    def test_an_empty_crop_folder_is_an_error_not_a_zero_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_labelled_crops(directory)


class ProtectedSurfacesTests(unittest.TestCase):
    def test_protected_files_are_unchanged(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/material.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/vocabulary.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/sorting_rules.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/realsense.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/depth.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/storage.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "Start-LocalLife-Demo.ps1",
        ]
        result = subprocess.run(
            ["git", "diff", "--name-only", SOURCE_SHA, "--", *protected],
            capture_output=True, text=True, cwd=PROJECT.parent, timeout=120,
        )
        if result.returncode != 0:
            self.skipTest("git or the source commit is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
