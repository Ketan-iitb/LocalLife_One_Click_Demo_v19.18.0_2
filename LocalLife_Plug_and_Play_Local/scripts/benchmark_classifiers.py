"""Compare material classifiers on labelled crops from the real bin.

Usage
-----
1. Save crops of objects from the bin, one folder per true material, named
   exactly as the pipeline's material labels so the zero-shot models can be
   scored against them:

       crops/
         polythene bag/        img001.jpg img002.jpg ...
         paper/                ...
         cardboard/            ...
         fabric or textile/    ...
         mixed or general waste/ ...

   A few dozen per class is enough to see a difference; a few hundred makes it
   a thesis result. Take them from both cameras, in the bin's real lighting --
   a benchmark on well-lit desk photos says nothing about a dark bin.

2. Run:

       python scripts/benchmark_classifiers.py --crops crops/ --out results/classifiers \\
           --models openai/clip-vit-base-patch32 google/siglip2-base-patch16-224 --probe

For every model this reports the zero-shot score (the model as the pipeline
runs it) and, with --probe, a cross-validated linear probe on the same model's
frozen image features (what training a small head on your own crops would
buy). `results/classifiers/summary.json` ranks them by macro-F1; the per-crop
CSVs show exactly which crops each model got wrong.

Whichever model wins here should become LOCALLIFE_MATERIAL_MODEL. Nothing in
this script changes the running system.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.classifier_benchmark import (  # noqa: E402
    evaluate_zero_shot,
    load_labelled_crops,
    nearest_centroid_probe,
    write_report,
)
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.material import _feature_tensor  # noqa: E402
from locallife_cloud.material_siglip import create_material_classifier  # noqa: E402


def _read(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    return image


def _embed(classifier, image_bgr: np.ndarray) -> np.ndarray | None:
    """The classifier's own normalised image embedding for one crop."""
    import torch
    from PIL import Image

    active = getattr(classifier, "_fallback", None) or classifier
    if active.model is None:
        return None
    inputs = active.processor(images=Image.fromarray(image_bgr[:, :, ::-1]), return_tensors="pt")
    inputs = {key: value.to(active.device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = _feature_tensor(active.model.get_image_features(**inputs))
    return features.squeeze(0).float().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", required=True, help="folder of <class>/<image> crops")
    parser.add_argument("--out", default="results/classifier_benchmark")
    parser.add_argument("--models", nargs="+", default=[
        "openai/clip-vit-base-patch32", "google/siglip2-base-patch16-224",
    ])
    parser.add_argument("--probe", action="store_true",
                        help="also score a cross-validated linear probe on each model's features")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    samples = load_labelled_crops(args.crops)
    print(f"{len(samples)} crops across {len({s.label for s in samples})} classes")
    base = AppConfig()
    results = []
    for model in args.models:
        config = replace(base, material_model=model, enable_material_classification=True,
                         material_labels=tuple(sorted({s.label for s in samples})))
        classifier = create_material_classifier(config, args.device)
        classifier.load()
        loaded = classifier.runtime.get("model", model)
        if loaded != model:
            # It fell back. Say so rather than report CLIP's score under
            # SigLIP's name.
            print(f"  {model}: could not load ({classifier.runtime.get('fallback_reason')}); skipped")
            continue
        images = {sample.path: _read(sample.path) for sample in samples}

        def predict(path: Path) -> str:
            image = images[path]
            height, width = image.shape[:2]
            label, _ = classifier.classify(image, None, (0, 0, width, height))
            return label

        zero_shot = evaluate_zero_shot(f"{model} zero-shot", samples, predict)
        results.append(zero_shot)
        print(f"  {model} zero-shot: accuracy {zero_shot.accuracy:.3f}  macro-F1 {zero_shot.macro_f1:.3f}")
        if args.probe:
            features = [_embed(classifier, images[sample.path]) for sample in samples]
            if any(item is None for item in features):
                print(f"  {model}: no image features available; probe skipped")
                continue
            probe = nearest_centroid_probe(
                f"{model} linear probe", np.stack(features), [s.label for s in samples],
                folds=args.folds, paths=[str(s.path) for s in samples],
            )
            results.append(probe)
            print(f"  {model} linear probe: accuracy {probe.accuracy:.3f}  macro-F1 {probe.macro_f1:.3f}")
    if not results:
        print("No classifier could be evaluated.")
        return 1
    summary = write_report(results, args.out)
    print(json.dumps(json.loads(summary.read_text())["ranking_by_macro_f1"], indent=2))
    print(f"Report: {summary.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
