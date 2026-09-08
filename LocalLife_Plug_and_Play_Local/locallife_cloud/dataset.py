"""Audit YOLO datasets before spending GPU time on broken labels or splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_root(yaml_path: Path, configured_root: Any) -> Path:
    if not configured_root:
        return yaml_path.parent.resolve()
    root = Path(str(configured_root)).expanduser()
    return root.resolve() if root.is_absolute() else (yaml_path.parent / root).resolve()


def _images_for_split(root: Path, value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({image for item in value for image in _images_for_split(root, item)})
    location = Path(str(value))
    location = location if location.is_absolute() else root / location
    if location.is_dir():
        return sorted(path for path in location.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if location.is_file() and location.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in location.read_text(encoding="utf-8").splitlines():
            item = Path(line.strip())
            if line.strip():
                images.append(item if item.is_absolute() else (location.parent / item).resolve())
        return images
    if location.is_file() and location.suffix.lower() in IMAGE_SUFFIXES:
        return [location]
    return []


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [index for index, value in enumerate(parts) if value == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def audit_dataset(yaml_file: str | Path, *, maximum_examples: int = 12) -> dict[str, Any]:
    yaml_path = Path(yaml_file).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {yaml_path}")

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset YAML must contain a mapping")

    names_value = payload.get("names", [])
    if isinstance(names_value, dict):
        names = {int(key): str(value) for key, value in names_value.items()}
    elif isinstance(names_value, list):
        names = {index: str(value) for index, value in enumerate(names_value)}
    else:
        raise ValueError("Dataset YAML names must be a list or mapping")
    if not names:
        raise ValueError("Dataset YAML must declare non-empty class names")

    root = _resolve_root(yaml_path, payload.get("path"))
    issues: list[str] = []
    if "nc" in payload and int(payload["nc"]) != len(names):
        issues.append("The declared nc value does not match the number of class names")
    examples: list[str] = []
    counts: Counter[int] = Counter()
    formats: Counter[str] = Counter()
    missing_labels = 0
    empty_labels = 0
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        images = _images_for_split(root, payload.get(split))
        split_counts[split] = len(images)
        if split in {"train", "val"} and not images:
            issues.append(f"The {split} split contains no readable images")

        for image in images:
            label_file = _label_for_image(image)
            if not label_file.exists():
                missing_labels += 1
                if len(examples) < maximum_examples:
                    examples.append(f"Missing label: {label_file}")
                continue

            rows = [row.strip() for row in label_file.read_text(encoding="utf-8").splitlines() if row.strip()]
            if not rows:
                empty_labels += 1
                continue

            for line_number, row in enumerate(rows, start=1):
                parts = row.split()
                try:
                    class_number = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                except (ValueError, IndexError):
                    issues.append(f"Malformed label at {label_file}:{line_number}")
                    continue

                if class_number not in names:
                    issues.append(f"Unknown class {class_number} at {label_file}:{line_number}")
                else:
                    counts[class_number] += 1

                if len(parts) == 5:
                    formats["detect"] += 1
                    if coordinates[2] <= 0 or coordinates[3] <= 0:
                        issues.append(f"Non-positive box size at {label_file}:{line_number}")
                elif len(parts) >= 7 and len(parts) % 2 == 1:
                    formats["segment"] += 1
                else:
                    issues.append(f"Unsupported label format at {label_file}:{line_number}")

                if any(value < -0.001 or value > 1.001 for value in coordinates):
                    issues.append(f"Coordinates outside [0, 1] at {label_file}:{line_number}")

    if len(formats) > 1:
        issues.append("The dataset mixes bounding-box and segmentation label formats")
    if sum(counts.values()) == 0:
        issues.append("No valid labeled objects were found")

    task = next(iter(formats), "unknown")
    classes = {
        str(class_id): {"name": label, "instances": counts[class_id]}
        for class_id, label in sorted(names.items())
    }
    warnings: list[str] = []
    if missing_labels:
        warnings.append(f"{missing_labels} images have no matching label file")
    if empty_labels:
        warnings.append(f"{empty_labels} images contain empty/background-only labels")
    if counts:
        smallest, largest = min(counts.values()), max(counts.values())
        if smallest and largest / smallest >= 5:
            warnings.append("Class imbalance exceeds 5:1; collect or augment minority classes")
    for class_id, label in names.items():
        if counts[class_id] == 0:
            warnings.append(f"Declared class '{label}' has no annotated objects")

    return {
        "dataset_yaml": str(yaml_path),
        "dataset_root": str(root),
        "task": task,
        "splits": split_counts,
        "classes": classes,
        "total_instances": sum(counts.values()),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "issues": issues,
        "warnings": warnings,
        "examples": examples,
        "ready": not issues,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a YOLO detection or segmentation dataset")
    parser.add_argument("data", help="Path to the YOLO dataset.yaml file")
    parser.add_argument("--output", help="Optional path for the JSON audit report")
    args = parser.parse_args()
    report = audit_dataset(args.data)
    print(json.dumps(report, indent=2))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
