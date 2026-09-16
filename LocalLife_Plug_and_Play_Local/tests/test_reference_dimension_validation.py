from __future__ import annotations

import unittest

from scripts.validate_reference_dimensions import (
    DEFAULT_MANIFEST,
    current_dimension_observation,
    find_reference,
    load_references,
)


class ReferenceDimensionValidationTests(unittest.TestCase):
    def test_manifest_exposes_all_annotated_objects_including_waste_negatives(self) -> None:
        references = load_references(DEFAULT_MANIFEST)
        self.assertEqual(len(references), 9)
        backpack = find_reference(references, "bagpack")
        self.assertFalse(backpack["accepted"])
        self.assertEqual(backpack["footprint_length_mm"], 400.0)
        self.assertEqual(backpack["footprint_width_mm"], 300.0)
        self.assertEqual(backpack["height_mm"], 150.0)

    def test_dimension_observation_requires_one_validation_object(self) -> None:
        object_item = {
            "label": "backpack", "source": "yoloe", "tracking_status": "confirmed",
            "accepted_class": "measurement_object", "color": "black",
            "dimensions_mm": {
                "footprint_length": 402.0, "footprint_width": 298.0, "height": 147.0,
            },
            "dimension_confidence": 0.82,
            "dimension_flags": ["single_view_geometry"],
            "dimension_method": "realsense_support_plane_pca",
            "realsense_volume_l": 12.0,
        }
        state = {
            "operating_mode": "geometry_validation", "build_version": "test",
            "cameras": {"realsense": {"latest": {"detections": [object_item]}}},
        }
        observation = current_dimension_observation(state)
        self.assertIsNotNone(observation)
        self.assertEqual(observation["length_mm"], 402.0)
        state["cameras"]["realsense"]["latest"]["detections"].append(dict(object_item))
        self.assertIsNone(current_dimension_observation(state))

    def test_waste_class_is_not_accepted_as_validation_evidence(self) -> None:
        state = {"cameras": {"realsense": {"latest": {"detections": [{
            "label": "cardboard box", "source": "yoloe", "tracking_status": "confirmed",
            "accepted_class": "cardboard_box",
            "dimensions_mm": {"footprint_length": 100, "footprint_width": 80, "height": 50},
        }]}}}}
        self.assertIsNone(current_dimension_observation(state))


if __name__ == "__main__":
    unittest.main()
