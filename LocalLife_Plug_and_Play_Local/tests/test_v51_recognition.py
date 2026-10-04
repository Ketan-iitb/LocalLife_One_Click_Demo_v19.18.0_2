"""V51: object name, visible material and colour are separate, honest attributes (logic tests)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud.colour_evidence import describe_colour  # noqa: E402
from locallife_cloud.recognition import canonical_name, object_name, visible_material  # noqa: E402


class NamingTests(unittest.TestCase):
    def test_confusing_pairs_keep_distinct_names(self) -> None:
        cases = {"garbage bag": "plastic waste bag", "backpack": "backpack", "rucksack": "backpack",
                 "pillow": "pillow", "cushion": "pillow", "cardboard shipping box": "cardboard box",
                 "book": "book", "sneaker": "shoe", "slipper": "slipper", "sandal": "slipper",
                 "table lamp": "lamp", "glass bottle": "bottle", "plastic bottle": "bottle",
                 "milk carton": "drink carton", "soda can": "can", "laptop bag": "laptop bag"}
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(canonical_name(label), expected)
        self.assertEqual(canonical_name("candle"), "candle")             # whole words: not a "can"

    def test_weak_evidence_gives_a_broad_name_not_a_confident_wrong_one(self) -> None:
        self.assertEqual(object_name("textile item", None, None)[0], "textile item")
        self.assertEqual(object_name("unknown deposited object", None, None)[0], "unknown object")
        name, basis = object_name("pillow", "backpack", 0.8)
        self.assertEqual(name, "backpack")
        self.assertIn("resolved", basis)
        self.assertIn("not yet confirmed", object_name("backpack", "pillow", 0.3)[1])


class MaterialTests(unittest.TestCase):
    def test_object_identity_does_not_decide_material(self) -> None:
        # A detector "garbage bag" whose pixels look like textile is reported as textile.
        votes = [("fabric or textile", 0.5), ("fabric or textile", 0.6), ("polythene bag", 0.4)]
        self.assertEqual(visible_material(votes, "plastic waste bag")["visible_material"], "textile")
        # A bottle has no class prior: plastic or glass must be seen.
        self.assertEqual(visible_material([], "bottle")["visible_material"], "unknown")
        glass = visible_material([("glass", 0.5), ("glass", 0.6)], "bottle")
        self.assertEqual(glass["visible_material"], "glass")

    def test_class_prior_only_as_a_labelled_fallback(self) -> None:
        box = visible_material([], "cardboard box")
        self.assertEqual(box["visible_material"], "paper/cardboard")
        self.assertIn("not observed", box["source"])
        self.assertEqual(visible_material([("plastic", 0.5)], "cardboard box")["visible_material"], "unknown")

    def test_split_votes_and_contents_stay_unknown(self) -> None:
        split = [("plastic", 0.5), ("metal", 0.5), ("fabric or textile", 0.5)]
        self.assertEqual(visible_material(split, "lamp")["visible_material"], "unknown")
        organic = [("food or organic waste", 0.7)] * 3
        self.assertEqual(visible_material(organic, "plastic waste bag")["visible_material"], "unknown")


class ColourTests(unittest.TestCase):
    @staticmethod
    def _scene(object_bgr, floor_bgr=(110, 110, 110), size=(120, 160)):
        frame = np.zeros((*size, 3), np.uint8)
        frame[:] = floor_bgr
        frame[0:6, 0:6] = (235, 235, 235)                     # a white patch somewhere in the room
        mask = np.zeros(size, bool)
        mask[30:90, 40:120] = True
        frame[mask] = object_bgr
        return frame, mask

    def test_lit_cardboard_is_brown_not_orange(self) -> None:
        frame, mask = self._scene((105, 150, 190))             # BGR of daylight cardboard, V 0.75, S 0.45
        self.assertEqual(describe_colour(frame, mask).colour, "brown")

    def test_saturated_orange_stays_orange(self) -> None:
        frame, mask = self._scene((0, 120, 245))
        self.assertEqual(describe_colour(frame, mask).colour, "orange")

    def test_white_under_dim_exposure_is_white(self) -> None:
        frame, mask = self._scene((180, 182, 184))
        frame[0:25, :] = (190, 190, 190)                       # a white wall, dimly exposed (V 0.75)
        self.assertEqual(describe_colour(frame, mask).colour, "white")

    def test_floor_outside_the_mask_does_not_leak(self) -> None:
        frame, mask = self._scene((30, 30, 30), floor_bgr=(40, 160, 40))     # black object on green mat
        self.assertEqual(describe_colour(frame, mask).colour, "black")

    def test_two_colour_object_reports_a_secondary(self) -> None:
        frame, mask = self._scene((235, 235, 235))
        frame[30:90, 40:75] = (20, 20, 20)                      # black stripe panel on a white object
        result = describe_colour(frame, mask)
        self.assertEqual(result.colour, "white")
        self.assertEqual(result.secondary, ["black"])

    def test_too_little_foreground_is_unknown(self) -> None:
        frame, mask = self._scene((30, 30, 200))
        tiny = np.zeros_like(mask)
        tiny[50:53, 50:53] = True
        self.assertEqual(describe_colour(frame, tiny).colour, "unknown")


if __name__ == "__main__":
    unittest.main()
