"""Colour support and the deterministic sorting table (playbook sections 11-12).

Both of these exist to stop the system asserting something it cannot support:
a colour no majority of the object's own pixels agrees with, or an
allowed/mis-sort verdict for an object the detector did not really recognise.
"""

from __future__ import annotations

import unittest

import numpy as np

from locallife_cloud.geometry import COLOUR_MIN_SUPPORT, classify_color, dominant_color
from locallife_cloud.sorting_rules import (
    CORRECT,
    MIS_SORT,
    UNKNOWN,
    classify_sorting,
    mis_sort_family,
)


def swatch(bgr, size=40):
    frame = np.tile(np.array(bgr, dtype=np.uint8), (size, size, 1))
    return frame, np.ones((size, size), dtype=bool)


def split_swatch(first_bgr, second_bgr, fraction=0.5, size=40):
    """One mask covering two differently coloured halves."""
    frame = np.tile(np.array(first_bgr, dtype=np.uint8), (size, size, 1))
    cut = int(size * fraction)
    frame[:, cut:] = np.array(second_bgr, dtype=np.uint8)
    return frame, np.ones((size, size), dtype=bool)


class ColourSupportTests(unittest.TestCase):
    def test_a_uniform_object_is_reported_with_full_support(self) -> None:
        for bgr, expected in (
            ((30, 30, 220), "red"),
            ((220, 40, 40), "blue"),
            ((40, 190, 40), "green"),
            ((20, 220, 230), "yellow"),
            ((240, 240, 240), "white"),
            ((15, 15, 15), "black"),
        ):
            with self.subTest(expected=expected):
                colour, support = classify_color(*swatch(bgr))
                self.assertEqual(colour, expected)
                self.assertGreater(support, 0.9)

    def test_an_object_with_no_dominant_colour_is_unknown_not_guessed(self) -> None:
        # Playbook section 11.6. Four-way split: no class holds enough of the
        # mask for an honest answer, so the event must say so.
        size = 40
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :10] = (30, 30, 220)      # red
        frame[:, 10:20] = (220, 40, 40)    # blue
        frame[:, 20:30] = (40, 190, 40)    # green
        frame[:, 30:] = (20, 220, 230)     # yellow
        colour, support = classify_color(frame, np.ones((size, size), dtype=bool))
        self.assertEqual(colour, "unknown")
        self.assertLess(support, COLOUR_MIN_SUPPORT)

    def test_a_clear_majority_still_wins_over_a_minority_contaminant(self) -> None:
        # The realistic case: a green bag with a patch of brown carpet caught
        # inside the mask must still be green, not averaged into something else.
        frame, mask = split_swatch((40, 190, 40), (60, 90, 130), fraction=0.85)
        colour, support = classify_color(frame, mask)
        self.assertEqual(colour, "green")
        self.assertGreater(support, COLOUR_MIN_SUPPORT)

    def test_support_falls_as_the_object_becomes_less_uniform(self) -> None:
        clean = classify_color(*swatch((40, 190, 40)))[1]
        contaminated = classify_color(*split_swatch((40, 190, 40), (60, 90, 130), 0.7))[1]
        self.assertGreater(clean, contaminated)

    def test_a_tiny_mask_reports_unknown_rather_than_a_pixel_guess(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, :5] = True
        self.assertEqual(dominant_color(frame, mask), "unknown")

    def test_the_caller_can_demand_stronger_support(self) -> None:
        frame, mask = split_swatch((40, 190, 40), (60, 90, 130), fraction=0.6)
        self.assertEqual(classify_color(frame, mask, min_support=0.05)[0], "green")
        self.assertEqual(classify_color(frame, mask, min_support=0.95)[0], "unknown")

    def test_the_pale_yellow_fix_survives_the_support_check(self) -> None:
        # Regression guard: the LAB b* rule that stopped a cream carton being
        # called grey runs per pixel now, so it has to hold up when the same
        # rule is also used to count agreement.
        colour, support = classify_color(*swatch((205, 238, 245)))
        self.assertEqual(colour, "yellow")
        self.assertGreater(support, 0.9)


class SortingRuleTests(unittest.TestCase):
    def test_waste_bags_are_correct(self) -> None:
        for label, accepted in (
            ("filled plastic waste bag", "plastic_bag"),
            ("black trash bag", "plastic_bag"),
            ("large garbage bag", "plastic_bag"),
            ("kraft paper bag", "paper_bag"),
            ("cardboard shipping box", "cardboard_box"),
        ):
            with self.subTest(label=label):
                verdict = classify_sorting(label, confidence=0.9, accepted_class=accepted)
                self.assertEqual(verdict.status, CORRECT)
                self.assertEqual(verdict.dashboard_text, "CORRECT")

    def test_the_playbook_mis_sort_families_are_caught(self) -> None:
        for label, family in (
            ("slipper", "footwear"),
            ("running shoe", "footwear"),
            ("cordless drill", "tool"),
            ("vacuum cleaner", "appliance"),
            ("office chair", "furniture"),
        ):
            with self.subTest(label=label):
                verdict = classify_sorting(label, confidence=0.9)
                self.assertEqual(verdict.status, MIS_SORT)
                self.assertEqual(verdict.family, family)
                self.assertEqual(verdict.dashboard_text, "MIS-SORT")

    def test_a_disallowed_family_wins_over_the_word_bag(self) -> None:
        # "vacuum cleaner bag" contains "bag" and must not be waved through on
        # that alone -- the disallowed family is checked first for this reason.
        verdict = classify_sorting("vacuum cleaner bag", confidence=0.9)
        self.assertEqual(verdict.status, MIS_SORT)
        self.assertEqual(verdict.family, "appliance")

    def test_low_confidence_is_never_guessed_either_way(self) -> None:
        verdict = classify_sorting("garbage bag", confidence=0.10, accepted_class="plastic_bag")
        self.assertEqual(verdict.status, UNKNOWN)
        self.assertEqual(verdict.dashboard_text, "UNKNOWN / MANUAL CHECK")
        self.assertIn("confidence", verdict.reason)

    def test_an_unrecognised_object_is_unknown_not_mis_sort(self) -> None:
        # Calling anything unfamiliar a mis-sort would make the mis-sort rate
        # meaningless; the playbook asks for manual check instead.
        verdict = classify_sorting("unclassified object", confidence=0.9)
        self.assertEqual(verdict.status, UNKNOWN)
        self.assertIsNone(verdict.family)

    def test_a_missing_label_is_unknown(self) -> None:
        for label in (None, "", "   "):
            with self.subTest(label=label):
                self.assertEqual(classify_sorting(label, confidence=1.0).status, UNKNOWN)

    def test_labels_are_matched_regardless_of_punctuation_and_case(self) -> None:
        self.assertEqual(mis_sort_family("Office_Chair"), "furniture")
        self.assertEqual(mis_sort_family("POWER-DRILL"), "tool")
        self.assertIsNone(mis_sort_family("garbage bag"))

    def test_the_verdict_serializes_for_the_event_record(self) -> None:
        payload = classify_sorting("slipper", confidence=0.9).to_dict()
        self.assertEqual(payload["sorting_status"], MIS_SORT)
        self.assertEqual(payload["sorting_text"], "MIS-SORT")
        self.assertEqual(payload["sorting_family"], "footwear")


if __name__ == "__main__":
    unittest.main()
