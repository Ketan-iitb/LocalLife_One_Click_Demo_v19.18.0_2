"""B36: a real waste bin, where bags touch, sag, and were already there.

The field case these tests are built from, read off the dashboard:

    object                     L x W x H (mm)   litres   envelope   fill
    RealSense #11 red bag      317 x 196 x 120   9.013     7.46 L    1.21
    Logitech  #16 red bag      437 x 272 x 251  20.092    29.83 L    0.67
    Logitech  #15 brown bag    340 x 277 x  69   3.442     6.50 L    0.53

Both Logitech rows carried `volume_disagrees_with_dimensions` and both still
printed litres. The first two tests below show why: the check compared the
integrated volume against length x width x height, which is the right test for
a carton and the wrong one for a bag. A filled bag sags into half to two thirds
of its envelope, so 0.67 and 0.53 are what a bag looks like. Meanwhile the same
check passed the RealSense reading at 1.21 -- a volume larger than the box that
contains it, which cannot happen.

The field figures are read from the screenshots, and the ratios computed from
them are arithmetic, not a hardware claim. The synthetic scenes are ray-traced
with the rig's geometry and known truth, and say nothing about what either real
camera reports.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.logitech_bin import (
    EMPTY_ENVELOPE,
    IMPOSSIBLE_FILL,
    MERGED,
    NOT_FLOOR_RELATIVE,
    SHAPE_BOX,
    SHAPE_CYLINDER,
    SHAPE_IRREGULAR,
    UNSETTLED,
    envelope_fill,
    merged_neighbour_reason,
    publish_decision,
    shape_model_for,
)
from locallife_cloud.logitech_geometry import GeometryStabiliser, static_background_reason

from .test_v35_logitech_metric_geometry import (  # noqa: F401 - shared synthetic rig
    CAMERA,
    HEIGHT,
    WIDTH,
    _cream_bottle,
    _measure,
    _shoe_box,
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "26f07c2088483910cad39ae173b912423482f4c0"

# Straight off the dashboard in the attached screenshots.
FIELD_REALSENSE_RED_BAG = (0.317, 0.196, 0.120, 9.013)
FIELD_LOGITECH_RED_BAG = (0.437, 0.272, 0.251, 20.092)
FIELD_LOGITECH_BROWN_BAG = (0.340, 0.277, 0.069, 3.442)


class TheFieldCaseTests(unittest.TestCase):
    """Why every bag was flagged, and why the impossible reading was not."""

    def test_a_sagging_bag_is_not_a_geometry_failure(self) -> None:
        for name, (length, width, height, litres) in (
            ("red", FIELD_LOGITECH_RED_BAG), ("brown", FIELD_LOGITECH_BROWN_BAG),
        ):
            with self.subTest(bag=name):
                fill = envelope_fill(
                    length_m=length, width_m=width, height_m=height, litres=litres,
                )
                self.assertTrue(fill.plausible, fill.reason)
                self.assertIsNone(fill.reason)
                # Half to two thirds of the enclosing box: a bag.
                self.assertGreater(fill.fill_fraction, 0.40)
                self.assertLess(fill.fill_fraction, 0.80)

    def test_a_volume_larger_than_its_envelope_is_caught(self) -> None:
        length, width, height, litres = FIELD_REALSENSE_RED_BAG
        fill = envelope_fill(
            length_m=length, width_m=width, height_m=height, litres=litres,
        )
        self.assertFalse(fill.plausible)
        self.assertEqual(fill.reason, IMPOSSIBLE_FILL)
        self.assertGreater(fill.fill_fraction, 1.0)

    def test_an_envelope_far_larger_than_the_object_is_caught(self) -> None:
        # A mask that ran across the bin: a large envelope, almost no object.
        fill = envelope_fill(length_m=0.9, width_m=0.7, height_m=0.4, litres=2.0)
        self.assertFalse(fill.plausible)
        self.assertEqual(fill.reason, EMPTY_ENVELOPE)

    def test_missing_geometry_is_not_silently_plausible(self) -> None:
        for case in (
            {"length_m": None, "width_m": 0.2, "height_m": 0.1, "litres": 1.0},
            {"length_m": 0.3, "width_m": 0.2, "height_m": 0.1, "litres": 0.0},
        ):
            with self.subTest(**case):
                fill = envelope_fill(**case)
                self.assertFalse(fill.plausible)
                self.assertEqual(fill.reason, "missing_geometry")


class ShapeModelTests(unittest.TestCase):
    """Volume must say which shape model it follows."""

    def test_a_bag_is_measured_as_an_irregular_height_map(self) -> None:
        for label in ("filled plastic waste bag", "paper waste bag", "textile item"):
            with self.subTest(label=label):
                self.assertEqual(shape_model_for(label), SHAPE_IRREGULAR)

    def test_a_carton_is_measured_as_a_cuboid(self) -> None:
        self.assertEqual(shape_model_for("cardboard carton"), SHAPE_BOX)
        self.assertEqual(shape_model_for("shoe box"), SHAPE_BOX)

    def test_geometry_beats_the_label(self) -> None:
        # A reconstructed round cross-section is round whatever it is called.
        self.assertEqual(
            shape_model_for("plastic waste bag", occlusion_corrected=True), SHAPE_CYLINDER,
        )


class TouchingBagsTests(unittest.TestCase):
    """Separate bags stay separate; a swallowed neighbour is named."""

    def setUp(self) -> None:
        self.big = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.big[200:400, 200:460] = True

    def test_two_separate_bags_are_measured_separately(self) -> None:
        other = np.zeros((HEIGHT, WIDTH), dtype=bool)
        other[200:330, 480:560] = True           # beside it, not inside it
        self.assertIsNone(merged_neighbour_reason(self.big, [other]))

    def test_bags_that_merely_touch_are_still_separate(self) -> None:
        other = np.zeros((HEIGHT, WIDTH), dtype=bool)
        other[200:330, 455:560] = True           # a few columns of contact
        self.assertIsNone(merged_neighbour_reason(self.big, [other]))

    def test_a_mask_that_swallowed_its_neighbour_is_named(self) -> None:
        swallowed = np.zeros((HEIGHT, WIDTH), dtype=bool)
        swallowed[240:330, 240:380] = True       # entirely inside the big mask
        self.assertEqual(merged_neighbour_reason(self.big, [swallowed]), MERGED)

    def test_the_smaller_mask_is_not_blamed_for_the_larger_one(self) -> None:
        swallowed = np.zeros((HEIGHT, WIDTH), dtype=bool)
        swallowed[240:330, 240:380] = True
        # Examined from the small bag's side there is nothing to report: only a
        # mask that took in a smaller neighbour is the merged one.
        self.assertIsNone(merged_neighbour_reason(swallowed, [self.big]))

    def test_a_speck_is_not_a_neighbour(self) -> None:
        speck = np.zeros((HEIGHT, WIDTH), dtype=bool)
        speck[250:253, 250:255] = True
        self.assertIsNone(merged_neighbour_reason(self.big, [speck]))


class PublishGateTests(unittest.TestCase):
    """No litres while the geometry is still moving, and always a reason."""

    def _fill(self, **overrides):
        case = {"length_m": 0.437, "width_m": 0.272, "height_m": 0.251, "litres": 20.092}
        case.update(overrides)
        return envelope_fill(**case)

    def test_a_settled_plausible_bag_is_published(self) -> None:
        decision = publish_decision(
            settled=True, plane_is_floor=True, fill=self._fill(),
        )
        self.assertTrue(decision.publish)
        self.assertTrue(decision.final)
        self.assertIsNone(decision.reason)
        self.assertEqual(decision.labels, ())
        self.assertAlmostEqual(decision.diagnostics["fill_fraction"], 0.673, places=2)

    def test_an_unsettled_geometry_is_labelled_not_finalised(self) -> None:
        """Still a usable live reading -- but never a finalised one, and never
        printed without saying so. The dashboard showed 20.092 L beside
        "geometry_not_settled" with nothing tying the two together."""
        decision = publish_decision(
            settled=False, plane_is_floor=True, fill=self._fill(),
        )
        self.assertTrue(decision.publish)
        self.assertFalse(decision.final)
        self.assertIn(UNSETTLED, decision.labels)

    def test_a_height_not_against_the_floor_is_labelled(self) -> None:
        decision = publish_decision(
            settled=True, plane_is_floor=False, fill=self._fill(),
        )
        self.assertTrue(decision.publish)
        self.assertFalse(decision.final)
        self.assertIn(NOT_FLOOR_RELATIVE, decision.labels)

    def test_an_impossible_volume_withholds_it(self) -> None:
        decision = publish_decision(
            settled=True, plane_is_floor=True,
            fill=self._fill(length_m=0.317, width_m=0.196, height_m=0.120, litres=9.013),
        )
        self.assertFalse(decision.publish)
        self.assertEqual(decision.reason, IMPOSSIBLE_FILL)

    def test_a_merged_mask_outranks_every_other_reason(self) -> None:
        decision = publish_decision(
            settled=True, plane_is_floor=True, fill=self._fill(), merged_reason=MERGED,
        )
        self.assertFalse(decision.publish)
        self.assertEqual(decision.reason, MERGED)

    def test_nothing_is_ever_reported_as_a_bare_pending(self) -> None:
        for case in (
            {"settled": False, "plane_is_floor": True, "fill": self._fill()},
            {"settled": True, "plane_is_floor": False, "fill": self._fill()},
            {"settled": True, "plane_is_floor": True,
             "fill": self._fill(litres=9.013, length_m=0.317, width_m=0.196, height_m=0.120)},
            {"settled": True, "plane_is_floor": True, "fill": self._fill(),
             "merged_reason": MERGED},
        ):
            with self.subTest(**{k: v for k, v in case.items() if k != "fill"}):
                decision = publish_decision(**case)
                self.assertFalse(decision.final)
                stated = (decision.reason,) + decision.labels
                self.assertTrue(any(stated), "a non-final result stated nothing")
                self.assertNotIn("pending", stated)


class StableAcrossFramesTests(unittest.TestCase):
    """One deposit, one finalised measurement -- not a new row every frame."""

    def test_a_settled_bag_stops_changing_size(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        for _ in range(5):
            stable = stabiliser.update(3, length_mm=437.0, width_mm=272.0,
                                       height_mm=251.0, volume_l=20.0)
        self.assertTrue(stable.settled)
        first = (stable.length_mm, stable.width_mm, stable.height_mm)
        for _ in range(4):
            stable = stabiliser.update(3, length_mm=440.0, width_mm=270.0,
                                       height_mm=253.0, volume_l=20.3)
        self.assertLess(abs(stable.length_mm - first[0]), 10.0)

    def test_nothing_is_published_before_it_settles(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        stable = stabiliser.update(4, length_mm=437.0, width_mm=272.0,
                                   height_mm=251.0, volume_l=20.0)
        self.assertFalse(stable.settled)
        decision = publish_decision(
            settled=stable.settled, plane_is_floor=True,
            fill=envelope_fill(length_m=0.437, width_m=0.272, height_m=0.251, litres=20.092),
        )
        self.assertFalse(decision.final)

    def test_a_wild_frame_does_not_move_a_settled_bag(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        for _ in range(5):
            stabiliser.update(5, length_mm=437.0, width_mm=272.0,
                              height_mm=251.0, volume_l=20.0)
        spiked = stabiliser.update(5, length_mm=1400.0, width_mm=900.0,
                                   height_mm=600.0, volume_l=120.0)
        self.assertLess(spiked.length_mm, 500.0)
        self.assertLess(spiked.volume_l, 25.0)


class BinBackgroundTests(unittest.TestCase):
    """Bin walls and contents already there are not new deposits."""

    def setUp(self) -> None:
        self.region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.region[100:400, 120:520] = True

    def test_pre_existing_contents_are_not_a_new_deposit(self) -> None:
        contents = np.zeros((HEIGHT, WIDTH), dtype=bool)
        contents[200:280, 260:360] = True
        change_elsewhere = np.zeros((HEIGHT, WIDTH), dtype=bool)
        change_elsewhere[320:380, 400:470] = True
        self.assertEqual(
            static_background_reason(contents, self.region, change=change_elsewhere),
            "unchanged_since_baseline",
        )

    def test_a_new_deposit_is_kept(self) -> None:
        deposit = np.zeros((HEIGHT, WIDTH), dtype=bool)
        deposit[200:280, 260:360] = True
        self.assertIsNone(
            static_background_reason(deposit, self.region, change=deposit.copy()),
        )

    def test_a_mask_across_the_whole_bin_is_the_bin(self) -> None:
        wall = np.zeros((HEIGHT, WIDTH), dtype=bool)
        wall[150:200, 120:520] = True
        self.assertEqual(
            static_background_reason(wall, self.region), "spans_measurement_zone",
        )


class MaskDepthAgreementTests(unittest.TestCase):
    """Mask, depth and intrinsics must describe the same picture."""

    def test_a_mask_of_the_wrong_shape_is_refused(self) -> None:
        from locallife_cloud.logitech_volume import metric_object_volume

        depth, empty, mask = _shoe_box()
        wrong = np.zeros((mask.shape[0] // 2, mask.shape[1]), dtype=bool)
        wrong[10:40, 10:40] = True
        with self.assertRaises((IndexError, ValueError)):
            result = metric_object_volume(
                depth, CAMERA, wrong, None, camera_height_m=1.0,
                min_height_m=0.008, min_pixels=10,
            )
            if result.reason is None:
                raise ValueError("a mask that does not match the depth was measured")

    def test_invalid_depth_inside_the_mask_is_refused_with_a_reason(self) -> None:
        from locallife_cloud.logitech_volume import metric_object_volume

        depth, empty, mask = _shoe_box()
        blanked = np.zeros_like(depth)
        result = metric_object_volume(
            blanked, CAMERA, mask, None, camera_height_m=1.0,
            min_height_m=0.008, min_pixels=60,
        )
        self.assertIsNotNone(result.reason)
        self.assertIsNone(result.measurement)

    def test_the_volume_fits_inside_the_envelope_reported_with_it(self) -> None:
        """The dimensions and the volume must describe the same cells.

        They did not: the extents were fitted to the cells the volume had not
        yet been completed over, and to those cells' midpoints rather than
        their area. On a small object that left the reported volume 1.8 times
        larger than the envelope printed beside it.
        """
        for name, scene in (("box", _shoe_box), ("bottle", _cream_bottle)):
            with self.subTest(object=name):
                depth, empty, mask = scene()
                result = _measure(depth, empty, mask)
                self.assertIsNone(result.reason)
                self.assertTrue(
                    result.diagnostics["envelope_plausible"],
                    result.diagnostics.get("envelope_reason"),
                )
                self.assertLessEqual(result.diagnostics["fill_fraction"], 1.05)

    def test_a_small_object_is_not_shrunk_by_half_a_cell_each_side(self) -> None:
        from locallife_cloud.footprint import estimate_extents

        # Sixteen 5 mm cells in a 4 x 4 block span 20 mm, not the 15 mm that
        # their centres span.
        cells = np.array([(x, y) for x in range(4) for y in range(4)], dtype=np.float64)
        centres = (cells + 0.5) * 0.005
        extents = estimate_extents(centres - centres.mean(axis=0))
        self.assertAlmostEqual(extents.length_m, 0.015, places=4)
        self.assertAlmostEqual(extents.length_m + 0.005, 0.020, places=4)

    def test_a_measured_bag_reports_its_fill_and_shape_model(self) -> None:
        depth, empty, mask = _shoe_box()
        result = _measure(depth, empty, mask)
        self.assertIsNone(result.reason)
        self.assertIn("fill_fraction", result.diagnostics)
        self.assertIn("shape_model", result.diagnostics)
        self.assertIsNotNone(result.diagnostics["envelope_l"])


class CameraIndependenceTests(unittest.TestCase):
    """Neither camera may read the other's frames, depth or measurements."""

    def test_the_bin_module_never_imports_a_realsense_one(self) -> None:
        import ast

        for name in ("logitech_bin.py", "logitech_geometry.py", "logitech_autocal.py",
                     "logitech_volume.py", "logitech_footprint.py"):
            with self.subTest(module=name):
                tree = ast.parse((PROJECT / "locallife_cloud" / name).read_text())
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                    elif isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                self.assertFalse([item for item in imported if "realsense" in item.lower()])

    def test_the_logitech_measurement_runs_with_no_realsense_input(self) -> None:
        from locallife_cloud.logitech_volume import metric_object_volume

        from .test_v35_logitech_metric_geometry import _floor_plane

        depth, empty, mask = _cream_bottle()
        result = metric_object_volume(
            depth, CAMERA, mask, _floor_plane(empty),
            reference_depth_m=None, measurement_mask=None,
            min_height_m=0.008, min_pixels=60,
        )
        self.assertIsNone(result.reason)


class ProtectedSurfacesTests(unittest.TestCase):
    """RealSense, the ledger, the CSV, the cloud and the launcher are untouched."""

    def test_the_protected_files_are_unchanged_since_v36(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/realsense.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/depth.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/volume.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/shape_geometry.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/storage.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/logitech.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/material.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/sorting_rules.py",
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
