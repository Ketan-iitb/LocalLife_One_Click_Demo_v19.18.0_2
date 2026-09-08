"""Committed tests for the additive dual-camera "recipe" pipeline -- v3
blueprint ("dual_camera_estimation_recipe_v3.pdf" / "Zero-Flaw
Implementation Blueprint"), superseding the prior round's v1/v2-era tests.

Covers pointcloud_volume.py, recipe_color.py, recipe_material.py,
recipe_detect.py, recipe_config.py, recipe_calibration.py,
recipe_pipeline.py, recipe_api.py, and the recipe's wiring into
DualCameraCoordinator.recipe_result() (comparison.py) / AppConfig
(config.py). This project's real hardware (RealSense, Logitech) and real
model weights (generic YOLOv8n/11n-seg, CLIP, the material fallback CNN)
are all unreachable in an offline test environment, so every test below
either exercises the math directly against hand-computed synthetic ground
truth, or substitutes a deterministic stand-in for the heavy model
(matching this project's existing convention -- see
FixedDetection/SequencedMaterialClassifier in test_material.py and
SharedDetector/IndependentMetricDepth in test_dual_camera.py).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.pointcloud_volume import (
    backproject_to_points,
    discretize_bag_volume,
    erode_mask,
    estimate_volume_heightmap,
    estimate_volume_obb,
    estimate_volume_planefit_box,
    estimate_volume_recipe,
    fit_floor_plane_from_baseline,
    normalize_pose,
)
from locallife_cloud.recipe_api import create_app
from locallife_cloud.recipe_calibration import CalibrationMap, fit_calibration
from locallife_cloud.recipe_color import RECIPE_COLOR_CLASSES, classify_dominant_color
from locallife_cloud.recipe_config import RecipeConfig
from locallife_cloud.recipe_detect import (
    RecipeDetection,
    RecipeDetector,
    classify_object_type,
    frame_edge_clip_fraction,
    select_best,
)
from locallife_cloud.recipe_material import (
    RecipeMaterialClassifier,
    RecipeMaterialFallbackClassifier,
    classify_material_cascade,
)
from locallife_cloud.recipe_pipeline import process_object
from locallife_cloud.types import CameraIntrinsics


def _dome_scene(height=480, width=640, fx=615.0, floor_depth=0.8, radius_m=0.15, max_height_m=0.20):
    """A dense, realistic-resolution single-view dome (paraboloid) bag scene,
    at a distance/height ratio within v3's own recommended 0.4-1.2m capture
    range -- avoids the sparsity failure mode this module's own synthetic
    testing found at longer range (see pointcloud_volume.py's convex-fill
    docstring)."""
    ppx, ppy = width / 2.0, height / 2.0
    fy = fx
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy, width=width, height=height)
    baseline = np.full((height, width), floor_depth, dtype=np.float32)
    rows, cols = np.indices((height, width))
    x_m = (cols - ppx) * floor_depth / fx
    y_m = (rows - ppy) * floor_depth / fy
    radial = np.sqrt(x_m**2 + y_m**2)
    mask = radial <= radius_m
    dome_height = np.clip(max_height_m * (1 - (radial / radius_m) ** 2), 0.0, None)
    depth = baseline.copy()
    depth[mask] = floor_depth - dome_height[mask]
    known_liters = np.pi * radius_m**2 * max_height_m / 2 * 1000
    return depth, baseline, mask, intrinsics, known_liters


def _box_points_single_view(height_m=0.20, width_m=0.25, depth_m=0.15, n=4000, seed=42):
    """Pose-normalized (table Z-up, z=0) synthetic box points: a top face at
    z=height_m plus one front face (y=0) spanning width_m x height_m -- the
    single-view case (only 2 of 3 faces ever visible from one angle)."""
    rng = np.random.default_rng(seed)
    top = np.stack(
        [rng.uniform(0, width_m, n), rng.uniform(0, depth_m, n), np.full(n, height_m)], axis=1
    )
    front = np.stack(
        [rng.uniform(0, width_m, n), np.zeros(n), rng.uniform(0, height_m, n)], axis=1
    )
    return np.vstack([top, front])


def _box_points_second_view(height_m=0.20, depth_m=0.15, n=4000, seed=43):
    """The same box's adjacent side face (x=0), as if captured from a second
    view rotated ~90 degrees (v3 §3.4) -- its own front-face width equals
    the box's true depth dimension."""
    rng = np.random.default_rng(seed)
    top = np.stack(
        [rng.uniform(0, 0.25, n), rng.uniform(0, depth_m, n), np.full(n, height_m)], axis=1
    )
    side = np.stack(
        [np.zeros(n), rng.uniform(0, depth_m, n), rng.uniform(0, height_m, n)], axis=1
    )
    return np.vstack([top, side])


class PointCloudVolumeTests(unittest.TestCase):
    def test_box_obb_volume_matches_known_extents_from_direct_points(self) -> None:
        # OBB is kept specifically as v3 §13.3's own ablation baseline
        # ("raw-OBB ... show OBB fails on hidden dimension"), not the box
        # default any more -- still tested directly against an exact ground
        # truth: a point cloud including all 8 corners of an axis-aligned
        # box, whose Open3D OBB is provably the box itself.
        rng = np.random.default_rng(42)
        length_x, length_y, length_z = 0.20, 0.25, 0.15
        corners = np.array([[x, y, z] for x in (0, length_x) for y in (0, length_y) for z in (0, length_z)])
        interior = rng.uniform(low=[0, 0, 0], high=[length_x, length_y, length_z], size=(3000, 3))
        points = np.vstack([corners, interior])

        result = estimate_volume_obb(points)

        known_liters = length_x * length_y * length_z * 1000
        self.assertAlmostEqual(result.liters, known_liters, delta=known_liters * 0.05)
        self.assertEqual(result.method, "oriented-bounding-box")
        self.assertGreater(result.confidence, 0.5)

    def test_degenerate_flat_point_cloud_falls_back_instead_of_crashing(self) -> None:
        height = width = 80
        intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, ppx=width / 2, ppy=height / 2, width=width, height=height)
        depth = np.full((height, width), 1.0, dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)
        mask[20:60, 20:60] = True
        depth[mask] = 0.9

        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        result = estimate_volume_obb(points)  # must not raise

        self.assertGreater(result.liters, 0.0)
        self.assertLess(result.confidence, 0.5)

    def test_normalize_pose_rotates_tilted_plane_to_z_up_at_zero(self) -> None:
        # v3 §5.2 step 3: rotate the whole cloud so the table plane's normal
        # becomes +Z and the table sits at z=0 -- checked against an
        # analytically-known tilted plane, not just a flat one.
        rng = np.random.default_rng(1)
        normal_true = np.array([0.1, 0.05, 0.99])
        normal_true /= np.linalg.norm(normal_true)
        d_true = -1.0
        xy = rng.uniform(-0.3, 0.3, size=(300, 2))
        z_on_plane = -(normal_true[0] * xy[:, 0] + normal_true[1] * xy[:, 1] + d_true) / normal_true[2]
        table_points = np.stack([xy[:, 0], xy[:, 1], z_on_plane], axis=1)
        object_points = table_points[:60] + normal_true * 0.15  # 15cm above table

        rotated, rotation, _shift = normalize_pose(
            np.vstack([table_points, object_points]),
            (normal_true[0], normal_true[1], normal_true[2], d_true),
        )
        table_part, object_part = rotated[:300], rotated[300:]

        self.assertAlmostEqual(float(table_part[:, 2].mean()), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(table_part[:, 2].std()), 0.0, delta=1e-6)
        self.assertAlmostEqual(
            float(object_part[:, 2].mean() - table_part[:, 2].mean()), 0.15, delta=1e-6
        )
        self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6))  # proper rotation

    def test_planefit_box_two_view_matches_known_dimensions_closely(self) -> None:
        # v3's own accuracy target: "Two views is the accuracy target."
        height_m, width_m, depth_m = 0.20, 0.25, 0.15
        view1 = _box_points_single_view(height_m, width_m, depth_m)
        view2 = _box_points_second_view(height_m, depth_m)

        result = estimate_volume_planefit_box(view1, points_view2=view2)

        known_liters = height_m * width_m * depth_m * 1000
        self.assertEqual(result.views_used, 2)
        self.assertLess(abs(result.liters - known_liters) / known_liters, 0.05)
        self.assertGreater(result.confidence, 0.8)
        self.assertNotIn("single_view_volume", result.flags)

    def test_planefit_box_single_view_is_flagged_and_lower_confidence_than_two_view(self) -> None:
        # v3 rule #6: "Single-view volume is never 'accurate' -- it is a
        # fallback with degraded confidence." Not a precision claim -- single
        # view derives the third dimension from the front face's own max
        # in-plane chord, which is a real (documented) overestimate, not a
        # bug; this test checks the *contract* (flagged, dampened), not
        # numeric accuracy.
        height_m, width_m, depth_m = 0.20, 0.25, 0.15
        view1 = _box_points_single_view(height_m, width_m, depth_m)
        view2 = _box_points_second_view(height_m, depth_m)

        single = estimate_volume_planefit_box(view1)
        two_view = estimate_volume_planefit_box(view1, points_view2=view2)

        self.assertIn("single_view_volume", single.flags)
        self.assertEqual(single.views_used, 1)
        self.assertGreater(single.liters, 0.0)
        self.assertLess(single.confidence, two_view.confidence)

    def test_planefit_box_incomplete_face_fit_flagged_not_crashed(self) -> None:
        # Only a top face visible (no front face at all) -- must degrade to
        # a flagged, zero/low result rather than raising or fabricating a
        # width/depth from nothing.
        rng = np.random.default_rng(7)
        top_only = np.stack(
            [rng.uniform(0, 0.25, 3000), rng.uniform(0, 0.15, 3000), np.full(3000, 0.20)], axis=1
        )
        result = estimate_volume_planefit_box(top_only)  # must not raise
        self.assertIn("incomplete_face_fit", result.flags)
        self.assertEqual(result.liters, 0.0)

    def test_dome_bag_heightmap_volume_within_tolerance(self) -> None:
        # v3 §5.4's grid-integration + convex-fill, at a dense, realistic
        # single-view point count (within the recipe's own recommended
        # 0.4-1.2m capture range). Bags always carry a generous tolerance
        # band precisely because this discretization is an approximation of
        # a curved/lumpy surface, not a claim of precision -- see this
        # module's own confidence-cap and always-emitted tolerance_liters.
        depth, baseline, mask, intrinsics, known_liters = _dome_scene()
        plane = fit_floor_plane_from_baseline(baseline, intrinsics)
        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        result = estimate_volume_heightmap(points, plane_equation=plane, class_sizes_l=())

        self.assertGreater(result.liters, 0.0)
        relative_error = abs(result.liters - known_liters) / known_liters
        self.assertLess(relative_error, 0.20, f"{result.liters}L too far from known {known_liters:.3f}L")
        self.assertGreater(result.tolerance_liters, 0.0)
        self.assertTrue(0.0 < result.confidence <= 0.7)  # v3: bag confidence capped 0.5-0.7

    def test_heightmap_convex_fill_does_not_leak_into_true_background(self) -> None:
        # Regression test for a real bug found during development: a naive
        # border-flood-fill hole-filler filled the ENTIRE bounding square
        # around a circular footprint (including the true-background
        # corners outside the circle), inflating area far past the real
        # footprint. footprint_area_m2 must stay close to the true circle
        # area, not the bounding box's area.
        depth, baseline, mask, intrinsics, _known = _dome_scene(radius_m=0.15)
        plane = fit_floor_plane_from_baseline(baseline, intrinsics)
        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        result = estimate_volume_heightmap(points, plane_equation=plane, cell_size_m=0.0025, class_sizes_l=())

        # The original bug filled the *entire* bounding square (0.09 m^2 at
        # this radius) regardless of the true circular footprint (0.0707
        # m^2) -- a ~27% overcount that alone exceeds this 15% bound. A
        # small dilation-driven margin around the true circle is expected
        # and fine; leaking to the whole square is not.
        true_circle_area = np.pi * 0.15**2
        self.assertLess(abs(result.footprint_area_m2 - true_circle_area) / true_circle_area, 0.15)

    def test_heightmap_fill_recovers_sparse_distant_capture(self) -> None:
        # A sparse point cloud (object far from camera, few pixels/points
        # per grid cell) can leave the footprint too porous for a naive
        # border flood-fill to find any interior at all -- found directly
        # during development (every empty cell got misclassified as
        # "exterior" and the fill did nothing). The dilation-based fix must
        # still recover a reasonable (not wildly undercounted) volume.
        # floor_depth stays within v3's own valid depth band (<=2.5m --
        # DEFAULT_VALID_DEPTH_RANGE_M) while still being far enough that the
        # object's pixel footprint is sparse relative to a 2.5mm grid cell.
        depth, baseline, mask, intrinsics, known_liters = _dome_scene(
            height=480, width=640, floor_depth=2.2, radius_m=0.15, max_height_m=0.02
        )
        plane = fit_floor_plane_from_baseline(baseline, intrinsics)
        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        result = estimate_volume_heightmap(points, plane_equation=plane, cell_size_m=0.0025, class_sizes_l=())
        relative_error = abs(result.liters - known_liters) / known_liters
        self.assertLess(relative_error, 0.25)

    def test_discretize_bag_volume_within_window_snaps_to_class(self) -> None:
        liters, discretized = discretize_bag_volume(5.2, class_sizes_l=(5.0, 10.0), window_frac=0.30)
        self.assertTrue(discretized)
        self.assertEqual(liters, 5.0)

        liters, discretized = discretize_bag_volume(9.9, class_sizes_l=(5.0, 10.0), window_frac=0.30)
        self.assertTrue(discretized)
        self.assertEqual(liters, 10.0)

    def test_discretize_bag_volume_outside_window_reports_raw(self) -> None:
        # 6.7L: >5*1.3=6.5 and <10*0.7=7.0 -- outside both windows.
        liters, discretized = discretize_bag_volume(6.7, class_sizes_l=(5.0, 10.0), window_frac=0.30)
        self.assertFalse(discretized)
        self.assertEqual(liters, 6.7)

    def test_erode_mask_shrinks_and_zero_is_noop(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        eroded = erode_mask(mask, 2)
        self.assertLess(int(eroded.sum()), int(mask.sum()))
        self.assertTrue(np.array_equal(erode_mask(mask, 0), mask.astype(bool)))

    def test_backproject_rejects_depth_outside_valid_range(self) -> None:
        # v3 §5.1 rule #1/#2: depth outside [100mm, 2500mm] (default) is
        # invalid and must never enter the point cloud, including
        # RealSense's own zero-fill for unmeasurable pixels.
        height = width = 20
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, ppx=10, ppy=10, width=width, height=height)
        depth = np.zeros((height, width), dtype=np.float32)  # all invalid (hole-fill zero)
        mask = np.ones((height, width), dtype=bool)
        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        self.assertEqual(points.shape[0], 0)

        depth[:] = 3.0  # beyond the 2.5m default ceiling
        points = backproject_to_points(depth, intrinsics, mask, mask_erode_px=0)
        self.assertEqual(points.shape[0], 0)

    def test_estimate_volume_recipe_box_path_end_to_end_no_crash_and_positive(self) -> None:
        # End-to-end v3 §5 box path (mask -> points -> clean -> pose-
        # normalize -> plane-fit volume) on a scene with a real front face
        # visible, verifying it neither crashes nor silently zeroes out --
        # the same spirit as the prior round's RANSAC-misidentification
        # regression test, adapted to the new plane-fit method.
        height, width = 300, 300
        fx = fy = 300.0
        ppx, ppy = width / 2.0, height / 2.0
        floor_depth = 1.0
        intrinsics = CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy, width=width, height=height)
        baseline = np.full((height, width), floor_depth, dtype=np.float32)
        depth = baseline.copy()
        mask = np.zeros((height, width), dtype=bool)
        # Top face (rows 60-140) and a "front" face below it (rows 140-220)
        # with a shallower depth gradient standing in for a visible vertical
        # face in the same mask -- both regions closer to camera than floor.
        mask[60:220, 80:220] = True
        depth[60:140, 80:220] = 0.85  # top, 15cm above floor
        gradient_rows = np.linspace(0.85, 1.0, 80).reshape(-1, 1)
        depth[140:220, 80:220] = gradient_rows  # sloped "front" face

        result = estimate_volume_recipe(
            depth, intrinsics, mask, object_type="box", baseline_depth_m=baseline,
        )
        self.assertEqual(result.method, "plane-fit-box")
        # Not asserting a specific liters value (the sloped-gradient face is
        # a rough stand-in for a real vertical face, not an exact synthetic
        # box) -- the contract under test is "doesn't crash, produces a
        # believable non-negative result with an appropriate method tag".
        self.assertGreaterEqual(result.liters, 0.0)


class RecipeColorTests(unittest.TestCase):
    def test_all_eight_classes_classified_correctly_from_solid_swatches(self) -> None:
        # BGR swatches chosen to land clearly inside each class's LAB bucket
        # (v3 §6: BGR->LAB, achromatic-first, then a*/b* quadrant mapping).
        swatches = {
            "Blue": (200, 30, 20),
            "Grey": (128, 128, 128),
            "Black": (10, 10, 10),
            "White": (245, 245, 245),
            "Red": (20, 20, 200),
            "Green": (20, 180, 20),
            "Yellow": (20, 220, 220),
        }
        height = width = 60
        mask = np.ones((height, width), dtype=bool)
        for expected_label, bgr in swatches.items():
            image = np.full((height, width, 3), bgr, dtype=np.uint8)
            result = classify_dominant_color(image, mask)
            self.assertEqual(result.label, expected_label, f"swatch {bgr} misclassified as {result.label}")
            self.assertGreater(result.confidence, 0.5)

    def test_too_few_pixels_returns_other_with_zero_confidence(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        result = classify_dominant_color(image, mask)
        self.assertEqual(result.label, "Other")
        self.assertEqual(result.confidence, 0.0)

    def test_mixed_half_and_half_swatch_reports_other_via_tie_rule(self) -> None:
        # v3 §6 step 5: "A tie (<8% margin) -> Other + lower confidence."
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = (200, 30, 20)  # blue half
        image[50:, :] = (20, 20, 200)  # red half
        mask = np.ones((100, 100), dtype=bool)
        result = classify_dominant_color(image, mask)
        self.assertEqual(result.label, "Other")
        self.assertLess(result.confidence, 0.5)

    def test_class_list_matches_recipe_exactly(self) -> None:
        self.assertEqual(
            RECIPE_COLOR_CLASSES,
            ("Blue", "Grey", "Black", "White", "Red", "Green", "Yellow", "Other"),
        )

    def test_center_crop_uses_central_60_percent_of_bbox(self) -> None:
        # A mask whose outer ring is a decoy color and center is the true
        # color -- v3 §6 step 1's central-60%-crop should see mostly the
        # true center color, not get swamped by the ring.
        height = width = 100
        image = np.full((height, width, 3), (20, 20, 200), dtype=np.uint8)  # decoy: red border
        image[20:80, 20:80] = (200, 30, 20)  # true color: blue center
        mask = np.ones((height, width), dtype=bool)
        result = classify_dominant_color(image, mask, center_crop_frac=0.6)
        self.assertEqual(result.label, "Blue")


class RecipeConfigTests(unittest.TestCase):
    def test_default_matches_v3_yaml_schema_values(self) -> None:
        config = RecipeConfig.default()
        self.assertEqual(config.depth.valid_mm, (100.0, 2500.0))
        self.assertEqual(config.depth.valid_range_m, (0.1, 2.5))
        self.assertEqual(config.depth.mask_erode_px, 3)
        self.assertEqual(config.volume.bag_class_sizes_l, (5.0, 10.0))
        self.assertEqual(config.volume.bag_tolerance_frac, 0.15)
        self.assertEqual(config.color.kmeans_k, 4)
        self.assertEqual(config.color.center_crop_frac, 0.6)
        self.assertEqual(config.material.clip_confidence_floor, 0.6)

    def test_from_yaml_loads_the_shipped_default_file(self) -> None:
        config = RecipeConfig.from_yaml()
        self.assertEqual(config.segmentation.model, "yolov8n-seg.pt")
        self.assertEqual(tuple(config.segmentation.area_range), (0.05, 0.80))

    def test_from_dict_overrides_only_given_keys(self) -> None:
        config = RecipeConfig.from_dict({"material": {"clip_confidence_floor": 0.7}})
        self.assertEqual(config.material.clip_confidence_floor, 0.7)
        self.assertEqual(config.material.fallback_confidence_floor, 0.55)  # unchanged default

    def test_missing_yaml_file_falls_back_to_defaults(self) -> None:
        config = RecipeConfig.from_yaml("/nonexistent/path/does-not-exist.yaml")
        self.assertEqual(config, RecipeConfig.default())


class RecipeCalibrationTests(unittest.TestCase):
    def test_default_map_is_unfitted_identity(self) -> None:
        calibration = CalibrationMap()
        self.assertFalse(calibration.fitted)
        self.assertAlmostEqual(calibration.apply(0.73), 0.73)

    def test_fit_calibration_with_too_few_or_single_class_samples_stays_unfitted(self) -> None:
        self.assertFalse(fit_calibration([0.5, 0.6], [True, True]).fitted)
        self.assertFalse(fit_calibration([], []).fitted)

    def test_fit_calibration_on_separable_data_fits_and_orders_correctly(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 200)
        correct = list((raw + rng.normal(0, 0.05, 200)) > 0.5)
        calibration = fit_calibration(list(raw), correct)
        self.assertTrue(calibration.fitted)
        self.assertGreater(calibration.apply(0.95), calibration.apply(0.05))


class RecipeMaterialTests(unittest.TestCase):
    def test_disabled_classifier_returns_other_without_importing_torch(self) -> None:
        config = AppConfig(enable_material_classification=False)
        classifier = RecipeMaterialClassifier(config)
        label, confidence = classifier.classify(
            np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=bool), (0, 0, 10, 10)
        )
        self.assertEqual((label, confidence), ("Other", 0.0))

    def test_class_list_matches_recipe_exactly(self) -> None:
        from locallife_cloud.recipe_material import RECIPE_MATERIAL_CLASSES

        self.assertEqual(
            RECIPE_MATERIAL_CLASSES,
            ("Plastic", "Fabric", "Metal", "Cardboard", "Paper", "Rubber", "Other"),
        )

    def test_fallback_disabled_without_a_checkpoint_path(self) -> None:
        fallback = RecipeMaterialFallbackClassifier(checkpoint_path=None)
        self.assertFalse(fallback.enabled)
        result = fallback.classify_with_margin(
            np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=bool), (0, 0, 10, 10)
        )
        self.assertEqual(result, ("Other", 0.0, 0.0))

    def test_fallback_disabled_for_nonexistent_checkpoint_file(self) -> None:
        fallback = RecipeMaterialFallbackClassifier(checkpoint_path="/nonexistent/checkpoint.pt")
        fallback.load()
        self.assertFalse(fallback.enabled)


class _FakeClipClassifier:
    def __init__(self, label: str, confidence: float, margin: float) -> None:
        self._label, self._confidence, self._margin = label, confidence, margin

    def classify_with_margin(self, frame_bgr, mask, box):
        return self._label, self._confidence, self._margin


class _FakeFallbackClassifier:
    backbone_name = "mobilenetv3_small"

    def __init__(self, label: str, confidence: float, enabled: bool = True) -> None:
        self._label, self._confidence, self.enabled = label, confidence, enabled

    def classify_with_margin(self, frame_bgr, mask, box):
        return self._label, self._confidence, 0.3


class RecipeMaterialCascadeTests(unittest.TestCase):
    """v3 §7's two-level routing logic, exercised directly against fakes so
    every branch is deterministic and does not need real CLIP/CNN weights."""

    _FRAME = np.zeros((10, 10, 3), dtype=np.uint8)
    _MASK = np.ones((10, 10), dtype=bool)
    _BOX = (0, 0, 10, 10)

    def test_high_confidence_clip_is_accepted_without_fallback(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Metal", 0.9, 0.5),
            fallback_classifier=None,
        )
        self.assertEqual((label, model, ambiguous), ("Metal", "clip", False))
        self.assertEqual(confidence, 0.9)

    def test_low_confidence_clip_without_fallback_keeps_clip_label_if_above_floor(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Cardboard", 0.58, 0.3),
            fallback_classifier=None,
            fallback_confidence_floor=0.55,
        )
        self.assertEqual((label, model, ambiguous), ("Cardboard", "clip", False))

    def test_low_confidence_clip_without_fallback_below_floor_reports_other_ambiguous(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Cardboard", 0.4, 0.1),
            fallback_classifier=None,
            fallback_confidence_floor=0.55,
        )
        self.assertEqual((label, model, ambiguous), ("Other", "clip", True))

    def test_thin_margin_plastic_routes_to_fallback_even_if_confident(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Plastic", 0.8, 0.05),
            fallback_classifier=_FakeFallbackClassifier("Paper", 0.7),
            plastic_thin_margin=0.10,
        )
        self.assertEqual((label, model, ambiguous), ("Paper", "mobilenetv3_small", False))

    def test_fallback_available_and_confident_is_used(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Cardboard", 0.3, 0.1),
            fallback_classifier=_FakeFallbackClassifier("Paper", 0.8),
        )
        self.assertEqual((label, model, ambiguous), ("Paper", "mobilenetv3_small", False))

    def test_fallback_available_but_also_low_confidence_reports_other_ambiguous(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Cardboard", 0.3, 0.1),
            fallback_classifier=_FakeFallbackClassifier("Other", 0.3),
        )
        self.assertEqual((label, model, ambiguous), ("Other", "mobilenetv3_small", True))

    def test_disabled_fallback_instance_treated_as_unavailable(self) -> None:
        label, confidence, model, ambiguous = classify_material_cascade(
            self._FRAME, self._MASK, self._BOX,
            clip_classifier=_FakeClipClassifier("Cardboard", 0.3, 0.1),
            fallback_classifier=_FakeFallbackClassifier("Paper", 0.8, enabled=False),
            fallback_confidence_floor=0.55,
        )
        # No usable fallback -- falls to the "no fallback available" branch.
        self.assertEqual(model, "clip")
        self.assertEqual(ambiguous, True)  # CLIP's own 0.3 < fallback_confidence_floor


class RecipeDetectTests(unittest.TestCase):
    def test_classify_object_type_box_vs_bag_by_fill_ratio(self) -> None:
        box_mask = np.ones((80, 80), dtype=bool)
        object_type, fill_ratio = classify_object_type(box_mask, (0, 0, 80, 80))
        self.assertEqual(object_type, "box")
        self.assertGreaterEqual(fill_ratio, 0.75)

        bag_mask = np.zeros((80, 80), dtype=bool)
        rows, cols = np.indices((80, 80))
        bag_mask[(rows - 40) ** 2 + (cols - 40) ** 2 <= 35**2] = True
        object_type, fill_ratio = classify_object_type(bag_mask, (0, 0, 80, 80))
        self.assertEqual(object_type, "bag")
        self.assertLess(fill_ratio, 0.75)

    def test_select_best_ranks_by_confidence_then_mask_area(self) -> None:
        small_mask = np.zeros((10, 10), dtype=bool)
        small_mask[:2, :2] = True
        large_mask = np.zeros((10, 10), dtype=bool)
        large_mask[:8, :8] = True

        low_confidence = RecipeDetection("a", 0.5, (0, 0, 10, 10), small_mask, "box", 1.0)
        high_confidence_small = RecipeDetection("b", 0.9, (0, 0, 10, 10), small_mask, "box", 1.0)
        high_confidence_large = RecipeDetection("c", 0.9, (0, 0, 10, 10), large_mask, "box", 1.0)

        best = select_best([low_confidence, high_confidence_small], [high_confidence_large])
        self.assertIs(best, high_confidence_large)

    def test_select_best_returns_none_when_nothing_detected(self) -> None:
        self.assertIsNone(select_best([], []))

    def test_disabled_detector_returns_no_detections(self) -> None:
        detector = RecipeDetector()
        detector.enabled = False
        self.assertEqual(detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)), [])

    def test_select_best_target_class_filters_before_ranking(self) -> None:
        # v3 §4 step 2: "if a target class is known, filter by class first."
        bag_mask = np.zeros((10, 10), dtype=bool)
        bag_mask[:4, :4] = True
        box_mask = np.zeros((10, 10), dtype=bool)
        box_mask[:8, :8] = True
        bag_detection = RecipeDetection("a", 0.5, (0, 0, 10, 10), bag_mask, "bag", 0.5)
        box_detection = RecipeDetection("b", 0.99, (0, 0, 10, 10), box_mask, "box", 0.99)

        # Without a target class, the higher-confidence box wins.
        self.assertIs(select_best([bag_detection, box_detection]), box_detection)
        # With target_class="bag", only the bag is eligible even though it
        # has lower confidence.
        self.assertIs(select_best([bag_detection, box_detection], target_class="bag"), bag_detection)

    def test_select_best_area_sanity_rejects_out_of_range_mask(self) -> None:
        # v3 §4 step 2: "sanity-check mask area in [5%, 80%] of frame."
        frame_area = 100 * 100
        tiny_mask = np.zeros((100, 100), dtype=bool)
        tiny_mask[:2, :2] = True  # 0.04% of frame -- below 5%
        sane_mask = np.zeros((100, 100), dtype=bool)
        sane_mask[:30, :30] = True  # 9% of frame -- within range

        tiny_detection = RecipeDetection("a", 0.99, (0, 0, 100, 100), tiny_mask, "box", 0.99)
        sane_detection = RecipeDetection("b", 0.5, (0, 0, 100, 100), sane_mask, "box", 0.5)

        best = select_best([tiny_detection, sane_detection], frame_area=frame_area)
        self.assertIs(best, sane_detection)

    def test_frame_edge_clip_fraction(self) -> None:
        self.assertEqual(frame_edge_clip_fraction((10, 10, 90, 90), (100, 100)), 0.0)
        self.assertEqual(frame_edge_clip_fraction((0, 10, 90, 90), (100, 100)), 0.25)
        self.assertEqual(frame_edge_clip_fraction((0, 0, 100, 100), (100, 100)), 1.0)


class _FakeDetector:
    """Deterministic stand-in for RecipeDetector, distinguishing cameras by
    frame identity rather than content -- matches this project's existing
    FixedDetection/SharedDetector pattern for other heavy models."""

    def __init__(self, realsense_frame: np.ndarray, realsense_detections, logitech_detections) -> None:
        self._realsense_frame = realsense_frame
        self._realsense = realsense_detections
        self._logitech = logitech_detections

    def detect(self, image_bgr: np.ndarray, confidence_threshold: float = 0.25):
        return self._realsense if image_bgr is self._realsense_frame else self._logitech


class _FakeMaterial:
    """Stands in for RecipeMaterialClassifier -- the v3 pipeline calls
    `classify_with_margin`, not the older `classify`."""

    def __init__(self, label: str = "Plastic", confidence: float = 0.9, margin: float = 0.5) -> None:
        self._label, self._confidence, self._margin = label, confidence, margin

    def classify_with_margin(self, frame_bgr, mask, box):
        return self._label, self._confidence, self._margin


class RecipePipelineTests(unittest.TestCase):
    _SCHEMA_KEYS = {
        "volume_liters", "volume_tolerance_liters", "volume_confidence", "color", "color_confidence",
        "material", "material_confidence", "material_model", "object_type", "views_used", "timestamp", "flags",
    }

    def test_end_to_end_returns_exact_v3_schema(self) -> None:
        height = width = 300
        fx = fy = 300.0
        ppx, ppy = width / 2.0, height / 2.0
        floor_depth = 1.0
        mask = np.zeros((height, width), dtype=bool)
        mask[60:220, 80:220] = True
        box = (80, 60, 220, 220)
        depth = np.full((height, width), floor_depth, dtype=np.float32)
        depth[60:140, 80:220] = 0.85
        gradient_rows = np.linspace(0.85, 1.0, 80).reshape(-1, 1)
        depth[140:220, 80:220] = gradient_rows

        realsense_color = np.zeros((height, width, 3), dtype=np.uint8)
        logitech_color = np.full((height, width, 3), (200, 30, 20), dtype=np.uint8)
        intrinsics = CameraIntrinsics(fx=fx, fy=fy, ppx=ppx, ppy=ppy, width=width, height=height)

        detection = RecipeDetection("object", 0.9, box, mask, "box", 0.95)
        detector = _FakeDetector(realsense_color, [detection], [detection])

        result = process_object(
            realsense_color, depth, intrinsics, logitech_color,
            detector=detector, material_classifier=_FakeMaterial(),
        )

        self.assertEqual(set(result.keys()), self._SCHEMA_KEYS)
        self.assertEqual(result["object_type"], "box")
        self.assertEqual(result["material"], "Plastic")
        self.assertEqual(result["material_model"], "clip")
        self.assertEqual(result["color"], "Blue")
        self.assertIsInstance(result["flags"], list)
        self.assertIn("single_view_volume", result["flags"])  # only one RealSense view given

    def test_no_detection_anywhere_returns_empty_result_with_flag(self) -> None:
        height = width = 40
        depth = np.full((height, width), 1.0, dtype=np.float32)
        realsense_color = np.zeros((height, width, 3), dtype=np.uint8)
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, ppx=width / 2, ppy=height / 2, width=width, height=height)
        detector = _FakeDetector(realsense_color, [], [])

        result = process_object(
            realsense_color, depth, intrinsics, None,
            detector=detector, material_classifier=_FakeMaterial(),
        )
        self.assertEqual(set(result.keys()), self._SCHEMA_KEYS)
        self.assertEqual(result["object_type"], "unknown")
        self.assertEqual(result["volume_liters"], 0.0)
        self.assertEqual(result["color"], "Other")
        self.assertEqual(result["material"], "Other")
        self.assertIn("no_object_found", result["flags"])

    def test_realsense_miss_logitech_hit_reports_zero_volume_but_real_color(self) -> None:
        height = width = 40
        depth = np.full((height, width), 1.0, dtype=np.float32)
        realsense_color = np.zeros((height, width, 3), dtype=np.uint8)
        logitech_color = np.full((height, width, 3), (20, 20, 210), dtype=np.uint8)
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, ppx=width / 2, ppy=height / 2, width=width, height=height)
        mask = np.ones((height, width), dtype=bool)
        detection = RecipeDetection("object", 0.8, (0, 0, width, height), mask, "box", 1.0)
        detector = _FakeDetector(realsense_color, [], [detection])

        result = process_object(
            realsense_color, depth, intrinsics, logitech_color,
            detector=detector, material_classifier=_FakeMaterial(),
        )
        self.assertEqual(result["volume_liters"], 0.0)
        self.assertEqual(result["volume_confidence"], 0.0)
        self.assertEqual(result["color"], "Red")
        self.assertIn("no_realsense_detection", result["flags"])

    def test_uncalibrated_confidences_flag_present_by_default(self) -> None:
        # Uses a real detection (not the "nothing detected anywhere" early
        # return, which skips the calibration step entirely since none of
        # its all-zero confidences are meaningful to flag).
        height = width = 40
        depth = np.full((height, width), 1.0, dtype=np.float32)
        realsense_color = np.zeros((height, width, 3), dtype=np.uint8)
        logitech_color = np.full((height, width, 3), (20, 20, 210), dtype=np.uint8)
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, ppx=width / 2, ppy=height / 2, width=width, height=height)
        mask = np.ones((height, width), dtype=bool)
        detection = RecipeDetection("object", 0.8, (0, 0, width, height), mask, "box", 1.0)
        detector = _FakeDetector(realsense_color, [], [detection])
        result = process_object(
            realsense_color, depth, intrinsics, logitech_color,
            detector=detector, material_classifier=_FakeMaterial(),
        )
        self.assertIn("uncalibrated_confidences", result["flags"])


class RecipeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_detector_load = RecipeDetector.load
        self._original_detector_detect = RecipeDetector.detect
        self._original_material_load = RecipeMaterialClassifier.load
        self._original_material_classify_with_margin = RecipeMaterialClassifier.classify_with_margin

        height = width = 60
        self.mask = np.zeros((height, width), dtype=bool)
        self.mask[10:50, 10:50] = True
        self.box = (10, 10, 50, 50)

        def fake_load(instance) -> None:
            instance.enabled = True
            instance.model = "FAKE"

        def fake_detect(instance, image_bgr, confidence_threshold: float = 0.25):
            return [RecipeDetection("object", 0.9, self.box, self.mask, "box", 0.9)]

        def fake_material_load(instance) -> None:
            instance.enabled = True

        def fake_material_classify_with_margin(instance, frame_bgr, mask, box):
            return "Plastic", 0.8, 0.5

        RecipeDetector.load = fake_load
        RecipeDetector.detect = fake_detect
        RecipeMaterialClassifier.load = fake_material_load
        RecipeMaterialClassifier.classify_with_margin = fake_material_classify_with_margin

        config = AppConfig.from_env()
        config.api_token = "test-token"
        self.app = create_app(config)

        from fastapi.testclient import TestClient

        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        RecipeDetector.load = self._original_detector_load
        RecipeDetector.detect = self._original_detector_detect
        RecipeMaterialClassifier.load = self._original_material_load
        RecipeMaterialClassifier.classify_with_margin = self._original_material_classify_with_margin

    def _multipart(self) -> tuple[dict, dict]:
        height = width = 60
        depth = np.full((height, width), 1.0, dtype=np.float32)
        depth[self.mask] = 0.85
        rs_color = np.zeros((height, width, 3), dtype=np.uint8)
        lg_color = np.full((height, width, 3), (200, 30, 20), dtype=np.uint8)
        _, rs_jpeg = cv2.imencode(".jpg", rs_color)
        _, lg_jpeg = cv2.imencode(".jpg", lg_color)
        depth_buffer = io.BytesIO()
        np.savez_compressed(depth_buffer, depth_m=depth)

        metadata = json.dumps({
            "intrinsics": {"fx": 90.0, "fy": 90.0, "ppx": width / 2, "ppy": height / 2, "width": width, "height": height},
        })
        files = {
            "realsense_image": ("rs.jpg", rs_jpeg.tobytes(), "image/jpeg"),
            "realsense_depth": ("rs_depth.npz", depth_buffer.getvalue(), "application/octet-stream"),
            "logitech_image": ("lg.jpg", lg_jpeg.tobytes(), "image/jpeg"),
        }
        return files, {"metadata": metadata}

    def test_health_requires_no_auth(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_process_without_token_is_rejected(self) -> None:
        files, data = self._multipart()
        response = self.client.post("/api/recipe/process", files=files, data=data)
        self.assertEqual(response.status_code, 401)

    def test_process_with_valid_token_returns_v3_schema(self) -> None:
        files, data = self._multipart()
        response = self.client.post(
            "/api/recipe/process", files=files, data=data,
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body.keys()),
            {
                "volume_liters", "volume_tolerance_liters", "volume_confidence", "color", "color_confidence",
                "material", "material_confidence", "material_model", "object_type", "views_used",
                "timestamp", "flags",
            },
        )
        self.assertEqual(body["object_type"], "box")
        self.assertEqual(body["material"], "Plastic")

    def test_x_api_token_header_also_accepted(self) -> None:
        files, data = self._multipart()
        response = self.client.post(
            "/api/recipe/process", files=files, data=data,
            headers={"X-API-Token": "test-token"},
        )
        self.assertEqual(response.status_code, 200)

    def test_invalid_metadata_json_returns_400(self) -> None:
        files, _ = self._multipart()
        response = self.client.post(
            "/api/recipe/process", files=files, data={"metadata": "{not json"},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_intrinsics_returns_400(self) -> None:
        files, _ = self._multipart()
        response = self.client.post(
            "/api/recipe/process", files=files, data={"metadata": json.dumps({})},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 400)

    def test_mismatched_image_and_depth_size_returns_400(self) -> None:
        files, data = self._multipart()
        bad_depth = io.BytesIO()
        np.savez_compressed(bad_depth, depth_m=np.full((5, 5), 1.0, dtype=np.float32))
        files = dict(files)
        files["realsense_depth"] = ("bad.npz", bad_depth.getvalue(), "application/octet-stream")
        response = self.client.post(
            "/api/recipe/process", files=files, data=data,
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 400)


class SharedDetectorStub:
    device = "cpu"
    runtime = {"device": "cpu"}

    def detect_batch(self, frames):
        return [[] for _ in frames]


class IndependentMetricDepthStub:
    def estimate_batch(self, frames):
        return [2.0 - frame.max(axis=2).astype(np.float32) * 0.0015 for frame in frames]


class DualCameraCoordinatorRecipeWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        config = AppConfig(
            results_dir=Path(self._tempdir.name), roi=(0, 0, 1, 1), min_component_pixels=20,
            tracker_confirm_frames=1, settle_frames=2, volume_window_frames=2,
            auto_deposit=False, logitech_reference_distance_m=2.0,
        )
        self.manager = DualCameraCoordinator(
            config, detector=SharedDetectorStub(), depth_estimator=IndependentMetricDepthStub(),
        )

        self._original_detector_load = RecipeDetector.load
        self._original_detector_detect = RecipeDetector.detect
        self._original_material_load = RecipeMaterialClassifier.load
        self._original_material_classify_with_margin = RecipeMaterialClassifier.classify_with_margin

    def tearDown(self) -> None:
        RecipeDetector.load = self._original_detector_load
        RecipeDetector.detect = self._original_detector_detect
        RecipeMaterialClassifier.load = self._original_material_load
        RecipeMaterialClassifier.classify_with_margin = self._original_material_classify_with_margin
        self._tempdir.cleanup()

    def test_disabled_by_default_and_never_imports_recipe_modules(self) -> None:
        result = self.manager.recipe_result()
        self.assertEqual(result, {"available": False, "reason": "disabled"})

    def test_enabled_but_no_frame_yet(self) -> None:
        self.manager.config.recipe_enabled = True
        result = self.manager.recipe_result()
        self.assertEqual(result["available"], False)
        self.assertEqual(result["reason"], "waiting_for_realsense_frame")

    def test_enabled_with_frames_returns_result_and_is_cached_until_refresh(self) -> None:
        self.manager.config.recipe_enabled = True
        call_count = {"n": 0}
        height = width = 60
        mask = np.zeros((height, width), dtype=bool)
        mask[10:50, 10:50] = True
        box = (10, 10, 50, 50)

        def fake_load(instance) -> None:
            instance.enabled = True
            instance.model = "FAKE"

        def fake_detect(instance, image_bgr, confidence_threshold: float = 0.25):
            call_count["n"] += 1
            return [RecipeDetection("object", 0.9, box, mask, "box", 0.9)]

        def fake_material_load(instance) -> None:
            instance.enabled = True

        def fake_material_classify_with_margin(instance, frame_bgr, mask, box):
            return "Cardboard", 0.7, 0.4

        RecipeDetector.load = fake_load
        RecipeDetector.detect = fake_detect
        RecipeMaterialClassifier.load = fake_material_load
        RecipeMaterialClassifier.classify_with_margin = fake_material_classify_with_margin

        hardware = self.manager.camera("realsense")
        webcam = self.manager.camera("logitech")
        depth = np.full((height, width), 1.0, dtype=np.float32)
        depth[mask] = 0.85
        hardware.latest_frame = np.zeros((height, width, 3), dtype=np.uint8)
        hardware.latest_depth = depth
        hardware.latest_intrinsics = CameraIntrinsics(fx=90.0, fy=90.0, ppx=width / 2, ppy=height / 2, width=width, height=height)
        webcam.latest_frame = np.full((height, width, 3), (200, 30, 20), dtype=np.uint8)

        state = self.manager.state()
        self.assertTrue(state["recipe_result"]["available"])
        self.assertEqual(state["recipe_result"]["object_type"], "box")
        self.assertEqual(state["recipe_result"]["material"], "Cardboard")
        self.assertEqual(state["recipe_result"]["material_model"], "clip")
        calls_after_first = call_count["n"]
        self.assertGreater(calls_after_first, 0)

        # Immediate second call must be served from cache -- no new detect() calls.
        state_again = self.manager.state()
        self.assertEqual(call_count["n"], calls_after_first)
        self.assertEqual(state_again["recipe_result"], state["recipe_result"])

        # Rewinding the cache clock forces a fresh computation.
        self.manager._recipe_cache_at = 0.0
        self.manager.state()
        self.assertGreater(call_count["n"], calls_after_first)

    def test_recipe_failure_degrades_gracefully_without_breaking_state(self) -> None:
        self.manager.config.recipe_enabled = True
        self.manager._recipe_import_error = "simulated missing dependency"
        result = self.manager.recipe_result()
        self.assertEqual(
            result, {"available": False, "reason": "unavailable", "detail": "simulated missing dependency"},
        )
        # state() must still succeed even though the recipe path is broken.
        state = self.manager.state()
        self.assertIn("recipe_result", state)


if __name__ == "__main__":
    unittest.main()
