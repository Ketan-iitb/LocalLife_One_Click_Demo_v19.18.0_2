"""V30: coordinate correctness, depth-mapping selection, analytic shapes, tooling.

Synthetic and offline only. Nothing here touches a camera.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.coordinates import (
    Letterbox,
    clip_to_region,
    frame_consistency,
    letterbox_params,
    restore_mask,
)
from locallife_cloud.depth_providers import candidate_providers
from locallife_cloud.logitech_volume import fit_plane_alignment, metric_object_volume, ray_plane_distance
from locallife_cloud.shape_geometry import CONE, HEMISPHERE, SPHERE, measure_shape
from locallife_cloud.types import CameraIntrinsics
from locallife_cloud.volume import fit_reference_plane

PROJECT = Path(__file__).resolve().parents[1]


def _letterbox(frame_shape: tuple[int, int], target: int) -> tuple[np.ndarray, Letterbox]:
    """A mask drawn in letterboxed model space, and the box that produced it."""
    import cv2

    box = letterbox_params(frame_shape, (target, target))
    canvas = np.zeros((target, target), dtype=np.uint8)
    object_pixels = np.zeros(frame_shape, dtype=np.uint8)
    object_pixels[60:180, 100:300] = 1          # 200 x 120 px object
    scaled = cv2.resize(object_pixels, (round(frame_shape[1] * box.scale), round(frame_shape[0] * box.scale)),
                        interpolation=cv2.INTER_NEAREST)
    canvas[box.pad_y:box.pad_y + scaled.shape[0], box.pad_x:box.pad_x + scaled.shape[1]] = scaled
    return canvas > 0, box


class CoordinateTests(unittest.TestCase):
    def test_letterbox_round_trip_returns_the_original_object(self) -> None:
        frame_shape = (480, 640)
        model_mask, box = _letterbox(frame_shape, 960)
        self.assertGreater(box.pad_y, 0)          # 4:3 into a square pads vertically
        self.assertEqual(box.pad_x, 0)
        restored = restore_mask(model_mask, frame_shape)
        rows, columns = np.nonzero(restored)
        self.assertAlmostEqual(int(rows.min()), 60, delta=2)
        self.assertAlmostEqual(int(rows.max()), 179, delta=2)
        self.assertAlmostEqual(int(columns.min()), 100, delta=2)
        self.assertAlmostEqual(int(columns.max()), 299, delta=2)
        # A plain stretch would have inflated the padded axis; this does not.
        self.assertAlmostEqual((rows.max() - rows.min()) / (columns.max() - columns.min()),
                               119 / 199, delta=0.05)

    def test_restoring_a_frame_sized_mask_changes_nothing(self) -> None:
        mask = np.zeros((480, 640), dtype=bool)
        mask[10:20, 30:40] = True
        self.assertTrue(np.array_equal(restore_mask(mask, (480, 640)), mask))

    def test_mismatched_coordinate_systems_are_named_not_guessed(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        logitech = CameraIntrinsics(fx=604, fy=604, ppx=320, ppy=240, width=640, height=480)
        realsense = CameraIntrinsics(fx=380, fy=380, ppx=320, ppy=240, width=640, height=480)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 100:200] = True
        depth = np.ones((480, 640), dtype=np.float32)
        region = np.zeros((480, 640), dtype=bool)
        region[50:400, 50:500] = True
        self.assertEqual(frame_consistency(frame, mask, depth, logitech, region=region,
                                           foreign_intrinsics=realsense), [])
        self.assertIn("mask_shape_(240, 320)_not_frame_(480, 640)",
                      frame_consistency(frame, np.zeros((240, 320), bool), depth, logitech))
        self.assertIn("depth_shape_(240, 320)_not_frame_(480, 640)",
                      frame_consistency(frame, mask, np.zeros((240, 320), np.float32), logitech))
        self.assertIn("intrinsics_for_1280x720_not_640x480", frame_consistency(
            frame, mask, depth, CameraIntrinsics(fx=900, fy=900, ppx=640, ppy=360, width=1280, height=720)))
        self.assertIn("intrinsics_belong_to_the_other_camera", frame_consistency(
            frame, mask, depth, realsense, foreign_intrinsics=realsense))
        outside = mask.copy()
        outside[0:10, 0:10] = True
        self.assertIn("mask_outside_calibrated_roi",
                      frame_consistency(frame, outside, depth, logitech, region=region))

    def test_a_measurement_mask_is_clipped_to_the_calibrated_region(self) -> None:
        region = np.zeros((100, 100), dtype=bool)
        region[20:80, 20:80] = True
        mask = np.ones((100, 100), dtype=bool)
        self.assertEqual(int(clip_to_region(mask, region).sum()), int(region.sum()))
        self.assertEqual(int(clip_to_region(mask, None).sum()), mask.size)


class DepthMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraIntrinsics(fx=500, fy=500, ppx=320, ppy=240, width=640, height=480)
        self.truth = ray_plane_distance((480, 640), self.camera, 1.4)
        self.region = np.zeros((480, 640), dtype=bool)
        self.region[60:420, 80:560] = True

    def test_a_linear_prediction_selects_the_linear_mapping(self) -> None:
        predicted = ((self.truth - 0.3) / 2.2).astype(np.float32)
        calibration, diagnostics = fit_plane_alignment(predicted, self.region, self.camera, 1.4)
        self.assertEqual(diagnostics["mapping"], "linear_depth")
        self.assertFalse(calibration.inverse)
        self.assertLess(diagnostics["held_out_rmse_linear_m"], diagnostics["held_out_rmse_inverse_m"])
        self.assertLess(float(np.abs(calibration.apply(predicted)[self.region] - self.truth[self.region]).max()),
                        0.005)

    def test_an_inverse_depth_prediction_selects_the_inverse_mapping(self) -> None:
        predicted = (1.0 / (0.55 * self.truth + 0.12)).astype(np.float32)
        calibration, diagnostics = fit_plane_alignment(predicted, self.region, self.camera, 1.4,
                                                       inverse=False)  # the hint is wrong on purpose
        self.assertEqual(diagnostics["mapping"], "inverse_depth")
        self.assertTrue(calibration.inverse)
        self.assertLess(diagnostics["held_out_rmse_inverse_m"], diagnostics["held_out_rmse_linear_m"])
        self.assertLess(float(np.abs(calibration.apply(predicted)[self.region] - self.truth[self.region]).max()),
                        0.01)


def _dome_points(semi_major: float, semi_minor: float, height: float, step: float = 0.002):
    x, y = np.meshgrid(np.arange(-semi_major, semi_major, step), np.arange(-semi_minor, semi_minor, step))
    points = np.column_stack((x.ravel(), y.ravel()))
    radial = (points[:, 0] / semi_major) ** 2 + (points[:, 1] / semi_minor) ** 2
    inside = radial <= 1
    return points[inside], height * np.sqrt(1 - radial[inside])


class AnalyticShapeTests(unittest.TestCase):
    def test_a_hemisphere_is_recognised_and_uses_two_thirds_pi_r_cubed(self) -> None:
        radius = 0.08
        points, heights = _dome_points(radius, radius, radius)
        integral = 2.0 / 3.0 * math.pi * radius ** 3 * 1000
        result = measure_shape(points, heights, mesh_volume_l=integral)
        self.assertEqual(result.geometry_method, HEMISPHERE)
        self.assertAlmostEqual(result.selected_volume_litres, integral, delta=0.12 * integral)

    def test_a_sphere_resting_on_the_plane_uses_four_thirds_pi_r_cubed(self) -> None:
        radius = 0.06
        points, heights = _dome_points(radius, radius, 2 * radius)
        integral = 4.0 / 3.0 * math.pi * radius ** 3 * 1000
        result = measure_shape(points, heights, mesh_volume_l=integral)
        self.assertEqual(result.geometry_method, SPHERE)
        self.assertAlmostEqual(result.selected_volume_litres, integral, delta=0.15 * integral)

    def test_a_cone_uses_a_third_pi_r_squared_h(self) -> None:
        radius, height = 0.07, 0.14
        step = 0.002
        x, y = np.meshgrid(np.arange(-radius, radius, step), np.arange(-radius, radius, step))
        points = np.column_stack((x.ravel(), y.ravel()))
        distance = np.hypot(points[:, 0], points[:, 1])
        inside = distance <= radius
        points, heights = points[inside], height * (1 - distance[inside] / radius)
        integral = math.pi * radius ** 2 * height / 3 * 1000
        result = measure_shape(points, heights, mesh_volume_l=integral)
        self.assertEqual(result.geometry_method, CONE)
        self.assertAlmostEqual(result.selected_volume_litres, integral, delta=0.15 * integral)

    def test_an_analytic_shape_that_disagrees_with_the_height_map_is_not_used(self) -> None:
        radius = 0.08
        points, heights = _dome_points(radius, radius, radius)
        # The integral says the object holds far more than a hemisphere would.
        result = measure_shape(points, heights, mesh_volume_l=5.0)
        self.assertNotIn(result.geometry_method, (SPHERE, HEMISPHERE, CONE))
        self.assertEqual(result.selected_volume_litres, 5.0)

    def test_a_flexible_label_keeps_the_occupied_volume_not_a_fitted_solid(self) -> None:
        radius = 0.12
        points, heights = _dome_points(radius, radius * 0.8, 0.15)
        integral = 2.0 / 3.0 * math.pi * radius * radius * 0.8 * 0.15 * 1000
        result = measure_shape(points, heights, mesh_volume_l=integral, label="filled waste bag")
        self.assertEqual(result.geometry_method, "flexible_or_unknown")
        self.assertEqual(result.volume_meaning, "current_external_occupied_volume")


class DepthProviderTests(unittest.TestCase):
    def test_every_candidate_reports_availability_with_a_reason(self) -> None:
        names = []
        for provider in candidate_providers():
            usable, reason = provider.available()
            self.assertIsInstance(usable, bool)
            self.assertTrue(reason)
            names.append(provider.name)
        self.assertIn("Depth-Anything-V2-Metric-Indoor-Small-hf", names)
        for expected in ("Metric3D-v2", "UniDepth-v2", "DepthPro"):
            self.assertIn(expected, names)

    def test_an_absent_model_is_reported_not_imported(self) -> None:
        provider = next(item for item in candidate_providers() if item.name == "Metric3D-v2")
        usable, reason = provider.available()
        if not usable:
            self.assertIn("not installed", reason)


class ToolingTests(unittest.TestCase):
    def _run(self, script: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(PROJECT / "scripts" / script), *arguments],
                              capture_output=True, text=True, cwd=PROJECT, timeout=300)

    def test_the_calibration_wizard_exposes_both_steps(self) -> None:
        result = self._run("calibrate_logitech_charuco.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("intrinsics", result.stdout)
        self.assertIn("plane", result.stdout)

    def test_the_accuracy_report_separates_rigid_from_irregular(self) -> None:
        import csv as csv_module

        from locallife_cloud.paired_events import COLUMNS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "comparison_measurements.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerow({"comparison_event_id": "e1", "measurement_id": "lg-1",
                                 "camera_source": "logitech", "object_type": "milk carton",
                                 "status": "accepted", "volume_liters": 1.1,
                                 "length_mm": 100, "width_mm": 100, "height_mm": 190,
                                 "processing_time_ms": 90})
                writer.writerow({"comparison_event_id": "e1", "measurement_id": "rs-1",
                                 "camera_source": "realsense", "object_type": "milk carton",
                                 "status": "accepted", "volume_liters": 0.95})
                writer.writerow({"comparison_event_id": "e2", "measurement_id": "lg-2",
                                 "camera_source": "logitech", "object_type": "folded textile",
                                 "status": "accepted", "volume_liters": 4.0})
            (root / "raw.jsonl").write_text(
                json.dumps({"measurement_id": "lg-1", "raw_volume_l": 2.2}) + "\n", encoding="utf-8")
            (root / "truth.json").write_text(json.dumps({
                "milk carton": {"litres": 0.95, "length_mm": 95, "width_mm": 95, "height_mm": 190,
                                "rigid": True, "small": False},
                "folded textile": {"litres": 3.0, "rigid": False},
            }), encoding="utf-8")
            result = self._run("logitech_benchmark.py", "--csv", str(csv_path), "--truth",
                               str(root / "truth.json"), "--raw-volumes", str(root / "raw.jsonl"),
                               "--output-dir", str(root / "out"))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(next((root / "out").glob("logitech_accuracy_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["rigid"]["logitech_calibrated"]["samples"], 1)
            self.assertEqual(report["rigid"]["logitech_raw"]["samples"], 1)
            self.assertEqual(report["irregular"]["logitech_calibrated"]["samples"], 1)
            # Rigid statistics never absorb the irregular object.
            self.assertAlmostEqual(report["rigid"]["logitech_calibrated"]["mae_litres"], 0.15, places=3)
            self.assertAlmostEqual(report["rigid"]["logitech_raw"]["mae_litres"], 1.25, places=3)
            self.assertIsNotNone(report["median_dimension_error_percent"])


class ProtectedFileTests(unittest.TestCase):
    def test_csv_history_and_cloud_files_are_untouched_on_this_branch(self) -> None:
        protected = [
            "LocalLife_Plug_and_Play_Local/locallife_cloud/event_log.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/paired_events.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/excel_export.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/ledger.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_ssh.py",
            "LocalLife_Plug_and_Play_Local/locallife_cloud/cloud_startup.py",
            "Start-LocalLife-Demo.ps1",
            "gpu.py",
        ]
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/Working_branch_v29_logitech_accuracy_cloud_fix", "--", *protected],
            capture_output=True, text=True, cwd=PROJECT.parent, timeout=120,
        )
        if result.returncode != 0:
            self.skipTest("git or the V29 ref is unavailable here")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
