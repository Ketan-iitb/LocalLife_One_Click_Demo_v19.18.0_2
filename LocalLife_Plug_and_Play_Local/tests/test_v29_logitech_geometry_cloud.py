"""V29: support-plane Logitech volume, and an optional host-key pin that cannot fail startup.

Synthetic geometry only -- these check the maths and the startup gate, not the
physical cameras.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.cloud_startup import STAGES
from locallife_cloud.logitech_volume import metric_object_volume
from locallife_cloud.types import CameraIntrinsics
from locallife_cloud.volume import fit_reference_plane

WIDTH = HEIGHT = 400
FX = FY = 500.0
CAMERA = CameraIntrinsics(fx=FX, fy=FY, ppx=WIDTH / 2, ppy=HEIGHT / 2, width=WIDTH, height=HEIGHT)
PLANE_DISTANCE_M = 1.5


def _rays() -> np.ndarray:
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    return np.stack(((columns - CAMERA.ppx) / FX, (rows - CAMERA.ppy) / FY, np.ones_like(columns, float)), axis=-1)


def upright_cylinder_scene(tilt_degrees: float, radius_m: float, height_m: float):
    """Depth map and mask of an upright cylinder on a plane seen at `tilt_degrees`.

    At 0 degrees the camera looks straight down; at 45 it sees the cylinder's
    side, which is exactly the real-hardware case that read four times high.
    """
    tilt = math.radians(tilt_degrees)
    normal = np.array([0.0, math.sin(tilt), math.cos(tilt)])
    normal /= np.linalg.norm(normal)
    rays = _rays()
    denominator = rays @ normal
    plane_t = np.where(denominator > 1e-9, PLANE_DISTANCE_M / np.where(denominator > 1e-9, denominator, 1), np.inf)
    floor_depth = (rays * plane_t[..., None])[..., 2]
    centre = rays[HEIGHT // 2, WIDTH // 2] * (PLANE_DISTANCE_M / (rays[HEIGHT // 2, WIDTH // 2] @ normal))
    axis = -normal
    flat = rays.reshape(-1, 3)
    best = np.full(flat.shape[0], np.inf)
    perpendicular = flat - np.outer(flat @ axis, axis)
    offset = -centre
    offset_perpendicular = offset - (offset @ axis) * axis
    a = np.einsum("ij,ij->i", perpendicular, perpendicular)
    b = 2 * (perpendicular @ offset_perpendicular)
    c = offset_perpendicular @ offset_perpendicular - radius_m ** 2
    discriminant = b * b - 4 * a * c
    usable = (discriminant > 0) & (a > 1e-12)
    root = np.sqrt(np.where(usable, discriminant, 0))
    for candidate in ((-b - root) / (2 * np.where(a > 1e-12, a, 1)), (-b + root) / (2 * np.where(a > 1e-12, a, 1))):
        along = (flat * candidate[:, None] - centre) @ axis
        hit = usable & (candidate > 0) & (along >= 0) & (along <= height_m) & (candidate < best)
        best = np.where(hit, candidate, best)
    top = centre + axis * height_m
    towards = flat @ axis
    cap_t = np.where(np.abs(towards) > 1e-9, ((top) @ axis) / np.where(np.abs(towards) > 1e-9, towards, 1), np.inf)
    cap_hit = (cap_t > 0) & (cap_t < best) & (np.linalg.norm(flat * cap_t[:, None] - top, axis=1) <= radius_m)
    best = np.where(cap_hit, cap_t, best)
    hit = np.isfinite(best)
    depth = floor_depth.reshape(-1).copy()
    depth[hit] = (flat[hit] * best[hit][:, None])[:, 2]
    return depth.reshape(HEIGHT, WIDTH), hit.reshape(HEIGHT, WIDTH), floor_depth


class SupportPlaneVolumeTests(unittest.TestCase):
    def test_obliquely_seen_bottle_is_no_longer_measured_from_its_silhouette(self) -> None:
        radius, height = 0.0425, 0.253  # the 1.5 L bottle from the hardware run
        truth_l = math.pi * radius ** 2 * height * 1000
        for tilt in (0.0, 25.0, 45.0, 60.0):
            with self.subTest(tilt=tilt):
                depth, mask, floor = upright_cylinder_scene(tilt, radius, height)
                plane = fit_reference_plane(floor.astype(np.float32), CAMERA)
                result = metric_object_volume(depth, CAMERA, mask, plane, min_height_m=0.01, min_pixels=50)
                self.assertIsNone(result.reason)
                self.assertAlmostEqual(result.measurement.liters, truth_l, delta=0.25 * truth_l)
                # The image-plane integral this replaced: silhouette area times
                # height, which is what produced ~4x on the real bottle.
                silhouette_l = (
                    int(mask.sum()) * float(np.median(depth[mask])) ** 2 / (FX * FY)
                    * result.diagnostics["height_p90_m"] * 1000
                )
                if tilt >= 25:
                    self.assertGreater(silhouette_l, 2.0 * truth_l)
                    self.assertLess(result.measurement.liters, 0.6 * silhouette_l)
                self.assertAlmostEqual(result.diagnostics["length_mm"], radius * 2000, delta=8)
                self.assertAlmostEqual(result.diagnostics["height_p90_m"] * 1000, height * 1000, delta=15)

    def test_occluded_far_side_is_completed_not_dropped(self) -> None:
        depth, mask, floor = upright_cylinder_scene(60.0, 0.0425, 0.253)
        plane = fit_reference_plane(floor.astype(np.float32), CAMERA)
        filled = metric_object_volume(depth, CAMERA, mask, plane, min_height_m=0.01, min_pixels=50)
        raw = metric_object_volume(depth, CAMERA, mask, plane, min_height_m=0.01, min_pixels=50,
                                   fill_occlusion=False)
        self.assertGreater(filled.diagnostics["occlusion_filled_cells"], 0)
        self.assertGreater(filled.measurement.liters, raw.measurement.liters)
        self.assertLess(filled.measurement.liters, 1.3 * math.pi * 0.0425 ** 2 * 0.253 * 1000)

    def test_a_calibrated_camera_height_stands_in_for_a_missing_plane(self) -> None:
        depth, mask, _ = upright_cylinder_scene(0.0, 0.0425, 0.20)
        without_plane = metric_object_volume(depth, CAMERA, mask, None, min_height_m=0.01, min_pixels=50)
        self.assertEqual(without_plane.reason, "no_support_plane")
        with_height = metric_object_volume(depth, CAMERA, mask, None, min_height_m=0.01, min_pixels=50,
                                           camera_height_m=PLANE_DISTANCE_M)
        self.assertIsNone(with_height.reason)
        self.assertEqual(with_height.diagnostics["plane_source"], "calibrated_camera_height")


class CloudStartupGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = (Path(__file__).resolve().parents[2] / "Start-LocalLife-Demo.ps1").read_text(encoding="utf-8")
        start = self.script.index("function Assert-CloudSshIdentity")
        self.identity = self.script[start:self.script.index("function Get-CloudSshOptions", start)]

    def test_only_a_failed_authenticated_probe_stops_startup(self) -> None:
        throws = [line.strip() for line in self.identity.splitlines() if line.strip().startswith("throw ")]
        self.assertEqual(len(throws), 1)
        self.assertIn("Cloud SSH probe failed", throws[0])

    def test_a_missing_published_key_continues_over_authenticated_ssh(self) -> None:
        self.assertIn("SSH HOST KEY NOT AUTOMATICALLY VERIFIED", self.identity)
        self.assertIn("startup continues", self.identity)
        self.assertIn("$script:CloudSshInteractive = $true", self.identity)
        # Security stays on: nothing is auto-accepted and no global relaxation.
        self.assertNotIn("StrictHostKeyChecking=no", self.script)
        self.assertNotIn("-batch", self.identity)

    def test_the_optional_pin_has_its_own_stage(self) -> None:
        self.assertIn("pinning_host_key", STAGES)
        self.assertIn("Set-CloudStage -Stage 'pinning_host_key'", self.identity)
        # Deployment is only claimed once the optional check is done.
        self.assertGreater(self.identity.count("Set-CloudStage -Stage 'checking_deployment'"), 1)

    def test_post_ssh_stages_are_published(self) -> None:
        for stage in ("uploading_bundle", "installing_dependencies", "starting_backend", "opening_tunnel"):
            with self.subTest(stage=stage):
                self.assertIn(stage, STAGES)
                self.assertIn(f"Set-CloudStage -Stage '{stage}'", self.script)


if __name__ == "__main__":
    unittest.main()


class MaskLeakageTests(unittest.TestCase):
    def test_detached_background_patch_is_dropped_before_integration(self) -> None:
        depth, mask, floor = upright_cylinder_scene(25.0, 0.0425, 0.20)
        plane = fit_reference_plane(floor.astype(np.float32), CAMERA)
        clean = metric_object_volume(depth, CAMERA, mask, plane, min_height_m=0.01, min_pixels=50)
        # A patch of "floor" 10 cm away that leaked into the mask, raised just
        # enough to pass the height gate -- a foot, or a bag's shadow edge.
        leaked_mask = mask.copy()
        leaked_depth = depth.copy()
        leaked_mask[60:110, 300:360] = True
        leaked_depth[60:110, 300:360] = floor[60:110, 300:360] - 0.05
        leaked = metric_object_volume(leaked_depth, CAMERA, leaked_mask, plane, min_height_m=0.01, min_pixels=50)
        self.assertGreater(leaked.diagnostics["background_cells_dropped"], 0)
        self.assertAlmostEqual(leaked.measurement.liters, clean.measurement.liters,
                               delta=0.15 * clean.measurement.liters)

    def test_volume_trace_reports_every_diagnostic_the_brief_asks_for(self) -> None:
        depth, mask, floor = upright_cylinder_scene(25.0, 0.0425, 0.20)
        plane = fit_reference_plane(floor.astype(np.float32), CAMERA)
        result = metric_object_volume(depth, CAMERA, mask, plane, min_height_m=0.01, min_pixels=50)
        for key in ("mask_pixels", "above_plane_pixels", "rejected_spike_pixels", "height_median_m",
                    "height_p90_m", "footprint_area_m2", "footprint_cells", "measured_cells",
                    "background_cells_dropped", "occlusion_filled_cells", "length_mm", "width_mm",
                    "object_depth_median_m", "plane_source", "raw_volume_l"):
            with self.subTest(key=key):
                self.assertIn(key, result.diagnostics)
