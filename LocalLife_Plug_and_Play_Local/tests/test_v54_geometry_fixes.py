"""V54 geometry fixes in the LIVE measurement path (volume.estimate_volume via AppConfig.volume_geometry,
VisionPipeline._apply_box_cuboid / calibrate_known_volume / volume history).

Ground truth comes from the scene definitions (a box's L x W x H; exact ray-box hits from
test_v49_box_fix._scene), never from an area formula of the estimator. Synthetic only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locallife_cloud.box_templates import BoxTemplate  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.pipeline import VisionPipeline, _integration_method  # noqa: E402
from locallife_cloud.types import BoxVolumeMeasurement, CameraIntrinsics, Detection  # noqa: E402
from locallife_cloud.volume import ReferencePlane, estimate_volume, fit_reference_plane  # noqa: E402
from test_v49_box_fix import KW, _obb, _scene  # noqa: E402

MODES = ("surface-columns", "reference-plane", "triangulated-surface", "ray-frustum", "height-map-grid",
         "support-plane-jacobian")


def front_facing_2l():
    """The exact supplied fixture: a 0.20 x 0.10 x 0.10 m box top at Z = 0.9 m over a floor at Z = 1.0 m;
    each top pixel spans 0.9/900 = 0.001 m, the mask covers 200 x 100 px -> 0.02 m^2 x 0.1 m = 2 L."""
    k = CameraIntrinsics(fx=900.0, fy=900.0, ppx=130.0, ppy=80.0, width=260, height=160)
    base = np.full((160, 260), 1.0, np.float32)
    depth = base.copy()
    depth[30:130, 30:230] = 0.9
    mask = np.zeros(base.shape, bool)
    mask[30:130, 30:230] = True
    return k, base, depth, mask, ReferencePlane(0.0, 0.0, 1000, (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))


class SupportPlaneIntegrationTests(unittest.TestCase):
    def test_supplied_two_litre_fixture_in_every_mode_and_the_configured_one(self):
        k, base, depth, mask, plane = front_facing_2l()
        for mode in MODES + (AppConfig().volume_geometry,):
            with self.subTest(mode=mode):
                r = estimate_volume(depth, base, k, object_mask=mask, geometry_mode=mode,
                                    fill_small_holes=False, reference_plane=plane)
                self.assertAlmostEqual(r.liters, 2.0, places=6)          # was 2.469 / 2.469 / 2.230 L
                self.assertTrue(r.geometry_mode.startswith("support-plane-jacobian"))
        legacy = estimate_volume(depth, base, k, object_mask=mask, geometry_mode="height-map-grid-legacy",
                                 fill_small_holes=False, reference_plane=plane)
        self.assertIsNotNone(legacy)                                       # still available for comparison

    def test_ray_traced_boxes_at_several_tilts_positions_and_orientations(self):
        mode = AppConfig().volume_geometry
        for dims, pitch, cam, yaw in (((0.10, 0.10, 0.10), 15.0, 0.9, 60.0), ((0.10, 0.10, 0.10), 45.0, 0.8, 10.0),
                                      ((0.20, 0.10, 0.10), 45.0, 0.8, 25.0), ((0.10, 0.10, 0.20), 30.0, 0.85, 40.0),
                                      ((0.20, 0.05, 0.20), 60.0, 0.6, 0.0)):
            with self.subTest(dims=dims, pitch=pitch, yaw=yaw):
                depth, mask = _scene(lambda x, y: 0 * x, _obb(*dims, 0, yaw, 0.0), pitch, cam=cam, far=cam * 3)
                empty, _ = _scene(lambda x, y: 0 * x, None, pitch, cam=cam, far=cam * 3)
                r = estimate_volume(depth, empty, KW, object_mask=mask, geometry_mode=mode, fill_small_holes=False,
                                    reference_plane=fit_reference_plane(empty, KW))
                truth = dims[0] * dims[1] * dims[2] * 1000.0
                # at 60 deg the 5 cm-deep top of the last box spans ~4 pixel rows of this 320 x 240 view:
                # its edge pixels' one-sided derivatives limit it to a few per cent (sampling, not formula)
                tolerance = 0.04 if pitch >= 60.0 else 0.025
                self.assertLess(abs(r.liters - truth) / truth, tolerance, (r.liters, truth))

    def test_empty_scene_and_dropout(self):
        k, base, depth, mask, plane = front_facing_2l()
        self.assertIsNone(estimate_volume(base.copy(), base, k, object_mask=mask, reference_plane=plane))
        holed = depth.copy()
        holed[30:130, 30:230] = np.nan                                       # no depth on the object at all
        self.assertIsNone(estimate_volume(holed, base, k, object_mask=mask, reference_plane=plane,
                                          fill_small_holes=False))

    def test_mask_leakage_onto_the_floor_adds_nothing(self):
        k, base, depth, mask, plane = front_facing_2l()
        leaky = np.zeros_like(mask)
        leaky[20:140, 20:240] = True
        r = estimate_volume(depth, base, k, object_mask=leaky, fill_small_holes=False, reference_plane=plane,
                            geometry_mode=AppConfig().volume_geometry)
        self.assertAlmostEqual(r.liters, 2.0, places=3)


class MeasurementContractTests(unittest.TestCase):
    def _cuboid(self, litres=1.9):
        return BoxVolumeMeasurement(volume_liters=litres, volume_confidence=0.8, volume_method="table-relative-cuboid",
                                    length_mm=200, width_mm=100, height_mm=95, depth_valid_ratio=1.0,
                                    object_points=900, table_plane_inliers=5000, table_plane_rmse_mm=1.0,
                                    height_p98_mm=95, height_top_median_mm=95, mask_clipped=False)

    def test_template_nominal_volume_never_replaces_the_measurement(self):
        fake = SimpleNamespace(box_templates=[BoxTemplate("2L-box", 2.0, 200, 100, 100, 15.0, measured=True)],
                               config=SimpleNamespace(realsense_max_item_volume_l=250.0))
        det = Detection("cardboard box", 0.9, (0, 0, 10, 10))
        VisionPipeline._apply_box_cuboid(fake, det, self._cuboid(1.9), [])
        self.assertEqual(det.box_template_id, "2L-box")
        self.assertEqual(det.box_template_nominal_volume_liters, 2.0)      # metadata only
        self.assertAlmostEqual(det.realsense_volume_l, 1.9)                # the measured value stays
        self.assertNotIn("template", det.measurement_method)

    def test_integration_factor_is_not_fitted_from_another_method(self):
        self.assertFalse(_integration_method("table-relative-cuboid"))
        self.assertFalse(_integration_method("table-relative-cuboid-template"))
        self.assertTrue(_integration_method("realsense-depth-integration"))
        self.assertTrue(_integration_method(None))
        det = Detection("cardboard box", 0.9, (0, 0, 10, 10))
        det.realsense_volume_l, det.measurement_method = 2.4, "table-relative-cuboid"
        det.tracking_status = "confirmed"
        import threading
        fake = SimpleNamespace(lock=threading.Lock(), camera_id="realsense",
                               latest_analysis=SimpleNamespace(detections=[det]),
                               config=SimpleNamespace(volume_calibration_factor=1.0, volume_geometry="x"))
        with self.assertRaises(ValueError):
            VisionPipeline.calibrate_known_volume(fake, 2.0)
        self.assertEqual(fake.config.volume_calibration_factor, 1.0)       # unchanged


if __name__ == "__main__":
    unittest.main()
