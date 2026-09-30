"""v44: bin profile, physical bounds, no clipped or pile-inflated Logitech heights, readable overlay.

Synthetic scenes only: they check the logic, not real-world accuracy.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_v37_logitech_pose_and_colour as rig  # noqa: E402

from locallife_cloud import bin_profile as bp  # noqa: E402
from locallife_cloud import logitech_volume as lv  # noqa: E402
from locallife_cloud.comparison import DualCameraCoordinator  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.server import _annotate_frame, create_app  # noqa: E402
from locallife_cloud.types import Detection, FrameAnalysis  # noqa: E402
from locallife_cloud.volume import fit_reference_plane  # noqa: E402

GOOD = {"optical_center_to_empty_floor_m": 1.45, "optical_center_above_rim_m": 0.40, "inner_depth_m": 1.00,
        "inner_length_top_m": 1.24, "inner_width_top_m": 0.71, "inner_length_floor_m": 1.08,
        "inner_width_floor_m": 0.60, "tilt_deg": 12.0}


def _refs(error=0.02):
    items = []
    for name, dims in (("box", (400.0, 300.0, 200.0)), ("tin", (100.0, 100.0, 150.0)), ("crate", (500, 350, 280))):
        for position in ("centre", "near-left", "far-right")[:2 if name != "box" else 3]:
            items.append(bp.ReferencePlacement(name, position, dims, tuple(d * (1 + error) for d in dims)))
    return items


class BinProfileTests(unittest.TestCase):
    def test_default_is_nominal_660l_and_unverified(self) -> None:
        profile = bp.BinProfile(camera_id="logitech")
        self.assertEqual(profile.status, "unverified-nominal")
        self.assertIn("unverified", profile.capacity_label)
        bounds = profile.bounds()
        self.assertIn("UNVERIFIED", bounds["source"])
        self.assertAlmostEqual(bounds["max_height_m"], 1.30)
        self.assertGreater(bounds["max_footprint_m"], 1.4)

    def test_validation_rejects_incomplete_or_inaccurate_calibration(self) -> None:
        missing = bp.validate(bp.BinProfile("realsense", measurements={"inner_depth_m": 1.0}))
        self.assertEqual(missing.status, "rejected")
        self.assertIn("optical_center_to_empty_floor_m", missing.status_reason)
        few = bp.validate(bp.BinProfile("realsense", measurements=dict(GOOD), references=_refs()[:2]))
        self.assertIn("reference check needs", few.status_reason)
        bad = bp.validate(bp.BinProfile("realsense", measurements=dict(GOOD), references=_refs(error=0.2)))
        self.assertEqual(bad.status, "rejected")
        self.assertIn("reference check failed", bad.status_reason)
        good = bp.validate(bp.BinProfile("realsense", measurements=dict(GOOD), references=_refs()))
        self.assertEqual(good.status, "measured", good.status_reason)
        self.assertEqual(good.bounds()["source"], "measured bin profile")

    def test_pose_check_needs_an_empty_bin_and_detects_a_moved_tripod(self) -> None:
        profile = bp.validate(bp.BinProfile("logitech", measurements=dict(GOOD), references=_refs()))
        self.assertEqual(bp.pose_check(profile, 12.2, 1.45, empty_bin=False)["state"], "not_checked")
        self.assertEqual(bp.pose_check(profile, 12.2, 1.46, empty_bin=True)["state"], "unchanged")
        moved = bp.pose_check(profile, 19.0, 1.30, empty_bin=True)
        self.assertEqual(moved["state"], "moved")
        self.assertEqual(len(moved["problems"]), 2)

    def test_physical_check_rejects_without_clamping(self) -> None:
        bounds = bp.BinProfile("realsense").bounds()
        self.assertIsNone(bp.physical_check(bounds, 450, 300, 250))
        self.assertEqual(bp.physical_check(bounds, 450, 300, -5), "physically_impossible_non_positive_height")
        self.assertEqual(bp.physical_check(bounds, 450, 300, 1500), "physically_impossible_height_exceeds_bin_depth")
        self.assertEqual(bp.physical_check(bounds, 2000, 300, 250), "physically_impossible_footprint_exceeds_bin")

    def test_store_round_trip_and_endpoints(self) -> None:
        with TemporaryDirectory() as d:
            store = bp.BinProfileStore(Path(d))
            profile = bp.validate(bp.BinProfile("logitech", measurements=dict(GOOD), references=_refs()))
            store.save(profile)
            self.assertEqual(store.load("logitech").status, "measured")
            self.assertEqual(store.load("realsense").status, "unverified-nominal")    # per camera

            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d) / "run")
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            got = client.get("/api/cameras/realsense/bin-profile").get_json()
            self.assertEqual(got["profile"]["status"], "unverified-nominal")
            rejected = client.post("/api/cameras/realsense/bin-profile", json={"measurements": {"inner_depth_m": 1}})
            self.assertEqual(rejected.status_code, 422)
            payload = {"measurements": GOOD, "method": "tape", "references": [
                {"object_name": r.object_name, "position": r.position, "true_mm": list(r.true_mm),
                 "measured_mm": list(r.measured_mm)} for r in _refs()]}
            accepted = client.post("/api/cameras/realsense/bin-profile", json=payload)
            self.assertEqual(accepted.status_code, 200, accepted.get_json())
            self.assertEqual(client.get("/api/cameras/realsense/bin-profile").get_json()["profile"]["status"], "measured")

    def test_pipeline_withholds_impossible_dimensions(self) -> None:
        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            station = DualCameraCoordinator(config).camera("realsense")
            ok = Detection("bag", 0.9, (0, 0, 10, 10))
            ok.footprint_length_mm, ok.footprint_width_mm, ok.physical_height_mm, ok.realsense_volume_l = 400, 300, 250, 20.0
            huge = Detection("bag", 0.9, (0, 0, 10, 10))
            huge.footprint_length_mm, huge.footprint_width_mm, huge.physical_height_mm, huge.realsense_volume_l = 3003, 538, 462, 77.0
            station._apply_bin_bounds([ok, huge])
            self.assertEqual(ok.realsense_volume_l, 20.0)
            self.assertIsNone(huge.realsense_volume_l)
            self.assertEqual(huge.volume_rejection_reason, "physically_impossible_footprint_exceeds_bin")


class LogitechHeightTests(unittest.TestCase):
    def _measure(self, max_height_m, surfaces, target=0):
        scene = rig.Scene(25.0)
        built = [build(scene) for build in surfaces]
        depth, _ = scene.render(*built)
        own_depth, own = scene.render(built[target])
        mask = own & (np.abs(depth - own_depth) < 1e-9)
        plane = fit_reference_plane(scene.empty, rig.CAMERA, mask=scene.region)
        return lv.metric_object_volume(depth, rig.CAMERA, mask, plane, reference_depth_m=scene.empty,
                                       min_height_m=0.004, min_pixels=25, cell_size_m=0.005,
                                       max_height_m=max_height_m)

    def test_an_object_above_the_cap_is_withheld_not_clipped(self) -> None:
        tall = [lambda s: s.box((0, 0), 0.20, 0.20, 0.50)]
        result = self._measure(0.40, tall)
        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "height_exceeds_physical_bound")
        # The same frame with the V42 behaviour (no guard) reports a clipped height under the cap.
        original = lv.OVER_HEIGHT_REJECT_FRACTION
        lv.OVER_HEIGHT_REJECT_FRACTION = 1.1
        try:
            old = self._measure(0.40, tall)
        finally:
            lv.OVER_HEIGHT_REJECT_FRACTION = original
        self.assertLessEqual(old.diagnostics["height_p90_m"], 0.40)        # V42: a number at the cap
        self.assertIsNotNone(self._measure(0.80, tall).measurement)        # inside the bound: measured

    def test_a_bag_on_an_uneven_pile_is_unreliable_not_pile_height(self) -> None:
        surfaces = [lambda s: s.box((0, 0), 0.60, 0.50, 0.20),
                    lambda s: s.cylinder((0, 0), 0.10, 0.15, base=0.20)]
        original = lv.local_support_height
        lv.local_support_height = lambda *a, **k: (None, "support_surface_uneven_height_above_bin_floor", {})
        try:
            result = self._measure(0.80, surfaces, target=1)
        finally:
            lv.local_support_height = original
        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "support_unknown_on_uneven_pile")


class OverlayTests(unittest.TestCase):
    def test_crowded_overlay_uses_short_non_overlapping_tags(self) -> None:
        import cv2

        with TemporaryDirectory() as d:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(d))
            station = DualCameraCoordinator(config).camera("logitech")
            detections = []
            for i in range(12):                       # a pile of boxes whose tag spots coincide
                det = Detection("filled plastic garbage bag", 0.8, (100 + i * 3, 100, 300, 300))
                det.track_id, det.accepted_class, det.color = i + 1, "waste_bag", "black"
                det.footprint_length_mm, det.footprint_width_mm, det.physical_height_mm = 400, 300, 790
                detections.append(det)
            station.latest_analysis = FrameAnalysis(timestamp=0.0, source="t", frame_width=640, frame_height=480, automatic_count=0,
                                                     detections=detections)
            calls = []
            original = cv2.putText
            cv2.putText = lambda img, text, org, *a, **k: calls.append((text, org))
            try:
                _annotate_frame(np.zeros((480, 640, 3), np.uint8), station)
            finally:
                cv2.putText = original
            texts = [t for t, _ in calls]
            self.assertTrue(all(len(t) <= 24 for t in texts), texts)          # short tags only
            self.assertFalse(any("LxWxH" in t or " L" in t[-3:] for t in texts))
            self.assertEqual(len({org for _, org in calls}), len(calls))      # no two at the same spot


if __name__ == "__main__":
    unittest.main()
