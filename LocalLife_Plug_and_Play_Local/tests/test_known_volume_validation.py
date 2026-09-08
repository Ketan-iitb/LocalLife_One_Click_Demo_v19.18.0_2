"""Synthetic known-volume (1 L / 2 L cuboid) accuracy regression guards.

This is the automated counterpart to `scripts/validate_known_volume.py` (the
tool the user runs against a real 1 L cube / 2 L box on their own hardware).
Real cameras are not reachable from this sandbox, so these tests build the
same kind of synthetic depth-array ground truth used by prior rounds'
regression suites (see `tests/test_bag_station.py`'s
`test_height_p90_reports_near_the_top_of_a_dome_shaped_object...` and
`test_patchy_dropout_below_the_old_seventy_percent_gate_is_now_recovered`):
a cuboid of an exactly known real-world volume is placed at a known depth in
front of a known-intrinsics camera, and the pixel footprint that volume
*should* occupy is derived analytically from the pinhole projection formula
`estimate_volume()` itself uses (`area_m2 = depth_m**2 / (fx * fy)`), not
guessed. A modest, deliberately-introduced mask-boundary erosion (a few
pixels shaved off each edge of the true footprint) then stands in for the
kind of imperfect real-world segmentation boundary that
`calibrate_known_volume()`'s own docstring names as exactly what a per-
installation calibration factor is meant to absorb -- so these tests can
honestly demonstrate both "the uncalibrated reading is in the right
ballpark" and "calibrating against this exact object corrects it", without
claiming that fitting and validating against the same object proves general
independent-object accuracy (it does not; see `validate_known_volume.py`
and the "Validate accuracy using separate objects not used to fit this
factor" warning already baked into `calibrate_known_volume()`'s own output).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.config import AppConfig
from locallife_cloud.pipeline import VisionPipeline
from locallife_cloud.types import CameraIntrinsics, Detection
from locallife_cloud.volume import estimate_volume

# Shared, arbitrary-but-realistic camera model for every synthetic object
# below: a 640x480-class sensor's focal length in pixels is commonly in the
# many-hundreds range for a RealSense-class RGB-D camera at this resolution.
FX = FY = 600.0
# Boundary erosion (pixels shaved off each of the four mask edges) standing
# in for realistic under-segmentation at an object's silhouette edge.
EDGE_EROSION_PX = 4
# The user's own proposal states a +/-10-15% accuracy target; the codebase's
# own settle/rejection tolerances elsewhere in this project also center on
# that range (see AppConfig.settle_volume_tolerance). Used as the ballpark
# (uncalibrated) tolerance below.
BALLPARK_TOLERANCE_FRACTION = 0.15
# After calibrating against the exact same object, the correction is exact
# by construction (calibration_factor = true / observed cancels the bias
# introduced above) modulo floating point -- this tight tolerance exists to
# prove the factor is actually wired through the full liters computation,
# not to claim independent-object accuracy.
CALIBRATED_TOLERANCE_FRACTION = 0.01


def _synthetic_cuboid_measurement(
    *, true_liters: float, height_m: float, baseline_depth_m: float,
    calibration_factor: float = 1.0,
) -> tuple[float, float, int]:
    """Build a synthetic cuboid depth array and measure it with `estimate_volume`.

    Returns (measured_liters, true_liters_for_the_actual_integer_pixel_mask,
    full_footprint_side_px). The full (true) footprint side length is solved
    analytically from the pinhole area formula so that the *un-eroded* mask
    integrates to exactly `true_liters` -- this is the same formula
    `estimate_volume()` itself uses, not an independent guess, so any
    remaining discrepancy after erosion is entirely attributable to the
    erosion, not to a mismatched area model.
    """
    object_depth_m = baseline_depth_m - height_m
    footprint_area_m2 = (true_liters / 1000.0) / height_m
    full_side_px = int(round((footprint_area_m2 * FX * FY / (object_depth_m ** 2)) ** 0.5))

    size = full_side_px + 4 * EDGE_EROSION_PX + 20
    margin = (size - full_side_px) // 2
    reference = np.full((size, size), baseline_depth_m, dtype=np.float32)
    depth = reference.copy()

    full_mask = np.zeros((size, size), dtype=bool)
    full_mask[margin: margin + full_side_px, margin: margin + full_side_px] = True
    exact_true_liters = float(
        (full_side_px ** 2) * (object_depth_m ** 2) / (FX * FY) * height_m * 1000.0
    )

    eroded_mask = np.zeros((size, size), dtype=bool)
    inset = margin + EDGE_EROSION_PX
    eroded_side = full_side_px - 2 * EDGE_EROSION_PX
    eroded_mask[inset: inset + eroded_side, inset: inset + eroded_side] = True
    depth[eroded_mask] = object_depth_m

    intrinsics = CameraIntrinsics(fx=FX, fy=FY, ppx=size / 2, ppy=size / 2, width=size, height=size)
    measurement = estimate_volume(
        depth, reference, intrinsics,
        object_mask=np.ones((size, size), dtype=bool),
        calibration_factor=calibration_factor,
    )
    assert measurement is not None
    return measurement.liters, exact_true_liters, full_side_px


class SyntheticKnownVolumeAccuracyTests(unittest.TestCase):
    """Direct `estimate_volume()`-level 1 L / 2 L ground-truth checks."""

    def test_one_liter_cuboid_is_measured_within_ballpark_tolerance_uncalibrated(self) -> None:
        measured, true_liters, side_px = _synthetic_cuboid_measurement(
            true_liters=1.0, height_m=0.10, baseline_depth_m=0.6,
        )
        self.assertAlmostEqual(true_liters, 1.0, delta=0.01)
        percent_error = abs(measured - true_liters) / true_liters
        self.assertLess(
            percent_error, BALLPARK_TOLERANCE_FRACTION,
            f"1L cuboid ({side_px}px footprint): measured {measured:.4f} L vs "
            f"true {true_liters:.4f} L ({percent_error * 100:.1f}% error)",
        )
        # The erosion is a real, non-trivial bias -- this guards against the
        # test accidentally becoming a no-op if the geometry math changes.
        self.assertGreater(percent_error, 0.03)

    def test_two_liter_cuboid_is_measured_within_ballpark_tolerance_uncalibrated(self) -> None:
        measured, true_liters, side_px = _synthetic_cuboid_measurement(
            true_liters=2.0, height_m=0.12, baseline_depth_m=0.6,
        )
        self.assertAlmostEqual(true_liters, 2.0, delta=0.02)
        percent_error = abs(measured - true_liters) / true_liters
        self.assertLess(
            percent_error, BALLPARK_TOLERANCE_FRACTION,
            f"2L cuboid ({side_px}px footprint): measured {measured:.4f} L vs "
            f"true {true_liters:.4f} L ({percent_error * 100:.1f}% error)",
        )
        self.assertGreater(percent_error, 0.03)

    def test_calibration_factor_meaningfully_improves_one_liter_accuracy(self) -> None:
        uncalibrated, true_liters, _ = _synthetic_cuboid_measurement(
            true_liters=1.0, height_m=0.10, baseline_depth_m=0.6,
        )
        factor = true_liters / uncalibrated
        calibrated, _, _ = _synthetic_cuboid_measurement(
            true_liters=1.0, height_m=0.10, baseline_depth_m=0.6, calibration_factor=factor,
        )
        uncalibrated_error = abs(uncalibrated - true_liters) / true_liters
        calibrated_error = abs(calibrated - true_liters) / true_liters
        self.assertLess(calibrated_error, CALIBRATED_TOLERANCE_FRACTION)
        self.assertLess(calibrated_error, uncalibrated_error)

    def test_calibration_factor_meaningfully_improves_two_liter_accuracy(self) -> None:
        uncalibrated, true_liters, _ = _synthetic_cuboid_measurement(
            true_liters=2.0, height_m=0.12, baseline_depth_m=0.6,
        )
        factor = true_liters / uncalibrated
        calibrated, _, _ = _synthetic_cuboid_measurement(
            true_liters=2.0, height_m=0.12, baseline_depth_m=0.6, calibration_factor=factor,
        )
        uncalibrated_error = abs(uncalibrated - true_liters) / true_liters
        calibrated_error = abs(calibrated - true_liters) / true_liters
        self.assertLess(calibrated_error, CALIBRATED_TOLERANCE_FRACTION)
        self.assertLess(calibrated_error, uncalibrated_error)

    def test_aggregate_mape_across_one_and_two_liter_objects_is_within_target(self) -> None:
        # Mirrors the user's own Master Thesis Proposal accuracy methodology
        # (median/90th-percentile absolute percentage error across multiple
        # known-volume trials), aggregated here as MAPE across the two
        # synthetic reference objects.
        errors = []
        for true_liters, height_m in ((1.0, 0.10), (2.0, 0.12)):
            measured, true_value, _ = _synthetic_cuboid_measurement(
                true_liters=true_liters, height_m=height_m, baseline_depth_m=0.6,
            )
            errors.append(abs(measured - true_value) / true_value)
        mape = sum(errors) / len(errors)
        self.assertLess(mape, BALLPARK_TOLERANCE_FRACTION)

    def test_max_item_volume_defaults_reject_a_near_full_frame_false_reading(self) -> None:
        # A real incident: something covering almost the whole camera frame
        # (a hand/pillow held close to the lens during testing, or a
        # monocular-depth hallucination on a shadow) integrated to ~116 L.
        # That was *under* the old 120 L cap, so it was reported as a real
        # measurement instead of being rejected as implausible. 90 L still
        # comfortably fits a large real waste bag/box but must reject this.
        config = AppConfig()
        self.assertEqual(config.logitech_max_item_volume_l, 90.0)
        self.assertEqual(config.realsense_max_item_volume_l, 90.0)
        self.assertLess(116.0, 120.0)  # the old cap really did let this through
        self.assertGreater(116.0, config.realsense_max_item_volume_l)  # the new one rejects it


class EndToEndKnownVolumeCalibrationTests(unittest.TestCase):
    """Full `VisionPipeline` + `calibrate_known_volume()` wiring, 1 L object."""

    def test_pipeline_calibrate_known_volume_corrects_a_real_detections_own_reading(self) -> None:
        true_liters = 1.0
        height_m = 0.10
        baseline_depth_m = 0.6
        object_depth_m = baseline_depth_m - height_m
        footprint_area_m2 = (true_liters / 1000.0) / height_m
        full_side_px = int(round((footprint_area_m2 * FX * FY / (object_depth_m ** 2)) ** 0.5))
        size = full_side_px + 4 * EDGE_EROSION_PX + 20
        margin = (size - full_side_px) // 2
        inset = margin + EDGE_EROSION_PX
        eroded_side = full_side_px - 2 * EDGE_EROSION_PX

        detected_mask = np.zeros((size, size), dtype=bool)
        detected_mask[inset: inset + eroded_side, inset: inset + eroded_side] = True
        box = (inset, inset, inset + eroded_side, inset + eroded_side)

        with tempfile.TemporaryDirectory() as temporary:
            detector = _SingleShotDetector(
                Detection("garbage bag", 0.9, box, mask=detected_mask.copy())
            )
            config = AppConfig(
                results_dir=Path(temporary),
                enable_monocular_depth=False,
                roi=(0, 0, 1, 1),
                min_component_pixels=200,
                tracker_confirm_frames=1,
                bag_only=True,
            )
            pipeline = VisionPipeline(config, detector=detector)
            empty = np.zeros((size, size, 3), dtype=np.uint8)
            baseline = np.full((size, size), baseline_depth_m, dtype=np.float32)
            camera = CameraIntrinsics(fx=FX, fy=FY, ppx=size / 2, ppy=size / 2, width=size, height=size)
            pipeline.set_baseline(empty, baseline, camera)

            frame = empty.copy()
            frame[detected_mask] = (0, 0, 255)
            depth = baseline.copy()
            depth[detected_mask] = object_depth_m

            first = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=1.0)
            confirmed = [item for item in first.detections if item.confidence > 0]
            self.assertEqual(len(confirmed), 1)
            observed_liters = confirmed[0].realsense_volume_l
            self.assertIsNotNone(observed_liters)
            uncalibrated_error = abs(observed_liters - true_liters) / true_liters
            # This scene's footprint (`full_side_px`) is deliberately sized
            # using the OBJECT's own depth, so a formula that also uses the
            # object's own depth for pixel footprint area (ray-frustum,
            # surface-columns) reproduces true_liters almost exactly at zero
            # tilt. reference-plane (default since round 16 -- see
            # config.py's `volume_geometry` comment) instead uses the
            # REFERENCE/baseline depth for footprint area, which is farther
            # from the camera than the object here (0.6 m vs 0.5 m) and so
            # overstates each pixel's real-world footprint by
            # (0.6/0.5)^2 = 1.44x before erosion/discretization effects --
            # numerically verified at exactly 0.2544 (25.44%) for this exact
            # scene. That is an expected, understood consequence of which
            # depth reference-plane's footprint area is anchored to, not a
            # regression: what this test actually verifies (below) is that
            # `calibrate_known_volume()` corrects this detection's own
            # reading to true_liters exactly, regardless of the starting bias.
            self.assertLess(uncalibrated_error, 0.30)

            # Regression guard for the `calibrate_known_volume()` fix: the
            # implicit `observed_liters` it solves the factor from must be
            # exactly this single confirmed detection's own displayed
            # reading, not some separately-computed camera-wide total.
            calibration = pipeline.calibrate_known_volume(known_liters=true_liters)
            self.assertAlmostEqual(calibration["observed_liters"], observed_liters, places=6)

            second = pipeline.process_frame(frame, depth_m=depth, intrinsics=camera, timestamp=2.0)
            recalibrated = [item for item in second.detections if item.confidence > 0][0]
            calibrated_error = abs(recalibrated.realsense_volume_l - true_liters) / true_liters
            self.assertLess(calibrated_error, CALIBRATED_TOLERANCE_FRACTION)
            self.assertLess(calibrated_error, uncalibrated_error)


class _SingleShotDetector:
    """Returns the same pre-built detection on every call, like a fixed scene."""

    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self, detection: Detection) -> None:
        self.detection = detection

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [
            [Detection(
                label=self.detection.label, confidence=self.detection.confidence,
                box=self.detection.box,
                mask=None if self.detection.mask is None else self.detection.mask.copy(),
                source=self.detection.source,
            )]
            for _ in frames
        ]


if __name__ == "__main__":
    unittest.main()
