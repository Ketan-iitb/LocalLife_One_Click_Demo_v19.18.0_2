"""V36: the Logitech camera has to start and measure without being told anything.

The bug the operator reported is here in one line. `/api/cameras/logitech/
reference-distance` used to call `station.set_baseline()`, which recaptures the
empty-scene reference from whatever is in front of the camera at that moment --
and the moment anyone types a distance is the moment an object is standing in
the zone. That object became part of the floor, nothing ever differed from the
reference again, every mask was refused for want of a foreground change, and
the dashboard filled with "pending". Typing a number switched the system off.

So the distance, the known volume and the baseline are three separate settings
and are no longer wired to each other, and neither of the first two is needed
in normal use: the camera height comes from the fitted floor plane and the
baseline is learned from still, empty startup frames.

The scenes are ray-traced with the rig's geometry and known ground truth. They
are synthetic and say nothing about what the real C920 reports.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.logitech_autocal import (
    LEARNED,
    NO_DEPTH,
    VIEW_NOT_STILL,
    WAITING,
    ZONE_NOT_EMPTY,
    BaselineLearner,
    camera_height_from_plane,
)
from locallife_cloud.logitech_calibration import METRIC_OUTPUT, depth_output_kind
from locallife_cloud.logitech_geometry import robust_object_height_m, static_background_reason
from locallife_cloud.logitech_volume import metric_object_volume
from locallife_cloud.volume import fit_reference_plane

from .test_v35_logitech_metric_geometry import (  # noqa: F401 - shared synthetic rig
    CAMERA,
    CAMERA_HEIGHT_M,
    HEIGHT,
    WIDTH,
    _cream_bottle,
    _floor_plane,
    _measure,
    _shoe_box,
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "5205c46d969455883082425d2a30b1fc52aef4b7"


def _pipeline(**attributes):
    """A VisionPipeline with only the fields under test, and no camera."""
    from locallife_cloud.pipeline import VisionPipeline

    station = VisionPipeline.__new__(VisionPipeline)
    station.camera_id = "logitech"
    station.reference_monocular = None
    station.reference_rgb = None
    station._logitech_plane_cache = None
    station._logitech_plane_reason = None
    station._logitech_plane_source = "none"
    station.logitech_distance_source = "none"
    station.logitech_derived_distance_m = None
    for name, value in attributes.items():
        setattr(station, name, value)
    return station


class StartsWithoutBeingToldAnythingTests(unittest.TestCase):
    """Test 1: no manual distance, no known volume, no calibration clicks."""

    def test_the_camera_height_comes_from_the_fitted_floor(self) -> None:
        _, empty, _ = _shoe_box()
        plane = fit_reference_plane(empty, CAMERA)
        derived = camera_height_from_plane(plane.coefficients)
        self.assertIsNotNone(derived)
        self.assertAlmostEqual(derived, CAMERA_HEIGHT_M, delta=0.05)

    def test_a_station_with_no_measured_distance_derives_one(self) -> None:
        from locallife_cloud.config import AppConfig

        depth, empty, _ = _shoe_box()
        config = AppConfig()
        config.logitech_reference_distance_m = 0.0
        station = _pipeline(config=config, reference_monocular=empty)
        plane = station._logitech_floor_plane(depth, CAMERA, None, None)
        height = station._auto_camera_height(plane)
        self.assertIsNotNone(height)
        # Geometry gets a height without anyone typing one...
        self.assertAlmostEqual(station.camera_height_m(), CAMERA_HEIGHT_M, delta=0.05)
        self.assertEqual(station.logitech_distance_source, "derived_from_floor_plane")
        # ... but a derived height never pretends to be a measured installation:
        # the research modes and the auto-deposit gate read this field, and a
        # number the code derived from its own depth cannot verify a rig.
        self.assertEqual(config.logitech_reference_distance_m, 0.0)

    def test_an_operator_measurement_is_kept_over_a_derived_one(self) -> None:
        from locallife_cloud.config import AppConfig

        depth, empty, _ = _shoe_box()
        config = AppConfig()
        config.logitech_reference_distance_m = 0.90
        station = _pipeline(config=config, reference_monocular=empty)
        station.logitech_distance_source = "operator_measured"
        station._auto_camera_height(station._logitech_floor_plane(depth, CAMERA, None, None))
        self.assertAlmostEqual(config.logitech_reference_distance_m, 0.90)
        self.assertEqual(station.logitech_distance_source, "operator_measured")
        # The derived value is still recorded, so the two can be compared.
        self.assertIsNotNone(station.logitech_derived_distance_m)

    def test_a_wall_or_a_plane_behind_the_camera_is_not_a_height(self) -> None:
        self.assertIsNone(camera_height_from_plane(None))
        self.assertIsNone(camera_height_from_plane((0.0, 0.0, 0.02)))    # too close
        self.assertIsNone(camera_height_from_plane((0.0, 0.0, 40.0)))    # too far
        self.assertIsNone(camera_height_from_plane((float("nan"), 0.0, 1.0)))

    def test_a_measurement_needs_no_manual_input_at_all(self) -> None:
        depth, empty, mask = _shoe_box()
        result = _measure(depth, empty, mask)
        self.assertIsNone(result.reason)
        self.assertIsNotNone(result.measurement)
        self.assertTrue(result.diagnostics["plane_is_floor"])


class OptionalReferencesNeverBreakAnythingTests(unittest.TestCase):
    """Test 2: editing or clearing a reference must not disable anything."""

    def test_setting_a_distance_does_not_recapture_the_baseline(self) -> None:
        import threading

        from locallife_cloud.config import AppConfig

        config = AppConfig()
        station = _pipeline(config=config, lock=threading.RLock())
        station.reference_rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        station.reference_monocular = np.full((HEIGHT, WIDTH), 1.2, np.float32)
        before_rgb = station.reference_rgb
        before_depth = station.reference_monocular

        station.note_manual_reference_distance(1.05)

        self.assertAlmostEqual(config.logitech_reference_distance_m, 1.05)
        self.assertIs(station.reference_rgb, before_rgb)
        self.assertIs(station.reference_monocular, before_depth)
        self.assertEqual(station.logitech_distance_source, "operator_measured")

    def test_the_distance_endpoint_no_longer_calls_set_baseline(self) -> None:
        """The coupling itself, read straight out of the route."""
        import inspect

        from locallife_cloud import server

        source = inspect.getsource(server.create_app)
        start = source.index("def set_logitech_distance")
        body = source[start : source.index("@app.post", start + 10)]
        code = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("set_baseline()", code)
        self.assertIn("note_manual_reference_distance", code)

    def test_a_known_volume_keeps_the_logitech_geometry(self) -> None:
        import threading

        from locallife_cloud.config import AppConfig
        from locallife_cloud.shape_geometry import GeometryLock

        class _Store:
            def save_json(self, *_args, **_kwargs) -> None:
                return None

        config = AppConfig()
        station = _pipeline(
            config=config, lock=threading.RLock(), store=_Store(),
            _volume_history={}, _box_measurement_history={}, _box_frames_considered={},
            _geometry_lock=GeometryLock(), _track_signatures={"sentinel": 1},
        )
        station.calibrate_known_volume(10.0, observed_liters=9.0)
        # The factor applies to litres; an object's metres do not depend on it,
        # so its shape lock and identity survive. Clearing them sent every
        # object on screen back to "not settled" with no dimensions.
        self.assertEqual(station._track_signatures, {"sentinel": 1})
        self.assertAlmostEqual(config.volume_calibration_factor, 10.0 / 9.0)


class BaselineIsLearnedTests(unittest.TestCase):
    """Automatic startup, and the refusals that keep it honest."""

    def setUp(self) -> None:
        self.learner = BaselineLearner(required_frames=4, depth_noise_m=0.004)
        self.depth = np.full((64, 64), 1.20, np.float32)
        self.region = np.ones((64, 64), dtype=bool)

    def _run(self, frames: int, **kwargs):
        decision = None
        for _ in range(frames):
            decision = self.learner.observe(
                depth=self.depth, region=self.region, objects_in_zone=0, **kwargs,
            )
        return decision

    def test_a_still_empty_view_is_captured_by_itself(self) -> None:
        decision = self._run(4)
        self.assertTrue(decision.capture)
        self.assertEqual(decision.reason, LEARNED)

    def test_an_object_in_the_zone_is_never_captured_as_floor(self) -> None:
        self._run(3)
        decision = self.learner.observe(
            depth=self.depth, region=self.region, objects_in_zone=1,
        )
        self.assertFalse(decision.capture)
        self.assertEqual(decision.reason, ZONE_NOT_EMPTY)
        # ... and the run starts again from nothing.
        self.assertFalse(self.learner.observe(
            depth=self.depth, region=self.region, objects_in_zone=0).capture)

    def test_a_moving_view_is_refused_with_its_own_reason(self) -> None:
        for index in range(4):
            noisy = self.depth + (0.2 if index % 2 else -0.2)
            decision = self.learner.observe(
                depth=noisy, region=self.region, objects_in_zone=0,
            )
        self.assertFalse(decision.capture)
        self.assertEqual(decision.reason, VIEW_NOT_STILL)

    def test_no_depth_yet_says_so(self) -> None:
        decision = self.learner.observe(depth=None, region=None, objects_in_zone=0)
        self.assertEqual(decision.reason, NO_DEPTH)

    def test_the_first_frames_report_progress_not_failure(self) -> None:
        decision = self.learner.observe(
            depth=self.depth, region=self.region, objects_in_zone=0)
        self.assertEqual(decision.reason, WAITING)
        self.assertEqual(decision.stable_frames, 1)


class RelativeDepthIsNeverMetricTests(unittest.TestCase):
    """Test 3."""

    def test_the_shipped_checkpoints_are_metric_ones(self) -> None:
        from locallife_cloud.config import LOCAL_DEPTH_MODEL, AppConfig

        self.assertEqual(depth_output_kind(LOCAL_DEPTH_MODEL), METRIC_OUTPUT)
        self.assertEqual(depth_output_kind(AppConfig().depth_model), METRIC_OUTPUT)

    def test_a_relative_checkpoint_is_recognised_as_relative(self) -> None:
        self.assertNotEqual(
            depth_output_kind("depth-anything/Depth-Anything-V2-Small-hf"), METRIC_OUTPUT,
        )

    def test_relative_output_without_a_distance_is_refused_not_scaled(self) -> None:
        from locallife_cloud.config import AppConfig

        config = AppConfig()
        config.depth_model = "depth-anything/Depth-Anything-V2-Small-hf"
        config.logitech_reference_distance_m = 0.0
        station = _pipeline(config=config, calibration_mode="")
        inverse = np.full((64, 64), 0.8, np.float32)
        out = station._metric_from_relative(inverse, np.ones((64, 64), dtype=bool))
        self.assertEqual(
            station.relative_depth_reason, "relative_depth_without_reference_distance",
        )
        # Returned unchanged: inverse depth is handed back as it came, never
        # relabelled as metres.
        np.testing.assert_allclose(out, inverse)


class MaskLeakageTests(unittest.TestCase):
    """Test 4: floor, furniture and merged masks are rejected."""

    def setUp(self) -> None:
        self.region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.region[100:400, 120:520] = True

    def test_a_floor_spanning_mask_is_rejected(self) -> None:
        strip = np.zeros((HEIGHT, WIDTH), dtype=bool)
        strip[180:230, 120:520] = True
        self.assertEqual(
            static_background_reason(strip, self.region), "spans_measurement_zone",
        )

    def test_a_mask_filling_the_zone_is_rejected_too(self) -> None:
        wide = np.zeros((HEIGHT, WIDTH), dtype=bool)
        wide[150:380, 140:500] = True
        self.assertEqual(
            static_background_reason(wide, self.region), "covers_measurement_zone",
        )

    def test_an_object_sized_mask_is_kept(self) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[220:300, 260:360] = True
        self.assertIsNone(static_background_reason(mask, self.region))

    def test_floor_leakage_beside_an_object_does_not_join_its_footprint(self) -> None:
        depth, empty, mask = _shoe_box()
        leaked = mask.copy()
        leaked[430:470, 80:560] = True          # a strip of floor along the bottom
        clean_result = _measure(depth, empty, mask)
        leaked_result = _measure(depth, empty, leaked)
        self.assertIsNone(clean_result.reason)
        if leaked_result.reason is None:
            # The floor is at zero height, so it never enters the footprint.
            self.assertLess(
                abs(leaked_result.diagnostics["length_mm"]
                    - clean_result.diagnostics["length_mm"]),
                40.0,
            )


class KnownGeometryTests(unittest.TestCase):
    """Test 5: height, length, breadth and volume against known truth."""

    def test_the_shoe_box(self) -> None:
        depth, empty, mask = _shoe_box()
        result = _measure(depth, empty, mask)
        self.assertIsNone(result.reason)
        for measured, truth, name in (
            (result.diagnostics["length_mm"], 338.0, "length"),
            (result.diagnostics["width_mm"], 253.0, "width"),
            (result.diagnostics["height_p90_m"] * 1000.0, 124.0, "height"),
        ):
            with self.subTest(dimension=name):
                self.assertLessEqual(abs(measured - truth) / truth, 0.20,
                                     f"{name}: {measured:.0f} against {truth:.0f} mm")
        litres = result.measurement.liters
        self.assertGreater(litres, 0.7 * 10.6)
        self.assertLess(litres, 1.3 * 10.6)

    def test_the_bottle_keeps_its_neck(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = _measure(depth, empty, mask)
        self.assertGreaterEqual(result.diagnostics["height_p90_m"] * 1000.0, 180.0)
        self.assertLessEqual(result.diagnostics["height_p90_m"] * 1000.0, 215.0)

    def test_a_few_depth_spikes_cannot_decide_a_flat_object_height(self) -> None:
        """The 78 mm bag that read 228 mm: a handful of cells must not set it.

        Three per cent of the footprint has to reach a height before it counts,
        so eight wild cells out of a thousand are ignored while a bottle's neck
        -- a twelfth of its footprint -- still counts.
        """
        flat = np.full(1000, 0.078)
        spiked = np.r_[flat, np.full(8, 0.320)]
        self.assertAlmostEqual(robust_object_height_m(spiked), 0.078, places=3)

        neck = np.r_[np.full(880, 0.150), np.full(120, 0.203)]
        self.assertAlmostEqual(robust_object_height_m(neck), 0.203, places=3)

    def test_a_small_object_still_needs_several_cells_to_agree(self) -> None:
        tiny = np.r_[np.full(30, 0.082), np.full(1, 0.400)]
        self.assertAlmostEqual(robust_object_height_m(tiny), 0.082, places=3)


class LogitechOnlyTests(unittest.TestCase):
    """Test 6: the Logitech path reads no RealSense frame or output."""

    def test_the_measurement_runs_with_no_realsense_input(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = metric_object_volume(
            depth, CAMERA, mask, _floor_plane(empty),
            reference_depth_m=None, measurement_mask=None,
            min_height_m=0.008, min_pixels=60,
        )
        self.assertIsNone(result.reason)

    def test_the_logitech_modules_never_import_the_realsense_one(self) -> None:
        import ast

        for name in (
            "logitech_geometry.py", "logitech_autocal.py",
            "logitech_volume.py", "logitech_footprint.py",
        ):
            with self.subTest(module=name):
                tree = ast.parse((PROJECT / "locallife_cloud" / name).read_text())
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                    elif isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                self.assertFalse(
                    [item for item in imported if "realsense" in item.lower()],
                    f"{name} imports a RealSense module: {imported}",
                )


class ProtectedSurfacesTests(unittest.TestCase):
    """Test 7: RealSense, the ledger, the cloud and the launcher are untouched."""

    def test_the_protected_files_are_unchanged_since_v35(self) -> None:
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
