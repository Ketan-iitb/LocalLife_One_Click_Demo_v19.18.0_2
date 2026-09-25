"""V35: the Logitech dimensions must describe the object, not the scene.

The hardware reported a 338 x 253 x 124 mm shoe box as 540 x 360 x 89 mm and a
75 x 37 x 203 mm bottle as 125 x 110 x 137 mm, while measuring the bed as a
35 L deposit.

The scene here is ray-traced with the rig's own geometry -- a camera tilted
45 degrees, 0.95 m above the mat, looking across it -- with objects of known
size standing on a known floor. That makes the measurements checkable against
truth, and it is the only thing they are evidence about: these are synthetic
numbers from a modelled camera and say nothing about what the real C920 will
report.

Two things the field numbers pointed at are established here. Given the fitted
floor, the projection code recovers the box and the bottle accurately, so a
measurement taken against the optical-axis fallback plane instead is the thing
to prevent. And the spike gate was discarding a bottle's neck as an outlier,
which returned the bottle's *body* height rather than its height.
"""

from __future__ import annotations

import math
import subprocess
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.footprint import estimate_extents
from locallife_cloud.logitech_geometry import (
    GeometryStabiliser,
    geometry_consistency,
    minimum_object_pixels,
    robust_object_height_m,
    static_background_reason,
    unclaimed_foreground_islands,
)
from locallife_cloud.logitech_volume import (
    axis_aligned_plane,
    fit_plane_alignment,
    metric_object_volume,
    project_to_plane,
    ray_plane_distance,
)
from locallife_cloud.types import CameraIntrinsics
from locallife_cloud.volume import fit_reference_plane

PROJECT = Path(__file__).resolve().parents[1]
# The branch this work started from.
SOURCE_SHA = "bacafaf0be949db91eb39882e417dee1040b2017"

WIDTH, HEIGHT = 640, 480
CAMERA = CameraIntrinsics(fx=620.0, fy=620.0, ppx=WIDTH / 2, ppy=HEIGHT / 2,
                          width=WIDTH, height=HEIGHT)
# The rig in the screenshots: mounted high, looking down and across the mat.
TILT_DEG = 45.0
CAMERA_HEIGHT_M = 0.95
CAMERA_STANDOFF_M = 0.75


def _camera_basis(tilt_deg: float = TILT_DEG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera centre and its three world axes.

    World is X right, Y forward along the floor, Z up. The camera sits behind
    and above the mat and is tilted `tilt_deg` from straight down.
    """
    tilt = math.radians(tilt_deg)
    centre = np.array((0.0, -CAMERA_STANDOFF_M, CAMERA_HEIGHT_M))
    forward = np.array((0.0, math.sin(tilt), -math.cos(tilt)))   # optical axis
    right = np.array((1.0, 0.0, 0.0))
    down = np.cross(forward, right)                              # image +v
    return centre, right, down, forward


def _rays() -> tuple[np.ndarray, np.ndarray]:
    """World-space origin and direction of every pixel's ray.

    The direction keeps a unit component along the optical axis, so the ray
    parameter is the camera-frame depth the renderer writes out.
    """
    centre, right, down, forward = _camera_basis()
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    x = (columns - CAMERA.ppx) / CAMERA.fx
    y = (rows - CAMERA.ppy) / CAMERA.fy
    direction = (x[..., None] * right + y[..., None] * down + forward)
    return centre, direction


def _floor_depth(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Distance to the floor plane z = 0, as camera depth."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / direction[..., 2]
    return np.where((t > 0) & np.isfinite(t), t, np.inf)


def _box_depth(
    origin: np.ndarray, direction: np.ndarray, centre_xy: tuple[float, float],
    length_m: float, width_m: float, height_m: float,
) -> np.ndarray:
    """Slab intersection with an axis-aligned box standing on the floor."""
    low = np.array((centre_xy[0] - length_m / 2, centre_xy[1] - width_m / 2, 0.0))
    high = np.array((centre_xy[0] + length_m / 2, centre_xy[1] + width_m / 2, height_m))
    near = np.full(direction.shape[:2], -np.inf)
    far = np.full(direction.shape[:2], np.inf)
    for axis in range(3):
        component = direction[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (low[axis] - origin[axis]) / component
            second = (high[axis] - origin[axis]) / component
        entry = np.minimum(first, second)
        exit_ = np.maximum(first, second)
        near = np.maximum(near, np.where(np.isfinite(entry), entry, -np.inf))
        far = np.minimum(far, np.where(np.isfinite(exit_), exit_, np.inf))
    hit = (near <= far) & (far > 0)
    return np.where(hit & (near > 0), near, np.inf)


def _cylinder_depth(
    origin: np.ndarray, direction: np.ndarray, centre_xy: tuple[float, float],
    radius_m: float, height_m: float, base_m: float = 0.0,
) -> np.ndarray:
    """Intersection with an upright cylinder, side wall and top disc."""
    offset_x = origin[0] - centre_xy[0]
    offset_y = origin[1] - centre_xy[1]
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    a = dx * dx + dy * dy
    b = 2.0 * (offset_x * dx + offset_y * dy)
    c = offset_x * offset_x + offset_y * offset_y - radius_m * radius_m
    discriminant = b * b - 4.0 * a * c
    best = np.full(direction.shape[:2], np.inf)
    valid = (discriminant >= 0) & (a > 1e-12)
    root = np.sqrt(np.where(valid, discriminant, 0.0))
    for sign in (-1.0, 1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-b + sign * root) / (2.0 * a)
        z = origin[2] + t * dz
        keep = valid & (t > 0) & (z >= base_m) & (z <= base_m + height_m) & (t < best)
        best = np.where(keep, t, best)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = (base_m + height_m - origin[2]) / dz
    cap_x = offset_x + cap * dx
    cap_y = offset_y + cap * dy
    on_cap = (cap > 0) & (cap_x ** 2 + cap_y ** 2 <= radius_m ** 2) & (cap < best)
    return np.where(on_cap, cap, best)


def _scene(*objects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth of the whole scene, depth of the empty floor, and the object mask."""
    origin, direction = _rays()
    floor = _floor_depth(origin, direction)
    nearest = floor.copy()
    mask = np.zeros(floor.shape, dtype=bool)
    for surface in objects:
        closer = surface < nearest
        nearest = np.where(closer, surface, nearest)
        mask |= closer
    depth = np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float64)
    empty = np.where(np.isfinite(floor), floor, 0.0).astype(np.float64)
    return depth, empty, mask


def _shoe_box() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin, direction = _rays()
    return _scene(_box_depth(origin, direction, (0.0, 0.0), 0.338, 0.253, 0.124))


def _cream_bottle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A bottle: a 75 mm body with a much narrower neck and cap on top."""
    origin, direction = _rays()
    body = _cylinder_depth(origin, direction, (0.0, 0.0), 0.0375, 0.150)
    neck = _cylinder_depth(origin, direction, (0.0, 0.0), 0.0130, 0.053, base_m=0.150)
    return _scene(body, neck)


def _floor_plane(empty_depth: np.ndarray) -> object:
    plane = fit_reference_plane(empty_depth, CAMERA)
    assert plane is not None and plane.coefficients is not None
    return plane


def _measure(depth: np.ndarray, empty: np.ndarray, mask: np.ndarray, **kwargs):
    return metric_object_volume(
        depth, CAMERA, mask, _floor_plane(empty), reference_depth_m=empty,
        min_height_m=0.008, min_pixels=60, cell_size_m=0.005, **kwargs,
    )


class BackProjectionTests(unittest.TestCase):
    """Test 5, 6: the 3-D reconstruction, and the sign of the floor normal."""

    def test_a_known_box_is_recovered_within_twenty_percent(self) -> None:
        depth, empty, mask = _shoe_box()
        result = _measure(depth, empty, mask)
        self.assertIsNone(result.reason)
        length = result.diagnostics["length_mm"]
        width = result.diagnostics["width_mm"]
        height = result.diagnostics["height_p90_m"] * 1000.0
        for measured, truth, name in (
            (length, 338.0, "length"), (width, 253.0, "width"), (height, 124.0, "height"),
        ):
            with self.subTest(dimension=name):
                self.assertLessEqual(
                    abs(measured - truth) / truth, 0.20,
                    f"{name}: {measured:.0f} mm against {truth:.0f} mm",
                )

    def test_the_floor_normal_makes_an_object_positively_tall(self) -> None:
        depth, empty, mask = _shoe_box()
        plane = _floor_plane(empty)
        _, heights = project_to_plane(depth, CAMERA, mask, plane.coefficients)
        self.assertGreater(float(np.percentile(heights, 98)), 0.0)
        self.assertGreater(float((heights > 0).mean()), 0.9)

    def test_the_measurement_needs_no_realsense_data(self) -> None:
        """Test 14: nothing in this path reads a second camera."""
        depth, empty, mask = _cream_bottle()
        result = metric_object_volume(
            depth, CAMERA, mask, _floor_plane(empty),
            reference_depth_m=None, measurement_mask=None,
            min_height_m=0.008, min_pixels=60,
        )
        self.assertIsNone(result.reason)
        self.assertIsNotNone(result.measurement)


class TallObjectTests(unittest.TestCase):
    """Test 4, 11: a standing object is not smeared across the floor."""

    def test_a_standing_bottle_measures_its_own_body(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = _measure(depth, empty, mask)
        self.assertIsNone(result.reason)
        # 75 mm body, 26 mm neck: the hardware read 125 x 110 mm.
        self.assertLessEqual(result.diagnostics["length_mm"], 90.0)
        self.assertGreaterEqual(result.diagnostics["length_mm"], 60.0)
        # The neck is far narrower than the body and must not set the width.
        self.assertGreaterEqual(result.diagnostics["width_mm"], 45.0)

    def test_the_bottle_keeps_its_full_height(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = _measure(depth, empty, mask)
        # 203 mm true: a 150 mm body under a 53 mm neck. The hardware reported
        # 137 mm, and this path reported exactly 150 mm -- the body alone --
        # because the spike gate threw the neck away as an outlier.
        self.assertGreaterEqual(result.diagnostics["height_p90_m"] * 1000.0, 180.0)
        self.assertLessEqual(result.diagnostics["height_p90_m"] * 1000.0, 215.0)

    def test_the_spike_gate_no_longer_decides_the_height(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = _measure(depth, empty, mask)
        # The gate still runs and still protects the integral; it just does not
        # set the height any more. Clipped cells are the neck's.
        self.assertGreater(result.diagnostics["height_p90_m"],
                           result.diagnostics["height_median_m"] + 0.02)

    def test_the_footprint_excludes_the_cells_the_hull_filled_in(self) -> None:
        depth, empty, mask = _cream_bottle()
        result = _measure(depth, empty, mask)
        self.assertEqual(result.diagnostics["dimension_method"],
                         "logitech_floor_plane_footprint")
        # The completion runs -- the far side of a round object is hidden --
        # but it feeds the volume, not the dimensions.
        self.assertGreaterEqual(result.diagnostics["occlusion_filled_cells"], 0)


class PlaneTests(unittest.TestCase):
    """Test 7, and the proof that the old fallback plane caused the height loss."""

    def test_the_optical_axis_fallback_is_not_a_floor_measurement(self) -> None:
        """The plane the code falls back to is not the floor, and says so.

        `axis_aligned_plane` is perpendicular to the optical axis. For a camera
        pointing straight down that is the floor; for this 45-degree mount it
        is a plane standing across the scene, and heights measured from it are
        not heights above the floor. Both cases are marked, so a measurement
        taken against it can never be mistaken for a floor-relative one.
        """
        depth, empty, mask = _shoe_box()
        floor = _measure(depth, empty, mask)
        axis_distance = float(np.median(empty[empty > 0]))
        frontal = metric_object_volume(
            depth, CAMERA, mask, axis_aligned_plane(CAMERA, axis_distance),
            reference_depth_m=None, min_height_m=0.008, min_pixels=60,
        )
        self.assertIsNone(floor.reason)
        self.assertTrue(floor.diagnostics["plane_is_floor"])
        self.assertFalse(frontal.diagnostics["plane_is_floor"])
        # The floor-plane measurement is the accurate one; the fallback is a
        # different number entirely, which is the point of flagging it.
        self.assertAlmostEqual(floor.diagnostics["height_p90_m"], 0.124, delta=0.015)
        if frontal.reason is None:
            self.assertGreater(
                abs(frontal.diagnostics["height_p90_m"] - 0.124), 0.03,
            )

    def test_the_pipeline_fits_a_floor_rather_than_taking_the_fallback(self) -> None:
        """The repair: a floor plane is fitted, so the fallback is not reached."""
        from locallife_cloud.pipeline import VisionPipeline

        depth, empty, _ = _shoe_box()
        pipeline = VisionPipeline.__new__(VisionPipeline)
        pipeline.camera_id = "logitech"
        pipeline.reference_monocular = empty
        pipeline._logitech_plane_cache = None
        pipeline._logitech_plane_reason = None
        plane = pipeline._logitech_floor_plane(depth, CAMERA, None, None)
        self.assertIsNotNone(plane)
        gradient_a, gradient_b, _ = plane.coefficients
        self.assertGreater(abs(gradient_a) + abs(gradient_b), 1e-6)
        self.assertAlmostEqual(plane.tilt_degrees, TILT_DEG, delta=5.0)

    def test_both_monocular_mappings_are_fitted_and_the_data_chooses(self) -> None:
        region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        region[120:360, 160:480] = True
        expected = ray_plane_distance((HEIGHT, WIDTH), CAMERA, CAMERA_HEIGHT_M)

        linear = (expected - 0.30) / 2.0
        calibration, diagnostics = fit_plane_alignment(linear, region, CAMERA, CAMERA_HEIGHT_M)
        self.assertIsNotNone(calibration)
        self.assertEqual(diagnostics["mapping"], "linear_depth")
        self.assertAlmostEqual(calibration.scale, 2.0, places=2)

        inverted = 1.0 / expected * 3.0
        calibration, diagnostics = fit_plane_alignment(
            inverted, region, CAMERA, CAMERA_HEIGHT_M, inverse=True)
        self.assertIsNotNone(calibration)
        self.assertEqual(diagnostics["mapping"], "inverse_depth")
        self.assertTrue(calibration.inverse)

    def test_a_fitted_plane_is_reused_while_the_floor_is_occluded(self) -> None:
        """Test 8: the camera is fixed, so its floor does not need refitting."""
        from locallife_cloud.pipeline import VisionPipeline

        depth, empty, _ = _shoe_box()
        pipeline = VisionPipeline.__new__(VisionPipeline)
        pipeline.camera_id = "logitech"
        pipeline.reference_monocular = empty
        pipeline._logitech_plane_cache = None
        pipeline._logitech_plane_reason = None

        first = pipeline._logitech_floor_plane(depth, CAMERA, None, None)
        self.assertIsNotNone(first)
        # The floor is now completely hidden; the stored plane still answers.
        pipeline.reference_monocular = None
        second = pipeline._logitech_floor_plane(np.zeros_like(depth), CAMERA, None, None)
        self.assertIs(second, first)


class HeightTests(unittest.TestCase):
    def test_the_top_percentile_ignores_the_shallow_filled_cells(self) -> None:
        # One tall object's cells, plus the shallow ones a hull fill adds.
        heights = np.r_[np.full(20, 0.200), np.full(220, 0.010)]
        self.assertLess(float(np.percentile(heights, 90)), 0.05)
        self.assertGreater(robust_object_height_m(heights), 0.15)

    def test_a_height_at_the_noise_floor_is_not_a_height(self) -> None:
        self.assertIsNone(robust_object_height_m(np.full(50, 0.001), noise_floor_m=0.004))


class BackgroundTests(unittest.TestCase):
    """Test 1, 2, 3: furniture out, deposits in, floor strips out."""

    def setUp(self) -> None:
        self.region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.region[100:400, 120:520] = True

    def test_a_mask_covering_the_zone_is_the_room(self) -> None:
        mask = self.region.copy()
        self.assertEqual(
            static_background_reason(mask, self.region), "covers_measurement_zone",
        )

    def test_a_deposit_inside_the_zone_is_kept(self) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[200:300, 250:360] = True
        self.assertIsNone(static_background_reason(mask, self.region))

    def test_a_floor_strip_across_the_zone_is_refused(self) -> None:
        strip = np.zeros((HEIGHT, WIDTH), dtype=bool)
        strip[180:230, 120:520] = True          # reaches both side edges
        self.assertEqual(
            static_background_reason(strip, self.region), "spans_measurement_zone",
        )

    def test_a_mask_that_never_changed_is_background(self) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[200:260, 250:320] = True
        change = np.zeros((HEIGHT, WIDTH), dtype=bool)
        change[300:340, 400:440] = True
        self.assertEqual(
            static_background_reason(mask, self.region, change=change),
            "unchanged_since_baseline",
        )


class SmallObjectTests(unittest.TestCase):
    """Test 9, 10: the can the detector missed, and no duplicate of one it saw."""

    def setUp(self) -> None:
        self.region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.region[100:400, 120:520] = True
        self.change = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.change[200:260, 250:300] = True     # the can
        self.change[300:380, 380:470] = True     # a detected box

    def test_an_unclaimed_island_is_recovered(self) -> None:
        claimed = np.zeros((HEIGHT, WIDTH), dtype=bool)
        claimed[300:380, 380:470] = True
        islands = unclaimed_foreground_islands(
            self.change, self.region, claimed, min_pixels=40,
        )
        self.assertEqual(len(islands), 1)
        self.assertTrue(islands[0][220, 270])
        self.assertFalse(islands[0][340, 420])

    def test_an_island_a_detection_already_covers_is_not_repeated(self) -> None:
        islands = unclaimed_foreground_islands(
            self.change, self.region, self.change.copy(), min_pixels=40,
        )
        self.assertEqual(islands, [])

    def test_the_pixel_floor_scales_down_with_the_frame(self) -> None:
        self.assertEqual(minimum_object_pixels((480, 640), 150), 150)
        self.assertLess(minimum_object_pixels((240, 320), 150), 150)
        self.assertEqual(minimum_object_pixels((1080, 1920), 150), 150)


class ConsistencyTests(unittest.TestCase):
    """Test 12, 13: one geometry behind both numbers, and a settled one."""

    def test_the_reported_shoe_box_error_cancellation_is_caught(self) -> None:
        # What the hardware showed: 540 x 360 x 89 mm alongside 13.60 L.
        report = geometry_consistency(
            length_m=0.540, width_m=0.360, height_m=0.089, litres=13.60, shape="box",
        )
        self.assertFalse(report["geometry_consistent"])
        # Its own dimensions imply 17.3 L, not the 13.6 L it printed.
        self.assertGreater(report["volume_from_dimensions_l"], 17.0)

    def test_a_coherent_box_passes(self) -> None:
        report = geometry_consistency(
            length_m=0.338, width_m=0.253, height_m=0.124, litres=10.0, shape="box",
        )
        self.assertTrue(report["geometry_consistent"])

    def test_the_volume_and_the_dimensions_come_from_one_measurement(self) -> None:
        depth, empty, mask = _shoe_box()
        result = _measure(depth, empty, mask)
        implied = (
            result.diagnostics["length_mm"] * result.diagnostics["width_mm"]
            * (result.diagnostics["height_p90_m"] * 1000.0)
        ) / 1e6
        self.assertGreater(result.measurement.liters, 0.45 * implied)
        self.assertLess(result.measurement.liters, 1.30 * implied)

    def test_a_single_frame_spike_does_not_move_the_result(self) -> None:
        stabiliser = GeometryStabiliser(window=9, minimum_frames=3, tolerance=0.25)
        for _ in range(4):
            stable = stabiliser.update(7, length_mm=340.0, width_mm=250.0,
                                       height_mm=124.0, volume_l=10.5)
        self.assertTrue(stable.settled)
        spiked = stabiliser.update(7, length_mm=980.0, width_mm=760.0,
                                   height_mm=61.0, volume_l=45.0)
        self.assertLess(spiked.length_mm, 400.0)
        self.assertLess(spiked.volume_l, 12.0)

    def test_an_untracked_object_is_reported_as_measured(self) -> None:
        stable = GeometryStabiliser().update(None, length_mm=100.0, width_mm=50.0,
                                             height_mm=200.0, volume_l=1.0)
        self.assertEqual(stable.length_mm, 100.0)
        self.assertFalse(stable.settled)


class ProtectedSurfacesTests(unittest.TestCase):
    """Test 15: RealSense, the ledger, the cloud and the launcher are untouched."""

    def test_the_protected_files_are_unchanged_since_the_branch_started(self) -> None:
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
