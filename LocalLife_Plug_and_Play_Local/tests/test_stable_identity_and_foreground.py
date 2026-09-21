"""Permanent measurement identity, and what may become a mesh.

Two field failures drive these. A stationary object was renumbered by the
detector (a pillow 10 -> 52, a can 54 -> 63) and each new number produced
another "measurement". And dark floor, shadow and furniture were meshed and
reported as 4-6 L of waste.
"""

from __future__ import annotations

import unittest

import numpy as np

from locallife_cloud.foreground_gate import (
    BACKGROUND,
    CALIBRATION_CHANGED,
    INSUFFICIENT_NEW_VOLUME,
    LOW_VALID_DEPTH,
    OUTSIDE_ROI,
    ForegroundSettings,
    has_new_volume,
    validate_foreground,
)
from locallife_cloud.stable_identity import (
    Observation,
    StabilitySettings,
    StableObjectRegistry,
)


def _observation(t: float, x: float = 100.0, y: float = 100.0, volume: float = 5.0,
                 track_id: int | None = 1, depth_mm: float = 300.0, size: float = 60.0):
    return Observation(
        timestamp=t, centroid=(x, y), box=(x - size, y - size, x + size, y + size),
        depth_mm=depth_mm, volume_l=volume, length_mm=400.0, width_mm=300.0,
        height_mm=depth_mm, label="bag", color="black", material="plastic",
        sorting_status="correct", confidence=0.9, detector_track_id=track_id,
    )


def _settle(registry: StableObjectRegistry, frames, start: float = 0.0):
    """Feed a sequence of observations; return the frozen result, if any."""
    frozen = None
    for index, observation in enumerate(frames):
        item = registry.observe(observation)
        result = registry.accept(item)
        if result is not None:
            frozen = result
            registry.mark_persisted(item)
            registry.mark_committed(item)
    return frozen


class StableIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = StableObjectRegistry(StabilitySettings(
            window_frames=8, min_valid_stable_frames=4, finalisation_hold_seconds=0.3,
        ))

    def _stationary(self, count: int, track_ids=None, **kwargs):
        ids = track_ids or [1] * count
        return [
            _observation(index * 0.1, track_id=ids[index % len(ids)], **kwargs)
            for index in range(count)
        ]

    # 11 / 12
    def test_a_renumbered_stationary_object_keeps_one_permanent_event(self) -> None:
        # The pillow: detector says 10 for a while, then 52, same object.
        frames = self._stationary(6, track_ids=[10, 10, 10, 52, 52, 52])
        frozen = _settle(self.registry, frames)
        self.assertIsNotNone(frozen)
        self.assertEqual(frozen["event_id"], 1)
        self.assertEqual(self.registry.accepted_count(), 1)
        self.assertEqual(len(self.registry.objects), 1)
        # The renumbering is recorded rather than hidden.
        self.assertEqual(frozen["detector_track_ids"], [10, 52])

    def test_a_can_renumbered_after_acceptance_creates_no_second_event(self) -> None:
        frames = self._stationary(6, track_ids=[54] * 6)
        self.assertIsNotNone(_settle(self.registry, frames))
        # It keeps sitting there, now called 63.
        for index in range(8):
            item = self.registry.observe(_observation(1.0 + index * 0.1, track_id=63))
            self.assertIsNone(self.registry.accept(item))
        self.assertEqual(self.registry.accepted_count(), 1)

    # 13
    def test_small_position_and_mask_jitter_does_not_create_a_new_event(self) -> None:
        frames = [
            _observation(index * 0.1, x=100 + (index % 3), y=100 - (index % 2),
                         size=60 + (index % 2))
            for index in range(8)
        ]
        self.assertIsNotNone(_settle(self.registry, frames))
        self.assertEqual(self.registry.accepted_count(), 1)
        self.assertEqual(len(self.registry.objects), 1)

    # 14
    def test_a_genuinely_new_deposit_gets_the_next_event(self) -> None:
        self.assertIsNotNone(_settle(self.registry, self._stationary(6)))
        second = [
            _observation(1.0 + index * 0.1, x=600.0, y=600.0, volume=3.0, track_id=77)
            for index in range(6)
        ]
        frozen = _settle(self.registry, second)
        self.assertIsNotNone(frozen)
        self.assertEqual(frozen["event_id"], 2)
        self.assertEqual(self.registry.accepted_count(), 2)

    # 15
    def test_completed_dimensions_and_litres_are_frozen(self) -> None:
        frames = self._stationary(6)
        frozen = _settle(self.registry, frames)
        self.assertAlmostEqual(frozen["volume_l"], 5.0, places=3)
        # Later noisy frames must not move the committed values.
        for index in range(6):
            item = self.registry.observe(_observation(2.0 + index * 0.1, volume=99.0))
            self.registry.accept(item)
        committed = self.registry.committed_objects()[0]
        self.assertAlmostEqual(committed.frozen["volume_l"], 5.0, places=3)
        self.assertEqual(committed.frozen["object_type"], "bag")

    def test_the_frozen_value_is_robust_not_the_last_frame(self) -> None:
        frames = [_observation(index * 0.1, volume=5.0) for index in range(5)]
        frames.append(_observation(0.5, volume=40.0))  # one badly segmented frame
        frozen = _settle(self.registry, frames)
        self.assertIsNotNone(frozen)
        self.assertLess(frozen["volume_l"], 6.0)

    def test_a_single_frame_never_finalises(self) -> None:
        item = self.registry.observe(_observation(0.0))
        self.assertIsNone(self.registry.accept(item))
        self.assertEqual(item.rejection_reason, "insufficient_stable_frames")

    def test_a_moving_object_is_not_finalised(self) -> None:
        frames = [_observation(index * 0.1, x=100 + index * 300) for index in range(8)]
        self.assertIsNone(_settle(self.registry, frames))

    def test_a_varying_volume_is_not_finalised(self) -> None:
        frames = [
            _observation(index * 0.1, volume=5.0 + index * 2.0) for index in range(8)
        ]
        self.assertIsNone(_settle(self.registry, frames))
        self.assertEqual(self.registry.accepted_count(), 0)

    def test_the_hold_time_must_elapse(self) -> None:
        # Six frames, but all inside 0.05 s: stable-looking, not yet held.
        frames = [_observation(index * 0.01) for index in range(6)]
        self.assertIsNone(_settle(self.registry, frames))

    def test_an_object_that_leaves_releases_its_slot(self) -> None:
        _settle(self.registry, self._stationary(6))
        self.assertEqual(len(self.registry.objects), 1)
        released = self.registry.release_absent(now=100.0)
        self.assertEqual(len(released), 1)
        self.assertEqual(self.registry.objects, {})
        # Permanent ids are never reused after a release.
        self.assertEqual(self.registry.accepted_count(), 1)
        frozen = _settle(self.registry, [
            _observation(200.0 + index * 0.1) for index in range(6)
        ])
        self.assertEqual(frozen["event_id"], 2)

    def test_the_settings_are_reported_for_the_dashboard(self) -> None:
        state = self.registry.state()
        self.assertIn("STABILITY_WINDOW_FRAMES", state["settings"])
        self.assertEqual(state["accepted_events"], 0)

    def test_an_impossible_window_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            StabilitySettings(window_frames=4, min_valid_stable_frames=9).validate()


class ForegroundGateTests(unittest.TestCase):
    """Only geometry may create a mesh -- never darkness or class alone."""

    def setUp(self) -> None:
        self.shape = (200, 200)
        self.plane = np.full(self.shape, 1.500)  # support plane 1.5 m away
        self.settings = ForegroundSettings()

    def _mask(self, size: int = 60, offset: int = 60) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        mask[offset:offset + size, offset:offset + size] = True
        return mask

    def _depth_with_object(self, mask: np.ndarray, height_m: float) -> np.ndarray:
        depth = self.plane.copy()
        depth[mask] = 1.500 - height_m
        return depth

    # 21
    def test_a_real_object_inside_the_roi_is_accepted(self) -> None:
        mask = self._mask()
        result = validate_foreground(mask, self._depth_with_object(mask, 0.12), self.plane)
        self.assertTrue(result.accepted, result.reason)
        self.assertAlmostEqual(result.detail["height_m"], 0.12, places=2)

    # 22
    def test_a_flat_flexible_bag_is_not_filtered_away(self) -> None:
        # Only 3 cm proud of the plane -- a slumped bag, and it must survive.
        mask = self._mask()
        result = validate_foreground(mask, self._depth_with_object(mask, 0.03), self.plane)
        self.assertTrue(result.accepted, result.reason)

    # 17
    def test_an_empty_scene_produces_nothing(self) -> None:
        empty = np.zeros(self.shape, dtype=bool)
        self.assertFalse(validate_foreground(empty, self.plane, self.plane).accepted)

    # 18 / 20
    def test_the_support_plane_itself_is_rejected_however_dark(self) -> None:
        # The mask is exactly the floor: no height, so no object, whatever the
        # detector thought of its colour.
        mask = self._mask()
        result = validate_foreground(mask, self.plane.copy(), self.plane)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, BACKGROUND)

    # 19
    def test_a_lighting_change_with_no_geometry_is_rejected(self) -> None:
        # A shadow moves across the floor: depth is unchanged.
        mask = self._mask(size=120, offset=40)
        depth = self.plane.copy()
        result = validate_foreground(mask, depth, self.plane)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, BACKGROUND)

    def test_a_background_scale_mask_is_rejected(self) -> None:
        mask = np.ones(self.shape, dtype=bool)
        result = validate_foreground(mask, self._depth_with_object(mask, 0.3), self.plane)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "background_scale_mask")

    def test_a_component_outside_the_measurement_roi_is_rejected(self) -> None:
        roi = np.zeros(self.shape, dtype=bool)
        roi[0:40, 0:40] = True
        mask = self._mask()
        result = validate_foreground(
            mask, self._depth_with_object(mask, 0.2), self.plane, roi_mask=roi,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, OUTSIDE_ROI)

    def test_missing_depth_never_creates_an_object(self) -> None:
        mask = self._mask()
        self.assertEqual(
            validate_foreground(mask, None, None).reason, LOW_VALID_DEPTH,
        )
        holes = self._depth_with_object(mask, 0.2)
        holes[mask] = 0.0  # every pixel invalid
        self.assertEqual(
            validate_foreground(mask, holes, self.plane).reason, LOW_VALID_DEPTH,
        )

    def test_a_tiny_speck_is_rejected(self) -> None:
        mask = np.zeros(self.shape, dtype=bool)
        mask[10:14, 10:14] = True
        self.assertEqual(
            validate_foreground(mask, self._depth_with_object(mask, 0.3), self.plane).reason,
            "component_too_small",
        )

    def test_a_thin_sliver_is_implausible(self) -> None:
        mask = np.zeros(self.shape, dtype=bool)
        mask[10:14, 10:190] = True  # a bin edge or a cable
        result = validate_foreground(mask, self._depth_with_object(mask, 0.2), self.plane)
        self.assertFalse(result.accepted)

    # 9: calibration protection
    def test_an_invalid_calibration_blocks_measurement(self) -> None:
        mask = self._mask()
        result = validate_foreground(
            mask, self._depth_with_object(mask, 0.2), self.plane, calibration_valid=False,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, CALIBRATION_CHANGED)

    def test_an_insignificant_scene_change_is_not_a_new_deposit(self) -> None:
        self.assertFalse(has_new_volume(0.01).accepted)
        self.assertEqual(has_new_volume(None).reason, INSUFFICIENT_NEW_VOLUME)
        self.assertTrue(has_new_volume(1.4).accepted)


if __name__ == "__main__":
    unittest.main()
