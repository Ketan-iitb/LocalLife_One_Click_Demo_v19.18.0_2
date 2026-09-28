"""V38: a moved Logitech stops measuring, material scores are fair, orange stays orange.

Synthetic frames only; these say what the code does, not how accurate the real
cameras are.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locallife_cloud.colour_evidence import describe_colour
from locallife_cloud.logitech_pose import ViewChangeGuard
from locallife_cloud.material_evidence import reconcile_material
from locallife_cloud.material_fair import balanced_label_scores
from locallife_cloud.material_siglip import aggregate_label_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))

H, W = 120, 160


def _room(seed: int = 0) -> np.ndarray:
    """A textured room: floor pattern, furniture edges, a bright stripe."""
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), 90, dtype=np.uint8)
    frame[:, :, 0] = 110
    for row in range(0, H, 6):
        frame[row:row + 2] = (70, 80, 85)
    frame[10:60, 20:30] = (40, 60, 120)            # a chair leg
    frame[5:25, 90:150] = (200, 200, 200)          # a bright rug stripe
    frame[70:110, 120:128] = (30, 30, 30)          # a table edge
    noise = rng.integers(-6, 7, size=frame.shape)
    return np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)


def _moved(frame: np.ndarray, dx: int = 25, dy: int = 18) -> np.ndarray:
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)


class ViewGuardTests(unittest.TestCase):
    def test_a_moved_camera_is_reported_after_consecutive_frames(self) -> None:
        guard, reference = ViewChangeGuard(), _room()
        moved = _moved(reference)
        self.assertEqual(guard.update(moved, reference), "camera_view_changed_since_empty_scene")

    def test_a_borderline_view_needs_consecutive_frames(self) -> None:
        from locallife_cloud import logitech_pose

        guard, reference = ViewChangeGuard(), _room()
        moved = _moved(reference)
        original = logitech_pose.VIEW_IMMEDIATE_CORRELATION
        logitech_pose.VIEW_IMMEDIATE_CORRELATION = -1.0      # treat the drop as borderline
        try:
            self.assertIsNone(guard.update(moved, reference))
            self.assertIsNone(guard.update(moved, reference))
            self.assertEqual(guard.update(moved, reference), "camera_view_changed_since_empty_scene")
        finally:
            logitech_pose.VIEW_IMMEDIATE_CORRELATION = original

    def test_a_lighting_change_is_not_a_move(self) -> None:
        guard, reference = ViewChangeGuard(), _room()
        darker = (reference.astype(np.float32) * 0.6).astype(np.uint8)
        for _ in range(4):
            self.assertIsNone(guard.update(darker, reference))
        self.assertGreater(guard.last["edge_correlation"], 0.8)

    def test_a_deposited_object_is_not_a_move(self) -> None:
        guard, reference = ViewChangeGuard(), _room()
        frame = reference.copy()
        mask = np.zeros((H, W), dtype=bool)
        mask[40:100, 40:110] = True
        frame[mask] = (30, 30, 200)
        for _ in range(4):
            self.assertIsNone(guard.update(frame, reference, exclude=mask))

    def test_a_new_reference_accepts_the_new_view(self) -> None:
        guard, reference = ViewChangeGuard(), _room()
        moved = _moved(reference)
        guard.update(moved, reference)
        self.assertIsNotNone(guard.reason)
        new_reference = moved.copy()
        self.assertIsNone(guard.update(moved, new_reference))

    def test_a_featureless_reference_proves_nothing(self) -> None:
        guard = ViewChangeGuard()
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        for _ in range(4):
            self.assertIsNone(guard.update(_room(), blank))
        self.assertEqual(guard.last["state"], "too_little_structure_to_compare")


class _RoomDepth:
    """Metric-checkpoint stand-in: 1.5 m floor, the red object 12 cm closer."""

    device = "cpu"

    def estimate_batch(self, frames):
        return [np.where((frame[:, :, 2] == 210) & (frame[:, :, 0] == 40), 1.38, 1.5).astype(np.float32)
                for frame in frames]


class MovedCameraPipelineTests(unittest.TestCase):
    def test_measurements_pause_after_a_move_and_resume_after_recapture(self) -> None:
        from locallife_cloud.types import CameraIntrinsics, Detection
        from test_v31_logitech_measurement_cascade import _station

        camera = CameraIntrinsics(fx=160, fy=160, ppx=80, ppy=60, width=W, height=H)
        with tempfile.TemporaryDirectory() as directory:
            manager, detector, _, _ = _station(directory, depth=_RoomDepth(),
                                               logitech_reference_distance_m=1.5)
            logitech = manager.camera("logitech")
            room = _room()
            detector.items = []
            logitech.process_frame(room, intrinsics=camera, persist=False, timestamp=40.0)
            logitech.set_baseline()

            clock = [50.0]

            def with_object(background, x):
                # A new deposit each time, at its own place: a finalised
                # track keeps republishing its own record by design.
                frame = background.copy()
                mask = np.zeros((H, W), dtype=bool)
                mask[50:90, x:x + 30] = True
                frame[mask] = (40, 40, 210)
                detector.items = [Detection("cosmetic bottle", 0.7, (x, 50, x + 30, 90), mask)]
                result = None
                for _ in range(6):
                    clock[0] += 1.0
                    result = logitech.process_frame(frame, intrinsics=camera, timestamp=clock[0])
                detector.items = []
                for _ in range(8):                       # it is taken away again
                    clock[0] += 1.0
                    logitech.process_frame(background, intrinsics=camera, timestamp=clock[0], persist=False)
                return result.detections[0]

            same_pose = with_object(room, 20)
            self.assertIsNotNone(same_pose.monocular_volume_l)

            moved_room = _moved(room)
            moved = with_object(moved_room, 60)
            self.assertEqual(moved.volume_rejection_reason, "camera_view_changed_since_empty_scene")
            self.assertIsNone(moved.monocular_volume_l)

            # The operator clears the zone and recaptures in the new pose.
            detector.items = []
            logitech.process_frame(moved_room, intrinsics=camera, persist=False, timestamp=clock[0] + 1)
            logitech.set_baseline()
            again = with_object(moved_room, 100)
            self.assertIsNone(logitech.logitech_view_reason)
            self.assertIsNotNone(again.monocular_volume_l)


class MaterialScoreTests(unittest.TestCase):
    PROMPTS = (["polythene bag"] * 6 + ["plastic"] * 4 + ["paper"] * 4 + ["mixed or general waste"] * 3)

    def test_the_old_sum_favoured_the_label_with_most_prompts(self) -> None:
        # An image equally similar to every prompt: no evidence at all.
        flat = np.zeros(len(self.PROMPTS))
        label, score, _ = aggregate_label_scores(flat, self.PROMPTS)
        self.assertEqual(label, "polythene bag")
        self.assertAlmostEqual(score, 6 / 17, places=3)

    def test_per_label_scoring_abstains_on_no_evidence(self) -> None:
        label, score, margin, scores = balanced_label_scores(np.zeros(len(self.PROMPTS)), self.PROMPTS)
        self.assertEqual(label, "unknown")
        self.assertAlmostEqual(margin, 0.0)
        self.assertAlmostEqual(scores["polythene bag"], scores["mixed or general waste"])

    def test_a_clear_image_still_wins(self) -> None:
        logits = np.array([0.0] * 6 + [4.0] * 4 + [0.0] * 4 + [0.0] * 3)
        label, score, margin, _ = balanced_label_scores(logits, self.PROMPTS)
        self.assertEqual(label, "plastic")
        self.assertGreater(margin, 0.5)

    def test_published_confidence_is_not_vote_agreement_alone(self) -> None:
        material, confidence, evidence = reconcile_material(
            "box", ["cardboard"] * 4, scores=[("cardboard", 0.4)] * 4)
        self.assertEqual(material, "cardboard")
        self.assertAlmostEqual(evidence["agreement"], 1.0)
        self.assertAlmostEqual(confidence, 0.4)        # was published as 1.0 ("100 %")
        self.assertIn("not a calibrated probability", evidence["confidence_meaning"])


def _object(size=(120, 120)):
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (110, 115, 95)                       # grey-blue mat
    mask = np.zeros(size, dtype=bool)
    mask[20:100, 35:85] = True
    return frame, mask


class ColourTests(unittest.TestCase):
    def test_a_shaded_orange_container_with_a_red_cap_is_orange(self) -> None:
        frame, mask = _object()
        frame[mask] = (30, 110, 235)                   # orange body
        frame[20:100, 70:85][mask[20:100, 70:85]] = (20, 70, 170)   # its shaded side
        frame[20:34, 35:85] = (35, 40, 190)            # a red cap
        evidence = describe_colour(frame, mask, label="can")
        self.assertEqual(evidence.colour, "orange")
        self.assertEqual(evidence.accent, "red")

    def test_red_stays_red(self) -> None:
        frame, mask = _object()
        frame[mask] = (35, 35, 200)
        self.assertEqual(describe_colour(frame, mask, label="can").colour, "red")

    def test_a_mask_that_leaks_onto_the_mat_keeps_the_object_colour(self) -> None:
        frame, mask = _object()
        frame[mask] = (30, 110, 235)
        leaked = mask.copy()
        leaked[16:104, 31:89] = True                   # a 4 px ring of mat inside the mask
        self.assertEqual(describe_colour(frame, leaked, label="can").colour, "orange")

    def test_an_open_bag_is_its_skin_and_a_solid_object_its_body(self) -> None:
        frame, mask = _object()
        frame[mask] = (40, 160, 40)                    # green skin ...
        frame[35:85, 45:75] = (20, 120, 230)           # ... orange contents showing in the middle
        self.assertEqual(describe_colour(frame, mask, label="plastic garbage bag").colour, "green")
        # The same pixels on a solid object: its visible surface is mostly... still
        # measured over the whole body, not a band.
        solid = describe_colour(frame, mask, label="toy")
        self.assertEqual(solid.sampled_region, "surface")


if __name__ == "__main__":
    unittest.main()
