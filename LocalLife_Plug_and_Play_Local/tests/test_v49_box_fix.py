"""V49 box fix: one resolved class per track, per-camera confidence floor, coherent geometry/records.

Synthetic scenes only: these check recognition logic and record consistency, not physical accuracy.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from locallife_cloud import session_deposits as sd  # noqa: E402
from locallife_cloud.comparison import DualCameraCoordinator  # noqa: E402
from locallife_cloud.config import AppConfig  # noqa: E402
from locallife_cloud.object_class import ClassResolver  # noqa: E402
from locallife_cloud.pipeline import apply_confidence_floor  # noqa: E402
from locallife_cloud.types import CameraIntrinsics, Detection  # noqa: E402

BOX_LABELS = (("book", 0.40), ("cardboard shipping box", 0.33), ("carton box", 0.30), ("pillow", 0.28))


class ResolverTests(unittest.TestCase):
    def _feed(self, frames, rigid=False, size=None):
        resolver = ClassResolver()
        for candidates in frames:
            resolver.update(1, candidates)
        if rigid:
            resolver.set_rigid(1, True)
        if size:
            resolver.set_size(1, *size)
        return resolver.resolve(1)

    def test_flickering_box_resolves_to_cardboard_box_with_its_own_size(self):
        frames = [BOX_LABELS if i % 2 else (("pillow", 0.42), ("cardboard shipping box", 0.35), ("book", 0.3))
                  for i in range(12)]
        name, share, _ = self._feed(frames, rigid=True, size=(43.0, 7.0))
        self.assertEqual(name, "cardboard box")
        self.assertGreaterEqual(share, 0.45)

    def test_a_book_sized_book_stays_a_book(self):
        name, _, _ = self._feed([BOX_LABELS] * 12, rigid=True, size=(24.0, 3.0))
        self.assertEqual(name, "book")

    def test_a_pillow_without_rigid_evidence_stays_a_pillow(self):
        name, _, _ = self._feed([(("pillow", 0.5), ("cardboard box", 0.2))] * 12)
        self.assertEqual(name, "pillow")

    def test_a_plastic_bag_misread_as_textile_now_and_then_stays_a_bag(self):
        frames = [(("plastic garbage bag", 0.45), ("textile item", 0.4))] * 8 + [(("textile item", 0.5),)] * 3
        name, _, _ = self._feed(frames)
        self.assertEqual(name, "plastic bag")

    def test_too_little_evidence_stays_unresolved(self):
        name, _, _ = self._feed([(("book", 0.3), ("pillow", 0.29), ("cardboard box", 0.28))])
        self.assertIsNone(name)


class ConfidenceFloorTests(unittest.TestCase):
    def _det(self, label, conf, rows=slice(10, 30)):
        mask = np.zeros((40, 40), bool)
        mask[rows, 10:30] = True
        return Detection(label, conf, (10, 10, 30, 30), mask, source="yoloe-segmentation")

    def test_realsense_set_is_unchanged_and_weak_proposals_become_evidence(self):
        strong, weak = self._det("book", 0.40), self._det("cardboard shipping box", 0.20)
        kept = apply_confidence_floor([strong, weak], 0.24)
        self.assertEqual(kept, [strong])
        self.assertIn(("cardboard shipping box", 0.20), strong.label_candidates)

    def test_shared_model_runs_at_the_lowest_camera_threshold(self):
        # Root cause: YOLOE ran at the RealSense 0.24 for BOTH cameras, so Logitech's 0.15 never applied.
        from unittest import mock
        from locallife_cloud import comparison
        seen = {}

        def fake(config):
            seen["conf"] = config.detector_confidence
            return _Detector()
        with TemporaryDirectory() as directory, mock.patch.object(comparison, "create_segmenter", fake):
            config = AppConfig(results_dir=Path(directory), enable_monocular_depth=False)
            comparison.DualCameraCoordinator(config, depth_estimator=_NoDepth())
            self.assertEqual(seen["conf"], min(config.detector_confidence, config.logitech_detector_confidence))
            self.assertEqual(config.detector_confidence, 0.24)                # RealSense keeps its own

    def test_logitech_keeps_its_own_lower_threshold(self):
        box = self._det("cardboard box", 0.18)
        self.assertEqual(apply_confidence_floor([box], 0.15), [box])


class _Detector:
    device = "cpu"
    runtime = {"device": "cpu"}

    def __init__(self):
        self.frames: list[list[Detection]] = []
        self.calls = 0

    def detect_batch(self, frames):
        items = self.frames[min(self.calls, len(self.frames) - 1)] if self.frames else []
        self.calls += 1
        return [[Detection(d.label, d.confidence, d.box, d.mask.copy(), source="yoloe-segmentation",
                           color=d.color) for d in items] for _ in frames]


class _NoDepth:
    def estimate_batch(self, frames):
        return [np.full(frame.shape[:2], 1.0, np.float32) for frame in frames]


K = CameraIntrinsics(fx=300.0, fy=300.0, ppx=160.0, ppy=120.0, width=320, height=240)


def _march(surface):
    rows, cols = np.mgrid[0:K.height, 0:K.width]
    xn, yn = (cols - K.ppx) / K.fx, (rows - K.ppy) / K.fy
    out = np.full(xn.shape, 1.10)
    for z in np.arange(0.20, 1.10, 0.002):
        hit = (out >= 1.10) & (surface(xn * z, yn * z) >= 1.10 - z)
        out[hit] = z
    return out.astype(np.float32)


def _leaning_box(deg=25.0, length=0.45, width=0.15, thick=0.07, pile=0.30):
    t = math.radians(deg)
    x0 = -length * math.cos(t) / 2
    xa = x0 - thick * math.sin(t)
    xb, xc = xa + length * math.cos(t), x0 + length * math.cos(t)

    def box(x, y):
        top = thick * math.cos(t) + (x - xa) * math.tan(t)
        end = thick * math.cos(t) + length * math.sin(t) - (x - xb) / math.tan(t)
        z = np.where((x >= xa) & (x < xb), top, np.where((x >= xb) & (x <= xc), end, -1))
        return np.where((np.abs(y) < width / 2) & (z > 0), z, 0.0)
    return box, pile


class RealSenseBoxEndToEndTests(unittest.TestCase):
    """A leaning 45 x 15 x 7 cm box whose detector label flickers book / pillow / shipping box."""

    def test_one_class_one_geometry_one_material_one_verdict(self):
        box, pile = _leaning_box()
        depth = _march(lambda x, y: pile + box(x, y))
        rows, cols = np.mgrid[0:K.height, 0:K.width]
        x, y = (cols - K.ppx) / K.fx * depth, (rows - K.ppy) / K.fy * depth
        mask = box(x, y) > 0.005
        frame = np.full((K.height, K.width, 3), 70, np.uint8)
        frame[mask] = (60, 110, 160)                                  # BGR brown
        r, c = np.nonzero(mask)
        bbox = (int(c.min()), int(r.min()), int(c.max()) + 1, int(r.max()) + 1)
        detector = _Detector()
        flicker = [[("book", 0.40), ("cardboard shipping box", 0.30), ("pillow", 0.27)],
                   [("pillow", 0.41), ("cardboard shipping box", 0.32), ("book", 0.30)],
                   [("cardboard shipping box", 0.39), ("book", 0.33), ("pillow", 0.25)]]
        detector.frames = [[Detection(lab, conf, bbox, mask, color="brown") for lab, conf in flicker[i % 3]]
                           for i in range(30)]
        with TemporaryDirectory() as directory:
            config = AppConfig(results_dir=Path(directory), roi=(0, 0, 1, 1), min_component_pixels=20,
                               tracker_confirm_frames=1, operating_mode="geometry_validation",
                               auto_deposit=False, research_mode="realsense_only",
                               enable_monocular_depth=False)
            manager = DualCameraCoordinator(config, detector=detector, depth_estimator=_NoDepth())
            station = manager.camera("realsense")
            empty = _march(lambda x, y: pile + 0 * x)
            station.fill.recalibrate(_march(lambda x, y: 0 * x), K, None)
            for i in range(24):
                result = station.process_frame(frame, depth_m=depth if i else empty, intrinsics=K,
                                               timestamp=100.0 + i, persist=False)
            tracked = [d for d in result.detections if d.track_id is not None]
        self.assertEqual(len(tracked), 1, [d.label for d in result.detections])
        item = tracked[0].to_dict()
        self.assertIn(item["raw_label"], {"book", "pillow", "cardboard shipping box"})   # raw kept as is
        self.assertEqual(item["resolved_label"], "cardboard box")
        self.assertEqual(item["material"], "cardboard")
        self.assertEqual(item["bin_sorting"], "mis_sort")
        length, width, height = item["support_length_cm"], item["support_width_cm"], item["support_height_cm"]
        # one coherent measurement: the volume IS the product of the shown dimensions (+-rounding)
        self.assertAlmostEqual(item["support_volume_l"], length * width * height / 1000.0, delta=0.06)
        self.assertTrue("slab" in item["support_method"] or "cuboid" in item["support_method"])
        # synthetic geometry sanity (not a physical-accuracy claim)
        self.assertLess(abs(length - 45.0), 5.0)
        self.assertLess(abs(height - 7.0), 3.0)


class DepositRecordTests(unittest.TestCase):
    """The deposit row takes class, colour, material and ONE geometry set from the same track."""

    def test_event_matches_the_live_track_and_late_values_update_the_same_event(self):
        from tests.test_v45_deposits_live import Scene, run  # noqa: E402
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            scene = Scene(camera="realsense")
            live = {"object_class": None, "colour": "grey", "material": "polythene bag",
                    "material_confidence": 0.7}
            original = scene.evidence

            def evidence(t, **kw):
                ev = original(t, **kw)
                for track in ev.tracks:
                    if track.track_id == 7:
                        for key, value in live.items():
                            setattr(track, key, value)
                        if live.get("object_class"):
                            track.support_length_cm, track.support_width_cm = 43.0, 14.5
                            track.support_height_cm, track.support_volume_l = 7.2, 4.49
                            track.support_method = "tilted rigid slab: top-face L x W x thickness"
                return ev
            scene.evidence = evidence
            run(counter, scene, 0.0, 7.0)
            scene.add(7, (40, 40, 90, 70), label="book")
            got, t = run(counter, scene, 7.0, 10.0)
            self.assertEqual(counter.count, 1)
            event = got[0]
            self.assertTrue(event["object_type"].startswith("unresolved"))   # not yet called a bag or a box
            live.update(object_class="cardboard box", colour="brown", material="cardboard",
                        material_confidence=0.8)
            run(counter, scene, t, t + 4.0)
            event = next(e for e in counter.events if e["event_id"] == got[0]["event_id"])
            self.assertEqual(counter.count, 1)                                 # metadata update, no new count
            self.assertEqual(len(counter.events), 1)
            self.assertEqual((event["object_type"], event["colour"], event["material"]),
                             ("cardboard box", "brown", "CARDBOARD"))
            self.assertEqual((event["length_cm"], event["width_cm"], event["height_cm"], event["envelope_l"]),
                             (43.0, 14.5, 7.2, 4.49))
            self.assertIn("slab", event["volume_method"])
            snap = counter.snapshot()["events"][0]
            self.assertEqual((snap["sorting"], snap["detector_label"]), ("MIS-SORT", "book"))


if __name__ == "__main__":
    unittest.main()
