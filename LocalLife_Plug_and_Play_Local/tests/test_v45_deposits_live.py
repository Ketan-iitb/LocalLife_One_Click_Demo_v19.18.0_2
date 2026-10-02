"""v45 live deposit counter: synthetic 2 fps sequences through the real state machine.

No real deposit video exists in the repository, so physical verification is
still pending; these check the logic only.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locallife_cloud import session_deposits as sd  # noqa: E402
from locallife_cloud.footprint import estimate_extents  # noqa: E402
from locallife_cloud.mask_leak import trim_mask_leak  # noqa: E402

H, W = 120, 160
FLOOR = 1.10


class Scene:
    """A bin seen from above: textured floor, bags as bright boxes with depth."""

    def __init__(self, camera="realsense", depth=True, seed=0):
        self.camera, self.depth = camera, depth
        rng = np.random.default_rng(seed)
        self.floor = rng.integers(40, 90, (H, W)).astype(np.uint8)
        self.noise = np.random.default_rng(seed + 1)
        self.bags: dict[int, dict] = {}
        self.untracked: list[dict] = []

    def add(self, tid, box, height=0.30, shade=200, label="waste bag", tracked=True):
        bag = dict(box=box, height=height, shade=shade, label=label)
        if tracked:
            self.bags[tid] = bag
        else:
            self.untracked.append(bag)

    def evidence(self, t, hand=False, with_heights=True):
        grey = self.floor.copy().astype(np.int16)
        depth = np.full((H, W), FLOOR, np.float32)
        for bag in list(self.bags.values()) + self.untracked:
            x1, y1, x2, y2 = bag["box"]
            grey[y1:y2, x1:x2] = bag["shade"]
            depth[y1:y2, x1:x2] = np.minimum(depth[y1:y2, x1:x2], FLOOR - bag["height"])
        if hand:                                      # something moving through the bin
            x = int(t * 37) % (W - 40)
            grey[20:90, x:x + 40] = 250 - (int(t * 10) % 3) * 60
            depth[20:90, x:x + 40] = 0.6
        grey = np.clip(grey + self.noise.integers(-4, 5, grey.shape), 0, 255).astype(np.uint8)
        tracks = [sd.TrackInfo(tid, tuple(float(v) for v in b["box"]), label=b["label"], colour="black",
                               material="plastic film", material_confidence=0.8,
                               length_mm=400.0, width_mm=300.0, height_mm=b["height"] * 1000)
                  for tid, b in self.bags.items()]
        use = self.depth
        return sd.FrameEvidence(self.camera, t, grey, np.ones((H, W), bool), tracks,
                                depth=depth if use else None,
                                heights=(FLOOR - depth) if use and with_heights else None,
                                height_status="approximate" if use and with_heights else None)


def run(counter, scene, start, end, step=0.5, **kw):
    t, out = start, []
    while t < end - 1e-9:
        r = counter.observe(scene.evidence(t, **kw))
        if r is not None:
            out.append(r)
        t += step
    return out, t


class SequenceTests(unittest.TestCase):
    def test_full_sequence_at_two_fps(self) -> None:
        with TemporaryDirectory() as d:
            clock = [0.0]
            counter = sd.SessionDeposits(Path(d), clock=lambda: clock[0])
            scene = Scene()
            for tid, box in ((1, (10, 10, 40, 40)), (2, (50, 10, 80, 40)), (3, (100, 60, 130, 90))):
                scene.add(tid, box)
            run(counter, scene, 0.0, 3.0)
            self.assertEqual(counter.snapshot()["status"], "initialising")
            run(counter, scene, 3.0, 7.0)
            snap = counter.snapshot()
            self.assertEqual((snap["status"], snap["new_bags_this_session"]), ("watching", 0))
            self.assertEqual(snap["cameras"]["realsense"]["baseline_tracks"], 3)

            # Bag A dropped after baseline readiness
            run(counter, scene, 7.0, 8.0, hand=True)
            scene.add(10, (20, 70, 60, 100), height=0.25)
            got, _ = run(counter, scene, 8.0, 12.0)
            self.assertEqual(counter.count, 1)
            a = got[0]
            self.assertEqual(a["evidence"]["realsense"]["evidence"], "new bag-like track over a persistent change")
            self.assertEqual(a["height_cm"], 25.0)                  # after top - before surface, local
            self.assertEqual((a["length_cm"], a["width_cm"], a["envelope_l"]), (40.0, 30.0, 30.0))
            self.assertEqual(a["material"], "PLASTIC FILM")         # temporal consensus

            # Bag B
            run(counter, scene, 12.0, 13.0, hand=True)
            scene.add(11, (70, 60, 95, 110), height=0.20)
            run(counter, scene, 13.0, 17.0)
            self.assertEqual(counter.count, 2)

            # An existing bag moves: still 2
            run(counter, scene, 17.0, 18.0, hand=True)
            scene.bags[2]["box"] = (120, 10, 150, 40)
            run(counter, scene, 18.0, 22.0)
            self.assertEqual(counter.count, 2)
            self.assertIn("existing bag moved", counter.rejected[-1]["reason"])

            # Temporary disappearance + ID switch of bag 3: still 2
            hidden = scene.bags.pop(3)
            run(counter, scene, 22.0, 23.0, hand=True)
            scene.bags[30] = hidden
            run(counter, scene, 23.0, 27.0)
            self.assertEqual(counter.count, 2)

            # New bag in a gap BELOW the pile's maximum height, detector missed it
            run(counter, scene, 27.0, 28.0, hand=True)
            scene.add(None, (130, 95, 155, 118), height=0.10, shade=170, tracked=False)
            got, _ = run(counter, scene, 28.0, 32.0)
            self.assertEqual(counter.count, 3)
            self.assertIn("persistent new foreground region", got[0]["evidence"]["realsense"]["evidence"])
            self.assertEqual(got[0]["height_cm"], 10.0)
            self.assertEqual(got[0]["measurement_status"], "partial")   # no tracked L x W: volume N/A
            self.assertIsNone(got[0]["envelope_l"])

            # A hand in and out, nothing left behind: rejected
            run(counter, scene, 32.0, 34.0, hand=True)
            run(counter, scene, 34.0, 37.0)
            self.assertEqual(counter.count, 3)
            self.assertIn("no persistent change", counter.rejected[-1]["reason"])

            # Never settles: timeout with a reason, not an endless candidate
            run(counter, scene, 37.0, 70.0, hand=True)
            self.assertIn("did not settle", counter.rejected[-1]["reason"])
            run(counter, scene, 70.0, 74.0)
            self.assertEqual(counter.snapshot()["cameras"]["realsense"]["state"], "watching")
            self.assertEqual(counter.count, 3)

            times = [e["deposit_time"] for e in counter.snapshot()["events"]]
            self.assertEqual(times, sorted(times))

            # "Browser refresh"/restart: a new instance resumes the same session
            clock[0] = 200.0
            again = sd.SessionDeposits(Path(d), clock=lambda: clock[0])
            self.assertEqual((again.session_id, again.count), (counter.session_id, 3))
            with (Path(d) / "session_deposits.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len([r for r in rows if r["counted"] == "True"]), 3)   # once per event
            self.assertEqual(len({r["event_id"] for r in rows}), len(rows))
            again.new_session()
            self.assertEqual(again.count, 0)

    def test_counting_needs_no_fill_profile_or_depth(self) -> None:
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            scene = Scene(camera="logitech", depth=False)
            scene.add(1, (10, 10, 40, 40), label="pillow")
            run(counter, scene, 0.0, 7.0)
            run(counter, scene, 7.0, 8.0, hand=True)
            scene.add(5, (60, 60, 100, 100), label="textile item")       # mislabelled bag still counts
            got, _ = run(counter, scene, 8.0, 12.0)
            self.assertEqual(counter.count, 1)
            self.assertEqual(got[0]["object_type"], "bag-like object")
            self.assertEqual(got[0]["detector_label"], "textile item")

    def test_a_red_bag_with_the_same_grey_level_is_seen(self) -> None:
        # Live replay found this: BGR (30, 30, 220) is grey ~87 on a ~65 pile, under the
        # threshold in grey, so the drop was invisible. Change is now per colour channel.
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            rng = np.random.default_rng(3)
            pile = np.repeat(rng.integers(55, 75, (H, W, 1)), 3, axis=2).astype(np.uint8)
            bag = pile.copy()
            bag[60:100, 60:110] = (30, 30, 220)
            self.assertLess(abs(float(bag[60:100, 60:110].mean(axis=-1).mean()) - 93) , 10)
            region = np.ones((H, W), bool)
            t = 0.0
            for img, hand in [(pile, False)] * 14 + [(pile, True)] * 2 + [(bag, False)] * 8:
                view = img.copy()
                if hand:
                    view[10:50, int(t * 20) % 100:int(t * 20) % 100 + 40] = (200, 180, 160)
                counter.observe(sd.FrameEvidence("logitech", t, view, region, []))
                t += 0.5
            self.assertEqual(counter.count, 1)

    def test_an_untracked_old_bag_that_moves_is_not_new(self) -> None:
        # Live replay found this: a bag the detector never tracked (absorbed into its
        # background) moved and got a fresh track id. Its vacated spot gives it away.
        for depth in (False, True):
            with self.subTest(depth=depth), TemporaryDirectory() as d:
                counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
                scene = Scene(depth=depth)
                scene.add(None, (10, 10, 50, 50), shade=210, tracked=False)       # never tracked
                run(counter, scene, 0.0, 7.0)
                run(counter, scene, 7.0, 8.0, hand=True)
                scene.untracked.clear()
                scene.add(44, (100, 60, 140, 100), shade=210)                    # same bag, new place
                run(counter, scene, 8.0, 12.0)
                self.assertEqual(counter.count, 0)
                self.assertIn("vacated", counter.rejected[-1]["reason"])

    def test_state_saved_by_an_older_version_resumes_without_breaking_the_panel(self) -> None:
        # Live bug: events saved before per-camera counting had no "camera" -> /api/bin-fill 500
        # -> "Bin fill data unavailable".
        import json
        with TemporaryDirectory() as d:
            (Path(d) / "session_state.json").write_text(json.dumps({
                "session_id": "old", "session_started_at": 1.0, "rejected": [],
                "events": [{"event_id": "D-1", "deposit_time": 2.0, "cameras": ["logitech"]}]}))
            counter = sd.SessionDeposits(Path(d), clock=lambda: 3.0)
            counter.observe(Scene("logitech", depth=False).evidence(3.0))
            snap = counter.snapshot()
            self.assertEqual(snap["cameras"]["logitech"]["new_bags"], 1)

    def test_removing_the_top_bag_does_not_count_the_one_below(self) -> None:
        for depth in (True, False):
            with self.subTest(depth=depth), TemporaryDirectory() as d:
                counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
                scene = Scene(depth=depth)
                scene.add(1, (40, 40, 100, 90), height=0.20, shade=120)            # lower bag
                scene.add(2, (45, 45, 95, 85), height=0.45, shade=220)             # bag on top of it
                run(counter, scene, 0.0, 7.0)
                run(counter, scene, 7.0, 8.0, hand=True)
                del scene.bags[2]                                                   # top bag lifted out
                scene.bags[7] = scene.bags.pop(1)                                   # lower bag re-detected, new id
                run(counter, scene, 8.0, 12.0)
                self.assertEqual(counter.count, 0)

    def test_a_person_is_not_a_bag(self) -> None:
        self.assertFalse(sd.bag_like("person"))
        self.assertFalse(sd.bag_like("hand"))
        self.assertTrue(sd.bag_like("pillow"))
        self.assertTrue(sd.bag_like("filled plastic garbage bag [waste_bag]"))

    def test_both_cameras_one_drop_count_once(self) -> None:
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            rs, lg = Scene("realsense"), Scene("logitech", depth=False, seed=5)
            for s in (rs, lg):
                s.add(1, (10, 10, 40, 40))
            for s in (rs, lg):
                run(counter, s, 0.0, 7.0)
            for s in (rs, lg):
                run(counter, s, 7.0, 8.0, hand=True)
            rs.add(10, (60, 60, 100, 90), height=0.25)
            lg.add(77, (70, 50, 110, 85))
            run(counter, rs, 8.0, 12.0)
            run(counter, lg, 8.5, 12.5)
            # Independent counters: each observing camera counts the drop once, nothing copied across.
            self.assertEqual((counter.count_for("realsense"), counter.count_for("logitech")), (1, 1))
            rs_event, lg_event = (next(e for e in counter.events if e["camera"] == c) for c in ("realsense", "logitech"))
            self.assertTrue(rs_event["event_id"].startswith("RS-") and lg_event["event_id"].startswith("LG-"))
            self.assertEqual(rs_event["height_cm"], 25.0)
            self.assertTrue(lg_event["height_source"].startswith("detector"))  # its own, never RealSense's


class DimensionTests(unittest.TestCase):
    def test_leaked_pixels_inflate_a_known_40cm_bag_and_are_trimmed(self) -> None:
        # Flat 40 x 30 cm bag seen from 1.1 m (f = 600 px): 1 px = 1.83 mm.
        scale = FLOOR / 600.0
        mask = np.zeros((480, 640), bool)
        mask[200:364, 200:419] = True                   # 219 x 164 px = 40 x 30 cm
        leaked = mask.copy()
        leaked[200:364, 660 - 50:660 - 40] = True        # a strip on the rim ~50 cm away
        spill = mask.copy()
        spill[276:286, 419:600] = True                   # thin spill into a neighbour

        def length_cm(m):
            rows, cols = np.nonzero(m)
            return 100 * estimate_extents(np.c_[cols * scale, rows * scale]).length_m

        self.assertAlmostEqual(length_cm(mask), 40.0, delta=0.5)
        self.assertGreater(length_cm(leaked), 70.0)      # 40 cm -> ~77 cm: the inflation symptom, reproduced
        self.assertGreater(length_cm(spill), 70.0)
        for bad in (leaked, spill):
            trimmed, flag = trim_mask_leak(bad)
            self.assertTrue(flag)
            self.assertLess(abs(length_cm(trimmed) - 40.0), 2.5)
        self.assertFalse(trim_mask_leak(mask)[1])
        thin = np.zeros((480, 640), bool)
        thin[300:306, 100:500] = True                    # a genuinely thin object is untouched
        self.assertFalse(trim_mask_leak(thin)[1])



class V47RecordTests(unittest.TestCase):
    def test_entry_and_confirmation_times_fill_at_drop_and_linked_metadata(self) -> None:
        with TemporaryDirectory() as d:
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            counter.fill_lookup = lambda camera: {"status": "ok", "height_fill_pct": 12.5, "rough_litres": 82.5}
            scene = Scene()
            scene.add(1, (10, 10, 40, 40))
            run(counter, scene, 0.0, 7.0)
            run(counter, scene, 7.0, 8.0, hand=True)
            scene.add(1, (60, 60, 100, 100), height=0.25)   # same id re-used by the detector for the new bag
            scene.bags[1]["box"] = (60, 60, 100, 100)
            scene.untracked.append(dict(box=(10, 10, 40, 40), height=0.30, shade=200, label="waste bag"))
            got, _ = run(counter, scene, 8.0, 12.0)
            e = got[0]
            self.assertLess(e["entered_at"], e["confirmed_at"])
            self.assertEqual((e["bin_fill_pct_after"], e["bin_fill_litres_after"]), (12.5, 82.5))
            self.assertEqual(e["colour"], "black")                      # linked to the covering detection

    def test_old_csv_schema_is_kept_aside_not_misaligned(self) -> None:
        with TemporaryDirectory() as d:
            (Path(d) / "session_deposits.csv").write_text("session_id,event_id\nold,1\n")
            counter = sd.SessionDeposits(Path(d), clock=lambda: 0.0)
            counter._write_csv({"event_id": "x", "cameras": ["realsense"], "deposit_time": 1.0, "evidence": {}})
            self.assertTrue(any("old_schema" in p.name for p in Path(d).iterdir()))
            header = (Path(d) / "session_deposits.csv").read_text().splitlines()[0].split(",")
            self.assertEqual(header, list(sd.CSV_FIELDS))



class V48ResetTests(unittest.TestCase):
    def test_reset_zeroes_both_cameras_ignores_late_frames_and_survives_restart(self) -> None:
        with TemporaryDirectory() as d:
            clock = [0.0]
            counter = sd.SessionDeposits(Path(d), clock=lambda: clock[0])
            scenes = {c: Scene(c, depth=(c == "realsense"), seed=i) for i, c in enumerate(("realsense", "logitech"))}
            for s in scenes.values():
                s.add(1, (10, 10, 40, 40))
                run(counter, s, 0.0, 7.0)
                run(counter, s, 7.0, 8.0, hand=True)
                s.add(5, (60, 60, 100, 100), height=0.25)
                run(counter, s, 8.0, 12.0)
            self.assertEqual((counter.count_for("realsense"), counter.count_for("logitech")), (1, 1))
            clock[0] = 20.0
            snap = counter.new_session()
            first_generation = snap["generation"]
            self.assertEqual(snap["new_bags_this_session"], 0)
            late = scenes["logitech"].evidence(19.5)                       # inference began before the reset
            late.started_at = 19.5
            self.assertIsNone(counter.observe(late))
            self.assertEqual(counter.late_frames_ignored, 1)
            again = sd.SessionDeposits(Path(d), clock=lambda: 30.0)         # restart: no old counts restored
            self.assertEqual((again.count, again.session_id), (0, counter.session_id))
            # Bags present at reset are baseline; the camera that reconnects later also starts at 0.
            run(counter, scenes["realsense"], 20.0, 27.0)
            self.assertEqual(counter.count_for("realsense"), 0)
            run(counter, scenes["logitech"], 40.0, 47.0)                   # reconnects late
            self.assertEqual(counter.count_for("logitech"), 0)
            self.assertEqual(counter.new_session()["generation"], first_generation + 1)   # repeated reset safe
            self.assertEqual(counter.new_session()["new_bags_this_session"], 0)


if __name__ == "__main__":
    unittest.main()
