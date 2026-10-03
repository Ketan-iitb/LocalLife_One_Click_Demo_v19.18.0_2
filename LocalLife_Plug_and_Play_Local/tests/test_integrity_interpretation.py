"""Provenance, freshness, disclosure, logging and counting regressions (synthetic inputs)."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locallife_cloud.colour_evidence import describe_colour
from locallife_cloud.config import AppConfig
from locallife_cloud.experiment_log import ExperimentLog, collect_attempt, evaluate
from locallife_cloud.geometry import is_phantom_detection, stamp_provenance
from locallife_cloud.material_evidence import reconcile_material
from locallife_cloud.pointcloud_volume import estimate_volume_heightmap
from locallife_cloud.tracking import ObjectTracker
from locallife_cloud.types import Detection


def _mask(shape=(60, 60), box=(20, 20, 40, 40)):
    mask = np.zeros(shape, bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


class PhantomProvenanceTests(unittest.TestCase):
    def test_a_silhouette_stays_unconfirmed_after_prediction(self) -> None:
        phantom = Detection("bag (silhouette)", 0.0, (20, 20, 40, 40), _mask(), source="fixed-bin-depth-silhouette")
        stamp_provenance(phantom, timestamp=1.0, processed_at=1.1, frame_id=1)
        tracker = ObjectTracker(confirmation_frames=1, max_missing_frames=5)
        tracker.update([phantom])
        for track in tracker.tracks.values():
            track.counted, track.last_detection, track.missing = True, phantom, 1
        predicted = tracker.predicted_detections(5)
        self.assertEqual(predicted[0].source, "tracked-prediction")
        self.assertEqual(predicted[0].observation_status, "predicted")
        self.assertTrue(is_phantom_detection(predicted[0]))            # source string no longer says so
        self.assertFalse(predicted[0].semantic_confirmed)
        self.assertEqual(predicted[0].measured_at, 1.0)                # age stays visible

    def test_stamp_never_overwrites_first_sighting(self) -> None:
        item = Detection("garbage bag", 0.8, (0, 0, 5, 5))
        stamp_provenance(item, timestamp=5.0, processed_at=5.2, frame_id=3)
        item.source = "held-through-dropout"
        stamp_provenance(item, timestamp=9.0, processed_at=9.1, frame_id=7)
        self.assertEqual((item.origin_source, item.measured_at, item.frame_id), ("yoloe", 5.0, 3))


class FusedFreshnessTests(unittest.TestCase):
    def _manager(self, directory):
        from locallife_cloud.comparison import DualCameraCoordinator

        config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                           enable_bucket_sync=False, results_dir=Path(directory))
        return DualCameraCoordinator(config)

    def _put(self, station, detections, processed_at):
        station.latest_analysis = SimpleNamespace(detections=detections, timestamp=processed_at)
        station.last_frame_processed_at = processed_at

    def _obj(self, volume, **kw):
        item = Detection("garbage bag", 0.9, (0, 0, 10, 10), accepted_class="plastic_bag",
                         tracking_status="confirmed", realsense_volume_l=volume, monocular_volume_l=volume, **kw)
        return item

    def test_stale_and_predicted_readings_are_not_fused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            self._put(manager.camera("realsense"), [self._obj(5.0)], time.time() - 60)
            self._put(manager.camera("logitech"), [self._obj(4.0, observation_status="predicted")], time.time())
            fused = manager.fused_result()
            self.assertFalse(fused["available"])
            self.assertIn("realsense", fused["stale_cameras"])

    def test_two_objects_in_one_view_are_not_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            now = time.time()
            self._put(manager.camera("realsense"), [self._obj(5.0)], now)
            self._put(manager.camera("logitech"), [self._obj(4.0), self._obj(1.0)], now)
            fused = manager.fused_result()
            self.assertEqual(fused["association"], "ambiguous-multiple-objects")
            self.assertIsNone(fused["agreement_l"])
            counting = manager.counting_summary()
            self.assertEqual(counting["cameras"]["logitech"]["objects_observed_now"], 2)
            self.assertIn("no count here is a physically confirmed drop", counting["physical_sensors"])


class DisclosureTests(unittest.TestCase):
    def test_bag_snapping_is_opt_in_and_flagged(self) -> None:
        rng = np.random.default_rng(0)
        xy = rng.uniform(-0.1, 0.1, (4000, 2))
        points = np.column_stack((xy, np.full(4000, 0.12)))               # 0.2 x 0.2 x 0.12 m = 4.8 L
        plain = estimate_volume_heightmap(points, plane_equation=(0.0, 0.0, 1.0, 0.0))
        self.assertFalse(plain.snapped_to_class)
        self.assertAlmostEqual(plain.liters, plain.raw_liters)
        snapped = estimate_volume_heightmap(points, plane_equation=(0.0, 0.0, 1.0, 0.0), class_sizes_l=(5.0, 10.0))
        self.assertTrue(snapped.snapped_to_class)
        self.assertEqual(snapped.liters, 5.0)
        self.assertIn("volume_snapped_to_class", snapped.flags)
        self.assertNotEqual(snapped.raw_liters, 5.0)

    def test_one_vote_is_not_one_hundred_percent_material(self) -> None:
        material, confidence, evidence = reconcile_material("garbage bag", ["polythene bag"],
                                                            scores=[("polythene bag", 0.41)])
        self.assertEqual(evidence["agreement"], 1.0)
        self.assertAlmostEqual(confidence, 0.41)
        exported = Detection("garbage bag", 0.9, (0, 0, 1, 1), material_label_agreement=1.0,
                             material_model_score=0.41, material_samples=1).to_dict()["material_scores"]
        self.assertEqual((exported["samples"], exported["model_score"]), (1, 0.41))
        self.assertIn("not a calibrated probability", exported["meaning"])


class ExperimentLogTests(unittest.TestCase):
    def test_failures_are_logged_and_counted_in_the_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = ExperimentLog(Path(directory))
            config = AppConfig(results_dir=Path(directory))
            idle = SimpleNamespace(latest_analysis=None, last_frame_processed_at=0.0)
            timeout = collect_attempt(idle, "realsense", requested_at=time.time(), timeout_s=0.5,
                                      sleep=lambda _s: None)
            self.assertEqual(timeout["outcome"], "timeout")
            unmeasured = Detection("garbage bag", 0.9, (0, 0, 5, 5), accepted_class="plastic_bag",
                                   volume_rejection_reason="rejected-sparse-height-inside-mask")
            measured = Detection("garbage bag", 0.9, (0, 0, 5, 5), accepted_class="plastic_bag",
                                 realsense_volume_l=5.5)
            rows = [timeout]
            for detection in (unmeasured, measured):
                station = SimpleNamespace(latest_analysis=SimpleNamespace(detections=[detection], timestamp=1.0),
                                          last_frame_processed_at=time.time() + 1)
                rows.append(collect_attempt(station, "realsense", requested_at=time.time(), timeout_s=0.5,
                                            sleep=lambda _s: None))
            self.assertEqual([r["outcome"] for r in rows], ["timeout", "missing_depth", "success"])
            run = log.start_run("t", config)
            for index, row in enumerate(rows):
                log.record({**{k: v for k, v in row.items() if not k.startswith("_")}, "role": "validation",
                            "object_id": f"obj{index}", "reference_volume_l": 5.0}, config)
            log.record({"role": "calibration", "camera": "realsense", "object_id": "obj2", "outcome": "success",
                        "reported_volume_l": 5.0, "reference_volume_l": 5.0}, config)
            result = evaluate(log.attempts(run["run_id"]))
            camera = result["cameras"]["realsense"]
            self.assertEqual(result["objects_used_for_both_calibration_and_validation"], ["obj2"])
            self.assertEqual(camera["attempts"], 2)                    # obj2 excluded: also a calibration object
            self.assertEqual(camera["valid_measurements"], 0)
            self.assertEqual(camera["failure_rate"], 1.0)
            self.assertIn("missing_depth", log.csv(run["run_id"]))


class ColourAndTokenTests(unittest.TestCase):
    def test_a_warm_light_cast_is_removed_before_naming_the_colour(self) -> None:
        background = np.full((80, 80, 3), (120, 120, 120), np.uint8)       # grey floor
        frame = np.full((80, 80, 3), (95, 120, 150), np.uint8)             # same floor under warm light
        frame[25:55, 25:55] = (175, 210, 255)                              # white bag under the same cast
        colour = describe_colour(frame, _mask((80, 80), (25, 25, 55, 55)), background_bgr=background)
        self.assertIsNotNone(colour.illumination_gains)
        self.assertEqual(colour.colour, "white")

    def test_a_remote_viewer_does_not_receive_the_api_token(self) -> None:
        from locallife_cloud.comparison import DualCameraCoordinator
        from locallife_cloud.server import create_app

        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(detector_model="local-opencv-background", enable_monocular_depth=False,
                               enable_bucket_sync=False, results_dir=Path(directory), api_token="s3cret-test")
            client = create_app(config, DualCameraCoordinator(config)).test_client()
            remote = client.get("/research", environ_base={"REMOTE_ADDR": "10.1.2.3"}).get_data(as_text=True)
            local = client.get("/research").get_data(as_text=True)
            self.assertNotIn("s3cret-test", remote)
            self.assertIn("s3cret-test", local)
            self.assertEqual(client.post("/api/experiment/run/start", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
