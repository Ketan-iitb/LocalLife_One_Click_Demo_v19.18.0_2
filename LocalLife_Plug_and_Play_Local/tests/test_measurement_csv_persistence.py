"""The measurement CSV: canonical events, idempotency, and the real download.

The export came back blank from real runs. The download route was never the
fault -- it read exactly the file the pipeline wrote. These pin the source of
truth instead: what counts as a finalised event, that each one lands exactly
once however many times it is offered, and that the bytes a browser receives
carry the values.

Numbered to match the required cases 1-16, with the end-to-end test last.
"""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.event_log import (
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    MeasurementEventLog,
    resolve_event_id,
)
from locallife_cloud.server import create_app


def _config(directory: str, **overrides) -> AppConfig:
    return AppConfig(
        detector_model="local-opencv-background",
        results_dir=Path(directory),
        enable_monocular_depth=False,
        enable_material_classification=False,
        enable_bucket_sync=False,
        restore_saved_baseline=False,
        **overrides,
    )


def _event(event_id: str, **fields) -> dict:
    row = {
        "event_id": event_id,
        "track_id": 1,
        "status": STATUS_ACCEPTED,
        "volume_l": 12.5,
        "color": "green",
        "material": "plastic",
        "sorting_status": "correct",
        "processing_mode": "local",
        "bag_count": 1,
    }
    row.update(fields)
    return row


class EventPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.log = MeasurementEventLog(self.root)

    # 1
    def test_one_accepted_event_creates_one_nonblank_row(self) -> None:
        result = self.log.record(_event("alpha"))
        self.assertTrue(result.written)
        rows = self.log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], "alpha")
        self.assertEqual(rows[0]["volume_l"], "12.5")
        # "Nonblank" means the row carries values, not just commas.
        self.assertTrue(any(value for value in rows[0].values()))

    # 2
    def test_two_accepted_events_create_two_rows(self) -> None:
        self.log.record(_event("alpha"))
        self.log.record(_event("beta", track_id=2, volume_l=3.25))
        rows = self.log.rows()
        self.assertEqual([row["event_id"] for row in rows], ["alpha", "beta"])
        self.assertEqual(rows[1]["volume_l"], "3.25")

    # 3
    def test_repeated_frames_do_not_duplicate_a_row(self) -> None:
        # The same physical object offered on twenty consecutive frames.
        results = [self.log.record(_event("alpha")) for _ in range(20)]
        self.assertTrue(results[0].written)
        self.assertTrue(all(item.duplicate for item in results[1:]))
        self.assertTrue(all(item.ok for item in results))
        self.assertEqual(len(self.log.rows()), 1)

    # 4
    def test_a_browser_refresh_does_not_duplicate_a_row(self) -> None:
        self.log.record(_event("alpha"))
        for _ in range(5):
            self.assertIn("alpha", self.log.csv_text())
        self.assertEqual(len(self.log.rows()), 1)

    # 6
    def test_a_rejected_event_keeps_its_status_and_reason(self) -> None:
        self.log.record(_event(
            "withheld", status=STATUS_REJECTED,
            reason="new_deposit_not_isolatable", bag_count=0,
        ))
        row = self.log.rows()[0]
        self.assertEqual(row["status"], STATUS_REJECTED)
        self.assertEqual(row["reason"], "new_deposit_not_isolatable")
        self.assertEqual(row["bag_count"], "0")

    # 11
    def test_a_reconnect_does_not_duplicate_an_event(self) -> None:
        # A dropped cloud link restarts the process: a fresh log over the same
        # directory must recognise what is already on disk.
        self.log.record(_event("alpha"))
        reconnected = MeasurementEventLog(self.root)
        self.assertTrue(reconnected.record(_event("alpha")).duplicate)
        self.assertEqual(len(reconnected.rows()), 1)
        self.assertEqual(reconnected.status()["events_persisted"], 1)

    # 12
    def test_writer_and_reader_resolve_the_same_file(self) -> None:
        self.log.record(_event("alpha"))
        self.assertEqual(self.log.path, self.root.resolve() / "measurements.csv")
        self.assertTrue(self.log.path.is_file())
        self.assertEqual(self.log.status()["csv_path"], str(self.log.path))

    def test_the_resolved_path_ignores_the_working_directory(self) -> None:
        # Relative paths must resolve from the configured data directory; the
        # launcher, a developer shell and the Windows service all differ.
        import os

        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(self.root)
        relative = MeasurementEventLog(Path(self._directory.name))
        self.assertTrue(relative.path.is_absolute())
        self.assertEqual(relative.path.parent, self.root.resolve())

    # 14
    def test_a_restart_preserves_earlier_rows(self) -> None:
        self.log.record(_event("alpha"))
        restarted = MeasurementEventLog(self.root)
        restarted.record(_event("beta", track_id=2))
        self.assertEqual([row["event_id"] for row in restarted.rows()], ["alpha", "beta"])

    # 15
    def test_commas_quotes_and_newlines_survive_the_round_trip(self) -> None:
        awkward = 'bag, "big"\nsecond line'
        self.log.record(_event("alpha", label=awkward))
        self.assertEqual(self.log.rows()[0]["label"], awkward)

    # 16
    def test_a_write_failure_is_visible_and_retryable(self) -> None:
        with mock.patch.object(
            MeasurementEventLog, "_append", side_effect=OSError("disk full"),
        ):
            result = self.log.record(_event("alpha"))
        self.assertFalse(result.ok)
        self.assertIn("disk full", result.error)
        status = self.log.status()
        self.assertFalse(status["healthy"])
        self.assertEqual(status["failed_event_ids"], ["alpha"])
        self.assertIn("disk full", status["last_error"])
        # ...and retryable once the condition clears.
        self.assertEqual(self.log.retry_failed(), {"recovered": 1, "still_failing": 0})
        self.assertTrue(self.log.status()["healthy"])
        self.assertEqual(len(self.log.rows()), 1)

    def test_an_empty_log_says_so_rather_than_looking_broken(self) -> None:
        status = self.log.status()
        self.assertEqual(status["events_persisted"], 0)
        self.assertEqual(status["empty_note"], "No completed measurements recorded")
        rows = list(csv.reader(io.StringIO(self.log.csv_text())))
        self.assertEqual(len(rows), 1)
        self.assertIn("event_id", rows[0])

    def test_a_missing_event_id_is_refused_rather_than_written(self) -> None:
        result = self.log.record({"volume_l": 1.0})
        self.assertFalse(result.ok)
        self.assertEqual(self.log.rows(), [])

    def test_ground_truth_error_columns_are_computed(self) -> None:
        self.log.record(_event("alpha", volume_l=10.0, ground_truth_litres=8.0))
        row = self.log.rows()[0]
        self.assertEqual(float(row["absolute_error_litres"]), 2.0)
        self.assertEqual(float(row["percentage_error"]), 25.0)

    def test_ground_truth_can_be_attached_after_the_fact(self) -> None:
        self.log.record(_event("alpha", volume_l=10.0))
        self.log.record(_event("beta", track_id=2, volume_l=4.0))
        self.assertTrue(self.log.set_ground_truth("alpha", 12.5))
        rows = {row["event_id"]: row for row in self.log.rows()}
        self.assertEqual(float(rows["alpha"]["absolute_error_litres"]), 2.5)
        self.assertEqual(float(rows["alpha"]["percentage_error"]), 20.0)
        # The untouched row must survive the rewrite intact.
        self.assertEqual(rows["beta"]["volume_l"], "4.0")
        self.assertFalse(self.log.set_ground_truth("absent", 1.0))

    def test_event_ids_are_stable_and_session_scoped(self) -> None:
        first = resolve_event_id("session-a", "realsense", 7)
        self.assertEqual(first, resolve_event_id("session-a", "realsense", 7))
        # A new session reusing track id 7 must not collide with the old one.
        self.assertNotEqual(first, resolve_event_id("session-b", "realsense", 7))
        self.assertNotEqual(first, resolve_event_id("session-a", "logitech", 7))


class StationPersistenceTests(unittest.TestCase):
    """The pipeline side: what counts as finalised, in each operating mode."""

    def _station(self, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = _config(directory.name, **overrides)
        coordinator = DualCameraCoordinator(config)
        return coordinator.camera("realsense"), Path(directory.name)

    def _detection(self, track_id=1, volume=12.5):
        from locallife_cloud.types import Detection

        return Detection(
            label="bag", confidence=0.9, box=(0, 0, 10, 10),
            track_id=track_id, realsense_volume_l=volume,
            color="green", accepted_class="bag",
        )

    # 5
    def test_a_pending_event_creates_no_accepted_row(self) -> None:
        station, _ = self._station()
        detection = self._detection()
        # One sample only: nowhere near settled, so nothing may be finalised.
        station._volume_history[1] = [12.5]
        station._finalise_settled_measurements([detection], 1.0)
        self.assertEqual(station.event_log.rows(), [])
        self.assertEqual(station.event_log.status()["events_persisted"], 0)

    # 7
    def test_geometry_validation_mode_records_readings(self) -> None:
        # The mode that disables the waste ledger -- previously the mode that
        # produced no spreadsheet at all.
        # auto_deposit off, so this exercises the settle-based finalisation.
        station, _ = self._station(operating_mode="geometry_validation", auto_deposit=False)
        detection = self._detection()
        station._volume_history[1] = [12.5] * station.config.settle_frames
        station._finalise_settled_measurements([detection], 1.0)
        rows = station.event_log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operating_mode"], "geometry_validation")
        self.assertEqual(rows[0]["status"], STATUS_ACCEPTED)

    # 8 + 9
    def test_waste_mode_local_run_records_a_finalised_deposit(self) -> None:
        station, _ = self._station(operating_mode="waste")
        station.persist_measurement_event(self._detection(), 1.0)
        rows = station.event_log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operating_mode"], "waste")
        self.assertEqual(rows[0]["processing_mode"], "local")
        self.assertEqual(rows[0]["bag_count"], "1")

    # 10
    def test_a_cloud_mode_result_is_recorded_with_the_same_schema(self) -> None:
        station, _ = self._station(cloud_enabled=True, cloud_project="demo")
        station.persist_measurement_event(self._detection(), 1.0)
        row = station.event_log.rows()[0]
        self.assertEqual(row["processing_mode"], "cloud")
        # Same columns as a local row: the two must stay comparable.
        self.assertEqual(set(row), set(MeasurementEventLog.COLUMNS))

    def test_a_withheld_deposit_is_recorded_as_rejected(self) -> None:
        station, _ = self._station()
        detection = self._detection()
        detection.volume_rejection_reason = "new_deposit_not_isolatable"
        station.persist_measurement_event(
            detection, 1.0, status=STATUS_REJECTED,
            status_reason=detection.volume_rejection_reason,
        )
        row = station.event_log.rows()[0]
        self.assertEqual(row["status"], STATUS_REJECTED)
        self.assertEqual(row["bag_count"], "0")

    def test_an_accepted_and_a_rejected_outcome_do_not_collide(self) -> None:
        # Same track, both outcomes: distinct events, so one must not suppress
        # the other through the idempotency key.
        station, _ = self._station()
        station.persist_measurement_event(self._detection(), 1.0)
        station.persist_measurement_event(
            self._detection(), 2.0, status=STATUS_REJECTED, status_reason="unstable_depth",
        )
        self.assertEqual(len(station.event_log.rows()), 2)

    def test_persisting_the_same_deposit_twice_writes_one_row(self) -> None:
        station, _ = self._station()
        for _ in range(4):
            station.persist_measurement_event(self._detection(), 1.0)
        self.assertEqual(len(station.event_log.rows()), 1)
        self.assertEqual(station._last_persist_result["duplicate"], True)


class DownloadEndToEndTests(unittest.TestCase):
    """13 + the required end-to-end test: finalise, download, parse, verify."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        config = _config(self._directory.name)
        self.app = create_app(config, DualCameraCoordinator(config))
        self.client = self.app.test_client()
        self.station = self.app.config["CAMERA_COORDINATOR"].camera("realsense")

    def test_a_finalised_measurement_survives_the_real_download_route(self) -> None:
        from locallife_cloud.types import Detection

        detection = Detection(
            label="black bin bag", confidence=0.93, box=(10, 10, 120, 140),
            track_id=7, realsense_volume_l=18.4, color="black",
            accepted_class="bag", material="plastic", sorting_status="correct",
            footprint_length_mm=410.0, footprint_width_mm=300.0,
            physical_height_mm=260.0,
        )
        persisted = self.station.persist_measurement_event(detection, 1234.5)
        self.assertTrue(persisted["written"])

        response = self.client.get("/api/cameras/realsense/measurements.csv")
        self.assertEqual(response.status_code, 200)
        # Content type and disposition, as an operator's browser sees them.
        self.assertEqual(response.headers["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment; filename=", response.headers["Content-Disposition"])
        self.assertEqual(
            response.headers["X-LocalLife-CSV-Path"], str(self.station.event_log.path)
        )

        # Parse the bytes the browser actually receives.
        body = response.get_data().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(body)))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["event_id"], persisted["event_id"])
        self.assertEqual(row["track_id"], "7")
        self.assertEqual(float(row["volume_l"]), 18.4)
        self.assertEqual(row["color"], "black")
        self.assertEqual(row["material"], "plastic")
        self.assertEqual(row["sorting_status"], "correct")
        self.assertEqual(row["processing_mode"], "local")
        self.assertEqual(row["status"], STATUS_ACCEPTED)
        self.assertEqual(row["bag_count"], "1")
        self.assertEqual(float(row["length_mm"]), 410.0)
        self.assertEqual(float(row["height_mm"]), 260.0)

    def test_downloading_repeatedly_never_grows_the_file(self) -> None:
        self.station.event_log.record(_event("alpha"))
        bodies = {
            self.client.get("/api/cameras/realsense/measurements.csv").get_data()
            for _ in range(4)
        }
        self.assertEqual(len(bodies), 1)

    def test_an_empty_run_downloads_a_header_only_csv(self) -> None:
        response = self.client.get("/api/cameras/realsense/measurements.csv")
        rows = list(csv.reader(io.StringIO(response.get_data().decode("utf-8-sig"))))
        self.assertEqual(len(rows), 1)
        self.assertIn("event_id", rows[0])
        state = self.client.get("/api/state").get_json()
        self.assertEqual(
            state["csv_persistence"]["empty_note"], "No completed measurements recorded"
        )

    def test_the_retry_route_reports_recovery(self) -> None:
        with mock.patch.object(
            MeasurementEventLog, "_append", side_effect=OSError("disk full"),
        ):
            self.station.event_log.record(_event("alpha"))
        self.assertFalse(
            self.client.get("/api/state").get_json()["csv_persistence"]["healthy"]
        )
        response = self.client.post("/api/cameras/realsense/measurements/retry", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["result"]["recovered"], 1)
        self.assertTrue(
            self.client.get("/api/state").get_json()["csv_persistence"]["healthy"]
        )

    def test_ground_truth_can_be_supplied_through_the_route(self) -> None:
        self.station.event_log.record(_event("alpha", volume_l=10.0))
        response = self.client.post(
            "/api/cameras/realsense/measurements/ground-truth",
            json={"event_id": "alpha", "litres": 8.0},
        )
        self.assertEqual(response.status_code, 200)
        row = self.station.event_log.rows()[0]
        self.assertEqual(float(row["absolute_error_litres"]), 2.0)
        for bad in ({"litres": 1.0}, {"event_id": "alpha"}, {"event_id": "alpha", "litres": -1}):
            with self.subTest(payload=bad):
                self.assertEqual(
                    self.client.post(
                        "/api/cameras/realsense/measurements/ground-truth", json=bad,
                    ).status_code,
                    400,
                )

    def test_an_unknown_camera_is_a_404(self) -> None:
        self.assertEqual(
            self.client.get("/api/cameras/nope/measurements.csv").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
