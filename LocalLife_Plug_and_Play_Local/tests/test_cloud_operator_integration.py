"""Operator landing page, cloud-mode reporting, and the measurements CSV route.

The page is the real Accuracy Deployment v3.0 operator interface ported from the
Windows v3.2 deployment; these pin the contract it depends on so a backend change
cannot silently leave it showing dashes.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.operator_dashboard import OPERATOR_DASHBOARD
from locallife_cloud.server import create_app


def _app(**overrides):
    directory = tempfile.TemporaryDirectory()
    config = AppConfig(
        # The OpenCV background segmenter keeps these tests free of torch and
        # ultralytics; the routes under test do not depend on the detector.
        detector_model="local-opencv-background",
        results_dir=Path(directory.name),
        enable_monocular_depth=False,
        enable_material_classification=False,
        enable_bucket_sync=False,
        restore_saved_baseline=False,
        **overrides,
    )
    application = create_app(config, DualCameraCoordinator(config))
    return application, application.test_client(), directory


class OperatorPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.client, self._directory = _app()
        self.addCleanup(self._directory.cleanup)

    def test_the_landing_page_is_the_operator_interface(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Accuracy Deployment v3.0", body)
        self.assertIn("Local Life Waste Measurement", body)

    def test_the_research_dashboard_is_still_reachable(self) -> None:
        response = self.client.get("/research")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<!doctype html>", response.get_data(as_text=True).lower())

    def test_the_page_shows_every_required_operator_field(self) -> None:
        body = self.client.get("/").get_data(as_text=True)
        for marker in (
            "CAMERA 1", "CAMERA 2", "GPU VM", "Raspberry Pi", "Processing",
            "OBJECT ID", "TYPE", "COLOR", "MATERIAL", "SORTING", "VOLUME",
            "SIZE L×W×H", "HEIGHT", "CONFIDENCE",
            "ACCEPTED DROPS", "OPERATIONAL VOLUME", "Totals by bag color",
            "DOWNLOAD SIMPLE EXCEL", "DOWNLOAD DETAILED RAW CSV", "RESEARCH / ADVANCED MODE",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_protected_posts_carry_the_api_token(self) -> None:
        # The page's own POSTs hit @protected routes; without the token the
        # operator's buttons would silently 401 once a token is configured.
        self.assertIn("X-API-Token", OPERATOR_DASHBOARD)
        self.assertIn("{{ api_token|tojson }}", OPERATOR_DASHBOARD)

    def test_the_page_only_calls_routes_the_backend_serves(self) -> None:
        served = {str(rule) for rule in self.app.url_map.iter_rules()}
        for route in (
            "/api/state", "/api/color-map", "/api/operator/session/new",
            "/api/cameras/<camera_id>/roi",
            "/api/cameras/logitech/reference-distance",
            "/api/cameras/<camera_id>/baseline",
            "/video/cameras/<camera_id>", "/research",
            "/api/cameras/<camera_id>/measurements.csv",
        ):
            with self.subTest(route=route):
                self.assertIn(route, served)


class OperatorStateContractTests(unittest.TestCase):
    def test_state_reports_local_mode_by_default(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        state = client.get("/api/state").get_json()
        self.assertEqual(state["cloud"]["enabled"], False)
        self.assertEqual(state["cloud"]["processing_mode"], "local")
        self.assertEqual(state["cloud"]["vm_status"], "not used")
        self.assertIn("session_id", state["operator_session"])

    def test_state_reports_cloud_mode_when_configured(self) -> None:
        app, client, directory = _app(
            cloud_enabled=True, cloud_project="demo-project",
            cloud_vm_name="depth-l4", cloud_zone="europe-west4-a",
        )
        self.addCleanup(directory.cleanup)
        cloud = client.get("/api/state").get_json()["cloud"]
        self.assertTrue(cloud["enabled"])
        self.assertEqual(cloud["processing_mode"], "cloud")
        self.assertEqual(cloud["vm_name"], "depth-l4")
        # The backend cannot see the VM; it must say so rather than guess.
        self.assertEqual(cloud["vm_status"], "unknown")

    def test_the_comparison_block_carries_the_operational_summary(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        operational = client.get("/api/state").get_json()["comparison"]["operational"]
        for key in ("deposited_count", "cumulative_volume_l", "colors", "color_map"):
            self.assertIn(key, operational)

    def test_the_colour_map_round_trips(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        response = client.post("/api/color-map", json={"mapping": {"Blue": " Plastic "}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/api/color-map").get_json()["mapping"], {"blue": "Plastic"})
        state = client.get("/api/state").get_json()
        self.assertEqual(state["comparison"]["operational"]["color_map"], {"blue": "Plastic"})

    def test_a_bad_colour_map_is_rejected(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        self.assertEqual(client.post("/api/color-map", json={"mapping": "blue"}).status_code, 400)


class MeasurementCsvRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.client, self._directory = _app()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def test_an_empty_run_returns_a_header_only_csv(self) -> None:
        response = self.client.get("/api/cameras/realsense/measurements.csv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.mimetype.startswith("text/csv"))
        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(len(rows), 1)
        for column in ("volume_l", "length_mm", "color", "sorting_status", "timestamp"):
            self.assertIn(column, rows[0])

    def test_recorded_rows_are_served_verbatim(self) -> None:
        station = self.app.config["CAMERA_COORDINATOR"].camera("realsense")
        for track_id, colour in ((1, "green"), (2, "black")):
            station.store.append_csv(
                "measurements.csv",
                {"track_id": track_id, "color": colour, "volume_l": 1.5 * track_id,
                 "label": 'bag, "big"', "sorting_status": "correct"},
                station.MEASUREMENT_CSV_COLUMNS,
            )
        body = self.client.get("/api/cameras/realsense/measurements.csv").get_data(as_text=True)
        rows = list(csv.DictReader(io.StringIO(body)))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["color"] for row in rows], ["green", "black"])
        # A comma and quotes inside a field must survive the round trip.
        self.assertEqual(rows[0]["label"], 'bag, "big"')

    def test_downloading_twice_does_not_duplicate_rows(self) -> None:
        station = self.app.config["CAMERA_COORDINATOR"].camera("realsense")
        station.store.append_csv(
            "measurements.csv", {"track_id": 1, "volume_l": 2.0},
            station.MEASUREMENT_CSV_COLUMNS,
        )
        for _ in range(3):
            body = self.client.get("/api/cameras/realsense/measurements.csv").get_data(as_text=True)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(body)))), 1)

    def test_an_unknown_camera_is_a_404(self) -> None:
        self.assertEqual(self.client.get("/api/cameras/nope/measurements.csv").status_code, 404)


class OperatorSessionTests(unittest.TestCase):
    def test_a_new_session_archives_the_previous_one(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        station = app.config["CAMERA_COORDINATOR"].camera("realsense")
        station.store.append_csv(
            "measurements.csv", {"track_id": 1, "volume_l": 3.0},
            station.MEASUREMENT_CSV_COLUMNS,
        )
        first = client.get("/api/state").get_json()["operator_session"]["session_id"]

        response = client.post("/api/operator/session/new", json={})
        self.assertEqual(response.status_code, 200)
        session = response.get_json()["session"]
        self.assertNotEqual(session["session_id"], first)

        # Restarting a session must never destroy the earlier recording.
        archives = list((root / "sessions").iterdir())
        self.assertEqual(len(archives), 1)
        archived = archives[0] / "realsense_measurements.csv"
        self.assertTrue(archived.is_file())
        self.assertIn("3.0", archived.read_text(encoding="utf-8"))
        self.assertEqual(
            json.loads((archives[0] / "session.json").read_text(encoding="utf-8"))["session_id"],
            first,
        )

    def test_the_session_id_is_stable_across_requests(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        ids = {client.get("/api/state").get_json()["operator_session"]["session_id"] for _ in range(3)}
        self.assertEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()


class MeasurementCountTests(unittest.TestCase):
    """The page must explain an empty CSV rather than just serving a header."""

    def test_state_reports_how_many_measurements_were_finalised(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        station = app.config["CAMERA_COORDINATOR"].camera("realsense")
        self.assertEqual(client.get("/api/state").get_json()["measurements_recorded"], 0)
        # Counted from what actually reached disk, not from an in-memory set of
        # track ids: a row the operator cannot download was never "recorded".
        for index in (1, 2, 3):
            station.event_log.record({"event_id": f"event-{index}", "volume_l": index})
        state = client.get("/api/state").get_json()
        self.assertEqual(state["measurements_recorded"], 3)
        self.assertTrue(state["csv_persistence"]["healthy"])
        self.assertEqual(state["csv_persistence"]["csv_path"], str(station.event_log.path))

    def test_the_page_explains_a_missing_baseline(self) -> None:
        app, client, directory = _app()
        self.addCleanup(directory.cleanup)
        body = client.get("/").get_data(as_text=True)
        self.assertIn("csv-note", body)
        self.assertIn("SETUP / RECALIBRATE", body)


class HistoryLedgerAlwaysOnTests(unittest.TestCase):
    """Recording is not a mode, a preference or a toggle.

    Live runs produced an empty CSV because the launcher defaulted to a mode
    that disabled the ledger. It can no longer be switched off from the UI, and
    it no longer depends on the operating mode.
    """

    def setUp(self) -> None:
        self.app, self.client, self._directory = _app()
        self.addCleanup(self._directory.cleanup)
        self.manager = self.app.config["CAMERA_COORDINATOR"]

    def test_the_ledger_is_enabled_in_every_mode(self) -> None:
        for mode in ("waste", "geometry_validation"):
            with self.subTest(mode=mode):
                self.assertTrue(AppConfig(operating_mode=mode).ledger_active)

    def test_normal_operation_defaults_to_the_full_waste_mode(self) -> None:
        self.assertEqual(AppConfig().operating_mode, "waste")
        self.assertFalse(AppConfig().diagnostic_mode)

    def test_the_ledger_is_enabled_for_cloud_results_too(self) -> None:
        app, client, directory = _app(cloud_enabled=True, cloud_project="demo")
        self.addCleanup(directory.cleanup)
        camera = client.get("/api/state").get_json()["cameras"]["realsense"]
        self.assertTrue(camera["waste_ledger_enabled"])
        self.assertEqual(
            app.config["CAMERA_COORDINATOR"].config.processing_mode, "cloud"
        )

    def test_the_ui_cannot_disable_it(self) -> None:
        response = self.client.post("/api/waste-ledger", json={"enabled": False})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.get_json()["waste_ledger_enabled"])
        self.assertTrue(self.manager.camera("realsense").config.ledger_active)

    def test_enabling_it_explicitly_is_accepted(self) -> None:
        response = self.client.post("/api/waste-ledger", json={"enabled": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["waste_ledger_enabled"])

    def test_a_non_boolean_is_rejected(self) -> None:
        for bad in ({"enabled": "yes"}, {"enabled": 1}, {}):
            with self.subTest(bad=bad):
                self.assertEqual(
                    self.client.post("/api/waste-ledger", json=bad).status_code, 400
                )

    def test_the_state_reports_it_as_enabled(self) -> None:
        camera = self.client.get("/api/state").get_json()["cameras"]["realsense"]
        self.assertTrue(camera["waste_ledger_enabled"])

    def test_startup_reports_the_three_required_lines(self) -> None:
        station = self.manager.camera("realsense")
        report = station.event_log.startup_report()
        self.assertEqual(report[0], "Measurement mode: normal")
        self.assertEqual(report[1], "History ledger: enabled")
        self.assertTrue(report[2].startswith("CSV persistence: enabled"))
        self.assertIn(str(station.event_log.path), report[2])

    def test_an_unwritable_store_blocks_rather_than_running_empty(self) -> None:
        from unittest import mock

        from locallife_cloud.event_log import MeasurementEventLog

        log = MeasurementEventLog(Path(self._directory.name))
        log.assert_writable()  # the healthy case must not raise
        with mock.patch.object(Path, "mkdir", side_effect=OSError("read-only")):
            with self.assertRaises(RuntimeError) as caught:
                log.assert_writable()
        self.assertIn("cannot start", str(caught.exception))
