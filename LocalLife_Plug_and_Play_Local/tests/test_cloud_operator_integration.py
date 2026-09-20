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
            "DOWNLOAD MEASUREMENTS (CSV)", "RESEARCH / ADVANCED MODE",
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


class WasteLedgerToggleTests(unittest.TestCase):
    """One button for the measurement history.

    The ledger only runs in `waste` mode, so a run left in `geometry_validation`
    records no history and produces no CSV rows -- which is exactly what
    happened in the field. This is the operator's switch for it.
    """

    def setUp(self) -> None:
        self.app, self.client, self._directory = _app()
        self.addCleanup(self._directory.cleanup)
        self.manager = self.app.config["CAMERA_COORDINATOR"]

    def test_the_default_mode_is_left_exactly_as_it_was(self) -> None:
        # The working local setup must not change until the button is pressed.
        self.assertEqual(AppConfig().operating_mode, "waste")
        state = self.client.get("/api/state").get_json()
        self.assertIn("waste_ledger_enabled", state["cameras"]["realsense"])

    def test_the_toggle_switches_every_station_not_just_the_coordinator(self) -> None:
        # Each station holds its own replace()d config copy.
        self.client.post("/api/waste-ledger", json={"enabled": False})
        self.assertFalse(self.manager.config.ledger_active)
        for camera_id in ("realsense", "logitech"):
            with self.subTest(camera=camera_id):
                self.assertFalse(self.manager.camera(camera_id).config.ledger_active)

    def test_the_toggle_never_changes_what_the_detector_accepts(self) -> None:
        # The bug this replaces: switching history on also switched the
        # classifier to strict waste mode, where only plastic bags, paper bags
        # and cardboard boxes count -- so a backpack stopped being detected at
        # all. operating_mode must not move.
        for mode in ("waste", "geometry_validation"):
            with self.subTest(mode=mode):
                app, client, directory = _app(operating_mode=mode)
                self.addCleanup(directory.cleanup)
                manager = app.config["CAMERA_COORDINATOR"]
                for enabled in (True, False, True):
                    client.post("/api/waste-ledger", json={"enabled": enabled})
                    self.assertEqual(manager.config.operating_mode, mode)
                    for camera_id in ("realsense", "logitech"):
                        self.assertEqual(
                            manager.camera(camera_id).config.operating_mode, mode
                        )

    def test_history_can_record_in_geometry_validation_mode(self) -> None:
        # The combination the operator actually needs: detect any object, and
        # still record the deposits.
        app, client, directory = _app(operating_mode="geometry_validation")
        self.addCleanup(directory.cleanup)
        manager = app.config["CAMERA_COORDINATOR"]
        self.assertFalse(manager.config.ledger_active)
        client.post("/api/waste-ledger", json={"enabled": True})
        self.assertTrue(manager.config.ledger_active)
        self.assertEqual(manager.config.operating_mode, "geometry_validation")
        camera = client.get("/api/state").get_json()["cameras"]["realsense"]
        self.assertTrue(camera["waste_ledger_enabled"])
        self.assertEqual(camera["operating_mode"], "geometry_validation")

    def test_untouched_config_still_follows_the_operating_mode(self) -> None:
        # Nothing changes until the button is pressed.
        self.assertTrue(AppConfig(operating_mode="waste").ledger_active)
        self.assertFalse(AppConfig(operating_mode="geometry_validation").ledger_active)

    def test_enabling_turns_the_ledger_back_on(self) -> None:
        self.client.post("/api/waste-ledger", json={"enabled": False})
        response = self.client.post("/api/waste-ledger", json={"enabled": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["waste_ledger_enabled"])
        self.assertTrue(self.manager.camera("realsense").config.ledger_active)

    def test_the_state_reports_the_switch_position(self) -> None:
        self.client.post("/api/waste-ledger", json={"enabled": False})
        camera = self.client.get("/api/state").get_json()["cameras"]["realsense"]
        self.assertFalse(camera["waste_ledger_enabled"])

    def test_a_non_boolean_is_rejected(self) -> None:
        for bad in ({"enabled": "yes"}, {"enabled": 1}, {}):
            with self.subTest(bad=bad):
                self.assertEqual(
                    self.client.post("/api/waste-ledger", json=bad).status_code, 400
                )

    def test_the_page_carries_the_button_and_its_label(self) -> None:
        body = self.client.get("/").get_data(as_text=True)
        for marker in ("ledger-toggle", "toggleLedger", "ENABLE HISTORY",
                       "DISABLE HISTORY", "/api/waste-ledger"):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)
