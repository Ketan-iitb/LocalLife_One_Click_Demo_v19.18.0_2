"""V37: RealSense is optional; a Pi without pyrealsense2 still runs the Logitech.

pyrealsense2 has no wheel for Python 3.13 on ARM64, which is what a Raspberry
Pi on Debian Trixie runs. In `--source dual` the RealSense thread went through
the resilient-camera loop, which retried the impossible import forever while
the dashboard said only "WAITING FOR REALSENSE CAMERA".
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from locallife_cloud import edge_client
from locallife_cloud.comparison import DualCameraCoordinator
from locallife_cloud.config import AppConfig
from locallife_cloud.server import create_app


def _run_main(argv: list[str], sdk_present: bool):
    started: list[str] = []
    reports: list[tuple[str, bool, str]] = []
    with mock.patch.object(sys, "argv", ["edge_client", *argv]), \
         mock.patch.object(edge_client, "realsense_sdk_available", return_value=sdk_present), \
         mock.patch.object(edge_client, "run_resilient_camera",
                           side_effect=lambda camera_id, factory, consumer: started.append(camera_id)), \
         mock.patch.object(edge_client, "report_camera_availability",
                           side_effect=lambda session, cloud, cid, ok, reason="": reports.append((cid, ok, reason))):
        edge_client.main()
    return started, reports


class EdgeStartupTests(unittest.TestCase):
    def test_dual_mode_without_the_sdk_starts_only_the_logitech(self) -> None:
        started, reports = _run_main(["--source", "dual"], sdk_present=False)
        self.assertEqual(started, ["logitech"])
        self.assertEqual(reports, [("realsense", False, "pyrealsense2 missing")])

    def test_dual_mode_with_the_sdk_is_unchanged(self) -> None:
        started, reports = _run_main(["--source", "dual"], sdk_present=True)
        self.assertEqual(sorted(started), ["logitech", "realsense"])
        self.assertEqual(reports, [])

    def test_realsense_only_mode_without_the_sdk_exits_with_a_clear_message(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            _run_main(["--source", "realsense"], sdk_present=False)
        self.assertIn("pyrealsense2 missing", str(raised.exception.code))
        self.assertIn("--source logitech", str(raised.exception.code))

    def test_the_sdk_check_does_not_import_the_sdk(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None) as find:
            self.assertFalse(edge_client.realsense_sdk_available())
        find.assert_called_once_with("pyrealsense2")


class ServerStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = AppConfig(
            detector_model="local-opencv-background", results_dir=Path(directory.name),
            enable_monocular_depth=False, enable_material_classification=False,
            enable_bucket_sync=False, restore_saved_baseline=False,
        )
        self.client = create_app(config, DualCameraCoordinator(config)).test_client()

    def test_a_reported_missing_sdk_shows_in_the_state(self) -> None:
        response = self.client.post("/api/cameras/realsense/availability",
                                    json={"available": False, "reason": "pyrealsense2 missing"})
        self.assertEqual(response.status_code, 200)
        state = self.client.get("/api/state").get_json()
        self.assertFalse(state["realsense_available"])
        self.assertEqual(state["cameras"]["realsense"]["unavailable_reason"], "pyrealsense2 missing")
        self.assertTrue(state["logitech_available"])

    def test_nothing_reported_means_available(self) -> None:
        state = self.client.get("/api/state").get_json()
        self.assertTrue(state["realsense_available"])
        self.assertIsNone(state["cameras"]["realsense"]["unavailable_reason"])

    def test_bad_reports_are_refused(self) -> None:
        self.assertEqual(self.client.post("/api/cameras/printer/availability",
                                          json={"available": False}).status_code, 404)
        self.assertEqual(self.client.post("/api/cameras/realsense/availability",
                                          json={"available": "no"}).status_code, 400)

    def test_the_dashboard_shows_the_reason(self) -> None:
        from locallife_cloud.dashboard import DUAL_DASHBOARD

        self.assertIn("unavailable_reason", DUAL_DASHBOARD)


if __name__ == "__main__":
    unittest.main()
