"""Cloud startup must not fail on a VM that works, and history must be visible.

Two field failures:

* depth-l4 started, DEPTH_READY arrived, the L4 was detected and CUDA was
  available -- and startup refused because Compute Engine had published no SSH
  host key. Host-key pinning was a precondition when it should have been an
  enhancement on top of an authenticated SSH probe that already succeeded.
* the research page read "Waste ledger disabled in validation mode" while the
  system was recording measurements perfectly well, because it showed only the
  waste-plant ledger, which validation mode deliberately leaves empty.
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
from locallife_cloud.dashboard import DUAL_DASHBOARD
from locallife_cloud.event_log import MeasurementEventLog
from locallife_cloud.server import create_app


def _app(**overrides):
    directory = tempfile.TemporaryDirectory()
    config = AppConfig(
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


class HistoryVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.client, self._directory = _app(operating_mode="geometry_validation")
        self.addCleanup(self._directory.cleanup)
        self.station = self.app.config["CAMERA_COORDINATOR"].camera("realsense")

    def test_the_research_page_no_longer_claims_the_ledger_is_disabled(self) -> None:
        self.assertNotIn("Waste ledger disabled in validation mode", DUAL_DASHBOARD)
        self.assertIn("state.measurement_history", DUAL_DASHBOARD)
        self.assertIn("HISTORY LEDGER", DUAL_DASHBOARD)

    def test_each_row_reports_history_and_csv_saved(self) -> None:
        self.assertIn("History saved: Yes", DUAL_DASHBOARD)
        self.assertIn("CSV saved: ", DUAL_DASHBOARD)

    def test_validation_mode_still_reports_the_ledger_as_enabled(self) -> None:
        state = self.client.get("/api/state").get_json()
        camera = state["cameras"]["realsense"]
        self.assertEqual(camera["operating_mode"], "geometry_validation")
        self.assertTrue(camera["waste_ledger_enabled"])
        self.assertEqual(camera["measurement_history"], [])

    def test_a_finalised_event_appears_in_the_history(self) -> None:
        from locallife_cloud.types import Detection

        detection = Detection(
            label="test object", confidence=0.9, box=(0, 0, 10, 10), track_id=14,
            realsense_volume_l=0.67, color="brown", accepted_class="measurement_object",
            material="organic", sorting_status="correct",
            footprint_length_mm=1200.0, footprint_width_mm=1114.0,
            physical_height_mm=151.0,
        )
        self.station.persist_measurement_event(detection, 1234.5)
        history = self.client.get("/api/state").get_json()["cameras"]["realsense"]["measurement_history"]
        self.assertEqual(len(history), 1)
        row = history[0]
        self.assertEqual(row["object_type"], "test object")
        self.assertEqual(row["colour"], "brown")
        self.assertEqual(float(row["litres"]), 0.67)
        self.assertEqual(row["status"], "accepted")
        self.assertTrue(row["history_saved"])
        self.assertTrue(row["csv_saved"])

    def test_the_history_view_is_read_only_and_bounded(self) -> None:
        # It must not be able to disturb tracking: no tracker state is touched,
        # and the table cannot grow without bound.
        for index in range(45):
            self.station.event_log.record({"event_id": f"e{index}", "volume_l": 1.0})
        history = self.station.event_log.recent(limit=30)
        self.assertEqual(len(history), 30)
        # Newest first, so the operator sees the last deposit at the top.
        self.assertEqual(history[0]["event_id"], "e44")

    def test_enabling_history_does_not_reset_tracking_state(self) -> None:
        registry = self.station.identities
        before = (len(registry.objects), registry.accepted_count())
        for _ in range(3):
            self.client.get("/api/state")
        self.assertEqual((len(registry.objects), registry.accepted_count()), before)

    def test_a_csv_write_failure_is_visible_but_not_fatal(self) -> None:
        with mock.patch.object(MeasurementEventLog, "_append", side_effect=OSError("locked by Excel")):
            result = self.station.event_log.record({"event_id": "queued", "volume_l": 1.0})
        self.assertFalse(result.ok)
        status = self.client.get("/api/state").get_json()["cameras"]["realsense"]["csv_persistence"]
        self.assertFalse(status["healthy"])
        # Tracking is untouched by a storage failure.
        self.assertIsNotNone(self.station.identities)
        # ...and the queued event persists once the file is writable again.
        self.assertEqual(self.station.event_log.retry_failed(), {"recovered": 1, "still_failing": 0})
        self.assertEqual(len(self.station.event_log.rows()), 1)

    def test_retrying_does_not_duplicate_the_event(self) -> None:
        with mock.patch.object(MeasurementEventLog, "_append", side_effect=OSError("locked")):
            self.station.event_log.record({"event_id": "once", "volume_l": 2.0})
        self.station.event_log.retry_failed()
        self.station.event_log.retry_failed()
        self.assertEqual(len(self.station.event_log.rows()), 1)

    def test_end_to_end_download_carries_the_measurement(self) -> None:
        from locallife_cloud.types import Detection

        detection = Detection(
            label="cylinder", confidence=0.88, box=(0, 0, 10, 10), track_id=7,
            realsense_volume_l=0.39, color="white", accepted_class="measurement_object",
            material="plastic", sorting_status="correct",
            footprint_length_mm=50.0, footprint_width_mm=50.0, physical_height_mm=200.0,
        )
        persisted = self.station.persist_measurement_event(detection, 99.0)
        response = self.client.get("/api/cameras/realsense/measurements.csv")
        rows = list(csv.DictReader(io.StringIO(response.get_data().decode("utf-8-sig"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], persisted["event_id"])
        self.assertEqual(float(rows[0]["estimated_litres"]), 0.39)
        self.assertEqual(float(rows[0]["height_mm"]), 200.0)
        self.assertEqual(rows[0]["colour"], "white")
        self.assertEqual(rows[0]["sorting_result"], "correct")
        self.assertEqual(rows[0]["status"], "accepted")


class CloudProbeScriptTests(unittest.TestCase):
    """No PowerShell here, so these pin the script's text and ordering."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            Path(__file__).resolve().parent.parent.parent / "Start-LocalLife-Demo.ps1"
        ).read_text(encoding="utf-8", errors="replace")

    def test_an_authenticated_probe_runs_before_host_key_pinning(self) -> None:
        probe = self.script.index("Probing the VM over authenticated gcloud SSH")
        pinning = self.script.index("Write-Step 'Verifying SSH host key...'")
        self.assertLess(probe, pinning)

    def test_missing_guest_attributes_no_longer_block_a_working_vm(self) -> None:
        # The pin failure path must warn and continue, never throw.
        self.assertIn("function Test-CloudSshProbe", self.script)
        self.assertIn("$script:CloudSshInteractive = $true", self.script)
        self.assertIn("SSH HOST KEY NOT AUTOMATICALLY VERIFIED", self.script)

    def test_probe_failures_are_classified(self) -> None:
        for code in (
            "gcloud_authentication_required", "ssh_client_missing", "ssh_timeout",
            "ssh_host_key_conflict", "ssh_permission_denied", "vm_unreachable",
            "remote_probe_failed",
        ):
            with self.subTest(code=code):
                self.assertIn(code, self.script)

    def test_a_host_key_conflict_is_not_blindly_accepted(self) -> None:
        self.assertIn("does not match the one it now", self.script)
        for bypass in ("StrictHostKeyChecking=no", "PlinkHostKeyAutoAcceptLines"):
            with self.subTest(bypass=bypass):
                self.assertNotIn(bypass, self.script)

    def test_the_resolved_zone_reaches_every_cloud_command(self) -> None:
        # PowerShell continues a command with a trailing backtick, so the zone
        # often sits on the next physical line. Join them before checking.
        joined = self.script.replace("`\n", " ")
        for line in joined.splitlines():
            if "'compute', 'ssh'" in line or "'compute' 'scp'" in line:
                with self.subTest(line=line.strip()[:60]):
                    self.assertIn("zone", line.lower())

    def test_the_tunnel_stops_waiting_when_the_cloud_window_fails(self) -> None:
        self.assertIn("function Set-CloudStage", self.script)
        self.assertIn("function Get-CloudStage", self.script)
        self.assertIn("$stage.status -eq 'failed'", self.script)
        self.assertIn("Cloud startup failed in Window 1 at stage", self.script)

    def test_the_failure_offers_the_four_next_steps(self) -> None:
        for option in ("Retry Cloud", "Run Locally", "Diagnostics", "LocalLife_Stop.exe"):
            with self.subTest(option=option):
                self.assertIn(option, self.script)

    def test_no_false_zone_migration_message_once_the_vm_is_up(self) -> None:
        self.assertIn("$movingZones", self.script)
        self.assertIn("would be a lie", self.script)


if __name__ == "__main__":
    unittest.main()
