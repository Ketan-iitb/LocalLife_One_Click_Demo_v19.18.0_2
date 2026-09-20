"""Welcome page, run-mode selection, and the cloud-to-local fallback.

The launch runner is injected throughout, so these drive the whole decision path
-- cloud refused, cloud timing out, fallback allowed or disallowed -- without
starting a real process or touching Google Cloud.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from locallife_cloud.config import AppConfig
from locallife_cloud.launcher_service import (
    RUN_MODES,
    LaunchController,
    create_launcher_app,
)


def _config(directory, **overrides):
    return AppConfig(
        results_dir=Path(directory),
        detector_model="local-opencv-background",
        enable_monocular_depth=False,
        enable_bucket_sync=False,
        **overrides,
    )


def _ok(command, timeout):
    return subprocess.CompletedProcess(command, 0, "started", "")


def _fails(message="boom"):
    def runner(command, timeout):
        return subprocess.CompletedProcess(command, 1, "", message)
    return runner


def _times_out(command, timeout):
    raise subprocess.TimeoutExpired(command, timeout)


class _Controller(LaunchController):
    """Controller with the network probes pinned, so tests are deterministic."""

    def __init__(self, *args, cloud_ready=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._cloud_ready = cloud_ready

    def readiness(self):
        return {
            "version": "test",
            "internet": self._cloud_ready,
            "gcloud_installed": self._cloud_ready,
            "cloud_configured": self._cloud_ready,
            "cloud_available": self._cloud_ready,
            "pi_host": "locallife@pi.local",
            "pi_reachable": None,
            "realsense": None,
            "logitech": None,
            "dashboard_up": False,
            "allow_local_fallback": self.config.allow_local_fallback,
            "default_run_mode": self.config.default_run_mode,
            "cloud_startup_timeout_seconds": self.config.cloud_startup_timeout_seconds,
        }


def _controller(directory, *, runner=_ok, cloud_ready=True, **overrides):
    return _Controller(
        _config(directory, **overrides), runner=runner, cloud_ready=cloud_ready,
    )


class ModeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = self._directory.name

    def test_local_mode_never_touches_the_cloud(self) -> None:
        seen: list[list[str]] = []

        def runner(command, timeout):
            seen.append(command)
            return _ok(command, timeout)

        control = _controller(self.directory, runner=runner)
        launch = control.start("local")
        self.assertEqual(launch["mode"], "local")
        self.assertEqual(launch["phase"], "running")
        self.assertEqual(len(seen), 1)
        self.assertIn("Local", seen[0])
        self.assertNotIn("Cloud", seen[0])

    def test_cloud_mode_starts_the_cloud_launcher(self) -> None:
        seen: list[list[str]] = []

        def runner(command, timeout):
            seen.append(command)
            return _ok(command, timeout)

        launch = _controller(self.directory, runner=runner).start("cloud")
        self.assertEqual(launch["mode"], "cloud")
        self.assertIn("Cloud", seen[0])
        self.assertFalse(launch["fallback_used"])

    def test_cloud_mode_fails_loudly_rather_than_running_locally(self) -> None:
        # Claiming cloud while processing locally would corrupt every
        # processing_mode field, so cloud mode must not silently downgrade.
        launch = _controller(self.directory, cloud_ready=False).start("cloud")
        self.assertEqual(launch["phase"], "failed")
        self.assertIsNone(launch["mode"])
        self.assertIn("Cloud mode unavailable", launch["error"])

    def test_an_unknown_mode_is_rejected(self) -> None:
        control = _controller(self.directory)
        for mode in ("", "LOCAL; rm -rf /", "gpu", None):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    control.start(mode)

    def test_cancelling_before_startup_blocks_the_run(self) -> None:
        control = _controller(self.directory)
        launch = control.cancel()
        self.assertEqual(launch["phase"], "cancelled")
        self.assertIsNone(launch["mode"])


class AutomaticFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = self._directory.name

    def test_automatic_uses_the_cloud_when_it_is_healthy(self) -> None:
        launch = _controller(self.directory).start("auto")
        self.assertEqual(launch["mode"], "cloud")
        self.assertFalse(launch["fallback_used"])

    def test_automatic_falls_back_when_the_cloud_is_unconfigured(self) -> None:
        launch = _controller(self.directory, cloud_ready=False).start("auto")
        self.assertEqual(launch["mode"], "local")
        self.assertTrue(launch["fallback_used"])
        self.assertEqual(launch["phase"], "running")
        self.assertIsNotNone(launch["cloud_error"])

    def test_automatic_falls_back_when_cloud_startup_fails(self) -> None:
        calls: list[str] = []

        def runner(command, timeout):
            mode = command[command.index("-Mode") + 1]
            calls.append(mode)
            if mode == "Cloud":
                return subprocess.CompletedProcess(command, 1, "", "no GPU capacity")
            return _ok(command, timeout)

        launch = _controller(self.directory, runner=runner).start("auto")
        self.assertEqual(calls, ["Cloud", "Local"])
        self.assertEqual(launch["mode"], "local")
        self.assertIn("no GPU capacity", launch["cloud_error"])

    def test_a_cloud_timeout_resolves_rather_than_pending_forever(self) -> None:
        def runner(command, timeout):
            if command[command.index("-Mode") + 1] == "Cloud":
                return _times_out(command, timeout)
            return _ok(command, timeout)

        launch = _controller(
            self.directory, runner=runner, cloud_startup_timeout_seconds=5,
        ).start("auto")
        self.assertEqual(launch["mode"], "local")
        self.assertIn("exceeded 5s", launch["cloud_error"])
        self.assertNotIn(launch["phase"], {"checking", "starting-cloud"})

    def test_fallback_can_be_disabled(self) -> None:
        launch = _controller(
            self.directory, cloud_ready=False, allow_local_fallback=False,
        ).start("auto")
        self.assertEqual(launch["phase"], "failed")
        self.assertIsNone(launch["mode"])
        self.assertIn("fallback is disabled", launch["error"])

    def test_a_failed_local_start_reports_the_reason(self) -> None:
        launch = _controller(self.directory, runner=_fails("pwsh missing")).start("local")
        self.assertEqual(launch["phase"], "failed")
        self.assertIn("pwsh missing", launch["error"])


class ControlApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.control = _controller(self._directory.name)
        self.client = create_launcher_app(self.control.config, self.control).test_client()

    def test_the_welcome_page_offers_all_three_modes(self) -> None:
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn("Welcome to Local Life Demo", body)
        self.assertIn("How would you like to run the system?", body)
        for marker in (
            "RUN LOCALLY", "RUN WITH CLOUD GPU", "TRY CLOUD, FALL BACK TO LOCAL",
            "Raspberry Pi", "RealSense D435", "Logitech C920", "Internet",
            "Cloud configuration", "Diagnostics", "Cancel and go back",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_status_reports_readiness_and_launch_state(self) -> None:
        payload = self.client.get("/api/launcher/status").get_json()
        self.assertIn("readiness", payload)
        self.assertEqual(payload["launch"]["phase"], "idle")
        # Camera state belongs to the backend, which is not up yet.
        self.assertIsNone(payload["readiness"]["realsense"])

    def test_cloud_requires_an_explicit_billing_confirmation(self) -> None:
        response = self.client.post("/api/launcher/start", json={"mode": "cloud"})
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.get_json()["needs_confirmation"])
        self.assertEqual(self.control.state.phase, "idle")

    def test_a_confirmed_cloud_request_starts(self) -> None:
        response = self.client.post(
            "/api/launcher/start", json={"mode": "cloud", "confirm_billing": True}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["launch"]["mode"], "cloud")

    def test_the_api_rejects_anything_that_is_not_a_known_mode(self) -> None:
        for payload in ({"mode": "shell"}, {"mode": 5}, {}, {"mode": ["local"]}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/launcher/start", json=payload)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.control.state.phase, "idle")

    def test_the_mode_enum_is_closed(self) -> None:
        self.assertEqual(RUN_MODES, ("local", "cloud", "auto"))

    def test_the_launcher_command_is_fixed_argv_with_no_shell(self) -> None:
        # The browser supplies only the enum; everything else comes from config.
        command = self.control._command("local")
        self.assertIn("-File", command)
        self.assertEqual(command[command.index("-Mode") + 1], "Local")
        self.assertTrue(all(isinstance(part, str) for part in command))
        self.assertFalse(any(";" in part or "&&" in part or "|" in part for part in command))

    def test_the_dashboard_url_points_at_the_configured_port(self) -> None:
        self.client.post("/api/launcher/start", json={"mode": "local"})
        launch = self.client.get("/api/launcher/status").get_json()["launch"]
        self.assertEqual(launch["dashboard_url"], f"http://127.0.0.1:{self.control.config.port}/")


if __name__ == "__main__":
    unittest.main()
