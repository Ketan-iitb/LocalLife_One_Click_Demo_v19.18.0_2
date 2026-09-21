"""Welcome page, run-mode selection, and the cloud-to-local fallback.

The launch runner is injected throughout, so these drive the whole decision path
-- cloud refused, cloud timing out, fallback allowed or disallowed -- without
starting a real process or touching Google Cloud.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        # Built from the REAL readiness with only the network probes pinned, so
        # this stub cannot silently drift out of step with the contract the
        # welcome page depends on. (An earlier hand-written dict did exactly
        # that and hid a missing key.)
        with mock.patch("locallife_cloud.launcher_service._port_open", return_value=self._cloud_ready), \
             mock.patch("locallife_cloud.launcher_service.shutil.which",
                        return_value="/usr/bin/gcloud" if self._cloud_ready else None):
            return super().readiness()


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


class LauncherScriptResolutionTests(unittest.TestCase):
    """Regression: the reported "-File ... does not exist" startup failure.

    The one-click .cmd chdirs into the Python package before starting the
    service, but Start-LocalLife-Demo.ps1 lives one level up at the repository
    root. Resolving it from the working directory therefore looked in
    LocalLife_Plug_and_Play_Local/ and failed.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.control = _controller(self._directory.name)

    def test_the_script_is_found_from_any_working_directory(self) -> None:
        expected = self.control.resolve_launcher_script()
        self.assertTrue(expected.is_file())
        self.assertEqual(expected.name, "Start-LocalLife-Demo.ps1")
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        for where in (Path(__file__).resolve().parent.parent, Path(self._directory.name)):
            with self.subTest(cwd=str(where)):
                os.chdir(where)
                self.assertEqual(self.control.resolve_launcher_script(), expected)

    def test_the_command_points_at_a_file_that_exists(self) -> None:
        command = self.control._command("local")
        self.assertTrue(Path(command[command.index("-File") + 1]).is_file())

    def test_a_missing_script_fails_the_launch_cleanly(self) -> None:
        control = _Controller(
            _config(self._directory.name),
            launcher_script=Path(self._directory.name) / "absent.ps1",
            runner=_ok,
        )
        launch = control.start("local")
        self.assertEqual(launch["phase"], "failed")
        self.assertIn("absent.ps1", launch["error"])


class CloudDefaultsTests(unittest.TestCase):
    def test_the_cloud_target_is_configured_out_of_the_box(self) -> None:
        # The welcome page showed "Cloud configuration: Not available" purely
        # because these defaulted to empty strings.
        config = AppConfig()
        self.assertEqual(config.gcp_project, "locallife-thesis-depth")
        self.assertEqual(config.cloud_vm_name, "depth-l4")
        self.assertTrue(config.pi_host)

    def test_cloud_project_overrides_the_default_project_id(self) -> None:
        self.assertEqual(AppConfig(cloud_project="other-project").gcp_project, "other-project")


class LauncherDiagnosticsTests(unittest.TestCase):
    """The page must be able to show which script path the service resolved."""

    def test_readiness_reports_the_resolved_script_and_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readiness = LaunchController(_config(directory)).readiness()
            self.assertIsNone(readiness["launcher_script_error"])
            self.assertTrue(readiness["launcher_script"].endswith("Start-LocalLife-Demo.ps1"))
            self.assertTrue(Path(readiness["launcher_script"]).is_file())
            self.assertTrue(readiness["working_directory"])

    def test_a_missing_script_is_reported_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            control = LaunchController(
                _config(directory), launcher_script=Path(directory) / "gone.ps1",
            )
            readiness = control.readiness()
            self.assertIsNone(readiness["launcher_script"])
            self.assertIn("gone.ps1", readiness["launcher_script_error"])


class LiveStartupStatusTests(unittest.TestCase):
    """The welcome page's live cloud status, served by the local control service.

    It has to work before the tunnel or the cloud dashboard exists, which is
    exactly why it lives here rather than on the backend.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.control = _controller(self._directory.name)
        self.client = create_launcher_app(
            _config(self._directory.name), self.control,
        ).test_client()

    def test_status_carries_the_startup_feed_and_checklist(self) -> None:
        payload = self.client.get("/api/launcher/status").get_json()
        for key in ("readiness", "launch", "startup", "checklist", "fully_ready"):
            self.assertIn(key, payload)
        self.assertEqual(payload["startup"]["stage"], "idle")
        self.assertFalse(payload["fully_ready"])
        # Nothing observed yet is Unknown, not False.
        self.assertIsNone(payload["checklist"]["camera_heartbeat"])

    def test_readiness_reports_the_cloud_target_and_its_timeouts(self) -> None:
        cloud = self.client.get("/api/launcher/status").get_json()["readiness"]["cloud"]
        self.assertEqual(cloud["vm_name"], "depth-l4")
        self.assertEqual(
            cloud["timeouts"]["first_frame_timeout_seconds"],
            AppConfig().first_frame_timeout_seconds,
        )

    def test_the_launcher_can_push_stage_updates(self) -> None:
        response = self.client.post("/api/launcher/startup-event", json={
            "stage": "verifying_ssh",
            "facts": {"ssh_verified": True, "external_ip": "34.6.166.251"},
            "message": "SSH identity verified",
        })
        self.assertEqual(response.status_code, 200)
        startup = response.get_json()["startup"]
        self.assertEqual(startup["stage"], "verifying_ssh")
        self.assertTrue(startup["facts"]["ssh_verified"])
        self.assertIn("SSH identity verified", startup["messages"])

    def test_a_pushed_failure_carries_stage_reason_and_next_steps(self) -> None:
        self.client.post("/api/launcher/startup-event", json={"stage": "verifying_ssh"})
        startup = self.client.post("/api/launcher/startup-event", json={
            "failure": "ssh_host_key_mismatch", "reason": "key is not the published one",
        }).get_json()["startup"]
        self.assertEqual(startup["failure"], "ssh_host_key_mismatch")
        self.assertEqual(startup["stage"], "verifying_ssh")
        self.assertEqual(
            startup["actions"], ["retry_cloud", "run_locally", "open_diagnostics"]
        )

    def test_the_status_api_rejects_anything_outside_the_known_enums(self) -> None:
        for payload in (
            {"stage": "rm -rf /"},
            {"failure": "; shutdown now"},
            {"facts": "not-an-object"},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(
                    self.client.post("/api/launcher/startup-event", json=payload).status_code,
                    400,
                )

    def test_the_stage_vocabulary_is_published_for_the_page(self) -> None:
        payload = self.client.get("/api/launcher/stages").get_json()
        self.assertIn("verifying_ssh", payload["stages"])
        self.assertIn("ssh_host_key_mismatch", payload["failures"])

    def test_the_page_renders_the_structured_feed_not_console_text(self) -> None:
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn("/api/launcher/status", body)
        for marker in (
            "SSH host key", "Deployment version", "Tunnel", "First frame",
            "Retry cloud", "Run locally instead", "Open diagnostics",
            "address, not identity",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)


class CloudStartupBlockingTests(unittest.TestCase):
    """Regression: "cloud wala atak raha hai" -- cloud startup sat there forever.

    gpu.py recreates the VM on a zone move, so the host key legitimately
    changes and PuTTY stops at "Update cached key?" in a window nobody is
    watching. The fix must unblock that WITHOUT accepting an unverified key, so
    these assert the security properties as much as the unblocking.

    There is no PowerShell interpreter in this environment, so these assert on
    the script's text: they cannot prove the script runs, only that the fix is
    still present and still ordered correctly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            Path(__file__).resolve().parent.parent.parent / "Start-LocalLife-Demo.ps1"
        ).read_text(encoding="utf-8", errors="replace")

    def test_identity_is_verified_against_the_authenticated_api(self) -> None:
        self.assertIn("function Assert-CloudSshIdentity", self.script)
        self.assertIn("locallife_cloud.cloud_ssh", self.script)
        self.assertIn("Assert-CloudSshIdentity -PythonExe", self.script)

    def test_the_verifier_is_run_with_the_package_on_the_python_path(self) -> None:
        # Regression: this script sits at the repository root while the package
        # is one level down, so `python -m locallife_cloud.cloud_ssh` could not
        # import it and failed with ModuleNotFoundError before producing any
        # verdict.
        self.assertIn("$env:PYTHONPATH = $projectRoot", self.script)
        self.assertIn("$env:PYTHONPATH = $previousPythonPath", self.script)

    def test_a_check_that_could_not_run_is_not_called_a_mismatch(self) -> None:
        # The empty result was reported as "SSH host key mismatch", sending the
        # operator after a security incident that had not happened.
        self.assertIn("The VM identity check could not run", self.script)
        self.assertIn("ModuleNotFoundError|No module named", self.script)
        self.assertNotIn("SSH host key mismatch: the VM identity check", self.script)

    def test_the_verifier_s_stderr_is_captured_rather_than_swallowed(self) -> None:
        self.assertIn("ssh-verify-stderr.txt", self.script)
        self.assertIn("2>$errorFile", self.script)

    def test_nothing_auto_accepts_an_unverified_host_key(self) -> None:
        # The v21 approach, and every other form of blanket acceptance.
        for bypass in (
            "PlinkHostKeyAutoAcceptLines",
            "strict-host-key-checking=no",
            "StrictHostKeyChecking=no",
            "StrictHostKeyChecking=accept-new' ('--zone",
        ):
            with self.subTest(bypass=bypass):
                self.assertNotIn(bypass, self.script)

    def test_cloud_connections_pin_a_known_hosts_file_and_check_strictly(self) -> None:
        self.assertIn("'StrictHostKeyChecking=yes'", self.script)
        self.assertIn("UserKnownHostsFile=", self.script)
        self.assertIn("function Get-CloudKnownHostsPath", self.script)

    def test_an_unverified_identity_hands_the_decision_to_a_person(self) -> None:
        # Hard-failing here blocked cloud mode entirely when Google had not
        # published a host key. It now falls back to the reference
        # deployment's interactive path -- a HUMAN confirms the fingerprint --
        # rather than either stopping dead or auto-accepting.
        self.assertIn("if (-not $report.verified)", self.script)
        self.assertIn("$script:CloudSshInteractive = $true", self.script)
        self.assertIn("SSH HOST KEY NOT AUTOMATICALLY VERIFIED", self.script)

    def test_identity_is_pinned_before_the_other_windows_are_released(self) -> None:
        # Windows 2 and 3 start their own SSH sessions as soon as the zone file
        # exists, so pinning after that write would let them race an unpinned host.
        pinned = self.script.index("Assert-CloudSshIdentity -PythonExe")
        zone_file_written = self.script.index("Set-Content -LiteralPath $zoneFile")
        self.assertLess(pinned, zone_file_written)

    def test_the_tunnel_and_command_paths_both_go_through_one_helper(self) -> None:
        self.assertIn("function Invoke-VerifiedCloudSsh", self.script)
        self.assertIn("function Invoke-VerifiedCloudScp", self.script)

    def test_the_gcloud_fallback_is_gated_on_verification_failing(self) -> None:
        # gcloud compute ssh is allowed again -- it is the v3.2 reference
        # deployment's path, where PuTTY shows the operator the fingerprint and
        # they answer once. It must be reachable ONLY when automatic
        # verification could not establish identity, never as the default.
        self.assertIn("$script:CloudSshInteractive = $false", self.script)
        for line_number, line in enumerate(self.script.splitlines(), 1):
            if "'compute', 'ssh'" in line or "'compute' 'scp'" in line:
                with self.subTest(line=line_number):
                    # Every such call sits inside the interactive-fallback block.
                    preceding = "\n".join(self.script.splitlines()[:line_number])
                    self.assertIn("if ($script:CloudSshInteractive)", preceding)

    def test_the_fallback_never_claims_the_key_was_verified(self) -> None:
        self.assertIn("SSH HOST KEY NOT AUTOMATICALLY VERIFIED", self.script)
        self.assertIn("before answering y", self.script)

    def test_the_wait_message_does_not_cry_wolf_after_one_minute(self) -> None:
        # Observed startup with capacity in the usual zone: ~79 seconds. The
        # first notice fires at one minute, so it must not announce 15-30.
        self.assertIn("A normal start takes about 1-3 minutes.", self.script)
        long_wait = "image capture plus a fresh VM in a new region can take 15-30 minutes"
        self.assertIn(long_wait, self.script)
        # The branch now also requires the VM to still be starting, so the
        # long-wait wording cannot appear for a VM that is already up.
        guard = "if ($minutesWaited -lt 5 -or -not $movingZones)"
        self.assertIn(guard, self.script)
        # The long-wait wording must live on the far side of that branch.
        self.assertGreater(
            self.script.index(long_wait), self.script.index(guard)
        )
