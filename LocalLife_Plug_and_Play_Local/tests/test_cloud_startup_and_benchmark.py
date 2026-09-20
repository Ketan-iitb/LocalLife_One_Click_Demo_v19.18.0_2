"""Verified SSH identity, the startup state machine, and benchmark metrics.

The security tests are the important ones here: they pin the properties that
make the host-key fix a fix rather than a bypass -- nothing auto-accepted, a
mismatch failing closed, and no unrelated host's trust touched.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from locallife_cloud.benchmark import (
    BENCHMARK_COLUMNS,
    BenchmarkSession,
    ObjectOutcome,
    compare,
)
from locallife_cloud.cloud_ssh import (
    STATUS_RESOLVING,
    STATUS_VERIFIED,
    STATUS_VERIFYING,
    CloudSshVerifier,
    HostKey,
    verify_cloud_ssh,
)
from locallife_cloud.cloud_startup import (
    FAILURES,
    STAGES,
    TIMING_KEYS,
    CloudStartupMachine,
    is_fully_ready,
    readiness_checklist,
)

def _ed25519(seed: int) -> str:
    """A well-formed ed25519 host key blob, exactly as guest attributes hold it.

    SSH wire format: length-prefixed "ssh-ed25519", then the 32-byte key. Built
    rather than hard-coded so the fingerprint test checks real decoding instead
    of falling back to the unreadable-key path.
    """
    import base64 as _base64

    name = b"ssh-ed25519"
    blob = len(name).to_bytes(4, "big") + name + (32).to_bytes(4, "big") + bytes([seed]) * 32
    return _base64.b64encode(blob).decode("ascii")


_ED25519 = _ed25519(1)
_OTHER_ED25519 = _ed25519(2)

_DESCRIBE = json.dumps({
    "name": "depth-l4",
    "id": "8123456789012345678",
    "zone": "https://www.googleapis.com/compute/v1/projects/p/zones/europe-west4-a",
    "networkInterfaces": [{
        "networkIP": "10.164.0.2",
        "accessConfigs": [{"natIP": "34.6.166.251"}],
    }],
})


def _fake_gcloud(describe: str = _DESCRIBE, hostkeys: str | None = None, fail: str = ""):
    """A gcloud stand-in, so the whole decision path runs without a project."""
    keys = hostkeys if hostkeys is not None else json.dumps(
        [{"key": "ssh-ed25519", "value": _ED25519}]
    )

    def runner(command):
        joined = " ".join(command)
        if fail and fail in joined:
            return subprocess.CompletedProcess(command, 1, "", "boom")
        if "get-guest-attributes" in joined:
            return subprocess.CompletedProcess(command, 0, keys, "")
        if "describe" in joined:
            return subprocess.CompletedProcess(command, 0, describe, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return runner


class HostKeyVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.known_hosts = Path(self._directory.name) / "cloud_known_hosts"

    def _verifier(self, **kwargs) -> CloudSshVerifier:
        return CloudSshVerifier(
            "depth-l4", "europe-west4-a", "locallife-thesis-depth",
            runner=_fake_gcloud(**kwargs),
        )

    def test_identity_comes_from_the_authenticated_api(self) -> None:
        identity = self._verifier().resolve_identity()
        self.assertEqual(identity.instance_id, "8123456789012345678")
        self.assertEqual(identity.external_ip, "34.6.166.251")
        self.assertEqual(identity.zone, "europe-west4-a")
        # The IP is reported but is not the identity: instance id is.
        self.assertIn("instance_id", identity.to_dict())

    def test_pinning_writes_only_this_vm_s_published_keys(self) -> None:
        result = self._verifier().pin(self.known_hosts)
        self.assertTrue(result.verified)
        lines = self.known_hosts.read_text(encoding="utf-8").strip().splitlines()
        # One line per alias: external IP, internal IP, instance name.
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertIn("ssh-ed25519", line)
            self.assertIn(_ED25519, line)
        hosts = {line.split()[0] for line in lines}
        self.assertEqual(hosts, {"34.6.166.251", "10.164.0.2", "depth-l4"})

    def test_a_recreated_vm_replaces_only_its_own_entry(self) -> None:
        # An unrelated host the operator already trusts, in their own file.
        unrelated = Path(self._directory.name) / "user_known_hosts"
        unrelated.write_text("github.com ssh-ed25519 AAAAsomething\n", encoding="utf-8")

        self._verifier().pin(self.known_hosts)
        # gpu.py recreates the VM: same name and IP, genuinely new key.
        rotated = CloudSshVerifier(
            "depth-l4", "europe-west4-a", "locallife-thesis-depth",
            runner=_fake_gcloud(
                hostkeys=json.dumps([{"key": "ssh-ed25519", "value": _OTHER_ED25519}]),
            ),
        )
        result = rotated.pin(self.known_hosts)
        self.assertTrue(result.verified)
        body = self.known_hosts.read_text(encoding="utf-8")
        self.assertIn(_OTHER_ED25519, body)
        self.assertNotIn(_ED25519, body)
        # The operator's own trusted hosts are untouched.
        self.assertEqual(
            unrelated.read_text(encoding="utf-8"), "github.com ssh-ed25519 AAAAsomething\n"
        )

    def test_an_unchanged_vm_is_reported_as_unchanged(self) -> None:
        self._verifier().pin(self.known_hosts)
        self.assertTrue(self._verifier().pin(self.known_hosts).unchanged)

    def test_missing_published_keys_stop_with_a_security_error(self) -> None:
        result = self._verifier(hostkeys="[]").pin(self.known_hosts)
        self.assertFalse(result.verified)
        self.assertEqual(result.status, STATUS_VERIFYING)
        self.assertIn("No usable SSH host key", result.error)
        # Nothing is pinned when identity could not be established.
        self.assertFalse(self.known_hosts.exists())

    def test_an_unresolvable_instance_stops_before_connecting(self) -> None:
        result = self._verifier(fail="describe").pin(self.known_hosts)
        self.assertFalse(result.verified)
        self.assertEqual(result.status, STATUS_RESOLVING)
        self.assertFalse(self.known_hosts.exists())

    def test_ssh_options_never_relax_host_key_checking(self) -> None:
        options = CloudSshVerifier.ssh_options(self.known_hosts)
        joined = " ".join(options)
        self.assertIn("StrictHostKeyChecking=yes", joined)
        self.assertIn(f"UserKnownHostsFile={self.known_hosts}", joined)
        self.assertIn("BatchMode=yes", joined)
        # The bypasses this fix exists to avoid.
        for forbidden in ("StrictHostKeyChecking=no", "accept-new", "CheckHostIP=no"):
            self.assertNotIn(forbidden, joined)

    def test_the_report_carries_fingerprints_not_key_material(self) -> None:
        report = verify_cloud_ssh(
            "depth-l4", "europe-west4-a", "p", self.known_hosts,
            runner=_fake_gcloud(),
        )
        self.assertTrue(report["verified"])
        self.assertEqual(report["status"], STATUS_VERIFIED)
        fingerprint = report["fingerprints"][0]["fingerprint"]
        self.assertTrue(fingerprint.startswith("SHA256:"))
        self.assertNotIn(_ED25519, json.dumps(report))
        self.assertEqual(report["identity"]["instance_id"], "8123456789012345678")

    def test_the_fingerprint_matches_the_openssh_form(self) -> None:
        import base64, hashlib

        key = HostKey("ssh-ed25519", _ED25519)
        expected = "SHA256:" + base64.b64encode(
            hashlib.sha256(base64.b64decode(_ED25519)).digest()
        ).decode().rstrip("=")
        self.assertEqual(key.fingerprint, expected)

    def test_an_unreadable_key_does_not_crash_the_report(self) -> None:
        self.assertEqual(HostKey("ssh-rsa", "not base64!!").fingerprint, "SHA256:<unreadable>")


class StartupStateMachineTests(unittest.TestCase):
    def test_the_happy_path_visits_every_stage_and_times_it(self) -> None:
        clock = iter([float(n) for n in range(200)])
        machine = CloudStartupMachine(clock=lambda: next(clock))
        machine.begin()
        for stage in STAGES[1:-1]:
            machine.enter(stage)
            machine.complete_stage()
        machine.enter("ready")
        state = machine.state.to_dict()
        self.assertTrue(state["ready"])
        self.assertIsNone(state["failure"])
        timings = state["timings"]
        self.assertEqual(set(timings), set(TIMING_KEYS))
        for key in ("vm_start_seconds", "ssh_verification_seconds", "tunnel_ready_seconds"):
            self.assertIsNotNone(timings[key])
        self.assertIsNotNone(timings["total_ready_seconds"])

    def test_a_failure_names_the_stage_reason_elapsed_and_next_steps(self) -> None:
        machine = CloudStartupMachine()
        machine.begin()
        machine.enter("verifying_ssh")
        state = machine.fail("ssh_host_key_mismatch", "presented key is not the published one")
        self.assertEqual(state["failure"], "ssh_host_key_mismatch")
        self.assertEqual(state["stage"], "verifying_ssh")
        self.assertIn("published", state["reason"])
        self.assertGreaterEqual(state["elapsed_seconds"], 0.0)
        # Never an endless spinner: there is always somewhere to go.
        self.assertEqual(
            state["actions"], ["retry_cloud", "run_locally", "open_diagnostics"]
        )

    def test_every_declared_failure_maps_to_a_real_stage(self) -> None:
        for failure in FAILURES:
            with self.subTest(failure=failure):
                machine = CloudStartupMachine()
                machine.begin()
                state = machine.fail(failure, "reason")
                self.assertIn(state["stage"], STAGES)

    def test_a_stage_that_overruns_its_budget_fails_rather_than_waits(self) -> None:
        ticks = iter([0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0])
        machine = CloudStartupMachine(clock=lambda: next(ticks))
        machine.begin()
        ok, _ = machine.run_stage(
            "starting_vm", lambda: "done", failure="vm_start_timeout", timeout=10.0,
        )
        self.assertFalse(ok)
        self.assertEqual(machine.state.failure, "vm_start_timeout")
        self.assertIn("over the 10s budget", machine.state.reason)

    def test_a_raising_stage_becomes_a_named_failure(self) -> None:
        machine = CloudStartupMachine()
        machine.begin()

        def boom():
            raise RuntimeError("no capacity in any zone")

        ok, value = machine.run_stage(
            "starting_vm", boom, failure="gpu_unavailable", timeout=60.0,
        )
        self.assertFalse(ok)
        self.assertIsNone(value)
        self.assertIn("no capacity", machine.state.reason)

    def test_an_unknown_stage_or_failure_is_rejected(self) -> None:
        machine = CloudStartupMachine()
        with self.assertRaises(ValueError):
            machine.enter("teleporting")
        with self.assertRaises(ValueError):
            machine.fail("vibes_wrong", "reason")

    def test_ready_means_the_whole_path_is_up(self) -> None:
        facts = {key: True for key in readiness_checklist({})}
        self.assertTrue(is_fully_ready(facts))
        # A running VM with a loaded model is not a working demonstration.
        facts["camera_heartbeat"] = None
        self.assertFalse(is_fully_ready(facts))
        facts["camera_heartbeat"] = False
        self.assertFalse(is_fully_ready(facts))

    def test_facts_are_recorded_for_the_status_page(self) -> None:
        machine = CloudStartupMachine()
        machine.begin()
        machine.fact(zone="europe-west4-a", external_ip="34.6.166.251", ssh_verified=True)
        state = machine.state.to_dict()
        self.assertEqual(state["facts"]["zone"], "europe-west4-a")
        self.assertTrue(state["facts"]["ssh_verified"])


class BenchmarkMetricTests(unittest.TestCase):
    def _session(self, mode: str = "local", input_id: str = "clip-1") -> BenchmarkSession:
        session = BenchmarkSession("s1", mode, recorded_input_id=input_id)
        base = session.started_at
        for index in range(5):
            sample = session.record_capture(index, at=base + index)
            session.record_submitted(sample, at=base + index + 0.010, queue_depth=index)
            sample.inference_started_at = base + index + 0.020
            sample.inference_finished_at = base + index + 0.120
            sample.result_available_at = base + index + 0.150
            session.record_displayed()
        return session

    def test_throughput_is_reported_as_separate_rates(self) -> None:
        rates = self._session().throughput()
        for key in ("capture_fps", "submitted_fps", "inference_fps", "completed_fps", "display_fps"):
            self.assertIn(key, rates)
        # There must be no single ambiguous "fps" key to quote.
        self.assertNotIn("fps", rates)

    def test_latency_is_broken_down_into_its_stages(self) -> None:
        latency = self._session("cloud").latency()
        self.assertAlmostEqual(latency["median_ms"], 150.0, places=1)
        self.assertAlmostEqual(latency["upload_ms"], 10.0, places=1)
        self.assertAlmostEqual(latency["queue_ms"], 10.0, places=1)
        self.assertAlmostEqual(latency["inference_ms"], 100.0, places=1)
        self.assertAlmostEqual(latency["return_ms"], 30.0, places=1)
        self.assertIsNotNone(latency["p95_ms"])
        self.assertIsNotNone(latency["max_ms"])

    def test_reliability_counts_drops_and_stale_frames(self) -> None:
        session = self._session()
        dropped = session.record_capture(99)
        dropped.dropped = True
        stale = session.record_capture(100)
        stale.stale = True
        reliability = session.reliability()
        self.assertEqual(reliability["frames_captured"], 7)
        self.assertEqual(reliability["frames_processed"], 5)
        self.assertEqual(reliability["frames_dropped"], 1)
        self.assertEqual(reliability["stale_frames_discarded"], 1)
        self.assertAlmostEqual(reliability["frame_drop_percent"], 14.29, places=1)

    def test_an_uncollectable_metric_is_unavailable_not_zero(self) -> None:
        resources = self._session().resources()
        # No GPU in this environment: it must say so rather than report 0%.
        self.assertIsNone(resources["gpu_utilization_percent"])
        self.assertIsNone(resources["pi_temperature_c"])

    def test_accuracy_scores_only_what_has_ground_truth(self) -> None:
        session = self._session()
        session.record_outcome(ObjectOutcome(
            "e1", "local", estimated_litres=10.0, ground_truth_litres=8.0,
            colour_correct=True, material_correct=False, sorting_correct=True,
            bag_count_correct=True,
        ))
        session.record_outcome(ObjectOutcome("e2", "local", estimated_litres=5.0))
        accuracy = session.accuracy()
        self.assertEqual(accuracy["objects"], 2)
        self.assertEqual(accuracy["objects_with_ground_truth"], 1)
        self.assertAlmostEqual(accuracy["mean_volume_error_percent"], 25.0)
        self.assertEqual(accuracy["colour_accuracy_percent"], 100.0)
        self.assertEqual(accuracy["material_accuracy_percent"], 0.0)

    def test_an_outcome_without_ground_truth_reports_no_error(self) -> None:
        outcome = ObjectOutcome("e1", "local", estimated_litres=5.0)
        self.assertIsNone(outcome.absolute_error_litres)
        self.assertIsNone(outcome.percentage_error)

    def test_the_export_carries_every_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = self._session()
            session.record_outcome(ObjectOutcome(
                "e1", "local", estimated_litres=10.0, ground_truth_litres=8.0,
            ))
            path = session.write_csv(Path(directory))
            body = path.read_text(encoding="utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(body)))
            self.assertEqual(list(rows[0]), BENCHMARK_COLUMNS)
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["processing_mode"], "local")
            self.assertEqual(float(rows[0]["absolute_error_litres"]), 2.0)

    def test_a_comparison_refuses_different_inputs(self) -> None:
        verdict = compare(self._session("local", "clip-1"), self._session("cloud", "clip-2"))["verdict"]
        self.assertFalse(verdict["comparable"])
        self.assertIn("same recorded input", verdict["note"])
        self.assertIsNone(verdict["faster"])

    def test_a_comparison_on_one_input_names_the_faster_mode(self) -> None:
        local = self._session("local", "clip-1")
        cloud = BenchmarkSession("s1", "cloud", recorded_input_id="clip-1")
        base = cloud.started_at
        for index in range(5):
            sample = cloud.record_capture(index, at=base + index)
            cloud.record_submitted(sample, at=base + index + 0.005)
            sample.inference_started_at = base + index + 0.010
            sample.inference_finished_at = base + index + 0.040
            sample.result_available_at = base + index + 0.050
        verdict = compare(local, cloud)["verdict"]
        self.assertTrue(verdict["comparable"])
        self.assertEqual(verdict["faster"], "cloud")
        self.assertGreater(verdict["p95_difference_ms"], 0)

    def test_a_comparison_with_one_side_missing_says_so(self) -> None:
        verdict = compare(self._session(), None)["verdict"]
        self.assertFalse(verdict["comparable"])
        self.assertIn("Run both modes", verdict["note"])


if __name__ == "__main__":
    unittest.main()
