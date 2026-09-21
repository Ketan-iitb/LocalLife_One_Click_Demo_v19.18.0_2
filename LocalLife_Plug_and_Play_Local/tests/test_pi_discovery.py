"""Finding the Raspberry Pi when its mDNS name will not resolve.

Field failure: "ssh: Could not resolve hostname locallife.local: No such host
is known", with the Pi powered on and on the same network. The name was the
only broken part.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from locallife_cloud.pi_discovery import (
    candidates,
    remember,
    resolve_pi,
    split_pi_host,
    unreachable_message,
    PiAddress,
)


def _probe(reachable: set[str]):
    return lambda host, port, timeout: host in reachable


class PiDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.cache = Path(self._directory.name) / "pi-address.json"

    def test_the_host_string_is_split_correctly(self) -> None:
        self.assertEqual(split_pi_host("locallife@locallife.local"),
                         ("locallife", "locallife.local", 22))
        self.assertEqual(split_pi_host("pi@10.0.0.5:2222"), ("pi", "10.0.0.5", 2222))
        self.assertEqual(split_pi_host("10.0.0.5"), ("", "10.0.0.5", 22))

    def test_the_configured_name_is_preferred_when_it_works(self) -> None:
        found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                           probe=_probe({"locallife.local"}))
        self.assertIsNotNone(found)
        self.assertEqual(found.host, "locallife.local")
        self.assertEqual(found.source, "configured")
        self.assertEqual(found.target, "locallife@locallife.local")

    def test_the_short_name_is_tried_when_mdns_fails(self) -> None:
        # Exactly the reported failure: .local does not resolve, the plain
        # hostname does (unicast DNS, LLMNR or a hosts entry).
        found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                           probe=_probe({"locallife"}))
        self.assertIsNotNone(found)
        self.assertEqual(found.host, "locallife")
        self.assertEqual(found.source, "short-name")
        self.assertEqual(found.target, "locallife@locallife")

    def test_an_explicit_ip_override_is_used(self) -> None:
        with mock.patch.dict("os.environ", {"LOCALLIFE_PI_IP": "192.168.1.42"}):
            found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                               probe=_probe({"192.168.1.42"}))
        self.assertEqual(found.host, "192.168.1.42")
        self.assertEqual(found.source, "LOCALLIFE_PI_IP")

    def test_a_working_address_is_remembered_and_reused(self) -> None:
        with mock.patch.dict("os.environ", {"LOCALLIFE_PI_IP": "192.168.1.42"}):
            resolve_pi("locallife@locallife.local", cache_path=self.cache,
                       probe=_probe({"192.168.1.42"}))
        self.assertTrue(self.cache.is_file())
        self.assertEqual(json.loads(self.cache.read_text())["host"], "192.168.1.42")
        # Next run: no env override, mDNS still broken, cache saves it.
        found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                           probe=_probe({"192.168.1.42"}))
        self.assertEqual(found.host, "192.168.1.42")
        self.assertEqual(found.source, "last-known-address")

    def test_a_stale_cached_address_does_not_win_over_a_working_name(self) -> None:
        remember(PiAddress(host="10.9.9.9", source="last-known-address"), self.cache)
        found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                           probe=_probe({"locallife.local"}))
        self.assertEqual(found.host, "locallife.local")

    def test_nothing_reachable_returns_none_rather_than_guessing(self) -> None:
        self.assertIsNone(resolve_pi("locallife@locallife.local",
                                     cache_path=self.cache, probe=_probe(set())))

    def test_the_failure_message_names_what_was_tried_and_what_to_do(self) -> None:
        message = unreachable_message("locallife@locallife.local", self.cache)
        self.assertIn("locallife.local", message)
        self.assertIn("locallife", message)
        # It must correct the wrong conclusion the old message invited.
        self.assertIn("not a power one", message)
        for remedy in ("hostname -I", "LOCALLIFE_PI_IP", "hosts", "Bonjour", "-PiHost"):
            with self.subTest(remedy=remedy):
                self.assertIn(remedy, message)

    def test_candidates_are_unique_and_ordered(self) -> None:
        found = [host for host, _ in candidates("locallife@locallife.local", self.cache)]
        self.assertEqual(found[0], "locallife.local")
        self.assertEqual(found[1], "locallife")
        self.assertEqual(len(found), len(set(found)))

    def test_an_unwritable_cache_is_not_fatal(self) -> None:
        with mock.patch.object(Path, "write_text", side_effect=OSError("read-only")):
            found = resolve_pi("locallife@locallife.local", cache_path=self.cache,
                               probe=_probe({"locallife.local"}))
        self.assertIsNotNone(found)


class ReadinessTests(unittest.TestCase):
    """The welcome page said "Not reachable" and offered nothing to act on."""

    def _readiness(self, reachable: set[str]):
        from locallife_cloud.config import AppConfig
        from locallife_cloud.launcher_service import LaunchController

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = AppConfig(results_dir=Path(directory.name))
        control = LaunchController(config)
        with mock.patch("locallife_cloud.pi_discovery._reachable",
                        side_effect=lambda h, p, t: h in reachable), \
             mock.patch("locallife_cloud.launcher_service._port_open", return_value=False):
            return control.readiness()

    def test_a_pi_found_by_fallback_reads_as_reachable(self) -> None:
        readiness = self._readiness({"locallife"})
        self.assertTrue(readiness["pi_reachable"])
        self.assertEqual(readiness["pi_address"]["host"], "locallife")
        self.assertIsNone(readiness["pi_hint"])

    def test_an_unreachable_pi_carries_an_actionable_hint(self) -> None:
        readiness = self._readiness(set())
        self.assertFalse(readiness["pi_reachable"])
        self.assertIsNone(readiness["pi_address"])
        self.assertIn("LOCALLIFE_PI_IP", readiness["pi_hint"])

    def test_the_page_shows_the_hint(self) -> None:
        from locallife_cloud.launcher_page import WELCOME_PAGE

        self.assertIn("pi-hint", WELCOME_PAGE)
        self.assertIn("ready.pi_address", WELCOME_PAGE)


if __name__ == "__main__":
    unittest.main()
