"""B37: finding the Raspberry Pi when its name will never resolve.

The welcome page read "Raspberry Pi: Not reachable" on every start, having tried
only `locallife.local` and `locallife`. The old discovery remembered the Pi's
address only after a successful connection -- and on a network without mDNS
the name never resolves, so it never connected, so nothing was ever
remembered. The failure was a closed loop, not bad luck.

These tests pin the way out: find the Pi by its Raspberry Pi hardware address,
passively from the neighbour table first, then with one brief check of the
laptop's own /24 -- and never accept some other machine that merely runs SSH.
No real network is touched here; every lookup is injected.
"""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from locallife_cloud import pi_discovery
from locallife_cloud.pi_discovery import (
    DISCOVERY_ENV,
    is_raspberry_pi_mac,
    normalise_mac,
    parse_neighbours,
    raspberry_pi_neighbours,
    resolve_pi,
    unreachable_message,
)

WINDOWS_ARP = """
Interface: 192.168.1.23 --- 0x7
  Internet Address      Physical Address      Type
  192.168.1.1           a4-2b-b0-11-22-33     dynamic
  192.168.1.40          dc-a6-32-9a-bc-de     dynamic
  192.168.1.77          00-1a-2b-3c-4d-5e     dynamic
  192.168.1.255         ff-ff-ff-ff-ff-ff     static
"""
LINUX_NEIGH = """
192.168.137.12 dev wlan0 lladdr d8:3a:dd:01:02:03 REACHABLE
192.168.137.1 dev wlan0 lladdr 10:20:30:40:50:60 STALE
"""
MAC_ARP = "? (10.0.0.9) at b8:27:eb:2:3:4 on en0 ifscope [ethernet]\n"

SUBNET = [ipaddress.IPv4Network("192.168.1.0/24")]


def _probe(open_hosts: set[str]):
    return lambda host, port, timeout: host in open_hosts


class NeighbourParsingTests(unittest.TestCase):
    def test_windows_linux_and_mac_formats_are_read(self) -> None:
        self.assertIn(("192.168.1.40", "dc:a6:32:9a:bc:de"), parse_neighbours(WINDOWS_ARP))
        self.assertIn(("192.168.137.12", "d8:3a:dd:01:02:03"), parse_neighbours(LINUX_NEIGH))
        self.assertEqual(parse_neighbours(MAC_ARP), [("10.0.0.9", "b8:27:eb:02:03:04")])

    def test_only_raspberry_pi_hardware_is_picked(self) -> None:
        self.assertEqual(raspberry_pi_neighbours(WINDOWS_ARP), ["192.168.1.40"])
        self.assertEqual(raspberry_pi_neighbours(LINUX_NEIGH), ["192.168.137.12"])

    def test_mac_forms_normalise(self) -> None:
        self.assertEqual(normalise_mac("B8-27-EB-1-2-3"), "b8:27:eb:01:02:03")
        self.assertTrue(is_raspberry_pi_mac("2C:CF:67:00:00:01"))
        self.assertFalse(is_raspberry_pi_mac("00:1a:2b:3c:4d:5e"))


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        pi_discovery._last_sweep.clear()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "pi-address.json"
        self.shared = Path(directory.name) / "shared.json"

    def test_the_closed_loop_is_broken_by_the_neighbour_table(self) -> None:
        """No name resolves, nothing is cached, and the Pi is still found."""
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache,
            probe=_probe({"192.168.1.40"}),
            neighbours=lambda: WINDOWS_ARP, subnets=lambda: SUBNET, allow_sweep=False,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.host, "192.168.1.40")
        self.assertEqual(found.target, "locallife@192.168.1.40")
        self.assertEqual(found.source, "network-neighbour-raspberry-pi")

    def test_what_was_found_is_remembered_so_the_search_happens_once(self) -> None:
        resolve_pi(
            "locallife@locallife.local", cache_path=self.cache, also_read=[self.shared],
            probe=_probe({"192.168.1.40"}),
            neighbours=lambda: WINDOWS_ARP, subnets=lambda: SUBNET, allow_sweep=False,
        )
        for path in (self.cache, self.shared):
            with self.subTest(cache=path.name):
                self.assertEqual(json.loads(path.read_text())["host"], "192.168.1.40")

    def test_the_sweep_finds_a_pi_the_table_had_not_seen_yet(self) -> None:
        tables = iter(["", WINDOWS_ARP])       # empty before the sweep, filled after
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache,
            probe=_probe({"192.168.1.40", "192.168.1.77"}),
            neighbours=lambda: next(tables), subnets=lambda: SUBNET, allow_sweep=True,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.host, "192.168.1.40")
        self.assertEqual(found.source, "local-network-search-raspberry-pi")

    def test_another_machine_running_ssh_is_never_taken_for_the_pi(self) -> None:
        # 192.168.1.77 answers on port 22 but its hardware is not a Pi's.
        tables = iter(["", WINDOWS_ARP.replace("dc-a6-32", "00-11-22")])
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache,
            probe=_probe({"192.168.1.77", "192.168.1.40"}),
            neighbours=lambda: next(tables), subnets=lambda: SUBNET, allow_sweep=True,
        )
        self.assertIsNone(found)

    def test_a_pi_that_does_not_answer_ssh_is_not_returned(self) -> None:
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache, probe=_probe(set()),
            neighbours=lambda: WINDOWS_ARP, subnets=lambda: SUBNET, allow_sweep=False,
        )
        self.assertIsNone(found)

    def test_the_configured_name_still_wins_when_it_works(self) -> None:
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache,
            probe=_probe({"locallife.local", "192.168.1.40"}),
            neighbours=lambda: WINDOWS_ARP, subnets=lambda: SUBNET,
        )
        self.assertEqual(found.host, "locallife.local")

    def test_the_sweep_can_be_switched_off(self) -> None:
        swept: list[str] = []

        def probe(host, port, timeout):
            swept.append(host)
            return False

        with mock.patch.dict(os.environ, {DISCOVERY_ENV: "0"}):
            resolve_pi("locallife@locallife.local", cache_path=self.cache, probe=probe,
                       neighbours=lambda: "", subnets=lambda: SUBNET)
        # Only the named candidates were tried; no address in the subnet was.
        self.assertFalse([host for host in swept if host.startswith("192.168.1.")])

    def test_a_sweep_is_not_repeated_on_every_status_refresh(self) -> None:
        calls: list[str] = []

        def probe(host, port, timeout):
            calls.append(host)
            return False

        for _ in range(3):
            resolve_pi("locallife@locallife.local", cache_path=self.cache, probe=probe,
                       neighbours=lambda: "", subnets=lambda: SUBNET, allow_sweep=True)
        swept = [host for host in calls if host.startswith("192.168.1.")]
        self.assertEqual(len(swept), 254)

    def test_discovery_can_be_turned_off_entirely_for_callers_that_need_it(self) -> None:
        found = resolve_pi(
            "locallife@locallife.local", cache_path=self.cache,
            probe=_probe({"192.168.1.40"}), discover=False,
            neighbours=lambda: WINDOWS_ARP, subnets=lambda: SUBNET,
        )
        self.assertIsNone(found)


class LocalSubnetTests(unittest.TestCase):
    def test_only_private_slash_24s_around_own_addresses(self) -> None:
        with mock.patch("socket.gethostbyname_ex",
                        return_value=("pc", [], ["192.168.1.23", "169.254.3.4", "8.8.4.4"])), \
             mock.patch("socket.socket") as fake:
            fake.return_value.__enter__.return_value.getsockname.return_value = ("192.168.137.1", 0)
            subnets = pi_discovery.local_subnets()
        self.assertEqual(
            sorted(str(net) for net in subnets), ["192.168.1.0/24", "192.168.137.0/24"],
        )


class MessageTests(unittest.TestCase):
    def test_the_message_says_the_network_was_searched_and_what_that_means(self) -> None:
        with mock.patch.dict(os.environ, {DISCOVERY_ENV: "1"}):
            message = unreachable_message("locallife@locallife.local")
        self.assertIn("searched for a Raspberry Pi by its hardware address", message)
        self.assertIn("same network", message)
        for remedy in ("hostname -I", "LOCALLIFE_PI_IP", "hosts", "Bonjour", "-PiHost"):
            with self.subTest(remedy=remedy):
                self.assertIn(remedy, message)

    def test_a_disabled_search_is_stated(self) -> None:
        with mock.patch.dict(os.environ, {DISCOVERY_ENV: "0"}):
            self.assertIn(f"{DISCOVERY_ENV}=0", unreachable_message("locallife@locallife.local"))


if __name__ == "__main__":
    unittest.main()
