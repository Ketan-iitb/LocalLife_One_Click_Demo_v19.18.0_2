"""Find the Raspberry Pi when its mDNS name will not resolve.

`locallife.local` is an mDNS name. Windows resolves those only when something
is answering multicast DNS -- Bonjour, or the built-in resolver on a network
that permits multicast. On a university or hotel network, or with client
isolation on, the lookup fails and ssh stops at

    ssh: Could not resolve hostname locallife.local: No such host is known

while the Pi is powered on, on the same network, and perfectly reachable by
address. The name is the only thing broken, so this tries the other ways of
naming the same machine before giving up, and remembers what worked.

Why that was not enough
-----------------------
The first version only ever tried names the operator had configured, plus the
address remembered from the last *successful* connection. On a network without
mDNS that is a closed loop: the name never resolves, so it never connects, so
nothing is ever remembered, so the next run fails the same way. "Not reachable,
every time" was guaranteed rather than bad luck.

So when every named candidate fails, the Pi is now found by what it *is*
rather than what it is called. Every Raspberry Pi's network adapter carries a
hardware (MAC) address from a small, published set of Raspberry Pi prefixes.

1. The computer's own neighbour table (`arp -a`) is read first. That sends
   nothing on the network; it lists machines this computer has already seen.
2. If no Pi is in it, the computer's own local /24 subnet is checked on the SSH
   port once, briefly, which also fills the neighbour table -- and then only a
   host whose hardware address is a Raspberry Pi's is accepted. Some other
   machine that happens to run SSH is never guessed at.

Set LOCALLIFE_PI_DISCOVERY=0 to turn step 2 off, for example on a managed
network where connection sweeps are not welcome. Step 1 stays on: it is
passive. Whatever is found is remembered, so the search happens once.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CACHE_NAME = "pi-address.json"
SSH_PORT = 22

# IEEE OUI prefixes assigned to Raspberry Pi Foundation / Raspberry Pi Ltd.
# Both the Ethernet and the Wi-Fi adapters of every model use one of these.
RASPBERRY_PI_OUIS = frozenset({
    "b8:27:eb", "dc:a6:32", "e4:5f:01", "28:cd:c1", "d8:3a:dd", "2c:cf:67",
})
DISCOVERY_ENV = "LOCALLIFE_PI_DISCOVERY"
SWEEP_TIMEOUT_S = 0.35
SWEEP_WORKERS = 64
MAX_SUBNETS = 4
# The welcome page refreshes its status; a sweep must not repeat every refresh.
SWEEP_COOLDOWN_S = 60.0
_last_sweep: dict[str, float] = {}
_sweep_lock = threading.Lock()

_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
_MAC = re.compile(r"\b([0-9a-fA-F]{1,2}(?:[:-][0-9a-fA-F]{1,2}){5})\b")


def default_cache_path() -> Path:
    """The one place both the launcher script and the welcome page remember the Pi.

    They used to keep separate caches, so a Pi the launcher had found could
    still read "Not reachable" on the welcome page.
    """
    if sys.platform.startswith("win") and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "LocalLifeDemo" / CACHE_NAME
    return Path.home() / ".cache" / "locallife" / CACHE_NAME


def normalise_mac(text: str) -> str:
    """"B8-27-EB-1-2-3" -> "b8:27:eb:01:02:03": the forms arp prints vary."""
    return ":".join(part.zfill(2) for part in re.split(r"[:-]", text.strip().lower()))


def parse_neighbours(text: str) -> list[tuple[str, str]]:
    """(ip, mac) pairs from `arp -a` (Windows or macOS) or `ip neigh` output."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        ip = _IPV4.search(line)
        mac = _MAC.search(line)
        if ip and mac:
            pairs.append((ip.group(1), normalise_mac(mac.group(1))))
    return pairs


def is_raspberry_pi_mac(mac: str) -> bool:
    return normalise_mac(mac)[:8] in RASPBERRY_PI_OUIS


def neighbour_table() -> str:
    """This computer's neighbour table. Passive: nothing is sent to read it."""
    commands = (["arp", "-a"],) if sys.platform.startswith("win") else (
        ["ip", "neigh"], ["arp", "-an"],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.stdout.strip():
            return result.stdout
    return ""


def raspberry_pi_neighbours(text: str) -> list[str]:
    """Addresses in the neighbour table whose hardware says Raspberry Pi."""
    found: list[str] = []
    for ip, mac in parse_neighbours(text):
        if is_raspberry_pi_mac(mac) and ip not in found:
            found.append(ip)
    return found


def local_subnets() -> list[ipaddress.IPv4Network]:
    """The /24 of each private IPv4 address this computer has.

    Only private ranges, never link-local (169.254/16 is far too large to
    check, and a Pi there shows up in the neighbour table anyway), and only a
    /24 around this computer's own address -- never a wider range.
    """
    addresses: set[str] = set()
    try:
        # A UDP "connect" picks the outbound interface without sending anything.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    subnets: list[ipaddress.IPv4Network] = []
    for text in sorted(addresses):
        try:
            address = ipaddress.IPv4Address(text)
        except ValueError:
            continue
        if not address.is_private or address.is_loopback or address.is_link_local:
            continue
        network = ipaddress.IPv4Network(f"{address}/24", strict=False)
        if network not in subnets:
            subnets.append(network)
    return subnets[:MAX_SUBNETS]


def discovery_enabled() -> bool:
    return os.environ.get(DISCOVERY_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def sweep(
    subnets: Iterable[ipaddress.IPv4Network], port: int,
    probe: Callable[[str, int, float], bool], *, timeout: float = SWEEP_TIMEOUT_S,
) -> list[str]:
    """Hosts in these subnets answering on `port`. Once per subnet per minute."""
    now = time.monotonic()
    hosts: list[str] = []
    with _sweep_lock:
        for network in subnets:
            key = f"{network}:{port}"
            if now - _last_sweep.get(key, -SWEEP_COOLDOWN_S * 2) < SWEEP_COOLDOWN_S:
                continue
            _last_sweep[key] = now
            hosts.extend(str(host) for host in network.hosts())
    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=SWEEP_WORKERS) as pool:
        answers = list(pool.map(lambda host: probe(host, port, timeout), hosts))
    return [host for host, answered in zip(hosts, answers) if answered]


def discover_on_network(
    port: int,
    probe: Callable[[str, int, float], bool],
    *,
    timeout: float,
    neighbours: Callable[[], str] = neighbour_table,
    subnets: Callable[[], list[ipaddress.IPv4Network]] = local_subnets,
    allow_sweep: bool | None = None,
) -> tuple[str, str] | None:
    """A Raspberry Pi on this network answering SSH, found by hardware address."""
    for host in raspberry_pi_neighbours(neighbours()):
        if probe(host, port, timeout):
            return host, "network-neighbour-raspberry-pi"
    if not (discovery_enabled() if allow_sweep is None else allow_sweep):
        return None
    answering = set(sweep(subnets(), port, probe))
    if not answering:
        return None
    # The sweep filled the neighbour table. Only a host whose hardware is a
    # Raspberry Pi's is accepted: another machine running SSH is not the Pi.
    for host in raspberry_pi_neighbours(neighbours()):
        if host in answering:
            return host, "local-network-search-raspberry-pi"
    return None


@dataclass
class PiAddress:
    host: str
    port: int = SSH_PORT
    source: str = "configured"
    user: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def to_dict(self) -> dict[str, object]:
        return {"host": self.host, "port": self.port, "source": self.source,
                "target": self.target}


def split_pi_host(pi_host: str) -> tuple[str, str, int]:
    """Split "user@host:port" into its parts, with sensible defaults."""
    user, _, rest = pi_host.rpartition("@")
    host = rest.strip()
    port = SSH_PORT
    if ":" in host:
        name, _, tail = host.rpartition(":")
        if tail.isdigit():
            host, port = name, int(tail)
    return user.strip(), host, port


def _reachable(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def candidates(
    pi_host: str, cache_path: Path | None = None, also_read: Iterable[Path] = (),
) -> list[tuple[str, str]]:
    """Every name worth trying, in the order worth trying it.

    The configured name first, because when mDNS works it is the right answer
    and survives the Pi changing address. Then the same name without `.local`,
    which unicast DNS, LLMNR and a hosts entry can all answer. Then an explicit
    override, then the address that worked last time.
    """
    _, host, _ = split_pi_host(pi_host)
    found: list[tuple[str, str]] = []

    def _add(value: str | None, source: str) -> None:
        value = (value or "").strip()
        if value and all(value != existing for existing, _ in found):
            found.append((value, source))

    _add(host, "configured")
    if host.lower().endswith(".local"):
        _add(host[: -len(".local")], "short-name")
    _add(os.environ.get("LOCALLIFE_PI_IP"), "LOCALLIFE_PI_IP")
    _add(os.environ.get("LOCALLIFE_PI_HOST_IP"), "LOCALLIFE_PI_HOST_IP")
    for path in ([cache_path] if cache_path is not None else []) + list(also_read):
        try:
            cached = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                _add(str(cached.get("host") or ""), "last-known-address")
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return found


def resolve_pi(
    pi_host: str,
    *,
    cache_path: Path | None = None,
    timeout: float = 1.5,
    probe: Callable[[str, int, float], bool] | None = None,
    also_read: Iterable[Path] = (),
    discover: bool = True,
    neighbours: Callable[[], str] | None = None,
    subnets: Callable[[], list[ipaddress.IPv4Network]] | None = None,
    allow_sweep: bool | None = None,
) -> PiAddress | None:
    """The first candidate that actually answers on the SSH port, or None.

    Reachability, not name resolution, is the test: a name that resolves to a
    stale address is worse than no name at all, because ssh then hangs on it.
    When no named candidate answers, the Pi is looked for on the local network
    by its hardware address -- see the module docstring.
    """
    reach = probe or _reachable
    user, _, port = split_pi_host(pi_host)
    extra = list(also_read)
    found: tuple[str, str] | None = None
    for host, source in candidates(pi_host, cache_path, extra):
        if reach(host, port, timeout):
            found = (host, source)
            break
    if found is None and discover:
        found = discover_on_network(
            port, reach, timeout=min(timeout, 1.0),
            neighbours=neighbours or neighbour_table,
            subnets=subnets or local_subnets,
            allow_sweep=allow_sweep,
        )
    if found is None:
        return None
    address = PiAddress(host=found[0], port=port, source=found[1], user=user)
    for path in ([cache_path] if cache_path is not None else []) + extra:
        remember(address, Path(path))
    return address


def remember(address: PiAddress, cache_path: Path) -> None:
    """Record what worked, so the next run tries it even if mDNS stays broken."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({
                "host": address.host, "port": address.port,
                "source": address.source, "recorded_at": time.time(),
            }),
            encoding="utf-8",
        )
    except OSError:
        # A cache that cannot be written is a lost optimisation, not a failure.
        pass


def unreachable_message(pi_host: str, cache_path: Path | None = None) -> str:
    """What to tell the operator: what was tried, and what to do next."""
    tried = ", ".join(host for host, _ in candidates(pi_host, cache_path))
    searched = (
        "This computer's local network was also searched for a Raspberry Pi by its "
        "hardware address, and none answered on the SSH port -- so the Pi is most "
        "likely not on the same network as this laptop (check it is on the same "
        "Wi-Fi, or connect it with an Ethernet cable), or SSH is not running on it.\n"
        if discovery_enabled() else
        f"The automatic local-network search is switched off ({DISCOVERY_ENV}=0).\n"
    )
    return (
        f"The Raspberry Pi could not be reached. Tried: {tried}.\n"
        "The Pi being powered on is not enough -- this is a name-resolution "
        "problem, not a power one: Windows can only resolve a .local name when "
        "mDNS is available on the network.\n"
        + searched +
        "Fix it in one of these ways:\n"
        "  1. Find the Pi's IP (on the Pi: `hostname -I`, or check your "
        "router's device list) and start again with -PiHost "
        "locallife@<that-ip>, or set LOCALLIFE_PI_IP=<that-ip>.\n"
        "  2. Add a line to C:\\Windows\\System32\\drivers\\etc\\hosts: "
        "`<that-ip>  locallife.local`\n"
        "  3. Install Bonjour (Apple Print Services) so Windows can resolve "
        ".local names at all.\n"
        "Once it connects once, the address is remembered for later runs."
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    """Called by Start-LocalLife-Demo.ps1; prints one JSON object on stdout.

    Never raises and never prints a bare traceback: the caller parses one JSON
    object, and a traceback on stderr tells the operator nothing (a real run
    showed exactly that -- "DEMONSTRATION ERROR: Traceback (most recent call
    last):" and no further detail).
    """
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Locate the Raspberry Pi.")
    parser.add_argument("--pi-host", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--timeout", type=float, default=1.5)
    arguments = parser.parse_args(argv)
    cache = Path(arguments.cache) if arguments.cache else None
    try:
        address = resolve_pi(arguments.pi_host, cache_path=cache, timeout=arguments.timeout)
        report = {
            "found": address is not None,
            "target": None if address is None else address.target,
            "host": None if address is None else address.host,
            "source": None if address is None else address.source,
            "hint": None if address is not None else unreachable_message(
                arguments.pi_host, cache,
            ),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "found": False, "target": None, "host": None, "source": None,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": unreachable_message(arguments.pi_host, cache),
            "traceback": traceback.format_exc().strip().splitlines()[-3:],
        }
    print(json.dumps(report))
    return 0 if report["found"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
