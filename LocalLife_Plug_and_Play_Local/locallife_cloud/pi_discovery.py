"""Find the Raspberry Pi when its mDNS name will not resolve.

`locallife.local` is an mDNS name. Windows resolves those only when something
is answering multicast DNS -- Bonjour, or the built-in resolver on a network
that permits multicast. On a university or hotel network, or with client
isolation on, the lookup fails and ssh stops at

    ssh: Could not resolve hostname locallife.local: No such host is known

while the Pi is powered on, on the same network, and perfectly reachable by
address. The name is the only thing broken, so this tries the other ways of
naming the same machine before giving up, and remembers what worked.

Nothing here scans the network or guesses at neighbours: every candidate is a
name or address the operator has already configured, or one this launcher
itself recorded from a previous successful connection.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CACHE_NAME = "pi-address.json"
SSH_PORT = 22


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


def candidates(pi_host: str, cache_path: Path | None = None) -> list[tuple[str, str]]:
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
    if cache_path is not None:
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
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
) -> PiAddress | None:
    """The first candidate that actually answers on the SSH port, or None.

    Reachability, not name resolution, is the test: a name that resolves to a
    stale address is worse than no name at all, because ssh then hangs on it.
    """
    reach = probe or _reachable
    user, _, port = split_pi_host(pi_host)
    for host, source in candidates(pi_host, cache_path):
        if reach(host, port, timeout):
            address = PiAddress(host=host, port=port, source=source, user=user)
            if cache_path is not None:
                remember(address, Path(cache_path))
            return address
    return None


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
    return (
        f"The Raspberry Pi could not be reached. Tried: {tried}.\n"
        "The Pi being powered on is not enough -- this is a name-resolution "
        "problem, not a power one: Windows can only resolve a .local name when "
        "mDNS is available on the network.\n"
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
