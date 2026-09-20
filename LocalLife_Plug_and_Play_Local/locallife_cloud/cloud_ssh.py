"""Verified SSH identity for the cloud GPU VM.

The problem this replaces
------------------------
`gcloud compute ssh` shells out to PuTTY on Windows. When `gpu.py` moves the
VM to another zone it deletes and recreates it, so the host key legitimately
changes and PuTTY stops with

    WARNING - POTENTIAL SECURITY BREACH!
    Update cached key? (y/n, Return cancels connection, i for more info)

waiting for a keystroke in a window nobody is watching. That is what "the cloud
is stuck" actually was.

Answering that prompt automatically -- piping "y", or blanket-disabling strict
host-key checking -- would turn a real security control into a rubber stamp:
the launcher would accept *any* key any host presented, which is precisely the
man-in-the-middle case the prompt exists to catch.

What this does instead
----------------------
Google Compute Engine's guest agent publishes each instance's own SSH host keys
as **guest attributes**, readable over the authenticated GCP API. That is a
trusted channel independent of the SSH connection itself, so it answers the
question the prompt is asking -- "is this really that machine?" -- without a
human and without trusting the key being presented.

So: resolve the instance, fetch its published host keys, pin exactly those into
a dedicated `known_hosts` file, and connect with OpenSSH under
`StrictHostKeyChecking=yes`. A key that does not match fails the connection
closed, with a clear security error. Nothing is auto-accepted, no global setting
is relaxed, and no unrelated host's cached key is touched.

The pinned file is per-session and holds only this VM, so a recreated VM's new
key replaces only its own entry.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

LOGGER = logging.getLogger(__name__)

# Explicit statuses, so the welcome page can show where identity checking got to
# rather than a spinner. These strings are the contract with the UI.
STATUS_RESOLVING = "Resolving VM identity"
STATUS_VERIFYING = "Verifying SSH host key"
STATUS_VERIFIED = "SSH identity verified"
STATUS_MISMATCH = "SSH host key mismatch"
STATUS_TUNNEL_CONNECTED = "Tunnel connected"
STATUS_TUNNEL_FAILED = "Tunnel failed"

# Key types the GCE guest agent publishes. Ordered by preference: ed25519 first,
# since it is the strongest and the smallest.
PUBLISHED_KEY_TYPES = ("ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa")

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


class HostKeyVerificationError(RuntimeError):
    """Identity could not be established. Never downgraded to a warning."""


@dataclass
class HostKey:
    key_type: str
    key_base64: str

    @property
    def fingerprint(self) -> str:
        """OpenSSH's own SHA256 fingerprint form, for display next to a prompt.

        Same string `ssh-keygen -lf` prints, so an operator can compare it with
        what the GCP console shows without converting anything.
        """
        try:
            raw = base64.b64decode(self.key_base64, validate=True)
        except (ValueError, TypeError):
            return "SHA256:<unreadable>"
        digest = hashlib.sha256(raw).digest()
        return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")

    def known_hosts_line(self, host: str) -> str:
        return f"{host} {self.key_type} {self.key_base64}"


@dataclass
class InstanceIdentity:
    """Who the VM actually is, from the authenticated GCP API.

    `instance_id` is the identity that matters: name, zone and IP are all
    reusable, so none of them on its own proves the machine is the same one.
    """

    name: str
    zone: str
    project: str
    instance_id: str = ""
    external_ip: str = ""
    internal_ip: str = ""

    @property
    def host_aliases(self) -> list[str]:
        """Every name a known_hosts entry might be matched against."""
        return [item for item in (self.external_ip, self.internal_ip, self.name) if item]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vm_name": self.name,
            "zone": self.zone,
            "project": self.project,
            # Logged deliberately: it distinguishes a recreated VM from a
            # restarted one. It is not a credential.
            "instance_id": self.instance_id,
            "external_ip": self.external_ip,
            "internal_ip": self.internal_ip,
        }


@dataclass
class VerificationResult:
    status: str
    identity: InstanceIdentity | None = None
    keys: list[HostKey] = field(default_factory=list)
    known_hosts_path: Path | None = None
    error: str | None = None
    # Which trusted channel the keys came from, so the log says how identity
    # was established rather than just asserting that it was.
    source: str | None = None
    # True when the pinned file already held exactly these keys, i.e. nothing
    # about the VM's identity changed since the last run.
    unchanged: bool = False

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verified": self.verified,
            "identity": None if self.identity is None else self.identity.to_dict(),
            # Fingerprints only -- a public host key is not secret, but there is
            # no reason to spray full key material through logs.
            "fingerprints": [
                {"key_type": key.key_type, "fingerprint": key.fingerprint}
                for key in self.keys
            ],
            "known_hosts": None if self.known_hosts_path is None else str(self.known_hosts_path),
            "source": self.source,
            "unchanged": self.unchanged,
            "error": self.error,
        }


def parse_serial_host_keys(output: str) -> list[HostKey]:
    """Pull host keys out of a VM's serial console text.

    Compute Engine's guest agent prints them at boot between
    "-----BEGIN SSH HOST KEY KEYS-----" and its END marker, one per line as
    "<type> <base64> <comment>". Lines are matched on shape rather than on the
    markers alone, because the console text is interleaved with every other
    boot message and the markers are occasionally split across reads.

    Fingerprint blocks are deliberately ignored: a fingerprint cannot be pinned
    into known_hosts, only compared, and the full key is what is needed here.
    """
    keys: list[HostKey] = []
    in_fingerprint_block = False
    for line in output.splitlines():
        stripped = line.strip()
        if "SSH HOST KEY FINGERPRINTS" in stripped:
            in_fingerprint_block = "BEGIN" in stripped
            continue
        if "SSH HOST KEY KEYS" in stripped:
            in_fingerprint_block = False
            continue
        if in_fingerprint_block:
            continue
        parts = stripped.split()
        if len(parts) < 2 or parts[0] not in PUBLISHED_KEY_TYPES:
            continue
        candidate = parts[1]
        # Only accept something that really is a key blob: the console also
        # carries prose mentioning these type names.
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except (ValueError, TypeError):
            continue
        if len(decoded) < 32:
            continue
        keys.append(HostKey(key_type=parts[0], key_base64=candidate))
    return keys


def _run(command: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a command, turning every launch failure into an ordinary result.

    Two Windows-specific traps, both of which crashed a real run:

    * gcloud on Windows is `gcloud.cmd`, and subprocess without a shell resolves
      only `.exe` -- so a bare "gcloud" raises FileNotFoundError before the
      command ever starts. `shutil.which` applies PATHEXT and finds it.
    * a raised OSError escaped all the way out of the CLI as a traceback, which
      told the operator nothing. Every failure here becomes a non-zero result
      the caller can report properly.
    """
    arguments = list(command)
    resolved = shutil.which(arguments[0])
    if resolved:
        arguments[0] = resolved
    try:
        return subprocess.run(
            arguments, capture_output=True, text=True, check=False, timeout=120,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            arguments, 1, "",
            f"{command[0]} was not found on PATH. Install the Google Cloud CLI, "
            "or open a new terminal so PATH is picked up.",
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(arguments, 1, "", f"{command[0]} timed out")
    except OSError as exc:
        return subprocess.CompletedProcess(arguments, 1, "", f"{command[0]} could not run: {exc}")


class CloudSshVerifier:
    """Resolve the VM, pin its published host keys, build the ssh arguments.

    `runner` is injected so the whole decision path -- including a genuine key
    mismatch -- is testable without a GCP project.
    """

    def __init__(
        self,
        vm_name: str,
        zone: str,
        project: str,
        *,
        gcloud: str = "gcloud",
        runner: Runner | None = None,
    ) -> None:
        self.vm_name = vm_name
        self.zone = zone
        self.project = project
        self.gcloud = gcloud
        self._run = runner or _run

    # ------------------------------------------------------------- identity
    def resolve_identity(self) -> InstanceIdentity:
        completed = self._run([
            self.gcloud, "compute", "instances", "describe", self.vm_name,
            f"--zone={self.zone}", f"--project={self.project}", "--format=json",
        ])
        if completed.returncode != 0:
            raise HostKeyVerificationError(
                f"Could not resolve {self.vm_name} in {self.zone} "
                f"(project {self.project}): {(completed.stderr or '').strip()}"
            )
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise HostKeyVerificationError(
                f"Unreadable instance description for {self.vm_name}: {exc}"
            ) from exc
        interfaces = payload.get("networkInterfaces") or [{}]
        first = interfaces[0] if interfaces else {}
        access = (first.get("accessConfigs") or [{}])[0] if first else {}
        return InstanceIdentity(
            name=str(payload.get("name") or self.vm_name),
            zone=str(payload.get("zone", self.zone)).rsplit("/", 1)[-1],
            project=self.project,
            instance_id=str(payload.get("id") or ""),
            external_ip=str(access.get("natIP") or ""),
            internal_ip=str(first.get("networkIP") or ""),
        )

    # ------------------------------------------------------------ host keys
    def guest_attribute_host_keys(self) -> list[HostKey]:
        """Host keys from GCP guest attributes.

        The tidiest source, but it only exists when the instance carries
        `enable-guest-attributes=TRUE` -- which is NOT the default on Compute
        Engine, so on most VMs this legitimately returns nothing and the serial
        console below is the real source.
        """
        completed = self._run([
            self.gcloud, "compute", "instances", "get-guest-attributes", self.vm_name,
            f"--zone={self.zone}", f"--project={self.project}",
            "--query-path=hostkeys/", "--format=json",
        ])
        if completed.returncode != 0:
            return []
        try:
            entries = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            return []
        keys: list[HostKey] = []
        for entry in entries if isinstance(entries, list) else []:
            key_type = str(entry.get("key") or "").strip()
            value = str(entry.get("value") or "").strip()
            if key_type in PUBLISHED_KEY_TYPES and value:
                keys.append(HostKey(key_type=key_type, key_base64=value))
        return keys

    def serial_console_host_keys(self) -> list[HostKey]:
        """Host keys as the VM printed them to its own serial console at boot.

        This is the documented way to verify a Compute Engine host key, and it
        needs no metadata flag. It is trusted for the same reason guest
        attributes are: the text is fetched over the authenticated GCP API, not
        over the SSH connection being checked, so a machine-in-the-middle on the
        SSH path cannot influence it.
        """
        completed = self._run([
            self.gcloud, "compute", "instances", "get-serial-port-output", self.vm_name,
            f"--zone={self.zone}", f"--project={self.project}", "--port=1",
        ])
        if completed.returncode != 0:
            return []
        return parse_serial_host_keys(completed.stdout or "")

    def published_host_keys(self) -> tuple[list[HostKey], str]:
        """The VM's own host keys, and which trusted channel they came from."""
        keys = self.guest_attribute_host_keys()
        source = "guest-attributes"
        if not keys:
            keys = self.serial_console_host_keys()
            source = "serial-console"
        if not keys:
            raise HostKeyVerificationError(
                f"Google Cloud published no SSH host key for {self.vm_name}: guest "
                "attributes are empty (they are off unless "
                "enable-guest-attributes=TRUE) and the serial console shows no host "
                "key block. A VM that has just booted may not have printed them "
                "yet -- retry in a moment. If it persists, run `gcloud compute "
                f"instances get-serial-port-output {self.vm_name} --zone={self.zone} "
                "--port=1` and look for BEGIN SSH HOST KEY KEYS."
            )
        # De-duplicate by key type, strongest first: the serial console can
        # carry several boots' worth of output, and only the newest matters.
        best: dict[str, HostKey] = {}
        for key in keys:
            best[key.key_type] = key
        ordered = sorted(best.values(), key=lambda item: PUBLISHED_KEY_TYPES.index(item.key_type))
        return ordered, source

    # ---------------------------------------------------------------- pinning
    def pin(self, known_hosts: Path) -> VerificationResult:
        """Resolve, fetch and pin. The single call the launcher makes."""
        try:
            identity = self.resolve_identity()
        except HostKeyVerificationError as exc:
            return VerificationResult(status=STATUS_RESOLVING, error=str(exc))
        try:
            keys, source = self.published_host_keys()
        except HostKeyVerificationError as exc:
            return VerificationResult(
                status=STATUS_VERIFYING, identity=identity, error=str(exc),
            )
        lines = sorted(
            key.known_hosts_line(host)
            for key in keys
            for host in identity.host_aliases
        )
        content = "\n".join(lines) + "\n"
        previous = ""
        if known_hosts.is_file():
            try:
                previous = known_hosts.read_text(encoding="utf-8")
            except OSError:
                previous = ""
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        # Replace wholesale rather than append: this file holds only this VM, so
        # a recreated instance's new key supersedes its own stale entry and
        # nothing else. No other host's trust is affected.
        known_hosts.write_text(content, encoding="utf-8")
        return VerificationResult(
            status=STATUS_VERIFIED, identity=identity, keys=keys,
            known_hosts_path=known_hosts, source=source,
            unchanged=bool(previous) and previous == content,
        )

    # ------------------------------------------------------------- ssh flags
    @staticmethod
    def ssh_options(known_hosts: Path, identity_file: Path | None = None) -> list[str]:
        """OpenSSH arguments that verify without ever prompting.

        `StrictHostKeyChecking=yes` with a pre-pinned file is the whole point:
        a matching key connects silently, and a mismatched or unknown key is
        refused outright instead of asking a question no one will answer. It is
        set per connection, never in a global config.
        """
        options = [
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "StrictHostKeyChecking=yes",
            # Do not fall back to the user's own known_hosts: this connection is
            # verified against GCP's published keys or not at all.
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20",
        ]
        if identity_file is not None:
            options += ["-i", str(identity_file), "-o", "IdentitiesOnly=yes"]
        return options


def verify_cloud_ssh(
    vm_name: str,
    zone: str,
    project: str,
    known_hosts: Path,
    *,
    gcloud: str = "gcloud",
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Entry point for the launcher: pin the VM's identity, return a status."""
    verifier = CloudSshVerifier(vm_name, zone, project, gcloud=gcloud, runner=runner)
    result = verifier.pin(Path(known_hosts))
    if result.verified:
        LOGGER.info(
            "SSH identity verified for %s (%s) in %s: %s",
            vm_name, result.identity.instance_id if result.identity else "?", zone,
            ", ".join(key.fingerprint for key in result.keys),
        )
    else:
        LOGGER.error("SSH identity NOT verified for %s: %s", vm_name, result.error)
    return result.to_dict()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Called by Start-LocalLife-Demo.ps1; prints one JSON object on stdout."""
    import argparse

    parser = argparse.ArgumentParser(description="Pin and verify the cloud VM's SSH host key.")
    parser.add_argument("--vm", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--gcloud", default="gcloud")
    arguments = parser.parse_args(argv)
    try:
        report = verify_cloud_ssh(
            arguments.vm, arguments.zone, arguments.project,
            Path(arguments.known_hosts), gcloud=arguments.gcloud,
        )
    except Exception as exc:  # noqa: BLE001
        # A bare traceback on stderr told a real operator nothing. The caller
        # parses one JSON object, so an unexpected failure has to arrive in that
        # shape too -- with the detail, not instead of it.
        import traceback

        report = VerificationResult(
            status=STATUS_VERIFYING,
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()
        report["traceback"] = traceback.format_exc().strip().splitlines()[-3:]
    print(json.dumps(report))
    return 0 if report["verified"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
