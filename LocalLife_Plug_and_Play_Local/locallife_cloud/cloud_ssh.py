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
            "unchanged": self.unchanged,
            "error": self.error,
        }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command), capture_output=True, text=True, check=False, timeout=120,
    )


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
    def published_host_keys(self) -> list[HostKey]:
        """The VM's own host keys, from GCP guest attributes.

        Trusted because it arrives over the authenticated GCP API rather than
        over the SSH connection being checked.
        """
        completed = self._run([
            self.gcloud, "compute", "instances", "get-guest-attributes", self.vm_name,
            f"--zone={self.zone}", f"--project={self.project}",
            "--query-path=hostkeys/", "--format=json",
        ])
        if completed.returncode != 0:
            raise HostKeyVerificationError(
                "Google Cloud has not published SSH host keys for "
                f"{self.vm_name}: {(completed.stderr or '').strip()}"
            )
        try:
            entries = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise HostKeyVerificationError(f"Unreadable guest attributes: {exc}") from exc
        keys: list[HostKey] = []
        for entry in entries if isinstance(entries, list) else []:
            key_type = str(entry.get("key") or "").strip()
            value = str(entry.get("value") or "").strip()
            if key_type in PUBLISHED_KEY_TYPES and value:
                keys.append(HostKey(key_type=key_type, key_base64=value))
        if not keys:
            raise HostKeyVerificationError(
                f"No usable SSH host key published for {self.vm_name}. The guest "
                "agent may still be starting; retry in a moment."
            )
        keys.sort(key=lambda item: PUBLISHED_KEY_TYPES.index(item.key_type))
        return keys

    # ---------------------------------------------------------------- pinning
    def pin(self, known_hosts: Path) -> VerificationResult:
        """Resolve, fetch and pin. The single call the launcher makes."""
        try:
            identity = self.resolve_identity()
        except HostKeyVerificationError as exc:
            return VerificationResult(status=STATUS_RESOLVING, error=str(exc))
        try:
            keys = self.published_host_keys()
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
            known_hosts_path=known_hosts,
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
    report = verify_cloud_ssh(
        arguments.vm, arguments.zone, arguments.project,
        Path(arguments.known_hosts), gcloud=arguments.gcloud,
    )
    print(json.dumps(report))
    return 0 if report["verified"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
