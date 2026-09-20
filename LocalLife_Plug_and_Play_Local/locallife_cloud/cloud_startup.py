"""Cloud startup as an explicit state machine with measured stage timings.

Two things went wrong in the field that this module exists to prevent.

The first was an endless spinner. Startup blocked on an interactive SSH prompt
with no timeout and no state, so the operator had a window that said "starting"
and nothing else -- for as long as they were willing to wait. Every stage here
has its own bounded timeout and every failure names the stage, the reason and
the elapsed time, with a retry and a run-locally option.

The second was a misdiagnosis. The whole delay was described as GPU allocation,
when the VM itself was ready in about 79 seconds and the rest of the time went
somewhere else entirely. So each stage is timed, the timings are reported, and
optimisation decisions come from them rather than from a guess.

"Ready" here deliberately means the *whole path* is up -- VM, verified SSH,
backend health, tunnel, Pi, and a first frame. A running VM with a loaded model
is not a working demonstration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# The happy path, in order. Each entry is a stage the operator can be told they
# are in, and each one is separately timed.
STAGES = (
    "idle",
    "resolving_zone",
    "starting_vm",
    "verifying_ssh",
    "checking_deployment",
    "starting_backend",
    "opening_tunnel",
    "connecting_pi",
    "waiting_for_frames",
    "ready",
)

# Failure states. Each names what actually went wrong, because "cloud failed" is
# not something an operator can act on.
FAILURES = (
    "gpu_unavailable",
    "vm_start_timeout",
    "ssh_host_key_mismatch",
    "ssh_authentication_failed",
    "deployment_failed",
    "backend_health_failed",
    "tunnel_failed",
    "pi_unreachable",
    "camera_stream_timeout",
    "cloud_disconnected",
)

# Which stage each failure belongs to, so the UI can show the failure against
# the step that produced it.
FAILURE_STAGE = {
    "gpu_unavailable": "starting_vm",
    "vm_start_timeout": "starting_vm",
    "ssh_host_key_mismatch": "verifying_ssh",
    "ssh_authentication_failed": "verifying_ssh",
    "deployment_failed": "checking_deployment",
    "backend_health_failed": "starting_backend",
    "tunnel_failed": "opening_tunnel",
    "pi_unreachable": "connecting_pi",
    "camera_stream_timeout": "waiting_for_frames",
    "cloud_disconnected": "ready",
}

# The duration keys written to the diagnostics log and shown in Research mode.
TIMING_KEYS = (
    "vm_resolution_seconds",
    "vm_start_seconds",
    "ssh_verification_seconds",
    "deployment_seconds",
    "model_load_seconds",
    "backend_ready_seconds",
    "tunnel_ready_seconds",
    "pi_connection_seconds",
    "camera_first_frame_seconds",
    "total_ready_seconds",
)

STAGE_TIMING_KEY = {
    "resolving_zone": "vm_resolution_seconds",
    "starting_vm": "vm_start_seconds",
    "verifying_ssh": "ssh_verification_seconds",
    "checking_deployment": "deployment_seconds",
    "starting_backend": "backend_ready_seconds",
    "opening_tunnel": "tunnel_ready_seconds",
    "connecting_pi": "pi_connection_seconds",
    "waiting_for_frames": "camera_first_frame_seconds",
}


@dataclass
class StageRecord:
    name: str
    started_at: float
    finished_at: float | None = None
    detail: str | None = None

    @property
    def seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "seconds": self.seconds,
            "complete": self.finished_at is not None,
            "detail": self.detail,
        }


@dataclass
class CloudStartupState:
    """Everything the welcome page needs, in one serialisable object."""

    stage: str = "idle"
    failure: str | None = None
    reason: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    history: list[StageRecord] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.stage == "ready" and self.failure is None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 2)

    def timings(self) -> dict[str, float | None]:
        durations: dict[str, float | None] = {key: None for key in TIMING_KEYS}
        for record in self.history:
            key = STAGE_TIMING_KEY.get(record.name)
            if key and record.finished_at is not None:
                durations[key] = record.seconds
        # model_load_seconds is reported by the remote backend rather than
        # measured from here; None until it says so.
        if "model_load_seconds" in self.facts:
            durations["model_load_seconds"] = self.facts["model_load_seconds"]
        if self.ready:
            durations["total_ready_seconds"] = self.elapsed_seconds
        return durations

    def to_dict(self) -> dict[str, Any]:
        current = self.history[-1] if self.history else None
        return {
            "stage": self.stage,
            "stage_index": STAGES.index(self.stage) if self.stage in STAGES else None,
            "stage_count": len(STAGES),
            "stage_elapsed_seconds": 0.0 if current is None else current.seconds,
            "ready": self.ready,
            "failure": self.failure,
            "reason": self.reason,
            "elapsed_seconds": self.elapsed_seconds,
            "timings": self.timings(),
            "history": [record.to_dict() for record in self.history],
            "facts": dict(self.facts),
            "messages": list(self.messages),
            # A failure is never a dead end: the UI renders these three.
            "actions": (
                []
                if self.failure is None
                else ["retry_cloud", "run_locally", "open_diagnostics"]
            ),
        }


class CloudStartupMachine:
    """Drives the stages, enforces per-stage timeouts, records the timings.

    Holds no subprocess logic of its own: the launcher supplies each stage's
    work as a callable, so this stays testable and the orchestration stays in
    one readable place.
    """

    def __init__(self, timeouts: dict[str, float] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.state = CloudStartupState()
        self.timeouts = dict(timeouts or {})
        self._clock = clock
        self._lock = threading.Lock()

    def begin(self) -> None:
        with self._lock:
            self.state = CloudStartupState(started_at=self._clock())

    def note(self, message: str) -> None:
        with self._lock:
            self.state.messages.append(message)

    def fact(self, **values: Any) -> None:
        """Record live facts the status page shows (zone, IP, fingerprint...)."""
        with self._lock:
            self.state.facts.update(values)

    def enter(self, stage: str, detail: str | None = None) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        with self._lock:
            if self.state.started_at is None:
                self.state.started_at = self._clock()
            if self.state.history and self.state.history[-1].finished_at is None:
                self.state.history[-1].finished_at = self._clock()
            self.state.stage = stage
            self.state.history.append(
                StageRecord(name=stage, started_at=self._clock(), detail=detail)
            )
            if stage == "ready":
                self.state.history[-1].finished_at = self._clock()
                self.state.finished_at = self._clock()

    def complete_stage(self, detail: str | None = None) -> None:
        with self._lock:
            if self.state.history and self.state.history[-1].finished_at is None:
                self.state.history[-1].finished_at = self._clock()
                if detail:
                    self.state.history[-1].detail = detail

    def fail(self, failure: str, reason: str) -> dict[str, Any]:
        if failure not in FAILURES:
            raise ValueError(f"Unknown failure: {failure}")
        with self._lock:
            if self.state.history and self.state.history[-1].finished_at is None:
                self.state.history[-1].finished_at = self._clock()
            self.state.failure = failure
            self.state.reason = reason
            self.state.stage = FAILURE_STAGE.get(failure, self.state.stage)
            self.state.finished_at = self._clock()
            self.state.messages.append(f"{failure}: {reason}")
            return self.state.to_dict()

    def timeout_for(self, stage: str, default: float) -> float:
        return float(self.timeouts.get(stage, default))

    def run_stage(
        self,
        stage: str,
        work: Callable[[], Any],
        *,
        failure: str,
        timeout: float,
        detail: str | None = None,
    ) -> tuple[bool, Any]:
        """Run one stage, bounded. Returns (ok, value-or-None).

        A stage that overruns its budget is a named failure with an elapsed
        time, never an indefinite wait.
        """
        self.enter(stage, detail=detail)
        started = self._clock()
        try:
            value = work()
        except TimeoutError as exc:
            self.fail(failure, f"timed out after {timeout:.0f}s: {exc}")
            return False, None
        except Exception as exc:  # noqa: BLE001 - every stage failure is reportable
            self.fail(failure, f"{type(exc).__name__}: {exc}")
            return False, None
        elapsed = self._clock() - started
        if elapsed > timeout:
            self.fail(
                failure,
                f"took {elapsed:.0f}s, over the {timeout:.0f}s budget for this stage",
            )
            return False, None
        self.complete_stage()
        return True, value


DEPLOYMENT_MARKER = ".locallife_deployment"


def deployment_version(project_root: Path) -> str:
    """A short hash of the project's Python sources.

    Two jobs. It stops startup re-uploading a project that has not changed --
    the expensive step when it does run. And it catches the opposite, quieter
    bug: the previous check asked only whether the directory existed, so a VM
    carrying an *old* copy of the code was treated as up to date and silently
    ran stale software, which is worse than a slow upload.

    Content-addressed rather than timestamp-based: a fresh checkout on the
    laptop must not look like a change to the VM.
    """
    import hashlib

    digest = hashlib.sha256()
    root = Path(project_root)
    for path in sorted(root.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "build", "dist"} for part in path.parts):
            continue
        try:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:16]


def readiness_checklist(facts: dict[str, Any]) -> dict[str, bool | None]:
    """Cloud mode is ready only when the whole path is up.

    A running VM with a loaded model is not a demonstration: the frames still
    have to arrive and the operator's browser still has to reach the dashboard.
    None means "not determined yet", never a hopeful True.
    """
    return {
        "vm_ready": facts.get("vm_ready"),
        "ssh_verified": facts.get("ssh_verified"),
        "backend_healthy": facts.get("backend_healthy"),
        "tunnel_active": facts.get("tunnel_active"),
        "pi_connected": facts.get("pi_connected"),
        "camera_heartbeat": facts.get("camera_heartbeat"),
        "dashboard_reachable": facts.get("dashboard_reachable"),
    }


def is_fully_ready(facts: dict[str, Any]) -> bool:
    return all(value is True for value in readiness_checklist(facts).values())
