"""Local control service: the welcome page and the run-mode selector.

A browser page cannot start Python, cameras, PowerShell or an SSH tunnel by
itself, so the one-click launcher starts this small service first and opens the
welcome page against it. The page asks which way to run, posts that choice back
here, and this module starts the matching pipeline by invoking the *existing*
`Start-LocalLife-Demo.ps1` with a fixed argument list.

Security: this binds to 127.0.0.1 by default and the only thing the browser can
influence is one enum (`local`, `cloud`, `auto`). Nothing from the request ever
reaches a shell -- the command is assembled from configuration, the mode picks
between two constant flags, and it runs without a shell.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, request

from . import __version__
from .cloud_startup import (
    FAILURES,
    STAGES,
    CloudStartupMachine,
    is_fully_ready,
    readiness_checklist,
)
from .config import AppConfig
from .launcher_page import COMPARE_PAGE, WELCOME_PAGE
from .pi_discovery import default_cache_path, resolve_pi, unreachable_message

RUN_MODES = ("local", "cloud", "auto")

# capturing -> processing -> stable mirrors the measurement lifecycle; a launch
# has its own small one. "failed" and "cancelled" are terminal, and so is
# "running" -- a launch never sits in an unexplained pending state because
# `CLOUD_STARTUP_TIMEOUT_SECONDS` always resolves it one way or the other.
LAUNCH_PHASES = (
    "idle", "checking", "starting-cloud", "starting-local",
    "running", "failed", "cancelled",
)


@dataclass
class LaunchState:
    mode: str | None = None
    requested_mode: str | None = None
    phase: str = "idle"
    messages: list[str] = field(default_factory=list)
    error: str | None = None
    cloud_error: str | None = None
    fallback_used: bool = False
    started_at: float | None = None
    dashboard_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "phase": self.phase,
            "messages": list(self.messages),
            "error": self.error,
            "cloud_error": self.cloud_error,
            "fallback_used": self.fallback_used,
            "started_at": self.started_at,
            "elapsed_seconds": (
                None if self.started_at is None else round(time.time() - self.started_at, 1)
            ),
            "dashboard_url": self.dashboard_url,
        }


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _split_pi_host(pi_host: str) -> tuple[str, int]:
    host = pi_host.split("@", 1)[-1].strip()
    if ":" in host:
        name, _, port = host.rpartition(":")
        if port.isdigit():
            return name, int(port)
    return host, 22


METRIC_MODES = ("local", "cloud")


def fetch_backend_telemetry(port: int, window_s: float, timeout: float = 2.0) -> dict[str, Any]:
    """The running backend's /api/telemetry; raises OSError/ValueError when unavailable."""
    url = f"http://127.0.0.1:{port}/api/telemetry?window_s={window_s:g}"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed localhost URL
        return json.loads(response.read().decode("utf-8"))


def cost_estimate(started_at: float | None, now: float | None = None) -> dict[str, Any]:
    """A labelled estimate from an operator-supplied hourly rate, never a live bill."""
    rate = os.environ.get("LOCALLIFE_CLOUD_COST_PER_HOUR", "").strip()
    currency = os.environ.get("LOCALLIFE_COST_CURRENCY", "kr").strip() or "kr"
    note = ("Estimate only: hours since this cloud launch x the rate in LOCALLIFE_CLOUD_COST_PER_HOUR. "
            "Excludes the stopped-VM disk (gpu.py: about 7 kr/day), images, bucket and network egress; "
            "the GCP billing console is authoritative.")
    try:
        hourly = float(rate) if rate else None
    except ValueError:
        hourly = None
    if hourly is None or started_at is None:
        return {"estimate": None, "currency": currency, "rate_per_hour": hourly,
                "note": "N/A: set LOCALLIFE_CLOUD_COST_PER_HOUR to show an estimate. " + note}
    hours = max(0.0, ((now or time.time()) - started_at) / 3600.0)
    return {"estimate": round(hours * hourly, 2), "currency": currency, "rate_per_hour": hourly,
            "hours": round(hours, 3), "note": note}


class LaunchController:
    """Readiness probes and the local/cloud/auto start sequence.

    `runner` is injected so tests drive the whole decision path -- including the
    cloud timeout and the fallback -- without starting a real process.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        launcher_script: Path | None = None,
        runner: Callable[[list[str], float], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.config = config
        self.launcher_script = launcher_script
        self.state = LaunchState()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._runner = runner or self._run_launcher
        # Last telemetry seen from each mode's backend, so switching modes keeps
        # the other mode's numbers on screen (clearly marked as not live).
        self.metric_snapshots: dict[str, dict[str, Any]] = {}
        for mode in METRIC_MODES:
            try:
                self.metric_snapshots[mode] = json.loads(
                    (self._telemetry_dir() / f"last_{mode}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        # Structured cloud startup state, so the welcome page shows which stage
        # is running and where it stopped instead of an unexplained spinner.
        # Served from here, the local control service, so it stays visible
        # before the tunnel or the cloud dashboard exists.
        self.startup = CloudStartupMachine(timeouts={
            "starting_vm": config.vm_start_timeout_seconds,
            "verifying_ssh": config.ssh_verify_timeout_seconds,
            "starting_backend": config.backend_ready_timeout_seconds,
            "opening_tunnel": config.tunnel_ready_timeout_seconds,
            "connecting_pi": config.pi_connect_timeout_seconds,
            "waiting_for_frames": config.first_frame_timeout_seconds,
        })

    # ---------------------------------------------------------------- probes
    def readiness(self) -> dict[str, Any]:
        """What is actually reachable right now.

        Anything that cannot be determined from this machine is reported as
        None (rendered "unknown"), never as a confident yes or no. Camera state
        in particular belongs to the backend, which is not running yet.
        """
        launcher_script: str | None = None
        launcher_error: str | None = None
        try:
            launcher_script = str(self.resolve_launcher_script())
        except FileNotFoundError as exc:
            launcher_error = str(exc)
        # Resolve the Pi by reachability rather than by name: "locallife.local"
        # is an mDNS name and Windows cannot resolve it on a network without
        # multicast DNS, which showed up as "Not reachable" for a Pi that was
        # powered on and fine.
        pi_cache = self.config.results_dir / "pi-address.json"
        # Also the launcher script's cache: a Pi it found must not read as
        # "Not reachable" here.
        pi = resolve_pi(self.config.pi_host, cache_path=pi_cache,
                        also_read=[default_cache_path()])
        internet = _port_open("8.8.8.8", 53, timeout=1.5)
        gcloud = shutil.which("gcloud") is not None
        return {
            "version": __version__,
            "internet": internet,
            "gcloud_installed": gcloud,
            "cloud_configured": bool(
                self.config.gcp_project and self.config.cloud_vm_name
            ),
            "cloud_available": bool(internet and gcloud and self.config.gcp_project),
            "pi_host": self.config.pi_host or None,
            "pi_reachable": pi is not None,
            "pi_address": None if pi is None else pi.to_dict(),
            # Actionable, because "Not reachable" on its own sent the operator
            # to check the power supply when the name lookup was the problem.
            "pi_hint": None if pi is not None else unreachable_message(
                self.config.pi_host, pi_cache,
            ),
            # The cameras hang off the Pi and are only visible once the backend
            # is up; the welcome page shows "unknown" until then rather than
            # claiming a state it cannot see.
            "realsense": None,
            "logitech": None,
            "dashboard_up": _port_open("127.0.0.1", self.config.port, timeout=0.6),
            "allow_local_fallback": self.config.allow_local_fallback,
            "default_run_mode": self.config.default_run_mode,
            "cloud_startup_timeout_seconds": self.config.cloud_startup_timeout_seconds,
            # Shown in the diagnostics block: when a start fails on the script
            # path, the page should say which file it found rather than making
            # the operator guess.
            "launcher_script": launcher_script,
            "launcher_script_error": launcher_error,
            "working_directory": str(Path.cwd()),
            # Live cloud facts for the status page. Anything the control service
            # cannot see from this machine stays None ("Unknown"), never a
            # hopeful guess -- an IP is not an identity, and a running VM is not
            # a working demonstration.
            "cloud": {
                "vm_name": self.config.cloud_vm_name or None,
                "preferred_zone": self.config.cloud_zone or None,
                "project": self.config.gcp_project or None,
                "gcp_authenticated": gcloud,
                "timeouts": {
                    "vm_start_timeout_seconds": self.config.vm_start_timeout_seconds,
                    "ssh_verify_timeout_seconds": self.config.ssh_verify_timeout_seconds,
                    "backend_ready_timeout_seconds": self.config.backend_ready_timeout_seconds,
                    "tunnel_ready_timeout_seconds": self.config.tunnel_ready_timeout_seconds,
                    "pi_connect_timeout_seconds": self.config.pi_connect_timeout_seconds,
                    "first_frame_timeout_seconds": self.config.first_frame_timeout_seconds,
                },
            },
        }

    # --------------------------------------------------------------- running
    def _run_launcher(self, command: list[str], timeout: float) -> subprocess.CompletedProcess:
        # No shell, fixed argv: nothing from the browser is interpolated here.
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
        )

    def resolve_launcher_script(self) -> Path:
        """Locate Start-LocalLife-Demo.ps1 relative to this package.

        It sits at the repository root, one level above the Python package,
        while the service is normally started with the working directory set to
        the package folder -- so resolving it from the current directory looked
        for it one level too deep and failed with "the argument ... does not
        exist". Anchoring to __file__ makes it independent of where the process
        was launched from.
        """
        if self.launcher_script is not None:
            if self.launcher_script.is_file():
                return self.launcher_script
            raise FileNotFoundError(
                f"Configured launcher script does not exist: {self.launcher_script}"
            )
        package_root = Path(__file__).resolve().parent.parent
        candidates = (
            package_root.parent / "Start-LocalLife-Demo.ps1",  # repository root
            package_root / "Start-LocalLife-Demo.ps1",
            package_root.parent / "app" / "Start-LocalLife-Demo.ps1",
            Path.cwd() / "Start-LocalLife-Demo.ps1",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        searched = "\n  ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            "Could not find Start-LocalLife-Demo.ps1. Looked in:\n  " + searched
        )

    def _command(self, mode: str) -> list[str]:
        script = self.resolve_launcher_script()
        executable = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return [
            executable, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script),
            "-Role", "Launcher",
            "-Mode", "Cloud" if mode == "cloud" else "Local",
        ]

    def _note(self, message: str) -> None:
        self.state.messages.append(message)

    def _attempt(self, mode: str, timeout: float | None) -> tuple[bool, str | None]:
        self.state.phase = "starting-cloud" if mode == "cloud" else "starting-local"
        self._note(f"Starting {mode} pipeline…")
        timeout = timeout if timeout and timeout > 0 else None   # 0 = wait until it opens
        try:
            completed = self._runner(self._command(mode), timeout)
        except FileNotFoundError as exc:
            return False, str(exc)
        except subprocess.TimeoutExpired:
            return False, f"{mode} startup exceeded {timeout:.0f}s (configured limit)"
        except OSError as exc:
            return False, f"{mode} startup could not run: {exc}"
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            return False, f"{mode} startup failed: {detail[-1] if detail else 'unknown error'}"
        return True, None

    def start_in_background(self, mode: str) -> dict[str, Any]:
        """Start without holding the HTTP request open.

        The launcher script only returns once the app is healthy, the Pi is
        streaming and the dashboard is open -- often longer than a browser or
        a fixed request budget waits. The page polls /api/launcher/status for
        progress instead.
        """
        if mode not in RUN_MODES:
            raise ValueError(f"Unsupported run mode: choose one of {', '.join(RUN_MODES)}")
        # start() holds the lock for the whole run, so a busy lock means a start
        # is in progress: refuse at once instead of queueing a second start.
        if not self._lock.acquire(blocking=False):
            raise ValueError("A startup is already in progress")
        try:
            busy = self.state.phase in {"checking", "starting-cloud", "starting-local"}
            if busy or (self._worker is not None and self._worker.is_alive()):
                raise ValueError("A startup is already in progress")
            self.state = LaunchState(requested_mode=mode, phase="checking", started_at=time.time())
            self._worker = threading.Thread(target=self._start_worker, args=(mode,),
                                            name="launcher-start", daemon=True)
            self._worker.start()
            return self.state.to_dict()
        finally:
            self._lock.release()

    def _start_worker(self, mode: str) -> None:
        try:
            self.start(mode, reserved=True)
        except Exception as exc:  # noqa: BLE001 - surfaced on the page, never swallowed
            self.state.phase = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"

    def start(self, mode: str, *, reserved: bool = False) -> dict[str, Any]:
        if mode not in RUN_MODES:
            raise ValueError(f"Unsupported run mode: choose one of {', '.join(RUN_MODES)}")
        with self._lock:
            if not reserved and self.state.phase in {"checking", "starting-cloud", "starting-local"}:
                raise ValueError("A startup is already in progress")
            self.state = LaunchState(
                requested_mode=mode, phase="checking", started_at=time.time(),
            )
            self._note(f"Selected mode: {mode}")

            if mode in {"cloud", "auto"}:
                # Start the structured feed before the first cloud step, so the
                # page has a stage to show from the very first poll.
                self.startup.begin()
                self.startup.enter("resolving_zone")
                self.startup.fact(
                    selected_mode=mode, vm_name=self.config.cloud_vm_name,
                    preferred_zone=self.config.cloud_zone or None,
                )
                ready = self.readiness()
                if not ready["cloud_available"]:
                    reason = (
                        "no internet connection" if not ready["internet"]
                        else "gcloud CLI not found" if not ready["gcloud_installed"]
                        else "cloud project is not configured"
                    )
                    self.state.cloud_error = reason
                    if mode == "cloud":
                        self.state.phase = "failed"
                        self.state.error = f"Cloud mode unavailable: {reason}"
                        return self.state.to_dict()
                    self._note(f"Cloud unavailable ({reason})")
                else:
                    ok, error = self._attempt(
                        "cloud", float(self.config.cloud_startup_timeout_seconds)
                    )
                    if ok:
                        self._finish("cloud")
                        return self.state.to_dict()
                    self.state.cloud_error = error
                    self._note(error or "Cloud startup failed")
                    if mode == "cloud":
                        self.state.phase = "failed"
                        self.state.error = error
                        return self.state.to_dict()

                # auto mode from here on
                if not self.config.allow_local_fallback:
                    self.state.phase = "failed"
                    self.state.error = (
                        f"Cloud unavailable ({self.state.cloud_error}) and local "
                        "fallback is disabled"
                    )
                    return self.state.to_dict()
                self.state.fallback_used = True
                self._note("Falling back to local processing")

            ok, error = self._attempt("local", float(self.config.local_startup_timeout_seconds))
            if not ok:
                self.state.phase = "failed"
                self.state.error = error
                return self.state.to_dict()
            self._finish("local")
            return self.state.to_dict()

    def _finish(self, mode: str) -> None:
        self.state.mode = mode
        self.state.phase = "running"
        self.state.dashboard_url = f"http://127.0.0.1:{self.config.port}/"
        self._note(f"{mode.capitalize()} pipeline running")

    # ------------------------------------------------------------- metrics
    def _telemetry_dir(self) -> Path:
        return Path(self.config.results_dir) / "telemetry"

    def metrics(self, window_s: float = 60.0,
                fetch: Callable[[int, float], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Live metrics from the running backend plus the last snapshot of the other mode.

        The mode of a snapshot is the one the *backend* reports for itself,
        never the one that was requested, so a local run can never be filed
        under cloud.
        """
        backend: dict[str, Any] = {"reachable": False, "processing_mode": None, "error": None,
                                   "port": self.config.port}
        try:
            summary = (fetch or fetch_backend_telemetry)(self.config.port, window_s)
            mode = str(summary.get("processing_mode") or "")
            backend.update(reachable=True, processing_mode=mode)
            if mode in METRIC_MODES:
                snapshot = {"fetched_at": time.time(), "summary": summary}
                self.metric_snapshots[mode] = snapshot
                try:
                    self._telemetry_dir().mkdir(parents=True, exist_ok=True)
                    (self._telemetry_dir() / f"last_{mode}.json").write_text(json.dumps(snapshot), encoding="utf-8")
                except OSError:
                    pass
        except (OSError, ValueError) as exc:
            backend["error"] = f"backend telemetry unavailable: {exc}"
        launched = self.state.mode
        mismatch = bool(backend["reachable"] and launched and backend["processing_mode"] != launched)
        modes = {}
        for mode in METRIC_MODES:
            snapshot = self.metric_snapshots.get(mode)
            modes[mode] = None if snapshot is None else {
                **snapshot, "live": backend["reachable"] and backend["processing_mode"] == mode,
                "age_s": round(time.time() - snapshot["fetched_at"], 1),
            }
        return {
            "window_s": window_s, "backend": backend, "modes": modes,
            "launched_mode": launched, "fallback_used": self.state.fallback_used,
            "mode_mismatch": mismatch,
            "benchmarks": self.latest_benchmarks(),
            "cost": cost_estimate(self.state.started_at if launched == "cloud" else None),
        }

    def latest_benchmarks(self) -> dict[str, Any]:
        """Newest replay-benchmark summary per mode from results/benchmark/*/run_summary.json."""
        newest: dict[str, Any] = {}
        root = Path(self.config.results_dir) / "benchmark"
        for path in sorted(root.glob("*/run_summary.json")) if root.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            mode = data.get("metadata", {}).get("server_processing_mode")
            if mode in METRIC_MODES:
                newest[mode] = {**data, "path": str(path)}
        return newest

    def gpu_script(self) -> Path | None:
        try:
            root = self.resolve_launcher_script().parent
        except FileNotFoundError:
            root = Path(__file__).resolve().parent.parent.parent
        for candidate in (root / "gpu.py", root.parent / "gpu.py", root / "app" / "gpu.py"):
            if candidate.is_file():
                return candidate
        return None

    def vm_status(self) -> dict[str, Any]:
        """`python gpu.py status` (read-only, no charges); the zone is parsed, never assumed."""
        script = self.gpu_script()
        if script is None:
            return {"ok": False, "error": "gpu.py not found next to the launcher"}
        try:
            completed = self._runner([sys.executable, str(script), "status"], 60.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"gpu.py status failed: {exc}"}
        output = (completed.stdout or "") + (completed.stderr or "")
        match = re.search(r"^\S+: (\w+) in ([a-z0-9-]+)", output, re.MULTILINE)
        return {"ok": completed.returncode == 0, "output": output.strip()[-2000:],
                "vm_state": match.group(1) if match else None, "zone": match.group(2) if match else None}

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self.state.phase == "running":
                raise ValueError("The system is already running; use the stop control instead")
            self.state.phase = "cancelled"
            self._note("Startup cancelled by the operator")
            return self.state.to_dict()


def create_launcher_app(
    config: AppConfig | None = None,
    controller: LaunchController | None = None,
) -> Flask:
    settings = config or AppConfig.from_env()
    control = controller or LaunchController(settings)
    app = Flask(__name__)
    app.config["LAUNCH_CONTROLLER"] = control

    @app.get("/")
    def welcome() -> str:
        return WELCOME_PAGE

    @app.get("/compare")
    def compare_page() -> str:
        """The local-versus-cloud table on its own page."""
        return COMPARE_PAGE

    @app.get("/api/launcher/status")
    def status() -> Any:
        return jsonify(
            readiness=control.readiness(),
            launch=control.state.to_dict(),
            # The structured startup feed the page polls: stage, elapsed time,
            # per-stage timings, live facts and the failure's next steps. The
            # page never scrapes console text for any of this.
            startup=control.startup.state.to_dict(),
            checklist=readiness_checklist(control.startup.state.facts),
            fully_ready=is_fully_ready(control.startup.state.facts),
        )

    @app.get("/api/launcher/stages")
    def stages() -> Any:
        """The vocabulary the page renders: every stage and failure it can show."""
        return jsonify(stages=list(STAGES), failures=list(FAILURES))

    @app.post("/api/launcher/startup-event")
    def startup_event() -> Any:
        """Stage updates pushed by the PowerShell launcher.

        The launcher is a separate process, so it reports progress here rather
        than the page trying to read its console. Strictly validated: `stage`
        and `failure` must be members of the known enums, and nothing from this
        request ever reaches a shell.
        """
        payload = request.get_json(silent=True) or {}
        stage = payload.get("stage")
        failure = payload.get("failure")
        facts = payload.get("facts")
        if facts is not None:
            if not isinstance(facts, dict):
                return jsonify(error="facts must be an object"), 400
            control.startup.fact(**{str(key): value for key, value in facts.items()})
        if failure is not None:
            if failure not in FAILURES:
                return jsonify(error=f"Unknown failure: {failure}"), 400
            return jsonify(ok=True, startup=control.startup.fail(
                failure, str(payload.get("reason") or "")
            ))
        if stage is not None:
            if stage not in STAGES:
                return jsonify(error=f"Unknown stage: {stage}"), 400
            control.startup.enter(stage, detail=payload.get("detail"))
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            control.startup.note(message.strip())
        return jsonify(ok=True, startup=control.startup.state.to_dict())

    @app.post("/api/launcher/start")
    def start() -> Any:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
        if not isinstance(mode, str):
            return jsonify(error="Provide mode as a string"), 400
        if mode == "cloud" and not payload.get("confirm_billing"):
            return jsonify(
                error="Starting a cloud GPU VM can incur charges; confirm to continue",
                needs_confirmation=True,
            ), 400
        try:
            return jsonify(ok=True, launch=control.start_in_background(mode))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/launcher/metrics")
    def metrics() -> Any:
        try:
            window = min(3600.0, max(5.0, float(request.args.get("window_s", 60))))
        except ValueError:
            window = 60.0
        return jsonify(control.metrics(window))

    @app.post("/api/launcher/vm-status")
    def vm_status() -> Any:
        return jsonify(control.vm_status())

    @app.post("/api/launcher/cancel")
    def cancel() -> Any:
        try:
            return jsonify(ok=True, launch=control.cancel())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    return app


def main() -> None:  # pragma: no cover - process entry point
    import sys as _sys
    import threading as _threading
    import webbrowser

    settings = AppConfig.from_env()
    app = create_launcher_app(settings)
    page = "compare" if "--compare" in _sys.argv[1:] else ""
    url = f"http://127.0.0.1:{settings.local_api_port}/{page}"
    if _port_open("127.0.0.1", settings.local_api_port, timeout=0.5):
        # The control service is already running (e.g. from START_LOCAL_LIFE.cmd):
        # just open the page in the browser instead of failing to bind the port.
        print(f"Local Life control service already running: {url}")
        webbrowser.open(url)
        return
    print(f"Local Life control service: {url}")
    _threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    # 127.0.0.1 only: these endpoints start processes and must never be
    # reachable from the network.
    app.run(host="127.0.0.1", port=settings.local_api_port, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main()
