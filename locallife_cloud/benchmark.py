"""Local-versus-cloud benchmark metrics, and the export that backs a claim.

The point of this module is to make "is the cloud actually better?" an
answerable question rather than an impression. Three rules shape it.

*Same input.* A fair comparison replays one recorded clip through both modes.
Comparing two different live moments and calling the difference "cloud versus
local" measures the moments, not the modes, so `BenchmarkSession` carries the
id of the recorded input each side processed.

*No single ambiguous FPS.* Capture, submission, inference, completion and
display rates are all different numbers, and quoting one of them as "FPS" is
how a slow pipeline looks fast. They are reported separately.

*Never fabricate.* A metric that cannot be collected here -- GPU utilisation
with no GPU, Pi temperature with no Pi -- is reported as None and rendered
"Unavailable". An invented number is worse than a missing one in a thesis.
"""

from __future__ import annotations

import csv
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BENCHMARK_CSV_NAME = "benchmark.csv"

# The required export schema, in order.
BENCHMARK_COLUMNS = [
    "session_id", "frame_id", "event_id", "timestamp", "processing_mode",
    "capture_fps", "inference_fps", "display_fps",
    "upload_latency_ms", "inference_latency_ms", "return_latency_ms",
    "end_to_end_latency_ms", "queue_depth", "frames_dropped",
    "cpu_percent", "ram_mb", "gpu_utilization_percent", "gpu_memory_mb",
    "estimated_litres", "ground_truth_litres", "absolute_error_litres",
    "percentage_error", "colour_correct", "material_correct",
    "sorting_correct", "bag_count_correct",
]


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile: no interpolation, no invented intermediate.

    Same convention the height-map volume code uses, so a P95 here and a cell
    percentile there mean the same thing.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))
    return round(ordered[index], 3)


@dataclass
class FrameSample:
    """One frame's journey, timed at each hop."""

    frame_id: int
    processing_mode: str
    captured_at: float
    submitted_at: float | None = None
    inference_started_at: float | None = None
    inference_finished_at: float | None = None
    result_available_at: float | None = None
    dropped: bool = False
    stale: bool = False
    queue_depth: int = 0

    def _ms(self, start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 3)

    @property
    def upload_latency_ms(self) -> float | None:
        """Capture to the moment the far side has the frame.

        Only meaningful in cloud mode; locally the two timestamps are the same
        machine and the number is near zero by construction.
        """
        return self._ms(self.captured_at, self.submitted_at)

    @property
    def queue_latency_ms(self) -> float | None:
        return self._ms(self.submitted_at, self.inference_started_at)

    @property
    def inference_latency_ms(self) -> float | None:
        return self._ms(self.inference_started_at, self.inference_finished_at)

    @property
    def return_latency_ms(self) -> float | None:
        return self._ms(self.inference_finished_at, self.result_available_at)

    @property
    def end_to_end_latency_ms(self) -> float | None:
        return self._ms(self.captured_at, self.result_available_at)


@dataclass
class ResourceSample:
    """Host metrics. Every field optional: absent means Unavailable, not zero."""

    cpu_percent: float | None = None
    ram_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_mb: float | None = None
    pi_cpu_percent: float | None = None
    pi_ram_mb: float | None = None
    pi_temperature_c: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectOutcome:
    """One accepted test object, scored against ground truth where it exists."""

    event_id: str
    processing_mode: str
    estimated_litres: float | None = None
    ground_truth_litres: float | None = None
    colour_correct: bool | None = None
    material_correct: bool | None = None
    sorting_correct: bool | None = None
    bag_count_correct: bool | None = None
    confidence: float | None = None
    processing_seconds: float | None = None

    @property
    def absolute_error_litres(self) -> float | None:
        if self.estimated_litres is None or self.ground_truth_litres is None:
            return None
        return round(abs(self.estimated_litres - self.ground_truth_litres), 4)

    @property
    def percentage_error(self) -> float | None:
        error = self.absolute_error_litres
        if error is None or not self.ground_truth_litres:
            return None
        return round(error / abs(self.ground_truth_litres) * 100.0, 3)


class BenchmarkSession:
    """Collects samples for one mode over one recorded input.

    Thread-safe; the capture thread records frames while the dashboard reads
    the summary.
    """

    def __init__(
        self,
        session_id: str,
        processing_mode: str,
        *,
        recorded_input_id: str | None = None,
        window_seconds: float = 10.0,
    ) -> None:
        self.session_id = session_id
        self.processing_mode = processing_mode
        # What makes the comparison fair: both modes must name the same input.
        self.recorded_input_id = recorded_input_id
        self.window_seconds = window_seconds
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._frames: list[FrameSample] = []
        self._outcomes: list[ObjectOutcome] = []
        self._resources = ResourceSample()
        self._captured = 0
        self._submitted = 0
        self._displayed = 0

    # ------------------------------------------------------------ recording
    def record_capture(self, frame_id: int, at: float | None = None) -> FrameSample:
        sample = FrameSample(
            frame_id=frame_id, processing_mode=self.processing_mode,
            captured_at=at if at is not None else time.time(),
        )
        with self._lock:
            self._frames.append(sample)
            self._captured += 1
        return sample

    def record_submitted(self, sample: FrameSample, at: float | None = None, queue_depth: int = 0) -> None:
        sample.submitted_at = at if at is not None else time.time()
        sample.queue_depth = queue_depth
        with self._lock:
            self._submitted += 1

    def record_displayed(self) -> None:
        with self._lock:
            self._displayed += 1

    def record_outcome(self, outcome: ObjectOutcome) -> None:
        with self._lock:
            self._outcomes.append(outcome)

    def update_resources(self, **values: float | None) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self._resources, key):
                    setattr(self._resources, key, value)

    # -------------------------------------------------------------- reading
    def _recent(self) -> list[FrameSample]:
        cutoff = time.time() - self.window_seconds
        return [item for item in self._frames if item.captured_at >= cutoff]

    def throughput(self) -> dict[str, float | None]:
        """Five separate rates. Never collapsed into one "FPS"."""
        elapsed = max(1e-6, time.time() - self.started_at)
        with self._lock:
            frames = list(self._frames)
            captured, submitted, displayed = self._captured, self._submitted, self._displayed
        inferred = [item for item in frames if item.inference_finished_at is not None]
        completed = [item for item in frames if item.result_available_at is not None]
        return {
            "capture_fps": round(captured / elapsed, 2),
            "submitted_fps": round(submitted / elapsed, 2),
            "inference_fps": round(len(inferred) / elapsed, 2),
            "completed_fps": round(len(completed) / elapsed, 2),
            "display_fps": round(displayed / elapsed, 2),
        }

    def latency(self) -> dict[str, float | None]:
        with self._lock:
            frames = list(self._frames)
        end_to_end = [
            item.end_to_end_latency_ms for item in frames
            if item.end_to_end_latency_ms is not None
        ]
        recent = [
            item.end_to_end_latency_ms for item in self._recent()
            if item.end_to_end_latency_ms is not None
        ]

        def _collect(attribute: str) -> list[float]:
            return [
                value for value in (getattr(item, attribute) for item in frames)
                if value is not None
            ]

        upload = _collect("upload_latency_ms")
        queue = _collect("queue_latency_ms")
        inference = _collect("inference_latency_ms")
        returned = _collect("return_latency_ms")
        return {
            "current_ms": round(recent[-1], 3) if recent else None,
            "median_ms": round(statistics.median(end_to_end), 3) if end_to_end else None,
            "p95_ms": _percentile(end_to_end, 0.95),
            "max_ms": round(max(end_to_end), 3) if end_to_end else None,
            # Cloud only: where the time actually goes.
            "upload_ms": round(statistics.median(upload), 3) if upload else None,
            "queue_ms": round(statistics.median(queue), 3) if queue else None,
            "inference_ms": round(statistics.median(inference), 3) if inference else None,
            "return_ms": round(statistics.median(returned), 3) if returned else None,
        }

    def reliability(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._frames)
            captured, submitted = self._captured, self._submitted
        processed = len([item for item in frames if item.result_available_at is not None])
        dropped = len([item for item in frames if item.dropped])
        stale = len([item for item in frames if item.stale])
        return {
            "frames_captured": captured,
            "frames_submitted": submitted,
            "frames_processed": processed,
            "frames_dropped": dropped,
            "stale_frames_discarded": stale,
            "frame_drop_percent": (
                round(dropped / captured * 100.0, 2) if captured else 0.0
            ),
            "queue_depth": frames[-1].queue_depth if frames else 0,
        }

    def resources(self) -> dict[str, Any]:
        with self._lock:
            return self._resources.to_dict()

    def accuracy(self) -> dict[str, Any]:
        """Scored only where ground truth exists; otherwise None, not zero."""
        with self._lock:
            outcomes = list(self._outcomes)
        errors = [item.percentage_error for item in outcomes if item.percentage_error is not None]

        def _rate(attribute: str) -> float | None:
            scored = [getattr(item, attribute) for item in outcomes]
            scored = [value for value in scored if value is not None]
            if not scored:
                return None
            return round(sum(1 for value in scored if value) / len(scored) * 100.0, 2)

        return {
            "objects": len(outcomes),
            "objects_with_ground_truth": len(errors),
            "mean_volume_error_percent": (
                round(statistics.fmean(errors), 3) if errors else None
            ),
            "colour_accuracy_percent": _rate("colour_correct"),
            "material_accuracy_percent": _rate("material_correct"),
            "sorting_accuracy_percent": _rate("sorting_correct"),
            "bag_count_accuracy_percent": _rate("bag_count_correct"),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "processing_mode": self.processing_mode,
            "recorded_input_id": self.recorded_input_id,
            "elapsed_seconds": round(time.time() - self.started_at, 2),
            "throughput": self.throughput(),
            "latency": self.latency(),
            "reliability": self.reliability(),
            "resources": self.resources(),
            "accuracy": self.accuracy(),
        }

    # --------------------------------------------------------------- export
    def rows(self) -> list[dict[str, Any]]:
        """One row per sampled frame, joined to its object outcome if any."""
        throughput = self.throughput()
        reliability = self.reliability()
        resources = self.resources()
        with self._lock:
            frames = list(self._frames)
            outcomes = {item.event_id: item for item in self._outcomes}
        by_index = list(outcomes.values())
        rows: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            outcome = by_index[index] if index < len(by_index) else None
            rows.append({
                "session_id": self.session_id,
                "frame_id": frame.frame_id,
                "event_id": None if outcome is None else outcome.event_id,
                "timestamp": round(frame.captured_at, 3),
                "processing_mode": frame.processing_mode,
                "capture_fps": throughput["capture_fps"],
                "inference_fps": throughput["inference_fps"],
                "display_fps": throughput["display_fps"],
                "upload_latency_ms": frame.upload_latency_ms,
                "inference_latency_ms": frame.inference_latency_ms,
                "return_latency_ms": frame.return_latency_ms,
                "end_to_end_latency_ms": frame.end_to_end_latency_ms,
                "queue_depth": frame.queue_depth,
                "frames_dropped": reliability["frames_dropped"],
                "cpu_percent": resources["cpu_percent"],
                "ram_mb": resources["ram_mb"],
                "gpu_utilization_percent": resources["gpu_utilization_percent"],
                "gpu_memory_mb": resources["gpu_memory_mb"],
                "estimated_litres": None if outcome is None else outcome.estimated_litres,
                "ground_truth_litres": None if outcome is None else outcome.ground_truth_litres,
                "absolute_error_litres": None if outcome is None else outcome.absolute_error_litres,
                "percentage_error": None if outcome is None else outcome.percentage_error,
                "colour_correct": None if outcome is None else outcome.colour_correct,
                "material_correct": None if outcome is None else outcome.material_correct,
                "sorting_correct": None if outcome is None else outcome.sorting_correct,
                "bag_count_correct": None if outcome is None else outcome.bag_count_correct,
            })
        return rows

    def write_csv(self, directory: Path, name: str = BENCHMARK_CSV_NAME) -> Path:
        destination = Path(directory).resolve() / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=BENCHMARK_COLUMNS, extrasaction="ignore",
            )
            writer.writeheader()
            for row in self.rows():
                writer.writerow(row)
        return destination


def compare(local: BenchmarkSession | None, cloud: BenchmarkSession | None) -> dict[str, Any]:
    """Side-by-side summary, with an explicit verdict the numbers support.

    The verdict refuses to favour either mode unless the measurement actually
    shows it, and says so when the inputs were not the same recording.
    """
    summaries = {
        "local": None if local is None else local.summary(),
        "cloud": None if cloud is None else cloud.summary(),
    }
    verdict: dict[str, Any] = {"comparable": False, "faster": None, "note": None}
    if local is None or cloud is None:
        verdict["note"] = "Run both modes before comparing."
        return {"modes": summaries, "verdict": verdict}
    if local.recorded_input_id != cloud.recorded_input_id or local.recorded_input_id is None:
        verdict["note"] = (
            "Both modes must process the same recorded input; these ran on "
            "different inputs, so the difference measures the inputs, not the modes."
        )
        return {"modes": summaries, "verdict": verdict}
    local_p95 = summaries["local"]["latency"]["p95_ms"]
    cloud_p95 = summaries["cloud"]["latency"]["p95_ms"]
    verdict["comparable"] = True
    if local_p95 is None or cloud_p95 is None:
        verdict["note"] = "Not enough completed frames on one side to compare latency."
        return {"modes": summaries, "verdict": verdict}
    verdict["faster"] = "cloud" if cloud_p95 < local_p95 else "local"
    verdict["p95_difference_ms"] = round(local_p95 - cloud_p95, 3)
    verdict["note"] = (
        f"P95 end-to-end latency: local {local_p95} ms, cloud {cloud_p95} ms, "
        f"over the same recorded input ({local.recorded_input_id})."
    )
    return {"modes": summaries, "verdict": verdict}


def summarise_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Session summary from an exported CSV, for offline analysis."""
    collected = list(rows)
    latencies = [
        float(row["end_to_end_latency_ms"]) for row in collected
        if row.get("end_to_end_latency_ms") not in (None, "")
    ]
    return {
        "frames": len(collected),
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
    }
