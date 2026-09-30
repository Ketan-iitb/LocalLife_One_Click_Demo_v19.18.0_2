"""One session-wide count of NEW bags deposited into the bin.

A deposit is counted from a finalised before/after occupancy event (a bag
entered, the scene moved, then settled) -- never from detections alone, so a
re-detected, re-boxed or re-tracked bag cannot add to the count. On top of
that:

* bags already in the bin during the start-up warm-up are the baseline: they
  are part of the initial fill but never "new this session";
* an event whose bag is a baseline track, or whose measured surface did not
  rise, is an old bag moving -- logged as a rejected candidate, not counted;
* the same event or the same track on one camera counts once;
* both cameras seeing one bag inside the merge window count once (the second
  camera is added as evidence);
* a new bag whose volume cannot be measured (hidden, occluded) still counts,
  with volume N/A and the reason.

Every candidate goes to an append-only CSV, so the history outlives a browser
refresh and a restart (rows carry the session id).
"""

from __future__ import annotations

import csv
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

WARMUP_S = 5.0            # tracks seen this long after start are the baseline
MERGE_WINDOW_S = 12.0     # two cameras' events this close are one bag
MIN_RISE_L = 2.0          # a measured occupancy change below this is not a new bag

ENVELOPE_LABEL = "new-bag outer envelope (L x W x added height box)"
DELTA_LABEL = "estimated change in bin occupancy"

CSV_FIELDS = (
    "session_id", "event_id", "counted", "count_after", "deposit_time", "cameras", "track_id",
    "colour", "type", "material", "length_cm", "width_cm", "height_cm", "height_source",
    "envelope_l", "delta_occupancy_l", "measurement_status", "reason",
)


class SessionDeposits:
    def __init__(self, directory: Path, *, clock: Callable[[], float] = time.time,
                 warmup_s: float = WARMUP_S, merge_window_s: float = MERGE_WINDOW_S,
                 min_rise_l: float = MIN_RISE_L) -> None:
        self.clock = clock
        self.warmup_s, self.merge_window_s, self.min_rise_l = warmup_s, merge_window_s, min_rise_l
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.session_started_at = clock()
        self.updated_at = self.session_started_at
        self.csv_path = Path(directory) / "session_deposits.csv"
        self.baseline: dict[str, set[int]] = {}
        self.counted_tracks: dict[str, set[int]] = {}
        self.seen_events: set[str] = set()
        self.events: list[dict[str, Any]] = []       # counted deposits, chronological
        self.rejected: list[dict[str, Any]] = []     # candidates that were not new bags
        self.height_lookup: Callable[[str, float, float], tuple[float | None, str | None]] | None = None
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self.events)

    def in_warmup(self, now: float | None = None) -> bool:
        return (self.clock() if now is None else now) - self.session_started_at < self.warmup_s

    def observe_tracks(self, camera: str, track_ids: list[int], now: float | None = None) -> None:
        """Tracks visible during the start-up warm-up are the baseline (already in the bin)."""
        if self.in_warmup(now):
            with self._lock:
                self.baseline.setdefault(camera, set()).update(t for t in track_ids if t is not None)

    def record(self, camera: str, event: Any, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Consider one finalised occupancy event; returns the counted/merged record or None."""
        extra = extra or {}
        with self._lock:
            key = f"{camera}:{event.event_id}"
            if key in self.seen_events:
                return None
            self.seen_events.add(key)
            self.updated_at = self.clock()
            track = event.track_id
            delta = event.delta_occupancy_l
            reason = None
            if event.started_at - self.session_started_at < self.warmup_s:
                reason = "started during the start-up warm-up (baseline settling)"
            elif track is not None and track in self.baseline.get(camera, set()):
                reason = "baseline bag moved or re-detected"
            elif track is not None and track in self.counted_tracks.get(camera, set()):
                reason = "this bag was already counted"
            elif delta is not None and delta < self.min_rise_l:
                reason = f"surface did not rise ({delta:.1f} L < {self.min_rise_l:.0f} L): old bag moved or occlusion"
            elif track is None and delta is None:
                reason = "no new bag tracked and no measurable surface change"
            record = self._build(camera, event, extra)
            if reason is not None:
                record.update(counted=False, reason=reason)
                self.rejected.append(record)
                self._write(record)
                return None
            if track is not None:
                self.counted_tracks.setdefault(camera, set()).add(track)
            twin = next((e for e in reversed(self.events)
                         if camera not in e["cameras"]
                         and abs(e["deposit_time"] - record["deposit_time"]) <= self.merge_window_s), None)
            if twin is not None:
                twin["cameras"].append(camera)
                twin["evidence"][camera] = record["evidence"][camera]
                rank = {"measured": 2, "partial": 1, "na": 0}
                better = rank[record["measurement_status"]] > rank[twin["measurement_status"]] or (
                    rank[record["measurement_status"]] == rank[twin["measurement_status"]] and camera == "realsense")
                if better:           # keep the better-measured camera's size (RealSense on a tie)
                    for field in ("length_cm", "width_cm", "height_cm", "height_source", "envelope_l", "delta_occupancy_l",
                                  "measurement_status", "reason"):
                        twin[field] = record[field]
                for field in ("colour", "type", "material"):
                    if twin[field] in (None, "unknown", "UNKNOWN") and record[field] not in (None, "unknown", "UNKNOWN"):
                        twin[field] = record[field]
                self._write({**twin, "event_id": record["event_id"], "reason": f"merged into {twin['event_id']}"})
                return twin
            record["count_after"] = len(self.events) + 1
            self.events.append(record)
            self._write(record)
            return record

    def _build(self, camera: str, event: Any, extra: dict[str, Any]) -> dict[str, Any]:
        length = width = height = None
        height_source = None
        reasons = []
        if event.envelope_mm:
            length, width = event.envelope_mm[0] / 10.0, event.envelope_mm[1] / 10.0
        added, why = (None, "no fill profile for this camera")
        if self.height_lookup is not None:
            added, why = self.height_lookup(camera, event.started_at, event.finalized_at)
        if added is not None:
            height, height_source = added * 100.0, "after top - before surface under the bag"
        elif event.envelope_mm:
            height, height_source = event.envelope_mm[2] / 10.0, "detector height (surface under bag not measured)"
            reasons.append(why)
        else:
            reasons.append(why)
        if length is None:
            reasons.insert(0, event.reason or "bag footprint not measured (hidden, occluded or out of zone)")
        envelope = None
        if length and width and height:
            envelope = round(length * width * height / 1000.0, 1)
        status = "measured" if envelope is not None and height_source.startswith("after") else (
            "partial" if envelope is not None else "na")
        material = extra.get("material")
        return {
            "session_id": self.session_id, "event_id": f"D-{event.event_id}", "counted": True,
            "count_after": None, "deposit_time": float(event.finalized_at), "cameras": [camera],
            "track_id": event.track_id, "colour": event.color or "unknown",
            "type": event.label or "unknown",
            "material": material.upper() if material and material != "unknown" else "UNKNOWN",
            "length_cm": _r(length), "width_cm": _r(width), "height_cm": _r(height),
            "height_source": height_source, "envelope_l": envelope, "envelope_label": ENVELOPE_LABEL,
            "delta_occupancy_l": event.delta_occupancy_l, "delta_label": DELTA_LABEL,
            "measurement_status": status, "reason": "; ".join(r for r in reasons if r) or None,
            "evidence": {camera: {"event_id": event.event_id, "status": event.status,
                                  "track_id": event.track_id, "occupancy_reason": event.reason}},
        }

    def _write(self, record: dict[str, Any]) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerow({**record, "cameras": "+".join(record["cameras"]),
                                 "deposit_time": time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.localtime(record["deposit_time"]))})
        except OSError:
            pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id, "session_started_at": self.session_started_at,
                "updated_at": self.updated_at, "new_bags_this_session": self.count,
                "warmup": self.in_warmup(), "baseline_tracks": {k: len(v) for k, v in self.baseline.items()},
                "events": [dict(e, evidence=dict(e["evidence"]), cameras=list(e["cameras"])) for e in self.events],
                "rejected_candidates": len(self.rejected),
                "recent_rejections": [{"event_id": r["event_id"], "camera": r["cameras"][0], "reason": r["reason"]}
                                      for r in self.rejected[-5:]],
                "envelope_label": ENVELOPE_LABEL, "delta_label": DELTA_LABEL,
            }


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
