"""One session-wide count of NEW bags deposited into the bin.

Each camera runs its own time-based watcher, fed on EVERY processed frame
(tracks, a small greyscale view of the event region, depth when available):

    initialising -> watching -> candidate -> settling -> confirmed | rejected -> watching

* initialising: after a short stable start-up period the current view, depth
  and tracks become the baseline -- bags already in the bin are part of the
  initial fill and never "new".
* candidate: frame-to-frame motion in the event region, or a track that is not
  a known one (by id or by box overlap, so an ID switch is not new).
* settling: the scene must stay still for SETTLE_S seconds (elapsed time, not
  frames, so a low frame rate still settles); CANDIDATE_TIMEOUT_S without
  settling records a rejection with that reason.
* evaluation compares the settled view with the last committed one: a new
  bag-like track under a persistent change, an existing mask that grew into a
  touching new bag, or a persistent, compact change region (with a local depth
  rise where depth exists) not explained by a known bag moving. Nothing needs
  a fill profile, litres or a rise of the bin's maximum height.

Each camera counts INDEPENDENTLY (its own counter and events; nothing copied
between cameras). Sizes arriving a few frames later update the same event.
The session and events persist to JSON and resume after a restart; a CSV row
(camera + session + event id) is appended once per event, after FINALISE_S,
and once per rejected candidate.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

SMALL = (160, 120)             # evaluation resolution (w, h)
WARMUP_S = 4.0                 # minimum time before the baseline is taken
BASELINE_MAX_S = 20.0          # take the baseline even if the scene never stills
SETTLE_S = 1.0                 # stillness needed to evaluate a candidate
CANDIDATE_TIMEOUT_S = 30.0     # candidate that never settles -> rejected
FINALISE_S = 12.0              # late measurements update an event this long; then its CSV row is written
PIXEL_DIFF = 28                # grey-level change counted as changed
MOTION_ENTER = 0.02            # changed fraction between frames that opens a candidate
MOTION_STILL = 0.008           # below this the frame is still
MIN_CHANGE = 0.006             # persistent changed fraction of the region worth evaluating
MIN_RISE_M = 0.03              # local depth rise counted as new material
ID_SWITCH_IOU = 0.3            # an unknown id overlapping a committed box this much is a known bag
GROWTH = 1.35                  # a known box grown this much can hold a touching new bag

BAG_WORDS = {"bag", "sack", "pillow", "cushion", "textile", "fabric", "garbage", "waste", "trash", "refuse",
             "rubbish", "liner", "parcel", "package", "packet", "box", "carton", "blanket", "clothing", "bin"}
NOT_BAG_WORDS = {"person", "hand", "arm", "finger", "human", "head", "face", "leg", "foot"}

ENVELOPE_LABEL = "new-bag outer envelope (visible L x W x added height box)"
DELTA_LABEL = "net before/after change in bin occupancy (whole-bin surface, separate method)"

CSV_FIELDS = (
    "session_id", "camera", "event_id", "counted", "count_after", "deposit_time", "cameras", "association",
    "confidence", "evidence",
    "track_id", "colour", "object_type", "detector_label", "material", "length_cm", "width_cm", "height_cm",
    "height_source", "envelope_l", "delta_occupancy_l", "measurement_status", "reason",
)


def bag_like(label: str | None) -> bool:
    words = set((label or "").lower().replace("-", " ").replace("_", " ").split())
    return bool(words & BAG_WORDS) and not (words & NOT_BAG_WORDS)


@dataclass
class TrackInfo:
    track_id: int
    box: tuple[float, float, float, float]          # in SMALL coordinates
    label: str = "unknown"
    colour: str = "unknown"
    material: str | None = None
    material_confidence: float = 0.0
    length_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    rejection: str | None = None


@dataclass
class FrameEvidence:
    camera: str
    timestamp: float
    grey: np.ndarray                                # SMALL uint8 view (BGR, or grey)
    region: np.ndarray                              # SMALL bool (event region)
    tracks: list[TrackInfo] = field(default_factory=list)
    depth: np.ndarray | None = None                 # SMALL metres (RealSense aligned depth)
    heights: np.ndarray | None = None               # SMALL height above floor (approximate/measured pose)
    height_status: str | None = None                # profile status behind `heights`
    area: np.ndarray | None = None                  # SMALL floor area per pixel (m^2)
    xs: np.ndarray | None = None                    # SMALL camera x, y (m) for footprint L x W
    ys: np.ndarray | None = None


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel change: the largest per-channel difference (a red bag can match the grey level).

    The median difference is removed first, per channel: an auto-exposure or
    lighting step shifts the whole view and is not an object.
    """
    diff = a.astype(np.int16) - b.astype(np.int16)
    offset = np.median(diff.reshape(-1, diff.shape[-1]) if diff.ndim == 3 else diff.ravel(), axis=0)
    diff = np.abs(diff - offset.astype(np.int16))
    return (diff.max(axis=-1) if diff.ndim == 3 else diff) > PIXEL_DIFF


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_mask(box, shape) -> np.ndarray:
    mask = np.zeros(shape, bool)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return mask


class CameraWatcher:
    """One camera's deposit state machine. Pure logic: fed FrameEvidence."""

    def __init__(self, camera: str, started_at: float) -> None:
        self.camera = camera
        self.started_at = started_at
        self.state = "initialising"
        self.reason: str | None = "waiting for a stable start-up view"
        self.prev_grey: np.ndarray | None = None
        self.still_since: float | None = None
        self.committed: FrameEvidence | None = None
        self.known_ids: set[int] = set()
        self.baseline_ids: set[int] = set()
        self.candidate_since: float | None = None
        self.candidate_trigger: str | None = None
        self.motion = 0.0
        self.last_frame_at: float | None = None
        self.resync = False
        self.votes: dict[int, dict[str, Counter]] = {}
        self.noise = 0.0                    # typical frame-to-frame change of a still scene (sensor/JPEG)
        self.pending: list[tuple[dict[str, Any], int, float]] = []   # (event, track id, until) for late sizes

    # ----------------------------------------------------------- helpers
    def _vote(self, tracks: list[TrackInfo]) -> None:
        for t in tracks:
            v = self.votes.setdefault(t.track_id, {"colour": Counter(), "material": Counter()})
            if t.colour and t.colour != "unknown":
                v["colour"][t.colour] += 1
            if t.material and t.material != "unknown" and t.material_confidence >= 0.5:
                v["material"][t.material] += 1
        if len(self.votes) > 400:
            for key in sorted(self.votes)[:200]:
                self.votes.pop(key, None)

    def consensus(self, track_id: int | None, kind: str, fallback: str) -> str:
        votes = self.votes.get(track_id, {}).get(kind) if track_id is not None else None
        if not votes:
            return fallback
        value, n = votes.most_common(1)[0]
        return value if n >= (3 if kind == "material" else 1) and n >= 0.6 * sum(votes.values()) else fallback

    def _known(self, track: TrackInfo) -> bool:
        if track.track_id in self.known_ids:
            return True
        boxes = [] if self.committed is None else [t.box for t in self.committed.tracks]
        if any(_iou(track.box, b) >= ID_SWITCH_IOU for b in boxes):
            self.known_ids.add(track.track_id)           # ID switch / re-detection of a known bag
            return True
        return False

    def _commit(self, ev: FrameEvidence) -> None:
        self.committed = ev
        self.known_ids |= {t.track_id for t in ev.tracks}

    # --------------------------------------------------------------- main
    def observe(self, ev: FrameEvidence) -> tuple[str, dict[str, Any]] | None:
        """Advance on one frame; returns ("confirmed"|"rejected", details) when a candidate ends."""
        now = ev.timestamp
        self.last_frame_at = now
        region = ev.region if ev.region.any() else np.ones_like(ev.region)
        if self.prev_grey is not None and self.prev_grey.shape == ev.grey.shape:
            diff = _changed(ev.grey, self.prev_grey)
            self.motion = float(np.count_nonzero(diff & region)) / max(1, int(np.count_nonzero(region)))
        self.prev_grey = ev.grey
        # Thresholds follow this camera's own noise floor, so a noisy dark view still settles.
        if self.motion < max(MOTION_ENTER, 4 * self.noise):
            self.noise = 0.9 * self.noise + 0.1 * self.motion
        still_limit = max(MOTION_STILL, 2.5 * self.noise)
        enter_limit = max(MOTION_ENTER, 4 * self.noise)
        still = self.motion < still_limit
        if still:
            self.still_since = self.still_since if self.still_since is not None else now
        else:
            self.still_since = None
        settled = self.still_since is not None and now - self.still_since >= SETTLE_S
        self._vote(ev.tracks)

        if self.state == "initialising":
            age = now - self.started_at
            if (age >= WARMUP_S and settled) or age >= BASELINE_MAX_S:
                self._commit(ev)
                self.baseline_ids = {t.track_id for t in ev.tracks}
                self.state, self.reason = "watching", None
                if not settled:
                    self.reason = "baseline taken while the scene was still moving"
                LOGGER.info("%s deposit watcher: baseline ready with %d existing tracks",
                            self.camera, len(self.baseline_ids))
            return None

        if self.state == "watching":
            if self.resync:
                if settled:
                    self._commit(ev)
                    self.resync = False
                return None
            unknown = [t for t in ev.tracks if bag_like(t.label) and not self._known(t)]
            if self.motion >= enter_limit or unknown:
                self.state = "candidate"
                self.candidate_since = now
                self.candidate_trigger = "motion" if self.motion >= enter_limit else "new track"
                self.reason = f"candidate: {self.candidate_trigger}"
            else:
                # Known tracks drift a little between frames; keep boxes current while nothing happens.
                if settled and self.committed is not None:
                    self.committed.tracks = [t for t in ev.tracks if t.track_id in self.known_ids] or \
                        self.committed.tracks
                return None

        # candidate / settling
        if now - (self.candidate_since or now) > CANDIDATE_TIMEOUT_S:
            reason = f"scene did not settle within {CANDIDATE_TIMEOUT_S:.0f} s (trigger: {self.candidate_trigger})"
            self._end(reason)
            self.resync = True
            return "rejected", {"reason": reason}
        if not settled:
            self.state = "settling" if still else "candidate"
            self.reason = "settling" if still else "scene moving"
            return None
        outcome = self._evaluate(ev)
        self._commit(ev)
        self._end(None if outcome[0] == "confirmed" else outcome[1]["reason"])
        return outcome

    def _end(self, reason: str | None) -> None:
        self.state, self.reason = "watching", reason
        self.candidate_since = None
        if reason:
            LOGGER.info("%s deposit candidate rejected: %s", self.camera, reason)

    @staticmethod
    def _vacated(ev: FrameEvidence, before: FrameEvidence, change: np.ndarray, added: np.ndarray,
                 region: np.ndarray) -> bool:
        """Did something LEAVE a spot while `added` appeared? Then it moved; a deposit only adds.

        With depth: a comparable area whose surface dropped. Without depth: a
        comparable changed area, away from the added object, whose BEFORE look
        matches the added object's AFTER look (the same bag, somewhere else).
        """
        size = int(np.count_nonzero(added))
        if size == 0:
            return False
        if ev.depth is not None and before.depth is not None and ev.depth.shape == before.depth.shape:
            valid = (ev.depth > 0.05) & (before.depth > 0.05)
            dropped = valid & (ev.depth - before.depth >= MIN_RISE_M) & region
            return int(np.count_nonzero(dropped)) >= 0.5 * size
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return False
        away = change & ~cv2.dilate(added.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(away.astype(np.uint8), 8)
        if count < 2 or before.grey.ndim != ev.grey.ndim:
            return False
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        spot = labels == biggest
        if np.count_nonzero(spot) < 0.5 * size:
            return False
        was_there = before.grey[spot].astype(np.float32).reshape(-1, before.grey.shape[-1] if before.grey.ndim == 3 else 1)
        is_here = ev.grey[added].astype(np.float32).reshape(-1, ev.grey.shape[-1] if ev.grey.ndim == 3 else 1)
        return float(np.abs(np.median(was_there, axis=0) - np.median(is_here, axis=0)).max()) < 25.0

    def _removed(self, ev: FrameEvidence, before: FrameEvidence, area_mask: np.ndarray) -> bool:
        """Was something TAKEN OUT here? Lifting the top bag reveals the one below: not a new deposit."""
        if ev.heights is not None and before.heights is not None and ev.heights.shape == before.heights.shape:
            diff = (ev.heights - before.heights)[area_mask]
            diff = diff[np.isfinite(diff)]
            if diff.size >= 10:
                return float(np.median(diff)) < -0.04
        current = {t.track_id for t in ev.tracks}
        gone = [t for t in before.tracks if t.track_id not in current]
        inside = np.zeros_like(area_mask)
        for t in gone:
            inside |= _box_mask(t.box, area_mask.shape)
        return bool(gone) and np.count_nonzero(area_mask & inside) >= 0.6 * max(1, np.count_nonzero(area_mask))

    def _evaluate(self, ev: FrameEvidence) -> tuple[str, dict[str, Any]]:
        before = self.committed
        region = ev.region if ev.region.any() else np.ones_like(ev.region)
        total = max(1, int(np.count_nonzero(region)))
        change = _changed(ev.grey, before.grey) & region
        rise = None
        if ev.depth is not None and before.depth is not None and ev.depth.shape == before.depth.shape:
            valid = (ev.depth > 0.05) & (before.depth > 0.05)
            rise = np.where(valid, before.depth - ev.depth, 0.0)
            change |= (rise >= MIN_RISE_M) & region
        changed = float(np.count_nonzero(change)) / total
        details: dict[str, Any] = {"changed_fraction": round(changed, 4), "trigger": self.candidate_trigger}
        old = {t.track_id: t for t in before.tracks}

        def changed_share(box) -> float:
            m = _box_mask(box, change.shape) & region
            return float(np.count_nonzero(change & m)) / max(1, int(np.count_nonzero(m)))

        new_tracks = [t for t in ev.tracks if bag_like(t.label) and not self._known(t)]
        for t in sorted(new_tracks, key=lambda t: -changed_share(t.box)):
            if changed_share(t.box) >= 0.10:
                added = _box_mask(t.box, change.shape) & change
                if self._removed(ev, before, added):
                    return "rejected", {**details, "reason": "a bag was removed; the one revealed below is not new"}
                if self._vacated(ev, before, change, added, region):
                    return "rejected", {**details, "reason": "an existing bag moved: its old spot was vacated"}
                return "confirmed", {**details, "evidence": "new bag-like track over a persistent change",
                                     "track": t, "mask": added, "rise": rise}
        for t in ev.tracks:
            prior = old.get(t.track_id)
            if prior is not None and bag_like(t.label) and _area(t.box) >= GROWTH * max(_area(prior.box), 1.0):
                grown = _box_mask(t.box, change.shape) & ~_box_mask(prior.box, change.shape) & change
                if np.count_nonzero(grown) >= MIN_CHANGE * total:
                    return "confirmed", {**details, "evidence": "existing mask grew into a touching new bag",
                                         "track": None, "mask": grown, "rise": rise}
        if changed < MIN_CHANGE:
            reason = ("new track id but the scene is unchanged (re-detection / ID switch)" if new_tracks
                      else "no persistent change after settling (hand, occlusion or transient)")
            return "rejected", {**details, "reason": reason}
        # Known boxes jitter on a crowded pile; that is not an explanation for new
        # material. A real move is recognised by the spot it vacated (_vacated).
        rest = change
        depth_note = ""
        if rise is not None:
            valid = (ev.depth > 0.05) & (before.depth > 0.05) & rest
            if np.count_nonzero(valid) >= 0.3 * np.count_nonzero(rest):
                risen = valid & (rise >= MIN_RISE_M)
                if np.count_nonzero(risen) < 0.35 * np.count_nonzero(valid):
                    return "rejected", {**details, "reason": "change without a local surface rise where depth "
                                                             "is valid (lighting, shadow or an item shifted)"}
                rest = risen
            else:
                depth_note = ", depth too sparse there to check the rise"
        try:
            import cv2
            n, labels, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), 8)
            largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        except ImportError:  # pragma: no cover
            largest, labels, n, stats = int(np.count_nonzero(rest)), None, 0, None
        if largest < 0.6 * np.count_nonzero(rest) or largest < MIN_CHANGE * total:
            return "rejected", {**details, "reason": "scattered change, not one new object"}
        if self._removed(ev, before, rest):
            return "rejected", {**details, "reason": "a bag was removed; the one revealed below is not new"}
        if self._vacated(ev, before, change, rest, region):
            return "rejected", {**details, "reason": "an existing bag moved: its old spot was vacated"}
        second = int(np.sort(stats[1:, cv2.CC_STAT_AREA])[-2]) if n > 2 else 0
        details["ambiguous"] = second >= 0.4 * largest and second >= MIN_CHANGE * total
        return "confirmed", {**details, "evidence": "persistent new foreground region"
                             + (" with local depth rise" if rise is not None and not depth_note else " (no depth)")
                             + depth_note, "track": None, "mask": rest, "rise": rise}


class SessionDeposits:
    def __init__(self, directory: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.directory = Path(directory)
        self.csv_path = self.directory / "session_deposits.csv"
        self.state_path = self.directory / "session_state.json"
        self._lock = threading.RLock()
        self.watchers: dict[str, CameraWatcher] = {}
        self._pending_csv: list[dict[str, Any]] = []
        self._last_frame_at = 0.0
        if not self._resume():
            self._fresh()

    # ------------------------------------------------------------ session
    def _fresh(self) -> None:
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.session_started_at = self.clock()
        self.events: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.written: set[str] = set()
        self.last_confirmed_at: float | None = None
        self.resumed = False
        self.ignored_redetections = 0
        self.watchers.clear()
        self._save()

    def _resume(self) -> bool:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.session_id, self.session_started_at = data["session_id"], float(data["session_started_at"])
            self.events, self.rejected = list(data["events"]), list(data.get("rejected", []))
            # Records saved by an earlier version lack newer fields: fill them, never crash the panel.
            for record in self.events + self.rejected:
                cameras = record.get("cameras") or [record.get("camera") or "unknown"]
                record.setdefault("cameras", cameras)
                record.setdefault("camera", cameras[0])
                for key in ("colour", "object_type", "detector_label", "material", "confidence",
                            "measurement_status", "association", "reason", "length_cm", "width_cm",
                            "height_cm", "height_source", "envelope_l", "delta_occupancy_l", "track_id"):
                    record.setdefault(key, None)
                record.setdefault("evidence", {})
            self.written = set(data.get("written", []))
            self.last_confirmed_at = data.get("last_confirmed_at")
            self.ignored_redetections = 0
            self._pending_csv = [e for e in self.events if e["event_id"] not in self.written]
            self.resumed = True
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def new_session(self) -> dict[str, Any]:
        with self._lock:
            self._flush_csv(force=True)
            self._fresh()
            return self.snapshot()

    def _save(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "session_id": self.session_id, "session_started_at": self.session_started_at,
                "events": self.events, "rejected": self.rejected[-200:], "written": sorted(self.written),
                "last_confirmed_at": self.last_confirmed_at}, default=str), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            LOGGER.exception("could not persist the deposit session")

    @property
    def count(self) -> int:
        return len(self.events)

    # -------------------------------------------------------------- frames
    def observe(self, ev: FrameEvidence) -> dict[str, Any] | None:
        """Feed one camera frame; returns the counted/merged event when one is confirmed."""
        with self._lock:
            self._last_frame_at = max(self._last_frame_at, ev.timestamp)
            watcher = self.watchers.get(ev.camera)
            if watcher is None:
                # The warm-up starts at this camera's first frame (cameras may connect late,
                # and a resumed session re-takes its baseline: track ids do not survive a restart).
                watcher = self.watchers[ev.camera] = CameraWatcher(ev.camera, ev.timestamp)
            outcome = watcher.observe(ev)
            if watcher.pending:
                self._late_sizes(ev, watcher)
            self._flush_csv()
            if outcome is None:
                return None
            kind, details = outcome
            if kind == "rejected" and details["reason"].startswith("new track id but"):
                self.ignored_redetections += 1           # counted, not logged per row: detector id churn
                return None
            if kind == "rejected":
                row = {"session_id": self.session_id, "event_id": f"R-{ev.camera}-{uuid.uuid4().hex[:8]}",
                       "counted": False, "deposit_time": ev.timestamp, "cameras": [ev.camera], "camera": ev.camera,
                       "reason": details["reason"], "evidence": details.get("trigger")}
                self.rejected.append(row)
                self._write_csv(row)
                self._save()
                return None
            return self._confirm(ev, watcher, details)

    def count_for(self, camera: str) -> int:
        return sum(1 for e in self.events if e["camera"] == camera)

    def _confirm(self, ev: FrameEvidence, watcher: CameraWatcher, details: dict[str, Any]) -> dict[str, Any]:
        """Each camera counts its own deposits: nothing is merged with, or copied from, the other camera."""
        track: TrackInfo | None = details.get("track")
        record = self._measure(ev, watcher, details, track)
        number = self.count_for(ev.camera) + 1
        record["event_id"] = f"{'RS' if ev.camera == 'realsense' else 'LG'}-{self.session_id[-6:]}-{number:03d}"
        record["count_after"] = number
        if details.get("ambiguous"):
            record["association"] = "ambiguous: may be 2 bags seen as one observation (counted once, check)"
        self.events.append(record)
        self.last_confirmed_at = record["deposit_time"]
        if track is not None:
            # Sizes often settle a few frames after the count: update this same event later.
            watcher.pending.append((record, track.track_id, ev.timestamp + FINALISE_S))
        self._pending_csv.append(record)
        self._save()
        LOGGER.info("NEW bag %s confirmed by %s (%s); %s count %d", record["event_id"], ev.camera,
                    details["evidence"], ev.camera, number)
        return record

    def _late_sizes(self, ev: FrameEvidence, watcher: CameraWatcher) -> None:
        keep = []
        changed = False
        for record, track_id, until in watcher.pending:
            if ev.timestamp > until:
                continue
            track = next((t for t in ev.tracks if t.track_id == track_id), None)
            if track is not None and not track.rejection and track.length_mm and track.width_mm:
                if record["length_cm"] is None:
                    record["length_cm"], record["width_cm"] = _r(track.length_mm / 10), _r(track.width_mm / 10)
                    changed = True
                if record["height_cm"] is None and track.height_mm:
                    record["height_cm"] = _r(track.height_mm / 10)
                    record["height_source"] = "detector height (surface under the bag not measured)"
                    changed = True
                if record["length_cm"] and record["width_cm"] and record["height_cm"]:
                    record["envelope_l"] = round(record["length_cm"] * record["width_cm"] * record["height_cm"]
                                                 / 1000.0, 1)
                    if record["measurement_status"] in ("na", "partial"):
                        record["measurement_status"] = "approximate"
                        record["reason"] = "size added after the count (same event)"
                    continue                       # complete: stop following it
            keep.append((record, track_id, until))
        watcher.pending = keep
        if changed:
            self._save()

    def _measure(self, ev, watcher, details, track) -> dict[str, Any]:
        reasons: list[str] = []
        length = width = height = None
        source = None
        if track is not None and track.rejection:
            reasons.append(f"detector dimensions withheld: {track.rejection}")
        elif track is not None and track.length_mm and track.width_mm:
            length, width = track.length_mm / 10.0, track.width_mm / 10.0
        elif track is None:
            reasons.append("no single tracked bag to take L x W from")
        mask = details.get("mask")
        before = watcher.committed
        added = None
        if mask is not None and mask.any():
            if ev.heights is not None and before is not None and before.heights is not None:
                diff = (ev.heights - before.heights)[mask]
                diff = diff[np.isfinite(diff) & (diff > 0.02)]
                if diff.size >= 6:
                    added = float(np.percentile(diff, 90))
                    source = f"after top - before surface under the bag ({ev.height_status} pose)"
            if added is None and details.get("rise") is not None:
                diff = details["rise"][mask]
                diff = diff[diff > 0.02]
                if diff.size >= 6:
                    added = float(np.percentile(diff, 90))
                    source = "local depth rise along the line of sight (pose unknown, approximate)"
        volume = None
        if mask is not None and mask.any() and ev.heights is not None and before is not None \
                and before.heights is not None and ev.area is not None:
            rise = (ev.heights - before.heights)
            bag = mask & np.isfinite(rise) & (rise > 0.02)
            if np.count_nonzero(bag) >= 6:
                volume = round(float(np.sum(rise[bag] * ev.area[bag])) * 1000.0, 1)   # litres under the bag top
                if ev.xs is not None and length is None:
                    try:
                        import cv2
                        pts = np.c_[ev.xs[bag], ev.ys[bag]].astype(np.float32)
                        (_, _), (w1, w2), _ = cv2.minAreaRect(pts)
                        length, width = max(w1, w2) * 100.0, min(w1, w2) * 100.0
                        reasons = [r for r in reasons if not r.startswith("no single tracked bag")]
                    except ImportError:  # pragma: no cover
                        pass
        if added is not None:
            height = added * 100.0
        elif track is not None and track.height_mm and not track.rejection:
            height, source = track.height_mm / 10.0, "detector height (surface under the bag not measured)"
        else:
            reasons.append("no reliable before/after surface under the bag (hidden, occluded or no depth)")
        envelope = volume if volume is not None else (
            round(length * width * height / 1000.0, 1) if length and width and height else None)
        if envelope is None:
            status = "partial" if (length or height) else "na"
        elif source and source.startswith("after top") and "measured" in source:
            status = "measured"
        else:
            status = "approximate"
        track_id = None if track is None else track.track_id
        label = "unknown" if track is None else track.label
        evidence = details["evidence"]
        confidence = ("high" if track is not None and "rise" in evidence or (track is not None and ev.depth is not None)
                      else "medium" if track is not None or "depth rise" in evidence else "low (foreground only)")
        material = watcher.consensus(track_id, "material", "UNKNOWN")
        return {
            "session_id": self.session_id, "event_id": None, "counted": True, "camera": ev.camera,
            "confidence": confidence,
            "count_after": None, "deposit_time": float(ev.timestamp), "cameras": [ev.camera],
            "association": "single camera", "track_id": track_id,
            "colour": watcher.consensus(track_id, "colour", "unknown" if track is None else track.colour),
            "object_type": "bag-like object" if track is None or bag_like(label) else label,
            "detector_label": label, "material": material.upper() if material != "UNKNOWN" else material,
            "length_cm": _r(length), "width_cm": _r(width), "height_cm": _r(height), "height_source": source,
            "envelope_l": envelope, "envelope_label": ENVELOPE_LABEL,
            "delta_occupancy_l": None, "delta_label": DELTA_LABEL,
            "measurement_status": status, "reason": "; ".join(reasons) or None,
            "evidence": {ev.camera: {"evidence": details["evidence"], "trigger": details.get("trigger"),
                                     "changed_fraction": details.get("changed_fraction")}},
        }

    def attach_occupancy(self, camera: str, event: Any, extra: dict[str, Any] | None = None) -> None:
        """A whole-bin before/after occupancy event is EVIDENCE only; it never counts on its own."""
        with self._lock:
            near = [e for e in self.events if abs(e["deposit_time"] - event.finalized_at) <= FINALISE_S
                    and e["camera"] == camera and e["delta_occupancy_l"] is None]
            if near and event.delta_occupancy_l is not None:
                near[-1]["delta_occupancy_l"] = event.delta_occupancy_l
                self._save()

    # ----------------------------------------------------------------- csv
    def _flush_csv(self, force: bool = False) -> None:
        now = max(self.clock(), self._last_frame_at)      # frame time: the clock the events use
        keep = []
        for record in self._pending_csv:
            if force or now - record["deposit_time"] > FINALISE_S:
                self._write_csv(record)
            else:
                keep.append(record)
        if len(keep) != len(self._pending_csv):
            self._pending_csv = keep
            self._save()

    def _write_csv(self, record: dict[str, Any]) -> None:
        if record["event_id"] in self.written:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            new = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new:
                    writer.writeheader()
                evidence = record.get("evidence")
                writer.writerow({**record, "cameras": "+".join(record["cameras"]),
                                 "evidence": "; ".join(f"{k}: {v.get('evidence')}" for k, v in evidence.items())
                                 if isinstance(evidence, dict) else evidence,
                                 "deposit_time": time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.localtime(record["deposit_time"]))})
            self.written.add(record["event_id"])        # event ids carry camera + session
        except OSError:
            LOGGER.exception("could not append the deposit CSV")

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            cams = {c: {"state": w.state, "reason": w.reason, "motion": round(w.motion, 4),
                        "baseline_tracks": len(w.baseline_ids), "last_frame_at": w.last_frame_at,
                        "candidate_age_s": None if w.candidate_since is None else round(now - w.candidate_since, 1)}
                    for c, w in self.watchers.items()}
            states = [w["state"] for w in cams.values()]
            status = ("initialising" if not states or "initialising" in states else
                      "settling" if "settling" in states else "candidate" if "candidate" in states else "watching")
            frames = [w["last_frame_at"] for w in cams.values() if w["last_frame_at"]]
            for camera, block in cams.items():
                mine = [e for e in self.events if e["camera"] == camera]
                block.update(new_bags=len(mine), last_confirmed_at=mine[-1]["deposit_time"] if mine else None,
                             events=[dict(e) for e in mine],
                             rejected=sum(1 for r in self.rejected if r["cameras"][0] == camera),
                             latest_rejection=next((r["reason"] for r in reversed(self.rejected)
                                                    if r["cameras"][0] == camera), None))
            return {
                "session_id": self.session_id, "session_started_at": self.session_started_at,
                "elapsed_s": round(now - self.session_started_at, 1), "resumed": self.resumed,
                "updated_at": max(frames) if frames else None, "status": status, "cameras": cams,
                "new_bags_this_session": self.count, "last_confirmed_at": self.last_confirmed_at,
                "events": [{k: v for k, v in e.items()} for e in self.events],
                "rejected_candidates": len(self.rejected), "ignored_redetections": self.ignored_redetections,
                "recent_rejections": [{"camera": r["cameras"][0], "reason": r["reason"], "at": r["deposit_time"]}
                                      for r in self.rejected[-5:]],
                "envelope_label": ENVELOPE_LABEL, "delta_label": DELTA_LABEL,
            }


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
