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

from .bin_policy import apply_to_event

LOGGER = logging.getLogger(__name__)

SMALL = (160, 120)             # evaluation resolution (w, h)
WARMUP_S = 4.0                 # minimum time before the baseline is taken
BASELINE_MAX_S = 20.0          # take the baseline even if the scene never stills
SETTLE_S = float(os.environ.get("LOCALLIFE_DEPOSIT_SETTLE_S", "1.0"))   # stillness in the event region
CANDIDATE_TIMEOUT_S = float(os.environ.get("LOCALLIFE_DEPOSIT_TIMEOUT_S", "30"))  # never settles -> rejected
FINALISE_S = 12.0              # late measurements update an event this long; then its CSV row is written
PIXEL_DIFF = 28                # grey-level change counted as changed
MOTION_ENTER = 0.02            # changed fraction between frames that opens a candidate
MOTION_STILL = 0.008           # below this the frame is still
MIN_CHANGE = 0.006             # persistent changed fraction of the region worth evaluating
MIN_RISE_M = 0.03              # local depth rise counted as new material
ID_SWITCH_IOU = 0.3            # an unknown id overlapping a committed box this much is a known bag
GROWTH = 1.35                  # a known box grown this much can hold a touching new bag
# Stereo depth error grows with distance squared (D435 RMS roughly 2-3 mm x z^2 per metre^2); a
# per-frame comparison at 3 m is otherwise dominated by noise. Depth changes are judged against
# max(MIN_RISE_M, DEPTH_NOISE_K x z^2) -- about 3 sigma of that engineering model, not a fitted value.
DEPTH_NOISE_K = 0.0075
LOCAL_MARGIN_PX = 6            # settling is judged in the changed area grown by this many SMALL pixels
MIN_LOCAL_SHARE = 0.01         # ...when that area is at least this share of the region

BAG_WORDS = {"bag", "sack", "pillow", "cushion", "textile", "fabric", "garbage", "waste", "trash", "refuse",
             "rubbish", "liner", "parcel", "package", "packet", "box", "carton", "blanket", "clothing", "bin"}
NOT_BAG_WORDS = {"person", "hand", "arm", "finger", "human", "head", "face", "leg", "foot"}

MAX_RISE_MARGIN_CM = 20.0
RESUME_MAX_GAP_S = float(os.environ.get("LOCALLIFE_SESSION_RESUME_MAX_GAP_S", "7200"))
MAX_DELTA_OCCUPANCY_L = 120.0     # larger than any single bag deposit into a 660 L bin
ENVELOPE_LABEL = "new-bag outer envelope (visible L x W x added height box)"
DELTA_LABEL = "net before/after change in bin occupancy (whole-bin surface, separate method)"

CSV_FIELDS = (
    "session_id", "camera", "event_id", "counted", "count_after", "deposit_time", "entered_at", "confirmed_at",
    "cameras", "association", "confidence", "evidence",
    "track_id", "colour", "object_type", "detector_label", "material", "length_cm", "width_cm", "height_cm",
    "height_source", "envelope_l", "delta_occupancy_l", "measurement_status", "reason",
    "volume_method", "units", "bin_fill_pct_after", "bin_fill_litres_after",
    "bin_fill_pct_before", "bin_fill_litres_before", "sorting", "sorting_reason", "object_class",
    "object_name", "visible_material", "visible_material_source", "colour_secondary",
)


FURNITURE_WORDS = {"chair", "sofa", "couch", "bed", "table", "desk", "door", "wall", "floor", "cabinet"}


def bag_like(label: str | None) -> bool:
    """Anything that can be thrown into a bin: bags, but also parcels and boxes the detector
    names 'book', 'rigid household object', 'packaging'. A box labelled 'book' had been left
    unlinked, so its count showed no size, colour or material. Never a person or furniture."""
    words = set((label or "").lower().replace("-", " ").replace("_", " ").replace("[", " ").replace("]", " ").split())
    return bool(words) and not (words & NOT_BAG_WORDS) and not (words & FURNITURE_WORDS)


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
    support_volume_l: float | None = None           # volume above the local surface (median of this track)
    support_height_cm: float | None = None
    support_length_cm: float | None = None          # oriented L x W of the same region (median of the track)
    support_width_cm: float | None = None
    support_method: str | None = None               # how the support L/W/H/volume set was measured
    object_class: str | None = None                 # class resolved over the track; `label` stays raw
    object_name: str | None = None                  # V51 descriptive name (recognition.py)
    visible_material: str | None = None             # V51 visible exterior material vocabulary
    visible_material_source: str | None = None
    colour_secondary: tuple = ()


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
    started_at: float | None = None                 # when this frame's processing (inference) began


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel change: the largest per-channel difference (a red bag can match the grey level).

    The median difference is removed first, per channel: an auto-exposure or
    lighting step shifts the whole view and is not an object.
    """
    diff = a.astype(np.int16) - b.astype(np.int16)
    offset = np.median(diff.reshape(-1, diff.shape[-1]) if diff.ndim == 3 else diff.ravel(), axis=0)
    diff = np.abs(diff - offset.astype(np.int16))
    return (diff.max(axis=-1) if diff.ndim == 3 else diff) > PIXEL_DIFF


def _depth_threshold(depth: np.ndarray) -> np.ndarray:
    return np.maximum(MIN_RISE_M, DEPTH_NOISE_K * depth.astype(np.float64) ** 2)


def _grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return mask
    size = 2 * pixels + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


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
        self.max_rise_m: float | None = None      # set by SessionDeposits from the bin's usable height
        self.state = "initialising"
        self.reason: str | None = "waiting for a stable start-up view"
        self.prev_grey: np.ndarray | None = None
        self.still_since: float | None = None
        self.committed: FrameEvidence | None = None
        self.known_ids: set[int] = set()
        self.baseline_ids: set[int] = set()
        self.candidate_since: float | None = None
        self.last_candidate_since: float | None = None
        self.candidate_trigger: str | None = None
        self.motion = 0.0
        self.motion_scope = "region"
        self.still_limit = MOTION_STILL
        self.last_frame_at: float | None = None
        self.resync = False
        self.resync_since: float | None = None
        self.votes: dict[int, dict[str, Counter]] = {}
        self.noise = 0.0                    # typical frame-to-frame change of a still scene (sensor/JPEG)
        self.views: list[np.ndarray] = []   # last few small views, for colour voted over several frames
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
            self.motion_scope = "region"
            if self.state in ("candidate", "settling") and self.committed is not None \
                    and self.committed.grey.shape == ev.grey.shape:
                # Settling is about the candidate and the scene around it, not the whole view: in a
                # room, someone moving at the far wall kept a placed bag "unsettled" for 30 s.
                # Persistent change only (present in every recent frame): a placed object stays, a
                # person walking through does not, so the walker is not part of the candidate area.
                local = region.copy()
                for view in [*self.views[-3:], ev.grey]:
                    if view.shape == self.committed.grey.shape:
                        local &= _changed(view, self.committed.grey)
                for t in ev.tracks:
                    if bag_like(t.label) and not self._known(t):
                        local |= _box_mask(t.box, region.shape) & region
                local = _grow(local, LOCAL_MARGIN_PX) & region
                if np.count_nonzero(local) >= MIN_LOCAL_SHARE * np.count_nonzero(region):
                    self.motion = float(np.count_nonzero(diff & local)) / int(np.count_nonzero(local))
                    self.motion_scope = "candidate area"
        self.prev_grey = ev.grey
        self.views = (self.views + [ev.grey])[-4:]
        # Thresholds follow this camera's own noise floor, so a noisy dark view still settles.
        if self.motion < max(MOTION_ENTER, 4 * self.noise):
            self.noise = 0.9 * self.noise + 0.1 * self.motion
        still_limit = max(MOTION_STILL, 2.5 * self.noise)
        self.still_limit = still_limit
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
            unknown = [t for t in ev.tracks if bag_like(t.label) and not self._known(t)]
            if self.resync:
                # After a timeout: re-baseline at the next still moment, or after 15 s regardless.
                # Waiting only for stillness froze the counter while people kept moving.
                self.resync_since = self.resync_since or now
                if settled or now - self.resync_since > 15.0:
                    self._commit(ev)
                    self.resync, self.resync_since = False, None
                if not unknown:
                    return None
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
            # People keep moving around the bin, so the scene may never be still. A NEW bag-like
            # track over a persistent change is strong enough to count even then; motion alone is not.
            outcome = self._evaluate(ev)
            if outcome[0] == "confirmed" and outcome[1].get("track") is not None:
                outcome[1]["evidence"] += " (confirmed at the settle timeout; scene still moving)"
                self._commit(ev)
                self._end(None)
                return outcome
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
        self.last_candidate_since = self.candidate_since
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
            # A vacated spot is ONE coherent area whose surface fell by more than the depth noise at
            # that range, away from the new object. Counting every pixel that fell >= 3 cm anywhere
            # in the view let far-wall stereo noise (several cm at 3 m) "vacate" a bag that was
            # simply placed, rejecting the deposit.
            dropped = valid & (ev.depth - before.depth >= _depth_threshold(before.depth)) & region
            dropped &= ~_grow(added, 2)
            try:
                import cv2
                count, _, stats, _ = cv2.connectedComponentsWithStats(dropped.astype(np.uint8), 8)
                largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
            except ImportError:  # pragma: no cover
                largest = int(np.count_nonzero(dropped))
            return largest >= 0.5 * size
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

    def _removed(self, ev: FrameEvidence, before: FrameEvidence, area_mask: np.ndarray,
                 change: np.ndarray | None = None) -> bool:
        """Was something TAKEN OUT here? Lifting the top bag reveals the one below: not a new deposit."""
        if ev.heights is not None and before.heights is not None and ev.heights.shape == before.heights.shape:
            full = ev.heights - before.heights
            diff = full[area_mask]
            diff = diff[np.isfinite(diff)]
            if diff.size >= 10:
                offset = 0.0
                if ev.depth is None:
                    # Monocular heights (Logitech, no hardware depth): the model's scale re-normalises
                    # when a large object enters, shifting the WHOLE map. A box laid on the pile read
                    # as a 4 cm "drop" and was rejected as a removal. Measure the change relative to
                    # the unchanged part of the bin instead.
                    still = np.isfinite(full) & ev.region & ~area_mask
                    if change is not None and change.shape == still.shape:
                        still &= ~change
                    if np.count_nonzero(still) >= 50:
                        offset = float(np.median(full[still]))
                return float(np.median(diff)) - offset < -0.04
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
            change |= (rise >= _depth_threshold(before.depth)) & region
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
                if self._removed(ev, before, added, change):
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
                risen = valid & (rise >= _depth_threshold(before.depth))
                if np.count_nonzero(risen) < 0.35 * np.count_nonzero(valid):
                    return "rejected", {**details, "reason": "change without a local surface rise where depth "
                                                             "is valid (lighting, shadow or an item shifted)"}
                rest = risen
                # V53: a "rise" taller than the bin is something between camera and bin (lid, hand,
                # person), not a deposit; such events counted as bags and read 219-355 cm tall.
                if (self.max_rise_m is not None and np.count_nonzero(risen)
                        and float(np.median(rise[risen])) > self.max_rise_m):
                    return "rejected", {**details, "reason": "occlusion: the depth rise is taller than the bin "
                                                             "(lid, hand or person in the line of sight)"}
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
        if self._removed(ev, before, rest, change):
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
        self.cross_reference: Any = None    # V53 Phase 2: cross_camera.CrossCameraReference
        self.fill_lookup: Callable[[str], dict[str, Any]] | None = None   # camera -> its current fill reading
        self.fill_history: dict[str, list[tuple[float, float, float]]] = {}  # camera -> (valid at, litres, %)
        self._awaiting_after: list[dict[str, Any]] = []
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
        self.late_frames_ignored = 0
        self._awaiting_after = []
        self.generation = getattr(self, "generation", 0) + 1
        self._pending_csv = []
        self.watchers.clear()
        self._save()

    def _resume(self) -> bool:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            # V53: resume a session only after a short interruption (a restart). A session saved many
            # hours ago belongs to another test: on the bin rig, yesterday's room tests (backpack,
            # headphones) were still counted as "new bags" the next day.
            last_activity = float(data.get("saved_at") or 0.0) or self.state_path.stat().st_mtime
            if self.clock() - last_activity > RESUME_MAX_GAP_S:
                LOGGER.info("deposit session %s not resumed: last activity %.1f h ago", data.get("session_id"),
                            (self.clock() - last_activity) / 3600.0)
                return False
            self.session_id, self.session_started_at = data["session_id"], float(data["session_started_at"])
            self.events, self.rejected = list(data["events"]), list(data.get("rejected", []))
            # Records saved by an earlier version lack newer fields: fill them, never crash the panel.
            for record in self.events + self.rejected:
                cameras = record.get("cameras") or [record.get("camera") or "unknown"]
                record.setdefault("cameras", cameras)
                record.setdefault("camera", cameras[0])
                for key in ("colour", "object_type", "detector_label", "material", "confidence",
                            "measurement_status", "association", "reason", "length_cm", "width_cm",
                            "height_cm", "height_source", "envelope_l", "delta_occupancy_l", "track_id",
                            "object_name", "visible_material", "visible_material_source", "colour_secondary"):
                    record.setdefault(key, None)
                record.setdefault("evidence", {})
            self.written = set(data.get("written", []))
            self.last_confirmed_at = data.get("last_confirmed_at")
            self.ignored_redetections = 0
            self.late_frames_ignored = 0
            self.generation = 1
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
                "last_confirmed_at": self.last_confirmed_at, "saved_at": self.clock()}, default=str),
                encoding="utf-8")
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
            if (ev.started_at if ev.started_at is not None else ev.timestamp) < self.session_started_at:
                # Captured before the last reset (in-flight inference): never part of the new session.
                self.late_frames_ignored += 1
                return None
            self._last_frame_at = max(self._last_frame_at, ev.timestamp)
            watcher = self.watchers.get(ev.camera)
            if watcher is None:
                # The warm-up starts at this camera's first frame (cameras may connect late,
                # and a resumed session re-takes its baseline: track ids do not survive a restart).
                watcher = self.watchers[ev.camera] = CameraWatcher(ev.camera, ev.timestamp)
            watcher.max_rise_m = self._max_rise_cm(ev.camera) / 100.0
            outcome = watcher.observe(ev)
            if watcher.pending:
                self._late_sizes(ev, watcher)
            self._track_fill(ev.camera)
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
        mask = details.get("mask")
        if track is None and mask is not None and mask.any():
            # The change was confirmed without a NEW track id (re-used id, merged mask): link it to the
            # detection that covers it, so colour/material/size come from the deposited bag itself.
            # Prefer the track that FITS the change (overlap / union), and one that was not in the
            # committed scene: a big old bag box covering the spot is the neighbour, not the deposit.
            old_ids = set() if watcher.committed is None else {t.track_id for t in watcher.committed.tracks}
            best, best_share, best_key = None, 0.0, (-1, -1.0)
            mask_area = np.count_nonzero(mask)
            for t in ev.tracks:
                if not bag_like(t.label):
                    continue
                box = _box_mask(t.box, mask.shape)
                inter = float(np.count_nonzero(mask & box))
                share = inter / mask_area
                fit = inter / max(1.0, float(np.count_nonzero(mask | box)))
                key = (int(share >= 0.4 and t.track_id not in old_ids), fit)
                if share >= 0.4 and key > best_key:
                    best, best_share, best_key = t, share, key
            if best is not None and best_share >= 0.4:
                track = details["track"] = best
                details["evidence"] += f" (linked to track {best.track_id})"
        record = self._measure(ev, watcher, details, track)
        record["entered_at"] = watcher.last_candidate_since
        record["confirmed_at"] = record["deposit_time"]
        if record["colour"] in (None, "unknown") and mask is not None and mask.any():
            record["colour"] = _voted_colour(watcher.views, mask)
        # Bin volume BEFORE: this camera's last valid fill reading from before the bag entered.
        # AFTER: its first valid reading once the bag has settled (filled in on later frames).
        entered = record["entered_at"] or record["deposit_time"]
        before = [h for h in self.fill_history.get(ev.camera, []) if h[0] <= entered]
        if before:
            record["bin_fill_litres_before"], record["bin_fill_pct_before"] = before[-1][1], before[-1][2]
        self._awaiting_after.append(record)
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

    def _track_fill(self, camera: str) -> None:
        if self.fill_lookup is None:
            return
        reading = self.fill_lookup(camera) or {}
        valid_at = reading.get("updated_at")
        if reading.get("status") != "ok" or reading.get("stale") or valid_at is None:
            return
        history = self.fill_history.setdefault(camera, [])
        if not history or valid_at > history[-1][0]:
            history.append((float(valid_at), reading.get("rough_litres"), reading.get("height_fill_pct")))
            del history[:-300]
            changed = False
            for record in list(self._awaiting_after):
                if record["camera"] == camera and valid_at > record["deposit_time"]:
                    record["bin_fill_litres_after"] = reading.get("rough_litres")
                    record["bin_fill_pct_after"] = reading.get("height_fill_pct")
                    self._awaiting_after.remove(record)
                    changed = True
            if changed:
                self._save()

    def _late_sizes(self, ev: FrameEvidence, watcher: CameraWatcher) -> None:
        keep = []
        changed = False
        for record, track_id, until in watcher.pending:
            if ev.timestamp > until:
                continue
            track = next((t for t in ev.tracks if t.track_id == track_id), None)
            if track is not None:
                changed = _refresh_from_track(record, track) or changed
                if _coherent_support(track) is not None:
                    keep.append((record, track_id, until))    # keep following until the window closes
                    continue
            if track is not None and track.support_height_cm and (
                    record["height_cm"] is None or str(record.get("height_source") or "").startswith("detector")):
                record["height_cm"] = _r(float(track.support_height_cm))
                record["height_source"] = "height above the local surface around the bag"
                changed = True
            if track is not None and track.support_length_cm and track.support_width_cm and (
                    record["length_cm"] is None or record.get("volume_method") != "surface rise integrated over the bag"):
                record["length_cm"], record["width_cm"] = _r(float(track.support_length_cm)), _r(float(track.support_width_cm))
                changed = True
            if track is not None and track.support_volume_l and record.get("volume_method") != "surface rise integrated over the bag":
                record["envelope_l"] = round(float(track.support_volume_l), 1)
                record["volume_method"] = "volume above the local surface (median over the track)"
                if record["measurement_status"] in ("na", "partial"):
                    record["measurement_status"] = "approximate"
                changed = True
            if track is not None and not track.rejection and track.length_mm and track.width_mm:
                if record["length_cm"] is None:
                    record["length_cm"], record["width_cm"] = _r(track.length_mm / 10), _r(track.width_mm / 10)
                    changed = True
                if record["height_cm"] is None and track.height_mm:
                    record["height_cm"] = _r(track.height_mm / 10)
                    record["height_source"] = "detector height from the bin floor (may include the pile)"
                    changed = True
                if record["length_cm"] and record["width_cm"] and record["height_cm"]:
                    if not (record.get("volume_method") or "").startswith(("volume above", "surface rise")):
                        record["envelope_l"] = round(record["length_cm"] * record["width_cm"] * record["height_cm"]
                                                     / 1000.0, 1)
                        record["volume_method"] = "L x W x H box"
                    if record["measurement_status"] in ("na", "partial"):
                        record["measurement_status"] = "approximate"
                        record["reason"] = "size added after the count (same event)"
                    # complete, but class/colour/material may still resolve: follow to the window end
            keep.append((record, track_id, until))
        watcher.pending = keep
        if changed:
            self._save()

    def _measure(self, ev, watcher, details, track) -> dict[str, Any]:
        reasons: list[str] = []
        length = width = height = None
        source = None
        coherent = _coherent_support(track)
        if track is not None and track.support_length_cm and track.support_width_cm:
            # L x W of the SAME risen region the volume and height come from (oriented rectangle).
            length, width = float(track.support_length_cm), float(track.support_width_cm)
        elif track is not None and track.rejection:
            reasons.append(f"detector dimensions withheld: {track.rejection}")
        elif track is not None and track.length_mm and track.width_mm:
            length, width = track.length_mm / 10.0, track.width_mm / 10.0
        elif track is None:
            reasons.append("no single tracked bag to take L x W from")
        mask = details.get("mask")
        before = watcher.committed
        added = None
        if coherent is None and mask is not None and mask.any():
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
        if coherent is None and mask is not None and mask.any() and ev.heights is not None and before is not None \
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
        if coherent is not None:
            # L, W, H and volume of ONE measurement of the track's own mask above its local support.
            length, width, height, volume, source = coherent
        elif added is not None:
            height = added * 100.0
        elif track is not None and track.support_height_cm:
            # Above the surface the bag lies on. The detector's own height is measured from the BIN
            # FLOOR and includes the pile under the bag (a bag read 77 cm tall).
            height, source = float(track.support_height_cm), "height above the local surface around the bag"
        elif track is not None and track.height_mm and not track.rejection:
            height, source = track.height_mm / 10.0, "detector height from the bin floor (may include the pile)"
        else:
            reasons.append("no reliable before/after surface under the bag (hidden, occluded or no depth)")
        if volume is None and track is not None and track.support_volume_l:
            volume = float(track.support_volume_l)          # surface-map integral, median over the track
        # V53 plausibility: nothing deposited can stand higher than the bin's usable height (+ rim).
        # A depth "rise" along the line of sight of 2-3.5 m came from a hand, a person or the lid
        # between camera and bin (live Phase 2 ledger: 219.8, 316.6, 355.4 cm). Reported, not clipped.
        limit_cm = self._max_rise_cm(ev.camera)
        if height is not None and height > limit_cm:
            reasons.append(f"implausible {height:.0f} cm rise (> {limit_cm:.0f} cm bin height): an occlusion "
                           "between camera and bin, not waste")
            height, source = None, None
            if volume is not None and coherent is None:
                volume = None
        envelope = volume if volume is not None else (
            round(length * width * height / 1000.0, 1) if length and width and height else None)
        if envelope is None:
            status = "partial" if (length or height) else "na"
        elif source and source.startswith("after top") and "measured" in source:
            status = "measured"
        else:
            status = "approximate"
        track_id = None if track is None else track.track_id
        label = "unknown" if track is None else track.label          # raw detector class
        evidence = details["evidence"]
        confidence = ("high" if track is not None and "rise" in evidence or (track is not None and ev.depth is not None)
                      else "medium" if track is not None or "depth rise" in evidence else "low (foreground only)")
        material = _track_material(track) or watcher.consensus(track_id, "material", "UNKNOWN")
        return {
            "session_id": self.session_id, "event_id": None, "counted": True, "camera": ev.camera,
            "confidence": confidence,
            "count_after": None, "deposit_time": float(ev.timestamp), "cameras": [ev.camera],
            "association": "single camera", "track_id": track_id,
            # The track's own colour consensus -- the value the live table shows -- not a one-frame fallback.
            "colour": track.colour if track is not None and track.colour not in (None, "", "unknown")
            else watcher.consensus(track_id, "colour", "unknown"),
            "object_class": None if track is None else track.object_class,
            **_descriptive(track),
            "object_type": ("unclassified (image change only)" if track is None
                            else track.object_class or f"unresolved (detector: {label})"),
            "detector_label": label, "material": material.upper() if material != "UNKNOWN" else material,
            "length_cm": _r(length), "width_cm": _r(width), "height_cm": _r(height), "height_source": source,
            "envelope_l": envelope, "envelope_label": ENVELOPE_LABEL,
            "delta_occupancy_l": None, "delta_label": DELTA_LABEL,
            "measurement_status": status, "reason": "; ".join(reasons) or None,
            "bin_fill_pct_after": None, "bin_fill_litres_after": None,
            "bin_fill_pct_before": None, "bin_fill_litres_before": None,
            "volume_method": (coherent[4] if coherent is not None
                              else "surface rise integrated over the bag" if volume is not None and added is not None
                              else "volume above the local surface (median over the track)" if volume is not None
                              else "L x W x H box" if envelope is not None else None),
            "units": "cm, L",
            "evidence": {ev.camera: {"evidence": details["evidence"], "trigger": details.get("trigger"),
                                     "changed_fraction": details.get("changed_fraction")}},
        }

    def _max_rise_cm(self, camera: str) -> float:
        reading = {}
        if self.fill_lookup is not None:
            try:
                reading = self.fill_lookup(camera) or {}
            except Exception:  # noqa: BLE001 - lookup is advisory
                reading = {}
        usable = reading.get("usable_height_cm") or 100.0
        return float(usable) + MAX_RISE_MARGIN_CM

    def attach_occupancy(self, camera: str, event: Any, extra: dict[str, Any] | None = None) -> None:
        """A whole-bin before/after occupancy event is EVIDENCE only; it never counts on its own."""
        with self._lock:
            near = [e for e in self.events if e["camera"] == camera and e["delta_occupancy_l"] is None
                    and event.started_at - FINALISE_S <= e["deposit_time"] <= event.finalized_at + FINALISE_S]
            if event.started_at < self.session_started_at:
                return                                   # an occupancy event from before the reset
            if near and event.delta_occupancy_l is not None:
                delta = float(event.delta_occupancy_l)
                record = near[-1]
                # V53: a deposit cannot change the bin by more than a large item; Logitech before/after
                # frames with a different monocular scale gave -542 L and +440 L. A decrease is existing
                # waste moving, compressing or leaving -- never a negative bag volume.
                if abs(delta) > MAX_DELTA_OCCUPANCY_L:
                    record["delta_occupancy_note"] = (f"rejected {delta:+.0f} L: before/after surfaces inconsistent "
                                                      "(depth scale changed or the view was blocked)")
                else:
                    record["delta_occupancy_l"] = round(delta, 3)
                    if delta < 0:
                        record["delta_occupancy_note"] = ("occupancy decreased: existing waste moved, compressed or "
                                                          "was removed; not a bag volume")
                self._save()

    # ----------------------------------------------------------------- csv
    def _flush_csv(self, force: bool = False) -> None:
        now = max(self.clock(), self._last_frame_at)      # frame time: the clock the events use
        keep = []
        for record in self._pending_csv:
            waiting_after = (self.fill_lookup is not None and record.get("bin_fill_litres_after") is None
                             and now - record["deposit_time"] <= 45.0)
            if force or (now - record["deposit_time"] > FINALISE_S and not waiting_after):
                self._write_csv(record)
            else:
                keep.append(record)
        if len(keep) != len(self._pending_csv):
            self._pending_csv = keep
            self._save()

    def _write_csv(self, record: dict[str, Any]) -> None:
        if record["event_id"] in self.written:
            return
        if record.get("counted") and record.get("camera") == "logitech" and self.cross_reference is not None:
            try:
                self.cross_reference.on_logitech_event(record, self.events)
            except Exception:  # noqa: BLE001 - a reference is advisory; the raw row is still written
                LOGGER.exception("cross-camera reference failed")
        if record.get("counted"):
            apply_to_event(record)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.csv_path.exists():
                with self.csv_path.open(encoding="utf-8") as handle:
                    header = handle.readline().strip().split(",")
                if header != list(CSV_FIELDS):        # older schema: keep it under a dated name
                    self.csv_path.rename(self.csv_path.with_name(
                        f"session_deposits_{time.strftime('%Y%m%d-%H%M%S')}_old_schema.csv"))
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
                        "motion_scope": w.motion_scope, "still_limit": round(w.still_limit, 4),
                        "still_for_s": None if w.still_since is None else round(max(0.0, now - w.still_since), 1),
                        "settle_needed_s": SETTLE_S, "timeout_s": CANDIDATE_TIMEOUT_S,
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
                "generation": self.generation, "late_frames_ignored": self.late_frames_ignored,
                "session_id": self.session_id, "session_started_at": self.session_started_at,
                "elapsed_s": round(now - self.session_started_at, 1), "resumed": self.resumed,
                "updated_at": max(frames) if frames else None, "status": status, "cameras": cams,
                "new_bags_this_session": self.count, "last_confirmed_at": self.last_confirmed_at,
                "events": [apply_to_event({k: v for k, v in e.items()}) for e in self.events],
                "rejected_candidates": len(self.rejected), "ignored_redetections": self.ignored_redetections,
                "recent_rejections": [{"camera": r["cameras"][0], "reason": r["reason"], "at": r["deposit_time"]}
                                      for r in self.rejected[-5:]],
                "envelope_label": ENVELOPE_LABEL, "delta_label": DELTA_LABEL,
            }


def _voted_colour(views: list[np.ndarray], mask: np.ndarray) -> str:
    """Colour of the deposit's own pixels, agreed over the last few frames; UNKNOWN when they disagree."""
    try:
        from .geometry import classify_color
    except ImportError:  # pragma: no cover
        return "unknown"
    names = [classify_color(v, mask)[0] for v in views if v.ndim == 3 and v.shape[:2] == mask.shape]
    names = [n for n in names if n != "unknown"]
    if not names:
        return "unknown"
    top = Counter(names).most_common(1)[0]
    return top[0] if top[1] >= max(2, 0.6 * len(names)) else "unknown"


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _coherent_support(track: "TrackInfo | None"):
    """(L cm, W cm, H cm, litres, method) of one support measurement of the track, or None."""
    if track is None or not (track.support_volume_l and track.support_length_cm and track.support_width_cm
                             and track.support_height_cm):
        return None
    return (float(track.support_length_cm), float(track.support_width_cm), float(track.support_height_cm),
            round(float(track.support_volume_l), 2),
            track.support_method or "volume above the local surface around the object")


def _descriptive(track: "TrackInfo | None") -> dict[str, Any]:
    """V51 descriptive attributes of the SAME track; metadata only, never a count."""
    if track is None:
        return {"object_name": None, "visible_material": "unknown",
                "visible_material_source": "no detection (image change only)", "colour_secondary": ""}
    return {"object_name": track.object_name, "visible_material": track.visible_material or "unknown",
            "visible_material_source": track.visible_material_source,
            "colour_secondary": "/".join(track.colour_secondary or ())}


def _track_material(track: "TrackInfo | None") -> str | None:
    if track is None or not track.material or str(track.material).lower() == "unknown" \
            or float(track.material_confidence or 0.0) < 0.5:
        return None
    return str(track.material).upper()


def _refresh_from_track(record: dict[str, Any], track: "TrackInfo") -> bool:
    """Late, better evidence for the SAME event (never a new count): resolved class, material,
    colour and one coherent L/W/H/volume set, all from this track."""
    before = dict(record)
    if track.object_class:
        record["object_class"] = record["object_type"] = track.object_class
    material = _track_material(track)
    if material:
        record["material"] = material
    if track.colour and track.colour != "unknown":
        record["colour"] = track.colour
    record.update({key: value for key, value in _descriptive(track).items()
                   if value not in (None, "", "unknown", "unknown object") or key not in record})
    coherent = _coherent_support(track)
    if coherent is not None:
        length, width, height, litres, method = coherent
        record.update({"length_cm": _r(length), "width_cm": _r(width), "height_cm": _r(height),
                       "envelope_l": litres, "volume_method": method, "height_source": method})
        if record.get("measurement_status") in ("na", "partial"):
            record["measurement_status"] = "approximate"
    return record != before
