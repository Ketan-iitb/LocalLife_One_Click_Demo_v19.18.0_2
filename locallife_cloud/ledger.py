"""Durable, deduplicated waste-plant observations and deposit summaries."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from .storage import ResultStore
from .types import Detection


LEDGER_NAME = "waste_plant_ledger.jsonl"


def waste_object_type(label: str) -> str:
    words = set(label.lower().replace("_", " ").replace("-", " ").split())
    if words & {"bag", "bags", "sack", "sacks", "tote"}:
        return "bag"
    if words & {"box", "boxes", "carton", "cartons", "parcel", "parcels"}:
        return "box"
    return "other"


class WastePlantLedger:
    """One logical record per tracked object, recovered across service restarts."""

    def __init__(
        self,
        store: ResultStore,
        *,
        color_streams: dict[str, str] | None = None,
        history_limit: int = 100,
        camera_id: str = "realsense",
    ) -> None:
        self.store = store
        self.color_streams = dict(color_streams or {})
        self.history_limit = history_limit
        self.camera_id = camera_id
        self.session_id = uuid4().hex[:12]
        self.records: dict[str, dict[str, Any]] = {}
        self.track_entries: dict[int, str] = {}
        for event in self.store.read_jsonl(LEDGER_NAME):
            entry_id = event.get("entry_id")
            record = event.get("record")
            if isinstance(entry_id, str) and isinstance(record, dict):
                self.records[entry_id] = dict(record)

    def observe(self, detection: Detection, *, timestamp: float | None = None) -> dict[str, Any]:
        if detection.track_id is None:
            raise ValueError("Only tracked objects can enter the waste ledger")
        existing_id = self.track_entries.get(detection.track_id)
        if existing_id is not None:
            return self.records[existing_id]
        entry_id = f"{self.session_id}-{detection.track_id}"
        color = detection.color or "unknown"
        record = {
            "entry_id": entry_id,
            "camera_id": self.camera_id,
            "track_id": detection.track_id,
            "object_type": waste_object_type(detection.label),
            "label": detection.label,
            "accepted_class": detection.accepted_class,
            "color": color,
            "material": detection.material or "unknown",
            "material_confidence": detection.material_confidence,
            "waste_stream": self.color_streams.get(color.lower()),
            "volume_l": self._measurement(detection),
            "volume_uncertainty_l": detection.volume_uncertainty_l,
            "depth_coverage_percent": detection.depth_coverage_percent,
            "measurement_method": detection.measurement_method,
            "measurement_quality": detection.measurement_quality,
            "calibration_mode": detection.calibration_mode,
            "confidence": detection.confidence,
            "dimensions_mm": None if detection.footprint_length_mm is None else {
                "footprint_length": detection.footprint_length_mm,
                "footprint_width": detection.footprint_width_mm,
                "height": detection.physical_height_mm,
            },
            "dimension_confidence": detection.dimension_confidence,
            "dimension_flags": list(detection.dimension_flags),
            "dimension_method": detection.dimension_method,
            "observed_at": float(time.time() if timestamp is None else timestamp),
            "deposited_at": None,
            "status": "observed",
        }
        self.records[entry_id] = record
        self.track_entries[detection.track_id] = entry_id
        self._persist("observed", record)
        return record

    def refresh(self, detection: Detection) -> None:
        if detection.track_id is None:
            return
        entry_id = self.track_entries.get(detection.track_id)
        if entry_id is None:
            return
        record = self.records[entry_id]
        if record["status"] == "deposited":
            return
        record.update(
            label=detection.label,
            accepted_class=detection.accepted_class,
            color=detection.color,
            material=detection.material or "unknown",
            material_confidence=detection.material_confidence,
            waste_stream=self.color_streams.get((detection.color or "unknown").lower()),
            volume_l=self._measurement(detection),
            volume_uncertainty_l=detection.volume_uncertainty_l,
            depth_coverage_percent=detection.depth_coverage_percent,
            measurement_method=detection.measurement_method,
            measurement_quality=detection.measurement_quality,
            calibration_mode=detection.calibration_mode,
            confidence=detection.confidence,
            dimensions_mm=None if detection.footprint_length_mm is None else {
                "footprint_length": detection.footprint_length_mm,
                "footprint_width": detection.footprint_width_mm,
                "height": detection.physical_height_mm,
            },
            dimension_confidence=detection.dimension_confidence,
            dimension_flags=list(detection.dimension_flags),
            dimension_method=detection.dimension_method,
        )

    def deposit(self, detection: Detection, *, timestamp: float | None = None) -> dict[str, Any]:
        record = self.observe(detection, timestamp=timestamp)
        if record["status"] == "deposited":
            return record
        self.refresh(detection)
        record["status"] = "deposited"
        record["deposited_at"] = float(time.time() if timestamp is None else timestamp)
        self._persist("deposited", record)
        return record

    def is_deposited(self, track_id: int | None) -> bool:
        if track_id is None:
            return False
        entry_id = self.track_entries.get(track_id)
        return entry_id is not None and self.records[entry_id]["status"] == "deposited"

    def quarantine_implausible(self, maximum_volume_l: float) -> int:
        """Retain invalid historic deposits for audit, but exclude them from totals."""
        quarantined = 0
        for record in list(self.records.values()):
            if record.get("status") != "deposited":
                continue
            try:
                liters = float(record.get("volume_l"))
            except (TypeError, ValueError):
                continue
            if not liters > maximum_volume_l:
                continue
            record["previous_status"] = "deposited"
            record["status"] = "quarantined"
            record["quarantined_at"] = time.time()
            record["quarantine_reason"] = (
                f"Recorded volume {liters:.3f} L exceeds the configured "
                f"{maximum_volume_l:.3f} L maximum for one item"
            )
            self._persist("quarantined", record)
            quarantined += 1
        return quarantined

    def summary(self) -> dict[str, Any]:
        records = self.all_records()
        deposited = [record for record in records if record["status"] == "deposited"]
        color_totals: dict[str, dict[str, Any]] = {}
        material_totals: dict[str, dict[str, Any]] = {}
        stream_totals: dict[str, dict[str, Any]] = {}
        for record in records:
            color = str(record.get("color") or "unknown")
            color_summary = color_totals.setdefault(
                color,
                {"color": color, "observed_count": 0, "deposited_count": 0, "volume_l": 0.0,
                 "waste_stream": self.color_streams.get(color.lower())},
            )
            color_summary["observed_count"] += 1
            material = str(record.get("material") or "unknown")
            material_summary = material_totals.setdefault(
                material,
                {"material": material, "observed_count": 0, "deposited_count": 0, "volume_l": 0.0},
            )
            material_summary["observed_count"] += 1
            if record["status"] == "deposited":
                color_summary["deposited_count"] += 1
                color_summary["volume_l"] += float(record.get("volume_l") or 0.0)
                material_summary["deposited_count"] += 1
                material_summary["volume_l"] += float(record.get("volume_l") or 0.0)
                stream = record.get("waste_stream")
                if stream:
                    item = stream_totals.setdefault(
                        stream, {"waste_stream": stream, "deposited_count": 0, "volume_l": 0.0}
                    )
                    item["deposited_count"] += 1
                    item["volume_l"] += float(record.get("volume_l") or 0.0)

        for item in list(color_totals.values()) + list(material_totals.values()) + list(stream_totals.values()):
            item["volume_l"] = round(float(item["volume_l"]), 6)

        return {
            "camera_id": self.camera_id,
            "observed_count": len(records),
            "observed_bags": sum(record["object_type"] == "bag" for record in records),
            "observed_boxes": sum(record["object_type"] == "box" for record in records),
            "deposited_count": len(deposited),
            "deposited_bags": sum(record["object_type"] == "bag" for record in deposited),
            "deposited_boxes": sum(record["object_type"] == "box" for record in deposited),
            "quarantined_count": sum(record.get("status") == "quarantined" for record in self.records.values()),
            "cumulative_volume_l": round(sum(float(item.get("volume_l") or 0.0) for item in deposited), 6),
            "colors": sorted(color_totals.values(), key=lambda item: (-item["deposited_count"], item["color"])),
            "materials": sorted(
                material_totals.values(), key=lambda item: (-item["deposited_count"], item["material"])
            ),
            "waste_streams": sorted(stream_totals.values(), key=lambda item: item["waste_stream"]),
            "history": records[: self.history_limit],
        }

    def all_records(self, *, include_quarantined: bool = False) -> list[dict[str, Any]]:
        return sorted(
            (
                dict(record) for record in self.records.values()
                if include_quarantined or record.get("status") != "quarantined"
            ),
            key=lambda item: item["observed_at"],
            reverse=True,
        )

    def _persist(self, event: str, record: dict[str, Any]) -> None:
        self.store.append_jsonl(
            LEDGER_NAME,
            {"event": event, "timestamp": time.time(), "entry_id": record["entry_id"], "record": dict(record)},
        )

    def _measurement(self, detection: Detection) -> float | None:
        return detection.monocular_volume_l if self.camera_id == "logitech" else detection.realsense_volume_l
