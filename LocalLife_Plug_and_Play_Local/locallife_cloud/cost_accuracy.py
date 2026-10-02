"""Local vs Cloud: Cost and Accuracy -- an additive REPORT over results that already exist.

Nothing here runs inference, changes a setting or touches the pipelines. Inputs:

* replay benchmark runs (scripts/benchmark_local_cloud.py run): run_summary.json + frames.csv per run,
  with per-frame predictions; accuracy needs an independent reference file (truth CSV) and a local and
  a cloud run over the SAME recorded input (same input_id);
* live run summaries from the durable Local-vs-Cloud store (speed and runtime only; live runs have no
  reference measurements, so their accuracy is never evaluated);
* a cost configuration with editable rates. No price is built in: every unknown stays unknown, and
  rates without an official source URL + retrieval date are labelled "Unverified estimate".

Every number is traceable to a stored measurement or a configured rate; nothing is invented.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

CAMERAS = ("realsense", "logitech")
HOURS_PER_MONTH = 730.0          # Google Cloud's monthly-estimate convention (24 x 365 / 12)

DEFAULT_CONFIG: dict[str, Any] = {
    "currency": "USD",
    "exchange_rate_note": "",
    "cloud": {
        "status": "Unverified estimate",
        "source_url": "",
        "retrieved_on": "",
        "region": "",
        "provisioning": "on-demand",
        "machine_type": "",
        "machine_rate_per_hour": None,
        "gpu_type": "",
        "gpu_count": 1,
        "gpu_rate_per_hour": None,
        "gpu_included_in_machine_rate": None,
        "disk_type": "",
        "disk_gb": None,
        "disk_rate_per_gb_month": None,
        "billable_hours_override": None,
        "other_per_hour": None,
        "billed_cost": None,
        "billed_cost_note": "",
        "notes": "Excludes taxes, discounts, network egress, images and buckets unless entered above.",
    },
    "local": {
        "power_w": None,
        "power_measured": False,
        "tariff_per_kwh": None,
        "hardware_cost": None,
        "hardware_lifetime_hours": None,
    },
    "truth_csv": "",
}

_NUMBER_FIELDS = {
    "cloud": {"machine_rate_per_hour", "gpu_count", "gpu_rate_per_hour", "disk_gb", "disk_rate_per_gb_month",
              "billable_hours_override", "other_per_hour", "billed_cost"},
    "local": {"power_w", "tariff_per_kwh", "hardware_cost", "hardware_lifetime_hours"},
}
_TEXT_FIELDS = {
    "cloud": {"status", "source_url", "retrieved_on", "region", "provisioning", "machine_type", "gpu_type",
              "disk_type", "billed_cost_note", "notes"},
    "local": set(),
}
_BOOL_FIELDS = {"cloud": {"gpu_included_in_machine_rate"}, "local": {"power_measured"}}


# ------------------------------------------------------------------ config
def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clean_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Only known fields, typed; anything else is dropped (nothing secret is ever stored here)."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    raw = raw or {}
    for top in ("currency", "exchange_rate_note", "truth_csv"):
        if isinstance(raw.get(top), str):
            config[top] = raw[top].strip()[:300]
    for section in ("cloud", "local"):
        given = raw.get(section) or {}
        for key in _NUMBER_FIELDS[section]:
            if key in given:
                value = _num(given[key])
                config[section][key] = value if value is None or value >= 0 else None
        for key in _TEXT_FIELDS[section]:
            if isinstance(given.get(key), str):
                config[section][key] = given[key].strip()[:300]
        for key in _BOOL_FIELDS[section]:
            if key in given:
                value = _bool(given[key])
                config[section][key] = bool(value) if section == "local" else value
    cloud = config["cloud"]
    verified = bool(cloud["source_url"].startswith("https://cloud.google.com")
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cloud["retrieved_on"] or ""))
    cloud["status"] = "Official-rate estimate" if verified else "Unverified estimate"
    return config


def load_config(path: Path) -> dict[str, Any]:
    try:
        return clean_config(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return clean_config(None)


def save_config(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    config = clean_config(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp.replace(path)
    return config


def configured_resources(gpu_script: Path) -> dict[str, Any]:
    """What gpu.py is configured to create (read-only text parse; the live VM is not queried)."""
    try:
        text = gpu_script.read_text(encoding="utf-8")
    except OSError:
        return {"source": str(gpu_script), "available": False}

    def grab(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None
    return {
        "source": "gpu.py (configuration, not the live VM)", "available": True,
        "l4_machine_types": re.findall(r'"(g2-standard-\d+)"', grab(r"L4_SHAPES = \[([^\]]*)\]") or ""),
        "t4_machine_type": grab(r'T4_SHAPE = "([^"]+)"'),
        "disk_gb": _num(grab(r'DISK_GB = "(\d+)"')),
        "disk_type": grab(r'"--boot-disk-type", "([^"]+)"'),
        "provisioning": "on-demand (standard) by default; Spot with `gpu.py up --spot`",
        "zones": "EU zones by default (US only with --us)",
        "stopped_vm_note": "a stopped VM still bills its persistent disk",
    }


# ------------------------------------------------------------------ cost
def cloud_compute_rate(cloud: dict[str, Any]) -> tuple[float | None, str]:
    machine = cloud.get("machine_rate_per_hour")
    gpu = cloud.get("gpu_rate_per_hour")
    count = cloud.get("gpu_count") or 0
    included = cloud.get("gpu_included_in_machine_rate")
    if machine is None:
        return None, "machine rate per hour not entered"
    if included is True:
        return machine, "machine rate (GPU included in it: GPU not added again)"
    if included is None:
        if gpu is None:
            return None, "say whether the GPU is included in the machine rate"
        return None, "GPU rate entered but 'GPU included in machine rate' not set (would risk double counting)"
    if gpu is None:
        return None, "GPU billed separately but its rate is not entered"
    return machine + gpu * count, f"machine rate + {count:g} x GPU rate"


def cloud_cost(cloud: dict[str, Any], runtime_s: float | None) -> dict[str, Any]:
    hours_measured = None if runtime_s is None else runtime_s / 3600.0
    override = cloud.get("billable_hours_override")
    hours = override if override is not None else hours_measured
    basis = ("entered billable VM hours" if override is not None
             else "evaluated runtime only (NOT total VM uptime: idle/boot time excluded)")
    rate, rate_note = cloud_compute_rate(cloud)
    compute = None if rate is None or hours is None else rate * hours
    disk = None
    if cloud.get("disk_gb") is not None and cloud.get("disk_rate_per_gb_month") is not None and hours is not None:
        disk = cloud["disk_gb"] * cloud["disk_rate_per_gb_month"] * hours / HOURS_PER_MONTH
    other = None if cloud.get("other_per_hour") is None or hours is None else cloud["other_per_hour"] * hours
    estimate = None if compute is None else compute + (disk or 0.0) + (other or 0.0)
    billed = cloud.get("billed_cost")
    return {
        "hours": hours, "hours_basis": basis, "compute_rate_per_hour": rate, "compute_rate_note": rate_note,
        "compute": compute, "storage": disk,
        "storage_basis": (f"{cloud.get('disk_gb'):g} GB x rate per GB-month x run hours / {HOURS_PER_MONTH:g} h"
                          if disk is not None else "disk size or rate not entered"),
        "other": other, "total": billed if billed is not None else estimate,
        "total_kind": "actual billed cost (entered)" if billed is not None else (
            f"{cloud.get('status')}" if estimate is not None else "unknown"),
        "estimate": estimate,
        "excluded": "taxes, discounts, egress, images, bucket, and disk billed while the VM is stopped "
                    "(that accrues per month regardless of runs)",
    }


def local_cost(local: dict[str, Any], runtime_s: float | None) -> dict[str, Any]:
    hours = None if runtime_s is None else runtime_s / 3600.0
    power, tariff = local.get("power_w"), local.get("tariff_per_kwh")
    energy = None if power is None or tariff is None or hours is None else power / 1000.0 * hours * tariff
    hardware = None
    if local.get("hardware_cost") is not None and local.get("hardware_lifetime_hours") and hours is not None:
        hardware = local["hardware_cost"] * hours / local["hardware_lifetime_hours"]
    return {
        "hours": hours, "hours_basis": "evaluated runtime", "energy": energy,
        "energy_basis": ("unknown: enter power (W) and tariff per kWh" if energy is None else
                         f"{power:g} W ({'measured' if local.get('power_measured') else 'assumed'}) / 1000 x h x tariff"),
        "hardware_allocated": hardware,
        "hardware_basis": ("not entered" if hardware is None
                           else "purchase cost x run hours / assumed lifetime operating hours (shown separately)"),
        "total": energy, "total_kind": "energy estimate (hardware amortisation separate)" if energy is not None else "unknown",
    }


def per_thousand(total: float | None, frames: float | None) -> float | None:
    if total is None or not frames or frames <= 0:
        return None
    return total / frames * 1000.0


# ------------------------------------------------------------------ runs
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def load_benchmark_runs(directories: Iterable[Path]) -> list[dict[str, Any]]:
    """One entry per (benchmark run, camera), from stored files only."""
    entries, seen = [], set()
    for directory in directories:
        if not directory or not Path(directory).is_dir():
            continue
        for summary_path in sorted(Path(directory).glob("*/run_summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                with (summary_path.parent / "frames.csv").open(newline="", encoding="utf-8") as handle:
                    frames = list(csv.DictReader(handle))
            except (OSError, ValueError):
                continue
            meta = summary.get("metadata") or {}
            run_id = str(meta.get("run_id") or summary_path.parent.name)
            if run_id in seen:
                continue
            seen.add(run_id)
            for camera in CAMERAS:
                rows = [f for f in frames if f.get("camera_id") == camera and str(f.get("warmup")).lower() not in ("true", "1")]
                if not rows:
                    continue
                done = [f for f in rows if f.get("status") == "ok"]
                e2e = [v for f in done if (v := _num(f.get("e2e_ms"))) is not None]
                runtime_s = sum(e2e) / 1000.0 if e2e else None
                entries.append({
                    "key": f"bench:{run_id}:{camera}", "kind": "replay benchmark", "run_id": run_id,
                    "mode": meta.get("server_processing_mode"), "camera": camera,
                    "input_id": meta.get("input_id"), "detector": meta.get("detector_model"),
                    "depth_model": meta.get("depth_model"), "truth_file": meta.get("truth_file"),
                    "gpu": (meta.get("server_gpu") or {}).get("name") if isinstance(meta.get("server_gpu"), dict) else None,
                    "started_wall": meta.get("started_wall"),
                    "sent": len(rows), "processed": len(done), "failed": len(rows) - len(done),
                    "runtime_s": runtime_s,
                    "runtime_basis": "sum of per-frame send->result time (sequential replay)",
                    "fps": (len(done) / runtime_s) if runtime_s else None,
                    "latency_p50_ms": _pct(e2e, 0.5), "latency_p95_ms": _pct(e2e, 0.95),
                    "latency_boundary": "client send -> result received (replay)",
                    "predictions": {f.get("frame_file"): {
                        "status": f.get("status"),
                        "detected": (_num(f.get("detections")) or 0) > 0,
                        "volume_l": _num(f.get("pred_volume_l"))} for f in rows},
                })
    return entries


def live_runs(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Speed/runtime of live runs from the durable store. No predictions: accuracy never evaluated."""
    entries = []
    for s in summaries:
        camera = s.get("camera_id")
        if camera not in CAMERAS or s.get("processing_mode") not in ("local", "cloud"):
            continue
        runtime = _num(s.get("frames_window_s")) or _num(s.get("duration_s"))
        processed = _num(s.get("completed_unique"))
        entries.append({
            "key": f"live:{s.get('source_host')}|{s.get('run_id')}:{camera}", "kind": "live run",
            "run_id": s.get("run_id"), "mode": s.get("processing_mode"), "camera": camera,
            "input_id": s.get("input_id"), "detector": s.get("detector"), "depth_model": s.get("depth_model"),
            "truth_file": None, "gpu": s.get("gpu_name"), "started_wall": _num(s.get("started_wall")),
            "sent": _num(s.get("received")), "processed": processed, "failed": _num(s.get("failed")),
            "runtime_s": runtime, "runtime_basis": "observation window of completed frames (server clock)",
            "fps": _num(s.get("fps")), "latency_p50_ms": _num(s.get("latency_p50_ms")),
            "latency_p95_ms": _num(s.get("latency_p95_ms")),
            "latency_boundary": "server receipt -> result ready (live)", "predictions": None,
        })
    return entries


# ------------------------------------------------------------------ accuracy
def load_truth(path: str | None) -> tuple[dict[str, dict[str, Any]] | None, str]:
    if not path:
        return None, "no reference (truth) CSV configured"
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = {r["frame"]: r for r in csv.DictReader(handle) if r.get("frame")}
    except (OSError, KeyError, ValueError) as exc:
        return None, f"reference CSV not readable ({type(exc).__name__})"
    return (rows, "") if rows else (None, "reference CSV has no rows")


def accuracy(entry: dict[str, Any], frames: list[str], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Volume and detection accuracy of ONE run on the paired frames. Missing stays missing."""
    preds = entry["predictions"] or {}
    eligible = [f for f in frames if _num(truth[f].get("volume_l")) is not None]
    valid = [f for f in eligible if preds.get(f, {}).get("volume_l") is not None]
    abs_err = [abs(preds[f]["volume_l"] - _num(truth[f]["volume_l"])) for f in valid]
    positive = [f for f in valid if _num(truth[f]["volume_l"]) > 0]
    pct_err = [abs(preds[f]["volume_l"] - _num(truth[f]["volume_l"])) / _num(truth[f]["volume_l"]) * 100.0
               for f in positive]
    result = {
        "eligible": len(eligible), "valid_volume": len(valid),
        "missing_volume": len(eligible) - len(valid),
        "failed_frames": sum(preds.get(f, {}).get("status") != "ok" for f in frames),
        "availability_pct": (100.0 * len(valid) / len(eligible)) if eligible else None,
        "mae_l": statistics.fmean(abs_err) if abs_err else None,
        "mape_pct": statistics.fmean(pct_err) if pct_err else None, "mape_n": len(pct_err),
        "zero_reference_excluded_from_mape": len(valid) - len(positive),
        "precision": None, "recall": None, "detection_note": "needs an object_present column (0/1) in the reference CSV",
    }
    labelled = [f for f in frames if str(truth[f].get("object_present", "")).strip() in {"0", "1"}]
    if labelled:
        tp = sum(truth[f]["object_present"].strip() == "1" and preds.get(f, {}).get("detected") for f in labelled)
        fp = sum(truth[f]["object_present"].strip() == "0" and preds.get(f, {}).get("detected") for f in labelled)
        fn = sum(truth[f]["object_present"].strip() == "1" and not preds.get(f, {}).get("detected") for f in labelled)
        result.update(precision=(tp / (tp + fp)) if tp + fp else None, recall=(tp / (tp + fn)) if tp + fn else None,
                      detection_note=f"{len(labelled)} labelled frames (TP {tp}, FP {fp}, FN {fn})")
    return result


def pairing(local: dict[str, Any] | None, cloud: dict[str, Any] | None,
            truth: dict[str, dict[str, Any]] | None, truth_reason: str) -> tuple[list[str], list[str]]:
    """(paired frame ids, reasons accuracy is not evaluated)."""
    reasons = []
    if local is None or cloud is None:
        return [], ["select a local and a cloud run of this camera"]
    if local["predictions"] is None or cloud["predictions"] is None:
        reasons.append("live runs carry no per-frame predictions: replay recorded frames with "
                       "scripts/benchmark_local_cloud.py in both modes")
    if not local.get("input_id") or local.get("input_id") != cloud.get("input_id"):
        reasons.append("runs are not over the same recorded input (input_id differs or is unknown)")
    for key, label in (("detector", "detector model"), ("depth_model", "depth model")):
        if local.get(key) != cloud.get(key):
            reasons.append(f"{label} differs ({local.get(key)} vs {cloud.get(key)})")
    if truth is None:
        reasons.append(truth_reason)
    if reasons:
        return [], reasons
    frames = sorted(set(local["predictions"]) & set(cloud["predictions"]) & set(truth))
    if not frames:
        return [], ["no recorded frame appears in both runs and in the reference CSV"]
    return frames, []


# ------------------------------------------------------------------ view
def build_view(camera: str, entries: list[dict[str, Any]], config: dict[str, Any],
               local_key: str | None = None, cloud_key: str | None = None) -> dict[str, Any]:
    camera = camera if camera in CAMERAS else CAMERAS[0]
    mine = [e for e in entries if e["camera"] == camera]
    options = {mode: [{"key": e["key"], "label": f"{e['kind']} {e['run_id']} ({int(e['processed'] or 0)} frames)",
                       "input_id": e.get("input_id")}
                      for e in mine if e["mode"] == mode] for mode in ("local", "cloud")}

    def pick(mode: str, key: str | None) -> dict[str, Any] | None:
        pool = [e for e in mine if e["mode"] == mode]
        chosen = next((e for e in pool if e["key"] == key), None)
        if chosen is None and pool:                    # default: newest replay run, else newest live run
            pool = sorted(pool, key=lambda e: (e["predictions"] is not None, e.get("started_wall") or 0))
            chosen = pool[-1]
        return chosen
    local, cloud = pick("local", local_key), pick("cloud", cloud_key)
    truth_path = config.get("truth_csv") or (local or {}).get("truth_file") or (cloud or {}).get("truth_file")
    truth, truth_reason = load_truth(truth_path)
    frames, reasons = pairing(local, cloud, truth, truth_reason)
    currency = config.get("currency") or "USD"
    rows, breakdown, points = [], {}, []
    for mode, entry in (("local", local), ("cloud", cloud)):
        if entry is None:
            rows.append({"mode": mode, "available": False, "reason": f"no {mode} run for {camera}"})
            continue
        acc = accuracy(entry, frames, truth) if frames else None
        cost = (local_cost(config["local"], entry["runtime_s"]) if mode == "local"
                else cloud_cost(config["cloud"], entry["runtime_s"]))
        per_1000 = per_thousand(cost["total"], entry["processed"])
        breakdown[mode] = cost
        row = {
            "mode": mode, "available": True, "run_key": entry["key"], "run_kind": entry["kind"],
            "run_id": entry["run_id"], "camera": camera, "input_id": entry.get("input_id"),
            "paired_frames": len(frames) if frames else 0,
            "eligible": acc["eligible"] if acc else None,
            "valid_volume": acc["valid_volume"] if acc else None,
            "missing_volume": acc["missing_volume"] if acc else None,
            "availability_pct": acc["availability_pct"] if acc else None,
            "mae_l": acc["mae_l"] if acc else None, "mape_pct": acc["mape_pct"] if acc else None,
            "mape_n": acc["mape_n"] if acc else None,
            "zero_reference_excluded": acc["zero_reference_excluded_from_mape"] if acc else None,
            "precision": acc["precision"] if acc else None, "recall": acc["recall"] if acc else None,
            "detection_note": acc["detection_note"] if acc else None,
            "failed_frames": entry.get("failed"), "processed_frames": entry["processed"],
            "sent_frames": entry.get("sent"),
            "fps": entry["fps"], "latency_p50_ms": entry["latency_p50_ms"],
            "latency_p95_ms": entry["latency_p95_ms"], "latency_boundary": entry["latency_boundary"],
            "runtime_s": entry["runtime_s"], "runtime_basis": entry["runtime_basis"],
            "run_cost": cost["total"], "cost_kind": cost["total_kind"], "cost_per_1000": per_1000,
            "currency": currency, "gpu": entry.get("gpu"),
        }
        rows.append(row)
        if per_1000 is not None:
            points.append({"mode": mode, "x_cost_per_1000": per_1000, "y_mae_l": row["mae_l"],
                           "y_fps": entry["fps"], "n": row["valid_volume"],
                           "availability_pct": row["availability_pct"], "cost_basis": cost["total_kind"],
                           "latency_p50_ms": entry["latency_p50_ms"], "camera": camera})
    status = ("evaluated" if frames else "Accuracy not evaluated: " + "; ".join(reasons))
    return {
        "camera": camera, "currency": currency, "options": options,
        "selected": {"local": (local or {}).get("key"), "cloud": (cloud or {}).get("key")},
        "accuracy_status": status, "accuracy_reasons": reasons, "paired_frames": len(frames),
        "volume_definition": "object volume the benchmark records per frame (pred_volume_l: stable / RealSense / "
                             "Logitech volume of the largest detection) vs the reference CSV volume_l -- the same "
                             "definition in both modes. Not bin occupancy, not container capacity.",
        "truth_file": truth_path or None,
        "rows": rows, "breakdown": breakdown, "points": points,
        "assumptions": assumptions(config),
        "config": config,
    }


def assumptions(config: dict[str, Any]) -> list[str]:
    cloud, local = config["cloud"], config["local"]
    return [
        f"Cloud rates: {cloud['status']}"
        + (f" (source {cloud['source_url']}, retrieved {cloud['retrieved_on']})" if cloud["source_url"] else
           " (no official source recorded; enter the rate from the Google Cloud pricing page/calculator)"),
        f"Provisioning: {cloud['provisioning'] or 'not stated'}; region: {cloud['region'] or 'not stated'}; "
        f"machine: {cloud['machine_type'] or 'not stated'}; GPU: {cloud['gpu_type'] or 'not stated'}"
        f" x {cloud['gpu_count'] or 0:g}; GPU included in machine rate: "
        + {True: "yes", False: "no (added once)", None: "not stated"}[cloud["gpu_included_in_machine_rate"]],
        "Cloud compute hours = evaluated runtime unless billable VM hours are entered; VM boot/idle time is "
        "not inferred.",
        f"Disk allocated pro rata: GB x rate per GB-month x hours / {HOURS_PER_MONTH:g}; the disk keeps billing "
        "while the VM is stopped (not allocated to the run).",
        f"Local energy: {('measured' if local['power_measured'] else 'assumed') if local['power_w'] is not None else 'unknown'}"
        " power x runtime x tariff; local processing is never treated as free. Hardware amortisation shown "
        "separately; shared camera/Pi hardware is identical in both modes and excluded from both.",
        f"Currency: {config['currency']}" + (f"; exchange rate: {config['exchange_rate_note']}"
                                            if config.get("exchange_rate_note") else ""),
        cloud["notes"],
    ]


CSV_COLUMNS = [
    "camera", "mode", "run_kind", "run_id", "run_key", "input_id", "paired_frames", "eligible", "valid_volume",
    "missing_volume", "availability_pct", "failed_frames", "mae_l", "mape_pct", "mape_n",
    "zero_reference_excluded", "precision", "recall", "detection_note", "processed_frames", "sent_frames",
    "fps", "latency_p50_ms", "latency_p95_ms", "latency_boundary", "runtime_s", "runtime_basis",
    "run_cost", "cost_kind", "cost_per_1000", "currency", "cost_compute", "cost_storage", "cost_other",
    "cost_energy", "cost_hardware_allocated", "compute_rate_per_hour", "rate_status", "rate_source",
    "rate_retrieved_on", "accuracy_status", "truth_file", "assumptions",
]


def view_csv(view: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    cloud_cfg = view["config"]["cloud"]
    for row in view["rows"]:
        if not row.get("available"):
            writer.writerow({"camera": view["camera"], "mode": row["mode"], "accuracy_status": row.get("reason")})
            continue
        cost = view["breakdown"].get(row["mode"], {})
        writer.writerow({**{k: ("" if v is None else v) for k, v in row.items()},
                         "cost_compute": cost.get("compute"), "cost_storage": cost.get("storage"),
                         "cost_other": cost.get("other"), "cost_energy": cost.get("energy"),
                         "cost_hardware_allocated": cost.get("hardware_allocated"),
                         "compute_rate_per_hour": cost.get("compute_rate_per_hour"),
                         "rate_status": cloud_cfg["status"] if row["mode"] == "cloud" else "local inputs",
                         "rate_source": cloud_cfg["source_url"] if row["mode"] == "cloud" else "",
                         "rate_retrieved_on": cloud_cfg["retrieved_on"] if row["mode"] == "cloud" else "",
                         "accuracy_status": view["accuracy_status"], "truth_file": view.get("truth_file") or "",
                         "assumptions": " | ".join(view["assumptions"])})
    return output.getvalue()
