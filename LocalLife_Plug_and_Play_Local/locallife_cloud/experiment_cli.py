"""V54 volume experiment CLI.

Offline (files under <results>/experiment):  object-add, objects, replay, calibrate, evaluate
Live (talks to the running server):          session, baseline, trial-start, trial-stop, status

    python -m locallife_cloud.experiment_cli --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


from . import volume_experiment as ve


def _root(args) -> Path:
    return Path(args.results or os.environ.get("LOCALLIFE_RESULTS_DIR", Path.cwd() / "artifacts")) / "experiment"


def _post(args, action: str, body: dict | None = None) -> dict:
    url = f"{args.server.rstrip('/')}/api/experiment/{action}"
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if action == "status" else "POST",
                                     headers={"Content-Type": "application/json",
                                              "X-API-Token": os.environ.get("LOCALLIFE_API_TOKEN", "")})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode(errors="replace"), "status": exc.code}


def _trial_dirs(root: Path, sessions: list[str]) -> list[Path]:
    bases = [root / "sessions" / s for s in sessions] if sessions else [root / "sessions"]
    return sorted({p.parent for b in bases for p in b.rglob("trial.json")})


def cmd_calibrate(args) -> int:
    root = _root(args)
    samples, metas = [], []
    for directory in _trial_dirs(root, args.session):
        record = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
        if record.get("designation") != "calibration" or args.camera not in record.get("cameras", {}):
            continue
        if (record.get("reference") or {}).get("reference_status") != "measured":
            print(f"skip {record['trial_id']}: reference not measured")
            continue
        replay = ve.replay_trial(directory, {args.camera: {"depth_scale": 1.0, "align_background": args.align_background,
                                                           "calibration_id": "fitting"}})
        result = replay["cameras"][args.camera]
        frames = [f for f in result["frames"] if f.get("saved")]
        if not frames or result["volume_l"] is None:
            print(f"skip {record['trial_id']}: {result.get('reasons')}")
            continue
        samples.append({"trial_id": record["trial_id"], "session_id": record["session_id"],
                        "object_id": record["object_id"], "reference_l": record["reference_volume_l"],
                        "raw_volume_l": result["volume_l"]})
        session_dir = directory.parent.parent
        baseline = json.loads((session_dir / frames[0]["baseline_file"].replace(".npz", ".json")).read_text(encoding="utf-8"))
        metas.append(baseline)
    if not metas:
        print("no usable calibration trials")
        return 2
    first = metas[0]
    for other in metas[1:]:
        problems = ve.compatibility(first, other)
        if problems:
            print(f"calibration trials come from incompatible setups: {problems}")
            return 2
    meta = {k: first.get(k) for k in ("shape", "intrinsics", "plane", "depth_model", "output_kind",
                                      "intrinsics_source", "depth_units")}
    meta["software"] = ve.software_version()
    calibration = ve.fit_calibration(samples, camera=args.camera, align_background=args.align_background, meta=meta)
    # verify by replay at the fitted scale (V ∝ s^3 is exact only up to discretisation)
    check = []
    for s in samples:
        directory = root / "sessions" / s["session_id"] / "trials" / s["trial_id"]
        r = ve.replay_trial(directory, {args.camera: {k: calibration[k] for k in ("depth_scale", "align_background", "calibration_id")}})
        v = r["cameras"][args.camera]["volume_l"]
        check.append(None if v is None else round(100 * (v - s["reference_l"]) / s["reference_l"], 3))
    calibration["fit_quality"]["replayed_residual_pct"] = check
    if args.perpendicular_floor_distance_m:
        # Independent check, NOT used in the fit: the baseline plane's perpendicular camera distance at the
        # fitted scale vs a tape measurement of the same quantity (perpendicular, not along the view axis).
        fitted = calibration["depth_scale"] * float(first["plane"]["d"])
        calibration["fit_quality"]["floor_distance_check"] = {
            "measured_perpendicular_m": args.perpendicular_floor_distance_m, "fitted_m": round(fitted, 4),
            "residual_pct": round(100 * (fitted - args.perpendicular_floor_distance_m)
                                  / args.perpendicular_floor_distance_m, 3)}
    path = root / "calibration" / f"{args.camera}.json"
    if path.exists() and not args.replace:
        print(f"{path} exists: pass --replace to supersede it (the old one is kept as history)")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        (path.parent / f"{args.camera}_{old['calibration_id']}.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
    path.write_text(json.dumps(calibration, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: calibration[k] for k in ("calibration_id", "depth_scale", "align_background",
                                                  "calibration_objects", "fit_quality")}, indent=1))
    return 0


def cmd_evaluate(args) -> int:
    root = _root(args)
    trials = []
    for directory in _trial_dirs(root, args.session):
        if args.replay:
            setups = {}
            for camera in ve.CAMERAS:
                cal = json.loads((root / "calibration" / f"{camera}.json").read_text(encoding="utf-8")) \
                    if (root / "calibration" / f"{camera}.json").exists() else None
                if cal:
                    setups[camera] = {k: cal[k] for k in ("depth_scale", "align_background", "calibration_id")}
            record = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
            record["cameras"] = ve.replay_trial(directory, setups)["cameras"]
        else:
            record = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
        trials.append(record)
    calibration_objects = {}
    for camera in ve.CAMERAS:
        path = root / "calibration" / f"{camera}.json"
        if path.exists():
            calibration_objects[camera] = set(json.loads(path.read_text(encoding="utf-8")).get("calibration_objects", []))
    criteria = ({"max_mape_pct": args.max_mape_pct, "min_availability": args.min_availability}
                if args.max_mape_pct is not None and args.min_availability is not None else ve.env_criteria())
    result = ve.evaluate(trials, calibration_objects=calibration_objects, criteria=criteria)
    for row in result["rows"]:
        print(f"{row['camera']:9} {str(row['object_id']):12} {row['condition']:11} {row['motion_state']:14} "
              f"{row['split']:32} n={row['attempted']:2} valid={row['valid']:2} avail={row['availability']:.2f} "
              f"ref={row['reference_l']} mean={row['estimate_mean_l']} sd={row['estimate_std_l']} "
              f"bias%={row.get('mean_signed_error_pct')} MAE={row.get('mae_l')} RMSE={row.get('rmse_l')} "
              f"MAPE%={row.get('mape_pct')} criteria={row['criteria']}")
    paired = result["paired"]
    print(f"paired valid trials: {paired['n_paired_valid']}; Logitech - RealSense (L): "
          f"{paired['logitech_minus_realsense_l']} ({paired['note']})")
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        ve._append_csv(out.with_suffix(".csv"), [{k: (json.dumps(v) if isinstance(v, dict) else v)
                                                  for k, v in row.items()} for row in result["rows"]])
        print(f"wrote {out} and {out.with_suffix('.csv')}")
    return 0


def cmd_replay(args) -> int:
    directory = Path(args.trial)
    record = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
    replay = ve.replay_trial(directory)
    worst = 0.0
    for camera, res in replay["cameras"].items():
        recorded = record["cameras"][camera]
        for f_new, f_old in zip(res["frames"], recorded.get("frames", [])):
            if f_new.get("saved") and f_new.get("volume_l") is not None and f_old.get("volume_l") is not None:
                worst = max(worst, abs(f_new["volume_l"] - f_old["volume_l"]))
        print(f"{camera}: recorded {recorded.get('volume_l')} L ({recorded.get('status')}), "
              f"replayed {res['volume_l']} L ({res['status']})")
    print(f"largest frame difference: {worst:.6f} L")
    return 0 if worst < 1e-4 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", help="results directory (default $LOCALLIFE_RESULTS_DIR or ./artifacts)")
    parser.add_argument("--server", default=os.environ.get("LOCALLIFE_SERVER", "http://127.0.0.1:8000"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("object-add", help="register a reference object and how its reference was measured")
    p.add_argument("object_id")
    p.add_argument("--volume-l", type=float)
    p.add_argument("--method", required=True, help="e.g. 'caliper L x W x H, external' or 'water displacement'")
    p.add_argument("--quantity", choices=ve.REFERENCE_QUANTITIES, required=True)
    p.add_argument("--status", choices=ve.REFERENCE_STATUSES, required=True)
    p.add_argument("--uncertainty-l", type=float)
    p.add_argument("--notes", default="")
    sub.add_parser("objects")
    p = sub.add_parser("session")
    p.add_argument("--note", default="")
    p = sub.add_parser("baseline", help="capture the empty-bin baseline (bin EMPTY, nothing moving)")
    p.add_argument("--camera", action="append", choices=ve.CAMERAS)
    p = sub.add_parser("trial-start")
    p.add_argument("object_id")
    p.add_argument("--designation", choices=ve.DESIGNATIONS, required=True)
    p.add_argument("--placement", default="")
    p.add_argument("--condition", choices=ve.CONDITIONS, default="isolated")
    p.add_argument("--motion", choices=("settled", "moving"), default="settled")
    sub.add_parser("trial-stop")
    sub.add_parser("status")
    p = sub.add_parser("replay", help="recompute a recorded trial from its raw files")
    p.add_argument("trial", help="results/experiment/sessions/<session>/trials/<trial_id>")
    p = sub.add_parser("calibrate", help="fit and freeze one camera's depth scale on CALIBRATION trials only")
    p.add_argument("--camera", choices=ve.CAMERAS, required=True)
    p.add_argument("--session", action="append", default=[])
    p.add_argument("--align-background", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--perpendicular-floor-distance-m", type=float,
                   help="tape-measured PERPENDICULAR camera-to-bin-floor distance: an independent check only")
    p = sub.add_parser("evaluate")
    p.add_argument("--session", action="append", default=[])
    p.add_argument("--replay", action="store_true", help="recompute with the current frozen calibrations")
    p.add_argument("--max-mape-pct", type=float)
    p.add_argument("--min-availability", type=float)
    p.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "object-add":
        recorder = ve.ExperimentRecorder(_root(args))
        print(json.dumps(recorder.add_object(args.object_id, reference_volume_l=args.volume_l, reference_method=args.method,
                                             reference_quantity=args.quantity, reference_status=args.status,
                                             reference_uncertainty_l=args.uncertainty_l, notes=args.notes), indent=1))
        return 0
    if args.command == "objects":
        print(json.dumps(ve.ExperimentRecorder(_root(args)).objects(), indent=1))
        return 0
    if args.command == "calibrate":
        if args.align_background is None:
            args.align_background = args.camera == "logitech"
        return cmd_calibrate(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "replay":
        return cmd_replay(args)
    body = {"session": {"note": getattr(args, "note", "")}, "baseline": {"cameras": getattr(args, "camera", None)},
            "trial-start": {k: getattr(args, k, None) for k in ("object_id", "designation", "placement", "condition", "motion")},
            "trial-stop": {}, "status": None}[args.command]
    print(json.dumps(_post(args, args.command, body), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
