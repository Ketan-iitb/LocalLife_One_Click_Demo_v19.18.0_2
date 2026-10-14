"""Phase 1 evaluation analysis (supervisor review, check C6).

Input: one log with one row per reading -- either the experiment tool's `trials.csv` (one or more
session folders) or a hand-kept sheet with the columns

    object, pose, trial, camera, volume_l, valid, reference_volume_l[, designation, commit, files]

Every attempted reading counts: a missing, negative or invalid reading is "not valid", never zero and
never dropped. Calibration readings (designation=calibration) are excluded from the evaluation.

Output (in --out): results.md (tables + hypothesis verdicts), readings.csv (the merged log),
errors.svg (MAPE per object per camera) and repeats.svg (spread of the fixed-pose readings).

    python scripts/phase1_analysis.py results/experiment/sessions/S001 --threshold 10 --out phase1_results
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

FIXED_POSE = "fixed"            # the tape-marked repeatability pose; other pose labels are pose-sensitivity runs


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        # a session folder: the one-click readings (readings_long.csv) if any, else the protocol trials
        f = (p / "readings_long.csv" if (p / "readings_long.csv").exists() else p / "trials.csv") if p.is_dir() else p
        with f.open(newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                vol = r.get("volume_l") or ""
                try:
                    v = float(vol)
                except ValueError:
                    v = None
                valid = str(r.get("valid", "")).strip().lower()
                ok = v is not None and v > 0 and valid not in ("no", "false", "0")
                rows.append({"object": r.get("object") or r.get("object_id"),
                             "pose": (r.get("pose") or r.get("placement") or "").strip() or FIXED_POSE,
                             "trial": r.get("trial") or r.get("trial_id"), "camera": r.get("camera"),
                             "volume_l": v if ok else None, "raw_volume": vol, "valid": "yes" if ok else "no",
                             "reference_volume_l": float(r["reference_volume_l"]) if r.get("reference_volume_l") else None,
                             "designation": (r.get("designation") or "test").strip(),
                             "commit": r.get("commit", ""), "files": r.get("files", "")})
    return rows


def stats(rows: list[dict], threshold: float) -> dict:
    ref = rows[0]["reference_volume_l"]
    vals = [r["volume_l"] for r in rows if r["volume_l"] is not None]
    out = {"n": len(rows), "valid": len(vals), "ref": ref, "mean": None, "sd": None, "mape": None,
           "bias": None, "within": 0}
    if vals:
        out["mean"] = st.mean(vals)
        out["sd"] = st.stdev(vals) if len(vals) > 1 else None
    err = [100.0 * (r["volume_l"] - r["reference_volume_l"]) / r["reference_volume_l"]
           for r in rows if r["volume_l"] is not None and r["reference_volume_l"]]   # each reading vs ITS reference
    if err:
        out["mape"] = st.mean(abs(e) for e in err)
        out["bias"] = st.mean(err)
        out["within"] = sum(abs(e) <= threshold for e in err)     # share counted over ALL attempted readings
    return out


def fmt(x, nd=3, suffix=""):
    return "–" if x is None else f"{x:.{nd}f}{suffix}"


def bar_svg(groups: list[tuple[str, dict[str, float | None]]], threshold: float) -> str:
    cams = sorted({c for _, d in groups for c in d})
    colours = {c: col for c, col in zip(cams, ("#2a6db0", "#d9822b", "#6a9a3a", "#8a4fb0"))}
    vmax = max([v for _, d in groups for v in d.values() if v is not None] + [threshold * 1.5, 1.0])
    w, h, left, bottom = 120 * max(1, len(groups)) + 80, 320, 60, 260
    bw = 80 / max(1, len(cams))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Arial" font-size="11">',
             f'<text x="{left}" y="16" font-size="13">Mean absolute percentage error per object and camera</text>',
             f'<line x1="{left}" y1="{bottom}" x2="{w - 10}" y2="{bottom}" stroke="#333"/>']
    ty = bottom - (threshold / vmax) * 220
    parts.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{w - 10}" y2="{ty:.1f}" stroke="#c00" stroke-dasharray="4 3"/>'
                 f'<text x="{w - 12}" y="{ty - 3:.1f}" text-anchor="end" fill="#c00">±{threshold:g} %</text>')
    for i, (name, d) in enumerate(groups):
        x0 = left + 20 + i * 120
        for j, c in enumerate(cams):
            v = d.get(c)
            if v is None:
                parts.append(f'<text x="{x0 + j * bw + bw / 2:.1f}" y="{bottom - 4}" text-anchor="middle">n/a</text>')
                continue
            bh = v / vmax * 220
            parts.append(f'<rect x="{x0 + j * bw:.1f}" y="{bottom - bh:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" '
                         f'fill="{colours[c]}"/><text x="{x0 + j * bw + bw / 2 - 2:.1f}" y="{bottom - bh - 3:.1f}" '
                         f'text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="{x0 + 40}" y="{bottom + 16}" text-anchor="middle">{name}</text>')
    for j, c in enumerate(cams):
        parts.append(f'<rect x="{left + j * 110}" y="{h - 22}" width="10" height="10" fill="{colours[c]}"/>'
                     f'<text x="{left + j * 110 + 14}" y="{h - 13}">{c}</text>')
    return "\n".join(parts + ["</svg>"])


def dot_svg(series: list[tuple[str, list[float], float | None]]) -> str:
    allv = [v for _, vs, _ in series for v in vs] + [r for _, _, r in series if r]
    if not allv:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="40"><text x="5" y="20">no fixed-pose readings</text></svg>'
    lo, hi = min(allv) * 0.9, max(allv) * 1.1
    w, rowh, left = 640, 34, 170
    h = 40 + rowh * len(series)
    xs = lambda v: left + (v - lo) / max(1e-9, hi - lo) * (w - left - 20)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Arial" font-size="11">',
             f'<text x="5" y="16" font-size="13">Repeatability: readings at the fixed pose (| = reference)</text>']
    for i, (name, vs, ref) in enumerate(series):
        y = 40 + i * rowh
        parts.append(f'<text x="5" y="{y + 4}">{name}</text><line x1="{left}" y1="{y}" x2="{w - 20}" y2="{y}" stroke="#ddd"/>')
        if ref:
            parts.append(f'<line x1="{xs(ref):.1f}" y1="{y - 10}" x2="{xs(ref):.1f}" y2="{y + 10}" stroke="#c00" stroke-width="2"/>')
        for v in vs:
            parts.append(f'<circle cx="{xs(v):.1f}" cy="{y}" r="3.5" fill="#2a6db0" fill-opacity="0.7"/>')
    parts.append(f'<text x="{left}" y="{h - 4}">{lo:.2f} L</text><text x="{w - 20}" y="{h - 4}" text-anchor="end">{hi:.2f} L</text>')
    return "\n".join(parts + ["</svg>"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="session folders (with trials.csv) or CSV logs")
    ap.add_argument("--threshold", type=float, default=10.0, help="agreed MAPE threshold, %% (default 10)")
    ap.add_argument("--h2-margin", type=float, default=None,
                    help="H2 alternative: Logitech MAPE within this many points of the RealSense")
    ap.add_argument("--out", type=Path, default=Path("phase1_results"))
    a = ap.parse_args()
    rows = [r for r in load(a.inputs) if r["designation"] != "calibration"]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "readings.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]) if rows else ["object"])
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["object"], r["camera"], r["pose"])].append(r)
        by[(r["object"], r["camera"], "ALL")].append(r)
    cams = sorted({r["camera"] for r in rows})
    objects = sorted({r["object"] for r in rows})
    commits = sorted({r["commit"] for r in rows if r["commit"]})
    md = [f"# Phase 1 evaluation results\n", f"Threshold: ±{a.threshold:g} % mean absolute percentage error. "
          f"Readings: {len(rows)} attempted (calibration readings excluded). Code version(s): {', '.join(commits) or 'not recorded'}.\n",
          "## Per object, camera and pose\n",
          "| Object | Camera | Pose | Attempted | Valid | Reference (L) | Mean (L) | SD (L) | Bias | MAPE | Within threshold |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(by):
        s = stats(by[key], a.threshold)
        md.append(f"| {key[0]} | {key[1]} | {key[2]} | {s['n']} | {s['valid']} | {fmt(s['ref'])} | {fmt(s['mean'])} | "
                  f"{fmt(s['sd'])} | {fmt(s['bias'], 1, ' %')} | {fmt(s['mape'], 1, ' %')} | {s['within']}/{s['n']} |")
    md += ["\n## Repeatability and pose sensitivity\n",
           "Repeatability = SD (and CV) of the readings at the fixed pose. Pose sensitivity = range of the pose means "
           "(excluding the fixed pose), as a share of the reference.\n",
           "| Object | Camera | Fixed-pose SD (L) | CV | Pose means (L) | Pose range / reference |",
           "|---|---|---:|---:|---|---:|"]
    repeats = []
    for o in objects:
        for c in cams:
            fixed = stats(by.get((o, c, FIXED_POSE), [{"reference_volume_l": None, "volume_l": None}]), a.threshold) \
                if (o, c, FIXED_POSE) in by else None
            poses = {k[2]: stats(v, a.threshold)["mean"] for k, v in by.items()
                     if k[0] == o and k[1] == c and k[2] not in ("ALL", FIXED_POSE)}
            means = [m for m in poses.values() if m is not None]
            ref = by[(o, c, "ALL")][0]["reference_volume_l"] if (o, c, "ALL") in by else None
            cv = None if not fixed or not fixed["sd"] or not fixed["mean"] else 100 * fixed["sd"] / fixed["mean"]
            rng = None if len(means) < 2 or not ref else 100 * (max(means) - min(means)) / ref
            md.append(f"| {o} | {c} | {fmt(fixed and fixed['sd'])} | {fmt(cv, 1, ' %')} | "
                      f"{', '.join(f'{p}: {fmt(m)}' for p, m in sorted(poses.items())) or '–'} | {fmt(rng, 1, ' %')} |")
            if (o, c, FIXED_POSE) in by:
                repeats.append((f"{o} · {c}", [r["volume_l"] for r in by[(o, c, FIXED_POSE)] if r["volume_l"] is not None], ref))
    md += ["\n## Hypotheses\n"]
    overall = {c: stats([r for r in rows if r["camera"] == c and r["reference_volume_l"]], a.threshold) for c in cams}
    for c, label in (("realsense", "H1"), ("logitech", "H2")):
        s = overall.get(c)
        if not s or s["mape"] is None:
            md.append(f"- **{label} ({c})**: no valid readings with a reference — not testable.")
            continue
        verdict = s["mape"] <= a.threshold
        line = (f"- **{label} ({c})**: MAPE {s['mape']:.1f} % over {s['valid']} valid of {s['n']} attempted readings "
                f"→ {'supported' if verdict else 'not supported'} at ±{a.threshold:g} %.")
        if c == "logitech" and a.h2_margin is not None and overall.get("realsense") and overall["realsense"]["mape"] is not None:
            gap = s["mape"] - overall["realsense"]["mape"]
            line += f" Gap to RealSense {gap:+.1f} points → {'within' if gap <= a.h2_margin else 'outside'} the ±{a.h2_margin:g} point margin."
        md.append(line)
    md.append("\nInvalid readings (missing, negative, rejected) are counted as attempted and as not within the threshold.")
    (a.out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    groups = [(o, {c: stats(by[(o, c, "ALL")], a.threshold)["mape"] if (o, c, "ALL") in by else None for c in cams})
              for o in objects]
    (a.out / "errors.svg").write_text(bar_svg(groups, a.threshold), encoding="utf-8")
    (a.out / "repeats.svg").write_text(dot_svg(repeats), encoding="utf-8")
    print(f"wrote {a.out}/results.md, readings.csv, errors.svg, repeats.svg ({len(rows)} readings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
