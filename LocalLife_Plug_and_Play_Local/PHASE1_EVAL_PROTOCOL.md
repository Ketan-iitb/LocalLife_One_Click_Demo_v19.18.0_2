# Phase 1 evaluation protocol (supervisor review, 14 Oct 2026)

Scope is frozen: only changes that serve the volume question. Code version: git tag `phase1-eval-v1`.
Every trial records the commit, measurement version, depth model, intrinsics, baseline id and calibration id.

## Before every session (C2)

```
python -m unittest tests.test_phase1_eval_synthetic
```
Must pass: 10x10x10 cm box = 1.000 L, 10x10x20 cm = 2.000 L, 20x10x10 cm lying = standing = 2.000 L, each within ±0.5 %.

## C3 Reference objects (once)

Register each object with its measured volume and method, e.g.:
```
python -m locallife_cloud.experiment_cli object-add box1L --volume-l 1.000 --method "ruler, 10.0 x 10.0 x 10.0 cm external" --quantity external_geometric --status measured --uncertainty-l 0.03
python -m locallife_cloud.experiment_cli object-add bottle --volume-l 0.62 --method "water displacement, mean of 3" --quantity displacement --status measured --uncertainty-l 0.02
```
Photograph each box next to the ruler.

## C4 Session

1. Rig fixed; write down height above floor and tilt; photo. Do not touch it during the session.
2. Start the server (normal launcher), then:
```
python -m locallife_cloud.experiment_cli session --note "rig 75 cm, tilt X deg"
python -m locallife_cloud.experiment_cli baseline            # bin/floor EMPTY, nothing moving
```
3. Calibration placements (C5, decided in advance: empty floor + box1L only):
```
python -m locallife_cloud.experiment_cli trial-start box1L --designation calibration --placement cal-near --top-height-m 0.100
python -m locallife_cloud.experiment_cli trial-stop
```
   Repeat at 3 positions at different distances, then
```
python -m locallife_cloud.experiment_cli calibrate --camera logitech --perpendicular-floor-distance-m 0.75
```
   It must report `validation: validated` (≥ 3 placements, ≥ 15 % depth span, held-out error ≤ 5 %).
4. Evaluation, every object except box1L, both cameras record automatically:
   * Repeatability: 10 x `trial-start <object> --designation test --placement fixed` (lift out, put back in the tape outline, `trial-stop`).
   * Pose sensitivity: 5 x each of `--placement lying`, `--placement standing`, `--placement rot45`.
   Never delete a trial; failed readings stay in the log as invalid.

## C6 Analysis

```
python scripts/phase1_analysis.py <results>/experiment/sessions/<SESSION> --threshold 10 --h2-margin 5 --out phase1_results
```
Writes `results.md` (per object/camera/pose: attempted, valid, mean, SD, bias, MAPE, within threshold;
repeatability SD/CV at the fixed pose; pose-mean range; H1/H2 verdicts), `readings.csv`, `errors.svg`
and `repeats.svg`. A hand-kept sheet with columns `object, pose, trial, camera, volume_l, valid,
reference_volume_l` works as input too.
