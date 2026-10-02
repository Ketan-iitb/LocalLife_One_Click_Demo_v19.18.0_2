"""Same synthetic inputs through one code version. Usage: python eval_v48.py <package root>."""
import sys, json, numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
from test_v45_bin_fill_events import _render, _profile, K
from locallife_cloud import bin_fill as bf
out = {}
frame = np.zeros((K.height, K.width, 3), np.uint8)
prof = _profile(camera_to_empty_floor_m=1.10, usable_height_m=1.00)
# A. same physical surface, different detector recall (many boxes vs few boxes)
pile = _render(prof, lambda x, y: np.where(x < 0.05, 0.10, 0.0))          # undetected 10 cm layer over ~half the floor
many = np.zeros(pile.shape, bool); many[:, :100] = True                    # boxes cover the pile
few = np.zeros(pile.shape, bool); few[40:70, 30:60] = True                 # one small box
fills = {}
for name, boxes in (("many_boxes", many), ("few_boxes", few)):
    with TemporaryDirectory() as d:
        est = bf.FillEstimator("realsense", Path(d), prof)
        est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
        fills[name] = est.update(frame, pile, K, None, 0.0, 1.0, objects=boxes)["height_fill_pct"]
out["fill_same_surface"] = fills
# B/C. bag on a pile: reference error and repeatability under depth noise
rng = np.random.default_rng(0)
errs, single, med = [], [], []
for half_x, half_y, h in ((0.10, 0.075, 0.25), (0.08, 0.06, 0.15), (0.12, 0.08, 0.20)):
    scene = lambda x, y: np.where((np.abs(x) < half_x) & (np.abs(y) < half_y), 0.30 + h, 0.30)
    depth = _render(prof, scene)
    top = depth < (1.10 - 0.30 - h / 2)
    truth = float(np.sum((depth[top] / K.fx) * (depth[top] / K.fy))) * h * 1000
    z = 1.10 - 0.30 - h
    box = (K.ppx + K.fx * -half_x / z, K.ppy + K.fy * -half_y / z, K.ppx + K.fx * half_x / z, K.ppy + K.fy * half_y / z)
    with TemporaryDirectory() as d:
        est = bf.FillEstimator("realsense", Path(d), prof)
        est.recalibrate(_render(prof, lambda x, y: 0 * x), K, None)
        vals = []
        for t in range(20):
            noisy = depth + rng.normal(0, 0.004, depth.shape)                 # ~4 mm stereo noise
            noisy[rng.random(depth.shape) < 0.03] = 0                          # 3 % dropouts
            est.fill_history.clear()
            est.update(frame, noisy, K, None, 0.0, float(t))
            r = est.object_volume(box, None, float(t))
            if r: vals.append(r[0])
        vals = np.array(vals)
        rolling = np.array([np.median(vals[max(0, i - 7):i + 1]) for i in range(len(vals))])
        errs.append(abs(np.median(vals) - truth) / truth * 100)
        single.append(vals.std() / vals.mean() * 100)
        med.append(rolling[7:].std() / rolling[7:].mean() * 100)
out["bag_volume_ref_error_pct"] = [round(e, 1) for e in errs]
out["single_frame_cv_pct"] = [round(v, 2) for v in single]
out["rolling_median_cv_pct"] = [round(v, 2) for v in med]
print(json.dumps(out))
