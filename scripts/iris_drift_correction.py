"""30-frame drift-shift correction (user's method) on top of iris mappings.
 drift30_roll : each consecutive 30-frame block: shift = mean(pred)-mean(GT) of
                the block, subtracted within the block (upper-bound variant).
 drift30_start: shift from the FIRST 30 matched frames of each task segment,
                applied to the remainder; error on non-anchor frames (deployable:
                the dot location at task start is known).
Base mappings: projective_2d (baseline), rbf_2d, svr_4d, svr_4d_mm."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from iris_calib_mapping_study import (load_subject_files, fixed_targets, gt_map,
                                      fit_homography, fit_svr, fit_rbfi, PX_PER_CM, DATA)
import glob, collections

res = collections.defaultdict(dict)
subs = sorted(int(os.path.basename(p).split("_")[1])
              for p in glob.glob(f"{DATA}/extracted_data/subid_*"))
for sub in subs:
    calib, segs = load_subject_files(sub)
    if calib is None:
        continue
    gt = gt_map(sub)
    if not gt:
        continue
    tgt = np.tile(fixed_targets(), (calib.shape[1] // 10, 1))
    hx, hy, ix, iy = calib
    C2, C4 = np.stack([ix, iy], 1), np.stack([hx, hy, ix, iy], 1)
    maps = {
        "projective_2d": (fit_homography(C2, tgt), 2, False),
        "rbf_2d": (fit_rbfi(C2, tgt, "gaussian", epsilon=5.0, smoothing=1.0), 2, False),
        "svr_4d": (fit_svr(C4, tgt), 4, False),
        "svr_4d_mm": (fit_svr(C4, tgt), 4, True),
    }
    acc = collections.defaultdict(list)
    for start, arr in segs.items():
        fr_ok = [j for j in range(arr.shape[1]) if start + j in gt]
        if len(fr_ok) < 60:
            continue
        Y = np.array([gt[start + j] for j in fr_ok])
        X2 = arr[2:4, fr_ok].T
        X4 = arr[0:4, fr_ok].T
        for name, (f, dim, use_mm) in maps.items():
            X = X2 if dim == 2 else X4
            if use_mm:
                X = (X - X.mean(0)) / (X.std(0) + 1e-9) * (C4.std(0)) + C4.mean(0)
            P = f(X)
            acc[name + "_raw"].append(np.linalg.norm(P - Y, axis=1).mean())
            # rolling 30-frame shift
            Pc = P.copy()
            for b in range(0, len(P), 30):
                sl = slice(b, min(b + 30, len(P)))
                Pc[sl] -= P[sl].mean(0) - Y[sl].mean(0)
            acc[name + "_drift30_roll"].append(np.linalg.norm(Pc - Y, axis=1).mean())
            # start-of-task shift
            shift = P[:30].mean(0) - Y[:30].mean(0)
            e = np.linalg.norm((P[30:] - shift) - Y[30:], axis=1)
            acc[name + "_drift30_start"].append(e.mean())
            # settle-skipped start anchor: frames 10-40, median (robust to saccade)
            sh = np.median(P[10:40], axis=0) - np.median(Y[10:40], axis=0)
            e = np.linalg.norm((P[40:] - sh) - Y[40:], axis=1)
            acc[name + "_drift30_settle"].append(e.mean())
            # causal rolling: block b corrected by block b-1's shift (deployable)
            Pc2 = P.copy()
            prev = None
            for b in range(0, len(P), 30):
                sl = slice(b, min(b + 30, len(P)))
                if prev is not None:
                    Pc2[sl] -= prev
                prev = P[sl].mean(0) - Y[sl].mean(0)
            e = np.linalg.norm(Pc2[30:] - Y[30:], axis=1)
            acc[name + "_drift30_causal"].append(e.mean())
    for k, v in acc.items():
        res[k][sub] = float(np.mean(v)) / PX_PER_CM

names = sorted(res, key=lambda n: np.nanmean(list(res[n].values())))
print(f"{'method':34s} {'cohort mean':>12} {'median':>9}")
for n in names:
    v = list(res[n].values())
    print(f"{n:34s} {np.nanmean(v):>12.3f} {np.nanmedian(v):>9.3f}")
out = "/springbrook/share/eng/esrpxk/results/iris_drift_correction.md"
with open(out, "w") as fh:
    fh.write("# 30-frame drift-shift correction (cm)\n\n| method | mean | median |\n|---|---|---|\n")
    for n in names:
        v = list(res[n].values())
        fh.write(f"| {n} | {np.nanmean(v):.3f} | {np.nanmedian(v):.3f} |\n")
print("wrote", out)
