"""Diagnose the iris mapping failure: separates (a) calib-correspondence errors
from (b) task frame-alignment / signal problems.
 1. print recovered 10-target sequence per subject (names+px)
 2. homography self-fit residual on the 10 calib pairs (bad correspondence -> big)
 3. ORACLE affine fit directly on task pairs (iris->GT, fit=eval): the best any
    linear map could do -- if this is small, iris signal + frame alignment are
    fine and only calib correspondence is broken.
 4. auto-correspondence: affine from iris bbox->screen bbox (both mirror flips),
    assign each calib fixation to nearest 3x3 target, refit homography, report.
"""
import sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from iris_calib_mapping_study import (load_subject_files, calib_targets, gt_map,
                                      fit_homography, fit_affine, PX_PER_CM, DATA)

GRID = np.array([(217,146),(115,540),(217,934),(960,92),(960,540),(960,988),
                 (1703,146),(1805,540),(1703,934)], float)

for sub in (6, 10, 15, 22):
    calib, segs = load_subject_files(sub)
    tgt = calib_targets(sub)
    gt = gt_map(sub)
    hx, hy, ix, iy = calib
    C2 = np.stack([ix, iy], 1)
    X2, Y = [], []
    for start, arr in segs.items():
        for j in range(arr.shape[1]):
            f = start + j
            if f in gt:
                X2.append([arr[2, j], arr[3, j]]); Y.append(gt[f])
    X2, Y = np.asarray(X2), np.asarray(Y)

    print(f"\n=== sub {sub}: {len(tgt)} targets, {len(X2)} task frames ===")
    print("  target seq px:", [tuple(int(v) for v in t) for t in tgt])
    print("  iris calib pts:", np.round(C2, 1).tolist())
    # 2. self-fit residual
    pred = fit_homography(C2, tgt)(C2)
    print(f"  homography SELF-fit residual: {np.linalg.norm(pred-tgt,axis=1).mean()/PX_PER_CM:.2f} cm")
    # 3. oracle affine on task
    pa = fit_affine(X2, Y)(X2)
    print(f"  ORACLE affine on task (fit=eval): {np.linalg.norm(pa-Y,axis=1).mean()/PX_PER_CM:.2f} cm")
    # also oracle quadratic
    from iris_calib_mapping_study import fit_poly
    pq = fit_poly(X2, Y, 2)(X2)
    print(f"  ORACLE poly2 on task  (fit=eval): {np.linalg.norm(pq-Y,axis=1).mean()/PX_PER_CM:.2f} cm")
    # 4. auto-correspondence, both flips
    for flip in (False, True):
        c = C2.copy()
        if flip: c[:,0] = -c[:,0]
        # scale iris bbox to screen bbox
        a = (c - c.min(0)) / (c.max(0)-c.min(0)+1e-9)
        s = a * (GRID.max(0)-GRID.min(0)) + GRID.min(0)
        assign = GRID[np.argmin(np.linalg.norm(s[:,None,:]-GRID[None], axis=2), axis=1)]
        pred = fit_homography(C2, assign)(X2 if not flip else X2)
        e = np.linalg.norm(pred - Y, axis=1).mean()/PX_PER_CM
        names = [int(np.argmin(np.linalg.norm(GRID-t,axis=1))) for t in assign]
        print(f"  auto-corr flip={flip}: cells {names}  task err {e:.2f} cm")
