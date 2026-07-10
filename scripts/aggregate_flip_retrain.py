"""Flip-right-eye retrain vs clean baseline: base val_error per backbone (cm)."""
import glob, os
import numpy as np, torch

RUNS = "/springbrook/share/eng/esrpxk/runs"

def mean_err(root, bb):
    v = []
    for f in glob.glob(f"{root}/base/seed42/fold*_best_{bb}_gaze_segmenter.pth"):
        try: v.append(float(torch.load(f, map_location="cpu", mmap=True)["val_error"]))
        except Exception: pass
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

rows = []
for fd in sorted(glob.glob(f"{RUNS}/meta_pipeline_flip_*")):
    name = os.path.basename(fd)
    suf = name[len("meta_pipeline_flip_"):]
    cd = f"{RUNS}/meta_pipeline_clean_{suf}"
    ck = glob.glob(f"{fd}/base/seed42/fold0_best_*_gaze_segmenter.pth")
    if not ck: continue
    bb = os.path.basename(ck[0])[len("fold0_best_"):-len("_gaze_segmenter.pth")]
    fe, nf = mean_err(fd, bb); ce, nc = mean_err(cd, bb)
    rows.append((bb, ce, fe, ce - fe, nf, nc))
rows.sort(key=lambda r: -r[3] if r[3] == r[3] else 0)
print(f"{'backbone':38s} {'baseline':>9} {'flipped':>9} {'delta':>8} {'nf/nc':>6}")
gains = []
for bb, ce, fe, d, nf, nc in rows:
    print(f"{bb:38s} {ce:>9.3f} {fe:>9.3f} {d:>+8.3f} {nf}/{nc}")
    if d == d: gains.append(d)
print(f"\nCOHORT: mean delta {np.mean(gains):+.3f}  median {np.median(gains):+.3f}  "
      f"flip better on {sum(1 for g in gains if g>0)}/{len(gains)}")
out = "/springbrook/share/eng/esrpxk/results/flip_right_eye_summary.md"
with open(out, "w") as fh:
    fh.write("# Flip-right-eye retrain vs clean baseline (base val_error, cm; delta>0 = flip better)\n\n")
    fh.write("| backbone | baseline | flipped | delta |\n|---|---|---|---|\n")
    for bb, ce, fe, d, nf, nc in rows:
        fh.write(f"| {bb} | {ce:.3f} | {fe:.3f} | {d:+.3f} |\n")
    fh.write(f"\nCohort: mean {np.mean(gains):+.3f}, median {np.median(gains):+.3f}, "
             f"flip better {sum(1 for g in gains if g>0)}/{len(gains)}\n")
print("wrote", out)
