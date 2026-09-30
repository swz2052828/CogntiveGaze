"""Stage 2 acceptance check: is the recovered geometry complete and exact?

Gate for Stage 3 -- every released crop must be reproducible from the original
frame, or the anonymised arm and the control arm are not comparable.
"""
import numpy as np
from pathlib import Path

GEOM = Path("/springbrook/share/eng/esrpxk/datasets/anon_geometry")
FOLDERS = ("appleFace", "appleLeftEye", "appleRightEye")

rows, tot, bad_tot, iods = [], 0, 0, []
for p in sorted(GEOM.glob("*.npz")):
    g = np.load(p)
    n = len(g["frames"]); tot += n
    worst, bad = 1.0, 0
    for f in FOLDERS:
        c = g[f][:, 4]
        worst = min(worst, float(c.min()))
        bad += int((c < 0.99).sum())
    bad_tot += bad
    lf, rt = g["appleLeftEye"], g["appleRightEye"]
    iod = np.abs((lf[:, 0] + lf[:, 2] / 2) - (rt[:, 0] + rt[:, 2] / 2))
    iods.append(iod)
    rows.append((p.stem, n, worst, bad, iod.mean(), g["appleFace"][:, 2].mean()))

print(f"{'rec':>8s} {'frames':>7s} {'min corr':>9s} {'<0.99':>6s} {'IOD px':>7s} {'face px':>8s}")
for r in rows:
    print(f"{r[0]:>8s} {r[1]:7d} {r[2]:9.4f} {r[3]:6d} {r[4]:7.1f} {r[5]:8.1f}")

allio = np.concatenate(iods)
lo, hi = min(r[4] for r in rows), max(r[4] for r in rows)
print(f"\nrecordings {len(rows)}  frames {tot}  frames below 0.99 corr: {bad_tot}")
print(f"IOD cohort: mean {allio.mean():.1f} sd {allio.std():.1f} min {allio.min():.1f} max {allio.max():.1f} px")
print(f"per-subject IOD: {lo:.1f} - {hi:.1f} px ({(hi/lo-1)*100:.0f}% between subjects)")
print("\nblur sigma as fraction of IOD -> absolute px range across subjects:")
for frac in (0.03, 0.10, 0.20, 0.35, 0.50):
    print(f"  {frac:.2f} IOD -> {frac*lo:5.1f} - {frac*hi:5.1f} px")
print("\nGATE:", "PASS" if bad_tot == 0 and len(rows) == 18 else "REVIEW NEEDED")
