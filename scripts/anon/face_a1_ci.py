"""Participant-level 95% intervals for face-crop A1 (results/face_a1_all/*.scores.npz).

TAR at FAR 1e-3 uses the FIXED full-data threshold (the attacker calibrates once),
resampling participants' genuine hits (2,000 replicates), as everywhere in the paper.
Closed-set Rank-1 resamples participants' probe sets. Writes face_a1_ci.json.
"""
import json
from pathlib import Path

import numpy as np

R = Path("/springbrook/share/eng/esrpxk/results/face_a1_all")
out = {}
for f in sorted(R.glob("*.scores.npz")):
    z = np.load(f); S, py, gs = z["S"].astype(np.float64), z["probe_subject"], z["gallery_subject"]
    own = (py[:, None] == gs[None, :])
    gen, imp = S[own], S[~own]
    thr = np.quantile(imp, 1 - 1e-3)
    pid = py
    hit = S[own] > thr                                  # one genuine score per probe
    r1 = gs[S.argmax(1)] == py
    subs = np.unique(pid); idx = {s: np.where(pid == s)[0] for s in subs}
    rng = np.random.default_rng(0); bt, br = [], []
    for _ in range(2000):
        pick = np.concatenate([idx[s] for s in rng.choice(subs, len(subs))])
        bt.append(hit[pick].mean()); br.append(r1[pick].mean())
    out[f.name.replace(".scores.npz", "")] = dict(
        TAR3=float(hit.mean()), TAR3_ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
        rank1=float(r1.mean()), rank1_ci=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
        n_probe=int(len(py)), n_subjects=int(len(subs)))
    o = out[f.name.replace(".scores.npz", "")]
    print(f"{f.name[:-11]:36s} TAR {100*o['TAR3']:5.1f} [{100*o['TAR3_ci'][0]:.1f}, {100*o['TAR3_ci'][1]:.1f}]   "
          f"R1 {100*o['rank1']:5.1f} [{100*o['rank1_ci'][0]:.1f}, {100*o['rank1_ci'][1]:.1f}]")
json.dump(out, open(R / "face_a1_ci.json", "w"), indent=2)
