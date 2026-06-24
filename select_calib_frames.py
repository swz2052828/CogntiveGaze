"""Deterministic calibration-frame selection (replaces the leak-prone random
K-from-test-set sampling in the meta-calibration pipeline).

Selection rules (per subject, on the curated 2-cycle clustering):
  R1 prefer cycle 2  |  R2 longer-duration cluster is higher priority
  R3 distinct point+cycle clusters (no duplicates; cycles end up mixed)
  R4 take the MIDDLE (median, non-blink) frame of the chosen cluster
Per (cell, cycle) the longest cluster wins (R2); its representative is the middle
non-blink frame (R4). The cycle for a single-per-point slot is the one whose
cluster is longer, ties -> cycle 2 (R2 with R1 tiebreak).

K levels (replace old 4/8/16/32/64):
  K=4   one frame at each of the 4 CORNERS (TL, TR, BL, BR)
  K=9   one frame at each of the 9 calib points
  K=18  9 points x 2 cycles (both visits), 1 frame per cluster
  K=36  same 18 clusters, 2 frames per cluster
  K=72  same 18 clusters, 4 frames per cluster
For K=18/36/72 each cluster contributes m = K/18 frames, evenly spaced: the k-th
of m frames sits at L*k/(m+1) along the cluster's L non-blink frames (k=1..m), so
e.g. L=30, m=4 -> frames 6, 12, 18, 24. (m=1 -> the middle frame.)

Output: datasets/calib_selection/calib_selection.csv (all subjects, all K) +
        per-K SUMMARY. Light (no video) -> fine on the login node.
"""
import csv, os
import numpy as np
import cluster_calib_points as M
from link_calib_frames import subject_clusters, cycle_index

OUT = "/springbrook/share/eng/esrpxk/datasets/calib_selection"
CORNERS = [0, 6, 2, 8]                     # TL, TR, BL, BR (cell ids)
KS = [4, 9, 18, 36, 72]


def nonblink_frames(runs_si, blink):
    return [f for s, e in runs_si for f in range(s, e) if not blink[f]]


def spaced_frames(fs, m):
    """m frames evenly spaced along fs: k-th at L*k/(m+1), k=1..m (R4 spacing).
    Distinct; if the cluster has <= m frames, return all of them."""
    L = len(fs)
    if L == 0:
        return []
    if L <= m:
        return list(fs)
    idxs = sorted({max(0, min(L - 1, int(round(L * k / (m + 1))) - 1)) for k in range(1, m + 1)})
    return [fs[i] for i in idxs]


def select(sub):
    d = subject_clusters(sub)
    if d is None:
        return []
    p0, p1, runs, cell, dur = d["p0"], d["p1"], d["runs"], d["cell"], d["dur"]
    before, removed, after, mistake = d["before"], d["removed"], d["after"], d["mistake"]
    n = len(p0); cb = M.CALIB[sub]
    blink = M.load_blink(sub, n)
    keep = ~(before | removed | after | mistake)
    cyc = cycle_index(cell, keep, dur)     # capped at 2

    # per (cell, cycle): keep the LONGEST cluster (R2); cache its non-blink frames
    slots = {}                             # (cell, cycle) -> si
    clfs = {}                              # si -> non-blink local frames
    for si in range(len(runs)):
        if not keep[si] or cyc[si] < 1:
            continue
        fs = nonblink_frames(runs[si], blink)
        if not fs:
            continue
        clfs[si] = fs
        k = (int(cell[si]), int(cyc[si]))
        if k not in slots or dur[si] > dur[slots[k]]:
            slots[k] = si

    def rec(si, lf):
        c = int(cell[si]); (sx, sy), snm = M.cell_to_screen(c)
        return dict(cell_id=c, cell_name=M.NAMES[c], cycle=int(cyc[si]), dur=int(dur[si]),
                    local=int(lf), absf=int(cb + lf),
                    iris_x=float(p0[lf]), iris_y=float(p1[lf]),
                    screen_x=sx, screen_y=sy, screen_name=snm)

    def pick_one(c):                       # best single cycle slot for a cell (R2, R1 tiebreak)
        cands = [(cy, slots[(c, cy)]) for cy in (1, 2) if (c, cy) in slots]
        if not cands:
            return None
        return max(cands, key=lambda t: (dur[t[1]], t[0]))[1]

    out = []
    for K in KS:
        recs = []
        if K in (18, 36, 72):
            m = K // 18
            for c in range(9):
                for cy in (1, 2):
                    if (c, cy) in slots:
                        si = slots[(c, cy)]
                        recs += [rec(si, lf) for lf in spaced_frames(clfs[si], m)]
        elif K == 9:
            for c in range(9):
                si = pick_one(c)
                if si is not None:
                    recs.append(rec(si, spaced_frames(clfs[si], 1)[0]))
        else:                              # K == 4 corners
            for c in CORNERS:
                si = pick_one(c)
                if si is not None:
                    recs.append(rec(si, spaced_frames(clfs[si], 1)[0]))
        for j, r in enumerate(recs):
            out.append(dict(sub=sub, K=K, slot=j, **r))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in M.SUBS:
        rows += select(sub)
    cols = ["sub", "K", "slot", "cell_id", "cell_name", "cycle", "dur",
            "local", "absf", "iris_x", "iris_y", "screen_x", "screen_y", "screen_name"]
    with open(f"{OUT}/calib_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("Per-subject selected-frame counts (expect K=" + "/".join(map(str, KS)) + "):")
    for sub in M.SUBS:
        cnt = {K: sum(1 for r in rows if r["sub"] == sub and r["K"] == K) for K in KS}
        flag = "" if all(cnt[K] == K for K in KS) else "  <-- SHORT"
        print(f"  sub{sub}: " + ", ".join(f"K{K}={cnt[K]}" for K in KS) + flag)
    print(f"\nWrote {OUT}/calib_selection.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
