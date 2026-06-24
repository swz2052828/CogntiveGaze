"""Link calibration-window frames to calibration points (Experiment 2).

UPDATED 2026-06-23: this now consumes the authoritative, hand-curated clustering
produced by cluster_calib_points.py (split fixations merged; per-subject overrides;
relabel / start / end / remove / only / keep edits; duration-aware MC-cycle
before/after detection), instead of the old standalone k-means reproduction.

Each KEPT fixation frame is linked to its 3x3 calibration target (iris cell ->
screen, horizontal flip via M.cell_to_screen) and tagged with its calibration
CYCLE (1 or 2 -- calibration runs MC -> 8 edges -> MC, twice). Discarded frames
carry their reason (before / removed / after / outlier / blink).

Output per subject: <out>/subid_<id>_calib_link.csv
  abs_frame, local_frame, iris_x, iris_y, status, cycle, cell_id, cell_name,
  screen_x, screen_y, screen_name
plus <out>/SUMMARY.csv.
"""
import csv, os
import numpy as np
import cluster_calib_points as M
from cluster_calib_points import load_blink   # EAR blink flags (shared)

OUT = "/springbrook/share/eng/esrpxk/datasets/calib_point_links"
CENTER = 4


def subject_clusters(sub):
    """Re-run the curated pipeline for one subject -> per-cluster arrays
    (single source of truth = cluster_calib_points)."""
    path = M.find_bin(sub)
    if path is None:
        return None
    with open(path, "rb") as f:
        np.load(f); np.load(f); p0 = np.load(f); p1 = np.load(f)
    anch = M.anchor_grid(sub)
    if anch is None:
        return None
    segs = M.segment_30(p0, p1)
    C = np.array([[np.median(p0[s:e]), np.median(p1[s:e])] for s, e in segs])
    dur = np.array([e - s for s, e in segs])
    pair = np.linalg.norm(anch[:, None, :] - anch[None, :, :], axis=2)
    np.fill_diagonal(pair, np.inf)
    thr = max(3.0, 0.8 * pair.min())
    inlier = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1) < thr
    cell, _ = M.cluster(C, anch, inlier)
    cell = M.apply_overrides(sub, C, cell)
    runs, C, dur, cell = M.merge_runs(segs, C, dur, cell, p0, p1)
    cell, before, removed, after, force_keep = M.apply_post_merge_edits(sub, cell, C, dur)
    inlier = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1) < thr
    discard = before | removed | after
    mistake = (~inlier) & ~discard & ~force_keep
    return dict(p0=p0, p1=p1, runs=runs, cell=cell, dur=dur,
                before=before, removed=removed, after=after, mistake=mistake)


def cycle_index(cell, keep, dur, min_edges=8, min_dur=8, max_cycles=2):
    """Cycle number (1, 2) per KEPT cluster, split at substantial MC returns once
    all 8 edges have been reached. Capped at max_cycles=2: the calibration is two
    cycles, so any kept clusters past the 2nd return are an EXTENSION of cycle 2
    (e.g. sub7 m20-24), not a separate cycle."""
    cyc = np.zeros(len(cell), int)
    cur = set(); started = False; c = 1
    for i in range(len(cell)):
        if not keep[i]:
            continue
        cyc[i] = c
        if cell[i] == CENTER:
            if started and len(cur) >= min_edges and dur[i] >= min_dur:
                c = min(c + 1, max_cycles); cur = set()
            started = True
        else:
            cur.add(int(cell[i]))
    return cyc


def process(sub):
    d = subject_clusters(sub)
    if d is None:
        return None
    p0, p1, runs, cell, dur = d["p0"], d["p1"], d["runs"], d["cell"], d["dur"]
    before, removed, after, mistake = d["before"], d["removed"], d["after"], d["mistake"]
    n = len(p0); cb = M.CALIB[sub]
    keep = ~(before | removed | after | mistake)
    cyc = cycle_index(cell, keep, dur)

    fr_status = np.array(["gap"] * n, dtype=object)
    fr_cyc = np.zeros(n, int); fr_cell = np.full(n, -1)
    fr_status[(p0 <= 0) | (p1 <= 0)] = "blink"
    for si, grp in enumerate(runs):
        st = ("before" if before[si] else "removed" if removed[si] else
              "after" if after[si] else "outlier" if mistake[si] else "fixation")
        for s, e in grp:
            fr_status[s:e] = st; fr_cell[s:e] = cell[si]; fr_cyc[s:e] = cyc[si]

    # overlay EAR blink detection: a blink frame is never a usable fixation, so
    # re-tag 'fixation'/'gap' frames flagged by the EAR detector as 'blink'.
    blink = load_blink(sub, n)
    if blink is not None:
        for i in np.where(blink)[0]:
            if fr_status[i] in ("fixation", "gap"):
                fr_status[i] = "blink"; fr_cell[i] = -1; fr_cyc[i] = 0

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/subid_{sub}_calib_link.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["abs_frame", "local_frame", "iris_x", "iris_y", "status",
                    "cycle", "cell_id", "cell_name", "screen_x", "screen_y", "screen_name"])
        for i in range(n):
            if fr_status[i] == "fixation":
                c = fr_cell[i]; (sx, sy), snm = M.cell_to_screen(c)
                w.writerow([cb + i, i, f"{p0[i]:.1f}", f"{p1[i]:.1f}", "fixation",
                            fr_cyc[i], c, M.NAMES[c], sx, sy, snm])
            else:
                w.writerow([cb + i, i, f"{p0[i]:.1f}", f"{p1[i]:.1f}", fr_status[i],
                            "", "", "", "", "", ""])
    linked = int((fr_status == "fixation").sum())
    nblink = int((fr_status == "blink").sum())
    return dict(sub=sub, n=n, fixations=int(keep.sum()), n_cycles=int(cyc.max()),
                linked=linked, blinks=nblink, pct=round(100 * linked / n, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in M.SUBS:
        r = process(sub)
        if r is None:
            print(f"sub{sub}: no binary"); continue
        rows.append(r)
        print(f"sub{sub}: {r['n']} frames, {r['fixations']} fixation clusters, "
              f"{r['n_cycles']} cycles, {r['linked']} linked ({r['pct']}%), "
              f"{r['blinks']} blink frames")
    with open(f"{OUT}/SUMMARY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sub", "n", "fixations", "n_cycles",
                                          "linked", "blinks", "pct"])
        w.writeheader(); w.writerows(rows)
    tot = sum(r["n"] for r in rows); lk = sum(r["linked"] for r in rows)
    print(f"\nTOTAL: {tot} calib frames, {lk} linked ({100*lk/tot:.0f}%)")
    print(f"Wrote {OUT}/subid_<id>_calib_link.csv + SUMMARY.csv")


if __name__ == "__main__":
    main()
