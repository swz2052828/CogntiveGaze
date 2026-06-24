"""Cluster the per-frame calibration iris data into the 9 calibration points.

Per subject (subid_<id>_<calib_begins> binary, [head_x,head_y,iris_x,iris_y]):
  Rule 1 - fixations last ~30 frames: detect stable runs (low frame-to-frame
           movement); SPLIT runs longer than ~45 into ~30-frame chunks; KEEP
           short runs (do NOT drop them).
  Rule 2 - 9 points on a 3x3 grid, mistakes flagged: hierarchical clustering
           (k-means(3) on iris-x -> columns, then k-means(3) on iris-y WITHIN
           each column -> rows, which absorbs the grid skew) computed from
           inliers; spatial-outlier / off-grid segments are flagged as mistakes
           (not removed). Every segment, short ones included, gets a cell label.

Outputs under datasets/calib_clusters/:
  subid_<id>_clusters.csv    per frame: abs_frame, local_frame, iris_x, iris_y,
                             in_fixation, seg_dur, cell_id (0-8), cell_name,
                             is_mistake
  subid_<id>_centers.csv     the 9 cluster centres (iris space) + counts
  subid_<id>_plot.png        verification scatter
  SUMMARY.csv
"""
import csv, glob, os
import numpy as np
from scipy.cluster.vq import kmeans2
import scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BIN = "/springbrook/share/eng/esrpxk/datasets/calib_binaries"
OUT = "/springbrook/share/eng/esrpxk/datasets/calib_clusters"
CALIB_PROC = "/springbrook/share/eng/esrpxk/datasets/calib_processed"


def load_blink(sub, n):
    """Per-frame EAR blink flags from calib_processed/<id05d>/metadata.mat (same
    local-frame indexing as the iris binary). Returns bool[n] (all-False if none)."""
    p = f"{CALIB_PROC}/{sub:05d}/metadata.mat"
    out = np.zeros(n, bool)
    if os.path.exists(p):
        b = np.array(sio.loadmat(p)["blink"]).ravel().astype(int)
        k = min(n, len(b)); out[:k] = b[:k] > 0
    return out
SUBS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB = dict(zip(SUBS, [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
                        1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]))
NAMES = {0: "top-left", 1: "mid-left", 2: "bot-left", 3: "top-center",
         4: "mid-center", 5: "bot-center", 6: "top-right", 7: "mid-right", 8: "bot-right"}

# Screen targets per (screen_col, screen_row), screen_col 0=left/1=center/2=right,
# screen_row 0=top/1=mid/2=bot.
_SCREEN = {(0, 0): (217, 146), (0, 1): (115, 540), (0, 2): (217, 934),
           (1, 0): (960, 92), (1, 1): (960, 540), (1, 2): (960, 988),
           (2, 0): (1703, 146), (2, 1): (1805, 540), (2, 2): (1703, 934)}
_SCRNAME = {(0, 0): "top-left", (0, 1): "left", (0, 2): "bot-left",
            (1, 0): "top", (1, 1): "center", (1, 2): "bottom",
            (2, 0): "top-right", (2, 1): "right", (2, 2): "bot-right"}


def cell_to_screen(cell_id):
    """iris cell -> screen target, with HORIZONTAL FLIP (iris col 0/1/2 ->
    screen col 2/1/0); vertical row unchanged."""
    col, row = cell_id // 3, cell_id % 3
    scol = 2 - col
    return _SCREEN[(scol, row)], _SCRNAME[(scol, row)]


def find_bin(sub):
    for p in glob.glob(f"{BIN}/subid_{sub}_*"):
        b = os.path.basename(p)
        if "template" in b or "calib" in b:
            continue
        return p
    return None


def anchor_grid(sub):
    """9 location-sorted anchor positions from the reorganised _calib_ file."""
    p = glob.glob(f"{BIN}/subid_{sub}_calib_*")
    if not p:
        return None
    with open(p[0], "rb") as f:
        a0 = np.load(f); a1 = np.load(f); a2 = np.load(f); a3 = np.load(f)
    A = np.c_[a2, a3].astype(float)
    cc, _ = kmeans2(A, 9, minit="++", seed=0)   # dedupe 10 -> 9
    return cc


def segment_30(p0, p1, move_thr=3, target=30, split_long=45, min_run=3):
    valid = (p0 > 0) & (p1 > 0); n = len(p0)
    d = np.r_[0, np.hypot(np.diff(p0), np.diff(p1))]
    raw = []; i = 0
    while i < n:
        if valid[i] and d[i] <= move_thr:
            j = i
            while j + 1 < n and valid[j + 1] and d[j + 1] <= move_thr:
                j += 1
            raw.append((i, j + 1)); i = j + 1
        else:
            i += 1
    segs = []
    for s, e in raw:
        L = e - s
        if L < min_run:
            continue
        if L > split_long:                       # split too-long
            k = max(2, int(round(L / target)))
            bn = np.linspace(s, e, k + 1).astype(int)
            for a, b in zip(bn[:-1], bn[1:]):
                segs.append((a, b))
        else:                                    # keep (incl short)
            segs.append((s, e))
    return segs


def mad_outlier(v, k=3.5):
    med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return np.abs(v - med) > k * mad


def grid_label(centers):
    """Map 9 cluster centres -> grid cell id (col*3+row): cols by x (L,C,R),
    rows by y within each column (top,mid,bot)."""
    xs = np.argsort(centers[:, 0])
    cols = [xs[:3], xs[3:6], xs[6:]]
    cell_of = {}
    for col, idxs in enumerate(cols):
        for row, ci in enumerate(idxs[np.argsort(centers[idxs, 1])]):
            cell_of[int(ci)] = col * 3 + row
    return cell_of


def cluster(C, anch, inlier):
    """k-means(9) seeded at the location-sorted anchors, fit on INLIERS only so
    noise/outlier segments can't drag a centroid off-grid; then every segment
    (short + outlier) is assigned to its nearest centre."""
    cc, _ = kmeans2(C[inlier].astype(float), anch.astype(float), minit="matrix")
    cell_of = grid_label(cc)
    # assign ALL segments (incl short + outliers) to nearest centre
    d = np.linalg.norm(C[:, None, :] - cc[None, :, :], axis=2)
    near = d.argmin(1)
    cell = np.array([cell_of[int(c)] for c in near])
    grid_cc = np.full((9, 2), np.nan)
    for cl, g in cell_of.items():
        grid_cc[g] = cc[cl]
    return cell, grid_cc


ABBR2ID = {"TL": 0, "ML": 1, "BL": 2, "TC": 3, "MC": 4,
           "BC": 5, "TR": 6, "MR": 7, "BR": 8}

# ---------------------------------------------------------------------------
# Manual post-clustering overrides, authored from visual review of the per-
# subject plots. The calibration shows every target TWICE (center->edges->
# center, repeated), so each cell should hold ~2 genuine fixations; the auto
# 3x3 grid mislabels segments for subjects whose iris barely moves. Each op is
#   (pool_cells, [selector, ...])
# pool_cells = source cell labels READ FROM THE ORIGINAL (pre-override) labels.
# Selectors partition that pool sorted by iris_y ASCENDING (top=min y, bot=max y):
#   ("all", tgt)        - relabel every pooled segment to tgt
#   ("seg_top", n, tgt) - the n smallest-y segments
#   ("seg_bot", n, tgt) - the n largest-y segments
#   ("grp_low", n, tgt) - the n lowest (largest-y) SPATIAL groups  ("clusters")
#   ("rest", tgt)       - whatever remains in the middle
# Segments in no pool keep their original label.
OVERRIDES = {
    7:  [(("BL",), [("grp_low", 2, "BL"), ("rest", "ML")]),
         (("BC",), [("grp_low", 2, "BC"), ("rest", "MC")])],
    8:  [(("MR",), [("all", "MC")]),
         (("BR",), [("seg_top", 2, "MR"), ("rest", "BR")])],
    10: [(("MR",), [("all", "MC")]),
         (("BR",), [("seg_top", 2, "MR"), ("rest", "BR")])],
    12: [(("BC",), [("grp_low", 2, "BC"), ("rest", "MC")]),
         (("BR",), [("grp_low", 2, "BR"), ("rest", "MR")])],
    16: [(("TC",), [("seg_top", 3, "TC"), ("rest", "MC")])],
    17: [(("BR",), [("seg_bot", 3, "BR"), ("rest", "MR")])],
    18: [(("TC",), [("seg_top", 2, "TC"), ("rest", "MC")]),
         (("BR",), [("seg_bot", 2, "BC"), ("rest", "MR")]),
         (("MR",), [("seg_bot", 2, "BR"), ("rest", "MR")]),
         (("BC",), [("all", "MC")])],
    19: [(("TC",), [("seg_top", 2, "TC"), ("rest", "MC")]),
         (("BR",), [("seg_bot", 1, "BR"), ("rest", "MR")]),
         (("MR",), [("seg_bot", 1, "BR"), ("seg_top", 2, "MR"), ("rest", "MR")])],
    20: [(("TR", "MR", "BR"), [("seg_top", 2, "TR"), ("seg_bot", 2, "BR"), ("rest", "MR")])],
    21: [(("TC",), [("seg_top", 2, "TC"), ("rest", "MC")]),
         (("BC",), [("seg_bot", 2, "BC"), ("rest", "MC")])],
    22: [(("TR", "MR", "BR"), [("seg_top", 2, "TR"), ("seg_bot", 3, "BR"), ("rest", "MR")])],
}


def apply_overrides(sub, C, cell):
    """Re-label segments per the manual OVERRIDES table (no-op if none)."""
    ops = OVERRIDES.get(sub)
    if not ops:
        return cell.copy()
    orig = cell.copy(); new = cell.copy()
    for pool_cells, parts in ops:
        ids = {ABBR2ID[c] for c in pool_cells}
        idx = [i for i in range(len(orig)) if orig[i] in ids]
        idx.sort(key=lambda i: C[i, 1])               # y ascending: top -> bot
        for sel in parts:                              # "all" first
            if sel[0] == "all":
                for i in idx:
                    new[i] = ABBR2ID[sel[1]]
                idx = []
        front, back = 0, len(idx)
        for sel in parts:                              # n smallest-y
            if sel[0] == "seg_top":
                n, tgt = sel[1], sel[2]
                for i in idx[front:front + n]:
                    new[i] = ABBR2ID[tgt]
                front += n
        for sel in parts:                              # n largest-y
            if sel[0] == "seg_bot":
                n, tgt = sel[1], sel[2]
                for i in idx[back - n:back]:
                    new[i] = ABBR2ID[tgt]
                back -= n
        for sel in parts:                              # n bottom spatial groups
            if sel[0] == "grp_low":
                n, tgt = sel[1], sel[2]
                mid = idx[front:back]
                groups = []
                for i in mid:
                    if groups and C[i, 1] - C[groups[-1][-1], 1] <= 1.5:
                        groups[-1].append(i)
                    else:
                        groups.append([i])
                low = groups[-n:]
                for g in low:
                    for i in g:
                        new[i] = ABBR2ID[tgt]
                back -= sum(len(g) for g in low)
        for sel in parts:                              # middle remainder
            if sel[0] == "rest":
                for i in idx[front:back]:
                    new[i] = ABBR2ID[sel[1]]
    return new


def merge_runs(segs, C, dur, cell, p0, p1, max_gap=12, max_dist=3.0):
    """Merge consecutive segments that are the SAME cell, temporally continuous
    (frame gap <= max_gap) and spatially neighbouring (centre dist <= max_dist)
    into one cluster -- e.g. a single fixation split by a brief tracking dropout
    (sub6 segs 8 & 9). Returns parallel lists: runs (each a list of (s,e) frame
    sub-ranges), merged centres, merged durations, merged cell labels."""
    runs = [[segs[0]]]; gcell = [int(cell[0])]
    for k in range(1, len(segs)):
        s, e = segs[k]
        gap = s - runs[-1][-1][1]
        same = int(cell[k]) == gcell[-1]
        d = float(np.hypot(*(C[k] - C[k - 1])))
        if same and gap <= max_gap and d <= max_dist:
            runs[-1].append((s, e))
        else:
            runs.append([(s, e)]); gcell.append(int(cell[k]))
    mC = []; mdur = []
    for grp in runs:
        coords = np.concatenate([np.c_[p0[s:e], p1[s:e]] for s, e in grp]).astype(float)
        coords = coords[(coords[:, 0] > 0) & (coords[:, 1] > 0)]
        mC.append(np.median(coords, axis=0))
        mdur.append(int(sum(e - s for s, e in grp)))
    return runs, np.array(mC), np.array(mdur), np.array(gcell)


def cycle_phases(cell_seq, dur, min_edges=8, min_dur=8):
    """Principled before/after detector from the calibration structure:
    MC -> 8 edges (one by one, may be interrupted) -> MC, repeated TWICE.
      'before' = clusters before the first MC (the start of cycle 1) -- stray
                 pre-calibration fixations.
      'after'  = clusters after the SECOND return-to-MC (end of cycle 2).
    A 'return' is a SUBSTANTIAL MC (dur >= min_dur) reached after >= min_edges
    distinct edge cells since the last return -- so neither a mid-cycle glance at
    centre nor a brief blip is miscounted as a cycle boundary."""
    CENTER = 4; n = len(cell_seq)
    before = np.zeros(n, bool); after = np.zeros(n, bool)
    mc = [i for i, c in enumerate(cell_seq) if c == CENTER]
    if not mc:
        return before, after
    start = mc[0]
    before[:start] = True
    i = start                                   # skip the opening centre run
    while i < n and cell_seq[i] == CENTER:
        i += 1
    seen = set(); returns = 0
    while i < n:
        c = cell_seq[i]
        if c == CENTER and len(seen) >= min_edges and dur[i] >= min_dur:
            returns += 1; seen = set()
            j = i + 1
            while j < n and cell_seq[j] == CENTER:
                j += 1
            if returns == 2:
                after[j:] = True
                return before, after
            i = j; continue
        if c != CENTER:
            seen.add(c)
        i += 1
    return before, after


EDGE_IDS = [0, 1, 2, 3, 5, 6, 7, 8]              # all cells except MC(4)


def cycle_completeness(cell_kept, dur_kept, min_edges=8, min_dur=8, n_expected=2):
    """Split the KEPT cluster cells (temporal order) into the (2 expected)
    calibration cycles at the SUBSTANTIAL MC returns, and report each cycle's
    edge coverage. Rule: a complete cycle reaches every one of the 8 edges at
    least once. Returns a list of sorted missing-edge-name lists ([] = ok)."""
    CENTER = 4
    cycles = []; cur = set(); started = False
    for c, d in zip(cell_kept, dur_kept):
        if c == CENTER:
            if started and cur is not None and len(cur) >= min_edges and d >= min_dur:
                cycles.append(cur); cur = set()
                if len(cycles) >= n_expected:    # ignore any trailing fragment
                    cur = None
            started = True
        elif started and cur is not None:
            cur.add(int(c))
    if cur is not None and len(cur) >= 4:        # final partial cycle (no MC close)
        cycles.append(cur)
    return [sorted(NAMES[e] for e in EDGE_IDS if e not in cy) for cy in cycles]


# ---------------------------------------------------------------------------
# Post-MERGE manual edits, indexed by the MERGED cluster index shown in the
# time-order plots. Each subject may set:
#   "start": N    - clusters 0..N-1 are pre-calibration -> flagged 'before'
#   "end":   N    - clusters N+1.. are post-calibration -> flagged 'after'
#                   (an explicit 'end' OVERRIDES the auto MC-cycle detector)
#   "remove": [i] - individual stray/too-short clusters -> flagged 'removed'
#   "relabel": {i: "CELL"} - force a merged cluster's grid cell
# Principle: each calibration is MC -> 8 edges (one by one, may be interrupted)
# -> MC, repeated twice; so the kept clusters should form two such passes.
POST_MERGE_EDITS = {
    7:  {"start": 1, "end": 24},
    8:  {"start": 3, "end": 27, "relabel": {11: "BL", 21: "MR"}},
    9:  {"remove": [5], "end": 23, "keep": [0]},
    11: {"start": 5, "end": 23, "relabel": {6: "TR"}, "keep": [8]},
    12: {"remove": [9], "relabel": {3: "MR", 4: "TC", 6: "TR", 13: "BR"}, "keep": [21]},
    13: {"start": 1, "remove": [19]},
    14: {"start": 1},
    15: {"remove": [5, 13]},
    16: {"start": 2, "relabel": {18: "TC"}},
    17: {"end": 25, "relabel": {0: "MC", 13: "MC"}},
    18: {"relabel": {0: "MC", 1: "TL", 10: "BR"}, "keep": [2]},
    20: {"start": 0, "end": 24, "remove": [10, 11, 12, 13],
         "relabel": {0: "MC", 1: "BL", 2: "BR", 4: "BC", 9: "MC",
                     5: "MR", 17: "MR", 14: "MC", 16: "MC", 24: "MC"},
         "only": {"MR": [5, 17]}},
    21: {"start": 2},
    22: {"start": 3, "end": 35, "remove": [7, 14, 21, 23, 26, 29, 33],
         "relabel": {3: "MC", 11: "BR", 16: "MR", 22: "MR", 24: "MR", 32: "BR"},
         "only": {"MR": [16, 22, 24]}, "keep": [3, 10, 12]},
}


def apply_post_merge_edits(sub, cell, C, dur):
    """Apply the POST_MERGE_EDITS for one subject to the merged clusters.
    Relabels first, THEN derives the auto before/after from the MC-cycle
    structure (so MC relabels move the cycle boundaries). A manual 'start'/'end'
    overrides the auto 'before'/'after'. An 'only' rule restricts a cell to the
    listed clusters, reassigning any other kept cluster in that cell to its
    nearest OTHER kept-cell centroid. A 'keep' list force-keeps clusters (clears
    before/removed/after, and -- via the returned mask -- the off-grid flag), so
    a genuine edge wrongly discarded can be recovered. Returns
    (cell, before, removed, after, force_keep)."""
    nseg = len(cell)
    ed = POST_MERGE_EDITS.get(sub, {})
    for i, name in ed.get("relabel", {}).items():
        if i < nseg:
            cell[i] = ABBR2ID[name]
    before_auto, after_auto = cycle_phases(cell, dur)   # principled default
    if "start" in ed:
        before = np.zeros(nseg, bool); before[:ed["start"]] = True
    else:
        before = before_auto
    removed = np.zeros(nseg, bool)
    for i in ed.get("remove", []):
        if i < nseg:
            removed[i] = True
    if "end" in ed:
        after = np.zeros(nseg, bool); after[ed["end"] + 1:] = True
    else:
        after = after_auto
    force_keep = np.zeros(nseg, bool)
    for i in ed.get("keep", []):
        if i < nseg:
            force_keep[i] = True
    before[force_keep] = False; removed[force_keep] = False; after[force_keep] = False
    keep = ~(before | removed | after)
    for name, idxs in ed.get("only", {}).items():
        cid = ABBR2ID[name]; keepset = set(idxs)
        cent = {}                                    # centroids of the other cells
        for g in range(9):
            if g == cid:
                continue
            pts = [C[i] for i in range(nseg) if keep[i] and cell[i] == g]
            if pts:
                cent[g] = np.mean(pts, axis=0)
        for i in range(nseg):
            if keep[i] and cell[i] == cid and i not in keepset and cent:
                cell[i] = min(cent, key=lambda g: float(np.hypot(*(C[i] - cent[g]))))
    return cell, before, removed, after, force_keep


def process(sub):
    path = find_bin(sub)
    if path is None:
        return None
    with open(path, "rb") as f:
        h0 = np.load(f); h1 = np.load(f); p0 = np.load(f); p1 = np.load(f)
    cb = CALIB[sub]; n = len(p0)
    anch = anchor_grid(sub)
    if anch is None:
        return None
    segs = segment_30(p0, p1)
    C = np.array([[np.median(p0[s:e]), np.median(p1[s:e])] for s, e in segs])
    dur = np.array([e - s for s, e in segs])
    # inlier = close to one of the 9 anchor positions (so real grid columns are
    # kept; only genuine off-grid noise is excluded from the centroid fit).
    anchd = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1)
    pair = np.linalg.norm(anch[:, None, :] - anch[None, :, :], axis=2)
    np.fill_diagonal(pair, np.inf)
    min_spacing = pair.min()
    thr = max(3.0, 0.8 * min_spacing)
    inlier = anchd < thr
    cell, cc = cluster(C, anch, inlier)
    cell = apply_overrides(sub, C, cell)            # manual per-subject fixes
    # merge split/interrupted fixations (same cell, temporally + spatially close)
    runs, C, dur, cell = merge_runs(segs, C, dur, cell, p0, p1)
    nseg = len(runs)
    anchd = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1)
    inlier = anchd < thr                            # recomputed on merged centres
    # post-merge manual edits (relabel + before/after/removed discards);
    # before/after default to the principled MC-cycle detector (cycle_phases)
    cell, before, removed, after, force_keep = apply_post_merge_edits(sub, cell, C, dur)
    discard = before | removed | after
    # recompute the 9 grid centres from the FINAL labels (non-discarded only)
    for g in range(9):
        m = (cell == g) & ~discard
        if m.any():
            cc[g] = np.median(C[m], axis=0)
        else:
            cc[g] = np.nan
    # Rule 2: mistakes = off-grid segments (kept, not removed); discards excluded.
    # A segment is a mistake only if it is far from EVERY calibration anchor
    # (~inlier); distance to its own assigned-cell centre is no longer tested,
    # so manual overrides no longer get flagged as mistakes.
    mistake = (~inlier) & ~discard & ~force_keep    # 'keep' recovers off-grid edges

    # cycle-completeness rule: each cycle must reach all 8 edges at least once
    keep_idx = [i for i in range(nseg) if not (discard[i] or mistake[i])]
    kept_seq = np.array([cell[i] for i in keep_idx], dtype=int)
    kept_dur = np.array([dur[i] for i in keep_idx], dtype=int)
    cyc_missing = cycle_completeness(kept_seq, kept_dur)

    # per-frame labels (iterate the sub-runs of each merged cluster)
    frame_blink = load_blink(sub, n)                 # EAR blink frames -> unlinked
    frame_cell = np.full(n, -1); frame_seg = np.full(n, -1)
    frame_mist = np.zeros(n, bool); frame_dur = np.zeros(n, int)
    frame_after = np.zeros(n, bool); frame_before = np.zeros(n, bool)
    frame_removed = np.zeros(n, bool)
    for si, grp in enumerate(runs):
        for s, e in grp:
            frame_cell[s:e] = cell[si]; frame_seg[s:e] = si
            frame_mist[s:e] = mistake[si]; frame_dur[s:e] = dur[si]
            frame_after[s:e] = after[si]; frame_before[s:e] = before[si]
            frame_removed[s:e] = removed[si]

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/subid_{sub}_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["abs_frame", "local_frame", "iris_x", "iris_y", "in_fixation",
                    "seg_dur", "cell_id", "cell_name", "screen_x", "screen_y",
                    "screen_name", "is_mistake", "is_before", "is_removed", "is_after",
                    "is_blink"])
        for i in range(n):
            inf = frame_seg[i] >= 0
            blk = bool(frame_blink[i])
            bef = inf and frame_before[i]; rem = inf and frame_removed[i]
            aft = inf and frame_after[i]
            disc = bef or rem or aft
            # an EAR blink frame is never a usable fixation -> not linked
            ok = inf and not frame_mist[i] and not disc and not blk
            if ok:
                (sx, sy), snm = cell_to_screen(frame_cell[i])
            tag = ("blink" if blk else "before" if bef else "removed" if rem
                   else "after" if aft else (NAMES.get(frame_cell[i], "") if ok else ""))
            for_row = ([cb + i, i, f"{p0[i]:.0f}", f"{p1[i]:.0f}", int(inf),
                        frame_dur[i] if inf else "",
                        frame_cell[i] if ok else "", tag,
                        sx if ok else "", sy if ok else "", snm if ok else "",
                        int(frame_mist[i]) if inf else "",
                        int(bef) if inf else "", int(rem) if inf else "",
                        int(aft) if inf else "", int(blk)])
            w.writerow(for_row)
    with open(f"{OUT}/subid_{sub}_centers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "cell_name", "iris_x", "iris_y", "screen_x",
                    "screen_y", "screen_name", "n_seg", "n_frames"])
        for g in range(9):
            m = (cell == g) & ~mistake & ~discard
            (sx, sy), snm = cell_to_screen(g)
            cx = "" if np.isnan(cc[g, 0]) else f"{cc[g,0]:.1f}"
            cy = "" if np.isnan(cc[g, 1]) else f"{cc[g,1]:.1f}"
            w.writerow([g, NAMES[g], cx, cy, sx, sy, snm,
                        int(m.sum()), int(dur[m].sum())])

    # plot
    fig, ax = plt.subplots(figsize=(8, 7)); cmap = plt.cm.tab10
    for g in range(9):
        m = (cell == g) & ~mistake & ~discard; col, row = g // 3, g % 3
        nm = ["T", "M", "B"][row] + ["L", "C", "R"][col]
        ax.scatter(C[m, 0], C[m, 1], s=dur[m] * 4, color=cmap(g))
        if not np.isnan(cc[g, 0]):
            ax.annotate(nm, (cc[g, 0], cc[g, 1]), fontsize=11, weight="bold")
    ax.scatter(C[mistake, 0], C[mistake, 1], s=45, c="gray", marker="x", label="mistake")
    ax.scatter(C[before, 0], C[before, 1], s=55, c="tab:cyan", marker="^", label="before (discard)")
    ax.scatter(C[removed, 0], C[removed, 1], s=55, c="red", marker="v", label="removed (too short)")
    ax.scatter(C[after, 0], C[after, 1], s=55, c="black", marker="+", label="after (discard)")
    ax.legend()
    # zoom to the real clusters (kept, non-outlier) so off-grid mistakes/discards
    # don't squash the view; outliers fall outside the axes.
    keptm = ~mistake & ~discard
    Ck = C[keptm] if keptm.any() else C
    xlo, xhi = np.percentile(Ck[:, 0], [1, 99]); ylo, yhi = np.percentile(Ck[:, 1], [1, 99])
    mx = max(2.0, 0.15 * (xhi - xlo)); my = max(2.0, 0.15 * (yhi - ylo))
    ax.set_xlim(xlo - mx, xhi + mx); ax.set_ylim(yhi + my, ylo - my)   # y inverted
    ax.set_title(f"sub{sub}: 3x3 calib clusters ({nseg} clusters, {mistake.sum()} mist, "
                 f"{before.sum()} bef, {removed.sum()} rem, {after.sum()} aft)")
    plt.savefig(f"{OUT}/subid_{sub}_plot.png", dpi=70, bbox_inches="tight"); plt.close()

    empty = int(np.sum([((cell == g) & ~mistake & ~discard).sum() == 0 for g in range(9)]))
    n_cycles = len(cyc_missing)
    incomplete = [(ci, miss) for ci, miss in enumerate(cyc_missing) if miss]
    return dict(sub=sub, n_frames=n, n_seg=nseg, mistakes=int(mistake.sum()),
                before=int(before.sum()), removed=int(removed.sum()),
                after=int(after.sum()), empty_cells=empty,
                n_cycles=n_cycles, incomplete=incomplete)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in SUBS:
        r = process(sub)
        if r is None:
            print(f"sub{sub}: no binary"); continue
        rows.append(r)
        flag = "  <-- has empty cell!" if r["empty_cells"] else ""
        kept = r["n_seg"] - r["before"] - r["removed"] - r["after"]
        cyc = (f"{r['n_cycles']} cycles" if not r["incomplete"]
               else f"{r['n_cycles']} cycles INCOMPLETE " +
                    "; ".join(f"cyc{ci} missing {miss}" for ci, miss in r["incomplete"]))
        print(f"sub{sub}: {r['n_seg']} clusters ({kept} kept), {r['mistakes']} mist, "
              f"{r['before']} bef, {r['removed']} rem, {r['after']} aft, "
              f"{r['empty_cells']} empty, {cyc}{flag}")
    with open(f"{OUT}/SUMMARY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sub", "n_frames", "n_seg", "mistakes",
                                          "before", "removed", "after", "empty_cells",
                                          "n_cycles", "incomplete"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT}/  (clusters.csv, centers.csv, plot.png per subject + SUMMARY.csv)")


if __name__ == "__main__":
    main()
