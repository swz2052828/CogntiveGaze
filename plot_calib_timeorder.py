"""Temporal-order view of the calibration clusters: each segment plotted at its
iris position, connected to the next in TIME order by an arrow coloured along a
time colormap, labelled with its order index. Reuses the cluster_calib_points
pipeline (segment -> cluster -> overrides -> after) so it matches the CSVs.

Outputs under datasets/calib_clusters/timeorder/:
  subid_<id>_timeorder.png   per subject
  ALL_timeorder_montage.png  6x3 grid of all subjects
Light (matplotlib only) -> fine on the login node.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cluster_calib_points as M

OUT = f"{M.OUT}/timeorder"
ABBR = {0: "TL", 1: "ML", 2: "BL", 3: "TC", 4: "MC",
        5: "BC", 6: "TR", 7: "MR", 8: "BR"}
TCMAP = plt.cm.plasma          # time progression
CCMAP = plt.cm.tab10           # cell identity (matches the main plots)


def compute(sub):
    """Re-run the pipeline for one subject; return segment centres, durations,
    final cell labels and the 'after' mask, all in temporal order."""
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
    anchd = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1)
    pair = np.linalg.norm(anch[:, None, :] - anch[None, :, :], axis=2)
    np.fill_diagonal(pair, np.inf)
    thr = max(3.0, 0.8 * pair.min())
    inlier = anchd < thr
    cell, _ = M.cluster(C, anch, inlier)
    cell = M.apply_overrides(sub, C, cell)
    runs, C, dur, cell = M.merge_runs(segs, C, dur, cell, p0, p1)
    cell, before, removed, after, force_keep = M.apply_post_merge_edits(sub, cell, C, dur)
    inlier = np.linalg.norm(C[:, None, :] - anch[None, :, :], axis=2).min(1) < thr
    mistake = (~inlier) & ~(before | removed | after) & ~force_keep   # off-grid outliers
    return C, dur, cell, before, removed, after, mistake


def draw(ax, sub, data, label_every=1):
    C, dur, cell, before, removed, after, mistake = data
    n = len(C)
    # show ONLY kept clusters (before/removed/after/outlier are discarded -> omitted),
    # but keep each cluster's ORIGINAL time index as its label.
    kept = [i for i in range(n)
            if not (before[i] or removed[i] or after[i] or mistake[i])]
    K = C[kept]
    # time-coloured arrows between consecutive kept fixations
    for a, b in zip(kept[:-1], kept[1:]):
        ax.annotate("", xy=C[b], xytext=C[a],
                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                    color=TCMAP(kept.index(a) / max(1, len(kept) - 1)),
                                    alpha=0.7, shrinkA=4, shrinkB=4))
    for k, i in enumerate(kept):
        ax.scatter(C[i, 0], C[i, 1], s=max(20, dur[i] * 4),
                   color=CCMAP(cell[i] % 10), edgecolors="none", zorder=3)
        if k % label_every == 0:
            ax.annotate(str(i), C[i], fontsize=6, weight="bold",
                        ha="center", va="center", zorder=4)
    # clip to the robust range of the KEPT clusters
    xlo, xhi = np.percentile(K[:, 0], [2, 98]); ylo, yhi = np.percentile(K[:, 1], [2, 98])
    mx = max(2.0, 0.12 * (xhi - xlo)); my = max(2.0, 0.12 * (yhi - ylo))
    ax.set_xlim(xlo - mx, xhi + mx); ax.set_ylim(yhi + my, ylo - my)   # y inverted
    ax.set_title(f"sub{sub}  ({len(kept)} kept / {n} clusters)", fontsize=9)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for sub in M.SUBS:
        d = compute(sub)
        if d is None:
            continue
        results[sub] = d
        fig, ax = plt.subplots(figsize=(7, 6))
        draw(ax, sub, d)
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=TCMAP)
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("time order (start -> end)")
        ax.set_xlabel("iris_x"); ax.set_ylabel("iris_y")
        fig.savefig(f"{OUT}/subid_{sub}_timeorder.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"sub{sub}: time-order plot done", flush=True)

    subs = list(results)
    cols, rows = 3, int(np.ceil(len(subs) / 3))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.2))
    for ax, sub in zip(axs.ravel(), subs):
        draw(ax, sub, results[sub], label_every=2)
    for ax in axs.ravel()[len(subs):]:
        ax.axis("off")
    fig.suptitle("Calibration clusters in TIME order (arrow colour = time; "
                 "black ring = 'after'/discarded)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(f"{OUT}/ALL_timeorder_montage.png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT}/ (per-subject + ALL_timeorder_montage.png)")


if __name__ == "__main__":
    main()
