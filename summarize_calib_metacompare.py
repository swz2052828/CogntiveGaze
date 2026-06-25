"""Summarize + plot the deploy-faithful cross-backbone calibration sweep.

Reports the IMPACT OF K and each backbone's accuracy at its BEST K (NOT a mean
over K -- averaging K=4..72 conflates sparse and dense calibration and hides the
real trend). Reads runs/calib_metacompare/calib_metacompare_<bb>_calib.csv and
the matching old calib_metacompare_<bb>.csv, and writes:
  - SUMMARY_calib.csv / .md : per-backbone per-method BEST-K error (+ the K) and
    the K-impact table (cross-backbone mean per method at each K)
  - plots/k_impact.png            : error vs K per method (cross-backbone mean)
  - plots/best_at_bestK.png       : each backbone's best calibrated error @ best K
  - plots/oldvsnew_bestK.png      : old vs new per method, each at its best K

Light: small CSV reads + 3 plots. OK on the login node.
  envs/gaze/bin/python summarize_calib_metacompare.py
"""
import csv, glob, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

D = "/springbrook/share/eng/esrpxk/runs/calib_metacompare"
OUT = os.path.join(D, "plots"); os.makedirs(OUT, exist_ok=True)
CALS = ["svr", "svr_embed", "fc_ft", "meta", "meta_adv"]
ALLM = ["base", "base_adv"] + CALS
KS = [4, 9, 18, 36, 72]
CMAP = {"fc_ft": "#1f77b4", "meta": "#2ca02c", "meta_adv": "#ff7f0e",
        "svr_embed": "#9467bd", "svr": "#d62728"}


def per_K(path):
    """mean over folds, per method, per K."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        try: K = int(float(r["K"]))
        except Exception: continue
        for c in ALLM:
            try: by[K][c].append(float(r[c]))
            except Exception: pass
    return {K: {c: (sum(v[c]) / len(v[c]) if v[c] else float("nan")) for c in ALLM}
            for K, v in by.items()}


# ---- load ----
new, old = {}, {}
for f in sorted(glob.glob(os.path.join(D, "calib_metacompare_*_calib.csv"))):
    bb = os.path.basename(f)[len("calib_metacompare_"):-len("_calib.csv")]
    d = list(csv.DictReader(open(f)))
    if len(d) < 25: continue
    new[bb] = per_K(f)
    op = os.path.join(D, f"calib_metacompare_{bb}.csv")
    if os.path.exists(op):
        od = list(csv.DictReader(open(op)))
        if len(od) >= 25: old[bb] = per_K(op)


def best_k(perk, method):
    """(err, K) minimizing over K for one method."""
    cand = [(perk[K][method], K) for K in KS if K in perk]
    return min(cand) if cand else (float("nan"), None)


# best calibrator (method, K, err) per backbone
best = {}
for bb in new:
    cand = [(new[bb][K][c], K, c) for c in CALS for K in KS if K in new[bb]]
    best[bb] = min(cand)  # (err, K, method)
order = sorted(new, key=lambda b: best[b][0])

# ---- SUMMARY md/csv ----
with open(os.path.join(D, "SUMMARY_calib.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["backbone", "base"] + [f"{c}_bestK" for c in CALS] + [f"{c}_K" for c in CALS] +
               ["best", "best_K", "best_err"])
    for bb in order:
        bk = {c: best_k(new[bb], c) for c in CALS}
        be, bK, bm = best[bb]
        w.writerow([bb, f"{new[bb][72]['base']:.3f}"] +
                   [f"{bk[c][0]:.3f}" for c in CALS] + [bk[c][1] for c in CALS] +
                   [bm, bK, f"{be:.3f}"])

with open(os.path.join(D, "SUMMARY_calib.md"), "w") as fh:
    fh.write("# Deploy-faithful calibration — cross-backbone summary (at BEST K, not mean over K)\n\n")
    fh.write("## Accuracy at each backbone's best K (per method best-K error; best = winning method @ its K)\n\n")
    fh.write("| backbone | base | svr | svr_embed | fc_ft | meta | meta_adv | **best @K** |\n")
    fh.write("|" + "---|" * 8 + "\n")
    for bb in order:
        bk = {c: best_k(new[bb], c) for c in CALS}
        be, bK, bm = best[bb]
        cells = " | ".join(f"{bk[c][0]:.2f}(K{bk[c][1]})" for c in CALS)
        fh.write(f"| {bb} | {new[bb][72]['base']:.2f} | {cells} | **{bm} {be:.2f}@K{bK}** |\n")
    fh.write("\n## Impact of K (cross-backbone mean error per method at each K)\n\n")
    fh.write("| method | " + " | ".join(f"K{k}" for k in KS) + " | K4->K72 |\n")
    fh.write("|" + "---|" * (len(KS) + 2) + "\n")
    for c in CALS:
        ys = [sum(new[bb][k][c] for bb in new if k in new[bb]) / len(new) for k in KS]
        fh.write(f"| {c} | " + " | ".join(f"{y:.2f}" for y in ys) + f" | {ys[0]-ys[-1]:+.2f} |\n")
    fh.write("\nError drops fastest by K9-K18 then plateaus; K36/72 add little. "
             "Best overall: {} + {} @K{} = {:.2f} cm.\n".format(
                 order[0], best[order[0]][2], best[order[0]][1], best[order[0]][0]))

# ---- plot 1: K-impact (cross-backbone mean per method) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in CALS:
    ys = [sum(new[bb][k][c] for bb in new if k in new[bb]) / len(new) for k in KS]
    ax.plot(KS, ys, "-o", color=CMAP[c], label=c)
ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("K (calibration frames)"); ax.set_ylabel("error (cm)")
ax.set_title("Impact of K: cross-backbone mean error per method")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "k_impact.png"), dpi=130); plt.close(fig)

# ---- plot 2: best calibrated error @ best K per backbone ----
fig, ax = plt.subplots(figsize=(8, 7))
errs = [best[bb][0] for bb in order]
ax.barh(range(len(order)), errs, color=[CMAP[best[bb][2]] for bb in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("best calibrated error @ best K (cm)")
ax.set_title("Best per-subject calibration accuracy by backbone (at its best K)")
for i, bb in enumerate(order):
    be, bK, bm = best[bb]
    ax.text(be + 0.03, i, f"{bm} {be:.2f} @K{bK}", va="center", fontsize=7)
ax.legend(handles=[mp.Patch(color=CMAP[k], label=k) for k in CMAP], fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "best_at_bestK.png"), dpi=130); plt.close(fig)

# ---- plot 3: old vs new at best K, per method (cross-backbone mean of best-K) ----
common = [b for b in order if b in old]
mo = {c: sum(best_k(old[b], c)[0] for b in common) / len(common) for c in CALS}
mn = {c: sum(best_k(new[b], c)[0] for b in common) / len(common) for c in CALS}
fig, ax = plt.subplots(figsize=(7, 4.5))
x = range(len(CALS)); w = 0.38
ax.bar([i - w / 2 for i in x], [mo[c] for c in CALS], w, label="old (leaky-K / fixed SVR)", color="#bbbbbb")
ax.bar([i + w / 2 for i in x], [mn[c] for c in CALS], w, label="new (deploy-faithful)", color="#1f77b4")
ax.set_xticks(list(x)); ax.set_xticklabels(CALS); ax.set_ylabel("error @ best K (cm)")
ax.set_title(f"Old vs deploy-faithful at best K, cross-backbone mean (n={len(common)})")
for i, c in enumerate(CALS):
    ax.text(i + w / 2, mn[c] + 0.05, f"{mn[c]:.2f}", ha="center", fontsize=8)
    ax.text(i - w / 2, mo[c] + 0.05, f"{mo[c]:.2f}", ha="center", fontsize=8, color="#555")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "oldvsnew_bestK.png"), dpi=130); plt.close(fig)

print(f"backbones: {len(order)} | best overall: {order[0]} {best[order[0]][2]} @K{best[order[0]][1]} = {best[order[0]][0]:.2f}")
print(f"wrote SUMMARY_calib.{{md,csv}} and plots/{{k_impact,best_at_bestK,oldvsnew_bestK}}.png")
