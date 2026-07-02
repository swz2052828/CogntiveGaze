"""Plots + markdown for the CLEAN metacompare (_clean.csv). Adapted from
summarize_calib_metacompare.py: k-impact (cohort MEDIAN, robust to meta_adv
divergences), best-@-best-K leaderboard, and an adv-effect plot (base_adv-base,
meta_adv-meta per backbone). Writes runs/calib_metacompare/plots_clean/.
Light single-threaded matplotlib (login-node ok)."""
import csv, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

D = "/springbrook/share/eng/esrpxk/runs/calib_metacompare"
OUT = os.path.join(D, "plots_clean"); os.makedirs(OUT, exist_ok=True)
CALS = ["svr", "svr_embed", "fc_ft", "meta", "meta_adv"]
ALLM = ["base", "base_adv"] + CALS
KS = [4, 9, 18, 36, 72]
CMAP = {"fc_ft": "#1f77b4", "meta": "#2ca02c", "meta_adv": "#ff7f0e",
        "svr_embed": "#9467bd", "svr": "#d62728"}


def per_K(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        try: K = int(float(r["K"]))
        except Exception: continue
        for c in ALLM:
            try: by[K][c].append(float(r[c]))
            except Exception: pass
    return {K: {c: (np.mean(v[c]) if v[c] else float("nan")) for c in ALLM} for K, v in by.items()}


data = {}
for f in sorted(glob.glob(os.path.join(D, "calib_metacompare_*_clean.csv"))):
    bb = os.path.basename(f)[len("calib_metacompare_"):-len("_clean.csv")]
    data[bb] = per_K(f)


def best_k(perk, c):
    cand = [(perk[K][c], K) for K in KS if K in perk and not np.isnan(perk[K].get(c, np.nan))]
    return min(cand) if cand else (float("nan"), None)


best = {bb: min((data[bb][K][c], K, c) for c in CALS for K in KS
                if K in data[bb] and not np.isnan(data[bb][K].get(c, np.nan))) for bb in data}
order = sorted(data, key=lambda b: best[b][0])

# ---- plot 1: K-impact, cohort MEDIAN per method (robust to meta_adv blowups) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in CALS:
    ys = [np.median([data[bb][k][c] for bb in data if k in data[bb] and not np.isnan(data[bb][k][c])]) for k in KS]
    ax.plot(KS, ys, "-o", color=CMAP[c], label=c)
ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("K (calibration frames)"); ax.set_ylabel("cohort MEDIAN error (cm)")
ax.set_title("Clean metacompare — K impact (median per method)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "k_impact_clean.png"), dpi=130); plt.close(fig)

# ---- plot 2: best calibrated error @ best K per backbone ----
fig, ax = plt.subplots(figsize=(8, 8))
errs = [best[bb][0] for bb in order]
ax.barh(range(len(order)), errs, color=[CMAP[best[bb][2]] for bb in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("best calibrated error @ best K (cm)")
ax.set_title("Clean: best per-subject calibration accuracy by backbone")
for i, bb in enumerate(order):
    be, bK, bm = best[bb]
    ax.text(be + 0.03, i, f"{bm} {be:.2f}@K{bK}", va="center", fontsize=7)
ax.legend(handles=[mp.Patch(color=CMAP[k], label=k) for k in CMAP], fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "best_at_bestK_clean.png"), dpi=130); plt.close(fig)

# ---- plot 3: adv effect per backbone (uncalibrated & meta-calibrated), K=72 ----
fig, ax = plt.subplots(figsize=(8, 8))
raw = [data[bb][72]["base"] - data[bb][72]["base_adv"] for bb in order]      # >0 adv better
met = [data[bb][72]["meta"] - data[bb][72]["meta_adv"] for bb in order]
CLIP = 1.5
rawc = [np.clip(v, -CLIP, CLIP) for v in raw]; metc = [np.clip(v, -CLIP, CLIP) for v in met]
y = np.arange(len(order)); h = 0.4
ax.barh(y - h/2, rawc, h, color="#888", label="adv_raw (base - base_adv)")
ax.barh(y + h/2, metc, h, color="#2ca02c", label="adv_meta (meta - meta_adv)")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
ax.set_xlim(-CLIP, CLIP); ax.set_xlabel("cm  (>0 = adv better;  clipped to ±1.5)")
ax.set_title("Clean: adv effect @K72 (bars right = adv helps)")
for i, (r, m) in enumerate(zip(raw, met)):
    if abs(m) > CLIP: ax.text(np.clip(m,-CLIP,CLIP), i + h/2, f"  {m:+.0f}", va="center", fontsize=6, color="red")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "adv_effect_clean.png"), dpi=130); plt.close(fig)

# ---- markdown ----
with open(os.path.join(D, "SUMMARY_clean.md"), "w") as fh:
    fh.write("# Clean metacompare — leaderboard (best method @ best K), cm\n\n")
    fh.write("| backbone | base | svr_embed | fc_ft | meta | **best @K** |\n|" + "---|"*6 + "\n")
    for bb in order:
        bk = {c: best_k(data[bb], c) for c in ["svr_embed", "fc_ft", "meta"]}
        be, bK, bm = best[bb]
        fh.write(f"| {bb} | {data[bb][72]['base']:.2f} | "
                 + " | ".join(f"{bk[c][0]:.2f}(K{bk[c][1]})" for c in ['svr_embed','fc_ft','meta'])
                 + f" | **{bm} {be:.2f}@K{bK}** |\n")

print(f"backbones={len(order)}  best={order[0]} {best[order[0]][2]}@K{best[order[0]][1]}={best[order[0]][0]:.2f}")
print("wrote plots_clean/{k_impact_clean,best_at_bestK_clean,adv_effect_clean}.png + SUMMARY_clean.md")
