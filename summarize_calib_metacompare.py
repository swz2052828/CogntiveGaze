"""Summarize + plot the deploy-faithful cross-backbone calibration sweep.

Reads runs/calib_metacompare/calib_metacompare_<bb>_calib.csv (deploy-faithful:
calib-trained meta/meta_adv + swarm-tuned SVR) and the matching old
calib_metacompare_<bb>.csv (in-task-support meta, fixed-default SVR), and writes:
  - SUMMARY_calib.csv / .md : per-backbone per-method mean (folds x K) + best calibrator
  - plots/best_calibrated_per_backbone.png
  - plots/oldvsnew_methods.png
  - plots/error_vs_K.png

Light: small CSV reads + 3 plots. OK on the login node.
  envs/gaze/bin/python summarize_calib_metacompare.py
"""
import csv, glob, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/springbrook/share/eng/esrpxk/runs/calib_metacompare"
OUT = os.path.join(D, "plots"); os.makedirs(OUT, exist_ok=True)
METHODS = ["base", "base_adv", "svr", "svr_embed", "fc_ft", "meta", "meta_adv"]
CALIB = ["svr", "svr_embed", "fc_ft", "meta", "meta_adv"]


def per_method_mean(path):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 25:
        return None
    a = {c: [] for c in METHODS}
    for r in rows:
        for c in METHODS:
            try: a[c].append(float(r[c]))
            except Exception: pass
    return {c: (sum(a[c]) / len(a[c]) if a[c] else float("nan")) for c in METHODS}


def per_K_mean(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        try: K = int(float(r["K"]))
        except Exception: continue
        for c in METHODS:
            try: by[K][c].append(float(r[c]))
            except Exception: pass
    return {K: {c: (sum(v[c]) / len(v[c]) if v[c] else float("nan")) for c in METHODS}
            for K, v in by.items()}


# ---- aggregate ----
new, old, perK = {}, {}, {}
for f in sorted(glob.glob(os.path.join(D, "calib_metacompare_*_calib.csv"))):
    bb = os.path.basename(f)[len("calib_metacompare_"):-len("_calib.csv")]
    m = per_method_mean(f)
    if not m: continue
    new[bb] = m
    perK[bb] = per_K_mean(f)
    o = per_method_mean(os.path.join(D, f"calib_metacompare_{bb}.csv"))
    if o: old[bb] = o

order = sorted(new, key=lambda b: min(new[b][c] for c in CALIB))

# ---- SUMMARY csv + md ----
with open(os.path.join(D, "SUMMARY_calib.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["backbone"] + METHODS + ["best", "best_err"])
    for bb in order:
        m = new[bb]; best = min(CALIB, key=lambda c: m[c])
        w.writerow([bb] + [f"{m[c]:.3f}" for c in METHODS] + [best, f"{m[best]:.3f}"])
with open(os.path.join(D, "SUMMARY_calib.md"), "w") as fh:
    fh.write("# Deploy-faithful calibration — cross-backbone summary (mean over folds x K)\n\n")
    fh.write("| backbone | " + " | ".join(METHODS) + " | best |\n")
    fh.write("|" + "---|" * (len(METHODS) + 2) + "\n")
    for bb in order:
        m = new[bb]; best = min(CALIB, key=lambda c: m[c])
        fh.write(f"| {bb} | " + " | ".join(f"{m[c]:.2f}" for c in METHODS) + f" | {best} {m[best]:.2f} |\n")

# ---- plot 1: best calibrated error per backbone ----
fig, ax = plt.subplots(figsize=(8, 7))
bes = [(bb, min(CALIB, key=lambda c: new[bb][c])) for bb in order]
errs = [new[bb][b] for bb, b in bes]
cmap = {"fc_ft": "#1f77b4", "meta": "#2ca02c", "meta_adv": "#ff7f0e", "svr_embed": "#9467bd", "svr": "#d62728"}
ax.barh(range(len(order)), errs, color=[cmap[b] for _, b in bes])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("best calibrated error (cm)"); ax.set_title("Best per-subject calibration error by backbone")
for i, (bb, b) in enumerate(bes): ax.text(errs[i] + 0.03, i, f"{b} {errs[i]:.2f}", va="center", fontsize=7)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=cmap[k], label=k) for k in cmap], fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "best_calibrated_per_backbone.png"), dpi=130); plt.close(fig)

# ---- plot 2: old vs new, cross-backbone mean, per method ----
common = [b for b in order if b in old]
mean_old = {c: sum(old[b][c] for b in common) / len(common) for c in CALIB}
mean_new = {c: sum(new[b][c] for b in common) / len(common) for c in CALIB}
fig, ax = plt.subplots(figsize=(7, 4.5))
x = range(len(CALIB)); w = 0.38
ax.bar([i - w / 2 for i in x], [mean_old[c] for c in CALIB], w, label="old (leaky-K / fixed SVR)", color="#bbbbbb")
ax.bar([i + w / 2 for i in x], [mean_new[c] for c in CALIB], w, label="new (deploy-faithful)", color="#1f77b4")
ax.set_xticks(list(x)); ax.set_xticklabels(CALIB); ax.set_ylabel("error (cm)")
ax.set_title(f"Old vs deploy-faithful, cross-backbone mean (n={len(common)})")
for i, c in enumerate(CALIB):
    ax.text(i + w / 2, mean_new[c] + 0.05, f"{mean_new[c]:.2f}", ha="center", fontsize=8)
    ax.text(i - w / 2, mean_old[c] + 0.05, f"{mean_old[c]:.2f}", ha="center", fontsize=8, color="#555")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "oldvsnew_methods.png"), dpi=130); plt.close(fig)

# ---- plot 3: error vs K, cross-backbone mean per method ----
Ks = sorted({K for bb in perK for K in perK[bb]})
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in CALIB + ["base"]:
    ys = []
    for K in Ks:
        vals = [perK[bb][K][c] for bb in perK if K in perK[bb]]
        ys.append(sum(vals) / len(vals) if vals else float("nan"))
    style = "--" if c == "base" else "-"
    ax.plot(Ks, ys, style, marker="o", label=c)
ax.set_xscale("log", base=2); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.set_xlabel("K (calibration frames)"); ax.set_ylabel("error (cm)")
ax.set_title("Error vs K, cross-backbone mean"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_vs_K.png"), dpi=130); plt.close(fig)

print(f"backbones summarized: {len(order)}")
print(f"wrote {D}/SUMMARY_calib.{{csv,md}} and {OUT}/*.png")
