"""Aggregate the CLEAN metacompare (Stage B) across backbones. Text-only (no
matplotlib); reads runs/calib_metacompare/calib_metacompare_*_clean.csv (7 methods:
base, base_adv, svr, svr_embed, fc_ft, meta, meta_adv), fold-means per K, and
writes results/clean_metacompare_summary.md.

Focus = the base-vs-adv question on clean-trained models AFTER calibration:
  adv_raw_gain  = base - base_adv         (>0 = adv better UNcalibrated)
  adv_meta_gain = meta - meta_adv         (>0 = adv better after meta calibration)
  best method/K per backbone; cohort means at K=72 (best) and K=9 (real-world low-K).
"""
import csv
import glob
import collections
from pathlib import Path

import numpy as np

D = Path("/springbrook/share/eng/esrpxk/runs/calib_metacompare")
OUT = Path("/springbrook/share/eng/esrpxk/results/clean_metacompare_summary.md")
CALS = ["svr", "svr_embed", "fc_ft", "meta", "meta_adv"]
ALLM = ["base", "base_adv"] + CALS
KS = [4, 9, 18, 36, 72]


def per_K(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        try:
            K = int(float(r["K"]))
        except Exception:
            continue
        for c in ALLM:
            try:
                by[K][c].append(float(r[c]))
            except Exception:
                pass
    return {K: {c: (np.mean(v[c]) if v[c] else float("nan")) for c in ALLM} for K, v in by.items()}


def nfolds(path):
    return sum(1 for _ in csv.DictReader(open(path)))


data, nf = {}, {}
for f in sorted(glob.glob(str(D / "calib_metacompare_*_clean.csv"))):
    bb = Path(f).name[len("calib_metacompare_"):-len("_clean.csv")]
    data[bb] = per_K(f)
    nf[bb] = nfolds(f)


def best_method(perk):
    cand = [(perk[K][c], K, c) for c in CALS for K in KS if K in perk and not np.isnan(perk[K].get(c, float("nan")))]
    return min(cand) if cand else (float("nan"), None, "-")


order = sorted(data, key=lambda b: best_method(data[b])[0])
L = ["# CLEAN metacompare — cross-backbone (deploy-faithful calib support), cm", "",
     "base/base_adv = uncalibrated clean base vs clean adv. svr/svr_embed/fc_ft/meta = "
     "calibrated (from clean base). meta_adv = calibrated adv (adapter on adv features).",
     "adv_raw = base - base_adv (>0 adv better UNcalib); adv_meta = meta - meta_adv "
     "(>0 adv better after meta calib).", ""]

# main table @ K=72
hdr = (f"{'backbone':30s} {'base':>7} {'base_adv':>8} {'svr_emb':>7} {'fc_ft':>7} "
       f"{'meta':>7} {'meta_adv':>8} {'adv_raw':>7} {'adv_meta':>8} {'best@K':>14} {'cf':>3}")
L += ["## At K=72 (best-K regime)", "```", hdr, "-" * len(hdr)]
raw_gains, meta_gains, best_errs = [], [], []
for bb in order:
    d72 = data[bb].get(72, {})
    g = lambda c: d72.get(c, float("nan"))
    be, bK, bm = best_method(data[bb])
    adv_raw = g("base") - g("base_adv")
    adv_meta = g("meta") - g("meta_adv")
    if not np.isnan(adv_raw): raw_gains.append(adv_raw)
    if not np.isnan(adv_meta): meta_gains.append(adv_meta)
    if not np.isnan(be): best_errs.append(be)
    f = lambda v, w=7: (f"{v:.3f}" if not np.isnan(v) else "--").rjust(w)
    sgn = lambda v, w=7: (f"{v:+.3f}" if not np.isnan(v) else "--").rjust(w)
    L.append(f"{bb:30s} {f(g('base'))} {f(g('base_adv'),8)} {f(g('svr_embed'))} {f(g('fc_ft'))} "
             f"{f(g('meta'))} {f(g('meta_adv'),8)} {sgn(adv_raw)} {sgn(adv_meta,8)} "
             f"{(bm+'@'+str(bK)):>14} {nf[bb]:>3}")
L += ["-" * len(hdr),
      f"{'COHORT MEAN':30s} adv_raw={np.mean(raw_gains):+.3f} (adv better raw {sum(1 for x in raw_gains if x>0)}/{len(raw_gains)})  "
      f"adv_meta={np.mean(meta_gains):+.3f} (adv better meta {sum(1 for x in meta_gains if x>0)}/{len(meta_gains)})  "
      f"best_err_mean={np.mean(best_errs):.3f}",
      "```", ""]

# low-K real-world regime table @ K=9
L += ["## At K=9 (real-world low-K regime)", "```",
      f"{'backbone':30s} {'base':>7} {'svr_emb':>7} {'fc_ft':>7} {'meta':>7} {'meta_adv':>8}", "-" * 70]
for bb in order:
    d9 = data[bb].get(9, {})
    f = lambda c: (f"{d9.get(c, float('nan')):.3f}" if not np.isnan(d9.get(c, float('nan'))) else "--").rjust(7)
    L.append(f"{bb:30s} {f('base')} {f('svr_embed')} {f('fc_ft')} {f('meta')} {f('meta_adv'):>8}")
L += ["```", "", f"# clean metacompare: {len(data)} backbones aggregated"]

OUT.write_text("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwrote {OUT}  ({len(data)} backbones)")
