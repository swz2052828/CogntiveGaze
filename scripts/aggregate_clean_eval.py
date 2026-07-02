"""Aggregate Step-1 clean-vs-dirty eval CSVs (one per backbone) into a summary
table: per backbone, fold-mean coord error over ALL val frames (dirty) vs the
non-blink subset (clean), and the inflation delta. Sorted by clean error.
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

# optional arg: results subdir name (e.g. clean_eval_adv); default clean_eval
_NAME = sys.argv[1] if len(sys.argv) > 1 else "clean_eval"
D = Path(f"/springbrook/share/eng/esrpxk/results/{_NAME}")
OUT = Path(f"/springbrook/share/eng/esrpxk/results/{_NAME}_summary.md")


def main():
    rows = []
    for fp in sorted(glob.glob(str(D / "*.csv"))):
        rr = list(csv.DictReader(open(fp)))
        if not rr:
            continue
        bb = rr[0]["backbone"]
        all_e = np.array([float(r["err_all"]) for r in rr])
        clean_e = np.array([float(r["err_clean"]) for r in rr])
        blink_e = np.array([float(r["err_blink"]) for r in rr if r["err_blink"] != "nan"])
        nval = sum(int(r["n_val"]) for r in rr)
        nbl = sum(int(r["n_blink"]) for r in rr)
        rows.append(dict(
            backbone=bb, folds=len(rr), n_val=nval, n_blink=nbl,
            err_all=all_e.mean(), err_clean=clean_e.mean(),
            err_blink=blink_e.mean() if len(blink_e) else float("nan"),
            delta=all_e.mean() - clean_e.mean()))
    rows.sort(key=lambda r: r["err_clean"])

    hdr = f"{'backbone':32s} {'folds':>5} {'n_val':>7} {'n_blk':>6} {'err_all':>8} {'err_clean':>9} {'err_blink':>9} {'delta':>7}"
    lines = ["# Step 1: blink-frame inflation of held-out val error (cm)",
             "", "Existing (blink-contaminated-trained) base checkpoints, evaluated on each",
             "fold's held-out val recordings. err_all = mean over all val frames; err_clean =",
             "mean over non-blink frames; err_blink = mean over the +/-1-dilated blink frames.",
             "delta = err_all - err_clean = how much blink frames inflate the reported number.",
             "", "```", hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['backbone']:32s} {r['folds']:>5} {r['n_val']:>7} {r['n_blink']:>6} "
                     f"{r['err_all']:>8.4f} {r['err_clean']:>9.4f} {r['err_blink']:>9.4f} {r['delta']:>+7.4f}")
    if rows:
        ma = np.mean([r["err_all"] for r in rows]); mc = np.mean([r["err_clean"] for r in rows])
        lines += ["-" * len(hdr),
                  f"{'COHORT MEAN':32s} {'':>5} {'':>7} {'':>6} {ma:>8.4f} {mc:>9.4f} {'':>9} {ma-mc:>+7.4f}"]
    lines.append("```")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}  ({len(rows)} backbones)")


if __name__ == "__main__":
    main()
