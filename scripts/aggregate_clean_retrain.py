"""Aggregate Step-2: clean-trained models' held-out val error, and compare to
the Step-1 dirty-trained numbers. All errors in cm, fold-mean.

Three quantities per backbone (identical clean val frame-sets, so comparable):
  dirty_dirty = dirty-trained, dirty-val   (original reported number; clean_eval err_all)
  dirty_clean = dirty-trained, clean-val   (Step 1; clean_eval err_clean)
  clean_clean = clean-trained, clean-val   (Step 2; best val_error of the retrained ckpts)

train_gain = dirty_clean - clean_clean  (does removing blink TRAINING frames help,
measured on the same clean val).
"""
import csv
import glob
import re
from pathlib import Path

import numpy as np
import torch

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
EVAL = Path("/springbrook/share/eng/esrpxk/results/clean_eval")
OUT = Path("/springbrook/share/eng/esrpxk/results/clean_retrain_summary.md")


def dirty_numbers(bb):
    fp = EVAL / f"{bb}.csv"
    if not fp.is_file():
        return None, None
    rr = list(csv.DictReader(open(fp)))
    da = np.mean([float(r["err_all"]) for r in rr])
    dc = np.mean([float(r["err_clean"]) for r in rr])
    return da, dc


def clean_numbers(bb):
    vals = []
    for d in RUNS.glob("meta_pipeline_clean_*"):
        for ck in d.glob(f"base/seed42/fold*_best_{bb}_gaze_segmenter.pth"):
            try:
                vals.append(float(torch.load(ck, map_location="cpu", mmap=True)["val_error"]))
            except Exception:
                pass
    return (np.mean(vals), len(vals)) if vals else (None, 0)


def main():
    backbones = sorted(p.stem for p in EVAL.glob("*.csv"))
    rows = []
    for bb in backbones:
        da, dc = dirty_numbers(bb)
        cc, nf = clean_numbers(bb)
        rows.append(dict(backbone=bb, dirty_dirty=da, dirty_clean=dc,
                         clean_clean=cc, nfolds=nf,
                         train_gain=(dc - cc) if (cc is not None and dc is not None) else None))
    rows.sort(key=lambda r: (r["clean_clean"] is None, r["clean_clean"] if r["clean_clean"] is not None else 0))

    hdr = f"{'backbone':32s} {'dirty/dirty':>11} {'dirty/clean':>11} {'clean/clean':>11} {'train_gain':>10} {'cf':>3}"
    L = ["# Step 2: retrain on blink-cleaned data vs originals (val coord error, cm)",
         "", "dirty/dirty = original (blink-contaminated train + val).",
         "dirty/clean = original model, blink frames removed from val (Step 1).",
         "clean/clean = retrained on clean data, blink frames removed from val (Step 2).",
         "train_gain  = dirty/clean - clean/clean  (>0 means cleaning the TRAIN data helped).",
         "cf = clean retrain folds done.", "", "```", hdr, "-" * len(hdr)]
    done = [r for r in rows if r["clean_clean"] is not None]
    for r in rows:
        dd = f"{r['dirty_dirty']:.4f}" if r['dirty_dirty'] is not None else "  --"
        dc = f"{r['dirty_clean']:.4f}" if r['dirty_clean'] is not None else "  --"
        cc = f"{r['clean_clean']:.4f}" if r['clean_clean'] is not None else "  pending"
        tg = f"{r['train_gain']:+.4f}" if r['train_gain'] is not None else "  --"
        L.append(f"{r['backbone']:32s} {dd:>11} {dc:>11} {cc:>11} {tg:>10} {r['nfolds']:>3}")
    if done:
        mdc = np.mean([r["dirty_clean"] for r in done])
        mcc = np.mean([r["clean_clean"] for r in done])
        L += ["-" * len(hdr),
              f"{'COHORT MEAN (done only)':32s} {'':>11} {mdc:>11.4f} {mcc:>11.4f} {mdc-mcc:>+10.4f} {len(done):>3}"]
    L.append("```")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}  ({len(done)}/{len(rows)} backbones have clean retrains)")


if __name__ == "__main__":
    main()
