"""Aggregate the ADV clean-retrain (Step-2 for subject-adversarial models) and
compare to: adv dirty/clean (existing adv on clean val) and base clean/clean
(plain clean-retrain). All cm, fold-mean.

  adv_dirty_clean  = existing adv model, clean val   (results/clean_eval_adv)
  adv_clean_clean  = adv retrained on clean data      (runs/meta_pipeline_clean_adv_*)
  adv_train_gain   = adv_dirty_clean - adv_clean_clean (>0 = cleaning train helped adv)
  base_clean_clean = plain base retrained on clean     (runs/meta_pipeline_clean_*)
  adv_vs_base      = base_clean_clean - adv_clean_clean (>0 = adv beats base, both clean-trained)
"""
import csv
import glob
from pathlib import Path

import numpy as np
import torch

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
EVAL_ADV = Path("/springbrook/share/eng/esrpxk/results/clean_eval_adv")
OUT = Path("/springbrook/share/eng/esrpxk/results/clean_retrain_adv_summary.md")


def ckpt_mean(globpat, exclude=None):
    vals = []
    for ck in glob.glob(globpat):
        if exclude and exclude in ck:
            continue
        try:
            vals.append(float(torch.load(ck, map_location="cpu", mmap=True)["val_error"]))
        except Exception:
            pass
    return (np.mean(vals), len(vals)) if vals else (None, 0)


def csv_mean(fp, col):
    if not Path(fp).is_file():
        return None
    rr = list(csv.DictReader(open(fp)))
    return np.mean([float(r[col]) for r in rr]) if rr else None


def main():
    backbones = sorted(p.stem for p in EVAL_ADV.glob("*.csv"))
    rows = []
    for bb in backbones:
        adc = csv_mean(EVAL_ADV / f"{bb}.csv", "err_clean")
        acc, nf = ckpt_mean(str(RUNS / f"meta_pipeline_clean_adv_*/base/seed42/fold*_best_{bb}_gaze_segmenter.pth"))
        # base clean dirs are meta_pipeline_clean_<x>; exclude the adv ones (_clean_adv_)
        bcc, _ = ckpt_mean(str(RUNS / f"meta_pipeline_clean_*/base/seed42/fold*_best_{bb}_gaze_segmenter.pth"),
                           exclude="meta_pipeline_clean_adv_")
        rows.append(dict(bb=bb, adc=adc, acc=acc, nf=nf, bcc=bcc,
                         tg=(adc - acc) if (adc is not None and acc is not None) else None,
                         vb=(bcc - acc) if (bcc is not None and acc is not None) else None))
    rows.sort(key=lambda r: (r["acc"] is None, r["acc"] if r["acc"] is not None else 0))

    hdr = f"{'backbone':32s} {'adv_dirty/cl':>12} {'adv_clean/cl':>12} {'adv_gain':>9} {'base_clean/cl':>13} {'adv_vs_base':>11} {'cf':>3}"
    L = ["# ADV clean-retrain (Step-2 for subject-adversarial models), cm",
         "", "adv_dirty/cl = existing adv on clean val; adv_clean/cl = adv retrained on clean.",
         "adv_gain = adv_dirty/cl - adv_clean/cl (>0 = cleaning train helped adv).",
         "adv_vs_base = base_clean/cl - adv_clean/cl (>0 = adv beats base, both clean-trained).",
         "", "```", hdr, "-" * len(hdr)]
    done = [r for r in rows if r["acc"] is not None]
    for r in rows:
        f = lambda v, w=12, p=4: (f"{v:.{p}f}" if v is not None else "pending").rjust(w)
        g = lambda v, w=9: (f"{v:+.4f}" if v is not None else "--").rjust(w)
        L.append(f"{r['bb']:32s} {f(r['adc'])} {f(r['acc'])} {g(r['tg'])} {f(r['bcc'],13)} {g(r['vb'],11)} {r['nf']:>3}")
    if done:
        mad = np.mean([r["adc"] for r in done if r["adc"] is not None])
        mac = np.mean([r["acc"] for r in done])
        nbv = [r for r in done if r["vb"] is not None]
        adv_beats_base = sum(1 for r in nbv if r["vb"] > 0)
        L += ["-" * len(hdr),
              f"{'COHORT MEAN (done)':32s} {mad:>12.4f} {mac:>12.4f} {mad-mac:>+9.4f}",
              f"# adv clean-retrain done: {len(done)} backbones; adv beats base (clean-trained) on {adv_beats_base}/{len(nbv)}"]
    L.append("```")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}  ({len(done)}/{len(rows)} adv retrains done)")


if __name__ == "__main__":
    main()
