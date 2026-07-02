"""Pick ONE canonical run-dir per distinct backbone architecture and emit a
manifest (TSV) that drives the clean eval + retrain arrays.

Selection per backbone: prefer a dedicated single-backbone run-dir whose name
matches *_lr1e4 (the canonical LR), else any dedicated dir with 5 folds, else
the dir with the most folds. The shared runs/meta_pipeline/base dir holds
several backbones; we only fall back to it for backbones with no dedicated dir.

Columns: run_subdir  backbone  output_activation  gaze_range  lr  epochs  use_grid
(recipe read from that run's fold0 base checkpoint args).
"""
import glob
import re
from pathlib import Path

import torch

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
OUT = Path("/springbrook/share/eng/esrpxk/CogntiveGaze/scripts/clean_canonical_runs.tsv")


def fold_ckpts(run, backbone):
    base = RUNS / run / "base" / "seed42"
    return sorted(base.glob(f"fold*_best_{backbone}_gaze_segmenter.pth"))


def main():
    # backbone -> list of (run_subdir, nfolds, is_dedicated, is_lr1e4)
    cand = {}
    for d in sorted(RUNS.glob("meta_pipeline*")):
        base = d / "base" / "seed42"
        if not base.is_dir():
            continue
        run = d.name
        bbs = {}
        for ck in base.glob("fold*_best_*_gaze_segmenter.pth"):
            bb = ck.name.split("_best_")[1].rsplit("_gaze_segmenter", 1)[0]
            bbs.setdefault(bb, 0)
            bbs[bb] += 1
        dedicated = len(bbs) == 1
        for bb, nf in bbs.items():
            cand.setdefault(bb, []).append(
                (run, nf, dedicated, bool(re.search(r"_lr1e4$", run))))

    def rank(c):
        run, nf, ded, lr1 = c
        return (ded, lr1, nf == 5, nf)  # prefer dedicated, lr1e4, exactly-5, more folds

    picks = {}
    for bb, lst in cand.items():
        picks[bb] = sorted(lst, key=rank, reverse=True)[0]

    rows = []
    for bb in sorted(picks):
        run, nf, ded, lr1 = picks[bb]
        cks = fold_ckpts(run, bb)
        a = torch.load(cks[0], map_location="cpu", mmap=True).get("args", {})
        oa = a.get("output_activation") or "none"
        gr = a.get("gaze_range")
        gr = 4.0 if gr is None else float(gr)
        lr = a.get("lr", 1e-4)
        ep = a.get("epochs", 20)
        ug = 1 if a.get("use_grid", False) else 0
        rows.append((run, bb, oa, gr, lr, ep, ug, len(cks)))

    with open(OUT, "w") as f:
        f.write("run_subdir\tbackbone\toutput_activation\tgaze_range\tlr\tepochs\tuse_grid\tnfolds\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {OUT} with {len(rows)} backbones")
    for r in rows:
        print(f"  {r[1]:32s} <- {r[0]:42s} oa={r[2]} gr={r[3]} lr={r[4]} ep={r[5]} grid={r[6]} folds={r[7]}")


if __name__ == "__main__":
    main()
