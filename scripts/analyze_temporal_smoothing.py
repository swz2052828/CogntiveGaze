"""Temporal smoothing analysis (#5) over per-frame prediction dumps from
calib_eval_extras.py (runs/calib_extras/dumps/preds_<bb>_fold<F>.csv.gz).

Filters (per recording, ordered by frame index; windows reset across gaps >
GAP_MAX frames so we never smooth across seek/segment boundaries):
  - centered median, width w in {3,5,9,15}       (offline)
  - causal EMA, alpha in {0.3,0.5,0.7}           (real-time)
  - blink-aware: linearly interpolate predictions over blink frames
    (clean==0) from neighbouring clean frames, then centered median-5.

Errors reported on ALL frames and on CLEAN frames only, for raw vs filtered,
for each prediction column (base / fcft / meta). CPU-light.
"""
import argparse
import glob
import gzip
from pathlib import Path

import numpy as np

GAP_MAX = 30


def _segments(frames):
    """Contiguous runs (index spans) where frame gaps <= GAP_MAX."""
    cuts = np.where(np.diff(frames) > GAP_MAX)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts + 1, len(frames)]
    return list(zip(starts, ends))


def median_filter(x, w):
    if w <= 1 or len(x) < 3:
        return x.copy()
    h = w // 2
    out = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - h), min(len(x), i + h + 1)
        out[i] = np.median(x[lo:hi], axis=0)
    return out


def ema_filter(x, alpha):
    out = x.copy()
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def blink_interp(x, clean):
    """Replace pred on blink frames by linear interp between clean neighbours."""
    out = x.copy()
    idx = np.arange(len(x))
    good = clean.astype(bool)
    if good.sum() < 2:
        return out
    for d in range(x.shape[1]):
        out[~good, d] = np.interp(idx[~good], idx[good], x[good, d])
    return out


def apply_per_rec(frames, x, clean, fn, needs_clean=False):
    out = np.empty_like(x)
    for s, e in _segments(frames):
        seg = fn(x[s:e], clean[s:e]) if needs_clean else fn(x[s:e])
        out[s:e] = seg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-glob", default="/springbrook/share/eng/esrpxk/runs/calib_extras/dumps/preds_*_fold*.csv.gz")
    ap.add_argument("--out", default="/springbrook/share/eng/esrpxk/results/temporal_smoothing_summary.md")
    args = ap.parse_args()

    # collect: backbone -> {col -> {filter -> [per-rec errors all], [clean]}}
    results = {}
    for fp in sorted(glob.glob(args.dump_glob)):
        name = Path(fp).name[len("preds_"):-len(".csv.gz")]
        bb = name.rsplit("_fold", 1)[0]
        raw = np.genfromtxt(gzip.open(fp, "rt"), delimiter=",", names=True)
        if raw.size == 0:
            continue
        recs = raw["rec"].astype(int)
        R = results.setdefault(bb, {})
        for rec in np.unique(recs):
            m = recs == rec
            fr = raw["frame"][m].astype(int)
            order = np.argsort(fr)
            fr = fr[order]
            gt = np.stack([raw["gt_x"][m][order], raw["gt_y"][m][order]], 1)
            clean = raw["clean"][m][order].astype(int)
            for col in ("base", "fcft", "meta"):
                p = np.stack([raw[f"{col}_x"][m][order], raw[f"{col}_y"][m][order]], 1)
                variants = {"raw": p}
                for w in (3, 5, 9, 15):
                    variants[f"med{w}"] = apply_per_rec(fr, p, clean, lambda s, w=w: median_filter(s, w))
                for a in (0.3, 0.5, 0.7):
                    variants[f"ema{a}"] = apply_per_rec(fr, p, clean, lambda s, a=a: ema_filter(s, a))
                bi = apply_per_rec(fr, p, clean, blink_interp, needs_clean=True)
                variants["blinkI"] = bi
                variants["blinkI+med5"] = apply_per_rec(fr, bi, clean, lambda s: median_filter(s, 5))
                D = R.setdefault(col, {})
                for k, v in variants.items():
                    e = np.linalg.norm(v - gt, axis=1)
                    d = D.setdefault(k, {"all": [], "clean": []})
                    d["all"].append(float(e.mean()))
                    if clean.sum():
                        d["clean"].append(float(e[clean.astype(bool)].mean()))

    L = ["# Temporal smoothing at inference (opt #5), cm", "",
         "Per-recording mean error, averaged over recordings+folds. 'all' includes blink",
         "frames; 'clean' = blink-free frames only. med=centered median (offline),",
         "ema=causal EMA (real-time), blinkI=linear interp over blink frames.", ""]
    for bb, R in sorted(results.items()):
        L.append(f"## {bb}")
        L.append("```")
        filters = list(next(iter(R.values())).keys())
        L.append(f"{'filter':14s}" + "".join(f"{c+'(all)':>12}{c+'(cl)':>12}" for c in R))
        for f in filters:
            row = f"{f:14s}"
            for c in R:
                row += f"{np.mean(R[c][f]['all']):>12.4f}{np.mean(R[c][f]['clean']):>12.4f}"
            L.append(row)
        # best filter per column on all-frames
        for c in R:
            best = min(filters, key=lambda f: np.mean(R[c][f]["all"]))
            gain = np.mean(R[c]["raw"]["all"]) - np.mean(R[c][best]["all"])
            L.append(f"# {c}: best={best} gain={gain:+.4f} (vs raw, all frames)")
        L.append("```")
        L.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
