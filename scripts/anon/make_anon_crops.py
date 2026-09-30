"""Cheap (non-generative) anonymisation operators over a crop root.

Smoke-scale version: operates directly on the stored crops. The full study
applies the same operators to the FULL frame and re-crops with the recovered
geometry, which is equivalent for these local operators but also produces the
eye crops consistently.

Operators: blur<sigma> | pixel<block> | blackbox | noise<sigma>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def apply_op(img, op):
    kind = op.rstrip("0123456789.")
    param = op[len(kind):]
    if kind == "blur":
        s = float(param)
        k = int(2 * round(3 * s) + 1)
        return cv2.GaussianBlur(img, (k, k), s)
    if kind == "pixel":
        b = int(param)
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if kind == "blackbox":
        return np.zeros_like(img)
    if kind == "noise":
        s = float(param)
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)
    raise ValueError(f"unknown operator {op!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/springbrook/share/eng/esrpxk/datasets/ProcessedData")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--folders", nargs="+", default=["appleFace"])
    ap.add_argument("--recs", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="frames per recording")
    args = ap.parse_args()

    src, dst = Path(args.src_root), Path(args.dst_root)
    recs = args.recs or sorted(d.name for d in src.iterdir() if d.is_dir() and d.name.isdigit())
    n = 0
    for rec in recs:
        for folder in args.folders:
            sd = src / rec / folder
            if not sd.is_dir():
                continue
            dd = dst / rec / folder
            dd.mkdir(parents=True, exist_ok=True)
            names = sorted(p.name for p in sd.glob("*.jpg"))
            if args.limit:
                names = names[::max(1, len(names) // args.limit)][: args.limit]
            for name in names:
                img = cv2.imread(str(sd / name))
                if img is None:
                    continue
                cv2.imwrite(str(dd / name), apply_op(img, args.op),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                n += 1
        print(f"  {rec} done ({n} images)", flush=True)
    print(f"[{args.op}] wrote {n} images to {dst}")


if __name__ == "__main__":
    main()
