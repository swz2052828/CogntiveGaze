"""Build a manifest restricted to the frames a subsampled data root actually holds.

Why this exists: the utility evaluator reads the full ~11,718-frame manifest. Point
it at a root generated with --limit and roughly 90% of frames are missing; it warns
per frame and proceeds, producing plausible-looking wrong numbers (measured:
mobile_vit base 10.41 against a published 5.12). That failure cost 28 result files.

The structural fix is not a bigger data root -- we are inode-constrained, and a
full-count arm is ~630k files per operator -- but a manifest that matches the root.
Both arms of any comparison must then use the SAME subset manifest, so the
comparison is like-for-like and the frame count is a stated property rather than an
accident.

  python scripts/anon/make_subset_manifest.py --root .../anon_unmasked/blur0.50 \
      --src-mean meanno7_clean --out-mean meanno7_clean_sub1200
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root whose frames define the subset")
    ap.add_argument("--src-mean", default="meanno7_clean")
    ap.add_argument("--out-mean", required=True, help="manifest subdir name to create")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    src = DATA / "ProcessedData" / args.src_mean
    md = sio.loadmat(str(src / "metadata.mat"))

    rec = np.asarray(md["labelRecNum"]).ravel().astype(int)
    frm = np.asarray(md["frameIndex"]).ravel().astype(int)

    have = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        r = int(d.name)
        for f in (d / "appleFace").glob("*.jpg"):
            have.add((r, int(f.stem)))
    keep = np.array([(int(a), int(b)) in have for a, b in zip(rec, frm)])

    print(f"root {root.name}: {len(have)} frames present")
    print(f"manifest {args.src_mean}: {len(rec)} rows -> {int(keep.sum())} kept "
          f"({keep.mean()*100:.1f}%)")
    if keep.sum() == 0:
        raise SystemExit("FATAL: no manifest rows match this root")

    out = {}
    for k, v in md.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if a.ndim == 2 and a.shape[0] == 1 and a.shape[1] == len(rec):
            out[k] = a[:, keep]
        elif a.ndim == 2 and a.shape[0] == len(rec):
            out[k] = a[keep, :]
        else:
            out[k] = a
            print(f"  (passed through unchanged: {k} {a.shape})")

    if args.dry_run:
        print("dry run, nothing written")
        return
    dst = DATA / "ProcessedData" / args.out_mean
    dst.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(dst / "metadata.mat"), out)
    for m in src.glob("mean_*.mat"):          # mean images are frame-independent
        link = dst / m.name
        if not link.exists():
            link.symlink_to(m)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
