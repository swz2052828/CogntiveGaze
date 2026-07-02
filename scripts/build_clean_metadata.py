"""Build a blink-cleaned metadata.mat by dropping the contaminated (rec,frame)
rows from meanno7/metadata.mat. Light single-file I/O -- login-node safe.

Removal set = datasets/ear_vs_gt/train_blink_overlap/blink_contaminated_frames_dilate1.csv
(the +/-1-dilated EAR|GT blink frames, 4888 rows across train+test splits).
The original metadata.mat is left untouched; output goes to meanno7_clean/.
"""
import csv
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path("/springbrook/share/eng/esrpxk/datasets")
SRC = ROOT / "ProcessedData" / "meanno7"
DST = ROOT / "ProcessedData" / "meanno7_clean"
CSV = ROOT / "ear_vs_gt" / "train_blink_overlap" / "blink_contaminated_frames_dilate1.csv"

PER_ROW = ["labelRecNum", "frameIndex", "labelDotXCam", "labelDotYCam",
           "labelFaceGrid", "labelVal", "labelTrain", "labelTest"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    drop = set()
    for r in csv.DictReader(open(CSV)):
        drop.add((int(r["recNum"]), int(r["frameIndex"])))
    print(f"contaminated (rec,frame) pairs to drop: {len(drop)}")

    md = sio.loadmat(SRC / "metadata.mat", squeeze_me=True, struct_as_record=False)
    rec = np.asarray(md["labelRecNum"]).astype(int)
    frm = np.asarray(md["frameIndex"]).astype(int)
    n = len(rec)
    keep = np.ones(n, bool)
    for i in range(n):
        if (rec[i], frm[i]) in drop:
            keep[i] = False
    print(f"rows: {n} -> {int(keep.sum())} kept ({n - int(keep.sum())} removed)")

    out = {}
    for k in PER_ROW:
        a = np.asarray(md[k])
        out[k] = a[keep]
    sio.savemat(DST / "metadata.mat", out, do_compression=True)
    print(f"wrote {DST/'metadata.mat'}")

    # mean_*.mat are not read by the multistream loader (ImageNet norm) but copy
    # them so meanno7_clean is a self-contained drop-in mean-path.
    for f in SRC.glob("mean_*.mat"):
        shutil.copy2(f, DST / f.name)
        print(f"copied {f.name}")

    # sanity: per-split removal counts
    import collections
    c = collections.Counter()
    for r in csv.DictReader(open(CSV)):
        c[r["split"]] += 1
    print("removed per split (from csv):", dict(c))


if __name__ == "__main__":
    main()
