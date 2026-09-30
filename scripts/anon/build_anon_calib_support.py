"""Build anonymised per-subject enrolment sets: calib_support_K{K}_<op>.

Condition (iii) of the three-condition utility design needs enrolment frames that
were anonymised by the SAME operator as deployment. This builds them.

Why it operates on stored crops rather than full frames: measured on task data,
where both paths exist, applying an operator directly to the stored crop differs
from applying it to the full frame and re-cropping by MAE 0.49-0.52 -- the JPEG
re-encode noise floor -- with the border no worse than the interior even at
sigma = 0.5*IOD. So the crop-direct path is equivalent within re-encoding, which
is what makes K=9/18/36 available from calib_processed (all 1200 calibration
frames) instead of requiring video decoding.

Two scale facts this depends on:
  * calibration crops are stored RESIZED to 224 (task crops are native), so
    operator strength must use IOD measured in 224-face-crop pixels, not the
    task-side native IOD.
  * that IOD is per-subject (cohort 59.9-75.7 px, 26% spread), so a cohort
    constant is not good enough.

The `none` operator is the control arm and is NOT a copy: it is regenerated
through the identical decode/encode path so condition (i) carries the same number
of interpolations as (ii) and (iii).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
# Geometry is per-K, because the K levels are NOT nested: the K9 selection is not
# a subset of K72, so geometry recovered on one does not cover the other.
def geom_dir(k):
    return DATA / f"anon_calib_geometry_k{k}"

# Pixels are read from the SAME root whose filenames we iterate -- the released
# calibration support -- and never from a second root. The first version of this
# script read filenames from `calib_support_K<K>` but pixels from
# `calib_processed`, which holds a 1.40x wider face box (MAE 39.4 levels between
# them, against a 0.11 re-encode floor). Nothing downstream noticed: the crops
# were the right size, the right frames, and visually the right face. It cost the
# whole first condition (iii) run. Anonymising the released crops in place makes
# that mistake unrepresentable.

# IOD in 224-face-crop pixels comes from the geometry file rather than a
# hard-coded table, so it can never drift out of step with the crops it describes.
TASK_RATIO = (0.36, 0.52)   # IOD / face-crop width in the task data: 0.40-0.47


def iod_for(rec, geo_npz):
    """Median inter-ocular distance in face-crop pixels, with a convention check."""
    iod = float(np.median(geo_npz["iod_crop"]))
    ratio = iod / 224.0
    if not (TASK_RATIO[0] <= ratio <= TASK_RATIO[1]):
        raise SystemExit(
            f"FATAL [{rec}] geometry gives IOD/face-width {ratio:.3f}, outside "
            f"{TASK_RATIO}. The geometry was recovered against a face-crop "
            f"convention that does not match the task data. Refusing to build.")
    return iod


def load_geometry(rec, k):
    """Per-frame eye boxes in face-crop coords, keyed by frame stem.

    Merges the K-selection recovery (needed_<rec>.npz, the frames this builder
    actually consumes) with the earlier general sample. Each entry carries the
    worse of the two eye-box match correlations so callers can screen on it.
    """
    out = {}
    for name in (f"{rec}.npz", f"needed_{rec}.npz"):
        p = geom_dir(k) / name
        if not p.is_file():
            continue
        z = np.load(p)
        for i, f in enumerate([str(x) for x in z["frames"]]):
            L, R = z["appleLeftEye"][i], z["appleRightEye"][i]
            out[Path(f).stem] = (L[:4], R[:4], min(float(L[5]), float(R[5])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--k", type=int, required=True, choices=(4, 9, 18, 36, 72))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--allow-median-geometry", action="store_true",
                    help="Fall back to the per-subject median eye box for frames "
                         "with no recovered geometry, or whose match correlation "
                         "is below --min-corr. Position sd inside the 224 crop is "
                         "<=2.6 px for 17/18 subjects but 5.4 px for 00020, so "
                         "this is a documented approximation, not a silent "
                         "default; the fallback count is reported per subject.")
    ap.add_argument("--min-corr", type=float, default=0.95,
                    help="Per-FRAME template-match floor. 00010 (10 frames) and "
                         "00012 (19 frames) fall below 0.95; both wear glasses, "
                         "so the likely cause is a lens specular reflection "
                         "defeating the match on particular frames.")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "maf", str(Path(__file__).with_name("make_anon_frames.py")))
    maf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maf)

    src_support = DATA / f"calib_support_K{args.k}"
    out = Path(args.out_root or DATA / f"calib_support_K{args.k}_{args.op}")
    recs = sorted(d.name for d in src_support.iterdir()
                  if d.is_dir() and d.name.isdigit())

    report = {"op": args.op, "k": args.k, "recs": {}, "median_fallbacks": 0,
              "missing_geometry": []}
    for rec in recs:
        geo = load_geometry(rec, args.k)
        med = None
        good = [v for v in geo.values() if v[2] >= args.min_corr]
        if good:
            med = (np.median(np.stack([v[0] for v in good]), 0),
                   np.median(np.stack([v[1] for v in good]), 0))
        iod = iod_for(rec, np.load(geom_dir(args.k) / f"{rec}.npz"))

        names = sorted(p.name for p in (src_support / rec / "appleFace").glob("*.jpg"))
        for sub in ("appleFace", "appleLeftEye", "appleRightEye"):
            (out / rec / sub).mkdir(parents=True, exist_ok=True)

        n_fallback = 0
        for name in names:
            face = cv2.imread(str(src_support / rec / "appleFace" / name))
            if face is None:
                report["missing_geometry"].append(f"{rec}/{name}:no_crop")
                continue
            entry = geo.get(Path(name).stem)
            if entry is not None and entry[2] >= args.min_corr:
                boxes = entry[:2]
            else:
                reason = "no_geometry" if entry is None else f"corr<{args.min_corr}"
                if not args.allow_median_geometry or med is None:
                    report["missing_geometry"].append(f"{rec}/{name}:{reason}")
                    continue
                boxes = med
                n_fallback += 1

            outimg = maf.apply_op(face.copy(), args.op, iod)
            for b in boxes:                       # eye preservation, byte-exact
                x, y, w, h = [int(round(v)) for v in b]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(face.shape[1], x + w), min(face.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    outimg[y0:y1, x0:x1] = face[y0:y1, x0:x1]

            q = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            cv2.imwrite(str(out / rec / "appleFace" / name), outimg, q)
            for sub in ("appleLeftEye", "appleRightEye"):
                eye = cv2.imread(str(src_support / rec / sub / name))
                if eye is not None:               # unchanged, but re-encoded
                    cv2.imwrite(str(out / rec / sub / name), eye, q)

        report["recs"][rec] = {"frames": len(names), "median_fallback": n_fallback}
        report["median_fallbacks"] += n_fallback

    # manifest subdirs (mean_path points at these, not at a mean image)
    for m in ("mean7", "meanno7", "meanno7_clean"):
        src = src_support / m
        dst = out / m
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    print(json.dumps({k: v for k, v in report.items() if k != "recs"}, indent=1))
    print(f"recordings {len(report['recs'])}, "
          f"frames/rec {sorted({v['frames'] for v in report['recs'].values()})}, "
          f"median fallbacks {report['median_fallbacks']}, "
          f"missing {len(report['missing_geometry'])}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
