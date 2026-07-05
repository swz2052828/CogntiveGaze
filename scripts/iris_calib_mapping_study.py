"""Iris-location calibration mapping study (Experiment 2 extracted_data).

Data (user's own iris detector):
  datasets/extracted_data/subid_<n>/subid_<n>_<segstart>   : 4 concatenated .npy
      arrays (head_x, head_y, iris_x_rel, iris_y_rel), one value per frame of a
      task segment starting at video frame <segstart>.
  .../subid_<n>_calib_<f> : same 4 arrays, length 10 = the subject's calibration
      fixations (MC -> 8 edges -> MC, first cycle).

Targets for the 10 calib fixations: recovered per subject from
datasets/calib_clusters/subid_<n>_clusters.csv -- the temporal sequence of kept
fixation runs (cell -> screen mapping incl. the horizontal flip) of cycle 1.

Ground truth for task frames: meanno7 manifest labelDotXCam/YCam (cm), converted
to screen px by the project convention sx=x/54.4*1920+960, sy=y/30.4*1080+540.
Errors reported in cm (px error / (1920/54.4)).

Methods (fit on 10 calib pairs, predict all task frames):
  projective (baseline, DLT homography, 2D iris)   affine (lstsq)
  poly2 / poly3 (ridge-regularised polynomial)     tps (thin-plate spline)
  rbf (gaussian)                                   svr (RBF, C=100)
  each in 2D (iris only) and 4D (head + iris) variants where applicable.
"""
import glob
import io
import os
import re
import sys
import collections

import numpy as np

DATA = "/springbrook/share/eng/esrpxk/datasets"
PX_PER_CM = 1920 / 54.4


def read_concat_npy(path):
    arrs = []
    with open(path, "rb") as fh:
        buf = fh.read()
    bio = io.BytesIO(buf)
    while bio.tell() < len(buf):
        arrs.append(np.lib.format.read_array(bio))
    return arrs


def load_subject_files(sub):
    d = f"{DATA}/extracted_data/subid_{sub}"
    calib, segs = None, {}
    for fp in glob.glob(f"{d}/subid_{sub}_*"):
        name = os.path.basename(fp)
        if name.endswith(".xlsx"):
            continue
        m = re.match(rf"subid_{sub}_calib_(\d+)$", name)
        if m:
            arr = np.stack(read_concat_npy(fp))            # (4, 10)
            calib = arr if calib is None else np.concatenate([calib, arr], axis=1)
            continue
        m = re.match(rf"subid_{sub}_(\d+)$", name)
        if m:
            segs[int(m.group(1))] = np.stack(read_concat_npy(fp))   # (4, N)
    return calib, segs


# The user's 10 calib values are in CANONICAL GRID ORDER (verified 2026-07-05
# via auto-correspondence): iris cells row-major with mid-center duplicated.
# cell_to_screen flip: iris col 0/1/2 -> screen col 2/1/0.
_SCREEN = {(0,0):(217,146),(0,1):(115,540),(0,2):(217,934),
           (1,0):(960,92),(1,1):(960,540),(1,2):(960,988),
           (2,0):(1703,146),(2,1):(1805,540),(2,2):(1703,934)}
CELL_ORDER = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8]

def fixed_targets():
    out = []
    for c in CELL_ORDER:
        col, row = c // 3, c % 3
        out.append(_SCREEN[(2 - col, row)])
    return np.array(out, float)


def calib_targets_UNUSED(sub):
    """First-cycle fixation-run sequence -> 10 screen targets (px)."""
    import csv
    fp = f"{DATA}/calib_clusters/subid_{sub}_clusters.csv"
    rows = [r for r in csv.DictReader(open(fp))
            if r["is_mistake"] == "0" and r["is_before"] == "0"
            and r["is_removed"] == "0" and r["is_after"] == "0" and r["is_blink"] == "0"]
    rows.sort(key=lambda r: int(r["abs_frame"]))
    runs = []
    for r in rows:
        key = (r["cell_id"], r["screen_x"], r["screen_y"])
        if not runs or runs[-1][0] != key:
            runs.append([key, int(r["abs_frame"])])
    seq = [(float(k[1]), float(k[2])) for k, _ in runs]
    return np.array(seq[:10])                              # cycle 1 = 10 fixations


def gt_map(sub):
    import scipy.io as sio
    md = sio.loadmat(f"{DATA}/ProcessedData/meanno7/metadata.mat", squeeze_me=True)
    recs = np.asarray(md["labelRecNum"]).astype(int)
    frs = np.asarray(md["frameIndex"]).astype(int)
    gx = np.asarray(md["labelDotXCam"], float)
    gy = np.asarray(md["labelDotYCam"], float)
    m = recs == sub
    sx = gx[m] / 54.4 * 1920 + 960
    sy = gy[m] / 30.4 * 1080 + 540
    return dict(zip(frs[m].tolist(), np.stack([sx, sy], 1)))


# ---------------- mapping methods ----------------
def fit_homography(src, dst):
    """DLT: src(N,2)->dst(N,2), N>=4."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)

    def predict(P):
        Ph = np.c_[P, np.ones(len(P))] @ H.T
        return Ph[:, :2] / Ph[:, 2:3]
    return predict


def fit_affine(src, dst):
    A = np.c_[src, np.ones(len(src))]
    W, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return lambda P: np.c_[P, np.ones(len(P))] @ W


def fit_poly(src, dst, deg, alpha=1e-6):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    pf = PolynomialFeatures(deg)
    X = pf.fit_transform(src)
    ms = [Ridge(alpha=alpha).fit(X, dst[:, i]) for i in (0, 1)]
    return lambda P: np.stack([m.predict(pf.transform(P)) for m in ms], 1)


def fit_rbfi(src, dst, kernel, **kw):
    from scipy.interpolate import RBFInterpolator
    r = RBFInterpolator(src, dst, kernel=kernel, **kw)
    return lambda P: r(P)


def fit_svr(src, dst, C=100.0, eps=1.0):
    from sklearn.svm import SVR
    ms = [SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale").fit(src, dst[:, i])
          for i in (0, 1)]
    return lambda P: np.stack([m.predict(P) for m in ms], 1)


def methods_for(dim):
    m = {
        f"affine_{dim}d": fit_affine,
        f"poly2_{dim}d": lambda s, d: fit_poly(s, d, 2),
        f"tps_{dim}d": lambda s, d: fit_rbfi(s, d, "thin_plate_spline", smoothing=1.0),
        f"rbf_{dim}d": lambda s, d: fit_rbfi(s, d, "gaussian", epsilon=5.0, smoothing=1.0),
        f"svr_{dim}d": fit_svr,
    }
    if dim == 2:
        m["projective_2d"] = fit_homography          # the user's baseline
        m["poly3_2d"] = lambda s, d: fit_poly(s, d, 3)
    return m


def main():
    subs = sorted(int(os.path.basename(p).split("_")[1])
                  for p in glob.glob(f"{DATA}/extracted_data/subid_*"))
    results = collections.defaultdict(dict)
    counts = {}
    for sub in subs:
        try:
            calib, segs = load_subject_files(sub)
            if calib is None or not segs:
                print(f"sub {sub}: missing calib or segs, skip", flush=True); continue
            ncal = calib.shape[1]
            tgt = np.tile(fixed_targets(), (ncal // 10, 1))
            gt = gt_map(sub)
        except Exception as e:
            print(f"sub {sub}: load error {e}", flush=True); continue

        hx, hy, ix, iy = calib
        C2 = np.stack([ix, iy], 1)                      # iris-relative
        C4 = np.stack([hx, hy, ix, iy], 1)

        # task frames: build X and matched GT
        X2, X4, Y = [], [], []
        for start, arr in segs.items():
            n = arr.shape[1]
            for j in range(n):
                f = start + j
                if f in gt:
                    X2.append([arr[2, j], arr[3, j]])
                    X4.append([arr[0, j], arr[1, j], arr[2, j], arr[3, j]])
                    Y.append(gt[f])
        X2, X4, Y = np.asarray(X2), np.asarray(X4), np.asarray(Y)
        # per-segment: median iris of each task segment, target = segment's modal GT
        S2, S4, SY = [], [], []
        for start, arr in segs.items():
            fr = [start + j for j in range(arr.shape[1])]
            ok = [j for j, f in enumerate(fr) if f in gt]
            if len(ok) < 10:
                continue
            g = np.array([gt[fr[j]] for j in ok])
            m = np.median(arr[:, ok], axis=1)
            S2.append([m[2], m[3]]); S4.append(m.tolist())
            SY.append(np.median(g, axis=0))
        S2, S4, SY = np.asarray(S2), np.asarray(S4), np.asarray(SY)
        counts[sub] = len(Y)
        if len(Y) < 100:
            print(f"sub {sub}: only {len(Y)} matched frames, skip"); continue

        # signal-ceiling oracles (fit = eval on task data; NOT deployable)
        for dim, Xt in ((2, X2), (4, X4)):
            results[f"ORACLE_affine_{dim}d"][sub] = float(np.mean(
                np.linalg.norm(fit_affine(Xt, Y)(Xt) - Y, axis=1)) / PX_PER_CM)
            results[f"ORACLE_poly2_{dim}d"][sub] = float(np.mean(
                np.linalg.norm(fit_poly(Xt, Y, 2)(Xt) - Y, axis=1)) / PX_PER_CM)
        def mm(Xt, Xc):
            """moment-match task features to the calib feature distribution (label-free)"""
            return (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9) * Xc.std(0) + Xc.mean(0)
        evals = {"": (X2, X4, Y), "_mm": (mm(X2, C2), mm(X4, C4), Y),
                 "_seg": (S2, S4, SY), "_segmm": (mm(S2, C2), mm(S4, C4), SY)}
        for suf, (E2, E4, EY) in evals.items():
            for dim, Xc, Xt in ((2, C2, E2), (4, C4, E4)):
                for name, fit in methods_for(dim).items():
                    try:
                        pred = fit(Xc, tgt)(Xt)
                        e = np.linalg.norm(pred - EY, axis=1) / PX_PER_CM
                        results[name + suf][sub] = float(np.mean(e))
                    except Exception as ex:
                        results[name + suf][sub] = float("nan")
        print(f"sub {sub}: {len(Y)} frames  " + " ".join(
            f"{k}={results[k][sub]:.2f}" for k in
            ("projective_2d", "affine_2d", "poly2_2d", "tps_2d", "poly2_4d") if sub in results[k]),
            flush=True)

    # summary
    names = sorted(results, key=lambda n: np.nanmean(list(results[n].values())))
    L = ["# Iris-location calibration mapping study (Experiment 2 extracted_data), cm",
         "", f"subjects: {[s for s in counts]}, matched task frames per subject: "
         f"{ {k: v for k, v in counts.items()} }", "",
         "| method | cohort mean | median | vs projective |", "|---|---|---|---|"]
    base = np.nanmean(list(results["projective_2d"].values()))
    for n in names:
        vals = [results[n].get(s, float("nan")) for s in counts]
        mu = np.nanmean(vals)
        L.append(f"| {n} | {mu:.3f} | {np.nanmedian(vals):.3f} | {base-mu:+.3f} |")
    L += ["", "## per-subject best method", "| sub | best | err | projective err |", "|---|---|---|---|"]
    for s in counts:
        per = {n: results[n].get(s, float("nan")) for n in names}
        b = min(per, key=lambda n: per[n] if per[n] == per[n] else 9e9)
        L.append(f"| {s} | {b} | {per[b]:.3f} | {per.get('projective_2d', float('nan')):.3f} |")
    out = "/springbrook/share/eng/esrpxk/results/iris_calib_mapping_summary.md"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
