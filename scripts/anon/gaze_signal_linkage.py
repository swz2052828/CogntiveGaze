"""Does the OPEN tier leak? Linkage on the released gaze signal, not on pixels.

Our release plan puts per-frame gaze estimates and oculomotor measures in an
open tier on the grounds that it contains no imagery and no face embeddings. But
eye movement is an established biometric, so "no pixels" is not by itself an
argument -- it has to be measured, exactly as the pixel tiers were.

We initially intended to defend the open tier with a resolution argument (our
instrument is 30 Hz at ~5 cm, well below the operating points quoted for
oculomotor-plant biometrics). That argument does not survive contact with the
literature: one line of work reports 0.1 deg / 30 Hz as sufficient, and identification
remains above chance as spatial noise is raised to 0.5 deg. So we measure instead.

Attacker A2 (release only): they hold the open tier and ask whether it partitions
by participant. Features are computed per WINDOW of consecutive frames within a
recording, so the attacker sees short fragments -- the realistic case, since a
single long series per participant would be trivially separable by construction.

Reported against the same k=18 ARI convention as the pixel linkage, plus a
label-permutation null: the same pipeline on shuffled participant labels, which
is what "no per-participant signal" scores for this feature set and window count.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def window_features(xy):
    """Per-window descriptors an oculomotor-biometrics attacker would use.

    Deliberately restricted to quantities the open tier actually publishes:
    position dispersion, velocity statistics, and the saccade/fixation balance.
    No identity-specific engineering.
    """
    d = np.diff(xy, axis=0)
    v = np.linalg.norm(d, axis=1)
    if len(v) < 2:
        return None
    a = np.abs(np.diff(v))
    # saccade/fixation split at the cohort-independent 75th percentile of speed
    thr = np.percentile(v, 75)
    fast = v >= thr
    feats = [
        xy[:, 0].std(), xy[:, 1].std(),
        np.linalg.norm(xy - xy.mean(0), axis=1).mean(),      # spatial dispersion
        v.mean(), v.std(), np.percentile(v, 50), np.percentile(v, 95),
        a.mean(), a.std(),
        fast.mean(),                                          # fraction "saccadic"
        v[fast].mean() if fast.any() else 0.0,                # peak-ish velocity
        v[~fast].mean() if (~fast).any() else 0.0,            # drift during fixation
        np.abs(d[:, 0]).mean(), np.abs(d[:, 1]).mean(),
        np.corrcoef(xy[:, 0], xy[:, 1])[0, 1] if xy[:, 0].std() > 0 and xy[:, 1].std() > 0 else 0.0,
    ]
    f = np.asarray(feats, dtype=np.float64)
    return f if np.isfinite(f).all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=str(DATA / "ProcessedData" / "meanno7_clean" / "metadata.mat"))
    ap.add_argument("--window", type=int, default=150,
                    help="frames per window (30 fps -> 150 = 5 s fragments)")
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=20,
                    help="label-permutation replicates for the null")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="gaze_signal")
    args = ap.parse_args()

    m = sio.loadmat(args.manifest)
    rec = np.asarray(m["labelRecNum"]).ravel()
    fidx = np.asarray(m["frameIndex"]).ravel()
    x = np.asarray(m["labelDotXCam"]).ravel().astype(np.float64)
    y = np.asarray(m["labelDotYCam"]).ravel().astype(np.float64)

    X, lab = [], []
    for r in np.unique(rec):
        sel = rec == r
        order = np.argsort(fidx[sel])
        xy = np.stack([x[sel][order], y[sel][order]], axis=1)
        n = 0
        for s in range(0, len(xy) - args.window + 1, args.stride):
            f = window_features(xy[s:s + args.window])
            if f is not None:
                X.append(f); lab.append(int(r)); n += 1
        print(f"  rec {r}: {len(xy)} frames -> {n} windows", flush=True)

    X = np.asarray(X); lab = np.asarray(lab)
    subs = sorted(set(lab.tolist()))
    y_true = np.array([subs.index(v) for v in lab])
    # z-score per feature: an attacker would standardise before clustering
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"[{args.tag}] {X.shape[0]} windows, {X.shape[1]} features, {len(subs)} participants")

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    def purity(yt, yp):
        return float(sum(np.bincount(yt[yp == c]).max() for c in np.unique(yp)) / len(yt))

    km = KMeans(n_clusters=len(subs), n_init=10, random_state=0).fit(X)
    ari_k = float(adjusted_rand_score(y_true, km.labels_))
    pur_k = purity(y_true, km.labels_)

    best = (-2.0, None, None)
    for k in range(2, min(args.k_max, len(X) - 1) + 1):
        l = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        if len(np.unique(l)) < 2:
            continue
        s = silhouette_score(X, l, metric="cosine")
        if s > best[0]:
            best = (s, k, l)

    # Null: same pipeline, participant labels shuffled across windows.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.n_perm):
        yp = rng.permutation(y_true)
        null.append(float(adjusted_rand_score(yp, km.labels_)))

    res = {
        "tag": args.tag, "window_frames": args.window, "stride": args.stride,
        "n_windows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_participants": len(subs),
        "ari_known_k": ari_k, "purity_known_k": pur_k,
        "k_selected": best[1],
        "ari_auto_k": float(adjusted_rand_score(y_true, best[2])) if best[2] is not None else None,
        "silhouette": float(best[0]),
        "null_ari_mean": float(np.mean(null)), "null_ari_max": float(np.max(null)),
    }
    print(json.dumps(res, indent=2))
    if args.json_out:
        o = Path(args.json_out); o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(o, "w"), indent=2)
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
