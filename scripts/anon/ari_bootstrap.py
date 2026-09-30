"""Confidence intervals and clustering variability for the ARI linkage numbers.

Every ARI in the paper so far is a single KMeans fit at random_state=0, reported
as a point estimate. Two things are missing and a reviewer will ask for both:

  seed spread   KMeans is not deterministic across initialisations. If ARI moves
                with the seed, a single fit is not a measurement.

  subject CI    Participants are the sampling unit. Resampling them with
                replacement and refitting gives an interval that reflects the
                n = 18 cohort rather than the n = 2160 frames, which are not
                independent.

On the bootstrap: a resample contains duplicated participants. A duplicate is
the SAME person with byte-identical embeddings, so it keeps its participant's
label and k is the number of DISTINCT participants drawn (about 11-12 of 18).

A first version gave each duplicate its own label. KMeans cannot split two
identical copies of the same points into two clusters, so every duplicate was
scored as an error by construction, and the resulting "interval" sat entirely
BELOW the point estimate for all ten arms (e.g. 0.844 with CI [0.442, 0.700]).
The guard below now reports where the point estimate falls in the bootstrap
distribution and refuses to write an interval that excludes it.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def cluster_ari(X, y, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    return float(adjusted_rand_score(y, lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb", help="npz with X1/y1/X2/y2 from periocular_attack --emb-out")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--tag", default="ari")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    z = np.load(args.emb, allow_pickle=True)
    X = np.vstack([z["X1"], z["X2"]]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    lab = np.concatenate([z["y1"], z["y2"]])
    subs = sorted(set(lab.tolist()))
    y = np.array([subs.index(v) for v in lab])
    n_sub = len(subs)
    rows = {s: np.flatnonzero(y == i) for i, s in enumerate(subs)}

    # --- clustering variability across KMeans seeds -----------------------
    seed_vals = [cluster_ari(X, y, n_sub, s) for s in range(args.n_seeds)]

    # --- subject-level cluster bootstrap ----------------------------------
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.n_boot):
        drawn = rng.choice(subs, size=n_sub, replace=True)
        idx, lbl = [], []
        for s in drawn:
            r = rows[s]
            idx.append(r)
            lbl.append(np.full(len(r), subs.index(s)))   # a duplicate is the same person
        idx = np.concatenate(idx)
        lbl = np.concatenate(lbl)
        k = len(set(drawn.tolist()))
        boot.append(cluster_ari(X[idx], lbl, k, 0))

    b = np.asarray(boot)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    frac_above = float((b > seed_vals[0]).mean())
    # An unbiased bootstrap puts the point estimate inside its own interval.
    # If it does not, the resampling is broken, and an interval must not be written.
    if not (lo <= seed_vals[0] <= hi):
        raise SystemExit(
            f"[{args.tag}] BIASED BOOTSTRAP: point {seed_vals[0]:.3f} lies outside "
            f"[{lo:.3f}, {hi:.3f}] ({frac_above:.2f} of replicates above it). "
            f"Refusing to write an interval.")
    res = {
        "tag": args.tag, "source": str(Path(args.emb).name),
        "n_frames": int(len(X)), "n_participants": n_sub,
        "ari_point": seed_vals[0],
        "ari_seed_mean": float(np.mean(seed_vals)),
        "ari_seed_sd": float(np.std(seed_vals, ddof=1)),
        "ari_seed_min": float(np.min(seed_vals)),
        "ari_seed_max": float(np.max(seed_vals)),
        "n_seeds": args.n_seeds,
        "ari_boot_mean": float(b.mean()),
        "ari_boot_ci95": [lo, hi],
        "frac_boot_above_point": frac_above,
        "n_boot": args.n_boot,
    }
    print(json.dumps(res, indent=2))
    if args.json_out:
        o = Path(args.json_out); o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(o, "w"), indent=2)
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
