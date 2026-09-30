"""Is the periocular leak carried by session imaging conditions rather than by the person?

The reviewer's objection, and ours: every recording is a single session, so
exposure, white balance, focus distance and the direction of the light source on
the cornea are constant within a participant and differ between participants.
The face-free background floor controls for the ROOM; it does not control for
those, because they are properties of the imaging of the PERSON. Without a second
session we cannot settle this, but we can attack the specific channel named.

Three probes, in increasing strength:

  illum   A positive control on the suspected confound. Cluster the eye crops
          using ONLY photometric summary statistics -- per-channel mean, SD,
          percentiles, channel ratios. If this alone recovers the participants,
          illumination is doing the work. If it does not, while the recogniser
          embedding does, illumination is ruled out as the sole carrier.

  norm    An ablation. Re-embed after removing per-crop brightness and contrast
          (per-channel z-score) or after local histogram equalisation (CLAHE).
          Linkage that survives is not carried by the illumination channel.
          (run via periocular_attack.py --normalise)

  block   A within-session illumination change. Gallery frames come from the
          synthetic-stimulus blocks (calibration, fixation, pro/anti-saccade,
          pursuit, OKN -- dark screen, high-contrast dots and gratings), probe
          frames from the natural-image blocks (free viewing, visual search).
          The screen is the dominant light source at 80-120 cm in a dim room, so
          these two groups differ substantially in face illumination while
          remaining the same session. (run via periocular_attack.py --split)

None of these replaces a second recording, which remains the decisive experiment.
What they do is convert "this might be an illumination artefact" from an open
objection into a tested and rejected hypothesis.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def photometric_features(img):
    """Summary statistics an illumination-only attacker could use. No structure."""
    f = []
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        f += [ch.mean(), ch.std(),
              np.percentile(ch, 5), np.percentile(ch, 50), np.percentile(ch, 95)]
    b, g, r = [img[:, :, i].astype(np.float32).mean() + 1e-6 for i in range(3)]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f += [r / g, b / g, r / b,                       # white balance
          grey.mean(), grey.std(),
          float(np.percentile(grey, 99) - np.percentile(grey, 1))]  # dynamic range
    return np.asarray(f, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA / "ProcessedData"))
    ap.add_argument("--folder", default="appleLeftEye")
    ap.add_argument("--n-per-subject", type=int, default=120)
    ap.add_argument("--tag", default="illum")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    recs = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    rng = np.random.default_rng(0)
    X, y = [], []
    for rec in recs:
        d = root / rec / args.folder
        names = sorted(p.name for p in d.glob("*.jpg"))
        if len(names) < args.n_per_subject:
            continue
        for n in rng.choice(names, size=args.n_per_subject, replace=False):
            img = cv2.imread(str(d / n))
            if img is None:
                continue
            X.append(photometric_features(img))
            y.append(rec)
        print(f"  {rec}: {sum(1 for v in y if v == rec)}", flush=True)

    X = np.asarray(X); y = np.asarray(y)
    subs = sorted(set(y.tolist()))
    yt = np.array([subs.index(v) for v in y])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"[{args.tag}] {X.shape[0]} crops, {X.shape[1]} photometric features, "
          f"{len(subs)} participants")

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    def purity(a, b):
        return float(sum(np.bincount(a[b == c]).max() for c in np.unique(b)) / len(a))

    km = KMeans(n_clusters=len(subs), n_init=10, random_state=0).fit(X)
    best = (-2.0, None, None)
    for k in range(2, min(30, len(X) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="cosine")
        if s > best[0]:
            best = (s, k, lab)

    res = {"tag": args.tag, "root": str(root), "folder": args.folder,
           "n_crops": int(X.shape[0]), "n_features": int(X.shape[1]),
           "n_participants": len(subs),
           "ari_known_k": float(adjusted_rand_score(yt, km.labels_)),
           "purity_known_k": purity(yt, km.labels_),
           "k_selected": best[1],
           "ari_auto_k": float(adjusted_rand_score(yt, best[2])) if best[2] is not None else None}
    print(json.dumps(res, indent=2))
    if args.json_out:
        o = Path(args.json_out); o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(o, "w"), indent=2)
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
