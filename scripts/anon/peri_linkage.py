"""T1 linkage on PERIOCULAR crops -- the control the face-crop result lacks.

The 120 px eye crop excludes hair, ears, neck, clothing and face outline, so a
high ARI here cannot be attributed to those channels. Clustering follows
linkage_attack.py exactly: KMeans at known k, silhouette-selected agglomerative
(cosine, average) at unknown k.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[v] = "2"
import json, sys
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

def purity(yt, yp):
    return float(sum(np.bincount(yt[yp == c]).max() for c in np.unique(yp)) / len(yt))

def score(X, ylab, tag, k_max=30):
    uy = sorted(set(ylab.tolist()))
    y = np.array([uy.index(v) for v in ylab])
    n_true = len(uy)
    km = KMeans(n_clusters=n_true, n_init=10, random_state=0).fit(X)
    ari = float(adjusted_rand_score(y, km.labels_))
    pur = purity(y, km.labels_)
    best = (-2.0, None, None)
    for k in range(2, min(k_max, len(X) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="cosine")
        if s > best[0]:
            best = (s, k, lab)
    return {"tag": tag, "n_frames": int(len(X)), "n_subjects": n_true,
            "ari_k18": ari, "purity_k18": pur,
            "k_selected": best[1],
            "ari_kauto": float(adjusted_rand_score(y, best[2])) if best[2] is not None else None,
            "silhouette": float(best[0])}

out = []
for path, side in [(sys.argv[1], "left"), (sys.argv[2], "right")]:
    z = np.load(path)
    for half in ("1", "2"):
        X, yl = z[f"X{half}"], z[f"y{half}"]
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        out.append(score(X, yl, f"{side}_half{half}"))
    Xp = np.vstack([z["X1"], z["X2"]]); Xp = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
    yp = np.concatenate([z["y1"], z["y2"]])
    out.append(score(Xp, yp, f"{side}_pooled"))

for r in out:
    print(f"{r['tag']:16s} n={r['n_frames']:5d}  ARI@k={r['n_subjects']}={r['ari_k18']:.3f}  "
          f"purity={r['purity_k18']:.3f}  k_auto={r['k_selected']:3d}  ARI_auto={r['ari_kauto']:.3f}")
json.dump(out, open(sys.argv[3], "w"), indent=2)
print("wrote", sys.argv[3])
