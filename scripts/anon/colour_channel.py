"""How much identity is in the mean face colour, independent of any recogniser?

If meanfill_skin scores at the blackbox level, that says the ArcFace embedding
does not exploit a uniform colour patch. It does NOT say the patch carries no
identity -- the fill colour demonstrably varies between subjects. This script
measures the channel's own capacity by running the identical protocol with the
3-D fill colour AS the embedding: temporal-half gallery/probe split, subject
centroids, nearest-centroid identification.

The two numbers together separate "the information is not there" from "the
recogniser does not use it", which are different claims about a released image.
"""
import json
from pathlib import Path

import cv2
import numpy as np

D = Path("/springbrook/share/eng/esrpxk/datasets/anon_extreme/meanfill_skin")


def main():
    recs = sorted(d.name for d in D.iterdir() if d.is_dir() and d.name.isdigit())
    G, Gy, P, Py = [], [], [], []
    per_subject = {}
    for rec in recs:
        names = sorted(p.name for p in (D / rec / "appleFace").glob("*.jpg"))
        if len(names) < 20:
            continue
        half = len(names) // 2
        cols = []
        for grp, X, y in ((names[:half], G, Gy), (names[half:], P, Py)):
            for n in grp[:: max(1, len(grp) // 60)][:60]:
                im = cv2.imread(str(D / rec / "appleFace" / n))
                if im is None:
                    continue
                # corner pixel: uniform fill, outside both eye boxes
                c = im[2, 2].astype(np.float64)
                X.append(c); y.append(rec); cols.append(c)
        per_subject[rec] = np.mean(cols, axis=0)

    G, P = np.asarray(G), np.asarray(P)
    Gy, Py = np.asarray(Gy), np.asarray(Py)
    subs = sorted(set(Gy))
    C = np.stack([G[Gy == s].mean(0) for s in subs])

    # nearest centroid in raw BGR
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    pred = np.asarray(subs)[d2.argmin(1)]
    rank1 = float((pred == Py).mean())

    # same, with the negative distance as a score, for AUC / d'
    S = -np.sqrt(d2)
    lab = (np.asarray(subs)[None, :] == Py[:, None]).astype(int).ravel()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(lab, S.ravel()))
    gen, imp = S.ravel()[lab == 1], S.ravel()[lab == 0]
    dprime = float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var())))

    M = np.stack(list(per_subject.values()))
    res = {"tag": "colour_channel_only", "n_subjects": len(subs),
           "chance_rank1": 1.0 / len(subs), "rank1": rank1,
           "roc_auc": auc, "dprime": dprime,
           "fill_mean_BGR": M.mean(0).tolist(),
           "fill_between_subject_sd_BGR": M.std(0).tolist()}
    print(json.dumps(res, indent=2))
    print("\nper-subject fill colour (BGR):")
    for k, v in sorted(per_subject.items()):
        print(f"  {k}  {v[0]:6.1f} {v[1]:6.1f} {v[2]:6.1f}")
    out = Path("/springbrook/share/eng/esrpxk/results/anon_stage3b/colour_channel.json")
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
