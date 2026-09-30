"""Session-nuisance floor: how much of the measured 'identity' is not the face?

Every subject was recorded in ONE session, so illumination, seating, camera
distance, background and clothing are constant within a subject and differ
between subjects. A recogniser can ride those cues instead of the face, and a
low-pass filter does not touch them at all -- which would explain both the
periocular result and the blur result without any biometric content.

Control: run the identical attack on a patch of the frame that contains NO face.
If background-only Rank-1 is near chance, the periocular/blur numbers are about
faces. If it is high, they are substantially about the session, and every
identity claim in the paper must be re-scoped.

Also reports blur strength relative to inter-ocular distance, since absolute
pixel sigma is not comparable across subjects (face crops are 300-350 px).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def face_box(frame, tmpl):
    r = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return loc[0], loc[1], tmpl.shape[1], tmpl.shape[0], float(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-half", type=int, default=60)
    ap.add_argument("--patch", type=int, default=120)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anon.id_attack import ArcFace, tar_at_far

    proc, orig = DATA / "ProcessedData", DATA / "OriginalData"
    recs = sorted(d.name for d in proc.iterdir() if d.is_dir() and d.name.isdigit())
    fr = ArcFace(ctx_id=args.ctx)
    rng = np.random.default_rng(0)

    X1, y1, X2, y2, iods, geom = [], [], [], [], [], []
    for rec in recs:
        names = sorted(p.name for p in (proc / rec / "appleFace").glob("*.jpg"))
        half = len(names) // 2
        first = rng.choice(names[:half], size=args.n_per_half, replace=False)
        second = rng.choice(names[half:], size=args.n_per_half, replace=False)
        for grp, X, y in ((first, X1, y1), (second, X2, y2)):
            for n in grp:
                frame = cv2.imread(str(orig / rec / n))
                if frame is None:
                    continue
                # background patch: top-left corner, verified clear of the face box
                fb = None
                if sum(1 for g in geom if g[0] == rec) < 3:   # 3 samples PER recording
                    t = cv2.imread(str(proc / rec / "appleFace" / n))
                    lt = cv2.imread(str(proc / rec / "appleLeftEye" / n))
                    rt = cv2.imread(str(proc / rec / "appleRightEye" / n))
                    if t is not None and lt is not None and rt is not None:
                        fb = face_box(frame, t)
                        lb, rb = face_box(frame, lt), face_box(frame, rt)
                        iod = abs((lb[0] + lb[2] / 2) - (rb[0] + rb[2] / 2))
                        iods.append(iod)
                        geom.append((rec, fb[0], fb[1], fb[2], iod))
                p = args.patch
                patch = frame[0:p, 0:p]
                if fb is not None and fb[0] < p and fb[1] < p:
                    patch = frame[0:p, -p:]            # face intrudes; use other corner
                X.append(fr.embed_aligned(cv2.resize(patch, (112, 112))))
                y.append(rec)
        print(f"  {rec}: {len(first)}+{len(second)}", flush=True)

    X1, X2 = np.asarray(X1), np.asarray(X2)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    subs = sorted(set(y1))

    T = np.stack([X1[y1 == s].mean(0) for s in subs])
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    S = X2 @ T.T
    direct = float((np.asarray(subs)[S.argmax(1)] == y2).mean())
    labels = (np.asarray(subs)[None, :] == y2[:, None]).astype(int).ravel()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(labels, S.ravel()))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X1, y1)
    probe = float(clf.score(X2, y2))

    # d-prime: keeps resolving after Rank-1 and TAR saturate
    gen, imp = S.ravel()[labels == 1], S.ravel()[labels == 0]
    dprime = float((gen.mean() - imp.mean()) /
                   np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))

    iods = np.asarray(iods)
    res = {"tag": "background_nuisance_control", "n_subjects": len(subs),
           "chance_rank1": 1.0 / len(subs),
           "background_direct_rank1": direct, "background_direct_auc": auc,
           "background_probe_rank1": probe, "background_dprime": dprime,
           "iod_px_mean": float(iods.mean()) if len(iods) else None,
           "iod_px_min": float(iods.min()) if len(iods) else None,
           "iod_px_max": float(iods.max()) if len(iods) else None}
    res.update(tar_at_far(S.ravel(), labels))
    print(json.dumps(res, indent=2))
    if len(iods):
        print("\nblur sigma as a fraction of inter-ocular distance:")
        for s in (5, 15, 30, 60):
            print(f"  sigma={s:3d} px -> {s / iods.mean():.3f} IOD")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
