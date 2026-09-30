"""Periocular identity leakage: can the EYE CROPS alone identify the subject?

This is the pivot experiment. The deployed gaze model is eyes-only, so if the
eye crops leak identity then 'we only keep the eyes' is not a privacy argument
and the pixels required for utility are the pixels that leak.

A FACE detector finds nothing in a 120 px eye crop, and detector failure must
not be scored as privacy. So two attackers are run without any face detection:

  arcface_direct : pretrained ArcFace embedding of the resized eye crop
                   (zero-effort attacker, no training)
  probe          : logistic regression on those embeddings, trained on the
                   FIRST temporal half of each recording and tested on the
                   SECOND -- the informed attacker of threat model T2/T5.

Caveat carried into the paper: one session per subject, so a learned probe can
exploit session nuisance (lighting, seating, clothing). These numbers are an
UPPER bound on periocular leakage and are reported as such.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA / "ProcessedData"))
    ap.add_argument("--folder", default="appleLeftEye")
    ap.add_argument("--n-per-half", type=int, default=60)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="periocular")
    ap.add_argument("--recogniser", default="arcface", choices=["arcface", "facenet"],
                    help="arcface (default; every published number here) or facenet "
                         "(Inception-ResNet-v1/VGGFace2) as an independent second "
                         "opinion on whether the leakage is a property of the data.")
    ap.add_argument("--normalise", default="none", choices=["none", "zscore", "clahe"],
                    help="photometric ablation for the single-session confound. "
                         "zscore standardises each crop per channel, removing "
                         "brightness and contrast; clahe additionally equalises "
                         "the local histogram. Linkage that survives either is "
                         "not carried by the illumination channel.")
    ap.add_argument("--split", default="half", choices=["half", "taskblock"],
                    help="half (default): gallery = first temporal half, probe = "
                         "second. taskblock: gallery = synthetic-stimulus blocks "
                         "(dark screen), probe = natural-image blocks. The screen "
                         "is the dominant light source, so the two groups differ "
                         "in face illumination within the same session.")
    ap.add_argument("--background", type=int, default=0, metavar="PX",
                    help="periocular-specific floor: instead of the eye crop, embed a "
                         "PX-square face-free patch from the top-left of the ORIGINAL "
                         "frame of the same recording, keeping frame selection and "
                         "every downstream step identical. The face-crop pipeline's "
                         "floor (t1bg_p120) is not strictly comparable to an eye-crop "
                         "number; this makes one that is. 120 matches the eye crops.")
    ap.add_argument("--emb-out", default=None,
                    help="also save the raw embeddings (X1/y1 first half, X2/y2 "
                         "second half) so a paired de-anonymisation map can be "
                         "fitted against another root. Frame selection is seeded, "
                         "and the eye folders of ProcessedData and ProcessedSwap "
                         "hold the same filenames, so two roots exported this way "
                         "are row-aligned.")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anon.id_attack import tar_at_far
    from anon.recognisers import build as build_recogniser

    root = Path(args.root)
    recs = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    fr = build_recogniser(args.recogniser, ctx_id=args.ctx)

    def normalise(img):
        if args.normalise == "none":
            return img
        if args.normalise == "zscore":
            f = img.astype(np.float32)
            for c in range(3):
                ch = f[:, :, c]
                f[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-6) * 40.0 + 128.0
            return np.clip(f, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Absolute frame windows per stimulus family. The task timeline is global and
    # each subject is offset by their own t_init (gaze_dynamics/config.py).
    if args.split == "taskblock":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from gaze_dynamics import config as _cfg
        tb, tl, _ = _cfg.build_task_timeline()
        # blocks 0-5 = calibration, fixation, pro/anti-saccade, pursuit, OKN
        # blocks 6+  = free viewing and visual search on natural images
        dark = [(tb[i], tb[i] + tl[i]) for i in range(6)]
        imgs = [(tb[i], tb[i] + tl[i]) for i in range(6, len(tb))]
        t_init = {f"{s:05d}": t for s, t in zip(_cfg.SUBJECT_IDS, _cfg.T_INITS)}

        def group_of(rec, name):
            """'dark', 'image' or None, from the absolute frame index in the filename."""
            t0 = t_init.get(rec)
            if t0 is None:
                return None
            rel = int(Path(name).stem) - t0
            if any(a <= rel < b for a, b in dark):
                return "dark"
            if any(a <= rel < b for a, b in imgs):
                return "image"
            return None

    X1, y1, X2, y2 = [], [], [], []
    rng = np.random.default_rng(0)
    for rec in recs:
        d = root / rec / args.folder
        names = sorted(p.name for p in d.glob("*.jpg"))
        if len(names) < 4 * args.n_per_half:
            continue
        if args.split == "taskblock":
            pool_a = [n for n in names if group_of(rec, n) == "dark"]
            pool_b = [n for n in names if group_of(rec, n) == "image"]
            if min(len(pool_a), len(pool_b)) < args.n_per_half:
                # Silently shrinking the sample would make the two splits
                # incomparable; refuse instead.
                print(f"  {rec}: SKIP taskblock (dark={len(pool_a)} "
                      f"image={len(pool_b)}, need {args.n_per_half})", flush=True)
                continue
        else:
            half = len(names) // 2
            pool_a, pool_b = names[:half], names[half:]
        first = rng.choice(pool_a, size=args.n_per_half, replace=False)
        second = rng.choice(pool_b, size=args.n_per_half, replace=False)
        for grp, X, y in ((first, X1, y1), (second, X2, y2)):
            for n in grp:
                if args.background:
                    # Same recording, same frames, same pipeline -- only the pixels
                    # change, from the eye region to a face-free corner of the room.
                    frame = cv2.imread(str(DATA / "OriginalData" / rec / n))
                    if frame is None:
                        continue
                    p = args.background
                    img = frame[0:p, 0:p]
                else:
                    img = cv2.imread(str(d / n))
                    if img is None:
                        continue
                X.append(fr.embed_aligned(normalise(cv2.resize(img, (112, 112)))))
                y.append(rec)
        print(f"  {rec}: {len(first)}+{len(second)}", flush=True)

    X1, X2 = np.asarray(X1), np.asarray(X2)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    subs = sorted(set(y1))
    print(f"[{args.tag}] {args.folder}: train {X1.shape} test {X2.shape} "
          f"{len(subs)} subjects")

    if args.emb_out:
        o = Path(args.emb_out); o.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o, X1=X1.astype(np.float32), y1=y1,
                            X2=X2.astype(np.float32), y2=y2)
        print(f"  wrote embeddings {o}")

    # --- attacker 1: zero-effort template matching (no training) ---
    T = np.stack([X1[y1 == s].mean(0) for s in subs])
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    S = X2 @ T.T
    rank1_direct = float((np.asarray(subs)[S.argmax(1)] == y2).mean())
    labels = (np.asarray(subs)[None, :] == y2[:, None]).astype(int).ravel()
    from sklearn.metrics import roc_auc_score
    auc_direct = float(roc_auc_score(labels, S.ravel()))

    # --- attacker 2: informed probe trained on anonymised-domain features ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X1, y1)
    rank1_probe = float(clf.score(X2, y2))
    proba = clf.predict_proba(X2)
    cls = list(clf.classes_)
    lab_p = (np.asarray(cls)[None, :] == y2[:, None]).astype(int).ravel()
    auc_probe = float(roc_auc_score(lab_p, proba.ravel()))

    gen, imp = S.ravel()[labels == 1], S.ravel()[labels == 0]
    dprime = float((gen.mean() - imp.mean()) /
                   np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))

    res = {"tag": args.tag, "folder": args.folder, "n_subjects": len(subs),
           "direct_dprime": dprime,
           "chance_rank1": 1.0 / len(subs),
           "arcface_direct_rank1": rank1_direct, "arcface_direct_auc": auc_direct,
           "informed_probe_rank1": rank1_probe, "informed_probe_auc": auc_probe}
    res.update({f"direct_{k}": v for k, v in tar_at_far(S.ravel(), labels).items()})
    res.update({f"probe_{k}": v for k, v in tar_at_far(proba.ravel(), lab_p).items()})
    print(json.dumps(res, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
