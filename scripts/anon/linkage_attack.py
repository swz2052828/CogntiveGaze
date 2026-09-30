"""T1 -- linkage within the release (threat model §3, primary).

The attacker holds *no reference at all*: only the published, anonymised corpus.
The question is whether the released frames can be partitioned back into
individuals. This is the attack that needs nothing external, and it is the one
that matters for a release-only deployment -- clustering the corpus enables
longitudinal tracking, and any side channel (e.g. a participant list) then
completes de-anonymisation.

Metrics
-------
* Adjusted Rand Index and cluster purity at **known k=18** and at **unknown k**
  (k chosen by silhouette over a range, so the attacker is not handed the answer).
* Same/different **d'** on the probe-vs-probe cosine similarities. This is
  reference-free and, unlike ARI, does not depend on a clustering algorithm --
  it is the quantity the operator actually has to move.

Alignment follows the conservative rule in §3: landmarks come from the ORIGINAL
frame, so an operator gets no credit for merely defeating face detection.
`--self-align` runs the strictly release-only variant (detect on the anonymised
image itself), which is the honest T1 premise; report both -- the gap between
them is exactly the protection that comes from breaking detection rather than
from removing identity.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.anon.id_attack import ArcFace, DATA, list_recordings


def sample_frames(prb_dir: Path, n: int, seed: int = 0):
    names = sorted(p.name for p in prb_dir.glob("*.jpg"))
    if not names:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(names, size=min(n, len(names)), replace=False).tolist())


def purity(labels_true, labels_pred):
    """Fraction of frames in the majority subject of their assigned cluster."""
    total = 0
    for c in np.unique(labels_pred):
        m = labels_pred == c
        if m.sum():
            total += np.bincount(labels_true[m]).max()
    return float(total / len(labels_true))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", required=True)
    ap.add_argument("--gallery-root", default=str(DATA / "ProcessedData"),
                    help="ORIGINALS -- used only as the alignment source (§3 rule)")
    ap.add_argument("--folder", default="appleFace")
    ap.add_argument("--n-per-subject", type=int, default=60)
    ap.add_argument("--ctx", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--adapted-align", action="store_true",
                    help="Release-only attacker who has SOLVED alignment. Detects on "
                         "the anonymised release; for the frames where detection "
                         "fails, substitutes the median landmark configuration of "
                         "the frames in that same recording where it succeeded. "
                         "Needs nothing but the release, exploits the fact that the "
                         "crop geometry is per-subject constant, and requires no "
                         "training -- so it is a lower bound on what an adaptive "
                         "attacker achieves, not an upper one. Sits between "
                         "--self-align (off-the-shelf) and the conservative regime "
                         "(landmarks handed over from the original).")
    ap.add_argument("--background", action="store_true",
                    help="NUISANCE CONTROL: cluster a face-free corner patch of the "
                         "frame instead of the released crop. Clustering is an easier "
                         "task than identification, and the T2 control already showed "
                         "a blank corner gives Rank-1 50.5%% / d' 0.46 -- so if this "
                         "reaches ARI 1.000 on its own, ARI is saturated by "
                         "single-session structure and cannot discriminate operators "
                         "at all in this dataset.")
    ap.add_argument("--patch", type=int, default=120)
    ap.add_argument("--allow-face-overlap", action="store_true",
                    help="proceed even if the background patch clips the face box")
    ap.add_argument("--self-align", action="store_true",
                    help="Detect on the released image (strict release-only attacker)")
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    prb_root, gal_root = Path(args.probe_root), Path(args.gallery_root)
    recs = [r for r in list_recordings(gal_root) if (prb_root / r / args.folder).is_dir()]

    # A background floor is only a floor if the patch contains NO face. A
    # top-left patch large enough to be informative can reach the face box:
    # measured, p=224 clips it on 11718/11718 frames of 00008 and 00009 and
    # 1125 of 00012, which inflated ARI from 0.089 (p=120) to 0.442 and would
    # have been read as "session nuisance alone half-recovers the partition".
    # The face boxes are already on disk in anon_geometry, so check rather than
    # assume.
    if args.background:
        bad = {}
        for rec in recs:
            gz = DATA / "anon_geometry" / f"{rec}.npz"
            if not gz.exists():
                continue
            fb = np.load(gz)["appleFace"]
            n = int(((fb[:, 0] < args.patch) & (fb[:, 1] < args.patch)).sum())
            if n:
                bad[rec] = (n, len(fb))
        if bad:
            msg = ", ".join(f"{r} {n}/{t}" for r, (n, t) in sorted(bad.items()))
            if not args.allow_face_overlap:
                raise SystemExit(
                    f"FATAL: patch={args.patch} clips the face box: {msg}. "
                    f"This is not a nuisance floor -- it contains face pixels. "
                    f"Use a smaller --patch, or --allow-face-overlap to record "
                    f"it as a contaminated upper bound.")
            print(f"WARNING: patch clips face box: {msg}", flush=True)
    print(f"[{args.tag}] {len(recs)} recordings | folder={args.folder} "
          f"| self_align={args.self_align}", flush=True)

    fr = ArcFace(ctx_id=args.ctx, intra_threads=args.threads)
    X, y = [], []
    n_det_fail, n_imputed = 0, 0
    self_align = args.self_align or args.adapted_align
    for si, rec in enumerate(recs):
        pdir, gdir = prb_root / rec / args.folder, gal_root / rec / args.folder
        names = sample_frames(pdir, args.n_per_subject)

        pending = []          # (name, anon) whose detection failed
        kps_ok = []           # landmark sets recovered from the release itself
        for name in names:
            if args.background:
                frame = cv2.imread(str(DATA / "OriginalData" / rec / name))
                if frame is None:
                    continue
                p = args.patch
                anon = frame[0:p, 0:p]                 # top-left corner, no face
                X.append(fr.embed_aligned(cv2.resize(anon, (112, 112))))
                y.append(si)
                continue
            anon = cv2.imread(str(pdir / name))
            if anon is None:
                continue
            src = anon if self_align else cv2.imread(str(gdir / name))
            if src is None:
                continue
            det = fr.detect(src)
            if det is None:
                # Under --self-align this is a real protection effect (the attacker
                # cannot even find a face), so it is counted, not silently dropped.
                n_det_fail += 1
                if args.adapted_align:
                    pending.append(anon)
                continue
            kps_ok.append(det[0])
            X.append(fr.embed_with_kps(anon, det[0]))
            y.append(si)

        # The adaptive step: the crop geometry is constant within a recording
        # (protocol SS2.2), so landmarks recovered from the frames that DID detect
        # transfer to the ones that did not. No training, no external data.
        if args.adapted_align and pending and kps_ok:
            med = np.median(np.stack(kps_ok), axis=0)
            for anon in pending:
                X.append(fr.embed_with_kps(anon, med))
                y.append(si)
                n_imputed += 1
        print(f"  {rec}: {sum(1 for v in y if v == si)} frames", flush=True)

    X, y = np.asarray(X), np.asarray(y)
    print(f"embeddings {X.shape}  det_fail {n_det_fail}  imputed {n_imputed}", flush=True)
    if len(np.unique(y)) < 2:
        # Total detection failure is a RESULT -- pixelation at 0.25*IOD defeats the
        # detector on every frame, where blur0.50 still leaves ~11% -- so it must be
        # recorded, not returned from silently with exit 0 and no JSON. That is the
        # same failure class as attacking a probe root whose generation is unfinished.
        res = {"tag": args.tag, "folder": args.folder, "n_subjects": len(recs),
               "n_frames": int(len(X)), "self_align": bool(args.self_align),
               "adapted_align": bool(args.adapted_align),
               "det_fail": n_det_fail, "n_alignment_imputed": n_imputed,
               "n_subjects_recovered": int(len(np.unique(y))) if len(y) else 0,
               "ari_k18": None, "purity_k18": None, "k_selected": None,
               "k18_undefined_reason": (
                   f"only {len(X)} embeddings from {len(recs)} subjects survived "
                   f"detection ({n_det_fail} failures); fewer than 2 subjects "
                   f"recoverable, so no partition exists to score")}
        if not (args.self_align or args.adapted_align):
            res["WARNING"] = ("conservative regime aligns from the originals; total "
                              "detection failure here is a bug, not protection")
        print(json.dumps(res, indent=2))
        if args.json_out:
            out = Path(args.json_out); out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out, "w"), indent=2)
            print(f"wrote {out}")
        return

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    n_true = len(recs)
    res = {"tag": args.tag, "folder": args.folder, "n_subjects": n_true,
           "n_frames": int(len(X)), "self_align": bool(args.self_align),
           "det_fail": n_det_fail, "n_alignment_imputed": n_imputed,
           "adapted_align": bool(args.adapted_align),
           "n_subjects_recovered": int(len(np.unique(y)))}

    # --- known k -----------------------------------------------------------
    # Too few surviving embeddings to form 18 clusters is a RESULT, not an error,
    # and it means different things in the two regimes. Under --self-align the
    # attacker genuinely cannot align the release, so this is the protection being
    # measured; it must be recorded as undefined with the surviving count, never
    # crashed on and never coerced to ARI 0 (which would read as "clustering failed
    # to separate subjects" rather than "there was nothing left to cluster").
    # Under the conservative regime landmarks come from the ORIGINAL, so detection
    # should not fail at all, and few survivors there would indicate a real bug.
    if len(X) < n_true:
        res["ari_k18"] = None
        res["purity_k18"] = None
        res["k18_undefined_reason"] = (
            f"only {len(X)} embeddings survived detection, fewer than the "
            f"{n_true} clusters required")
        if not args.self_align:
            res["WARNING"] = ("conservative regime should align from the originals; "
                              "detection failure here indicates a bug, not protection")
    else:
        km = KMeans(n_clusters=n_true, n_init=10, random_state=0).fit(X)
        res["ari_k18"] = float(adjusted_rand_score(y, km.labels_))
        res["purity_k18"] = purity(y, km.labels_)

    # --- unknown k: attacker picks k by silhouette --------------------------
    best = (-2.0, None, None)
    for k in range(2, max(2, min(args.k_max, len(X) - 1)) + 1):
        if k >= len(X):
            break
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="cosine")
        if s > best[0]:
            best = (s, k, lab)
    if best[1] is not None:
        res["k_selected"] = int(best[1])
        res["silhouette"] = float(best[0])
        res["ari_kauto"] = float(adjusted_rand_score(y, best[2]))
        res["purity_kauto"] = purity(y, best[2])

    # --- reference-free same/different d' -----------------------------------
    S = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    sim = S[iu]
    same = (y[iu[0]] == y[iu[1]])
    gen, imp = sim[same], sim[~same]
    res["dprime_same_diff"] = float((gen.mean() - imp.mean()) /
                                    np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))
    from sklearn.metrics import roc_auc_score
    res["pair_auc"] = float(roc_auc_score(same.astype(int), sim))

    print(json.dumps(res, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
        np.savez_compressed(out.with_suffix(".emb.npz"), X=X.astype(np.float32), y=y)


if __name__ == "__main__":
    main()
