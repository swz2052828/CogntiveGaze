"""Identity-leakage attack suite for anonymised gaze imagery (Threat model A).

Attacker model
--------------
The attacker holds enrolment photographs of the subjects (gallery, always the
ORIGINAL frames) and observes the released/anonymised frames (probe). The
attacker knows where the face is: face alignment landmarks are taken from the
ORIGINAL frame and applied to the anonymised frame, so a blur that merely
defeats face *detection* gets no credit. This is the conservative (strong
attacker) choice and is what makes the privacy axis comparable across operators.

Metrics: Rank-1 identification, verification ROC-AUC, TAR@FAR in {1e-1,1e-2,1e-3},
and (with --attributes) gender/age inference agreement against the original.

Gallery/probe frames are drawn from DISJOINT temporal halves of each recording
so that neither trivially matches the other.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def list_recordings(root: Path):
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name.isdigit())


def pick_frames(gal_dir: Path, prb_dir: Path, n_gallery: int, n_probe: int, seed: int = 0):
    """Disjoint temporal halves: gallery from the first half of the ORIGINAL
    recording, probe from the second half of what the PROBE root actually holds.

    Probe roots may be subsampled (generative operators are too costly to run on
    every frame), so probe names must be listed from the probe root -- otherwise
    most probes silently fail to load and the condition is scored on whatever
    frames happened to coincide."""
    gal = sorted(p.name for p in gal_dir.glob("*.jpg"))
    prb = sorted(p.name for p in prb_dir.glob("*.jpg"))
    if not gal or not prb:
        return [], []
    cut = gal[len(gal) // 2]                       # temporal midpoint by frame name
    gal_first = [n for n in gal if n < cut]
    prb_second = [n for n in prb if n >= cut]
    if not gal_first or not prb_second:
        return [], []
    rng = np.random.default_rng(seed)
    g = rng.choice(gal_first, size=min(n_gallery, len(gal_first)), replace=False)
    p = rng.choice(prb_second, size=min(n_probe, len(prb_second)), replace=False)
    return sorted(g.tolist()), sorted(p.tolist())


class ArcFace:
    """insightface w600k_r50 embeddings + genderage, aligned by supplied kps."""

    def __init__(self, ctx_id=-1, det_size=320, intra_threads=4):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        # The default pool spawns one thread per core (168 here) and thrashes.
        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 \
            else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers,
                                allowed_modules=["detection", "recognition", "genderage"],
                                session_options=so)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.rec = self.app.models["recognition"]
        self.ga = self.app.models.get("genderage")

    def detect(self, bgr):
        """Return (kps, bbox) of the largest detected face, or None."""
        faces = self.app.get(bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return faces[-1].kps, faces[-1].bbox

    def align(self, bgr, kps):
        from insightface.utils import face_align
        return face_align.norm_crop(bgr, landmark=kps, image_size=112)

    def embed_aligned(self, aimg):
        emb = self.rec.get_feat(aimg).flatten()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def embed_with_kps(self, bgr, kps):
        return self.embed_aligned(self.align(bgr, kps))

    def gender_age(self, bgr, kps, bbox):
        """Attribute inference on the RELEASED image, using geometry from the original."""
        if self.ga is None:
            return None, None
        from insightface.app.common import Face
        face = Face(bbox=np.asarray(bbox, dtype=np.float32),
                    kps=np.asarray(kps, dtype=np.float32), det_score=1.0)
        g, a = self.ga.get(bgr, face)
        return int(g), float(a)


class FaceNet:
    """InceptionResnetV1 (vggface2). Shares ArcFace's alignment so the two
    attackers differ only in the embedding, not in the crop they see."""

    def __init__(self, device="cuda"):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed_aligned(self, aimg112):
        import numpy as _np
        torch = self.torch
        img = cv2.resize(aimg112, (160, 160))[:, :, ::-1].copy()   # BGR->RGB
        x = torch.from_numpy(img).float().permute(2, 0, 1)[None]
        x = (x - 127.5) / 128.0
        with torch.no_grad():
            e = self.net(x.to(self.device)).cpu().numpy().ravel()
        return e / (_np.linalg.norm(e) + 1e-9)


def tar_at_far(scores, labels, far_targets=(1e-1, 1e-2, 1e-3)):
    """labels: 1 = genuine (same subject), 0 = impostor."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    gen, imp = scores[labels == 1], scores[labels == 0]
    out = {}
    for far in far_targets:
        if len(imp) == 0:
            out[f"TAR@FAR={far:g}"] = float("nan")
            continue
        thr = np.quantile(imp, 1.0 - far)
        out[f"TAR@FAR={far:g}"] = float((gen >= thr).mean()) if len(gen) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-root", default=str(DATA / "ProcessedData"),
                    help="Root holding ORIGINAL crops (attacker's enrolment photos)")
    ap.add_argument("--probe-root", required=True,
                    help="Root holding the RELEASED (possibly anonymised) crops")
    ap.add_argument("--folder", default="appleFace",
                    help="appleFace, or appleLeftEye for the periocular attack")
    ap.add_argument("--n-gallery", type=int, default=20)
    ap.add_argument("--n-probe", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=-1, help="gpu id, -1 = cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lowfreq", action="store_true",
                    help="Also build a low-frequency descriptor (8x8 thumbnail of "
                         "the aligned crop) and report scores for ArcFace alone, "
                         "the descriptor alone, and their fusion over a sweep of "
                         "weights. Tests whether the ArcFace-only attacker is "
                         "under-powered on the colour/luminance field that every "
                         "blur arm leaves behind.")
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip face detection; resize the crop straight to 112x112. "
                         "Required for periocular crops, where a FACE detector finds "
                         "nothing -- detector failure must not be scored as privacy.")
    ap.add_argument("--attributes", action="store_true")
    ap.add_argument("--truth-csv",
                    default="/springbrook/share/eng/esrpxk/CogntiveGaze/results/subject_attributes.csv",
                    help="ground-truth gender/age per recording; enables balanced "
                         "accuracy against truth rather than only self-consistency")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--recogniser", default="arcface", choices=["arcface", "facenet"],
                    help="arcface (default; every published number here) or facenet "
                         "(Inception-ResNet-v1/VGGFace2) as an independent second "
                         "opinion. facenet is embeddings-only, so it requires "
                         "--no-detect and is incompatible with --attributes.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gal_root, prb_root = Path(args.gallery_root), Path(args.probe_root)
    recs = [r for r in list_recordings(gal_root) if (prb_root / r / args.folder).is_dir()]
    print(f"[{args.tag}] {len(recs)} recordings | folder={args.folder}", flush=True)

    if getattr(args, "recogniser", "arcface") == "arcface":
        fr = ArcFace(ctx_id=args.ctx, intra_threads=args.threads)
    else:
        # FaceNet has no detector; it is only valid on the --no-detect path, which
        # is the eye-crop attack. Refuse rather than silently fall through to an
        # AttributeError halfway through a gallery.
        if not args.no_detect:
            raise SystemExit("--recogniser facenet requires --no-detect "
                             "(it supplies embeddings only, no face detector)")
        if getattr(args, "attributes", False):
            raise SystemExit("--recogniser facenet has no gender/age head; "
                             "drop --attributes")
        # Same directory, but this file is run both as a script (sys.path =
        # scripts/anon) and imported as anon.id_attack (sys.path = scripts).
        # Import the class directly rather than via recognisers.build(), whose
        # arcface branch imports this module back.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from recognisers import FaceNetRecogniser
        fr = FaceNetRecogniser(ctx_id=args.ctx)
    G, Gy, P, Py = [], [], [], []
    Glf, Plf = [], []

    def lowfreq(bgr):
        t = cv2.resize(bgr, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
        return t / (np.linalg.norm(t) + 1e-9)
    attr_ref, attr_prb, attr_subject = [], [], []
    n_kps_fail = 0

    for rec in recs:
        gdir, pdir = gal_root / rec / args.folder, prb_root / rec / args.folder
        gframes, pframes = pick_frames(gdir, pdir, args.n_gallery, args.n_probe)
        for name in gframes:
            img = cv2.imread(str(gdir / name))
            if img is None:
                continue
            if args.no_detect:
                G.append(fr.embed_aligned(cv2.resize(img, (112, 112))))
                Gy.append(rec)
                continue
            det = fr.detect(img)
            if det is None:
                n_kps_fail += 1
                continue
            kps, _ = det
            G.append(fr.embed_with_kps(img, kps))
            Gy.append(rec)
            if args.lowfreq:
                Glf.append(lowfreq(fr.align(img, kps)))
        for name in pframes:
            orig = cv2.imread(str(gdir / name))       # alignment source: ORIGINAL
            anon = cv2.imread(str(pdir / name))       # released image
            if orig is None or anon is None:
                continue
            if args.no_detect:
                P.append(fr.embed_aligned(cv2.resize(anon, (112, 112))))
                Py.append(rec)
                continue
            det = fr.detect(orig)
            if det is None:
                n_kps_fail += 1
                continue
            kps, bbox = det
            P.append(fr.embed_with_kps(anon, kps))
            Py.append(rec)
            if args.lowfreq:
                Plf.append(lowfreq(fr.align(anon, kps)))
            if args.attributes:
                attr_ref.append(fr.gender_age(orig, kps, bbox))
                attr_prb.append(fr.gender_age(anon, kps, bbox))
                attr_subject.append(rec)
        print(f"  {rec}: gallery {len(gframes)} probe {len(pframes)}", flush=True)

    G, P = np.asarray(G), np.asarray(P)
    Gy, Py = np.asarray(Gy), np.asarray(Py)
    print(f"embeddings: gallery {G.shape} probe {P.shape}  kps_fail {n_kps_fail}")
    if not len(G) or not len(P):
        # Hard failure, not a quiet return: an empty arm otherwise writes a JSON
        # that is indistinguishable from a real measurement and would read as a
        # spectacular privacy win in the aggregate table. The usual cause is
        # attacking a probe root whose generation job is still writing.
        sys.stderr.write(
            f"FATAL [{args.tag}]: empty gallery ({len(G)}) or probe ({len(P)}). "
            f"Probe root {prb_root} is missing frames -- is its generation job "
            f"still running? No JSON written.\n")
        sys.exit(2)

    # subject-level gallery templates (mean embedding, renormalised)
    subs = sorted(set(Gy))
    T = np.stack([G[Gy == s].mean(0) for s in subs])
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

    S = P @ T.T                                    # cosine, [n_probe, n_subj]
    pred = np.asarray(subs)[S.argmax(1)]
    rank1 = float((pred == Py).mean())
    chance = 1.0 / len(subs)

    labels = (np.asarray(subs)[None, :] == Py[:, None]).astype(int).ravel()
    scores = S.ravel()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(labels, scores))

    # d-prime: the pre-registered primary privacy metric alongside TAR@FAR=1e-3.
    # Rank-1/AUC/TAR@FAR=1e-2 all saturate at 1.0 across several operators (§2.1),
    # so d' is what still resolves differences at the top of the range.
    gen, imp = scores[labels == 1], scores[labels == 0]
    dprime = float((gen.mean() - imp.mean()) /
                   np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))

    res = {"tag": args.tag, "folder": args.folder, "n_subjects": len(subs),
           "n_probe": int(len(P)), "rank1": rank1, "chance_rank1": chance,
           "dprime": dprime, "roc_auc": auc, "kps_fail": n_kps_fail}
    res.update(tar_at_far(scores, labels))

    if args.attributes and attr_ref:
        gr = np.array([a[0] for a in attr_ref]); gp = np.array([a[0] for a in attr_prb])
        ar = np.array([a[1] for a in attr_ref]); apr = np.array([a[1] for a in attr_prb])
        attr_rec = np.asarray(attr_subject)

        # (a) self-consistency, kept -- but scored against the agreement two
        # INDEPENDENT predictors with these marginals would reach, p*q+(1-p)*(1-q),
        # not against the cohort's majority-class rate. This is an agreement
        # statistic between two predictions, not accuracy against truth, so the
        # majority-class baseline does not apply to it.
        p_ref = float((gr == 1).mean()); p_prb = float((gp == 1).mean())
        res["gender_agreement"] = float((gr == gp).mean())
        res["gender_marginal_original"] = p_ref
        res["gender_marginal_anonymised"] = p_prb
        res["gender_agreement_chance"] = float(p_ref * p_prb + (1 - p_ref) * (1 - p_prb))
        res["gender_marginal_shift"] = p_prb - p_ref

        # (b) balanced accuracy against ground truth, now that truth exists
        truth = {}
        try:
            import csv as _csv
            with open(args.truth_csv) as fh:
                for row in _csv.DictReader(l for l in fh if not l.startswith("#")):
                    if row.get("gender"):
                        truth[row["rec"]] = row["gender"]
        except Exception as exc:                                    # noqa: BLE001
            res["truth_csv_error"] = str(exc)
        if truth:
            # insightface codes gender as 0/1; fix the convention empirically on the
            # ORIGINAL predictions rather than assuming it, then apply it to both.
            y = np.array([1 if truth.get(r) == "Man" else 0 for r in attr_rec])
            keep = np.array([r in truth for r in attr_rec])
            if keep.any():
                acc_direct = float((gr[keep] == y[keep]).mean())
                flip = acc_direct < 0.5
                res["gender_code_flipped"] = bool(flip)
                for tag, pred in (("original", gr), ("anonymised", gp)):
                    pv = (1 - pred) if flip else pred
                    bal = []
                    for cls in (0, 1):
                        m = keep & (y == cls)
                        if m.any():
                            bal.append(float((pv[m] == cls).mean()))
                    res[f"gender_balanced_acc_{tag}"] = float(np.mean(bal)) if bal else None
                    res[f"gender_raw_acc_{tag}"] = float((pv[keep] == y[keep]).mean())
                res["gender_majority_baseline"] = float(max((y[keep] == c).mean() for c in (0, 1)))

        # (c) age drift between two predictions -- retained only as a diagnostic.
        # It is NOT an accuracy: the cohort's bands are 1/8/5/2/1 subjects, which
        # cannot support a per-band claim, and this quantity never touches truth.
        res["age_pred_drift_years"] = float(np.abs(ar - apr).mean())
        res["age_mae_years"] = float(np.abs(ar - apr).mean())   # legacy key

    Slf = None
    if args.lowfreq and len(Glf) and len(Plf):
        Glf, Plf = np.asarray(Glf), np.asarray(Plf)
        Tlf = np.stack([Glf[Gy == s_].mean(0) for s_ in subs])
        Tlf /= np.linalg.norm(Tlf, axis=1, keepdims=True) + 1e-9
        Slf = Plf @ Tlf.T

        # z-score each score matrix so the fusion weight is interpretable
        Za = (S - S.mean()) / (S.std() + 1e-12)
        Zl = (Slf - Slf.mean()) / (Slf.std() + 1e-12)
        WS = (0.0, 0.25, 0.5, 0.75, 1.0)
        subs_arr = np.asarray(subs)

        def _metrics(M, mask=None):
            m = np.ones(len(M), dtype=bool) if mask is None else mask
            lab = (subs_arr[None, :] == Py[m][:, None]).astype(int).ravel()
            sc = M[m].ravel()
            g, i = sc[lab == 1], sc[lab == 0]
            dp = float((g.mean() - i.mean()) / np.sqrt(0.5 * (g.var() + i.var()) + 1e-12))
            r1 = float((subs_arr[M[m].argmax(1)] == Py[m]).mean())
            return dp, r1, tar_at_far(sc, lab)["TAR@FAR=0.001"]

        fixed = {}
        for w in WS:
            dp, r1, t3 = _metrics((1 - w) * Za + w * Zl)
            fixed[f"w={w}"] = {"dprime": dp, "rank1": r1, "TAR@FAR=0.001": t3}

        # Leave-one-subject-out weight selection: the weight applied to a probe is
        # chosen WITHOUT that probe's subject, so nothing is selected on the data it
        # is scored on. Reporting max-over-w would be selection on test, which no
        # appeal to "the attacker may choose w" repairs -- a real attacker has no
        # answer key either.
        M_loo = np.empty_like(Za)
        chosen = {}
        for s_held in subs:
            held = (Py == s_held)
            rest = ~held
            if not held.any() or not rest.any():
                M_loo[held] = Za[held]
                continue
            best_w, best_d = 0.0, -np.inf
            for w in WS:
                dp, _, _ = _metrics((1 - w) * Za + w * Zl, mask=rest)
                if dp > best_d:
                    best_d, best_w = dp, w
            chosen[s_held] = best_w
            M_loo[held] = ((1 - best_w) * Za + best_w * Zl)[held]
        dp, r1, t3 = _metrics(M_loo)

        res["lowfreq_fixed_w"] = fixed
        res["lowfreq_loo"] = {"dprime": dp, "rank1": r1, "TAR@FAR=0.001": t3}
        res["lowfreq_loo_weights"] = chosen
        res["lowfreq_loo_gain_dprime"] = dp - fixed["w=0.0"]["dprime"]
        res["lowfreq_loo_gain_TAR"] = t3 - fixed["w=0.0"]["TAR@FAR=0.001"]

    print(json.dumps(res, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
        # raw similarities, so a new metric costs a rescore rather than a GPU rerun
        _extra = {} if Slf is None else {"S_lowfreq": Slf}
        np.savez_compressed(out.with_suffix(".scores.npz"), **_extra,
                            S=S.astype(np.float32), probe_subject=Py,
                            gallery_subject=np.asarray(subs))


if __name__ == "__main__":
    main()
