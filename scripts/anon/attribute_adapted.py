"""Fourth control: does the gender signal survive an ADAPTED attribute attacker?

The off-the-shelf head collapses to a constant class under blur -- P(male|anon)
goes to 0.000, balanced accuracy to exactly 0.500. That is the same shape as the
alignment result: an off-the-shelf model failing on off-distribution input, which
is not the same thing as the information being gone.

The cheap discriminating test is not fine-tuning. The head emits two logits and
the collapse is in the ARGMAX; if the logit DIFFERENCE still ranks men above
women, the information survived and only the decision threshold moved. So:

  off-the-shelf : argmax(pred[:2])                       -- what we reported
  adapted       : threshold on (pred[1]-pred[0]), fitted leave-one-subject-out

Leave-one-subject-out means no subject's threshold is fitted on its own frames.
Zero training, no external data, uses only what an attacker who knows the
operator already has -- so like the adapted-alignment regime it is a LOWER bound
on an adaptive attacker, not an upper one.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
TRUTH = Path("/springbrook/share/eng/esrpxk/CogntiveGaze/results/subject_attributes.csv")


def gender_logits(app, bgr, kps, bbox):
    """Raw two-logit gender output, bypassing the argmax that collapses."""
    from insightface.app.common import Face
    from insightface.utils import face_align
    ga = app.models["genderage"]
    face = Face(bbox=np.asarray(bbox, np.float32), kps=np.asarray(kps, np.float32),
                det_score=1.0)
    b = face.bbox
    w, h = b[2] - b[0], b[3] - b[1]
    center = (b[2] + b[0]) / 2, (b[3] + b[1]) / 2
    scale = ga.input_size[0] / (max(w, h) * 1.5)
    aimg, _ = face_align.transform(bgr, center, ga.input_size[0], scale, 0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / ga.input_std, tuple(aimg.shape[:2][::-1]),
                                 (ga.input_mean,) * 3, swapRB=True)
    pred = ga.session.run(ga.output_names, {ga.input_name: blob})[0][0]
    return float(pred[1] - pred[0]), int(np.argmax(pred[:2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", required=True)
    ap.add_argument("--gallery-root", default=str(DATA / "ProcessedData"))
    ap.add_argument("--n-per-subject", type=int, default=60)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--tag", default="adapted")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anon.id_attack import ArcFace

    truth = {}
    with open(TRUTH) as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            if row.get("gender"):
                truth[row["rec"]] = 1 if row["gender"] == "Man" else 0

    prb, gal = Path(args.probe_root), Path(args.gallery_root)
    recs = sorted(d.name for d in gal.iterdir() if d.is_dir() and d.name.isdigit())
    fr = ArcFace(ctx_id=args.ctx)

    score, argm, subj = [], [], []
    for rec in recs:
        if rec not in truth:
            continue
        gd, pd = gal / rec / "appleFace", prb / rec / "appleFace"
        names = sorted(p.name for p in pd.glob("*.jpg"))
        names = names[:: max(1, len(names) // args.n_per_subject)][: args.n_per_subject]
        for n in names:
            orig, anon = cv2.imread(str(gd / n)), cv2.imread(str(pd / n))
            if orig is None or anon is None:
                continue
            det = fr.detect(orig)          # alignment from the original, per SS3
            if det is None:
                continue
            s, a = gender_logits(fr.app, anon, det[0], det[1])
            score.append(s); argm.append(a); subj.append(rec)
        print(f"  {rec}: {sum(1 for x in subj if x == rec)}", flush=True)

    score = np.asarray(score); argm = np.asarray(argm); subj = np.asarray(subj)
    y = np.asarray([truth[r] for r in subj])
    if not len(y):
        print("NOTHING TO SCORE"); return

    def bal(pred, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        out = [float((pred[m & (y == c)] == c).mean())
               for c in (0, 1) if (m & (y == c)).any()]
        return float(np.mean(out)) if out else float("nan")

    # adapted: threshold fitted WITHOUT the held-out subject
    adapted = np.empty_like(argm)
    directions = {}
    for rec in sorted(set(subj)):
        held = subj == rec
        rest = ~held
        if not rest.any():
            adapted[held] = argm[held]; continue
        # BOTH directions must be searched. Blur inverts the logit ranking (AUC
        # falls below 0.5), so a one-sided `score > t` attacker is structurally
        # unable to use information that is present but sign-flipped -- it would
        # sit at ~0.5 and be reported as protection. An adaptive attacker would
        # obviously try the other direction, so not fitting it understates the
        # attacker in exactly the way this project keeps warning about.
        cand = np.quantile(score[rest], np.linspace(0.02, 0.98, 49))
        best, best_bal = None, -1.0
        for t in cand:
            for sign in (1, -1):
                pred = ((score * sign) > (t * sign)).astype(int)
                b = bal(pred, rest)
                if b > best_bal:
                    best_bal, best = b, (t, sign)
        t, sign = best
        directions[rec] = int(sign)
        adapted[held] = ((score[held] * sign) > (t * sign)).astype(int)

    res = {"tag": args.tag, "n_frames": int(len(y)), "n_subjects": len(set(subj)),
           "offtheshelf_balanced_acc": bal(argm),
           "offtheshelf_p_male": float(argm.mean()),
           "adapted_balanced_acc": bal(adapted),
           "adapted_p_male": float(adapted.mean()),
           "score_auc_vs_truth": None,
           "adapted_directions": directions,
           "adapted_direction_flipped_folds": None}
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) == 2:
        # threshold-free: does the logit difference still RANK the classes?
        res["score_auc_vs_truth"] = float(roc_auc_score(y, score))
    res["adapted_direction_flipped_folds"] = int(sum(1 for v in directions.values() if v < 0))
    if res["score_auc_vs_truth"] is not None:
        # threshold-free capability of a sign-aware attacker
        res["score_auc_sign_aware"] = float(max(res["score_auc_vs_truth"],
                                                1 - res["score_auc_vs_truth"]))
    res["recovery"] = res["adapted_balanced_acc"] - res["offtheshelf_balanced_acc"]
    print(json.dumps(res, indent=2))
    if args.json_out:
        out = Path(args.json_out); out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
