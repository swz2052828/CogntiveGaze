"""T4: an attacker who knows the anonymisation method and adapts to it.

The zero-effort attacks (T2) apply a pretrained recogniser to the anonymised
image as-is. T4 is the more honest threat: the attacker knows the operator, can
run it themselves, and therefore holds PAIRED (original, anonymised) imagery for
people who are not the target. They can learn what the operator does to an
embedding and undo as much of it as is learnable, then apply that to the target.

Protocol, matching the T2 premise (attacker holds original enrolment photos):

  * gallery  = ORIGINAL embeddings, first temporal half, ALL 18 subjects
  * probe    = ANONYMISED embeddings, second temporal half, mapped through a
               de-anonymisation map
  * the map is fitted LEAVE-ONE-SUBJECT-OUT: for probes of subject s the map
    never sees any frame of s, so the reported figure is what the attacker gets
    on a person they could not train on.

Maps, in increasing order of attacker effort:

  none        no adaptation. Reproduces T2 on this frame set, and is the
              baseline every other row is a delta against.
  shift       add the mean (original - anonymised) offset. The cheapest possible
              adaptation: one 512-vector, no fitting.
  procrustes  the best orthogonal map. Cannot rescale or mix dimensions, so it
              tests whether the operator acts like a rotation of the space.
  ridge       full linear map, alpha chosen on the training subjects only.

Metrics follow id_attack.py exactly -- cosine scores, Rank-1 over the 18-way
gallery, d' between genuine and impostor score distributions, TAR at fixed FAR --
so the numbers are comparable with every other row in the protocol.

  python scripts/anon/adaptive_attack.py --orig <none.emb.npz> --anon <op.emb.npz>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def halves(y):
    """First/second temporal half within each subject, as index arrays.

    Frames are stored in temporal order per subject, so a positional split is a
    temporal split. Splitting by subject rather than globally keeps the gallery
    balanced.
    """
    g, p = [], []
    for s in np.unique(y):
        idx = np.where(y == s)[0]
        h = len(idx) // 2
        g.append(idx[:h])
        p.append(idx[h:])
    return np.concatenate(g), np.concatenate(p)


def fit_map(kind, A, B, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Return f(X)->X' mapping anonymised space A onto original space B."""
    if kind == "none":
        return lambda X: X
    if kind == "shift":
        d = B.mean(0) - A.mean(0)
        return lambda X: X + d
    if kind == "procrustes":
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
        R = U @ Vt
        am, bm = A.mean(0), B.mean(0)
        return lambda X: (X - am) @ R + bm
    if kind == "ridge":
        # alpha is chosen inside the training subjects, on a held-out slice of
        # them, so the reported result never sees the test subject at any stage.
        n = len(A)
        cut = int(n * 0.8)
        perm = np.random.default_rng(0).permutation(n)
        tr, va = perm[:cut], perm[cut:]
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        best, best_a = None, None
        G = Ac[tr].T @ Ac[tr]
        H = Ac[tr].T @ Bc[tr]
        for a in alphas:
            W = np.linalg.solve(G + a * np.eye(G.shape[0]), H)
            err = np.linalg.norm(Ac[va] @ W - Bc[va])
            if best is None or err < best:
                best, best_a = err, a
        W = np.linalg.solve(Ac.T @ Ac + best_a * np.eye(Ac.shape[1]), Ac.T @ Bc)
        am, bm = A.mean(0), B.mean(0)
        f = lambda X: (X - am) @ W + bm
        f.alpha = best_a
        return f
    raise ValueError(kind)


def evaluate(gal, gal_y, prb, prb_y, subs):
    """Cosine scoring against per-subject gallery centroids, as id_attack.py."""
    C = unit(np.stack([gal[gal_y == s].mean(0) for s in subs]))
    S = unit(prb) @ C.T
    rank1 = float((np.asarray(subs)[S.argmax(1)] == prb_y).mean())
    labels = (np.asarray(subs)[None, :] == prb_y[:, None]).astype(int).ravel()
    scores = S.ravel()
    gen, imp = scores[labels == 1], scores[labels == 0]
    dp = float((gen.mean() - imp.mean())
               / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))
    out = {"rank1": rank1, "dprime": dp, "chance_rank1": 1.0 / len(subs)}
    # Threshold exactly as id_attack.tar_at_far: the (1 - FAR) quantile of the
    # impostor scores. An earlier version took the k-th highest impostor score,
    # which differs only marginally on ~18k impostor pairs but made the claim
    # "metrics follow id_attack.py exactly" untrue.
    for far in (0.1, 0.01, 0.001):
        thr = np.quantile(imp.astype(np.float64), 1.0 - far)
        out[f"TAR@FAR={far}"] = float((gen >= thr).mean())
    return out




def run_maps(Xo, Xa, y, gi, pi, subs, res):
    """Fit and evaluate every map, leave-one-subject-out and in-sample.

    Each map is run twice. LOO is the attack. IN-SAMPLE (fitted on all subjects,
    including the one being probed) is not a threat model -- it is a diagnostic,
    and the gap between the two is the point: if in-sample works and LOO does
    not, the map has memorised *these identities* rather than learned what the
    operator does, and the operator is not invertible in a subject-independent
    way. Without this row a below-chance LOO number is indistinguishable from a
    bug in the fitting.
    """
    for kind in ("none", "shift", "procrustes", "ridge"):
        for mode in ("loo", "insample"):
            alphas = []
            if mode == "insample":
                f = fit_map(kind, Xa, Xo)
                if hasattr(f, "alpha"):
                    alphas.append(f.alpha)
                mapped = f(Xa[pi])
            else:
                mapped = np.empty_like(Xa[pi])
                for s in subs:
                    tr = np.isin(y, [t for t in subs if t != s])
                    f = fit_map(kind, Xa[tr], Xo[tr])
                    if hasattr(f, "alpha"):
                        alphas.append(f.alpha)
                    sel = y[pi] == s
                    mapped[sel] = f(Xa[pi][sel])
            r = evaluate(Xo[gi], y[gi], mapped, y[pi], subs)
            if alphas:
                r["ridge_alpha_median"] = float(np.median(alphas))
            res["maps"][kind if mode == "loo" else f"{kind}_insample"] = r
            lbl = kind if mode == "loo" else f"{kind} (in-sample)"
            print(f"  {lbl:22} d'={r['dprime']:6.3f}  Rank-1={r['rank1']*100:5.1f}%  "
                  f"TAR@1e-3={r['TAR@FAR=0.001']*100:5.1f}%  "
                  f"TAR@1e-2={r['TAR@FAR=0.01']*100:5.1f}%")


def finish(res, args):
    b = res["maps"]["none"]
    for v in res["maps"].values():
        v["gain_dprime"] = v["dprime"] - b["dprime"]
        v["gain_TAR@FAR=0.001"] = v["TAR@FAR=0.001"] - b["TAR@FAR=0.001"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="embeddings of the ORIGINAL crops")
    ap.add_argument("--anon", required=True, help="embeddings of the ANONYMISED crops")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    O, A = np.load(args.orig), np.load(args.anon)

    if "X1" in O.files:
        # Pre-split export (periocular_attack --emb-out): the gallery half and
        # the probe half are already separated, so do not re-split.
        Xo = np.concatenate([O["X1"], O["X2"]]).astype(np.float64)
        Xa = np.concatenate([A["X1"], A["X2"]]).astype(np.float64)
        y = np.concatenate([O["y1"], O["y2"]])
        if not np.array_equal(y, np.concatenate([A["y1"], A["y2"]])):
            raise SystemExit("pre-split embedding files are not row-aligned.")
        n1 = len(O["X1"])
        gi, pi = np.arange(n1), np.arange(n1, len(y))
    else:
        if not np.array_equal(O["y"], A["y"]):
            raise SystemExit(
                "the two embedding files are not row-aligned (labels differ), so "
                "a paired map cannot be fitted. Regenerate both with the same "
                "frame set.")
        y = O["y"]
        Xo, Xa = O["X"].astype(np.float64), A["X"].astype(np.float64)
        gi, pi = halves(y)

    subs = sorted(np.unique(y).tolist())
    res = {"tag": args.tag or Path(args.anon).stem, "n_subjects": len(subs),
           "n_gallery": int(len(gi)), "n_probe": int(len(pi)), "maps": {}}
    run_maps(Xo, Xa, y, gi, pi, subs, res)
    finish(res, args)


if __name__ == "__main__":
    main()
