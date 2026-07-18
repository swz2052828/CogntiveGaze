"""SVR calibration optimizer sweep: PSO vs JAYA vs MVO on every backbone.

Per backbone x fold:
  1. cache 128-d calibration embeddings once (train + val subjects, task +
     calib-support-K72 frames),
  2. tune (C, gamma, epsilon) with each swarm optimizer on the LEAK-FREE
     deploy-faithful fitness (train subjects only: support = their pre-task
     calib frames, query = their task frames; Zhu et al. bounds),
  3. evaluate each tuned triple + the fixed-HP baseline (C=1, gamma=scale,
     eps=0.1) on the HELD-OUT val subjects, svr_embed protocol @K9/K72.

One CSV row per (fold, optimizer|fixed). Backbones discovered via the
flip_calib_eval clean-run map (33 with the forward_features contract).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.dataset import MultiStreamGazeDataset                             # noqa: E402
from vit_gaze.metacompare import _features_and_preds, _svr_embed_predict, \
    _load_base_checkpoint                                                       # noqa: E402
from vit_gaze.splits import recording_kfolds                                    # noqa: E402
from vit_gaze.svr_search import optimize, DEFAULT_LB, DEFAULT_UB                # noqa: E402
from scripts.flip_calib_eval import FLIP_ROOT                                   # noqa: E402

DATA = "/springbrook/share/eng/esrpxk/datasets/ProcessedData"
SUPP = "/springbrook/share/eng/esrpxk/datasets"
RUNS = Path("/springbrook/share/eng/esrpxk/runs")


def ds(root):
    return MultiStreamGazeDataset(data_path=root, mean_path="meanno7_clean",
                                  eye_path=root, metadata_path=None,
                                  image_size=224, eye_size=224,
                                  grid_size=25, use_grid=True)


def err(p, g):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(g), axis=1).mean())


def group(dset, idxs):
    g = {}
    for row, i in enumerate(idxs):
        g.setdefault(int(dset.samples[i][-2]), []).append(row)
    return g


def make_fitness(sup_by_rec, qry_by_rec, seed, query_cap=400):
    """Deploy-faithful: per train subject fit SVR on its calib embeddings,
    score on (capped) task embeddings. sup/qry: rec -> (X emb, y gaze)."""
    from sklearn.svm import SVR
    rng = np.random.default_rng(seed)
    capped = {}
    for r, (Xq, yq) in qry_by_rec.items():
        if len(Xq) > query_cap:
            sel = rng.choice(len(Xq), size=query_cap, replace=False)
            capped[r] = (Xq[sel], yq[sel])
        else:
            capped[r] = (Xq, yq)

    def fitness(params):
        C, gamma, eps = float(params[0]), float(params[1]), float(params[2])
        es = []
        for r, (Xs, ys) in sup_by_rec.items():
            Xq, yq = capped[r]
            try:
                sx = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps).fit(Xs, ys[:, 0])
                sy = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps).fit(Xs, ys[:, 1])
                pred = np.stack([sx.predict(Xq), sy.predict(Xq)], 1)
                es.append(err(pred, yq))
            except Exception:
                return 1e6
        return float(np.mean(es))
    return fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(FLIP_ROOT))
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--csv-out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    bb, fold = a.backbone, a.fold_index

    ck = RUNS / FLIP_ROOT[bb][0] / f"base/seed42/fold{fold}_best_{bb}_gaze_segmenter.pth"
    model, mean, std = _load_base_checkpoint(str(ck), device)
    task = ds(DATA)
    calib = ds(f"{SUPP}/calib_support_K72")
    splits = recording_kfolds(task.unique_recordings(), folds=5, seed=a.seed)
    sp = [s for s in splits if s["fold"] == fold][0]

    # ---- cache embeddings (train + val, task + calib) ----
    emb = {}
    for tag, dset, recs in (("t_tr", task, sp["train_recordings"]),
                            ("t_va", task, sp["val_recordings"]),
                            ("c_tr", calib, sp["train_recordings"]),
                            ("c_va", calib, sp["val_recordings"])):
        idx = dset.indices_for_recordings(recs)
        f, g, _, e = _features_and_preds(model, dset, idx, mean, std, device,
                                         a.batch_size, a.num_workers)
        emb[tag] = (e.numpy(), g.numpy(), group(dset, idx))

    # ---- leak-free fitness on TRAIN subjects ----
    sup_by, qry_by = {}, {}
    ec, gc, grc = emb["c_tr"]
    et, gt_, grt = emb["t_tr"]
    for r, cs in grc.items():
        if r in grt:
            sup_by[r] = (ec[cs], gc[cs])
            qry_by[r] = (et[grt[r]], gt_[grt[r]])

    # ---- evaluate a triple on VAL subjects (deploy-faithful) ----
    ecv, gcv, grcv = emb["c_va"]
    etv, gtv, grtv = emb["t_va"]

    def val_err(C, gamma, eps, K):
        es = []
        for r, tq in grtv.items():
            cs = grcv.get(r)
            if not cs:
                continue
            sup_src = cs if K == 72 else cs[::8][:9]
            n = len(sup_src)
            feats = torch.from_numpy(np.concatenate([ecv[sup_src], etv[tq]]))
            gz = torch.from_numpy(np.concatenate([gcv[sup_src], gtv[tq]]))
            pred = _svr_embed_predict(feats, gz, list(range(n)),
                                      list(range(n, n + len(tq))),
                                      C=C, gamma=gamma, epsilon=eps)
            es.append(err(pred, gtv[tq]))
        return float(np.mean(es))

    rows = []
    rows.append(dict(fold=fold, backbone=bb, method="fixed",
                     C=1.0, gamma=-1.0, eps=0.1, fit_err=-1.0, tune_s=0.0,
                     val_K9=val_err(1.0, "scale", 0.1, 9),
                     val_K72=val_err(1.0, "scale", 0.1, 72)))
    for opt in ("pso", "jaya", "mvo"):
        t0 = time.time()
        fitness = make_fitness(sup_by, qry_by, seed=a.seed + fold)
        bx, bf, _ = optimize(opt, fitness, DEFAULT_LB, DEFAULT_UB,
                             pop=a.pop, iters=a.iters, seed=a.seed + fold)
        C, gamma, eps = float(bx[0]), float(bx[1]), float(bx[2])
        rows.append(dict(fold=fold, backbone=bb, method=opt,
                         C=C, gamma=gamma, eps=eps, fit_err=float(bf),
                         tune_s=round(time.time() - t0, 1),
                         val_K9=val_err(C, gamma, eps, 9),
                         val_K72=val_err(C, gamma, eps, 72)))
        print(rows[-1], flush=True)

    import csv
    fp = Path(a.csv_out)
    fp.parent.mkdir(parents=True, exist_ok=True)
    hdr = not fp.exists()
    with open(fp, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if hdr:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", fp, flush=True)


if __name__ == "__main__":
    main()
