"""Unified eval-side experiments on EXISTING clean checkpoints (one GPU pass per fold):

  #1 fc_ft hyperparameter tuning -- leak-free grid search on the TRAINING subjects'
     (calib-support -> task-frames) transfer, mirroring the SVR swarm-search protocol.
     Reports fc_ft_default (Zhu: lr 5e-5, 20 steps, wd 5e-4) vs fc_ft_tuned.
  #2 per-subject calibrator selection -- leave-one-calibration-point-out across the
     9 points picks the best method (fc_ft_tuned / meta / svr_embed) per subject
     ('auto'); 'oracle' = hindsight-best per subject (upper bound).
  #5 per-frame prediction dump (rec, frame, gt, preds) for temporal-smoothing analysis.

The task dataset uses the DIRTY manifest (meanno7) so timelines are complete (blink
frames included -- needed for smoothing); errors are reported BOTH on all frames and
on the clean-frame mask (the meanno7_clean set) for comparability with the clean
metacompare. Models/adapters are the clean-trained ones from the metacmp roots.

  python scripts/calib_eval_extras.py --run-root runs/meta_pipeline_clean_metacmp_convnext \
    --backbone convnext --fold-index 0 --ks 9 72 --csv-out ... --dump-dir ...
"""
import argparse
import copy
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.dataset import MultiStreamGazeDataset                     # noqa: E402
from vit_gaze.metacompare import (                                       # noqa: E402
    _features_and_preds, _fc_ft_predict, _meta_predict, _svr_embed_predict,
    _load_base_checkpoint, _load_meta_checkpoint)
from vit_gaze.splits import recording_kfolds                             # noqa: E402

DATA = "/springbrook/share/eng/esrpxk/datasets/ProcessedData"
SUPP = "/springbrook/share/eng/esrpxk/datasets"

FCFT_DEFAULT = dict(lr=5e-5, steps=20, weight_decay=5e-4)
FCFT_GRID = [dict(lr=lr, steps=st, weight_decay=wd)
             for lr in (2e-5, 5e-5, 1e-4, 3e-4, 1e-3)
             for st in (20, 60, 150)
             for wd in (0.0, 5e-4, 5e-3)]


def build_task_dataset(mean_path, use_grid=True):
    return MultiStreamGazeDataset(
        data_path=DATA, mean_path=mean_path, eye_path=DATA, metadata_path=None,
        image_size=224, eye_size=224, grid_size=25, use_grid=use_grid)


def build_calib_dataset(k, mean_path, use_grid=True):
    root = f"{SUPP}/calib_support_K{k}"
    return MultiStreamGazeDataset(
        data_path=root, mean_path=mean_path, eye_path=root, metadata_path=None,
        image_size=224, eye_size=224, grid_size=25, use_grid=use_grid)


def group_rows_by_rec(dataset, indices):
    out = {}
    for row, i in enumerate(indices):
        out.setdefault(int(dataset.samples[i][-2]), []).append(row)
    return out


def frames_of(dataset, indices):
    return np.array([int(dataset.samples[i][-1]) for i in indices], dtype=np.int64)


def clean_frame_set():
    """(rec, frame) pairs present in the blink-cleaned manifest."""
    import scipy.io as sio
    md = sio.loadmat(f"{DATA}/meanno7_clean/metadata.mat", squeeze_me=True)
    return set(zip(np.asarray(md["labelRecNum"]).astype(int).ravel(),
                   np.asarray(md["frameIndex"]).astype(int).ravel()))


def err(pred, gt, mask=None):
    e = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=1)
    if mask is not None:
        e = e[mask]
    return float(e.mean()) if len(e) else float("nan")


def fcft_tune(base_model, mean, std, feats_c, gz_c, feats_t, gz_t,
              rows_c, rows_t, device, query_cap=200, seed=42):
    """Leak-free grid: fit readout on each TRAIN rec's calib support, score on a
    capped sample of its task frames; mean over recs; return best HP dict."""
    rng = np.random.default_rng(seed)
    per_combo = []
    caps = {}
    for rec, tq in rows_t.items():
        caps[rec] = list(rng.choice(len(tq), size=min(query_cap, len(tq)), replace=False))
    for hp in FCFT_GRID:
        errs = []
        for rec, cs in rows_c.items():
            tq = rows_t.get(rec)
            if not tq:
                continue
            qsel = [tq[j] for j in caps[rec]]
            n = len(cs)
            feats = torch.cat([feats_c[cs], feats_t[qsel]])
            gz = torch.cat([gz_c[cs], gz_t[qsel]])
            p = _fc_ft_predict(base_model, mean, std, feats, gz,
                               list(range(n)), list(range(n, n + len(qsel))), device, **hp)
            errs.append(err(p, gz[n:].numpy()))
        per_combo.append((float(np.mean(errs)), hp))
    per_combo.sort(key=lambda x: x[0])
    return per_combo[0][1], per_combo[0][0], per_combo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[9, 72])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mean-path", default="meanno7")          # dirty = full timeline
    ap.add_argument("--meta-subdir", default="meta_on_base_calib")
    ap.add_argument("--svr-hp-dir", default=None, help="dir of embed_K{K}_fold{F}.json")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--inner-steps", type=int, default=20)
    ap.add_argument("--inner-lr", type=float, default=1.0)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--dump-dir", default=None, help="per-frame prediction dumps (K=max only)")
    ap.add_argument("--hp-out", default=None, help="write tuned fc_ft HPs json here")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = args.fold_index
    seed = args.seed

    root = Path(args.run_root)
    bb = args.backbone
    base_ckpt = root / f"base/seed{seed}/fold{fold}_best_{bb}_gaze_segmenter.pth"
    meta_ckpt = root / f"{args.meta_subdir}/seed{seed}/fold{fold}_meta_film_{bb}_gaze.pth"

    dataset = build_task_dataset(args.mean_path)
    splits = recording_kfolds(dataset.unique_recordings(), folds=5, seed=seed)
    split = [s for s in splits if s["fold"] == fold][0]
    val_recs, trn_recs = split["val_recordings"], split["train_recordings"]
    val_idx = dataset.indices_for_recordings(val_recs)
    trn_idx = dataset.indices_for_recordings(trn_recs)

    base_model, b_mean, b_std = _load_base_checkpoint(str(base_ckpt), device)
    meta_model, adapter, m_mean, m_std = _load_meta_checkpoint(str(meta_ckpt), device)

    print(f"[fold {fold}] caching task features: val={len(val_idx)} trn={len(trn_idx)}", flush=True)
    bf_t, gz_t, bp_t, be_t = _features_and_preds(
        base_model, dataset, val_idx, b_mean, b_std, device, args.batch_size, args.num_workers)
    mf_t, _, _, _ = _features_and_preds(
        meta_model, dataset, val_idx, m_mean, m_std, device, args.batch_size, args.num_workers)
    bf_trn, gz_trn, _, _ = _features_and_preds(
        base_model, dataset, trn_idx, b_mean, b_std, device, args.batch_size, args.num_workers)

    rows_t = group_rows_by_rec(dataset, val_idx)
    rows_trn = group_rows_by_rec(dataset, trn_idx)
    frames_t = frames_of(dataset, val_idx)
    cleanset = clean_frame_set()

    out_rows = []
    for K in args.ks:
        cds = build_calib_dataset(K, args.mean_path)
        cidx_v = cds.indices_for_recordings(val_recs)
        cidx_t = cds.indices_for_recordings(trn_recs)
        bf_cv, gz_cv, _, be_cv = _features_and_preds(
            base_model, cds, cidx_v, b_mean, b_std, device, args.batch_size, args.num_workers)
        mf_cv, _, _, _ = _features_and_preds(
            meta_model, cds, cidx_v, m_mean, m_std, device, args.batch_size, args.num_workers)
        bf_ct, gz_ct, _, _ = _features_and_preds(
            base_model, cds, cidx_t, b_mean, b_std, device, args.batch_size, args.num_workers)
        rows_cv = group_rows_by_rec(cds, cidx_v)
        rows_ct = group_rows_by_rec(cds, cidx_t)

        # ---- #1: leak-free fc_ft grid on TRAIN recs ----
        hp, hp_err, _ = fcft_tune(base_model, b_mean, b_std, bf_ct, gz_ct,
                                  bf_trn, gz_trn, rows_ct, rows_trn, device)
        print(f"[fold {fold} K={K}] tuned fc_ft: {hp} (train transfer {hp_err:.4f})", flush=True)
        if args.hp_out:
            Path(args.hp_out).parent.mkdir(parents=True, exist_ok=True)
            j = json.load(open(args.hp_out)) if os.path.exists(args.hp_out) else {}
            j[f"K{K}_fold{fold}"] = hp
            json.dump(j, open(args.hp_out, "w"), indent=1)

        # tuned svr_embed HPs from the clean-run swarm search, if available
        svr_hp = dict(C=1.0, gamma="scale", epsilon=0.1)
        if args.svr_hp_dir:
            p = Path(args.svr_hp_dir) / f"embed_K{K}_fold{fold}.json"
            if p.is_file():
                h = json.load(open(p))[str(fold)]
                svr_hp = dict(C=float(h["C"]), gamma=float(h["gamma"]), epsilon=float(h["epsilon"]))

        # ---- #2 eval per val rec ----
        acc = {k: [] for k in ("base", "fcft_def", "fcft_tuned", "svr_embed", "meta",
                               "auto", "auto_cv", "oracle",
                               "base_cl", "fcft_def_cl", "fcft_tuned_cl", "svr_embed_cl",
                               "meta_cl", "auto_cl", "auto_cv_cl", "oracle_cl")}
        pick_cv = {}
        pick_count = {}
        dump_rows = []
        for rec, tq in rows_t.items():
            cs = rows_cv.get(rec)
            if not cs:
                continue
            n = len(cs)
            sup = list(range(n)); qry = list(range(n, n + len(tq)))
            bfe = torch.cat([bf_cv[cs], bf_t[tq]]); mfe = torch.cat([mf_cv[cs], mf_t[tq]])
            bee = torch.cat([be_cv[cs], be_t[tq]]); gz = torch.cat([gz_cv[cs], gz_t[tq]])
            gt_q = gz[qry].numpy()
            # frame numbers + clean mask for this rec's query rows
            fr = np.array([int(dataset.samples[val_idx[r]][-1]) for r in tq])
            clean_mask = np.array([(rec, f) in cleanset for f in fr])

            preds = {}
            preds["base"] = bp_t[tq].numpy()
            preds["fcft_def"] = _fc_ft_predict(base_model, b_mean, b_std, bfe, gz, sup, qry,
                                               device, **FCFT_DEFAULT)
            preds["fcft_tuned"] = _fc_ft_predict(base_model, b_mean, b_std, bfe, gz, sup, qry,
                                                 device, **hp)
            preds["svr_embed"] = _svr_embed_predict(bee, gz, sup, qry, **svr_hp)
            preds["meta"] = _meta_predict((meta_model, adapter, mfe, m_mean, m_std),
                                          sup, qry, gz, device, args.inner_lr, args.inner_steps)

            # LOO by calibration point (unique gaze targets among support)
            y_sup = gz[sup].numpy()
            pts = np.unique(np.round(y_sup, 3), axis=0)
            loo = {"fcft_def": [], "svr_embed": [], "meta": []}
            for p9 in pts:
                is_p = np.all(np.isclose(np.round(y_sup, 3), p9), axis=1)
                s8 = [sup[i] for i in range(n) if not is_p[i]]
                qp = [sup[i] for i in range(n) if is_p[i]]
                if not s8 or not qp:
                    continue
                gt_p = gz[qp].numpy()
                loo["fcft_def"].append(err(_fc_ft_predict(
                    base_model, b_mean, b_std, bfe, gz, s8, qp, device, **FCFT_DEFAULT), gt_p))
                loo["svr_embed"].append(err(_svr_embed_predict(bee, gz, s8, qp, **svr_hp), gt_p))
                loo["meta"].append(err(_meta_predict(
                    (meta_model, adapter, mfe, m_mean, m_std), s8, qp, gz, device,
                    args.inner_lr, args.inner_steps), gt_p))
            sel = min(loo, key=lambda m: np.mean(loo[m]))
            pick_count[sel] = pick_count.get(sel, 0) + 1
            preds["auto"] = preds[sel]

            # cycle-CV selector: per point, first half of frames (by frame id) =
            # cycle 1, second half = cycle 2; fit on one cycle, validate on the
            # other (both directions). Validates across the temporal gap between
            # cycles -- a closer proxy for the calib->task shift than LOO points.
            if n >= 18:
                fr_c = np.array([int(cds.samples[cidx_v[r]][-1]) for r in cs])
                cyc1, cyc2 = [], []
                for p9 in pts:
                    ip = np.where(np.all(np.isclose(np.round(y_sup, 3), p9), axis=1))[0]
                    o = ip[np.argsort(fr_c[ip])]
                    h = len(o) // 2
                    cyc1 += [sup[i] for i in o[:h]]
                    cyc2 += [sup[i] for i in o[h:]]
                cvres = {}
                fits = (
                    ("fcft_def", lambda s8, qp: _fc_ft_predict(
                        base_model, b_mean, b_std, bfe, gz, s8, qp, device, **FCFT_DEFAULT)),
                    ("svr_embed", lambda s8, qp: _svr_embed_predict(bee, gz, s8, qp, **svr_hp)),
                    ("meta", lambda s8, qp: _meta_predict(
                        (meta_model, adapter, mfe, m_mean, m_std), s8, qp, gz, device,
                        args.inner_lr, args.inner_steps)))
                for mname, fitfn in fits:
                    e1 = err(fitfn(cyc1, cyc2), gz[cyc2].numpy())
                    e2 = err(fitfn(cyc2, cyc1), gz[cyc1].numpy())
                    cvres[mname] = 0.5 * (e1 + e2)
                selcv = min(cvres, key=cvres.get)
                pick_cv[selcv] = pick_cv.get(selcv, 0) + 1
                preds["auto_cv"] = preds[selcv]
            else:
                preds["auto_cv"] = preds[sel]
            cand = ("fcft_def", "svr_embed", "meta")
            preds["oracle"] = preds[min(cand, key=lambda m: err(preds[m], gt_q))]

            for name in ("base", "fcft_def", "fcft_tuned", "svr_embed", "meta", "auto", "auto_cv", "oracle"):
                acc[name].append(err(preds[name], gt_q))
                acc[name + "_cl"].append(err(preds[name], gt_q, clean_mask))

            if args.dump_dir and K == max(args.ks):
                for j, (f_, g_) in enumerate(zip(fr, gt_q)):
                    dump_rows.append((rec, int(f_), g_[0], g_[1],
                                      *preds["base"][j], *preds["fcft_tuned"][j],
                                      *preds["meta"][j], int(clean_mask[j])))
            print(f"[fold {fold} K={K}] rec={rec} sel={sel} "
                  + " ".join(f"{k}={acc[k][-1]:.3f}" for k in
                             ("base", "fcft_def", "fcft_tuned", "svr_embed", "meta", "auto", "oracle")),
                  flush=True)

        row = dict(fold=fold, K=K, seed=seed, backbone=bb,
                   fcft_lr=hp["lr"], fcft_steps=hp["steps"], fcft_wd=hp["weight_decay"],
                   picks=";".join(f"{k}:{v}" for k, v in sorted(pick_count.items())),
                   picks_cv=";".join(f"{k}:{v}" for k, v in sorted(pick_cv.items())))
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else float("nan")
        out_rows.append(row)

        if args.dump_dir and dump_rows and K == max(args.ks):
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            fp = Path(args.dump_dir) / f"preds_{bb}_fold{fold}.csv.gz"
            with gzip.open(fp, "wt") as fh:
                fh.write("rec,frame,gt_x,gt_y,base_x,base_y,fcft_x,fcft_y,meta_x,meta_y,clean\n")
                for r in dump_rows:
                    fh.write(",".join(str(x) for x in r) + "\n")
            print(f"[fold {fold}] dumped {len(dump_rows)} frames -> {fp}", flush=True)

    # append csv
    import csv as _csv
    fp = Path(args.csv_out); fp.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not fp.exists()
    with open(fp, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        if write_hdr:
            w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {len(out_rows)} rows -> {fp}", flush=True)


if __name__ == "__main__":
    main()
