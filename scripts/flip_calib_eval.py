"""Calibrated flip-vs-baseline comparison: fc_ft + svr_embed (fixed HPs) on the
flip-right-eye checkpoints vs the clean baseline checkpoints. The eval dataset
mirrors the right eye for the flip arm (matching training). K in {9, 72},
deploy-faithful calib support. No meta (no adapters trained for flip runs)."""
import argparse, sys, glob
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.dataset import MultiStreamGazeDataset
from vit_gaze.metacompare import (_features_and_preds, _fc_ft_predict,
                                  _svr_embed_predict, _load_base_checkpoint)
from vit_gaze.splits import recording_kfolds

DATA = "/springbrook/share/eng/esrpxk/datasets/ProcessedData"
SUPP = "/springbrook/share/eng/esrpxk/datasets"
RUNS = Path("/springbrook/share/eng/esrpxk/runs")
FLIP_ROOT = {  # backbone -> (clean run root, flip run root)
    "itracker": ("meta_pipeline_clean_metacmp_itracker", "meta_pipeline_flip_itracker_lr1e4"),
    "mgazenet": ("meta_pipeline_clean_metacmp_mgazenet", "meta_pipeline_flip_mgazenet_lr1e4"),
    "eyes_only_mobilevitv2": ("meta_pipeline_clean_metacmp_eyes_only_mobilevitv2",
                              "meta_pipeline_flip_eyes_only_mobilevitv2_lr1e4"),
    "eyes_only_convnextv2_binocular": ("meta_pipeline_clean_metacmp_eyes_only_convnextv2_binocular",
                                       "meta_pipeline_flip_eyes_only_convnextv2_binocular_lr1e4"),
    "eyes_only_convnextv2_atto_binocular": ("meta_pipeline_clean_metacmp_eyes_only_convnextv2_atto_binocular",
                                            "meta_pipeline_flip_eyes_only_convnextv2_atto_binocular_lr1e4"),
}

def ds(root, flip):
    return MultiStreamGazeDataset(data_path=root, mean_path="meanno7_clean",
        eye_path=root, metadata_path=None, image_size=224, eye_size=224,
        grid_size=25, use_grid=True, flip_right_eye=flip)

def err(p, g): return float(np.linalg.norm(np.asarray(p)-np.asarray(g), axis=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(FLIP_ROOT))
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda")
    bb, fold = a.backbone, a.fold_index
    out = dict(fold=fold, backbone=bb)
    for arm, rootname, flip in (("clean", FLIP_ROOT[bb][0], False),
                                ("flip", FLIP_ROOT[bb][1], True)):
        ck = RUNS / rootname / f"base/seed42/fold{fold}_best_{bb}_gaze_segmenter.pth"
        model, mean, std = _load_base_checkpoint(str(ck), device)
        task = ds(DATA, flip)
        splits = recording_kfolds(task.unique_recordings(), folds=5, seed=42)
        val = [s for s in splits if s["fold"] == fold][0]["val_recordings"]
        ti = task.indices_for_recordings(val)
        tf_, tg, tp, te = _features_and_preds(model, task, ti, mean, std, device,
                                              a.batch_size, a.num_workers)
        def group(dset, idxs):
            g = {}
            for row, i in enumerate(idxs):
                g.setdefault(int(dset.samples[i][-2]), []).append(row)
            return g
        rt = group(task, ti)
        out[f"{arm}_base"] = err(tp.numpy(), tg.numpy())
        for K in (9, 72):
            cds = ds(f"{SUPP}/calib_support_K{K}", flip)
            ci = cds.indices_for_recordings(val)
            cf, cg, _, ce = _features_and_preds(model, cds, ci, mean, std, device,
                                                a.batch_size, a.num_workers)
            rc = group(cds, ci)
            efc, esv = [], []
            for rec, tq in rt.items():
                cs = rc.get(rec)
                if not cs: continue
                n = len(cs)
                feats = torch.cat([cf[cs], tf_[tq]]); gz = torch.cat([cg[cs], tg[tq]])
                emb = torch.cat([ce[cs], te[tq]])
                sup, qry = list(range(n)), list(range(n, n+len(tq)))
                gt = gz[qry].numpy()
                efc.append(err(_fc_ft_predict(model, mean, std, feats, gz, sup, qry,
                               device, lr=5e-5, steps=20, weight_decay=5e-4), gt))
                esv.append(err(_svr_embed_predict(emb, gz, sup, qry,
                               C=1.0, gamma="scale", epsilon=0.1), gt))
            out[f"{arm}_fcft_K{K}"] = float(np.mean(efc))
            out[f"{arm}_svrE_K{K}"] = float(np.mean(esv))
        del model; torch.cuda.empty_cache()
    import csv
    fp = Path(a.csv_out); fp.parent.mkdir(parents=True, exist_ok=True)
    hdr = not fp.exists()
    with open(fp, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out.keys()))
        if hdr: w.writeheader()
        w.writerow(out)
    print(out, flush=True)

if __name__ == "__main__":
    main()
