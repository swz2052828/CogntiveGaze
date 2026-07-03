"""Deploy-faithful calibration for the RAW single-input models (task #6.3).

Support = the subject's pre-task calibration frames from OriginalCalib (same
frame IDs as calib_support_K72, extracted from the videos and crop-aligned to
OriginalData framing); labels from calib_support_K72/meanno7/metadata.mat.
Query = ALL the subject's task frames. Methods: fc_ft (Zhu defaults) and
svr_embed (fixed C=1/scale). K subsampling: 72 (all) and 9 (first frame of
each of the 9 points x 1 cycle -- deterministic: every 8th frame).

  python scripts/raw_calib_eval.py --backbone raw_vit --fold-index 0 --csv-out ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.dataset import PairedFaceGazeDataset                              # noqa: E402
from vit_gaze.metacompare import _fc_ft_predict, _svr_embed_predict             # noqa: E402
from vit_gaze.splits import recording_kfolds                                    # noqa: E402
from scripts.raw_vs_multistream_eval import (                                   # noqa: E402
    raw_features_and_preds, load_raw_ckpt, err, DATA, RAW, RUNS)

CAL = "/springbrook/share/eng/esrpxk/datasets/OriginalCalib"
CALMETA = "/springbrook/share/eng/esrpxk/datasets/calib_support_K72/meanno7/metadata.mat"
RUN_OF = {"raw_vit": "meta_pipeline_clean_raw_vit_384",
          "raw_mobile_vit": "meta_pipeline_clean_raw_mobile_vit_384",
          "raw_foveal_vit": "meta_pipeline_clean_raw_foveal_vit_384"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(RUN_OF))
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    fold, seed, bb = args.fold_index, args.seed, args.backbone

    ckpt = RUNS / f"{RUN_OF[bb]}/base/seed{seed}/fold{fold}_best_{bb}_gaze_segmenter.pth"
    model, mean, std = load_raw_ckpt(str(ckpt), device)

    task_ds = PairedFaceGazeDataset(
        data_path=DATA, mean_path="meanno7_clean", metadata_path=None,
        raw_root=RAW, synthetic_root=None, raw_folder="raw", synthetic_folder="synthetic",
        image_size=384, use_synthetic=False, require_synthetic=False)
    calib_ds = PairedFaceGazeDataset(
        data_path=DATA, mean_path="meanno7_clean", metadata_path=CALMETA,
        raw_root=CAL, synthetic_root=None, raw_folder="raw", synthetic_folder="synthetic",
        image_size=384, use_synthetic=False, require_synthetic=False)

    splits = recording_kfolds(task_ds.unique_recordings(), folds=5, seed=seed)
    val_recs = [s for s in splits if s["fold"] == fold][0]["val_recordings"]
    t_idx = task_ds.indices_for_recordings(val_recs)
    c_idx = calib_ds.indices_for_recordings(val_recs)

    tf, tg, tp = raw_features_and_preds(model, task_ds, t_idx, mean, std, device,
                                        args.batch_size, args.num_workers)
    cf, cg, _ = raw_features_and_preds(model, calib_ds, c_idx, mean, std, device,
                                       args.batch_size, args.num_workers)

    def group(ds, idxs):
        out = {}
        for row, i in enumerate(idxs):
            out.setdefault(int(ds.samples[i][-2]), []).append(row)
        return out
    rows_t, rows_c = group(task_ds, t_idx), group(calib_ds, c_idx)

    out = dict(fold=fold, backbone=bb, base=err(tp.numpy(), tg.numpy()))
    for K in (9, 72):
        e_fc, e_sv = [], []
        for rec, tq in rows_t.items():
            cs = rows_c.get(rec)
            if not cs:
                continue
            sup_src = cs if K == 72 else cs[::8][:9]     # every 8th = 1 frame/point/cycle
            n = len(sup_src)
            feats = torch.cat([cf[sup_src], tf[tq]])
            gz = torch.cat([cg[sup_src], tg[tq]])
            sup = list(range(n)); qry = list(range(n, n + len(tq)))
            gt = gz[qry].numpy()
            e_fc.append(err(_fc_ft_predict(model, mean, std, feats, gz, sup, qry,
                                           device, lr=5e-5, steps=20, weight_decay=5e-4), gt))
            # embedding-space SVR on the 128-d calibration feature
            with torch.no_grad():
                emb = model.calibration_feature(feats.to(device)).cpu()
            e_sv.append(err(_svr_embed_predict(emb, gz, sup, qry,
                                               C=1.0, gamma="scale", epsilon=0.1), gt))
        out[f"fcft_K{K}"] = float(np.mean(e_fc))
        out[f"svr_embed_K{K}"] = float(np.mean(e_sv))
    import csv
    fp = Path(args.csv_out); fp.parent.mkdir(parents=True, exist_ok=True)
    hdr = not fp.exists()
    with open(fp, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out.keys()))
        if hdr:
            w.writeheader()
        w.writerow(out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
