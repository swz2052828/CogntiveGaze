"""Head-to-head: raw-frame single-input ViT vs facemesh+multistream (task #6).

Per fold: (a) BASE accuracy of both models on the val subjects' task frames;
(b) CALIBRATED accuracy via fc_ft (Zhu defaults) on MATCHED random-K support
draws (K in {9,72}, same seed/indices for both pipelines -- raw frames have no
pre-task calib-support set, so random-K keeps the comparison fair; flagged as
leaky-K in the writeup); (c) GPU forward latency + params + peak memory for
both models (batch 1 and 32). Facemesh CPU latency is measured separately
(facedet env) and added in the analysis.

  python scripts/raw_vs_multistream_eval.py --fold-index 0 \
    --ms-backbone eyes_only_convnextv2_atto_binocular --csv-out ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_gaze.dataset import MultiStreamGazeDataset, PairedFaceGazeDataset      # noqa: E402
from vit_gaze.metacompare import (_features_and_preds, _fc_ft_predict,          # noqa: E402
                                  _load_base_checkpoint)
from vit_gaze.models import create_model                                        # noqa: E402
from vit_gaze.splits import recording_kfolds                                    # noqa: E402
from vit_gaze.training import normalize_gaze, denormalize_gaze                  # noqa: E402

DATA = "/springbrook/share/eng/esrpxk/datasets/ProcessedData"
RAW = "/springbrook/share/eng/esrpxk/datasets/OriginalData"
RUNS = Path("/springbrook/share/eng/esrpxk/runs")


@torch.no_grad()
def raw_features_and_preds(model, dataset, indices, mean, std, device, bs, nw):
    import torch.utils.data as data
    loader = data.DataLoader(data.Subset(dataset, list(indices)), batch_size=bs,
                             shuffle=False, num_workers=nw, pin_memory=True)
    feats, gazes, preds = [], [], []
    for batch in loader:
        img = batch["raw"].to(device, non_blocking=True)
        f = model.forward_features(img)
        p = denormalize_gaze(model.readout(f).float(), mean, std)
        feats.append(f.float().cpu()); gazes.append(batch["gaze"]); preds.append(p.cpu())
    return torch.cat(feats), torch.cat(gazes), torch.cat(preds)


def load_raw_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    saved = ckpt.get("args", {})
    model = create_model("raw", weights="none",
                         backbone=str(saved.get("backbone", "raw_vit")),
                         image_size=int(saved.get("image_size", 384)),
                         output_activation=str(saved.get("output_activation", "none")),
                         gaze_range=float(saved.get("gaze_range", 4.0))).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt["gaze_mean"].to(device), ckpt["gaze_std"].to(device)


def err(pred, gt):
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=1).mean())


def bench(model, example, device, iters=50):
    """Median GPU forward latency (ms) + peak mem (MB)."""
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(*example)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(*example)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), torch.cuda.max_memory_allocated() / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--ms-backbone", default="eyes_only_convnextv2_atto_binocular")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[9, 72])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    fold, seed = args.fold_index, args.seed
    bb = args.ms_backbone

    raw_ckpt = RUNS / f"meta_pipeline_clean_raw_vit_384/base/seed{seed}/fold{fold}_best_raw_vit_gaze_segmenter.pth"
    ms_ckpt = RUNS / f"meta_pipeline_clean_metacmp_{bb}/base/seed{seed}/fold{fold}_best_{bb}_gaze_segmenter.pth"

    # datasets share the meanno7_clean manifest -> identical (rec, frame) sample order
    ms_ds = MultiStreamGazeDataset(data_path=DATA, mean_path="meanno7_clean", eye_path=DATA,
                                   metadata_path=None, image_size=224, eye_size=224,
                                   grid_size=25, use_grid=True)
    raw_ds = PairedFaceGazeDataset(data_path=DATA, mean_path="meanno7_clean", metadata_path=None,
                                   raw_root=RAW, synthetic_root=None, raw_folder="raw",
                                   synthetic_folder="synthetic", image_size=384,
                                   use_synthetic=False, require_synthetic=False)
    splits = recording_kfolds(ms_ds.unique_recordings(), folds=5, seed=seed)
    val_recs = [s for s in splits if s["fold"] == fold][0]["val_recordings"]
    ms_idx = ms_ds.indices_for_recordings(val_recs)
    raw_idx = raw_ds.indices_for_recordings(val_recs)

    ms_model, ms_mean, ms_std = _load_base_checkpoint(str(ms_ckpt), device)
    rw_model, rw_mean, rw_std = load_raw_ckpt(str(raw_ckpt), device)

    ms_f, ms_g, ms_p, _ = _features_and_preds(ms_model, ms_ds, ms_idx, ms_mean, ms_std,
                                              device, args.batch_size, args.num_workers)
    rw_f, rw_g, rw_p = raw_features_and_preds(rw_model, raw_ds, raw_idx, rw_mean, rw_std,
                                              device, args.batch_size, args.num_workers)

    # group rows per rec (same manifest order for both datasets)
    def group(ds, idxs):
        out = {}
        for row, i in enumerate(idxs):
            out.setdefault(int(ds.samples[i][-2]), []).append(row)
        return out
    ms_rows, rw_rows = group(ms_ds, ms_idx), group(raw_ds, raw_idx)

    rng = np.random.default_rng(seed * 1000 + fold)
    out = dict(fold=fold, ms_backbone=bb,
               ms_base=err(ms_p.numpy(), ms_g.numpy()),
               raw_base=err(rw_p.numpy(), rw_g.numpy()))
    for K in args.ks:
        ems, erw = [], []
        for rec in ms_rows:
            tq_ms, tq_rw = ms_rows[rec], rw_rows.get(rec)
            if not tq_rw or len(tq_ms) != len(tq_rw) or len(tq_ms) <= K + 10:
                continue
            for _ in range(args.trials):
                sel = rng.choice(len(tq_ms), size=K, replace=False)
                sup_ms = [tq_ms[j] for j in sel]; qry_ms = [tq_ms[j] for j in range(len(tq_ms)) if j not in set(sel)]
                sup_rw = [tq_rw[j] for j in sel]; qry_rw = [tq_rw[j] for j in range(len(tq_rw)) if j not in set(sel)]
                pms = _fc_ft_predict(ms_model, ms_mean, ms_std, ms_f, ms_g, sup_ms, qry_ms,
                                     device, lr=5e-5, steps=20, weight_decay=5e-4)
                prw = _fc_ft_predict(rw_model, rw_mean, rw_std, rw_f, rw_g, sup_rw, qry_rw,
                                     device, lr=5e-5, steps=20, weight_decay=5e-4)
                ems.append(err(pms, ms_g[qry_ms].numpy()))
                erw.append(err(prw, rw_g[qry_rw].numpy()))
        out[f"ms_fcft_K{K}"] = float(np.mean(ems)); out[f"raw_fcft_K{K}"] = float(np.mean(erw))

    # latency / size (GPU forward only; facemesh CPU cost added in analysis)
    for tag, model, ex in (
            ("ms", ms_model, (torch.randn(1, 3, 224, 224, device=device),
                              torch.randn(1, 3, 224, 224, device=device),
                              torch.randn(1, 3, 224, 224, device=device),
                              torch.randn(1, 625, device=device))),
            ("raw", rw_model, (torch.randn(1, 3, 384, 384, device=device),))):
        lat, mem = bench(model, ex, device)
        out[f"{tag}_lat_ms_b1"] = round(lat, 3); out[f"{tag}_peakMB_b1"] = round(mem, 1)
        out[f"{tag}_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

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
