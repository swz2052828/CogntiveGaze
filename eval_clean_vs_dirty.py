"""Step 1: re-evaluate EXISTING (blink-contaminated-trained) base checkpoints on
the held-out val recordings, reporting the gaze coord-error over ALL val frames
(dirty) vs over the non-blink subset (clean). The gap = how much the blink
frames were inflating the reported error.

For each fold the model NEVER saw its val recordings in training, so its val set
is a faithful "test set" for that fold; we just split each fold's val frames into
blink / non-blink using the +/-1-dilated contamination list and score each.

One forward pass over the DIRTY val set per fold; no retraining. Runs on a GPU
SLURM node. Writes one CSV row per (backbone, fold) + a cohort summary line.
"""
import argparse
import csv
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from vit_gaze.dataset import build_multistream_dataset
from vit_gaze.models import batch_multistream_for_mode, forward_multistream
from vit_gaze.training import denormalize_gaze, load_checkpoint

CONTAM = "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/train_blink_overlap/blink_contaminated_frames_dilate1.csv"


def load_contam():
    s = set()
    for r in csv.DictReader(open(CONTAM)):
        s.add((int(r["recNum"]), int(r["frameIndex"])))
    return s


def dataset_from_args(saved):
    """Build the DIRTY multistream val dataset from a checkpoint's saved args."""
    ns = Namespace(
        data_path=saved["data_path"],
        mean_path="meanno7",            # dirty metadata (includes blink frames)
        eye_path=saved.get("eye_path"),
        metadata_path=None,
        face_folder=saved.get("face_folder", "appleFace"),
        left_eye_folder=saved.get("left_eye_folder", "appleLeftEye"),
        right_eye_folder=saved.get("right_eye_folder", "appleRightEye"),
        image_size=saved.get("image_size", 224),
        eye_size=saved.get("eye_size", 224),
        grid_size=saved.get("grid_size", 25),
        use_grid=saved.get("use_grid", False),
    )
    return build_multistream_dataset(ns)


@torch.no_grad()
def per_sample_error(model, loader, gaze_mean, gaze_std, device):
    """Return (recs, frames, errs_cm) arrays over the loader."""
    recs, frames, errs = [], [], []
    for batch in loader:
        inputs = batch_multistream_for_mode(batch, device)
        pred_norm = forward_multistream(model, inputs).float()
        pred = denormalize_gaze(pred_norm, gaze_mean, gaze_std)
        gaze = batch["gaze"].to(device)
        e = torch.linalg.norm(pred - gaze, dim=1)
        errs.append(e.cpu().numpy())
        recs.append(batch["rec"].numpy())
        frames.append(batch["frame"].numpy())
    return (np.concatenate(recs), np.concatenate(frames), np.concatenate(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="runs/meta_pipeline_<run> dir with base/seed42/foldN_best_*.pth")
    ap.add_argument("--backbone", default=None,
                    help="filter checkpoints to this backbone (needed for the "
                         "shared meta_pipeline dir that holds several backbones)")
    ap.add_argument("--subdir", default="base",
                    help="checkpoint subdir under the run dir: 'base' or 'adv'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contam = load_contam()
    base = Path(args.run_dir) / args.subdir / f"seed{args.seed}"
    pat = f"fold*_best_{args.backbone}_gaze_segmenter.pth" if args.backbone \
        else "fold*_best_*_gaze_segmenter.pth"
    ckpts = sorted(base.glob(pat))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {base} matching {pat}")
    backbone = ckpts[0].name.split("_best_")[1].rsplit("_gaze_segmenter", 1)[0]
    print(f"run={args.run_dir} backbone={backbone} folds={len(ckpts)} device={device}", flush=True)

    dataset = None
    rows = []
    for ck in ckpts:
        fold = int(ck.name.split("fold")[1].split("_")[0])
        model, gaze_mean, gaze_std, checkpoint, _ = load_checkpoint(ck, device)
        if dataset is None:
            dataset = dataset_from_args(checkpoint["args"])
        val_recs = checkpoint["val_recordings"]
        idx = dataset.indices_for_recordings(val_recs)
        loader = data.DataLoader(
            data.Subset(dataset, idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False)
        recs, frames, errs = per_sample_error(model, loader, gaze_mean, gaze_std, device)
        is_blink = np.array([(int(r), int(f)) in contam for r, f in zip(recs, frames)])
        n = len(errs); nb = int(is_blink.sum())
        err_all = float(errs.mean())
        err_clean = float(errs[~is_blink].mean()) if (~is_blink).any() else float("nan")
        err_blink = float(errs[is_blink].mean()) if nb else float("nan")
        rows.append(dict(backbone=backbone, fold=fold, n_val=n, n_blink=nb,
                         err_all=err_all, err_clean=err_clean, err_blink=err_blink,
                         val_recordings="|".join(map(str, val_recs))))
        print(f"  fold{fold}: n={n} blink={nb} "
              f"err_all={err_all:.4f} err_clean={err_clean:.4f} err_blink={err_blink:.4f}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    # cohort (sample-weighted) summary across folds
    tot_n = sum(r["n_val"] for r in rows)
    tot_nb = sum(r["n_blink"] for r in rows)
    w_all = sum(r["err_all"] * r["n_val"] for r in rows) / tot_n
    w_clean = sum(r["err_clean"] * (r["n_val"] - r["n_blink"]) for r in rows) / (tot_n - tot_nb)
    fold_all = np.mean([r["err_all"] for r in rows])
    fold_clean = np.mean([r["err_clean"] for r in rows])
    print(f"SUMMARY {backbone}: fold-mean err_all={fold_all:.4f} err_clean={fold_clean:.4f} "
          f"(delta={fold_all-fold_clean:+.4f}); sample-wt err_all={w_all:.4f} err_clean={w_clean:.4f}",
          flush=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
