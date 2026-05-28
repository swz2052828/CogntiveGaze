"""Bridge: write vit_gaze model predictions into the per-item gaze files that
``gaze_dynamics.io.load_gaze_data`` reads.

File contract (one file per item, named ``"{sub_id}_{item}"``): three arrays
written back-to-back with ``np.save`` -- predicted xy ``[2, N]``, ground-truth
xy ``[2, N]``, frame indices ``[N]``. This is the exact inverse of
``io.load_gaze_data``, so analyzers can consume model output unchanged.

Coordinate note: ``MakeMeta.py`` stores ``labelDotXCam/Y`` as screen-cm (e.g.
``[0, 54.4]`` x ``[0, 30.4]``), so ``vit_gaze`` predictions land in cm too.
The gaze_dynamics analyzers default to pixels (``screen_res=(1920, 1080)``).
Pass ``transform=geometry.cm_to_pixels`` to ``ViTGazeExporter`` so the exported
files are already in pixels; otherwise call the analyzers with
``screen_res=(54.4, 30.4)`` instead.
"""

import os

import numpy as np


def save_gaze_item(out_dir, sub_id, item, pred_xy, gt_xy, frames):
    """Write one ``{sub_id}_{item}`` file: pred_xy[2,N], gt_xy[2,N], frames[N]."""
    os.makedirs(out_dir, exist_ok=True)
    pred_xy = np.asarray(pred_xy)
    gt_xy = np.asarray(gt_xy)
    frames = np.asarray(frames)
    if pred_xy.ndim != 2 or pred_xy.shape[0] != 2:
        raise ValueError(f"pred_xy must be [2, N], got {pred_xy.shape}")
    if gt_xy.ndim != 2 or gt_xy.shape[0] != 2:
        raise ValueError(f"gt_xy must be [2, N], got {gt_xy.shape}")
    if frames.shape[0] != pred_xy.shape[1]:
        raise ValueError(f"frames length {frames.shape[0]} != N {pred_xy.shape[1]}")
    path = os.path.join(out_dir, f"{sub_id}_{item}")
    with open(path, "wb") as f:
        np.save(f, pred_xy)
        np.save(f, gt_xy)
        np.save(f, frames)
    return path


def _frame_of(dataset, idx):
    """Frame index of a sample (last element of the sample tuple in both datasets)."""
    return int(dataset.samples[idx][-1])


class ViTGazeExporter:
    """Run a trained vit_gaze checkpoint and dump predictions in the gaze-file format.

    torch / vit_gaze are imported lazily so this module loads without them.
    """

    def __init__(self, checkpoint_path, device=None):
        import torch
        from vit_gaze.training import load_checkpoint

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        (self.model, self.gaze_mean, self.gaze_std,
         self.checkpoint, self.input_mode) = load_checkpoint(checkpoint_path, self.device)
        self.model.eval()

    def predict_indices(self, dataset, indices, batch_size=64, num_workers=4, transform=None):
        """Predict gaze for ``dataset[indices]`` in order.

        Returns ``(pred_xy[2,N], gt_xy[2,N], frames[N])``. ``transform`` (if given)
        maps an ``[N, 2]`` array into the analysis coordinate space and is applied
        to both prediction and ground truth.
        """
        import torch
        import torch.utils.data as data
        from vit_gaze.training import denormalize_gaze
        from vit_gaze.models import (
            batch_multistream_for_mode, forward_multistream,
            batch_images_for_mode, forward_for_mode,
        )

        indices = list(indices)
        subset = data.Subset(dataset, indices)
        loader = data.DataLoader(subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=torch.cuda.is_available())
        preds = []
        with torch.no_grad():
            for batch in loader:
                if self.input_mode == "multistream":
                    inputs = batch_multistream_for_mode(batch, self.device)
                    out = forward_multistream(self.model, inputs)
                else:
                    first, second = batch_images_for_mode(batch, self.input_mode, self.device)
                    out = forward_for_mode(self.model, self.input_mode, first, second)
                out = denormalize_gaze(out.float(), self.gaze_mean, self.gaze_std)
                preds.append(out.cpu().numpy())

        pred = np.concatenate(preds, axis=0)        # [N, 2]
        gt = np.asarray(dataset.gazes)[indices]     # [N, 2]
        frames = np.array([_frame_of(dataset, i) for i in indices])
        if transform is not None:
            pred = transform(pred)
            gt = transform(gt)
        return pred.T, gt.T, frames

    def export_items(self, dataset, items, out_dir, transform=None, **kw):
        """Write one file per item. ``items`` maps ``(sub_id, item) -> [indices]``."""
        written = []
        for (sub_id, item), indices in items.items():
            pred_xy, gt_xy, frames = self.predict_indices(
                dataset, indices, transform=transform, **kw)
            written.append(save_gaze_item(out_dir, sub_id, item, pred_xy, gt_xy, frames))
        return written

    def export_by_recording(self, dataset, out_dir, item=0, transform=None, **kw):
        """One file per recording (``sub_id = recording id``), full sequence, time-ordered."""
        written = []
        for rec in np.unique(dataset.recordings):
            indices = dataset.indices_for_recordings([rec])
            indices = sorted(indices, key=lambda i: _frame_of(dataset, i))
            if not indices:
                continue
            pred_xy, gt_xy, frames = self.predict_indices(
                dataset, indices, transform=transform, **kw)
            written.append(save_gaze_item(out_dir, int(rec), item, pred_xy, gt_xy, frames))
        return written


def _build_dataset(args):
    """Build the right vit_gaze dataset for ``args.input_mode``."""
    from vit_gaze.dataset import build_dataset, build_multistream_dataset
    if args.input_mode == "multistream":
        return build_multistream_dataset(args)
    use_synthetic = args.input_mode in ("synthetic", "paired")
    require_synthetic = use_synthetic and not args.allow_missing_synthetic
    return build_dataset(args, use_synthetic=use_synthetic, require_synthetic=require_synthetic)


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Export vit_gaze predictions into gaze_dynamics per-item files "
                    "(one file per recording: pred_xy[2,N], gt_xy[2,N], frames[N]).")
    p.add_argument("--checkpoint", required=True, help="Trained vit_gaze checkpoint (.pth).")
    p.add_argument("--out-dir", default="gaze_dynamics_gaze", help="Where to write the gaze files.")
    p.add_argument("--input-mode", choices=("raw", "synthetic", "paired", "multistream"),
                   default="multistream")
    # dataset location
    p.add_argument("--data-path", required=True)
    p.add_argument("--eye-path", default=None)
    p.add_argument("--mean-path", default="mean7")
    p.add_argument("--metadata-path", default=None)
    p.add_argument("--raw-root", default=None)
    p.add_argument("--synthetic-root", default=None)
    p.add_argument("--raw-folder", default="appleFace")
    p.add_argument("--synthetic-folder", default="appleFaceFake")
    p.add_argument("--face-folder", default="appleFace")
    p.add_argument("--left-eye-folder", default="appleLeftEye")
    p.add_argument("--right-eye-folder", default="appleRightEye")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--eye-size", type=int, default=224)
    p.add_argument("--grid-size", type=int, default=25)
    p.add_argument("--use-grid", action="store_true")
    p.add_argument("--allow-missing-synthetic", action="store_true")
    # inference
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    return p


def main():
    args = build_parser().parse_args()
    dataset = _build_dataset(args)
    exporter = ViTGazeExporter(args.checkpoint, device=args.device)
    written = exporter.export_by_recording(
        dataset, args.out_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Wrote {len(written)} gaze files to {args.out_dir}")


if __name__ == "__main__":
    main()
