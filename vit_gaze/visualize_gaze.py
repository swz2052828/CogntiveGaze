"""Animate predicted vs ground-truth gaze over a recording, before/after calibration.

Renders a GIF of one recording's gaze trace on the screen rectangle:
the ground-truth dot, the raw model prediction, and (optionally) one or more
calibrated predictions (SVR / meta-adapter), each with a fading trail and an
error vector to the ground truth. This lets you *see* where prediction error
occurs and how calibration corrects it, frame by frame.

The first --enroll-k frames are the calibration phase (SVR is fit on them and
the meta-adapter is enrolled on them); they are shaded in the animation so the
"before calibration data arrives vs after" effect is visible.

Usage:
  python -m vit_gaze.visualize_gaze \
      --data-path ../datasets/ProcessedData --eye-path ../datasets/ProcessedData \
      --mean-path meanno7 --input-mode multistream \
      --base-checkpoint base/fold2_best_mobilenet_v3_gaze_segmenter.pth \
      --meta-checkpoint meta_on_base/fold2_meta_film_mobilenet_v3_gaze.pth \
      --rec 6 --methods base,svr,meta --enroll-k 16 \
      --out gaze_rec6.gif --fps 10 --max-frames 200
"""

import argparse

import numpy as np

SCREEN_CM = (54.4, 30.4)   # labelDotXCam/Y span for a 1920x1080 / 54.4x30.4 cm screen

_STYLE = {
    "gt":   dict(color="black",      marker="o", label="Ground truth"),
    "base": dict(color="tab:red",    marker="x", label="Base (no calibration)"),
    "svr":  dict(color="tab:orange", marker="s", label="SVR calibrated"),
    "meta": dict(color="tab:blue",   marker="^", label="Meta calibrated"),
}


def _frame_of(dataset, idx):
    return int(dataset.samples[idx][-1])


def _ordered_indices(dataset, rec):
    idx = dataset.indices_for_recordings([rec])
    if not idx:
        raise ValueError(f"Recording {rec} not found. Available: "
                         f"{sorted(int(r) for r in dataset.unique_recordings())}")
    return sorted(idx, key=lambda i: _frame_of(dataset, i))


def _predict_methods(args, dataset, indices, device):
    """Return frames[N], gts[N,2], and a dict method_name -> preds[N,2] (cm)."""
    import torch
    from .metacompare import _adapt_meta, _features_and_preds, _load_meta_checkpoint
    from .calibration import SVRCalibrator
    from .training import load_checkpoint, normalize_gaze, denormalize_gaze

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    frames = np.array([_frame_of(dataset, i) for i in indices])
    out = {}

    # Base model: predictions + (reused) features for SVR fitting.
    base_model, base_mean, base_std, _, _ = load_checkpoint(args.base_checkpoint, device)
    _, gts_t, base_preds_t = _features_and_preds(
        base_model, dataset, indices, base_mean, base_std, device,
        args.batch_size, args.num_workers)
    gts = gts_t.numpy()
    base_preds = base_preds_t.numpy()
    if "base" in methods:
        out["base"] = base_preds

    K = args.enroll_k
    if "svr" in methods:
        if len(indices) <= K:
            raise ValueError(f"Recording has {len(indices)} frames <= enroll-k {K}.")
        svr = SVRCalibrator(C=args.svr_C, epsilon=args.svr_eps, gamma=args.svr_gamma).fit(
            base_preds[:K], gts[:K])
        out["svr"] = svr.transform(base_preds)

    if "meta" in methods:
        if not args.meta_checkpoint:
            raise ValueError("--methods includes meta but --meta-checkpoint not given.")
        meta_model, adapter, meta_mean, meta_std = _load_meta_checkpoint(
            args.meta_checkpoint, device)
        feats_t, gts_m, _ = _features_and_preds(
            meta_model, dataset, indices, meta_mean, meta_std, device,
            args.batch_size, args.num_workers)
        feats = feats_t.to(device)
        f_sup = feats[:K]
        y_sup = normalize_gaze(gts_m[:K].to(device), meta_mean, meta_std)
        fast = _adapt_meta(meta_model, adapter, list(adapter.parameters()),
                           f_sup, y_sup, args.inner_lr, args.inner_steps)
        with torch.no_grad():
            preds = denormalize_gaze(
                meta_model.readout(adapter.func(feats, fast)).float(),
                meta_mean, meta_std).cpu().numpy()
        out["meta"] = preds

    return frames, gts, out


def _subsample(n, max_frames):
    if n <= max_frames:
        return np.arange(n)
    return np.linspace(0, n - 1, max_frames).round().astype(int)


def _compute_extent(gts, preds_by_method, screen_cm=None, pad_frac=0.15):
    """Axis extent that contains gt + all preds + (optionally) the screen rect.

    Returns ``(x0, x1, y0, y1)`` already padded. Use ``screen_cm=None`` to
    pure-auto-fit; pass ``(W, H)`` to also keep the screen rectangle in view.
    """
    all_xy = np.concatenate([gts] + list(preds_by_method.values()), axis=0)
    finite = all_xy[np.isfinite(all_xy).all(axis=1)]
    if len(finite) == 0:
        finite = all_xy
    xmin, ymin = float(finite[:, 0].min()), float(finite[:, 1].min())
    xmax, ymax = float(finite[:, 0].max()), float(finite[:, 1].max())
    if screen_cm is not None:
        sw, sh = screen_cm
        xmin = min(xmin, 0.0)
        xmax = max(xmax, float(sw))
        ymin = min(ymin, 0.0)
        ymax = max(ymax, float(sh))
    xpad = max(xmax - xmin, 1.0) * pad_frac
    ypad = max(ymax - ymin, 1.0) * pad_frac
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def render_gif(frames, gts, preds_by_method, out_path, enroll_k,
               fps=10, trail=12, max_frames=200, screen_cm=SCREEN_CM, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sel = _subsample(len(frames), max_frames)
    methods = list(preds_by_method.keys())

    # Auto-fit axes to gt + preds, optionally keeping the screen rectangle in view.
    x0, x1, y0, y1 = _compute_extent(gts, preds_by_method, screen_cm=screen_cm)
    aspect = (y1 - y0) / max(x1 - x0, 1e-6)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 7 * aspect)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)          # invert y so screen-top is up
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    ax.set_title(title or "Gaze: prediction vs ground truth")

    # Faint screen rectangle as a reference frame.
    if screen_cm is not None:
        sw, sh = screen_cm
        ax.add_patch(plt.Rectangle((0, 0), sw, sh, fill=False,
                                   edgecolor="0.5", lw=1.2, ls="--", zorder=0))
        ax.text(0, 0, " screen", va="bottom", ha="left",
                color="0.5", fontsize=7, zorder=0)

    # Static legend.
    for name in ["gt"] + methods:
        ax.scatter([], [], **{k: v for k, v in _STYLE[name].items()})
    ax.legend(loc="upper right", fontsize=8)

    # Artists updated per frame.
    gt_pt = ax.scatter([], [], s=90, **{k: v for k, v in _STYLE["gt"].items() if k != "label"})
    pred_pts = {m: ax.scatter([], [], s=70, **{k: v for k, v in _STYLE[m].items() if k != "label"})
                for m in methods}
    err_lines = {m: ax.plot([], [], color=_STYLE[m]["color"], lw=0.8, alpha=0.6)[0]
                 for m in methods}
    trail_lines = {m: ax.plot([], [], color=_STYLE[m]["color"], lw=1.0, alpha=0.3)[0]
                   for m in methods}
    gt_trail = ax.plot([], [], color="black", lw=1.0, alpha=0.3)[0]
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                   family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    def update(k):
        i = sel[k]
        lo = max(0, i - trail)
        gt_trail.set_data(gts[lo:i + 1, 0], gts[lo:i + 1, 1])
        gt_pt.set_offsets([gts[i]])
        lines = [f"frame {frames[i]}"]
        if i < enroll_k:
            lines.append("[CALIBRATING]")
        for m in methods:
            p = preds_by_method[m]
            pred_pts[m].set_offsets([p[i]])
            err_lines[m].set_data([p[i, 0], gts[i, 0]], [p[i, 1], gts[i, 1]])
            trail_lines[m].set_data(p[lo:i + 1, 0], p[lo:i + 1, 1])
            e = float(np.hypot(p[i, 0] - gts[i, 0], p[i, 1] - gts[i, 1]))
            lines.append(f"{m:>5}: {e:5.2f} cm")
        # shade background during the enrollment phase
        ax.set_facecolor("#fff3e0" if i < enroll_k else "white")
        info.set_text("\n".join(lines))
        return [gt_pt, gt_trail, info] + list(pred_pts.values()) \
            + list(err_lines.values()) + list(trail_lines.values())

    anim = FuncAnimation(fig, update, frames=len(sel), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def _build_dataset(args):
    from .dataset import build_multistream_dataset_maybe_video
    return build_multistream_dataset_maybe_video(args)


def _screen_cm_arg(s):
    """argparse type for --screen-cm: 'WxH' (e.g. '54.4x30.4') or 'none' to disable."""
    if s is None or s.lower() in ("none", "off", "false", ""):
        return None
    try:
        w, h = s.lower().replace("x", ",").replace(" ", ",").split(",")[:2]
        return (float(w), float(h))
    except Exception as exc:
        import argparse as _ap
        raise _ap.ArgumentTypeError(
            f"--screen-cm must be 'WxH' (e.g. '54.4x30.4') or 'none'; got {s!r}") from exc


def add_visualize_args(p):
    """Register the visualize-tool flags on an argparse parser. Used by both
    the standalone ``__main__`` and the ``visualize`` subcommand in cli.py."""
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--meta-checkpoint", default=None)
    p.add_argument("--methods", default="base,svr,meta",
                   help="Comma-separated subset of base,svr,meta to overlay.")
    p.add_argument("--rec", type=int, required=True, help="Recording id to animate.")
    p.add_argument("--out", default="gaze.gif")
    p.add_argument("--enroll-k", type=int, default=16,
                   help="Calibration frames (first K, time-ordered).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--trail", type=int, default=12, help="Trailing positions drawn per dot.")
    p.add_argument("--max-frames", type=int, default=200,
                   help="Cap on animated frames (evenly subsampled if longer).")
    p.add_argument(
        "--screen-cm", type=_screen_cm_arg, default=SCREEN_CM,
        help="Physical screen size in cm as 'WxH' (default '54.4x30.4', the "
             "standard 1920x1080 / 24-in monitor). Drawn as a dashed reference "
             "rectangle; the plot axes auto-fit to include both the screen and "
             "the data, so off-screen predictions are still visible. Pass "
             "'none' to suppress the rectangle entirely.",
    )
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument("--svr-C", type=float, default=1.0)
    p.add_argument("--svr-eps", type=float, default=0.1)
    p.add_argument("--svr-gamma", default="scale")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)


def visualize(args):
    """Entry point used by both the standalone script and cli.py."""
    import torch
    try:
        args.svr_gamma = float(args.svr_gamma)
    except (TypeError, ValueError):
        pass  # 'scale' / 'auto'

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_dataset(args)
    indices = _ordered_indices(dataset, args.rec)
    frames, gts, preds = _predict_methods(args, dataset, indices, device)

    # Diagnostic ranges so the user can sanity-check the coordinate system the
    # data actually uses against --screen-cm. If gt range is way outside the
    # screen rect, --screen-cm probably needs adjusting to your dataset's
    # actual screen size.
    print(f"  gt range:      x=[{gts[:, 0].min():+.2f}, {gts[:, 0].max():+.2f}] cm, "
          f"y=[{gts[:, 1].min():+.2f}, {gts[:, 1].max():+.2f}] cm")
    for name, p in preds.items():
        print(f"  {name} range:    x=[{p[:, 0].min():+.2f}, {p[:, 0].max():+.2f}] cm, "
              f"y=[{p[:, 1].min():+.2f}, {p[:, 1].max():+.2f}] cm")
    if args.screen_cm is not None:
        sw, sh = args.screen_cm
        print(f"  screen-cm:     [0, {sw:.2f}] x [0, {sh:.2f}] (dashed rectangle reference)")
    else:
        print("  screen-cm:     none (no reference rectangle)")

    path = render_gif(frames, gts, preds, args.out, enroll_k=args.enroll_k,
                      fps=args.fps, trail=args.trail, max_frames=args.max_frames,
                      screen_cm=args.screen_cm,
                      title=f"Recording {args.rec}: gaze vs ground truth")
    print(f"Wrote {path} ({len(frames)} frames, methods={list(preds.keys())})")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_visualize_args(p)
    # Dataset location flags duplicated here so the standalone script is
    # self-sufficient; the cli.py subcommand uses add_common_args instead.
    p.add_argument("--input-mode", choices=("multistream",), default="multistream")
    p.add_argument("--backbone", default=None)
    p.add_argument("--data-path", required=True)
    p.add_argument("--eye-path", default=None)
    p.add_argument("--mean-path", default="mean7")
    p.add_argument("--metadata-path", default=None)
    p.add_argument("--face-folder", default="appleFace")
    p.add_argument("--left-eye-folder", default="appleLeftEye")
    p.add_argument("--right-eye-folder", default="appleRightEye")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--eye-size", type=int, default=224)
    p.add_argument("--grid-size", type=int, default=25)
    p.add_argument("--use-grid", action="store_true")
    visualize(p.parse_args())


if __name__ == "__main__":
    main()
