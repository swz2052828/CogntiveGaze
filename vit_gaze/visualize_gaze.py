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
        svr_C, svr_g, svr_e = _resolve_svr_hp(args)
        print(f"  svr hp:        C={svr_C} gamma={svr_g} epsilon={svr_e}")
        # Range of the support set the SVR sees: if pred range is tiny here
        # (e.g. the first K enrollment frames all look at one dot), the SVR
        # can't extrapolate beyond it -- the output will look 'small-scale'.
        sup_lo = base_preds[:K].min(axis=0)
        sup_hi = base_preds[:K].max(axis=0)
        print(f"  svr support range (base_preds[:K]): "
              f"x=[{sup_lo[0]:+.2f}, {sup_hi[0]:+.2f}], "
              f"y=[{sup_lo[1]:+.2f}, {sup_hi[1]:+.2f}] cm")
        svr = SVRCalibrator(C=svr_C, epsilon=svr_e, gamma=svr_g).fit(
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


def _compute_extent(gts, preds_by_method, screen_box=None, pad_frac=0.15):
    """Axis extent containing gt + all preds + (optionally) the screen rect.

    ``screen_box`` is ``(x0, y0, x1, y1)`` (anchored, not just (W, H)) or None.
    Returns ``(x0, x1, y0, y1)`` already padded.
    """
    all_xy = np.concatenate([gts] + list(preds_by_method.values()), axis=0)
    finite = all_xy[np.isfinite(all_xy).all(axis=1)]
    if len(finite) == 0:
        finite = all_xy
    xmin, ymin = float(finite[:, 0].min()), float(finite[:, 1].min())
    xmax, ymax = float(finite[:, 0].max()), float(finite[:, 1].max())
    if screen_box is not None:
        sx0, sy0, sx1, sy1 = screen_box
        xmin = min(xmin, sx0)
        xmax = max(xmax, sx1)
        ymin = min(ymin, sy0)
        ymax = max(ymax, sy1)
    xpad = max(xmax - xmin, 1.0) * pad_frac
    ypad = max(ymax - ymin, 1.0) * pad_frac
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _screen_box(gts, screen_cm):
    """Place the screen rectangle in the data's coordinate system.

    The calibration dots in ``gts`` are *shown on the screen* by construction,
    so their bounding box is a lower bound on where the screen is. We compute
    the GT bbox and either return it as-is (when no explicit size given) or
    center a ``screen_cm`` (W, H) rectangle on the GT bbox center (when the
    user wants the full physical screen drawn, even if the dots don't span it).
    """
    if len(gts) == 0:
        return None
    gx0, gy0 = float(gts[:, 0].min()), float(gts[:, 1].min())
    gx1, gy1 = float(gts[:, 0].max()), float(gts[:, 1].max())
    if screen_cm is None:
        # Just bound the dots themselves.
        return (gx0, gy0, gx1, gy1)
    sw, sh = float(screen_cm[0]), float(screen_cm[1])
    cx, cy = 0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1)
    return (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2)


def render_gif(frames, gts, preds_by_method, out_path, enroll_k,
               fps=10, trail=12, max_frames=200, screen_cm=SCREEN_CM, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sel = _subsample(len(frames), max_frames)
    methods = list(preds_by_method.keys())

    # Anchor the screen rectangle in the data's coordinate system using the gt
    # bounding box (the calibration dots define where on the screen they appear).
    # Works for any convention -- screen-origin (positive cm) or camera-centered
    # (negative cm) -- without requiring the user to specify the origin offset.
    screen_box = _screen_box(gts, screen_cm)
    x0, x1, y0, y1 = _compute_extent(gts, preds_by_method, screen_box=screen_box)
    aspect = (y1 - y0) / max(x1 - x0, 1e-6)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 7 * aspect)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)          # NOT inverted: lets either convention render naturally
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    ax.set_title(title or "Gaze: prediction vs ground truth")

    # Faint screen rectangle as a reference frame.
    if screen_box is not None:
        sx0, sy0, sx1, sy1 = screen_box
        ax.add_patch(plt.Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, fill=False,
                                   edgecolor="0.5", lw=1.2, ls="--", zorder=0))
        ax.text(sx0, sy1, " screen", va="top", ha="left",
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


def _resolve_svr_hp(args):
    """Pick (C, gamma, epsilon) for SVR: --svr-hp-json beats explicit flags.

    The svrsearch output JSON looks like ``{"<fold>": {"C": ..., "gamma": ...,
    "epsilon": ..., ...}}``. ``--svr-hp-fold`` says which fold to read (so a
    multi-fold JSON works); falls back to the first key.
    """
    if not getattr(args, "svr_hp_json", None):
        return args.svr_C, args.svr_gamma, args.svr_eps
    import json
    hp_table = json.loads(open(args.svr_hp_json).read())
    fold = getattr(args, "svr_hp_fold", None)
    if fold is None:
        # If JSON has one fold, use it; otherwise require --svr-hp-fold.
        if len(hp_table) == 1:
            (_, hp), = hp_table.items()
        else:
            raise ValueError(
                f"--svr-hp-json {args.svr_hp_json} has folds {list(hp_table.keys())}; "
                f"pass --svr-hp-fold to disambiguate.")
    else:
        key = str(fold)
        if key not in hp_table:
            raise ValueError(
                f"fold {key!r} not in {args.svr_hp_json} (have {list(hp_table.keys())}).")
        hp = hp_table[key]
    return float(hp["C"]), float(hp["gamma"]), float(hp["epsilon"])


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
        "--screen-cm", type=_screen_cm_arg, default=None,
        help="Physical screen size in cm as 'WxH' (e.g. '54.4x30.4'). The "
             "rectangle is *centered on the calibration-dot bounding box* in "
             "data coordinates, so it works for any convention (screen-origin "
             "positive cm or GazeCapture-style camera-centered negative cm). "
             "Default (omitted): use the dot bounding box itself as the screen "
             "rectangle. Pass 'none' to suppress the rectangle entirely.",
    )
    p.add_argument("--inner-steps", type=int, default=20)
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument(
        "--svr-hp-json", default=None,
        help="JSON written by `svrsearch --json-out` (per-fold tuned "
             "(C, gamma, epsilon) for the prediction-space SVR). Overrides "
             "--svr-C / --svr-gamma / --svr-eps when given. STRONGLY recommended "
             "-- the sklearn defaults (C=1) produce an under-regularised SVR "
             "whose predictions cluster near the training-target mean.",
    )
    p.add_argument("--svr-hp-fold", type=int, default=None,
                   help="Fold key to read from --svr-hp-json. Optional if the "
                        "JSON contains only one fold.")
    p.add_argument("--svr-C", type=float, default=1.0,
                   help="Manual SVR C (used only if --svr-hp-json not set). "
                        "sklearn default 1.0 is usually too small for gaze cm "
                        "scales; use the value from `svrsearch` (often 100-300).")
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

    # Loud warning when SVR is requested with sklearn defaults: at gaze cm
    # scales this gives a near-constant SVR output near the target mean
    # ("scale of svr is much smaller than base/meta").
    if "svr" in [m.strip() for m in args.methods.split(",")] \
            and not getattr(args, "svr_hp_json", None) \
            and float(args.svr_C) <= 1.0:
        import sys
        print(
            "WARNING: --svr-C=1.0 (sklearn default). RBF-SVR on gaze cm-scale "
            "targets with C=1 is heavily under-regularised; predictions will "
            "cluster near the training-target mean and look 'small-scale'. "
            "Pass --svr-hp-json runs/.../svr_hp_seedX_foldY.json for the "
            "PSO-tuned hyperparameters.", file=sys.stderr)

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
    # Report where the screen rectangle ended up (data-coord-anchored, not (0,0)).
    sb = _screen_box(gts, args.screen_cm)
    if sb is None:
        print("  screen:        none (no reference rectangle)")
    else:
        sx0, sy0, sx1, sy1 = sb
        src = "centered on gt bbox" if args.screen_cm is not None else "= gt bbox"
        print(f"  screen rect:   x=[{sx0:+.2f}, {sx1:+.2f}], y=[{sy0:+.2f}, {sy1:+.2f}] cm ({src})")

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
