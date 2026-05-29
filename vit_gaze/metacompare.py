"""Apples-to-apples comparison at matched K: base / SVR / meta.

For each held-out recording in the current CV split:
* Cache fused features for all of the recording's frames using both checkpoints'
  encoders (they share the same architecture; usually the same weights).
* Repeat ``--trials`` times with different random K-subset draws:
  - **base**: predict on the query set with no calibration.
  - **SVR**: fit two RBF-SVRs on K (base_pred_xy, true_xy) pairs from the support
    set; transform the base predictions on the query set.
  - **meta**: from the meta-learned adapter init, run ``--inner-steps`` of SGD
    on the K support features; predict on the query features with the adapted
    adapter and the (meta-tuned) head.
* Report per-recording mean coord error (cm) for the three methods, then per
  fold and overall.

This is the headline number against the ~5 cm geometric floor. By scoring all
three on **the same support/query draws and the same K**, the comparison is
not confounded by sampling.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from . import accel
from .calibration import SVRCalibrator
from .dataset import build_multistream_dataset
from .models import batch_multistream_for_mode, create_model
from .multistream_backbones.adapter import MultistreamBackboneBase
from .multistream_backbones.adapters import make_adapter
from .splits import recording_kfolds, select_splits
from .training import denormalize_gaze, log, normalize_gaze


def _require_meta_support(model):
    if type(model).forward_features is MultistreamBackboneBase.forward_features:
        raise ValueError(
            f"backbone {type(model).__name__} does not implement forward_features; "
            f"it cannot be used with the meta-learned calibration path.")
    return model


def metacompare(args):
    log.open(getattr(args, "log_file", None))
    try:
        _run_metacompare(args)
    finally:
        log.close()


def _run_metacompare(args):
    if args.input_mode != "multistream":
        raise ValueError("metacompare is multistream-only.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))

    dataset = build_multistream_dataset(args)
    splits_all = recording_kfolds(dataset.unique_recordings(), folds=args.folds, seed=args.seed)
    splits = select_splits(splits_all, args.fold_index)

    log(f"Device: {device}")
    log(f"metacompare K={args.k} trials={args.trials} inner_steps={args.inner_steps} "
        f"inner_lr={args.inner_lr} svr_C={args.svr_C}")

    fold_rows = []
    for split in splits:
        fold_rows.append(_compare_one_fold(args, dataset, split, device))

    if fold_rows:
        methods = _active_methods(args)
        agg = {k: float(np.nanmean([row[k] for row in fold_rows])) for k in methods}
        parts = " ".join(f"mean_{k}={agg[k]:.4f}" for k in methods)
        line = f"metacompare CV summary folds={len(fold_rows)} K={args.k} {parts}"
        line += f" svr_gain={agg['base'] - agg['svr']:.4f} meta_gain={agg['base'] - agg['meta']:.4f}"
        line += f" meta_vs_svr={agg['svr'] - agg['meta']:.4f}"
        if "svr_embed" in methods:
            line += (f" svr_embed_gain={agg['base'] - agg['svr_embed']:.4f}"
                     f" svr_embed_vs_svr={agg['svr'] - agg['svr_embed']:.4f}"
                     f" meta_vs_svr_embed={agg['svr_embed'] - agg['meta']:.4f}")
        if "fc_ft" in methods:
            line += (f" fc_ft_gain={agg['base'] - agg['fc_ft']:.4f}"
                     f" fc_ft_vs_svr={agg['svr'] - agg['fc_ft']:.4f}"
                     f" meta_vs_fc_ft={agg['fc_ft'] - agg['meta']:.4f}")
        if "meta_adv" in methods:
            line += (f" meta_adv_gain={agg['base'] - agg['meta_adv']:.4f}"
                     f" meta_adv_vs_svr={agg['svr'] - agg['meta_adv']:.4f}"
                     f" meta_adv_vs_meta={agg['meta'] - agg['meta_adv']:.4f}")
        log(line)


def _active_methods(args):
    methods = ["base", "svr"]
    if getattr(args, "svr_embed", False):
        methods.append("svr_embed")
    if getattr(args, "fc_ft", False):
        methods.append("fc_ft")
    methods.append("meta")
    if getattr(args, "meta_adv_checkpoint", None):
        methods.append("meta_adv")
    return methods


def _load_meta_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    saved = ckpt.get("args", {})
    model = create_model(
        input_mode="multistream",
        weights="none",
        freeze_encoder=True,
        use_grid=bool(saved.get("use_grid", False)),
        grid_size=int(saved.get("grid_size", 25)),
        backbone=str(saved.get("backbone", "vit")),
    ).to(device)
    _require_meta_support(model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    adapter = make_adapter(
        ckpt["adapter_kind"], int(ckpt["adapter_dim"]),
        rank=int(ckpt.get("lora_rank", 8)),
        alpha=float(ckpt.get("lora_alpha", 8.0)),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    return model, adapter, ckpt["gaze_mean"].to(device), ckpt["gaze_std"].to(device)


def _load_base_checkpoint(path, device):
    """Load a standard `train` checkpoint (encoder+head, no adapter)."""
    ckpt = torch.load(path, map_location=device)
    saved = ckpt.get("args", {})
    model = create_model(
        input_mode="multistream",
        weights="none",
        freeze_encoder=False,
        use_grid=bool(saved.get("use_grid", False)),
        grid_size=int(saved.get("grid_size", 25)),
        backbone=str(saved.get("backbone", "vit")),
    ).to(device)
    _require_meta_support(model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["gaze_mean"].to(device), ckpt["gaze_std"].to(device)


@torch.no_grad()
def _features_and_preds(model, dataset, indices, gaze_mean, gaze_std,
                        device, batch_size, num_workers):
    """Return (feats[N,dim], gazes[N,2], base_pred_xy[N,2]) on CPU as numpy/torch."""
    loader = data.DataLoader(
        data.Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    feats, gazes, preds = [], [], []
    for batch in loader:
        inputs = batch_multistream_for_mode(batch, device)
        f = model.forward_features(
            inputs["face"], inputs["eye_left"], inputs["eye_right"], inputs.get("grid"))
        p_norm = model.readout(f).float()
        p = denormalize_gaze(p_norm, gaze_mean, gaze_std)
        feats.append(f.float().cpu())
        gazes.append(batch["gaze"])
        preds.append(p.cpu())
    return torch.cat(feats), torch.cat(gazes), torch.cat(preds)


def _adapt_meta(model, adapter, init_params, f_sup, y_sup, inner_lr, inner_steps):
    fast = [p.detach().clone().requires_grad_(True) for p in init_params]
    for _ in range(inner_steps):
        pred = model.readout(adapter.func(f_sup, fast))
        loss = F.smooth_l1_loss(pred, y_sup)
        grads = torch.autograd.grad(loss, fast)
        fast = [(p - inner_lr * g).detach().requires_grad_(True)
                for p, g in zip(fast, grads)]
    return fast


def _fc_ft_predict(base_model, base_mean, base_std, base_feats, gazes,
                   sup_rows, qry_rows, device, lr, steps, weight_decay):
    """Fc-only fine-tune baseline (Zhu et al., finetuning_freezen.py).

    Clones the base model's readout, freezes everything else (implicit -- only
    readout params are passed to the optimizer), Adam-trains on the K support
    features for ``steps`` full-batch SGD steps, predicts on query features.
    The encoder is never re-run: we reuse the cached fused features that the
    base predictions were computed from, which is exactly what their fc-only
    fine-tune would see if the encoder were frozen (and it is).
    """
    import copy
    readout = copy.deepcopy(base_model.readout).to(device)
    readout.train()
    opt = torch.optim.Adam(readout.parameters(), lr=lr, weight_decay=weight_decay)
    f_sup = base_feats[sup_rows].to(device)
    y_sup = normalize_gaze(gazes[sup_rows].to(device), base_mean, base_std)
    f_qry = base_feats[qry_rows].to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(readout(f_sup), y_sup)
        loss.backward()
        opt.step()
    readout.eval()
    with torch.no_grad():
        return denormalize_gaze(readout(f_qry).float(), base_mean, base_std).cpu().numpy()


def _svr_embed_predict(base_feats, gazes, sup_rows, qry_rows, C, gamma, epsilon):
    """SVR-on-embeddings baseline (Zhu et al., SwarmIntelligentCalibration).

    Fits two RBF-SVRs on K support pairs of (fused feature -> coord), then
    predicts on the query features. This replaces the readout entirely with a
    per-subject SVR -- structurally what Zhu et al. do, but applied to OUR
    cached fused features so the comparison is fair (same encoder, same K).

    Numerics: the feature dim (~2304) > K at small K, so the SVR is in the
    underdetermined regime. That is part of what the comparison is meant to
    expose: meta-adapter and prediction-space SVR both anchor on a working
    base model, whereas this method must learn the full feature -> gaze map
    from K points.
    """
    from sklearn.svm import SVR
    f_sup = base_feats[sup_rows].numpy()
    f_qry = base_feats[qry_rows].numpy()
    y_sup = gazes[sup_rows].numpy()
    svr_x = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(f_sup, y_sup[:, 0])
    svr_y = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(f_sup, y_sup[:, 1])
    return np.stack([svr_x.predict(f_qry), svr_y.predict(f_qry)], axis=1)


def _meta_predict(bundle, sup_rows, qry_rows, gazes, device, inner_lr, inner_steps):
    """Adapt a (model, adapter, feats, mean, std) bundle on support, predict on query."""
    model, adapter, feats, mean, std = bundle
    init_params = list(adapter.parameters())
    f_sup = feats[sup_rows].to(device)
    y_sup = normalize_gaze(gazes[sup_rows].to(device), mean, std)
    f_qry = feats[qry_rows].to(device)
    fast = _adapt_meta(model, adapter, init_params, f_sup, y_sup, inner_lr, inner_steps)
    with torch.no_grad():
        return denormalize_gaze(
            model.readout(adapter.func(f_qry, fast)).float(), mean, std).cpu().numpy()


def _compare_one_fold(args, dataset, split, device):
    fold = split["fold"]
    val_idx = dataset.indices_for_recordings(split["val_recordings"])
    if not val_idx:
        raise RuntimeError(f"Fold {fold} has no validation indices.")

    methods = _active_methods(args)

    # Base model supplies the predictions SVR calibrates; each meta model supplies
    # its own fused features (subject-adv features differ from the plain ones).
    base_model, base_mean, base_std = _load_base_checkpoint(args.base_checkpoint, device)
    meta_model, adapter, meta_mean, meta_std = _load_meta_checkpoint(args.meta_checkpoint, device)

    log(f"Fold {fold} caching features val_recordings={split['val_recordings']}")
    base_feats, gazes_b, base_preds = _features_and_preds(
        base_model, dataset, val_idx, base_mean, base_std, device,
        args.batch_size, args.num_workers)
    meta_feats, gazes_m, _ = _features_and_preds(
        meta_model, dataset, val_idx, meta_mean, meta_std, device,
        args.batch_size, args.num_workers)
    assert torch.allclose(gazes_b, gazes_m), "Datasets returned different ground truth orderings."
    meta_bundle = (meta_model, adapter, meta_feats, meta_mean, meta_std)

    adv_bundle = None
    if "meta_adv" in methods:
        adv_model, adv_adapter, adv_mean, adv_std = _load_meta_checkpoint(
            args.meta_adv_checkpoint, device)
        adv_feats, gazes_a, _ = _features_and_preds(
            adv_model, dataset, val_idx, adv_mean, adv_std, device,
            args.batch_size, args.num_workers)
        assert torch.allclose(gazes_b, gazes_a), "meta-adv dataset ordering mismatch."
        adv_bundle = (adv_model, adv_adapter, adv_feats, adv_mean, adv_std)

    recs = torch.tensor([int(dataset.samples[i][-2]) for i in val_idx])
    rows_by_rec = {}
    for row, r in enumerate(recs.tolist()):
        rows_by_rec.setdefault(int(r), []).append(row)

    rng = random.Random(args.seed + fold)
    K = args.k
    per_rec = {k: [] for k in methods}

    for rec, rows in rows_by_rec.items():
        if len(rows) <= K:
            log(f"Fold {fold} rec={rec} skipped (only {len(rows)} frames, need > {K})")
            continue
        errs = {k: [] for k in methods}
        for _ in range(args.trials):
            shuffled = rows[:]
            rng.shuffle(shuffled)
            sup_rows, qry_rows = shuffled[:K], shuffled[K:]
            gt_q = gazes_b[qry_rows].numpy()

            # base (no calibration)
            pred_q_base = base_preds[qry_rows].numpy()
            errs["base"].append(float(np.linalg.norm(pred_q_base - gt_q, axis=1).mean()))

            # SVR fit on K support pairs from base predictions
            svr = SVRCalibrator(C=args.svr_C, epsilon=args.svr_eps, gamma=args.svr_gamma).fit(
                base_preds[sup_rows].numpy(), gazes_b[sup_rows].numpy())
            pred_q_svr = svr.transform(pred_q_base)
            errs["svr"].append(float(np.linalg.norm(pred_q_svr - gt_q, axis=1).mean()))

            # SVR on embeddings (Zhu et al.'s actual recipe): SVR replaces the readout.
            if "svr_embed" in methods:
                pred_q_se = _svr_embed_predict(
                    base_feats, gazes_b, sup_rows, qry_rows,
                    C=args.svr_embed_C, gamma=args.svr_embed_gamma,
                    epsilon=args.svr_embed_eps)
                errs["svr_embed"].append(float(np.linalg.norm(pred_q_se - gt_q, axis=1).mean()))

            # fc_ft: head-only fine-tune on support features (Zhu et al. baseline)
            if "fc_ft" in methods:
                pred_q_fc = _fc_ft_predict(
                    base_model, base_mean, base_std, base_feats, gazes_b,
                    sup_rows, qry_rows, device,
                    lr=args.fc_ft_lr, steps=args.fc_ft_steps,
                    weight_decay=args.fc_ft_weight_decay)
                errs["fc_ft"].append(float(np.linalg.norm(pred_q_fc - gt_q, axis=1).mean()))

            # meta: adapt on support features, predict on query features
            pred_q_meta = _meta_predict(meta_bundle, sup_rows, qry_rows, gazes_m, device,
                                        args.inner_lr, args.inner_steps)
            errs["meta"].append(float(np.linalg.norm(pred_q_meta - gt_q, axis=1).mean()))

            # meta on subject-adv features (optional 4th method)
            if adv_bundle is not None:
                pred_q_adv = _meta_predict(adv_bundle, sup_rows, qry_rows, gazes_m, device,
                                           args.inner_lr, args.inner_steps)
                errs["meta_adv"].append(float(np.linalg.norm(pred_q_adv - gt_q, axis=1).mean()))

        means = {k: float(np.mean(errs[k])) for k in methods}
        for k in methods:
            per_rec[k].append(means[k])
        log(f"Fold {fold} rec={rec} K={K} trials={args.trials} "
            + " ".join(f"{k}={means[k]:.4f}" for k in methods))

    fold_summary = {"fold": fold}
    for k in methods:
        fold_summary[k] = float(np.mean(per_rec[k])) if per_rec[k] else float("nan")
    # Keep the CSV schema stable regardless of which methods ran this time.
    for k in ("svr_embed", "fc_ft", "meta_adv"):
        fold_summary.setdefault(k, float("nan"))
    log(f"Fold {fold} done K={K} "
        + " ".join(f"mean_{k}={fold_summary[k]:.4f}" for k in methods))

    if args.csv_out:
        _append_csv(args.csv_out, fold_summary, args)
    return fold_summary


def _append_csv(path, row, args):
    import csv
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["fold", "seed", "K", "trials", "inner_steps", "inner_lr",
                        "svr_C", "base", "svr", "svr_embed", "fc_ft", "meta", "meta_adv"])
        w.writerow([row["fold"], args.seed, args.k, args.trials, args.inner_steps, args.inner_lr,
                    args.svr_C, row["base"], row["svr"],
                    row.get("svr_embed", float("nan")),
                    row.get("fc_ft", float("nan")), row["meta"],
                    row.get("meta_adv", float("nan"))])
