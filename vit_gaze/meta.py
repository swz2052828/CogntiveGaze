"""Meta-learned per-subject calibration (FOMAML / ANIL) for multistream gaze.

Motivation: every backbone plateaus near a ~5 cm floor set by per-subject
head/distance geometry, and the standard remedy is a post-hoc per-subject SVR
that linearly corrects the 2D output. This module learns a *feature-space*,
nonlinear correction instead, and meta-trains it so that adapting on only K
calibration frames generalizes to the rest of the subject's session -- i.e. the
enrollment scenario, optimized end to end. It composes with --subject-adv and
is meant to be compared head-to-head with (and on top of) SVR.

Method (ANIL; Raghu et al. 2020 + FOMAML; Finn et al. 2017):
* Each recording is a task. An episode splits a recording into a support set
  (K calibration frames) and a query set (the rest).
* The encoder (everything before the readout) is frozen, so the fused
  per-stream feature (``model.forward_features``) is constant per frame -- we
  cache it once per fold and meta-train purely on cached ``[N, dim]`` features
  (no backbone in the loop). Works for any multistream backbone that
  implements ``forward_features`` (all five: vit + the four CNN baselines).
  The meta-learned parameters are the shared gaze *readout* and the *adapter
  init* (FiLM or LoRA).
* Inner loop: from the adapter init, take a few SGD steps on the support set
  (only the adapter moves). Outer loop (first-order): minimize the post-
  adaptation query loss w.r.t. the head and the adapter init.

Evaluation reports pre- vs post-adaptation coordinate error on held-out
recordings -- the apples-to-apples number against the SVR-calibrated floor.
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data

from . import accel
from .dataset import build_multistream_dataset
from .models import batch_multistream_for_mode, create_model
from .multistream_backbones.adapter import MultistreamBackboneBase
from .multistream_backbones.adapters import make_adapter
from .splits import recording_kfolds, select_splits
from .training import denormalize_gaze, log, normalize_gaze


def meta_train(args):
    log.open(getattr(args, "log_file", None))
    try:
        _run_meta(args)
    finally:
        log.close()


def _run_meta(args):
    if args.input_mode != "multistream":
        raise ValueError("metatrain is multistream-only; pass --input-mode multistream.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))

    dataset = build_multistream_dataset(args)
    all_splits = recording_kfolds(dataset.unique_recordings(), folds=args.folds, seed=args.seed)
    splits = select_splits(all_splits, args.fold_index)

    log(f"Device: {device}")
    log(f"Meta cross validation: {args.folds} folds by recording id")
    log(f"Meta config adapter={args.adapter} inner_steps={args.inner_steps} "
        f"inner_lr={args.inner_lr} outer_lr={args.outer_lr} "
        f"meta_iters={args.meta_iters} tasks_per_batch={args.tasks_per_batch} "
        f"support={args.meta_support} query={args.meta_query}")

    summaries = []
    for split in splits:
        summaries.append(_meta_one_fold(args, dataset, split, device))

    if summaries:
        pre = sum(s["pre_error"] for s in summaries) / len(summaries)
        post = sum(s["post_error"] for s in summaries) / len(summaries)
        log(f"Meta CV summary folds={len(summaries)} "
            f"mean_pre_adapt_error={pre:.6f} mean_post_adapt_error={post:.6f} "
            f"mean_improvement={pre - post:.6f}")


def _build_base_model(args, device):
    model = create_model(
        args.input_mode,
        weights=args.weights,
        freeze_encoder=True,  # ANIL: encoder frozen so features can be cached
        use_grid=getattr(args, "use_grid", False),
        grid_size=getattr(args, "grid_size", 25),
        backbone=getattr(args, "backbone", "vit"),
    ).to(device)
    if not _supports_meta(model):
        raise ValueError(
            f"--backbone {getattr(args, 'backbone', 'vit')} does not implement "
            f"forward_features; metatrain needs it."
        )
    if getattr(args, "init_checkpoint", None):
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"Loaded init checkpoint {args.init_checkpoint} "
            f"(missing={len(missing)} unexpected={len(unexpected)})")
    # ANIL: freeze everything except the readout so the fused features are
    # constant per frame (cacheable) and only the readout + adapter init are
    # meta-learned. Generalizes the ViT-specific encoder freeze to any backbone.
    readout_params = set(model.readout.parameters())
    for p in model.parameters():
        p.requires_grad = p in readout_params
    return model


def _supports_meta(model):
    """True if the backbone overrides forward_features (i.e. opts into meta)."""
    return type(model).forward_features is not MultistreamBackboneBase.forward_features


@torch.no_grad()
def _cache_features(args, model, dataset, indices, device):
    """Run the frozen encoder once over ``indices`` and cache fused features.

    Returns CPU tensors (feats[N, dim], gazes[N, 2], recs[N]) so meta-training
    indexes them without re-decoding images or re-running the ViT.
    """
    model.eval()
    loader = data.DataLoader(
        data.Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    amp_enabled, amp_dtype = accel.resolve_amp(device, getattr(args, "amp", False))
    feats, gazes, recs = [], [], []
    for batch in loader:
        inputs = batch_multistream_for_mode(batch, device)
        with accel.autocast(device, amp_enabled, amp_dtype):
            f = model.forward_features(
                inputs["face"], inputs["eye_left"], inputs["eye_right"], inputs.get("grid"))
        feats.append(f.float().cpu())
        gazes.append(batch["gaze"])
        recs.append(batch["rec"])
    return torch.cat(feats), torch.cat(gazes), torch.cat(recs)


def _rec_to_rows(recs):
    """Map each recording id to the row indices (into the cached arrays) it owns."""
    out = {}
    recs = recs.tolist()
    for row, r in enumerate(recs):
        out.setdefault(int(r), []).append(row)
    return out


def _inner_adapt(model, adapter, init_params, f_sup, y_sup, inner_lr, inner_steps):
    """FOMAML inner loop: SGD on cloned adapter params over the support set.

    Returns the adapted "fast" params (detached leaves). The encoder is frozen
    and features are cached, so only the adapter moves; the head is shared.
    """
    fast = [p.detach().clone().requires_grad_(True) for p in init_params]
    for _ in range(inner_steps):
        pred = model.readout(adapter.func(f_sup, fast))
        loss = F.smooth_l1_loss(pred, y_sup)
        grads = torch.autograd.grad(loss, fast)
        fast = [(w - inner_lr * g).detach().requires_grad_(True)
                for w, g in zip(fast, grads)]
    return fast


def _meta_one_fold(args, dataset, split, device):
    fold = split["fold"]
    train_idx = dataset.indices_for_recordings(split["train_recordings"])
    val_idx = dataset.indices_for_recordings(split["val_recordings"])
    if not train_idx or not val_idx:
        raise RuntimeError(f"Fold {fold} has empty train or validation indices.")

    model = _build_base_model(args, device)

    log(f"Fold {fold} caching features "
        f"train_recordings={split['train_recordings']} "
        f"val_recordings={split['val_recordings']}")
    tr_feats, tr_gazes, tr_recs = _cache_features(args, model, dataset, train_idx, device)
    va_feats, va_gazes, va_recs = _cache_features(args, model, dataset, val_idx, device)

    gaze_mean = tr_gazes.mean(dim=0).to(device)
    gaze_std = tr_gazes.std(dim=0).clamp_min(1e-6).to(device)
    dim = tr_feats.shape[1]

    adapter = make_adapter(
        args.adapter, dim, rank=getattr(args, "lora_rank", 8),
        alpha=getattr(args, "lora_alpha", 8.0)).to(device)

    meta_params = list(model.readout.parameters()) + list(adapter.parameters())
    meta_opt = torch.optim.AdamW(meta_params, lr=args.outer_lr)

    tr_rows = _rec_to_rows(tr_recs)
    task_recs = [r for r, rows in tr_rows.items()
                 if len(rows) >= args.meta_support + 1]
    if not task_recs:
        raise RuntimeError(
            f"Fold {fold}: no training recording has >= {args.meta_support + 1} "
            f"frames; lower --meta-support.")

    rng = random.Random(args.seed + fold)
    model.eval()  # disable head dropout for stable adaptation
    K, Q = args.meta_support, args.meta_query

    for it in range(1, args.meta_iters + 1):
        meta_opt.zero_grad(set_to_none=True)
        tasks = rng.sample(task_recs, k=min(args.tasks_per_batch, len(task_recs)))
        adapter_grad = [torch.zeros_like(p) for p in adapter.parameters()]

        for rec in tasks:
            rows = tr_rows[rec][:]
            rng.shuffle(rows)
            sup_rows = rows[:K]
            qry_rows = rows[K:K + Q] if len(rows) > K else rows[:Q]
            f_sup = tr_feats[sup_rows].to(device)
            y_sup = normalize_gaze(tr_gazes[sup_rows].to(device), gaze_mean, gaze_std)
            f_qry = tr_feats[qry_rows].to(device)
            y_qry = normalize_gaze(tr_gazes[qry_rows].to(device), gaze_mean, gaze_std)

            fast = _inner_adapt(model, adapter, list(adapter.parameters()),
                                f_sup, y_sup, args.inner_lr, args.inner_steps)
            fast_leaf = [w.detach().requires_grad_(True) for w in fast]
            pred_q = model.readout(adapter.func(f_qry, fast_leaf))
            loss_q = F.smooth_l1_loss(pred_q, y_qry) / len(tasks)
            loss_q.backward()  # head grads accumulate; fast_leaf.grad set
            for acc, fl in zip(adapter_grad, fast_leaf):
                acc += fl.grad

        # head grads were accumulated by backward; set adapter-init grads (FOMAML)
        for p, g in zip(adapter.parameters(), adapter_grad):
            p.grad = g.clone()
        meta_opt.step()

        if it % args.print_freq == 0 or it == args.meta_iters:
            log(f"Fold {fold} meta_iter {it}/{args.meta_iters} "
                f"query_loss={loss_q.item() * len(tasks):.6f}")

    pre_err, post_err = _meta_eval(args, model, adapter, va_feats, va_gazes, va_recs,
                                   gaze_mean, gaze_std, device)
    log(f"Fold {fold} done meta_pre_adapt_error={pre_err:.6f} "
        f"meta_post_adapt_error={post_err:.6f} improvement={pre_err - post_err:.6f}")

    _save_meta_checkpoint(args, model, adapter, gaze_mean, gaze_std, split, dim)
    return {"fold": fold, "pre_error": pre_err, "post_error": post_err}


def _meta_eval(args, model, adapter, feats, gazes, recs, gaze_mean, gaze_std, device):
    """Per held-out recording: adapt on K frames, report pre/post coord error (cm)."""
    model.eval()
    rows_by_rec = _rec_to_rows(recs)
    K = args.meta_support
    steps = getattr(args, "adapt_steps", None) or args.inner_steps
    rng = random.Random(args.seed)
    pre_errors, post_errors = [], []

    for rec, rows in rows_by_rec.items():
        if len(rows) <= K:
            continue
        rows = rows[:]
        rng.shuffle(rows)
        sup_rows, qry_rows = rows[:K], rows[K:]
        f_sup = feats[sup_rows].to(device)
        y_sup = normalize_gaze(gazes[sup_rows].to(device), gaze_mean, gaze_std)
        f_qry = feats[qry_rows].to(device)
        g_qry = gazes[qry_rows].to(device)

        # pre-adaptation: adapter at meta-learned init (identity-ish)
        with torch.no_grad():
            pred_pre = denormalize_gaze(
                model.readout(adapter(f_qry)).float(), gaze_mean, gaze_std)
            pre_errors.append(torch.linalg.norm(pred_pre - g_qry, dim=1).mean().item())

        fast = _inner_adapt(model, adapter, list(adapter.parameters()),
                            f_sup, y_sup, args.inner_lr, steps)
        with torch.no_grad():
            pred_post = denormalize_gaze(
                model.readout(adapter.func(f_qry, fast)).float(), gaze_mean, gaze_std)
            post_errors.append(torch.linalg.norm(pred_post - g_qry, dim=1).mean().item())

    pre = sum(pre_errors) / max(1, len(pre_errors))
    post = sum(post_errors) / max(1, len(post_errors))
    return pre, post


def _save_meta_checkpoint(args, model, adapter, gaze_mean, gaze_std, split, dim):
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fold = split["fold"]
    checkpoint = {
        "model": model.state_dict(),
        "adapter": adapter.state_dict(),
        "adapter_kind": args.adapter,
        "adapter_dim": dim,
        "lora_rank": getattr(args, "lora_rank", 8),
        "lora_alpha": getattr(args, "lora_alpha", 8.0),
        "gaze_mean": gaze_mean.cpu(),
        "gaze_std": gaze_std.cpu(),
        "args": vars(args),
        "input_mode": args.input_mode,
        "fold": fold,
        "train_recordings": split["train_recordings"],
        "val_recordings": split["val_recordings"],
        "inner_lr": args.inner_lr,
        "inner_steps": args.inner_steps,
    }
    path = out_path / f"fold{fold}_meta_{args.adapter}_{args.backbone}_gaze.pth"
    torch.save(checkpoint, path)
    log(f"Fold {fold} saved meta checkpoint {path}")
