import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data

from . import accel
from .dataset import (
    AugmentedSubset,
    build_dataset,
    build_multistream_dataset,
    make_augment_transform,
)
from .models import (
    batch_images_for_mode,
    batch_multistream_for_mode,
    create_model,
    forward_for_mode,
    forward_multistream,
)
from .splits import recording_kfolds, select_splits

print_freq = 100


class _Logger:
    """Routes every training/optimization log line to stdout and, when
    ``--log-file`` is set, also appends a timestamped copy to that file.

    The file is line-buffered so it can be followed live with ``tail -f`` and
    survives a crash. Stdout output is left exactly as before, so Slurm parsing
    and live monitoring keep working.
    """

    def __init__(self):
        self._fh = None

    def open(self, path):
        if path:
            self._fh = open(path, "a", buffering=1)
            self._fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === training log started ===\n")

    def __call__(self, msg=""):
        print(msg)
        if self._fh is not None:
            self._fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


log = _Logger()


def normalize_gaze(gaze, mean, std):
    return (gaze - mean) / std


def denormalize_gaze(gaze, mean, std):
    return gaze * std + mean


def make_loader(dataset, indices, batch_size, shuffle, num_workers, augment_transform=None):
    subset = data.Subset(dataset, indices)
    if augment_transform is not None:
        subset = AugmentedSubset(subset, augment_transform)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        # Keep workers and their prefetch buffers alive between epochs so a fast
        # GPU is not starved waiting on PIL decode/resize. With 32 GB RAM the
        # extra prefetch buffers are cheap.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return data.DataLoader(subset, **loader_kwargs)


def train(args):
    log.open(getattr(args, "log_file", None))
    try:
        _run_training(args)
    finally:
        log.close()


def _run_training(args):
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))
    if args.input_mode == "multistream":
        dataset = build_multistream_dataset(args)
    else:
        use_synthetic = args.input_mode in ("synthetic", "paired")
        require_synthetic = (
            args.input_mode in ("synthetic", "paired") and not args.allow_missing_synthetic
        )
        dataset = build_dataset(
            args, use_synthetic=use_synthetic, require_synthetic=require_synthetic
        )

    all_splits = recording_kfolds(dataset.unique_recordings(), folds=args.folds, seed=args.seed)
    splits = select_splits(all_splits, args.fold_index)

    log(f"Device: {device}")
    log(f"Cross validation: {args.folds} folds by recording id")
    backbone_str = (f" backbone={args.backbone}"
                    if args.input_mode == "multistream" and hasattr(args, "backbone")
                    else "")
    log(f"Model: input_mode={args.input_mode}{backbone_str} "
        f"weights={args.weights} freeze_encoder={args.freeze_encoder}")
    if args.fold_index is not None:
        log(f"Running only fold {args.fold_index}")

    summaries = []
    for split in splits:
        summary = train_one_fold(args, dataset, split, device)
        summaries.append(summary)

    total_time = time.perf_counter() - start_time
    if summaries:
        mean_val_loss = sum(item["best_val_loss"] for item in summaries) / len(summaries)
        mean_val_error = sum(item["best_val_error"] for item in summaries) / len(summaries)
        log(
            f"CV summary folds={len(summaries)} "
            f"mean_best_val_loss={mean_val_loss:.6f} "
            f"mean_best_val_coord_error={mean_val_error:.6f} "
            f"total_time_sec={total_time:.2f}"
        )


def train_one_fold(args, dataset, split, device):
    fold_start = time.perf_counter()
    fold = split["fold"]
    train_indices = dataset.indices_for_recordings(split["train_recordings"])
    val_indices = dataset.indices_for_recordings(split["val_recordings"])
    if not train_indices or not val_indices:
        raise RuntimeError(f"Fold {fold} has empty train or validation indices.")

    log(
        f"Fold {fold} start "
        f"train_recordings={split['train_recordings']} "
        f"val_recordings={split['val_recordings']} "
        f"train_samples={len(train_indices)} "
        f"val_samples={len(val_indices)}"
    )

    gaze_mean = torch.from_numpy(dataset.gazes[train_indices].mean(axis=0)).float()
    gaze_std = torch.from_numpy(dataset.gazes[train_indices].std(axis=0)).float().clamp_min(1e-6)

    model = create_model(
        args.input_mode,
        weights=args.weights,
        freeze_encoder=args.freeze_encoder,
        use_grid=getattr(args, "use_grid", False),
        grid_size=getattr(args, "grid_size", 25),
        backbone=getattr(args, "backbone", "vit"),
    ).to(device)
    if getattr(args, "compile", False):
        model = _maybe_compile(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_enabled, amp_dtype = accel.resolve_amp(device, getattr(args, "amp", False))
    scaler = accel.make_grad_scaler(amp_enabled, amp_dtype)
    log(f"Fold {fold} accel {accel.describe(device, amp_enabled, amp_dtype)}")
    scheduler = _make_scheduler(optimizer, args)
    if scheduler is not None:
        log(f"Fold {fold} lr_scheduler={getattr(args, 'lr_scheduler', 'none')}")

    augment = getattr(args, "augment", "none")
    aug_transform = None
    if args.input_mode == "multistream" and augment and augment != "none":
        aug_transform = make_augment_transform(augment, args.image_size)
        log(f"Fold {fold} augment={augment}")

    train_loader = make_loader(dataset, train_indices, args.batch_size, True, args.num_workers,
                               augment_transform=aug_transform)
    val_loader = make_loader(dataset, val_indices, args.batch_size, False, args.num_workers)

    gaze_mean_device = gaze_mean.to(device)
    gaze_std_device = gaze_std.to(device)
    best_val_loss = float("inf")
    best_val_error = float("inf")
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Early-stopping bookkeeping. patience=None disables it; the monitored
    # metric also drives "best" checkpoint selection so the two are consistent.
    # min_delta sets the minimum improvement that counts as progress.
    patience = getattr(args, "patience", None)
    monitor = getattr(args, "early_stop_metric", "val_error")
    min_delta = float(getattr(args, "min_delta", 0.0))
    if monitor not in ("val_loss", "val_error"):
        raise ValueError(f"--early-stop-metric must be val_loss or val_error, got {monitor}")
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gaze_mean=gaze_mean_device,
            gaze_std=gaze_std_device,
            device=device,
            input_mode=args.input_mode,
            fold=fold,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_loss, val_error = evaluate(
            model=model,
            loader=val_loader,
            gaze_mean=gaze_mean_device,
            gaze_std=gaze_std_device,
            device=device,
            input_mode=args.input_mode,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"Fold {fold} epoch {epoch + 1}/{args.epochs} validation "
            f"val_loss={val_loss:.6f} "
            f"val_coord_error={val_error:.6f} "
            f"train_loss_mean={train_loss:.6f} "
            f"lr={current_lr:.2e} "
            f"epoch_time_sec={epoch_time:.2f}"
        )
        if scheduler is not None:
            scheduler.step()

        checkpoint = {
            "model": _unwrap(model).state_dict(),
            "gaze_mean": gaze_mean,
            "gaze_std": gaze_std,
            "args": vars(args),
            "input_mode": args.input_mode,
            "fold": fold,
            "train_recordings": split["train_recordings"],
            "val_recordings": split["val_recordings"],
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_error": val_error,
        }
        torch.save(checkpoint, out_path / f"fold{fold}_last_{args.backbone}_gaze_segmenter.pth")

        current = val_loss if monitor == "val_loss" else val_error
        best_so_far = best_val_loss if monitor == "val_loss" else best_val_error
        if current < best_so_far - min_delta:
            best_val_loss = val_loss
            best_val_error = val_error
            epochs_since_improve = 0
            torch.save(checkpoint, out_path / f"fold{fold}_best_{args.backbone}_gaze_segmenter.pth")
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience:
                log(
                    f"Fold {fold} early stop at epoch {epoch + 1}/{args.epochs}: "
                    f"{monitor} did not improve by >= {min_delta} for "
                    f"{patience} epochs (best {monitor}={best_so_far:.6f})"
                )
                break

    fold_time = time.perf_counter() - fold_start
    log(
        f"Fold {fold} done "
        f"best_val_loss={best_val_loss:.6f} "
        f"best_val_coord_error={best_val_error:.6f} "
        f"fold_time_sec={fold_time:.2f}"
    )
    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "best_val_error": best_val_error,
        "fold_time_sec": fold_time,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    gaze_mean,
    gaze_std,
    device,
    input_mode,
    fold,
    epoch,
    total_epochs,
    scaler=None,
    amp_enabled=False,
    amp_dtype=None,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with accel.autocast(device, amp_enabled, amp_dtype):
            pred, batch_size = _predict(model, batch, input_mode, device)
            target = normalize_gaze(batch["gaze"].to(device), gaze_mean, gaze_std)
            loss = F.smooth_l1_loss(pred, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_count += batch_size
        running_loss = total_loss / max(1, total_count)
        batch_time = time.perf_counter() - batch_start
        if (batch_idx % print_freq) == 0:
            log(
                f"Fold {fold} epoch {epoch + 1}/{total_epochs} "
                f"batch {batch_idx}/{total_batches} "
                f"batch_loss={loss.item():.6f} "
                f"running_train_loss={running_loss:.6f} "
                f"batch_time_sec={batch_time:.2f}"
            )
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, gaze_mean, gaze_std, device, input_mode, amp_enabled=False, amp_dtype=None):
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_count = 0
    for batch in loader:
        with accel.autocast(device, amp_enabled, amp_dtype):
            pred_norm, batch_size = _predict(model, batch, input_mode, device)
        pred_norm = pred_norm.float()
        gaze = batch["gaze"].to(device)
        target = normalize_gaze(gaze, gaze_mean, gaze_std)

        pred = denormalize_gaze(pred_norm, gaze_mean, gaze_std)
        loss = F.smooth_l1_loss(pred_norm, target, reduction="sum")
        error = torch.linalg.norm(pred - gaze, dim=1).sum()

        total_loss += loss.item()
        total_error += error.item()
        total_count += batch_size

    return total_loss / max(1, total_count), total_error / max(1, total_count)


def _unwrap(model):
    """Return the underlying module, unwrapping a torch.compile wrapper if present.

    Keeps saved checkpoints loadable by the plain (uncompiled) model used at
    inference time.
    """
    return getattr(model, "_orig_mod", model)


def _maybe_compile(model):
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        log("torch.compile unavailable; running eagerly.")
        return model

    # On clustered filesystems the default Triton/Inductor caches land in
    # $HOME/.triton and $HOME/.cache, which routinely blows the home quota and
    # crashes the first compiled step (OSError: Disk quota exceeded). Point them
    # at node-local scratch ($TMPDIR) unless the user has already chosen a dir.
    cache_base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_base, "triton_cache"))
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.join(cache_base, "torchinductor_cache")
    )
    # Best effort: a runtime compile failure (e.g. unsupported op, cache I/O)
    # should degrade to eager rather than kill a multi-hour job, matching the
    # documented behaviour of --compile.
    try:
        from torch import _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:
        pass

    try:
        # dynamic=True avoids guard recompiles on the ragged final batch and on
        # the stacked (3*B) multistream encoder input.
        compiled = compile_fn(model, dynamic=True)
        log(
            f"torch.compile enabled (dynamic=True); Triton cache at "
            f"{os.environ['TRITON_CACHE_DIR']}. The first step compiles and "
            f"will be slower, then speeds up."
        )
        return compiled
    except Exception as exc:  # pragma: no cover - backend/hardware dependent
        log(f"torch.compile failed ({exc}); running eagerly.")
        return model


def _make_scheduler(optimizer, args):
    name = getattr(args, "lr_scheduler", "none")
    if not name or name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.0
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(args, "step_size", 3),
            gamma=getattr(args, "step_gamma", 0.5),
        )
    raise ValueError(f"Unknown --lr-scheduler: {name!r}")


def _predict(model, batch, input_mode, device):
    """Dispatch a batch through the right forward path; return (pred, batch_size)."""
    if input_mode == "multistream":
        inputs = batch_multistream_for_mode(batch, device)
        pred = forward_multistream(model, inputs)
        return pred, inputs["face"].size(0)
    first, second = batch_images_for_mode(batch, input_mode, device)
    pred = forward_for_mode(model, input_mode, first, second)
    return pred, first.size(0)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    input_mode = checkpoint.get("input_mode", saved_args.get("input_mode", "paired"))
    model = create_model(
        input_mode=input_mode,
        weights="none",
        freeze_encoder=bool(saved_args.get("freeze_encoder", False)),
        use_grid=bool(saved_args.get("use_grid", False)),
        grid_size=int(saved_args.get("grid_size", 25)),
        backbone=str(saved_args.get("backbone", "vit")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    gaze_mean = checkpoint["gaze_mean"].to(device)
    gaze_std = checkpoint["gaze_std"].to(device)
    return model, gaze_mean, gaze_std, checkpoint, input_mode
