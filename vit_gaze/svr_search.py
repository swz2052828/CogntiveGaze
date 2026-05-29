"""Swarm-style global hyperparameter search for the per-subject SVR baseline.

Inspired by PhoneRealTimeGazeEstimation (Zhu et al., SwarmIntelligentCalibration),
which tunes one global (C, gamma, epsilon) triple for an RBF-SVR calibrator by
running MVO/JAYA/PSO across multiple subjects -- the swarm sits *upstream* of
the per-subject SVR fit and is run once to produce a population-best triple
that is reused for every subject at validation time.

We use the same fitness design (mean Euclidean error of per-subject SVR fits
over support/query draws) with a small dependency-free PSO. The output is one
triple per fold; paste it into ``metacompare`` as ``--svr-C/--svr-eps/--svr-gamma``
so the SVR baseline is tuned rather than sklearn-default.

Note: Zhu et al. fit SVR on CNN *embeddings* (so the SVR is the readout). Our
``SVRCalibrator`` -- and this tuner -- fit SVR on the base model's *predicted
xy* (so the SVR is a per-subject correction on top of an already-trained
readout). The hyperparameter scales transfer because both are RBF-SVR with
the same (C, gamma, epsilon) family, but the tuned values are calibrated to
the prediction-space mapping our metacompare uses, not theirs.

Protocol note: HP search uses the **training** subjects of the current fold so
the held-out subjects are never seen by the tuner, matching the same CV
protocol metacompare evaluates on.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from . import accel
from .dataset import build_multistream_dataset
from .models import batch_multistream_for_mode, create_model
from .splits import recording_kfolds, select_splits
from .training import denormalize_gaze, log

# Search bounds match Zhu et al. (Swarm Intelligent Calibration):
#   C       in [0.1, 1000]
#   gamma   in [0.001, 10]
#   epsilon in [0.01, 0.1]
DEFAULT_LB = np.array([0.1, 0.001, 0.01])
DEFAULT_UB = np.array([1000.0, 10.0, 0.1])


def pso(fitness, lb, ub, pop=30, iters=50, w=0.7, c1=1.5, c2=1.5, seed=0,
        progress=None):
    """Minimal Particle Swarm Optimization. Minimizes ``fitness(x: ndarray)``.

    Returns ``(best_x, best_f, history)`` where history is per-iter best_f.
    Standard inertia + cognitive + social update with bound clamping.
    """
    rng = np.random.default_rng(seed)
    dim = len(lb)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    x = lb + rng.random((pop, dim)) * (ub - lb)
    v = (rng.random((pop, dim)) - 0.5) * (ub - lb) * 0.1
    f = np.array([float(fitness(xi)) for xi in x])
    pbest = x.copy()
    pbest_f = f.copy()
    g_idx = int(np.argmin(pbest_f))
    gbest = pbest[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    history = [gbest_f]
    if progress is not None:
        progress(0, gbest_f, gbest)
    for it in range(1, iters + 1):
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lb, ub)
        f = np.array([float(fitness(xi)) for xi in x])
        better = f < pbest_f
        pbest[better] = x[better]
        pbest_f[better] = f[better]
        g_idx = int(np.argmin(pbest_f))
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest = pbest[g_idx].copy()
        history.append(gbest_f)
        if progress is not None:
            progress(it, gbest_f, gbest)
    return gbest, gbest_f, history


@torch.no_grad()
def _cache_base_preds(model, dataset, indices, gaze_mean, gaze_std,
                      device, batch_size, num_workers):
    """Return (preds[N,2], gazes[N,2], recs[N]) for `indices`, all CPU."""
    loader = data.DataLoader(
        data.Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    preds, gazes, recs = [], [], []
    for batch in loader:
        inputs = batch_multistream_for_mode(batch, device)
        f = model.forward_features(
            inputs["face"], inputs["eye_left"], inputs["eye_right"], inputs.get("grid"))
        p = denormalize_gaze(model.readout(f).float(), gaze_mean, gaze_std)
        preds.append(p.cpu())
        gazes.append(batch["gaze"])
        recs.append(batch["rec"])
    return torch.cat(preds).numpy(), torch.cat(gazes).numpy(), torch.cat(recs).numpy()


def _make_fitness(preds_by_rec, gazes_by_rec, k, trials, seed):
    """Mean Euclidean error across recordings x trials for candidate (C, gamma, epsilon)."""
    from sklearn.svm import SVR
    rng = random.Random(seed)
    # Precompute support/query draws so every candidate triple sees identical draws
    # (fair comparison; eliminates per-call sampling variance).
    draws_by_rec = {}
    for rec, preds in preds_by_rec.items():
        n = len(preds)
        if n <= k:
            continue
        rows = list(range(n))
        draws = []
        for _ in range(trials):
            shuffled = rows[:]
            rng.shuffle(shuffled)
            draws.append((shuffled[:k], shuffled[k:]))
        draws_by_rec[rec] = draws

    def fitness(params):
        C, gamma, epsilon = float(params[0]), float(params[1]), float(params[2])
        errs = []
        for rec, draws in draws_by_rec.items():
            preds = preds_by_rec[rec]
            gts = gazes_by_rec[rec]
            for sup, qry in draws:
                svr_x = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(preds[sup], gts[sup, 0])
                svr_y = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(preds[sup], gts[sup, 1])
                px = svr_x.predict(preds[qry])
                py = svr_y.predict(preds[qry])
                errs.append(float(np.mean(np.sqrt((px - gts[qry, 0]) ** 2 +
                                                  (py - gts[qry, 1]) ** 2))))
        return float(np.mean(errs)) if errs else float("inf")

    return fitness


def _load_base_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    saved = ckpt.get("args", {})
    model = create_model(
        input_mode="multistream", weights="none", freeze_encoder=False,
        use_grid=bool(saved.get("use_grid", False)),
        grid_size=int(saved.get("grid_size", 25)),
        backbone=str(saved.get("backbone", "vit")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["gaze_mean"].to(device), ckpt["gaze_std"].to(device)


def svrsearch(args):
    log.open(getattr(args, "log_file", None))
    try:
        _run_svrsearch(args)
    finally:
        log.close()


def _run_svrsearch(args):
    if args.input_mode != "multistream":
        raise ValueError("svrsearch is multistream-only.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))

    dataset = build_multistream_dataset(args)
    splits_all = recording_kfolds(dataset.unique_recordings(), folds=args.folds, seed=args.seed)
    splits = select_splits(splits_all, args.fold_index)
    model, gaze_mean, gaze_std = _load_base_checkpoint(args.base_checkpoint, device)

    log(f"Device: {device}")
    log(f"svrsearch K={args.k} trials={args.trials} pop={args.pop} iters={args.iters} "
        f"bounds=[{list(DEFAULT_LB)}, {list(DEFAULT_UB)}]")

    out = {}
    for split in splits:
        fold = split["fold"]
        # HP search uses TRAINING-fold subjects only so held-out subjects are never
        # seen by the tuner. Per-fold output: paste into metacompare for the same fold.
        train_idx = dataset.indices_for_recordings(split["train_recordings"])
        log(f"Fold {fold} caching base preds on {len(split['train_recordings'])} train subjects")
        preds, gazes, recs = _cache_base_preds(
            model, dataset, train_idx, gaze_mean, gaze_std,
            device, args.batch_size, args.num_workers)
        preds_by_rec, gazes_by_rec = {}, {}
        for r in np.unique(recs):
            mask = recs == r
            preds_by_rec[int(r)] = preds[mask]
            gazes_by_rec[int(r)] = gazes[mask]

        fitness = _make_fitness(preds_by_rec, gazes_by_rec,
                                k=args.k, trials=args.trials, seed=args.seed + fold)

        def progress(it, best_f, best_x):
            if it == 0 or it == args.iters or it % max(1, args.iters // 10) == 0:
                log(f"Fold {fold} pso iter {it}/{args.iters} best_err={best_f:.4f} "
                    f"C={best_x[0]:.4f} gamma={best_x[1]:.6f} epsilon={best_x[2]:.4f}")

        best_x, best_f, _ = pso(
            fitness, DEFAULT_LB, DEFAULT_UB,
            pop=args.pop, iters=args.iters, seed=args.seed + fold, progress=progress)
        log(f"Fold {fold} done best_err={best_f:.4f} "
            f"C={best_x[0]:.6f} gamma={best_x[1]:.6f} epsilon={best_x[2]:.6f}")
        out[fold] = {"C": float(best_x[0]), "gamma": float(best_x[1]),
                     "epsilon": float(best_x[2]), "best_err": float(best_f)}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        log(f"Wrote tuned SVR hyperparameters to {args.json_out}")
    # Convenient one-liner the user can paste into a metacompare command:
    if len(out) == 1:
        ((fold, hp),) = out.items()
        log(f"Fold {fold} paste: --svr-C {hp['C']:.4f} --svr-gamma {hp['gamma']:.6f} "
            f"--svr-eps {hp['epsilon']:.4f}")
