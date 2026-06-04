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

Two spaces are supported (``--space``):

* ``prediction`` -- fit SVR on the base model's *predicted xy* (so the SVR is a
  per-subject correction on top of an already-trained readout). This is what our
  ``SVRCalibrator`` and metacompare's ``svr`` baseline use.
* ``embedding`` -- fit SVR on the compact penultimate readout activation (the
  128-d ``calibration_feature``), so the SVR *replaces* the final linear readout.
  This is the faithful analogue of Zhu et al.'s recipe: they fit SVR on the
  256-d ``gaze_feature`` bottleneck that feeds their final Linear(.,2), NOT on
  the raw backbone output. Earlier revisions mistakenly used the 2432-d
  ``forward_features`` vector here, which is ~19x wider than their bottleneck and
  made each RBF fit/predict dominate the whole pipeline; the 128-d activation
  restores both fidelity and speed.

The tuned (C, gamma, epsilon) are written per fold; ``svrsearch --space embedding``
output feeds metacompare's ``--svr-embed-*`` (which evaluates on the same 128-d
feature), and ``--space prediction`` output feeds its ``--svr-*``.

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
from .dataset import build_multistream_dataset_maybe_video
from .models import vivit_kwargs_from_args, batch_multistream_for_mode, create_model
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


def jaya(fitness, lb, ub, pop=30, iters=50, seed=0, progress=None):
    """JAYA (Rao, 2016): parameter-less optimizer. Each candidate moves toward
    the best and away from the worst solution; the move is accepted only if it
    improves. Minimizes ``fitness(x)``. Returns ``(best_x, best_f, history)``.
    """
    rng = np.random.default_rng(seed)
    dim = len(lb)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    x = lb + rng.random((pop, dim)) * (ub - lb)
    f = np.array([float(fitness(xi)) for xi in x])
    best_i = int(np.argmin(f))
    best, best_f = x[best_i].copy(), float(f[best_i])
    history = [best_f]
    if progress is not None:
        progress(0, best_f, best)
    for it in range(1, iters + 1):
        worst = x[int(np.argmax(f))]
        cur_best = x[int(np.argmin(f))]
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))
        cand = x + r1 * (cur_best - np.abs(x)) - r2 * (worst - np.abs(x))
        cand = np.clip(cand, lb, ub)
        cand_f = np.array([float(fitness(xi)) for xi in cand])
        improved = cand_f < f
        x[improved] = cand[improved]
        f[improved] = cand_f[improved]
        bi = int(np.argmin(f))
        if f[bi] < best_f:
            best_f, best = float(f[bi]), x[bi].copy()
        history.append(best_f)
        if progress is not None:
            progress(it, best_f, best)
    return best, best_f, history


def _roulette(weights, rng):
    """Roulette-wheel index selection proportional to non-negative ``weights``."""
    total = weights.sum()
    if total <= 0:
        return int(rng.integers(len(weights)))
    r = rng.random() * total
    idx = int(np.searchsorted(np.cumsum(weights), r))
    return min(idx, len(weights) - 1)


def mvo(fitness, lb, ub, pop=30, iters=50, seed=0,
        wep_min=0.2, wep_max=1.0, p=6.0, progress=None):
    """Multi-Verse Optimizer (Mirjalili et al., 2016). Universes exchange
    objects via white/black holes (roulette by inflation rate = fitness) and
    teleport toward the best universe via wormholes whose existence probability
    (WEP) rises and travelling-distance rate (TDR) shrinks over iterations.
    Minimizes ``fitness(x)``. Returns ``(best_x, best_f, history)``.
    """
    rng = np.random.default_rng(seed)
    dim = len(lb)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    univ = lb + rng.random((pop, dim)) * (ub - lb)
    f = np.array([float(fitness(xi)) for xi in univ])
    best_i = int(np.argmin(f))
    best, best_f = univ[best_i].copy(), float(f[best_i])
    history = [best_f]
    if progress is not None:
        progress(0, best_f, best)
    for it in range(1, iters + 1):
        wep = wep_min + it * ((wep_max - wep_min) / iters)
        tdr = 1.0 - (it ** (1.0 / p)) / (iters ** (1.0 / p))

        order = np.argsort(f)                       # ascending: best first
        sorted_univ = univ[order]
        sorted_f = f[order]
        # Normalized inflation in [0,1]; higher = worse universe.
        rng_f = sorted_f.max() - sorted_f.min()
        norm_inf = (sorted_f - sorted_f.min()) / (rng_f + 1e-12)
        # White-hole source weights: better (lower-fitness) universes more likely.
        wh_weights = (1.0 - norm_inf) + 1e-6

        new = univ.copy()
        for i in range(pop):
            for j in range(dim):
                if rng.random() < norm_inf[i]:
                    # black hole receives an object from a roulette-selected
                    # (good) universe's white hole.
                    src = _roulette(wh_weights, rng)
                    new[i, j] = sorted_univ[src, j]
                if rng.random() < wep:
                    span = (ub[j] - lb[j]) * rng.random() + lb[j]
                    if rng.random() < 0.5:
                        new[i, j] = best[j] + tdr * span
                    else:
                        new[i, j] = best[j] - tdr * span
        univ = np.clip(new, lb, ub)
        f = np.array([float(fitness(xi)) for xi in univ])
        bi = int(np.argmin(f))
        if f[bi] < best_f:
            best_f, best = float(f[bi]), univ[bi].copy()
        history.append(best_f)
        if progress is not None:
            progress(it, best_f, best)
    return best, best_f, history


def optimize(method, fitness, lb, ub, pop, iters, seed, progress=None):
    """Dispatch to the requested swarm optimizer (Zhu et al. use MVO/JAYA/PSO)."""
    method = method.lower()
    if method == "pso":
        return pso(fitness, lb, ub, pop=pop, iters=iters, seed=seed, progress=progress)
    if method == "jaya":
        return jaya(fitness, lb, ub, pop=pop, iters=iters, seed=seed, progress=progress)
    if method == "mvo":
        return mvo(fitness, lb, ub, pop=pop, iters=iters, seed=seed, progress=progress)
    raise ValueError(f"Unknown optimizer {method!r}; choose pso / mvo / jaya.")


@torch.no_grad()
def _cache(model, dataset, indices, gaze_mean, gaze_std, device,
           batch_size, num_workers, want_features):
    """Cache (X, gazes, recs) for ``indices``.

    ``X`` is either the model's predicted xy (prediction space) or the compact
    penultimate readout activation (embedding space), depending on
    ``want_features``. The embedding is the 128-d ``calibration_feature`` -- the
    analogue of Zhu et al.'s 256-d ``gaze_feature`` bottleneck -- NOT the wide
    2432-d ``forward_features`` vector, so the SVR replaces only the final linear
    readout and each RBF fit/predict stays cheap.
    """
    loader = data.DataLoader(
        data.Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    xs, gazes, recs = [], [], []
    for batch in loader:
        inputs = batch_multistream_for_mode(batch, device)
        f = model.forward_features(
            inputs["face"], inputs["eye_left"], inputs["eye_right"], inputs.get("grid"))
        if want_features:
            xs.append(model.calibration_feature(f).float().cpu())
        else:
            p = denormalize_gaze(model.readout(f).float(), gaze_mean, gaze_std)
            xs.append(p.cpu())
        gazes.append(batch["gaze"])
        recs.append(batch["rec"])
    return torch.cat(xs).numpy(), torch.cat(gazes).numpy(), torch.cat(recs).numpy()


def _make_fitness(X_by_rec, gazes_by_rec, k, trials, seed):
    """Mean Euclidean error across recordings x trials for candidate (C, gamma, epsilon).

    ``X_by_rec[rec]`` is the SVR input for that recording -- predicted xy in
    prediction-space tuning, fused features in embedding-space tuning. The
    target is always 2D screen coordinates from ``gazes_by_rec``.
    """
    from sklearn.svm import SVR
    rng = random.Random(seed)
    # Precompute support/query draws so every candidate triple sees identical draws
    # (fair comparison; eliminates per-call sampling variance).
    draws_by_rec = {}
    for rec, X in X_by_rec.items():
        n = len(X)
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
            X = X_by_rec[rec]
            gts = gazes_by_rec[rec]
            for sup, qry in draws:
                svr_x = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(X[sup], gts[sup, 0])
                svr_y = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(X[sup], gts[sup, 1])
                px = svr_x.predict(X[qry])
                py = svr_y.predict(X[qry])
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
        **vivit_kwargs_from_args(saved),
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

    dataset = build_multistream_dataset_maybe_video(args)
    splits_all = recording_kfolds(dataset.unique_recordings(), folds=args.folds, seed=args.seed)
    splits = select_splits(splits_all, args.fold_index)
    model, gaze_mean, gaze_std = _load_base_checkpoint(args.base_checkpoint, device)

    want_features = (getattr(args, "space", "prediction") == "embedding")
    optimizer_name = getattr(args, "optimizer", "pso").lower()
    log(f"Device: {device}")
    log(f"svrsearch optimizer={optimizer_name} "
        f"space={'embedding' if want_features else 'prediction'} "
        f"K={args.k} trials={args.trials} pop={args.pop} iters={args.iters} "
        f"bounds=[{list(DEFAULT_LB)}, {list(DEFAULT_UB)}]")

    out = {}
    for split in splits:
        fold = split["fold"]
        # HP search uses TRAINING-fold subjects only so held-out subjects are never
        # seen by the tuner. Per-fold output: paste into metacompare for the same fold.
        train_idx = dataset.indices_for_recordings(split["train_recordings"])
        log(f"Fold {fold} caching on {len(split['train_recordings'])} train subjects "
            f"(space={'embedding' if want_features else 'prediction'})")
        X, gazes, recs = _cache(
            model, dataset, train_idx, gaze_mean, gaze_std,
            device, args.batch_size, args.num_workers, want_features=want_features)
        X_by_rec, gazes_by_rec = {}, {}
        for r in np.unique(recs):
            mask = recs == r
            X_by_rec[int(r)] = X[mask]
            gazes_by_rec[int(r)] = gazes[mask]

        fitness = _make_fitness(X_by_rec, gazes_by_rec,
                                k=args.k, trials=args.trials, seed=args.seed + fold)

        def progress(it, best_f, best_x):
            if it == 0 or it == args.iters or it % max(1, args.iters // 10) == 0:
                log(f"Fold {fold} {optimizer_name} iter {it}/{args.iters} "
                    f"best_err={best_f:.4f} C={best_x[0]:.4f} "
                    f"gamma={best_x[1]:.6f} epsilon={best_x[2]:.4f}")

        best_x, best_f, _ = optimize(
            optimizer_name, fitness, DEFAULT_LB, DEFAULT_UB,
            pop=args.pop, iters=args.iters, seed=args.seed + fold, progress=progress)
        log(f"Fold {fold} done optimizer={optimizer_name} best_err={best_f:.4f} "
            f"C={best_x[0]:.6f} gamma={best_x[1]:.6f} epsilon={best_x[2]:.6f}")
        out[fold] = {"C": float(best_x[0]), "gamma": float(best_x[1]),
                     "epsilon": float(best_x[2]), "best_err": float(best_f),
                     "space": "embedding" if want_features else "prediction",
                     "optimizer": optimizer_name}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        log(f"Wrote tuned SVR hyperparameters to {args.json_out}")
    # Convenient one-liner the user can paste into a metacompare command:
    if len(out) == 1:
        ((fold, hp),) = out.items()
        prefix = "--svr-embed" if want_features else "--svr"
        log(f"Fold {fold} paste: {prefix}-C {hp['C']:.4f} {prefix}-gamma {hp['gamma']:.6f} "
            f"{prefix}-eps {hp['epsilon']:.4f}")
