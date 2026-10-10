"""The attacks. Each returns a plain dict; nothing here decides pass/fail.

Threat models
  A2  release only: the attacker holds nothing but the release. Verification
      (first-half templates vs second-half probes) and linkage (unsupervised
      partition of the release). For a public release, A2 is the operative one.
  A1  enrolment: the attacker also holds photographs of the participants.

Every interval is a participant-level cluster bootstrap: frames of one person are
not independent. A duplicated participant keeps its own label (a duplicate is the
same person; giving it a new label scores every duplicate as an error and pushes
the interval below its own point estimate -- the guard below refuses that).
"""
import numpy as np

from . import data


# --------------------------------------------------------------- embedding
def embed_halves(root, stream, rec, sample, align_detect=False, loader=None):
    """X1,y1 (first half) and X2,y2 (second half). `loader(pid, name)` overrides
    image loading (used for the floor patch). Detection failures under
    align_detect are counted, never imputed: on the release they are a real
    effect, but they are reported beside the number."""
    import cv2
    X1, y1, X2, y2, fail = [], [], [], [], 0
    for pid, (a, b) in sample.items():
        for names, X, y in ((a, X1, y1), (b, X2, y2)):
            for n in names:
                img = loader(pid, n) if loader else data.load(root, pid, stream, n)
                if img is None:
                    fail += 1
                    continue
                if align_detect:
                    e = rec.embed_detected(img)
                    if e is None:
                        fail += 1
                        continue
                else:
                    e = rec.embed(cv2.resize(img, (112, 112)))
                X.append(e)
                y.append(pid)
    return (np.asarray(X1), np.asarray(y1), np.asarray(X2), np.asarray(y2), fail)


# ------------------------------------------------------------ verification
def tar_at_far(scores, labels, fars=(1e-1, 1e-2, 1e-3)):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    gen, imp = scores[labels == 1], scores[labels == 0]
    return {f"TAR@FAR={f:g}": (float((gen >= np.quantile(imp, 1.0 - f)).mean())
                               if len(imp) and len(gen) else float("nan"))
            for f in fars}


def _scores(X1, y1, X2, y2, subs):
    T = np.stack([X1[y1 == s].mean(0) for s in subs])
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    S = X2 @ T.T
    lab = (np.asarray(subs)[None, :] == y2[:, None]).astype(int)
    return S, lab


def verification(X1, y1, X2, y2, n_boot=1000, seed=0):
    """Zero-effort template matching: mean first-half embedding per participant,
    cosine against second-half probes. Rank-1 is reported but is NOT evidence on
    single-session data (it rides session nuisance); TAR at FAR 1e-3 is primary."""
    from sklearn.metrics import roc_auc_score
    subs = sorted(set(y1.tolist()) & set(y2.tolist()))
    keep1, keep2 = np.isin(y1, subs), np.isin(y2, subs)
    X1, y1, X2, y2 = X1[keep1], y1[keep1], X2[keep2], y2[keep2]
    S, lab = _scores(X1, y1, X2, y2, subs)
    gen, imp = S[lab == 1], S[lab == 0]
    res = {"n_participants": len(subs), "n_probes": int(len(X2)),
           "rank1": float((np.asarray(subs)[S.argmax(1)] == y2).mean()),
           "chance_rank1": 1.0 / len(subs),
           "auc": float(roc_auc_score(lab.ravel(), S.ravel())),
           "dprime": float((gen.mean() - imp.mean()) /
                           np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))}
    res.update(tar_at_far(S.ravel(), lab.ravel()))

    # participant bootstrap of TAR@FAR=1e-3: rows/columns of drawn participants,
    # genuine iff same ORIGINAL participant (duplicates stay the same person)
    rng = np.random.default_rng(seed)
    pid_of_probe = np.array([subs.index(v) for v in y2])
    rows = {i: np.flatnonzero(pid_of_probe == i) for i in range(len(subs))}
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(subs), len(subs))
        r = np.concatenate([rows[i] for i in drawn])
        Sb = S[np.ix_(r, drawn)]
        Lb = (pid_of_probe[r][:, None] == drawn[None, :]).astype(int)
        vals.append(tar_at_far(Sb.ravel(), Lb.ravel(), fars=(1e-3,))["TAR@FAR=0.001"])
    vals = np.asarray(vals)
    # A resample holds ~63% of the participants, so it has fewer impostor scores,
    # and an extreme quantile (the 99.9th, for FAR 1e-3) of a smaller set is biased
    # low -> the threshold drops -> bootstrap TAR is biased HIGH. A plain
    # percentile interval then sits above its own point estimate. Use the basic
    # (bias-reflecting) interval 2*theta - q, and refuse it if it still excludes
    # the point.
    pt = res["TAR@FAR=0.001"]
    q_lo, q_hi = np.percentile(vals, [2.5, 97.5])
    lo, hi = float(np.clip(2 * pt - q_hi, 0, 1)), float(np.clip(2 * pt - q_lo, 0, 1))
    res["TAR@FAR=0.001_boot_bias"] = float(vals.mean() - pt)
    res["TAR@FAR=0.001_ci95"] = [lo, hi] if lo <= pt <= hi else None
    res["TAR_ci_method"] = "basic bootstrap (bias-reflecting), participant level"
    return res


# ----------------------------------------------------------------- linkage
def _kmeans_ari(X, y, k, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    return float(adjusted_rand_score(y, lab)), lab


def _purity(y, lab):
    return float(sum(np.bincount(y[lab == c]).max() for c in np.unique(lab)) / len(y))


def auto_k(X, y, k_max=30):
    """The attacker does not know the cohort size: agglomerative (cosine, average)
    for k = 2..k_max, k chosen by silhouette."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    best = (-2.0, None, None)
    for k in range(2, min(k_max, len(X) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="cosine")
        if s > best[0]:
            best = (s, k, lab)
    if best[1] is None:
        return {"k_selected": None}
    return {"k_selected": int(best[1]), "silhouette": float(best[0]),
            "ari_auto_k": float(adjusted_rand_score(y, best[2])),
            "purity_auto_k": _purity(y, best[2])}


def linkage(X1, y1, X2, y2, n_boot=1000, n_perm=200, k_max=30, seed=0):
    """Can the release be partitioned into individuals with no reference at all?"""
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    X = np.vstack([X1, X2]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    lab = np.concatenate([y1, y2])
    subs = sorted(set(lab.tolist()))
    y = np.array([subs.index(v) for v in lab])
    n = len(subs)

    point, km = _kmeans_ari(X, y, n, seed)
    res = {"n_participants": n, "n_frames": int(len(X)),
           "ari_known_k": point, "purity_known_k": _purity(y, km)}

    # label-permutation null for the known-k partition (statistical chance only;
    # it does NOT control session nuisance -- that is what the floor is for)
    rng = np.random.default_rng(seed)
    null = [adjusted_rand_score(rng.permutation(y), km) for _ in range(n_perm)]
    res["perm_null_ari_max"] = float(np.max(null))

    rows = {i: np.flatnonzero(y == i) for i in range(n)}
    boot = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        # rng.choice over participant indices, as in the study's ari_bootstrap.py,
        # so the published intervals are reproduced draw for draw
        drawn = rng.choice(np.arange(n), size=n, replace=True)
        idx = np.concatenate([rows[i] for i in drawn])
        lb = np.concatenate([np.full(len(rows[i]), i) for i in drawn])
        boot.append(_kmeans_ari(X[idx], lb, len(set(drawn.tolist())), seed)[0])
    b = np.asarray(boot)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    res["ari_known_k_ci95"] = [lo, hi]
    res["frac_boot_above_point"] = float((b > point).mean())
    res["bootstrap_valid"] = bool(lo <= point <= hi)
    if not res["bootstrap_valid"]:
        res["ari_known_k_ci95"] = None   # never report an interval that excludes its point

    res.update(auto_k(X, y, k_max))
    S = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    same = (y[iu[0]] == y[iu[1]])
    sim = S[iu]
    gen, imp = sim[same], sim[~same]
    res["pair_auc"] = float(roc_auc_score(same.astype(int), sim))
    res["pair_dprime"] = float((gen.mean() - imp.mean()) /
                               np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))
    return res


# ------------------------------------------------------- non-pixel channels
def photometric_features(img):
    import cv2
    f = []
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        f += [ch.mean(), ch.std(), np.percentile(ch, 5), np.percentile(ch, 50),
              np.percentile(ch, 95)]
    b, g, r = [img[:, :, i].astype(np.float32).mean() + 1e-6 for i in range(3)]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f += [r / g, b / g, r / b, grey.mean(), grey.std(),
          float(np.percentile(grey, 99) - np.percentile(grey, 1))]
    return np.asarray(f, dtype=np.float64)


def photometric(root, stream, n_per_participant=120, k_max=30, seed=0):
    """21 summary statistics per crop, no structure, no recogniser. If these alone
    partition the release, the release is linkable without any face model."""
    sample = data.flat_sample(root, stream, n_per_participant, seed)
    X, lab = [], []
    for pid, names in sample.items():
        for n in names:
            img = data.load(root, pid, stream, n)
            if img is not None:
                X.append(photometric_features(img))
                lab.append(pid)
    X = np.asarray(X)
    subs = sorted(set(lab))
    y = np.array([subs.index(v) for v in lab])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    point, km = _kmeans_ari(X, y, len(subs), seed)
    res = {"n_participants": len(subs), "n_crops": int(len(X)),
           "ari_known_k": point, "purity_known_k": _purity(y, km)}
    res.update(auto_k(X, y, k_max))
    return res


def geometry(root, streams, sample):
    """Crop dimensions as an identifier. Templates: the modal (w, h) tuple across
    streams over a participant's first-half frames; probes: each second-half
    frame's tuple. Ties are broken uniformly, so the score is an EXPECTED Rank-1.
    Also reports participant-level distinctness and the identifying information
    in bits (entropy of the grouping of participants by template)."""
    from collections import Counter
    from PIL import Image

    def tup(pid, name):
        try:
            return tuple(Image.open(f"{root}/{pid}/{s}/{name}").size for s in streams)
        except (FileNotFoundError, OSError):
            return None

    tmpl = {}
    for pid, (a, _) in sample.items():
        ts = [t for t in (tup(pid, n) for n in a) if t]
        if ts:
            tmpl[pid] = Counter(ts).most_common(1)[0][0]
    pids = sorted(tmpl)
    hits = []
    for pid in pids:
        for n in sample[pid][1]:
            t = tup(pid, n)
            if t is None:
                continue
            exact = [q for q in pids if tmpl[q] == t]
            if not exact:     # nearest template by L1 over all dimensions
                d = {q: sum(abs(x - z) for u, v in zip(tmpl[q], t) for x, z in zip(u, v))
                     for q in pids}
                m = min(d.values())
                exact = [q for q in pids if d[q] == m]
            hits.append(1.0 / len(exact) if pid in exact else 0.0)
    groups = Counter(tmpl[p] for p in pids)
    n = len(pids)
    bits = float(sum(g / n * np.log2(n / g) for g in groups.values()))
    return {"n_participants": n, "expected_rank1": float(np.mean(hits)) if hits else None,
            "chance_rank1": 1.0 / n, "distinct_templates": len(groups),
            "identifying_bits": bits, "max_bits": float(np.log2(n)),
            "constant_within_participant": bool(all(
                len({tup(p, x) for x in sample[p][0][:10]} - {None}) == 1 for p in pids))}


def null_control(release, original, stream, sample, max_frames=60):
    """Does the release actually differ from the original? A transform that was
    not applied scores as 'leaky' for the wrong reason, or -- worse -- a contrast
    between two arms is computed between identical images."""
    d, skipped = [], 0
    for pid, (_, b) in sample.items():
        for n in b[:max_frames // max(1, len(sample)) + 1]:
            r, o = data.load(release, pid, stream, n), data.load(original, pid, stream, n)
            if r is None or o is None or r.shape != o.shape:
                skipped += 1
                continue
            d.append(np.abs(r.astype(np.int16) - o.astype(np.int16)).mean())
    return {"mean_abs_diff": float(np.mean(d)) if d else None, "n_pairs": len(d),
            "n_skipped": skipped}
