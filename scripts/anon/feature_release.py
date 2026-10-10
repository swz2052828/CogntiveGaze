"""Item 3: release eye-geometry FEATURES instead of pixels. Does it land anywhere
near the private-and-useful corner?

Features come from FaceMesh landmarks already extracted for 1,200 frames per
participant spread over the whole recording (datasets/anon_eye_landmarks), plus the
face box from the stored geometry. Three tiers, each a strict superset:

  T1 iris     per eye: iris centre in the eye's own corner-to-corner frame,
              (u along the corner axis, v across it), normalised by eye width -> 4
  T2 +head    T1 + face-box centre and size in the frame (head position)       -> 7
  T3 +shape   T2 + both 16-point eyelid contours in the same corner frame        -> 71

Utility: participant-wise 5-fold CV with the study's split (recording_kfolds, seed
42); gradient-boosted trees per coordinate; no per-participant calibration, so it
compares with Table 3's uncalibrated `base`. Reported beside the predict-the-mean
error on the same frames.

Privacy: exactly the pixel attacks' statistics (privaudit.attacks): first-half
templates vs second-half probes for verification, KMeans partition for linkage,
participant-level intervals; features standardised then L2-normalised. Two attacker
granularities: single frames, and 10-frame windows averaged (the release is
temporally ordered, so an attacker can pool neighbours).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path("/springbrook/share/eng/esrpxk")
sys.path.insert(0, str(ROOT / "CogntiveGaze"))
from privaudit import attacks                      # noqa: E402
from vit_gaze.splits import recording_kfolds       # noqa: E402

LM = ROOT / "datasets" / "anon_eye_landmarks"
GEO = ROOT / "datasets" / "anon_geometry"
OUT = ROOT / "results" / "feature_release"


def corner_frame(contour, pts):
    """Express `pts` in the eye's frame: origin at corner 0, x along corner0->corner8,
    unit = eye width. contour (N,16,2), pts (N,K,2)."""
    c0, c8 = contour[:, 0, :], contour[:, 8, :]
    ax = c8 - c0
    w = np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ex = ax / w
    ey = np.stack([-ex[:, 1], ex[:, 0]], 1)
    rel = pts - c0[:, None, :]
    return np.stack([(rel * ex[:, None, :]).sum(-1), (rel * ey[:, None, :]).sum(-1)], -1) / w[:, None, :]


def load():
    md = sio.loadmat(ROOT / "datasets/ProcessedData/meanno7_clean/metadata.mat", squeeze_me=True)
    lab = {(int(r), int(f)): (float(x), float(y)) for r, f, x, y in
           zip(md["labelRecNum"], md["frameIndex"], md["labelDotXCam"], md["labelDotYCam"])}
    rows = []
    for r in range(6, 24):
        z = np.load(LM / f"{r:05d}.npz", allow_pickle=True)
        g = np.load(GEO / f"{r:05d}.npz", allow_pickle=True)
        gi = {str(n): i for i, n in enumerate(g["frames"])}
        fr = [str(n) for n in z["frames"]]
        # Privacy uses every landmark frame (no labels needed, all 18 participants, as
        # for the pixel attacks); utility uses the labelled ones (participant 00007
        # has no gaze ground truth and is outside every meanno7 manifest).
        idx = np.array([i for i, n in enumerate(fr) if n in gi], dtype=int)
        feats = {}
        for e in ("appleLeftEye", "appleRightEye"):
            con = z[f"{e}_contour"][idx].astype(np.float64)
            # contour and iris are in EYE-CROP coordinates; map to frame coordinates
            box = np.stack([g[e][gi[fr[i]]][:2] for i in idx]).astype(np.float64)
            con_f = con + box[:, None, :]
            cen_f = z[f"{e}_centre"][idx].astype(np.float64) + box
            feats[f"{e}_iris"] = corner_frame(con_f, cen_f[:, None, :])[:, 0, :]
            feats[f"{e}_shape"] = corner_frame(con_f, con_f).reshape(len(idx), -1)
        fb = np.stack([g["appleFace"][gi[fr[i]]][:4] for i in idx]).astype(np.float64)
        head = np.stack([fb[:, 0] + fb[:, 2] / 2, fb[:, 1] + fb[:, 3] / 2, fb[:, 2]], 1)
        y = np.array([lab.get((r, int(fr[i].split(".")[0])), (np.nan, np.nan)) for i in idx])
        order = np.argsort([int(fr[i].split(".")[0]) for i in idx])   # temporal order
        T1 = np.hstack([feats["appleLeftEye_iris"], feats["appleRightEye_iris"]])
        T2 = np.hstack([T1, head])
        T3 = np.hstack([T2, feats["appleLeftEye_shape"], feats["appleRightEye_shape"]])
        rows.append(dict(rec=r, T1=T1[order], T2=T2[order], T3=T3[order], y=y[order]))
    return rows


def utility(rows, tier):
    from sklearn.ensemble import HistGradientBoostingRegressor
    ok = {d["rec"]: ~np.isnan(d["y"]).any(1) for d in rows}
    rows = [d for d in rows if ok[d["rec"]].sum() > 0]
    recs = np.array([d["rec"] for d in rows])
    X = {d["rec"]: d[tier][ok[d["rec"]]] for d in rows}
    Y = {d["rec"]: d["y"][ok[d["rec"]]] for d in rows}
    errs, mp = [], []
    for s in recording_kfolds(recs, 5, 42):
        Xtr = np.vstack([X[r] for r in s["train_recordings"]]); ytr = np.vstack([Y[r] for r in s["train_recordings"]])
        Xte = np.vstack([X[r] for r in s["val_recordings"]]); yte = np.vstack([Y[r] for r in s["val_recordings"]])
        pred = np.stack([HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                         random_state=0).fit(Xtr, ytr[:, k]).predict(Xte) for k in range(2)], 1)
        errs.append(float(np.linalg.norm(pred - yte, axis=1).mean()))
        mp.append(float(np.linalg.norm(ytr.mean(0) - yte, axis=1).mean()))
    return dict(n_participants=len(rows), n_frames=int(sum(len(v) for v in Y.values())),
                error_by_fold=errs, error=float(np.mean(errs)),
                mean_predictor_by_fold=mp, mean_predictor=float(np.mean(mp)))


def privacy(rows, tier, n_per_half=60, window=1, n_boot=1000):
    rng = np.random.default_rng(0)
    X1, y1, X2, y2 = [], [], [], []
    allX = np.vstack([d[tier] for d in rows])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    for d in rows:
        F = (d[tier] - mu) / sd
        if window > 1:   # pool consecutive released frames
            n = len(F) // window
            F = F[: n * window].reshape(n, window, -1).mean(1)
        half = len(F) // 2
        k = min(n_per_half, half)
        for part, XX, yy in ((F[:half], X1, y1), (F[half:], X2, y2)):
            pick = rng.choice(len(part), size=k, replace=False)
            XX.append(part[pick]); yy += [f"{d['rec']:05d}"] * k
    X1, X2 = np.vstack(X1), np.vstack(X2)
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True) + 1e-12
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12
    y1, y2 = np.array(y1), np.array(y2)
    v = attacks.verification(X1, y1, X2, y2, n_boot=2000)
    lk = attacks.linkage(X1, y1, X2, y2, n_boot=n_boot)
    return dict(n_per_half=k, window=window,
                TAR=v["TAR@FAR=0.001"], TAR_ci95=v["TAR@FAR=0.001_ci95"], rank1=v["rank1"],
                ARI=lk["ari_known_k"], ARI_ci95=lk["ari_known_k_ci95"],
                k_selected=lk.get("k_selected"), ARI_auto=lk.get("ari_auto_k"),
                pair_auc=lk["pair_auc"], perm_null_ari_max=lk["perm_null_ari_max"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    print(f"{sum(len(d['y']) for d in rows)} landmark frames, {len(rows)} participants; "
          f"{sum(int((~np.isnan(d['y']).any(1)).sum()) for d in rows)} labelled", flush=True)
    res = {}
    for tier in ("T1", "T2", "T3"):
        u = utility(rows, tier)
        p1 = privacy(rows, tier, window=1)
        p10 = privacy(rows, tier, window=10, n_per_half=30)
        res[tier] = dict(dims=int(rows[0][tier].shape[1]), utility=u, privacy_frame=p1,
                         privacy_window10=p10)
        print(f"{tier} ({res[tier]['dims']}d): error {u['error']:.2f} cm (mean-pred {u['mean_predictor']:.2f}) | "
              f"frame TAR {p1['TAR']:.3f} {p1['TAR_ci95']} ARI {p1['ARI']:.3f} {p1['ARI_ci95']} | "
              f"10-frame TAR {p10['TAR']:.3f} ARI {p10['ARI']:.3f} {p10['ARI_ci95']}", flush=True)
        json.dump(res, open(OUT / "feature_release.json", "w"), indent=2)
    print(f"wrote {OUT / 'feature_release.json'}")


if __name__ == "__main__":
    main()
