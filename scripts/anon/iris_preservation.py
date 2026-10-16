"""Model-free test: does a de-identification method keep the PUPIL where it was?

For the same frames in every arm, FaceMesh (refine_landmarks) is run on the
released face crop, and each iris centre is expressed in its eye's own corner
frame: u along the corner-to-corner axis, v across it, in units of eye width.
That coordinate is what gaze direction moves; head pose and crop placement do
not. Compared per frame with the ORIGINAL crop of the same frame:

  det        fraction of frames where FaceMesh finds the face (both arms)
  mae_u/v    mean |u_arm - u_orig|, |v_arm - v_orig| (eye widths)
  r_u/r_v    within-participant Pearson r(u_orig, u_arm), averaged over participants
             -- does the pupil follow the original as the person looks around?
  gt_r_x/y   within-participant r(u_arm, gaze x) and r(v_arm, gaze y) against the
             EyeLink ground truth (labelled frames), the same for the original

Frames: the 260-per-participant privacy sample (the frames every FAMS strength in
the sweep was generated on), so every arm is scored on identical frames.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

D = Path("/springbrook/share/eng/esrpxk/datasets")
SAMPLE = D / "ProcessedFAMS_per_frame"
EYES = {"a": (33, 133, 468), "b": (362, 263, 473)}   # corner, corner, iris centre


def uv(P, c0, c1, ic):
    a, b, p = P[c0], P[c1], P[ic]
    ax = b - a; w = np.linalg.norm(ax) + 1e-9; ex = ax / w; ey = np.array([-ex[1], ex[0]])
    r = p - a
    return np.array([r @ ex, r @ ey]) / w


def one(job):
    root, rec, names = job
    import mediapipe as mp
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                           refine_landmarks=True, min_detection_confidence=0.3)
    out = {}
    for n in names:
        im = cv2.imread(str(D / root / rec / "appleFace" / n))
        if im is None:
            continue
        res = mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            out[n] = None; continue
        h, w = im.shape[:2]
        P = np.array([[q.x * w, q.y * h] for q in res.multi_face_landmarks[0].landmark])
        out[n] = np.concatenate([uv(P, *EYES["a"]), uv(P, *EYES["b"])])
    mesh.close()
    return root, rec, out


def labels():
    import scipy.io as sio
    md = sio.loadmat(D / "ProcessedData/meanno7_clean/metadata.mat", squeeze_me=True)
    return {(int(r), int(f)): (float(x), float(y)) for r, f, x, y in
            zip(md["labelRecNum"], md["frameIndex"], md["labelDotXCam"], md["labelDotYCam"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="dataset roots; ProcessedData first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    recs = sorted(p.name for p in SAMPLE.iterdir() if p.is_dir())
    names = {rec: sorted(p.name for p in (SAMPLE / rec / "appleFace").glob("*.jpg")) for rec in recs}
    # identical frames in every arm
    for root in args.arms:
        for rec in recs:
            have = {p.name for p in (D / root / rec / "appleFace").glob("*.jpg")}
            names[rec] = [n for n in names[rec] if n in have]
    jobs = [(root, rec, names[rec]) for root in args.arms for rec in recs]
    with Pool(args.procs) as pool:
        res = pool.map(one, jobs)
    U = {}
    for root, rec, out in res:
        U.setdefault(root, {})[rec] = out
    lab = labels()
    orig = args.arms[0]
    summary = {}
    for root in args.arms:
        det, du, dv, ru, rv, gx, gy = [], [], [], [], [], [], []
        for rec in recs:
            o, a = U[orig][rec], U[root][rec]
            both = [n for n in names[rec] if o.get(n) is not None and a.get(n) is not None]
            det.append(len(both) / max(1, len(names[rec])))
            if len(both) < 20:
                continue
            Xo = np.array([o[n] for n in both]); Xa = np.array([a[n] for n in both])
            du.append(np.abs(Xa[:, [0, 2]] - Xo[:, [0, 2]]).mean())
            dv.append(np.abs(Xa[:, [1, 3]] - Xo[:, [1, 3]]).mean())
            ru.append(np.mean([np.corrcoef(Xo[:, k], Xa[:, k])[0, 1] for k in (0, 2)]))
            rv.append(np.mean([np.corrcoef(Xo[:, k], Xa[:, k])[0, 1] for k in (1, 3)]))
            g = [(i, lab[(int(rec), int(n[:-4]))]) for i, n in enumerate(both) if (int(rec), int(n[:-4])) in lab]
            if len(g) >= 20:
                ii = [i for i, _ in g]; G = np.array([v for _, v in g])
                gx.append(np.mean([abs(np.corrcoef(Xa[ii, k], G[:, 0])[0, 1]) for k in (0, 2)]))
                gy.append(np.mean([abs(np.corrcoef(Xa[ii, k], G[:, 1])[0, 1]) for k in (1, 3)]))
        summary[root] = dict(det=float(np.mean(det)), mae_u=float(np.mean(du)), mae_v=float(np.mean(dv)),
                             r_u=float(np.mean(ru)), r_v=float(np.mean(rv)),
                             gt_r_x=float(np.mean(gx)) if gx else None, gt_r_y=float(np.mean(gy)) if gy else None,
                             r_u_by_rec=[float(x) for x in ru], gt_r_x_by_rec=[float(x) for x in gx],
                             n_recs=len(ru), n_frames=int(sum(len(v) for v in names.values())))
        s = summary[root]
        print(f"{root:34s} det {s['det']:.3f}  |du| {s['mae_u']:.3f} |dv| {s['mae_v']:.3f}  "
              f"r_u {s['r_u']:.3f} r_v {s['r_v']:.3f}  gt|r| x {s['gt_r_x']} y {s['gt_r_y']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    np.savez_compressed(Path(args.out).with_suffix(".npz"),
                        **{f"{root}__{rec}": np.array([[int(n[:-4])] + (list(U[root][rec][n]) if U[root][rec].get(n) is not None else [np.nan] * 4)
                                                       for n in names[rec]]) for root in args.arms for rec in recs})


if __name__ == "__main__":
    main()
