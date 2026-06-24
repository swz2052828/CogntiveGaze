"""Find an iris-based blink metric that actually collapses on closure.

The current IrisVisibilityBlinkDetector uses max iris radius / inter-canthi,
which barely dips (refine_landmarks fits a full circle even when occluded).
Here we re-run FaceMesh on a few subjects and, per frame, compute several
candidate iris metrics + the EAR label, then report each candidate's best-F1
threshold treating EAR<0.2 as ground truth.

Iris landmark order in FaceMesh refine: [center, right, top, left, bottom].
Runs on a SLURM compute node.
"""
import sys, csv, os
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.blink import _ear

SUBS = [11, 18, 23, 6, 13]            # mix of strong blinkers + sub13
CALIB = {6: 3750, 11: 6300, 13: 2400, 18: 11250, 23: 2250}
ROT = cv2.ROTATE_90_CLOCKWISE


def metrics(lm):
    out = {}
    lo, ro = lm.left_eye[0], lm.right_eye[8]
    span = float(np.linalg.norm(ro - lo)) or 1.0
    for side, iris, eye_ear in (("l", lm.left_iris, lm.left_eye_ear),
                                ("r", lm.right_iris, lm.right_eye_ear)):
        c = iris.mean(axis=0)
        max_r = float(np.linalg.norm(iris - c, axis=1).max())
        vert = float(iris[:, 1].max() - iris[:, 1].min())      # top-bottom
        horiz = float(iris[:, 0].max() - iris[:, 0].min())     # left-right
        out[f"max_radius_{side}"] = max_r / span
        out[f"vert_extent_{side}"] = vert / span
        out[f"vh_ratio_{side}"] = vert / horiz if horiz > 1e-6 else 0.0
    out["ear"] = min(_ear(lm.left_eye_ear), _ear(lm.right_eye_ear))
    return out


def main():
    rows = []
    for sub in SUBS:
        cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, CALIB[sub])
        mesh = VideoFaceDetector()
        r = 0
        while r < 1200:
            ok, f = cap.read()
            if not ok:
                break
            r += 1
            lm = mesh.detect(cv2.cvtColor(cv2.rotate(f, ROT), cv2.COLOR_BGR2RGB))
            if lm is None:
                continue
            m = metrics(lm)
            rows.append(m)
        mesh.close(); cap.release()
        print(f"[sub {sub}] {r} frames", flush=True)

    labels = [m["ear"] < 0.2 for m in rows]
    P = sum(labels); N = len(labels) - P
    print(f"\nframes={len(rows)} ear-blink={P} open={N}\n")

    def best_f1(metric_min, lower_is_blink=True):
        vals = sorted(set(round(metric_min[i], 4) for i in range(len(rows))))
        best = (None, 0, 0, 0)
        for thr in vals:
            if lower_is_blink:
                pred = [v < thr for v in metric_min]
            else:
                pred = [v > thr for v in metric_min]
            tp = sum(1 for p, l in zip(pred, labels) if p and l)
            fp = sum(1 for p, l in zip(pred, labels) if p and not l)
            fn = P - tp
            prec = tp/(tp+fp) if tp+fp else 0
            rec = tp/P if P else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
            if f1 > best[1]:
                best = (thr, f1, prec, rec)
        return best

    for name in ("max_radius", "vert_extent", "vh_ratio"):
        mn = [min(m[f"{name}_l"], m[f"{name}_r"]) for m in rows]
        thr, f1, prec, rec = best_f1(mn)
        print(f"{name:12} best-F1 thr={thr:.4f}  F1={f1:.3f} prec={prec:.2f} rec={rec:.2f}")


if __name__ == "__main__":
    main()
