"""Validate the template-matching blink detector on subjects with templates.

Checks whether our (upright, native-res) eye crops are at the same scale as the
pupil templates: if so, peak correlation is high on open eyes and drops on
EAR-flagged blinks. Reports correlation stats + best-F1 threshold vs EAR.
Runs on a SLURM compute node.
"""
import sys, csv
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points
from video_preprocess.eye_detectors import build_eye_detector
from video_preprocess.blink_detectors import build_blink_detector
from video_preprocess.blink import _ear

CALIB = {6: 3750, 7: 13170, 11: 6300, 13: 2400, 12: 1500}
TEMPLATES = "/springbrook/share/eng/esrpxk/datasets/eye_templates"
ROT = cv2.ROTATE_90_CLOCKWISE


def run(sub):
    cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, CALIB[sub])
    mesh = VideoFaceDetector()
    eye = build_eye_detector("facemesh_contour")
    tm = build_blink_detector("template_match", templates_dir=TEMPLATES, subject_id=sub)
    print(f"[sub {sub}] {len(tm.templates)} templates "
          f"(sizes {[t.shape[0] for t in tm.templates]})", flush=True)
    rows = []
    r = 0
    while r < 1200:
        ok, f = cap.read()
        if not ok:
            break
        r += 1
        rgb = cv2.cvtColor(cv2.rotate(f, ROT), cv2.COLOR_BGR2RGB)
        lm = mesh.detect(rgb)
        if lm is None:
            continue
        boxes = eye.detect(rgb, bbox_from_points(lm.face_oval), lm)
        if boxes is None:
            continue
        L, R = boxes
        _, sl, sr = tm.detect(rgb, L, R, lm)
        ear = min(_ear(lm.left_eye_ear), _ear(lm.right_eye_ear))
        rows.append((ear < 0.2, sl, sr))
    mesh.close(); eye.close(); cap.release()

    score = [min(s for s in (sl, sr) if s == s) for _, sl, sr in rows]
    blink = [b for b, _, _ in rows]
    op = [s for s, b in zip(score, blink) if not b]
    bl = [s for s, b in zip(score, blink) if b]
    print(f"[sub {sub}] frames={len(rows)} ear-blink={sum(blink)}")
    print(f"  template peak-corr OPEN : mean={np.mean(op):.3f} p25={np.percentile(op,25):.3f} min={min(op):.3f}")
    if bl:
        print(f"  template peak-corr BLINK: mean={np.mean(bl):.3f} p75={np.percentile(bl,75):.3f} max={max(bl):.3f}")
    # best-F1 threshold (lower corr => blink)
    P = sum(blink); best = (None, 0, 0, 0)
    for thr in np.arange(0.1, 0.9, 0.02):
        tp = sum(1 for s, b in zip(score, blink) if s < thr and b)
        fp = sum(1 for s, b in zip(score, blink) if s < thr and not b)
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/P if P else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        if f1 > best[1]:
            best = (round(thr, 2), f1, prec, rec)
    print(f"  best-F1 thr={best[0]} F1={best[1]:.3f} prec={best[2]:.2f} rec={best[3]:.2f}\n", flush=True)


if __name__ == "__main__":
    for sub in [6, 7, 11, 13]:
        run(sub)
