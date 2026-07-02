"""Measure the NATIVE pixel span of the eye region in the raw OriginalData frames
(1080x750) via MediaPipe FaceMesh. Decides opt #4: if the native eye box is
substantially larger than the 120x120 ProcessedData crops, a high-res recrop adds
real iris detail; if it's ~120px, the existing crops are already near-lossless.

Samples N frames per recording. facedet env (mediapipe, numpy<2). CPU/SLURM.
"""
import argparse
import glob
import os
import random

import cv2
import mediapipe as mp
import numpy as np

# FaceMesh landmark index sets around each eye (incl. brow-to-cheek margin used by
# GazeCapture-style crops). Iris indices 468-477 exist with refine_landmarks=True.
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]


def eye_box_px(landmarks, idxs, w, h, margin=1.7):
    xs = np.array([landmarks[i].x for i in idxs]) * w
    ys = np.array([landmarks[i].y for i in idxs]) * h
    cx, cy = xs.mean(), ys.mean()
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) * margin  # square, GC-style margin
    return span, cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/springbrook/share/eng/esrpxk/datasets/OriginalData")
    ap.add_argument("--per-rec", type=int, default=30)
    ap.add_argument("--out", default="/springbrook/share/eng/esrpxk/results/native_eye_span.md")
    args = ap.parse_args()
    random.seed(42)

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5)

    spans, face_spans = {}, {}
    for rec_dir in sorted(glob.glob(os.path.join(args.root, "0*"))):
        rec = os.path.basename(rec_dir)
        frames = sorted(glob.glob(os.path.join(rec_dir, "*.jpg")))
        if not frames:
            continue
        pick = random.sample(frames, min(args.per_rec, len(frames)))
        s, fs = [], []
        for fp in pick:
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            l, _, _ = eye_box_px(lm, LEFT_EYE, w, h)
            r, _, _ = eye_box_px(lm, RIGHT_EYE, w, h)
            s += [l, r]
            xs = np.array([p.x for p in lm]) * w
            ys = np.array([p.y for p in lm]) * h
            fs.append(max(xs.max() - xs.min(), ys.max() - ys.min()))
        if s:
            spans[rec] = (float(np.median(s)), float(np.percentile(s, 90)), len(s))
            face_spans[rec] = float(np.median(fs))
            print(f"{rec}: eye median {spans[rec][0]:.0f}px p90 {spans[rec][1]:.0f}px "
                  f"face {face_spans[rec]:.0f}px (n={len(s)})", flush=True)

    allmed = [v[0] for v in spans.values()]
    L = ["# Native eye-region span in raw OriginalData (1080x750), px",
         "", "ProcessedData eye crops are 120x120 at source (upscaled to 224 in the loader).",
         "Native span = square eye box (GazeCapture-style 1.7 margin) from FaceMesh.", "",
         "| rec | eye median px | eye p90 px | face median px | n |", "|---|---|---|---|---|"]
    for rec in sorted(spans):
        m, p90, n = spans[rec]
        L.append(f"| {rec} | {m:.0f} | {p90:.0f} | {face_spans[rec]:.0f} | {n} |")
    L += ["", f"**Cohort: eye median {np.median(allmed):.0f}px, min {min(allmed):.0f}, max {max(allmed):.0f}.**",
          "", f"Verdict: {'RECROP WORTH IT (native >> 120px)' if np.median(allmed) > 150 else 'existing 120px crops are near-native; recrop adds little'}"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
