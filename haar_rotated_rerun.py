"""Re-run Haar after de-rotating frames. The calib footage is rotated, so the
Viola-Jones frontal cascade (tolerates only ~+/-15 deg roll) fails. Here we test
the four fixed orientations and measure, per orientation:

  - FaceMesh eye-line angle  (upright orientation -> angle near 0 deg)
  - Haar selected-box recall  (vs FaceMesh GT in the SAME rotated frame)
  - Haar background-FP rate

FaceMesh is rotation-robust, so it gives valid ground truth in every orientation;
the eye-line angle tells us which orientation is truly upright. If Haar recall
jumps from ~25% (original) to high on the upright orientation, rotation is the
confirmed root cause and the fix is to rotate frames before detection.

Runs on a SLURM compute node (never the login node).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

import mediapipe  # noqa
import mediapipe.python.solutions as _sol  # noqa
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
sys.modules.setdefault("mediapipe.solutions.face_detection", _sol.face_detection)

import cv2  # noqa: E402
from video_preprocess.detector import VideoFaceDetector  # noqa: E402
from video_preprocess.bbox import bbox_from_points  # noqa: E402

ROOT = "/springbrook/share/eng/esrpxk/datasets"
VIDEOS = f"{ROOT}/videos"
SUB_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB = dict(zip(SUB_IDS,
    [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
     1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]))

ORIENTS = {
    "identity": None,
    "rot90_cw": cv2.ROTATE_90_CLOCKWISE,
    "rot180": cv2.ROTATE_180,
    "rot90_ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def classify(cand, gt):
    cx = (cand[0] + cand[2]) / 2.0
    cy = (cand[1] + cand[3]) / 2.0
    if not (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]):
        return "background"
    return "on_face" if (cand[2] - cand[0]) >= 0.30 * (gt[2] - gt[0]) else "partial"


def eyeline_angle(lm):
    """Absolute eye-line angle from horizontal, degrees (0 = upright)."""
    lc = lm.left_eye.mean(axis=0)
    rc = lm.right_eye.mean(axis=0)
    ang = np.degrees(np.arctan2(rc[1] - lc[1], rc[0] - lc[0]))
    # fold to [0,90]: distance from horizontal (0 or 180)
    a = abs(ang) % 180
    return min(a, 180 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--out", default=f"{ROOT}/face_detector_comparison/haar_rotated")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # one FaceMesh session per orientation (tracking mode is per-stream)
    acc = {o: dict(gt=0, sel_hit=0, sel_bg=0, ncand=0, angle=[], fm=0, frames=0)
           for o in ORIENTS}

    t0 = time.perf_counter()
    for sub in SUB_IDS:
        begin = CALIB[sub]
        # buffer the sampled frames for this subject so each orientation's
        # FaceMesh tracking sees a clean sequential stream
        cap = cv2.VideoCapture(f"{VIDEOS}/subid_{sub}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
        frames = []
        read = 0
        while read < 1200:
            ok, fr = cap.read()
            if not ok:
                break
            if read % args.stride == 0:
                frames.append(fr)
            read += 1
        cap.release()

        for oname, ocode in ORIENTS.items():
            mesh = VideoFaceDetector()
            for f in frames:
                img = f if ocode is None else cv2.rotate(f, ocode)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lm = mesh.detect(rgb)
                a = acc[oname]
                a["frames"] += 1
                if lm is None:
                    continue
                a["fm"] += 1
                a["angle"].append(eyeline_angle(lm))
                gt = bbox_from_points(lm.face_oval)
                a["gt"] += 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = HAAR.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(60, 60))
                cands = [(float(x), float(y), float(x + w), float(y + h))
                         for (x, y, w, h) in rects]
                a["ncand"] += len(cands)
                if cands:
                    largest = max(cands, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    if classify(largest, gt) in ("on_face", "partial"):
                        a["sel_hit"] += 1
                    else:
                        a["sel_bg"] += 1
            mesh.close()
        print(f"[sub {sub}] done  elapsed {time.perf_counter()-t0:.0f}s", flush=True)

    rows = []
    for oname in ORIENTS:
        a = acc[oname]
        gt = max(1, a["gt"])
        rows.append(dict(
            orientation=oname,
            facemesh_recall=a["fm"] / max(1, a["frames"]),
            mean_eyeline_deg=float(np.mean(a["angle"])) if a["angle"] else float("nan"),
            haar_sel_recall=a["sel_hit"] / gt,
            haar_bg_fp=a["sel_bg"] / gt,
            haar_mean_cand=a["ncand"] / gt,
            n_gt=a["gt"],
        ))
    with open(f"{args.out}/orientation_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"{'orientation':12}{'fm_recall':>10}{'eyeline_deg':>13}{'haar_recall':>13}{'haar_bgFP':>11}{'cand':>7}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: -r["haar_sel_recall"]):
        print(f"{r['orientation']:12}{r['facemesh_recall']*100:>9.1f}%{r['mean_eyeline_deg']:>12.1f}"
              f"{r['haar_sel_recall']*100:>12.1f}%{r['haar_bg_fp']*100:>10.1f}%{r['haar_mean_cand']:>7.1f}")
    print("=" * 78)
    print(f"Wrote {args.out}/orientation_sweep.csv")


if __name__ == "__main__":
    main()
