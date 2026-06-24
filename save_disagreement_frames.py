"""Render the EAR-vs-GT blink disagreement frames for one subject as labeled
eye crops, so they can be eyeballed.

Reads datasets/ear_vs_gt/subid_<id>_disagreements.csv and, for each event,
renders the middle frame (upright, FaceMesh eye region):
  disagreement_frames/subid_<id>/FP/f<frame>.jpg   EAR said blink, no GT
  disagreement_frames/subid_<id>/FN/f<frame>.jpg   GT blink, EAR missed
"""
import argparse, csv, sys, os
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points

ROT = cv2.ROTATE_90_CLOCKWISE
D = "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt"


def eye_crop(img, lm, W=420, H=210):
    el = bbox_from_points(lm.left_eye); er = bbox_from_points(lm.right_eye)
    x0 = int(min(el[0], er[0])); y0 = int(min(el[1], er[1]))
    x1 = int(max(el[2], er[2])); y1 = int(max(el[3], er[3]))
    pw = int((x1 - x0) * 0.15); ph = int((y1 - y0) * 0.8)
    return cv2.resize(img[max(0, y0-ph):y1+ph, max(0, x0-pw):x1+pw], (W, H))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", type=int, required=True)
    sub = ap.parse_args().subject
    rows = list(csv.DictReader(open(f"{D}/subid_{sub}_disagreements.csv")))
    # build render tasks: (frame, type, label)
    tasks = []
    for r in rows:
        if r["type"] == "FN_gt_no_taskframe":
            continue
        s, e = int(r["start"]), int(r["end"])
        mid = (s + e) // 2
        kind = "FP" if r["type"].startswith("FP") else "FN"
        tasks.append((mid, kind, r["type"], r["min_ear"], r["note"]))
    tasks.sort()
    base = f"{D}/disagreement_frames/subid_{sub}"
    cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
    mesh = VideoFaceDetector()
    saved = 0
    for mid, kind, typ, mn, note in tasks:
        crop = None
        for off in (0, -2, 2, -4, 4, -6, 6):
            fr = mid + off
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, f = cap.read()
            if not ok:
                continue
            img = cv2.rotate(f, ROT)
            lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if lm is None:
                continue
            crop = eye_crop(img, lm)
            cv2.putText(crop, f"{kind} f{mid} minEAR={mn}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if kind == "FP" else (255, 180, 0),
                        1, cv2.LINE_AA)
            cv2.putText(crop, note[:38], (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (220, 220, 220), 1, cv2.LINE_AA)
            break
        if crop is None:
            continue
        d = f"{base}/{kind}"; os.makedirs(d, exist_ok=True)
        cv2.imwrite(f"{d}/f{mid}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88]); saved += 1
    mesh.close(); cap.release()
    print(f"sub{sub}: saved {saved}/{len(tasks)} disagreement frames", flush=True)


if __name__ == "__main__":
    main()
