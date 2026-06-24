"""Per-subject review pack:
  review/subid_<id>/montage.jpg                  the 3x3 screen-target montage
  review/subid_<id>/<screen_name>/f<frame>.jpg   linked eye crops, every 10 frames
Uses datasets/calib_clusters/subid_<id>_clusters.csv. Runs on a SLURM compute node.
"""
import sys, csv, os, shutil
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points

CL = "/springbrook/share/eng/esrpxk/datasets/calib_clusters"
OUT = f"{CL}/review"
MONT = f"{CL}/screen_montages"
SUBS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ROT = cv2.ROTATE_90_CLOCKWISE
INTERVAL = 10


def eye_crop(img, lm, W=360, H=180):
    el = bbox_from_points(lm.left_eye); er = bbox_from_points(lm.right_eye)
    x0 = int(min(el[0], er[0])); y0 = int(min(el[1], er[1]))
    x1 = int(max(el[2], er[2])); y1 = int(max(el[3], er[3]))
    pw = int((x1 - x0) * 0.15); ph = int((y1 - y0) * 0.8)
    return cv2.resize(img[max(0, y0-ph):y1+ph, max(0, x0-pw):x1+pw], (W, H))


def main():
    mesh = VideoFaceDetector()
    for sub in SUBS:
        path = f"{CL}/subid_{sub}_clusters.csv"
        if not os.path.exists(path):
            continue
        subdir = f"{OUT}/subid_{sub}"; os.makedirs(subdir, exist_ok=True)
        if os.path.exists(f"{MONT}/sub{sub}_screen_montage.jpg"):
            shutil.copy(f"{MONT}/sub{sub}_screen_montage.jpg", f"{subdir}/montage.jpg")
        # collect frames per screen target
        bytgt = {}
        for r in csv.DictReader(open(path)):
            if r["in_fixation"] == "1" and r["is_mistake"] == "0" and r["screen_name"]:
                bytgt.setdefault((r["screen_name"], int(r["screen_x"]), int(r["screen_y"])), []).append(int(r["abs_frame"]))
        # sample every INTERVAL frames per target; collect (frame, target)
        tasks = []
        for (snm, sx, sy), frs in bytgt.items():
            frs = sorted(frs)
            for fr in frs[::INTERVAL]:
                tasks.append((fr, snm, sx, sy))
        tasks.sort()  # frame order -> tracking-mode friendly
        cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
        saved = 0
        for fr, snm, sx, sy in tasks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, f = cap.read()
            if not ok:
                continue
            img = cv2.rotate(f, ROT)
            lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if lm is None:
                continue
            crop = eye_crop(img, lm)
            cv2.putText(crop, f"{snm} ({sx},{sy})  f{fr}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            d = f"{subdir}/{snm}"; os.makedirs(d, exist_ok=True)
            cv2.imwrite(f"{d}/f{fr}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88]); saved += 1
        cap.release()
        print(f"sub{sub}: montage + {saved} example crops (interval {INTERVAL})", flush=True)
    mesh.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
