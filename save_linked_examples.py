"""Save example LINKED eye images for the 9 calibration points, sampled every
STEP=10 frames, from the updated calib-frame link
(datasets/calib_point_links/subid_<id>_calib_link.csv).

Per subject:
  - individual eye crops: linked_examples/subid_<id>/<screen_name>/f<frame>_c<cycle>.jpg
  - a filmstrip montage:   linked_examples/sub<id>_linked_examples.jpg
    (9 rows = the 9 screen targets in raster order; columns = the interval-10
     samples across both cycles).
Runs on a SLURM compute node (video decode + FaceMesh).
"""
import sys, csv, os
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points

LINK = "/springbrook/share/eng/esrpxk/datasets/calib_point_links"
VIDEOS = "/springbrook/share/eng/esrpxk/datasets/videos"
OUT = "/springbrook/share/eng/esrpxk/datasets/calib_clusters/linked_examples"
SUBS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ROT = cv2.ROTATE_90_CLOCKWISE
STEP = 10                      # sample every 10 linked frames
CW, CH = 150, 95
MAXCOL = 16
# screen target (px) -> raster (row, col) for ordering the 9 rows
SCREEN_CELL = {(217, 146): (0, 0), (960, 92): (0, 1), (1703, 146): (0, 2),
               (115, 540): (1, 0), (960, 540): (1, 1), (1805, 540): (1, 2),
               (217, 934): (2, 0), (960, 988): (2, 1), (1703, 934): (2, 2)}


def eye_crop(img, lm, W, H):
    el = bbox_from_points(lm.left_eye); er = bbox_from_points(lm.right_eye)
    x0 = int(min(el[0], er[0])); y0 = int(min(el[1], er[1]))
    x1 = int(max(el[2], er[2])); y1 = int(max(el[3], er[3]))
    pw = int((x1 - x0) * 0.15); ph = int((y1 - y0) * 0.8)
    return cv2.resize(img[max(0, y0-ph):y1+ph, max(0, x0-pw):x1+pw], (W, H))


def crop_at(cap, mesh, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, f = cap.read()
    if not ok:
        return None
    img = cv2.rotate(f, ROT)
    lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if lm is None:
        return None
    return eye_crop(img, lm, CW, CH)


def main():
    os.makedirs(OUT, exist_ok=True)
    mesh = VideoFaceDetector()
    for sub in SUBS:
        path = f"{LINK}/subid_{sub}_calib_link.csv"
        if not os.path.exists(path):
            continue
        bytgt = {}                                  # tgt -> [(frame, cyc, name)]
        for r in csv.DictReader(open(path)):
            if r["status"] == "fixation" and r["screen_x"]:
                tgt = (int(r["screen_x"]), int(r["screen_y"]))
                bytgt.setdefault(tgt, []).append((int(r["abs_frame"]), int(r["cycle"]), r["screen_name"]))
        cap = cv2.VideoCapture(f"{VIDEOS}/subid_{sub}.mp4")
        sdir = f"{OUT}/subid_{sub}"; os.makedirs(sdir, exist_ok=True)
        order = sorted(bytgt, key=lambda t: SCREEN_CELL[t])
        strip = np.full((9 * (CH + 8) + 10, MAXCOL * (CW + 4) + 130, 3), 35, np.uint8)
        nsaved = 0
        for ri, tgt in enumerate(order):
            recs = sorted(bytgt[tgt])                # by frame
            name = recs[0][2]
            samples = recs[::STEP]                   # interval 10
            pdir = f"{sdir}/{name}"; os.makedirs(pdir, exist_ok=True)
            cv2.putText(strip, name, (4, ri * (CH + 8) + CH // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            col = 0
            for fr, cyc, _ in samples:
                crop = crop_at(cap, mesh, fr)
                if crop is None:
                    continue
                cv2.imwrite(f"{pdir}/f{fr}_c{cyc}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                nsaved += 1
                if col < MAXCOL:
                    c2 = crop.copy()
                    cv2.putText(c2, f"f{fr}c{cyc}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (200, 255, 200), 1, cv2.LINE_AA)
                    y0 = ri * (CH + 8) + 4; x0 = 120 + col * (CW + 4)
                    strip[y0:y0 + CH, x0:x0 + CW] = c2
                    col += 1
        cv2.imwrite(f"{OUT}/sub{sub}_linked_examples.jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cap.release()
        print(f"sub{sub}: saved {nsaved} linked example crops + strip", flush=True)
    mesh.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
