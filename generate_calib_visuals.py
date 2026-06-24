"""Generate calibration-link visuals per subject:
  1. 3x3 eye-crop montage arranged by linked screen target.
  2. Full-frame examples (face + mini screen diagram with the active target lit).
Uses the anchor-based frame->point links in calib_point_links/.
Runs on a SLURM compute node.
"""
import sys, csv, os
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points

LINKS = "/springbrook/share/eng/esrpxk/datasets/calib_point_links"
OUT = f"{LINKS}/visuals"
SUBS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ROT = cv2.ROTATE_90_CLOCKWISE
SCREEN = {0: (960, 540), 1: (960, 92), 2: (960, 988), 3: (115, 540), 4: (1805, 540),
          5: (217, 146), 6: (1703, 146), 7: (217, 934), 8: (1703, 934)}
CELL = {5: (0, 0), 1: (0, 1), 6: (0, 2), 3: (1, 0), 0: (1, 1), 4: (1, 2),
        7: (2, 0), 2: (2, 1), 8: (2, 2)}


def eye_crop(img, lm, W, H):
    el = bbox_from_points(lm.left_eye); er = bbox_from_points(lm.right_eye)
    x0 = int(min(el[0], er[0])); y0 = int(min(el[1], er[1]))
    x1 = int(max(el[2], er[2])); y1 = int(max(el[3], er[3]))
    pw = int((x1 - x0) * 0.15); ph = int((y1 - y0) * 0.8)
    x0 = max(0, x0 - pw); y0 = max(0, y0 - ph)
    x1 = min(img.shape[1], x1 + pw); y1 = min(img.shape[0], y1 + ph)
    return cv2.resize(img[y0:y1, x0:x1], (W, H))


def mini_screen(active_pt, w=300):
    h = int(w * 1080 / 1920)
    panel = np.full((h, w, 3), 30, np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    for p, (sxp, syp) in SCREEN.items():
        x = int(sxp / 1920 * w); y = int(syp / 1080 * h)
        if p == active_pt:
            cv2.circle(panel, (x, y), 9, (0, 215, 255), -1)
            cv2.circle(panel, (x, y), 11, (0, 255, 255), 2)
        else:
            cv2.circle(panel, (x, y), 5, (120, 120, 120), 1)
    cv2.putText(panel, "screen target", (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return panel


def read_frame(cap, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, f = cap.read()
    return cv2.rotate(f, ROT) if ok else None


def main():
    os.makedirs(OUT, exist_ok=True)
    mesh = VideoFaceDetector()
    for sub in SUBS:
        path = f"{LINKS}/subid_{sub}_calib_link.csv"
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        vid = f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4"
        cap = cv2.VideoCapture(vid)
        # frames per point
        byp = {}
        for r in rows:
            if r["status"] == "fixation" and r["point_id"] != "":
                byp.setdefault(int(r["point_id"]), []).append(int(r["abs_frame"]))

        # ---- 3x3 montage ----
        CW, CH = 300, 150
        mont = np.full((3 * CH + 40, 3 * CW + 40, 3), 40, np.uint8)
        for p, frs in byp.items():
            fr = frs[len(frs) // 2]
            img = read_frame(cap, fr)
            if img is None:
                continue
            lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if lm is None:
                continue
            crop = eye_crop(img, lm, CW, CH)
            cv2.putText(crop, f"target {SCREEN[p]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop, f"f{fr}", (5, CH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            rr, cc = CELL[p]
            mont[20 + rr * (CH + 10):20 + rr * (CH + 10) + CH,
                 20 + cc * (CW + 10):20 + cc * (CW + 10) + CW] = crop
        cv2.putText(mont, f"sub{sub}: eye crops by linked target (3x3 screen grid)",
                    (20, mont.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUT}/sub{sub}_montage.jpg", mont, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # ---- full-frame examples (one per several distinct points) ----
        ex_pts = [0, 5, 6, 7, 8]  # center + 4 corners
        for p in ex_pts:
            if p not in byp:
                continue
            fr = byp[p][len(byp[p]) // 2]
            img = read_frame(cap, fr)
            if img is None:
                continue
            lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            big = cv2.resize(img, (540, 960))
            if lm is not None:
                fb = bbox_from_points(lm.face_oval); sc = 540 / img.shape[1]
                cv2.rectangle(big, (int(fb[0] * sc), int(fb[1] * 960 / img.shape[0])),
                              (int(fb[2] * sc), int(fb[3] * 960 / img.shape[0])), (0, 255, 0), 2)
            ms = mini_screen(p, w=300)
            canvas = np.full((960, 540 + 320, 3), 25, np.uint8)
            canvas[:, :540] = big
            canvas[20:20 + ms.shape[0], 550:550 + ms.shape[1]] = ms
            cv2.putText(canvas, f"sub{sub} f{fr}", (550, ms.shape[0] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"target {SCREEN[p]}", (550, ms.shape[0] + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imwrite(f"{OUT}/sub{sub}_fullframe_pt{p}_f{fr}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cap.release()
        print(f"sub{sub}: montage + {len(ex_pts)} full-frame examples", flush=True)
    mesh.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
