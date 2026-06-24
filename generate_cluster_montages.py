"""Per-subject 3x3 eye-crop montage arranged by the linked SCREEN target, built
from the UPDATED calib-frame link (datasets/calib_point_links/subid_<id>_calib_link.csv).
Each of the 9 screen cells shows the eye crop for cycle 1 (left) and cycle 2
(right), so the two calibration visits to the same target can be compared.
Runs on a SLURM compute node (video decode + FaceMesh). Output:
datasets/calib_clusters/screen_montages/sub<id>_screen_montage.jpg
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
OUT = "/springbrook/share/eng/esrpxk/datasets/calib_clusters/screen_montages"
SUBS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ROT = cv2.ROTATE_90_CLOCKWISE
CW, CH = 200, 120
# screen target (px) -> 3x3 montage cell (row, col)
SCREEN_CELL = {(217, 146): (0, 0), (960, 92): (0, 1), (1703, 146): (0, 2),
               (115, 540): (1, 0), (960, 540): (1, 1), (1805, 540): (1, 2),
               (217, 934): (2, 0), (960, 988): (2, 1), (1703, 934): (2, 2)}


def eye_crop(img, lm, W, H):
    el = bbox_from_points(lm.left_eye); er = bbox_from_points(lm.right_eye)
    x0 = int(min(el[0], er[0])); y0 = int(min(el[1], er[1]))
    x1 = int(max(el[2], er[2])); y1 = int(max(el[3], er[3]))
    pw = int((x1 - x0) * 0.15); ph = int((y1 - y0) * 0.8)
    return cv2.resize(img[max(0, y0-ph):y1+ph, max(0, x0-pw):x1+pw], (W, H))


def best_crop(cap, mesh, frames):
    """Try frames near the middle of the fixation; return the first that yields
    a face/eye crop (FaceMesh runs in tracking mode and can miss after a seek)."""
    order = sorted(range(len(frames)), key=lambda i: abs(i - len(frames) // 2))
    for i in order[:15]:
        fr = frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, f = cap.read()
        if not ok:
            continue
        img = cv2.rotate(f, ROT)
        lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if lm is None:
            continue
        return eye_crop(img, lm, CW, CH), fr
    return None, None


def main():
    os.makedirs(OUT, exist_ok=True)
    mesh = VideoFaceDetector()
    for sub in SUBS:
        path = f"{LINK}/subid_{sub}_calib_link.csv"
        if not os.path.exists(path):
            continue
        bykey = {}                                  # (target, cycle) -> [frames]
        names = {}
        for r in csv.DictReader(open(path)):
            # show the two calibration cycles; clamp any partial extra sweep to c2
            if r["status"] == "fixation" and r["screen_x"]:
                tgt = (int(r["screen_x"]), int(r["screen_y"]))
                cyc = min(int(r["cycle"]), 2)
                bykey.setdefault((tgt, cyc), []).append(int(r["abs_frame"]))
                names[tgt] = r["screen_name"]
        cap = cv2.VideoCapture(f"{VIDEOS}/subid_{sub}.mp4")
        mont = np.full((3 * CH + 50, 3 * (2 * CW + 6) + 40, 3), 35, np.uint8)
        for (tgt, cyc), frs in bykey.items():
            crop, fr = best_crop(cap, mesh, frs)
            if crop is None:
                continue
            crop = crop.copy()
            cv2.putText(crop, f"{names[tgt]} c{cyc}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop, f"f{fr}", (5, CH - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            rr, cc = SCREEN_CELL[tgt]
            y0 = 20 + rr * (CH + 10)
            x0 = 20 + cc * (2 * CW + 6) + (cyc - 1) * (CW + 6)
            mont[y0:y0 + CH, x0:x0 + CW] = crop
        cv2.putText(mont, f"sub{sub}: eye crops by linked SCREEN target (cycle1 | cycle2)",
                    (20, mont.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUT}/sub{sub}_screen_montage.jpg", mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cap.release()
        print(f"sub{sub}: montage done ({len(bykey)} target-visits)", flush=True)
    mesh.close()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
