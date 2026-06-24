"""Build the per-K calibration SUPPORT datasets that the meta-calibration
pipeline consumes via `metacompare --calib-support-root` (replaces the leak-prone
random-K-from-test sampling).

Driven by datasets/calib_selection/calib_selection.csv (from select_calib_frames.py):
for each K in {4,9,18,36,72} and each selected frame, re-crop face/eyes in the
ProcessedData convention (face = square face_oval/0.94; eyes = square 2.0x eye
width), assign the cm gaze label (screen target px -> cm), compute the iTracker
face grid, and write a GazeCapture-format dataset:

  datasets/calib_support_K<K>/<rec:05d>/appleFace|appleLeftEye|appleRightEye/<frame:05d>.jpg
  datasets/calib_support_K<K>/<mean>/metadata.mat
    (labelRecNum, frameIndex, labelDotXCam, labelDotYCam, labelFaceGrid)

Each frame is cropped ONCE per subject and shared across the K datasets that
select it. Runs on a SLURM compute node (video decode + FaceMesh).
"""
import sys, csv, os
import numpy as np
import scipy.io as sio
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points
from video_preprocess.face_grid import face_grid_params

SEL = "/springbrook/share/eng/esrpxk/datasets/calib_selection/calib_selection.csv"
DS = "/springbrook/share/eng/esrpxk/datasets"
VIDEOS = "/springbrook/share/eng/esrpxk/datasets/videos"
MEAN = "mean7"
KS = [4, 9, 18, 36, 72]
ROT = cv2.ROTATE_90_CLOCKWISE
SCREEN_W_CM, SCREEN_H_CM = 54.4, 30.4
SCREEN_W_PX, SCREEN_H_PX = 1920.0, 1080.0
FACE_FILL = 0.94
EYE_FACTOR = 2.0
FACE_SIZE = EYE_SIZE = 224


def sq_crop(img, cx, cy, side, out):
    h, w = img.shape[:2]; s = int(round(side))
    x0 = int(round(cx - s/2)); y0 = int(round(cy - s/2)); x1 = x0 + s; y1 = y0 + s
    pl, pt = max(0, -x0), max(0, -y0); pr, pb = max(0, x1-w), max(0, y1-h)
    pad = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    crop = pad[y0+pt:y1+pt, x0+pl:x1+pl]
    return cv2.resize(crop, (out, out))


def px_to_cm(sx, sy):
    # screen pixel -> camera/screen-CENTERED cm, matching the task labelDotXCam/
    # YCam space (x in [-W/2, W/2], y in [-H/2, H/2]); screen centre -> (0, 0).
    return ((sx - SCREEN_W_PX / 2) / SCREEN_W_PX * SCREEN_W_CM,
            (sy - SCREEN_H_PX / 2) / SCREEN_H_PX * SCREEN_H_CM)


def crop_frame(cap, mesh, fr):
    """Crop the EXACT selected frame; if FaceMesh misses (tracking mode after a
    seek), jitter +-1..3 frames -- still inside the same fixation, same gaze."""
    for d in (0, 1, -1, 2, -2, 3, -3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr + d)
        ok, f = cap.read()
        if not ok:
            continue
        img = cv2.rotate(f, ROT)
        lm = mesh.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if lm is None:
            continue
        fo = bbox_from_points(lm.face_oval)
        fcx, fcy, fw = (fo[0]+fo[2])/2, (fo[1]+fo[3])/2, fo[2]-fo[0]
        side = fw / FACE_FILL
        face = sq_crop(img, fcx, fcy - 0.02*side, side, FACE_SIZE)

        def ecrop(pts):
            b = bbox_from_points(pts); cx, cy, ew = (b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0]
            return sq_crop(img, cx, cy, ew*EYE_FACTOR, EYE_SIZE)
        le = ecrop(lm.left_eye); re = ecrop(lm.right_eye)
        fb = (fcx - side/2, fcy - 0.02*side - side/2, fcx + side/2, fcy - 0.02*side + side/2)
        grid = face_grid_params(fb, (img.shape[1], img.shape[0]), grid_size=25)
        return face, le, re, list(grid)
    return None


def main():
    sel = {}; uniq = {}
    for r in csv.DictReader(open(SEL)):
        sub = int(r["sub"]); K = int(r["K"]); fr = int(r["absf"])
        sx, sy = int(r["screen_x"]), int(r["screen_y"])
        sel.setdefault(sub, []).append((K, fr, sx, sy))
        uniq.setdefault(sub, set()).add(fr)

    for K in KS:
        for sub in uniq:
            for k in ("appleFace", "appleLeftEye", "appleRightEye"):
                os.makedirs(f"{DS}/calib_support_K{K}/{sub:05d}/{k}", exist_ok=True)

    mesh = VideoFaceDetector()
    meta = {K: [] for K in KS}
    for sub in sorted(uniq):
        cap = cv2.VideoCapture(f"{VIDEOS}/subid_{sub}.mp4")
        cache = {fr: crop_frame(cap, mesh, fr) for fr in sorted(uniq[sub])}
        miss = [fr for fr, c in cache.items() if c is None]
        for (K, fr, sx, sy) in sel[sub]:
            c = cache.get(fr)
            if c is None:
                continue
            face, le, re, grid = c
            root = f"{DS}/calib_support_K{K}/{sub:05d}"; name = f"{fr:05d}.jpg"
            cv2.imwrite(f"{root}/appleFace/{name}", face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{root}/appleLeftEye/{name}", le, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{root}/appleRightEye/{name}", re, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gx, gy = px_to_cm(sx, sy)
            meta[K].append((sub, fr, gx, gy, grid))
        cap.release()
        print(f"sub{sub}: {len(uniq[sub])} unique frames cropped, {len(miss)} undetected {miss}", flush=True)
    mesh.close()

    for K in KS:
        base = f"{DS}/calib_support_K{K}/{MEAN}"; os.makedirs(base, exist_ok=True)
        rows = meta[K]
        sio.savemat(f"{base}/metadata.mat", {
            "labelRecNum": np.array([r[0] for r in rows], np.int32),
            "frameIndex": np.array([r[1] for r in rows], np.int32),
            "labelDotXCam": np.array([r[2] for r in rows], np.float32),
            "labelDotYCam": np.array([r[3] for r in rows], np.float32),
            "labelFaceGrid": np.array([r[4] for r in rows], np.float32),
        })
        print(f"K={K}: wrote {base}/metadata.mat ({len(rows)} rows, "
              f"{len(rows)//18 if K>=18 else len(rows)//(K)} ~per-subject)")


if __name__ == "__main__":
    main()
