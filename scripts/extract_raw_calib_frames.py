"""Extract RAW calibration-support frames from the session videos (task #6.3).
Decodes each subid video SEQUENTIALLY from frame 0 (never cap.set(POS_FRAMES):
known H.264 seek artifact) and saves the frames whose IDs appear in
calib_support_K72/<rec>/appleFace/ to OriginalCalib/<rec05d>/<frame05d>.jpg.
Sanity check: also decodes one TASK frame per rec and compares to the
OriginalData jpg (mean abs diff) to verify frameIndex == video frame number.
"""
import glob, os, sys
import cv2
import numpy as np

VID = "/springbrook/share/eng/esrpxk/datasets/videos"
SUP = "/springbrook/share/eng/esrpxk/datasets/calib_support_K72"
ORIG = "/springbrook/share/eng/esrpxk/datasets/OriginalData"
OUT = "/springbrook/share/eng/esrpxk/datasets/OriginalCalib"

for rec_dir in sorted(glob.glob(f"{SUP}/0*")):
    rec = int(os.path.basename(rec_dir))
    want = sorted(int(f[:-4]) for f in os.listdir(f"{rec_dir}/appleFace"))
    task_probe = sorted(int(f[:-4]) for f in os.listdir(f"{ORIG}/{rec:05d}"))[0]
    targets = set(want) | {task_probe}
    os.makedirs(f"{OUT}/{rec:05d}", exist_ok=True)
    cap = cv2.VideoCapture(f"{VID}/subid_{rec}.mp4")
    n, got, last = 0, 0, max(targets)
    while True:
        ok, frame = cap.read()
        if not ok or n > last:
            break
        if n in targets:
            if n == task_probe:
                ref = cv2.imread(f"{ORIG}/{rec:05d}/{task_probe:05d}.jpg")
                # OriginalData is a crop of the full frame; compare via min-shape overlap MAD
                h, w = min(ref.shape[0], frame.shape[0]), min(ref.shape[1], frame.shape[1])
                mad = float(np.abs(ref[:h,:w].astype(int) - frame[:h,:w].astype(int)).mean())
                print(f"rec {rec}: probe frame {task_probe} MAD={mad:.1f} "
                      f"(video {frame.shape[:2]}, orig {ref.shape[:2]})", flush=True)
            if n in set(want):
                cv2.imwrite(f"{OUT}/{rec:05d}/{n:05d}.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                got += 1
        n += 1
    cap.release()
    print(f"rec {rec}: saved {got}/{len(want)} calib frames (decoded {n})", flush=True)
print("done")
