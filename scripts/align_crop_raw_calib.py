"""Find each recording's OriginalData crop window inside the full 1920x1080
video frame (template match on the task probe frame), then apply that crop to
the extracted OriginalCalib frames so raw calib frames match the raw-model
training distribution (1080x750 above-shoulder crop)."""
import glob, os
import cv2
import numpy as np

VID = "/springbrook/share/eng/esrpxk/datasets/videos"
ORIG = "/springbrook/share/eng/esrpxk/datasets/OriginalData"
CAL = "/springbrook/share/eng/esrpxk/datasets/OriginalCalib"

for rec_dir in sorted(glob.glob(f"{CAL}/0*")):
    rec = int(os.path.basename(rec_dir))
    probe_id = sorted(int(f[:-4]) for f in os.listdir(f"{ORIG}/{rec:05d}")
                      if f.endswith(".jpg") and f[:-4].isdigit())[0]
    ref = cv2.imread(f"{ORIG}/{rec:05d}/{probe_id:05d}.jpg")           # 750x1080 crop
    # decode probe frame sequentially (no seek)
    cap = cv2.VideoCapture(f"{VID}/subid_{rec}.mp4"); n = 0; frame = None
    while n <= probe_id:
        ok, frame = cap.read()
        if not ok: break
        n += 1
    cap.release()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)      # video is portrait-in-landscape
    res = cv2.matchTemplate(frame, ref, cv2.TM_SQDIFF_NORMED)
    mn, _, loc, _ = cv2.minMaxLoc(res)
    x, y = loc; h, w = ref.shape[:2]
    crop_mad = float(np.abs(frame[y:y+h, x:x+w].astype(int) - ref.astype(int)).mean())
    print(f"rec {rec}: crop offset=({x},{y}) size=({w}x{h}) sqdiff={mn:.5f} MAD={crop_mad:.2f}", flush=True)
    if crop_mad > 8:
        print(f"rec {rec}: WARN alignment poor, skipping crop"); continue
    for fp in glob.glob(f"{rec_dir}/*.jpg"):
        img = cv2.imread(fp)
        if img.shape[0] > h:   # still full-size -> rotate + crop in place
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(fp, img[y:y+h, x:x+w], [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"rec {rec}: cropped {len(glob.glob(f'{rec_dir}/*.jpg'))} calib frames", flush=True)
print("done")
