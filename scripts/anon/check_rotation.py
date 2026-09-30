"""Which rotation makes each source video upright?

generate_calib_support.py applies cv2.ROTATE_90_CLOCKWISE unconditionally, on
the documented assumption that "the source video is stored 90 deg CCW". That is
true of the 15 portrait recordings (1080x1920). Three recordings -- subid_12, 19
and 20 -- are stored 1920x1080, so the assumption needs checking rather than
inheriting.

Frame identity is deliberately not used: cap.set(POS_FRAMES) lands on the wrong
frame for these H.264 files, which is why a template-match test against the
stored crop returns 0.38-0.52 for every rotation and ranks them by noise.
Orientation is a property of the stream, not of the frame, so this samples
frames sequentially from the start and asks FaceMesh which rotation yields an
upright face: eyes above nose above mouth, and a small eye-line tilt.
"""
import sys, cv2, numpy as np
sys.path.insert(0, "/springbrook/share/eng/esrpxk/CogntiveGaze")
from video_preprocess.detector import VideoFaceDetector

ROTS = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
VID = "/springbrook/share/eng/esrpxk/datasets/videos"


def probe(sub, n_frames=5, stride=400, skip=600):
    cap = cv2.VideoCapture(f"{VID}/subid_{sub}.mp4")
    w, h = int(cap.get(3)), int(cap.get(4))
    frames, i = [], 0
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i >= skip and (i - skip) % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    res = {}
    for nm, code in ROTS.items():
        det = VideoFaceDetector(refine_landmarks=True)
        hits, tilts, upright = 0, [], 0
        for f in frames:
            img = f if code is None else cv2.rotate(f, code)
            lm = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if lm is None:
                continue
            hits += 1
            le = np.asarray(lm.left_eye).mean(0)
            re = np.asarray(lm.right_eye).mean(0)
            ov = np.asarray(lm.face_oval)
            tilts.append(abs(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))))
            # upright = eye line sits in the upper half of the face oval
            eye_y = (le[1] + re[1]) / 2
            y0, y1 = ov[:, 1].min(), ov[:, 1].max()
            if y1 > y0 and (eye_y - y0) / (y1 - y0) < 0.5:
                upright += 1
        res[nm] = (hits, len(frames), float(np.median(tilts)) if tilts else None, upright)
    return w, h, res


if __name__ == "__main__":
    subs = [int(s) for s in sys.argv[1:]] or list(range(6, 24))
    print(f"{'subid':8}{'raw':>12}   " + "".join(f"{k:>22}" for k in ROTS))
    print(f"{'':8}{'':>12}   " + "".join(f"{'det/tilt/upright':>22}" for _ in ROTS))
    for s in subs:
        w, h, r = probe(s)
        line = f"subid_{s:<3}{w}x{h:<7}   "
        for k in ROTS:
            hits, n, tilt, up = r[k]
            t = f"{tilt:.0f}d" if tilt is not None else "--"
            line += f"{f'{hits}/{n} {t} up{up}':>22}"
        ok = [k for k in ROTS if r[k][0] == r[k][1] and r[k][3] == r[k][1]
              and r[k][2] is not None and r[k][2] < 20]
        print(line + f"   -> {ok}")
