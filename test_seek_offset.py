"""Diagnose the per-subject EAR-vs-GT offset: is it a cv2 POS_FRAMES seek error?

For each subject, compare the frame returned by cap.set(POS_FRAMES, lo)+read()
against the SAME video decoded sequentially from frame 0. If seeking is exact the
match is at `lo` (seek_err 0); if it lands early/late the match index differs, and
seek_err == -(observed EAR-GT offset) would confirm the offset is a seek artifact.
Runs on a SLURM compute node."""
import sys
import numpy as np
import cv2
sys.path.insert(0, ".")
from gaze_dynamics import config

VID = "/springbrook/share/eng/esrpxk/datasets/videos/subid_{}.mp4"
GT = "/springbrook/share/eng/esrpxk/datasets/extracted_blinks/{:05d}"
MARGIN = 30
W = 8  # search window around lo

def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int32)

def seek_err(sub):
    gt = np.load(open(GT.format(sub), "rb")).astype(int)
    lo = int(gt[:, 0].min()) - MARGIN
    vid = VID.format(sub)

    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    ok, seeked = cap.read(); cap.release()
    if not ok:
        return None
    gseek = gray(seeked)

    cap = cv2.VideoCapture(vid)
    store = {}; fr = 0
    while fr <= lo + W:
        ok, f = cap.read()
        if not ok:
            break
        if lo - W <= fr <= lo + W:
            store[fr] = gray(f)
        fr += 1
    cap.release()

    best, bd = None, 1e18
    for k, g in store.items():
        d = float(np.abs(gseek - g).mean())
        if d < bd:
            bd, best = d, k
    return lo, best, bd

if __name__ == "__main__":
    subs = [int(x) for x in sys.argv[1:]] or config.SUBJECT_IDS
    print(f"{'sub':>3} {'req_lo':>7} {'match':>6} {'seek_err':>9} {'predicts_offset':>16} {'pixdiff':>8}")
    for s in subs:
        r = seek_err(s)
        if r is None:
            print(f"{s:>3}  read failed"); continue
        lo, best, bd = r
        err = best - lo
        print(f"{s:>3} {lo:>7} {best:>6} {err:>+9} {(-err):>+16} {bd:>8.2f}")
