"""EAR vs ground-truth blink events for one subject; save all disagreements.

GT = extracted_blinks/000<id> (recorded blink events [start, duration]).
Task-frame mask = ProcessedData/000<id>/appleFace frame indices (non-task frames
are excluded, since the GT blinks only live in the task period).

Process the GT span (upright, rotate cw), FaceMesh + EAR per frame, threshold 0.2,
restrict to task frames, build EAR blink events, and emit:
  FP_ear_only       : an EAR blink event overlapping no GT event
  FN_gt_missed      : a GT blink event with no EAR-blink frame
Output: <out>/subid_<id>_disagreements.csv  (+ per_frame_ear.csv, summary line)
"""
import argparse, csv, sys, os, glob
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.blink import _ear
from gaze_dynamics import config

ROT = cv2.ROTATE_90_CLOCKWISE
EAR_THR = 0.23  # default if no per-subject value; cohort-optimal fixed threshold
MARGIN = 30
THR_MAP = "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/per_subject_ear_thr.csv"


def ear_threshold(sub, default=EAR_THR):
    """Per-subject EAR threshold (global-min FP+FN, plateau-centered). Falls back
    to the fixed default if the map is missing."""
    try:
        for r in csv.DictReader(open(THR_MAP)):
            if int(r["subject"]) == sub:
                return float(r["thr"])
    except FileNotFoundError:
        pass
    return default


def task_frames(sub):
    """Authoritative GT task windows = t_init + task_begin, length task_len
    (gaze_dynamics.config). NOT the appleFace glob: appleFace is a superset that
    also holds ~958 inter-task gap frames where GT counted no blinks, which
    inflated EAR false positives."""
    idx = config.SUBJECT_IDS.index(sub)
    tinit = config.T_INITS[idx]
    tb, tl, _ = config.build_task_timeline()
    kept = set()
    for a, l in zip(tb, tl):
        kept.update(range(int(a + tinit), int(a + tinit + l)))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", default="/springbrook/share/eng/esrpxk/datasets/ear_vs_gt")
    ap.add_argument("--tol", type=int, default=5,
                    help="frames of slack for matching an EAR event to a GT event "
                         "(absorbs per-subject frame-sync offsets, e.g. sub23 ~+3-4f)")
    args = ap.parse_args()
    sub = args.subject
    os.makedirs(args.out, exist_ok=True)
    EAR_THR = ear_threshold(sub)
    print(f"sub{sub}: EAR_THR={EAR_THR}", flush=True)

    with open(f"/springbrook/share/eng/esrpxk/datasets/extracted_blinks/{sub:05d}", "rb") as f:
        gt = np.load(f)
    gt_iv = [(int(s), int(s + d)) for s, d in gt]
    kept = task_frames(sub)
    if not kept:
        print(f"sub{sub}: no task frames"); return
    lo = min(s for s, _ in gt_iv) - MARGIN
    hi = max(e for _, e in gt_iv) + MARGIN

    cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
    mesh = VideoFaceDetector()
    # Decode sequentially from frame 0 -- do NOT cap.set(POS_FRAMES): for H.264 it
    # lands on the nearest keyframe, mislabeling frames by a per-video constant and
    # creating a spurious per-subject EAR-vs-GT offset. fr is then the TRUE index.
    ear = {}  # frame -> ear (only task frames)
    fr = 1; noface = 0  # 1-indexed to match ProcessedData/appleFace + GT numbering
    while fr <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if fr >= lo and fr in kept:
            lm = mesh.detect(cv2.cvtColor(cv2.rotate(f, ROT), cv2.COLOR_BGR2RGB))
            if lm is None:
                noface += 1
            else:
                ear[fr] = min(_ear(lm.left_eye_ear), _ear(lm.right_eye_ear))
        fr += 1
    mesh.close(); cap.release()

    # EAR blink events: contiguous kept-frame runs with ear < thr
    kf = sorted(ear)
    blink = {f: ear[f] < EAR_THR for f in kf}
    ev = []; i = 0
    while i < len(kf):
        if blink[kf[i]]:
            j = i
            while j + 1 < len(kf) and blink[kf[j+1]] and kf[j+1] == kf[j] + 1:
                j += 1
            ev.append((kf[i], kf[j])); i = j + 1
        else:
            i += 1

    def gap(a, b, c, d):
        """Frame gap between intervals [a,b] and [c,d]; 0 if they touch/overlap."""
        if b < c: return c - b
        if a > d: return a - d
        return 0

    TOL = args.tol
    rows = []
    # FP: EAR events with no GT event within TOL frames
    for es, ee in ev:
        if not any(gap(es, ee, gs, ge) <= TOL for gs, ge in gt_iv):
            mn = min(ear[f] for f in range(es, ee + 1) if f in ear)
            rows.append(["FP_ear_only", es, ee, ee - es + 1, f"{mn:.3f}",
                         "fully-closed (likely GT missed)" if mn < 0.05 else "partial/squint"])
    # FN: GT events with no EAR event within TOL frames (tolerant of frame-sync offset)
    for gs, ge in gt_iv:
        seg = [f for f in range(gs, ge) if f in ear]
        if not seg:
            rows.append(["FN_gt_no_taskframe", gs, ge, ge - gs, "", "GT event outside task frames"])
        elif not any(gap(es, ee, gs, ge) <= TOL for es, ee in ev):
            mn = min(ear[f] for f in seg)
            rows.append(["FN_gt_missed", gs, ge, ge - gs, f"{mn:.3f}", f"no EAR event within {TOL}f"])

    with open(f"{args.out}/subid_{sub}_disagreements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "start", "end", "dur_frames", "min_ear", "note"])
        w.writerows(rows)
    nfp = sum(r[0] == "FP_ear_only" for r in rows)
    nfn = sum(r[0].startswith("FN") for r in rows)
    # GT detected count
    gt_hit = sum(1 for gs, ge in gt_iv
                 if any(ear.get(f, 1) < EAR_THR for f in range(gs, ge)))
    print(f"sub{sub}: GT={len(gt_iv)} task_ear_frames={len(ear)} noface={noface} "
          f"GT_detected={gt_hit} FP={nfp} FN={nfn}", flush=True)


if __name__ == "__main__":
    main()
