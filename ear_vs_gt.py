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

ROT = cv2.ROTATE_90_CLOCKWISE
EAR_THR = 0.2
MARGIN = 30


def task_frames(sub):
    d = f"/springbrook/share/eng/esrpxk/datasets/ProcessedData/{sub:05d}/appleFace"
    return set(int(os.path.splitext(os.path.basename(p))[0])
               for p in glob.glob(f"{d}/*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", default="/springbrook/share/eng/esrpxk/datasets/ear_vs_gt")
    args = ap.parse_args()
    sub = args.subject
    os.makedirs(args.out, exist_ok=True)

    with open(f"/springbrook/share/eng/esrpxk/datasets/extracted_blinks/{sub:05d}", "rb") as f:
        gt = np.load(f)
    gt_iv = [(int(s), int(s + d)) for s, d in gt]
    kept = task_frames(sub)
    if not kept:
        print(f"sub{sub}: no task frames"); return
    lo = min(s for s, _ in gt_iv) - MARGIN
    hi = max(e for _, e in gt_iv) + MARGIN

    cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    mesh = VideoFaceDetector()
    ear = {}  # frame -> ear (only task frames)
    fr = lo; noface = 0
    while fr <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if fr in kept:
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

    def overlaps(a, b, c, d):
        return not (b < c or a >= d)

    rows = []
    # FP: EAR events overlapping no GT
    for es, ee in ev:
        if not any(overlaps(es, ee + 1, gs, ge) for gs, ge in gt_iv):
            mn = min(ear[f] for f in range(es, ee + 1) if f in ear)
            rows.append(["FP_ear_only", es, ee, ee - es + 1, f"{mn:.3f}",
                         "fully-closed (likely GT missed)" if mn < 0.05 else "partial/squint"])
    # FN: GT events with no EAR-blink frame
    for gs, ge in gt_iv:
        seg = [f for f in range(gs, ge) if f in ear]
        if seg and not any(ear[f] < EAR_THR for f in seg):
            mn = min(ear[f] for f in seg)
            rows.append(["FN_gt_missed", gs, ge, ge - gs, f"{mn:.3f}", "EAR stayed >= 0.20"])
        elif not seg:
            rows.append(["FN_gt_no_taskframe", gs, ge, ge - gs, "", "GT event outside task frames"])

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
