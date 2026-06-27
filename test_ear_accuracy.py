"""Test EAR blink-detector accuracy on subject 11 vs the recorded blink events.

Ground truth: extracted_blinks/00011 -> 121 events [start_frame, duration].
We process the GT span (frames 7999..25789) of subid_11.mp4, upright (cw),
run FaceMesh + EAR per frame, and score EAR (threshold 0.2 default) at:

  frame level : each frame blink/not (GT frame = inside any GT interval)
  event level : GT event hit if >=1 of its frames is EAR-blink (recall);
                EAR event is TP if it overlaps a GT interval else FP (precision)

Also sweeps the EAR threshold to show the precision/recall trade-off.
Runs on a SLURM compute node.
"""
import sys, csv
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.blink import _ear

SUB = 11
VIDEO = f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{SUB}.mp4"
GT_FILE = "/springbrook/share/eng/esrpxk/datasets/extracted_blinks/00011"
OUT = "/springbrook/share/eng/esrpxk/datasets/ear_accuracy_sub11"
ROT = cv2.ROTATE_90_CLOCKWISE
MARGIN = 30  # frames of context around the GT span
_THR_MAP = "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/per_subject_ear_thr.csv"
def _ear_threshold(sub, default=0.23):
    try:
        for r in csv.DictReader(open(_THR_MAP)):
            if int(r["subject"]) == sub:
                return float(r["thr"])
    except FileNotFoundError:
        pass
    return default
EAR_THR = _ear_threshold(SUB)  # per-subject optimum (sub11); default 0.23


def runs_from_flags(frame_ids, flags):
    """Contiguous runs of True -> list of (start_frame, end_frame_inclusive)."""
    events = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1] and frame_ids[j + 1] == frame_ids[j] + 1:
                j += 1
            events.append((frame_ids[i], frame_ids[j]))
            i = j + 1
        else:
            i += 1
    return events


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(GT_FILE, "rb") as f:
        gt = np.load(f)
    gt_intervals = [(int(s), int(s + d)) for s, d in gt]  # [start, end)
    lo = min(s for s, _ in gt_intervals) - MARGIN
    hi = max(e for _, e in gt_intervals) + MARGIN
    print(f"sub{SUB}: {len(gt_intervals)} GT events, processing frames {lo}..{hi}", flush=True)

    cap = cv2.VideoCapture(VIDEO)
    mesh = VideoFaceDetector()

    fids, ears, noface = [], [], 0
    # Decode sequentially from frame 0 -- do NOT cap.set(POS_FRAMES): for H.264 it
    # lands on the nearest keyframe, mislabeling frames by a per-video constant and
    # creating a spurious EAR-vs-GT offset. fr is then the TRUE video frame index.
    fr = 1  # 1-indexed to match ProcessedData/appleFace + GT numbering
    fh = open(f"{OUT}/per_frame_ear.csv", "w", newline="")
    w = csv.writer(fh); w.writerow(["frame", "ear", "no_face"])
    while fr <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if fr < lo:
            fr += 1
            continue
        lm = mesh.detect(cv2.cvtColor(cv2.rotate(f, ROT), cv2.COLOR_BGR2RGB))
        if lm is None:
            noface += 1
            fids.append(fr); ears.append(np.nan)
            w.writerow([fr, "", 1])
        else:
            e = min(_ear(lm.left_eye_ear), _ear(lm.right_eye_ear))
            fids.append(fr); ears.append(e)
            w.writerow([fr, f"{e:.4f}", 0])
        fr += 1
        if (fr - lo) % 2000 == 0:
            print(f"  {fr-lo} frames", flush=True)
    fh.close(); mesh.close(); cap.release()
    ears = np.array(ears)
    fids = np.array(fids)
    print(f"processed {len(fids)} frames, no_face={noface}", flush=True)

    # restrict to the GT task windows (t_init + task_begin, length task_len);
    # GT only counts blinks there, and the inter-task gap frames otherwise inflate
    # EAR false positives. NOT the appleFace glob (a superset incl. gap frames).
    from gaze_dynamics import config
    idx = config.SUBJECT_IDS.index(SUB)
    tinit = config.T_INITS[idx]
    tb, tl, _ = config.build_task_timeline()
    taskset = set()
    for a, l in zip(tb, tl):
        taskset.update(range(int(a + tinit), int(a + tinit + l)))
    keep = np.array([int(f) in taskset for f in fids])
    fids, ears = fids[keep], ears[keep]
    print(f"task-window frames kept: {int(keep.sum())} of {len(keep)}", flush=True)

    # GT frame membership
    gt_frame = np.zeros(len(fids), bool)
    for s, e in gt_intervals:
        gt_frame |= (fids >= s) & (fids < e)

    def score(thr):
        ear_blink = ears < thr  # NaN < thr -> False (no_face = not blink)
        # frame-level
        tp = int(np.sum(ear_blink & gt_frame))
        fp = int(np.sum(ear_blink & ~gt_frame))
        fn = int(np.sum(~ear_blink & gt_frame))
        tn = int(np.sum(~ear_blink & ~gt_frame))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        acc = (tp + tn) / len(fids)
        # event-level
        ev = runs_from_flags(list(fids), list(ear_blink))
        gt_hit = 0
        for s, e in gt_intervals:
            seg = (fids >= s) & (fids < e)
            if np.any(ear_blink & seg):
                gt_hit += 1
        ear_tp = 0
        for es, ee in ev:
            if any(not (ee < s or es >= e) for s, e in gt_intervals):
                ear_tp += 1
        ev_rec = gt_hit / len(gt_intervals)
        ev_prec = ear_tp / len(ev) if ev else 0
        ev_f1 = 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if ev_prec + ev_rec else 0
        return dict(thr=thr, f_prec=prec, f_rec=rec, f_f1=f1, f_acc=acc,
                    tp=tp, fp=fp, fn=fn, n_ear_events=len(ev),
                    ev_recall=ev_rec, ev_prec=ev_prec, ev_f1=ev_f1,
                    gt_hit=gt_hit, ear_tp=ear_tp)

    rows = [score(t) for t in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]]
    with open(f"{OUT}/ear_accuracy.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    d = score(EAR_THR)
    print("\n==================== EAR vs recorded blinks (sub11) ====================")
    print(f"GT events: {len(gt_intervals)}   frames scored: {len(fids)}   no_face: {noface}")
    print(f"\n--- DEFAULT EAR threshold {EAR_THR} ---")
    print(f"FRAME-level : precision={d['f_prec']:.3f} recall={d['f_rec']:.3f} "
          f"F1={d['f_f1']:.3f} accuracy={d['f_acc']:.4f}")
    print(f"EVENT-level : recall={d['ev_recall']:.3f} ({d['gt_hit']}/{len(gt_intervals)} GT blinks detected) "
          f"precision={d['ev_prec']:.3f} ({d['ear_tp']}/{d['n_ear_events']} EAR events valid) F1={d['ev_f1']:.3f}")
    print("\n--- threshold sweep (event-level) ---")
    print(f"{'thr':>6}{'ev_recall':>11}{'ev_prec':>9}{'ev_f1':>8}{'n_ear_ev':>10}")
    for r in rows:
        print(f"{r['thr']:>6}{r['ev_recall']:>11.3f}{r['ev_prec']:>9.3f}{r['ev_f1']:>8.3f}{r['n_ear_events']:>10}")
    print(f"\nWrote {OUT}/ear_accuracy.csv and per_frame_ear.csv")


if __name__ == "__main__":
    main()
