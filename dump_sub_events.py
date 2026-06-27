"""Dump the FULL EAR event list and GT event list for one subject, with match
status (not just disagreements). Mirrors ear_vs_gt.py processing: task-frame
masked, upright, FaceMesh + EAR, threshold 0.2. Writes a readable .txt + CSVs.
Runs on a SLURM compute node (never the login node)."""
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
EAR_THR = 0.23  # default if no per-subject value
MARGIN = 30
TOL = 5  # frames of slack for the "matched" decision
THR_MAP = "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/per_subject_ear_thr.csv"


def ear_threshold(sub, default=EAR_THR):
    """Per-subject EAR threshold (global-min FP+FN); falls back to default."""
    try:
        for r in csv.DictReader(open(THR_MAP)):
            if int(r["subject"]) == sub:
                return float(r["thr"])
    except FileNotFoundError:
        pass
    return default

def task_frames(sub):
    """Authoritative task-window frames = t_init + task_begin, length task_len
    (per GazeDataLoader). NOT the appleFace glob, which is a superset that also
    contains ~958 inter-task gap frames where GT counted no blinks."""
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
    ap.add_argument("--out", default="/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/events")
    args = ap.parse_args()
    sub = args.subject
    os.makedirs(args.out, exist_ok=True)
    EAR_THR = ear_threshold(sub)
    print(f"sub{sub}: EAR_THR={EAR_THR}", flush=True)

    gt = np.load(open(f"/springbrook/share/eng/esrpxk/datasets/extracted_blinks/{sub:05d}", "rb")).astype(int)
    gt_iv = [(int(s), int(s + d)) for s, d in gt]
    kept = task_frames(sub)
    lo = min(s for s, _ in gt_iv) - MARGIN
    hi = max(e for _, e in gt_iv) + MARGIN

    cap = cv2.VideoCapture(f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4")
    mesh = VideoFaceDetector()
    # Decode sequentially from frame 0 -- do NOT cap.set(POS_FRAMES): for H.264 it
    # lands on the nearest keyframe, mislabeling frames by a per-video constant and
    # creating a spurious per-subject EAR-vs-GT offset. fr is then the TRUE index.
    ear = {}; fr = 1  # 1-indexed to match ProcessedData/appleFace + GT numbering
    while fr <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if fr >= lo and fr in kept:
            lm = mesh.detect(cv2.cvtColor(cv2.rotate(f, ROT), cv2.COLOR_BGR2RGB))
            if lm is not None:
                ear[fr] = min(_ear(lm.left_eye_ear), _ear(lm.right_eye_ear))
        fr += 1
    mesh.close(); cap.release()

    # persist per-frame EAR so re-masking never needs another video pass
    with open(f"{args.out}/subid_{sub}_per_frame_ear.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "ear"])
        for fr in sorted(ear):
            w.writerow([fr, f"{ear[fr]:.4f}"])

    # EAR events: contiguous kept-frame runs with ear < thr
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
        if b < c: return c - b
        if a > d: return a - d
        return 0

    # match EAR events <-> GT events within TOL
    ear_rows = []
    for es, ee in ev:
        mn = min(ear[f] for f in range(es, ee + 1) if f in ear)
        hits = [k for k, (gs, ge) in enumerate(gt_iv) if gap(es, ee, gs, ge) <= TOL]
        ear_rows.append((es, ee, ee - es + 1, mn, hits))
    gt_rows = []
    for gs, ge in gt_iv:
        hits = [k for k, (es, ee) in enumerate(ev) if gap(es, ee, gs, ge) <= TOL]
        seg = [ear[f] for f in range(gs, ge) if f in ear]
        gt_rows.append((gs, ge, ge - gs, min(seg) if seg else None, hits))

    # write CSVs
    with open(f"{args.out}/subid_{sub}_ear_events.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","start","end","dur","min_ear","matches_gt","status"])
        for k,(es,ee,du,mn,hits) in enumerate(ear_rows,1):
            w.writerow([k,es,ee,du,f"{mn:.3f}","|".join(str(h+1) for h in hits),
                        "matched" if hits else "FP_ear_only"])
    with open(f"{args.out}/subid_{sub}_gt_events.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","start","end","dur","min_ear_in_span","matches_ear","status"])
        for k,(gs,ge,du,mn,hits) in enumerate(gt_rows,1):
            w.writerow([k,gs,ge,du,"" if mn is None else f"{mn:.3f}","|".join(str(h+1) for h in hits),
                        "detected" if hits else ("FN_missed" if mn is not None else "FN_no_taskframe")])

    # readable txt
    with open(f"{args.out}/subid_{sub}_events.txt", "w") as f:
        f.write(f"Subject {sub}: {len(ev)} EAR events, {len(gt_iv)} GT events "
                f"(EAR_THR={EAR_THR}, match tol={TOL}f, task-frame masked)\n\n")
        f.write("=== EAR EVENTS (EAR<0.20 runs) ===\n")
        f.write(f"{'#':>3} {'start':>6} {'end':>6} {'dur':>4} {'minEAR':>7}  match->GT\n")
        for k,(es,ee,du,mn,hits) in enumerate(ear_rows,1):
            tag = "GT#" + ",".join(str(h+1) for h in hits) if hits else "-- FP (no GT) --"
            f.write(f"{k:>3} {es:>6} {ee:>6} {du:>4} {mn:>7.3f}  {tag}\n")
        f.write("\n=== GT EVENTS (recorded blinks) ===\n")
        f.write(f"{'#':>3} {'start':>6} {'end':>6} {'dur':>4} {'minEAR':>7}  detected?\n")
        for k,(gs,ge,du,mn,hits) in enumerate(gt_rows,1):
            tag = "EAR#" + ",".join(str(h+1) for h in hits) if hits else "-- MISSED --"
            me = "  n/a" if mn is None else f"{mn:>7.3f}"
            f.write(f"{k:>3} {gs:>6} {ge:>6} {du:>4} {me}  {tag}\n")

    nfp = sum(1 for *_, h in ear_rows if not h)
    nfn = sum(1 for *_, h in gt_rows if not h)
    print(f"sub{sub}: EAR_events={len(ev)} GT_events={len(gt_iv)} "
          f"FP={nfp} FN={nfn} -> {args.out}/subid_{sub}_events.txt", flush=True)

if __name__ == "__main__":
    main()
