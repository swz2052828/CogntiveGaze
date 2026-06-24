"""Full diagnosis of the opencv_haar face detector on the calibration windows.

Ground truth = the FaceMesh face_oval box (≥99% reliable), loaded from the
existing per_frame CSVs, so we never re-run FaceMesh.

Key fairness point: FaceMesh's face_oval box (forehead->chin, full width) is
~2x larger than Haar's tight frontal-face box, so a perfect Haar-on-face hit
only reaches IoU~0.28. We therefore classify each Haar *candidate* by where it
lands, not by raw IoU:

  ON_FACE      center inside GT box AND box width >= 0.30 * GT width
  PARTIAL      center inside GT box but box too small  (<0.30 GT width)
  BACKGROUND   center outside GT box                   (true false positive)

Two passes:
  A) Candidate diagnosis at the repo-default config (default cascade, gray,
     scaleFactor=1.1, minNeighbors=5, minSize=60): how many candidates per
     frame, is the face among them, does the 'pick largest' heuristic select it.
  B) Sweep over {cascade} x {preproc} x {scaleFactor} x {minNeighbors} x
     {minSize}: selected-box recall + oracle recall + background-FP rate, to see
     if Haar can be salvaged and with what settings.
"""

import argparse
import csv
import glob
import os
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = "/springbrook/share/eng/esrpxk/datasets"
PF = f"{ROOT}/face_detector_comparison/per_frame"
VIDEOS = f"{ROOT}/videos"

SUB_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB = dict(zip(SUB_IDS,
    [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
     1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]))

CASC_DIR = cv2.data.haarcascades
CASCADES = {
    "default": "haarcascade_frontalface_default.xml",
    "alt": "haarcascade_frontalface_alt.xml",
    "alt2": "haarcascade_frontalface_alt2.xml",
}


def load_gt(sub):
    """frame -> facemesh box (x0,y0,x1,y1)."""
    gt = {}
    for r in csv.DictReader(open(f"{PF}/subid_{sub}.csv")):
        if r["detector"] == "mediapipe_facemesh" and r["found"] == "1":
            gt[int(r["frame"])] = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
    return gt


def classify(cand, gt):
    """Return 'on_face' | 'partial' | 'background' for one candidate vs GT."""
    cx = (cand[0] + cand[2]) / 2.0
    cy = (cand[1] + cand[3]) / 2.0
    inside = gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]
    if not inside:
        return "background"
    cw = cand[2] - cand[0]
    gw = gt[2] - gt[0]
    return "on_face" if cw >= 0.30 * gw else "partial"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rects_to_boxes(rects):
    return [(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in rects]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=20,
                    help="sample every Nth frame of the 1200-frame window")
    ap.add_argument("--out", default=f"{ROOT}/face_detector_comparison/haar_diagnosis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cascades = {k: cv2.CascadeClassifier(CASC_DIR + v) for k, v in CASCADES.items()}

    # ---- sweep grid ----
    sweep_cascades = ["default", "alt", "alt2"]
    sweep_preproc = ["gray", "equalized"]
    sweep_scale = [1.05, 1.1]
    sweep_neighbors = [3, 5]
    sweep_minsize = [60, 150, 250]

    # Pass A accumulators (repo default config)
    A = defaultdict(int)
    A_ncand = []
    A_best_iou = []
    # Pass B accumulators: config -> counters
    B = defaultdict(lambda: dict(gt=0, sel_hit=0, oracle_hit=0, sel_bg=0,
                                 ncand=0, frames=0, time=0.0, ious=[]))

    t0 = time.perf_counter()
    for sub in SUB_IDS:
        gt_all = load_gt(sub)
        begin = CALIB[sub]
        cap = cv2.VideoCapture(f"{VIDEOS}/subid_{sub}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
        read = 0
        while read < 1200:
            ok, frame = cap.read()
            if not ok:
                break
            fr = begin + read
            do = (read % args.stride == 0)
            read += 1
            if not do:
                continue
            gt = gt_all.get(fr)
            if gt is None:
                continue  # no ground-truth face this frame; skip
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            imgs = {"gray": gray, "equalized": eq}

            # ---- Pass A: repo default ----
            rects = cascades["default"].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            cands = rects_to_boxes(rects)
            A["gt_frames"] += 1
            A_ncand.append(len(cands))
            if cands:
                cls = [classify(c, gt) for c in cands]
                largest = max(cands, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                largest_cls = classify(largest, gt)
                A[f"sel_{largest_cls}"] += 1          # what the heuristic picked
                if any(c in ("on_face", "partial") for c in cls):
                    A["oracle_face_present"] += 1      # face was among candidates
                best = max((iou(c, gt) for c in cands), default=0.0)
                A_best_iou.append(best)
            else:
                A["no_candidates"] += 1

            # ---- Pass B: sweep ----
            for cname in sweep_cascades:
                casc = cascades[cname]
                for pp in sweep_preproc:
                    img = imgs[pp]
                    for sf in sweep_scale:
                        for mn in sweep_neighbors:
                            for ms in sweep_minsize:
                                key = (cname, pp, sf, mn, ms)
                                st = time.perf_counter()
                                rr = casc.detectMultiScale(
                                    img, scaleFactor=sf, minNeighbors=mn,
                                    minSize=(ms, ms))
                                el = (time.perf_counter() - st) * 1000.0
                                cc = rects_to_boxes(rr)
                                d = B[key]
                                d["gt"] += 1
                                d["frames"] += 1
                                d["time"] += el
                                d["ncand"] += len(cc)
                                if cc:
                                    classes = [classify(c, gt) for c in cc]
                                    if any(c in ("on_face", "partial") for c in classes):
                                        d["oracle_hit"] += 1
                                    largest = max(cc, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                                    lc = classify(largest, gt)
                                    if lc in ("on_face", "partial"):
                                        d["sel_hit"] += 1
                                        d["ious"].append(iou(largest, gt))
                                    else:
                                        d["sel_bg"] += 1
        cap.release()
        print(f"[sub {sub}] sampled, elapsed {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- write Pass A ----
    gtf = max(1, A["gt_frames"])
    with open(f"{args.out}/passA_default.txt", "w") as f:
        def p(*a):
            line = " ".join(str(x) for x in a)
            print(line); f.write(line + "\n")
        p("=" * 70)
        p("PASS A — repo default (default cascade, gray, sf=1.1, mn=5, minSize=60)")
        p("=" * 70)
        p(f"GT frames sampled           : {gtf}")
        p(f"mean candidates / frame     : {np.mean(A_ncand):.2f}")
        p(f"frames with 0 candidates    : {A['no_candidates']} ({100*A['no_candidates']/gtf:.1f}%)")
        p(f"face present among candidates: {A['oracle_face_present']} ({100*A['oracle_face_present']/gtf:.1f}%)  <- oracle recall")
        p("")
        p("What the 'pick largest' heuristic actually selected:")
        for k in ("sel_on_face", "sel_partial", "sel_background"):
            p(f"  {k:18}: {A[k]:5d} ({100*A[k]/gtf:.1f}%)")
        p(f"  {'no_candidates':18}: {A['no_candidates']:5d} ({100*A['no_candidates']/gtf:.1f}%)")
        p(f"mean best-candidate IoU vs GT (cap ~0.28): {np.mean(A_best_iou) if A_best_iou else 0:.3f}")

    # ---- write Pass B ----
    rows = []
    for key, d in B.items():
        gt = max(1, d["gt"])
        rows.append(dict(
            cascade=key[0], preproc=key[1], scaleFactor=key[2],
            minNeighbors=key[3], minSize=key[4],
            sel_recall=d["sel_hit"] / gt,
            oracle_recall=d["oracle_hit"] / gt,
            bg_fp_rate=d["sel_bg"] / gt,
            mean_cand=d["ncand"] / gt,
            ms_per_frame=d["time"] / max(1, d["frames"]),
            mean_iou=float(np.mean(d["ious"])) if d["ious"] else float("nan"),
        ))
    rows.sort(key=lambda r: -r["sel_recall"])
    with open(f"{args.out}/passB_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nTOP 8 configs by selected-box recall:")
    hdr = f"{'cascade':8}{'preproc':10}{'sf':>5}{'mn':>3}{'minSz':>6}{'sel_rec':>8}{'oracle':>8}{'bg_fp':>7}{'cand':>6}{'ms':>6}"
    print(hdr)
    for r in rows[:8]:
        print(f"{r['cascade']:8}{r['preproc']:10}{r['scaleFactor']:>5}{r['minNeighbors']:>3}"
              f"{r['minSize']:>6}{r['sel_recall']*100:>7.1f}{r['oracle_recall']*100:>7.1f}"
              f"{r['bg_fp_rate']*100:>6.1f}{r['mean_cand']:>6.1f}{r['ms_per_frame']:>6.0f}")
    print(f"\nWrote {args.out}/passA_default.txt and passB_sweep.csv")


if __name__ == "__main__":
    main()
