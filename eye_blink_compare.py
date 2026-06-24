"""Compare all eye detectors and all blink detectors on the calib windows.

Per subject, on the UPRIGHT (rotate cw) 1200-frame calibration window, run
FaceMesh once per frame and then every eye + blink detector:

  eye   : facemesh_contour, facemesh_iris, opencv_haar
  blink : ear, contour_ratio, iris_visibility

opencv_haar eye runs inside the FaceMesh face bbox ROI. The three blink
detectors all read mesh landmarks directly, so they are independent per-frame.

Outputs under --out-root:
  eye_per_frame/subid_<id>.csv     frame, method, found, L+R boxes
  blink_per_frame/subid_<id>.csv   frame, method, is_blink, score_l, score_r
  samples/subid_<id>/*.jpg         eye boxes (3 methods) + blink flags overlaid
  eye_summary.csv                  per (subject,method): det_rate, box size, jitter
  eye_agreement.csv                per subject: pairwise mean IoU (L & R)
  blink_summary.csv                per (subject,method): blink count + rate
  blink_agreement.csv              per subject: pairwise frame-agreement of flags

Runs on a SLURM compute node (never the login node).
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np

import mediapipe  # noqa
import mediapipe.python.solutions as _sol  # noqa
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
sys.modules.setdefault("mediapipe.solutions.face_detection", _sol.face_detection)

import cv2  # noqa: E402
from video_preprocess.detector import VideoFaceDetector  # noqa: E402
from video_preprocess.bbox import bbox_from_points  # noqa: E402
from video_preprocess.eye_detectors import build_eye_detector  # noqa: E402
from video_preprocess.blink_detectors import build_blink_detector  # noqa: E402

SUB_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB = dict(zip(SUB_IDS,
    [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
     1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]))

EYE_METHODS = ["facemesh_contour", "facemesh_iris", "opencv_haar"]
BLINK_METHODS = ["ear", "contour_ratio", "iris_visibility", "template_match"]
TEMPLATES_DIR = "/springbrook/share/eng/esrpxk/datasets/eye_templates"
EYE_COL = {"facemesh_contour": (0, 255, 0), "facemesh_iris": (255, 255, 0),
           "opencv_haar": (255, 0, 255)}
ROT = cv2.ROTATE_90_CLOCKWISE


def iou(a, b):
    if a is None or b is None:
        return None
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)


def process_subject(sub, videos_dir, out_root, window, sample_stride):
    video = os.path.join(videos_dir, f"subid_{sub}.mp4")
    if not os.path.exists(video):
        print(f"[sub {sub}] MISSING {video}", flush=True)
        return None
    begin = CALIB[sub]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)

    mesh = VideoFaceDetector()
    eyes = {m: build_eye_detector(m) for m in EYE_METHODS}
    blinks = {}
    for m in BLINK_METHODS:
        if m == "template_match":
            blinks[m] = build_blink_detector(m, templates_dir=TEMPLATES_DIR, subject_id=sub)
        else:
            blinks[m] = build_blink_detector(m)

    eye_dir = os.path.join(out_root, "eye_per_frame"); os.makedirs(eye_dir, exist_ok=True)
    blink_dir = os.path.join(out_root, "blink_per_frame"); os.makedirs(blink_dir, exist_ok=True)
    samp_dir = os.path.join(out_root, "samples", f"subid_{sub}"); os.makedirs(samp_dir, exist_ok=True)
    efh = open(os.path.join(eye_dir, f"subid_{sub}.csv"), "w", newline="")
    ew = csv.writer(efh)
    ew.writerow(["frame", "method", "found", "lx0", "ly0", "lx1", "ly1",
                 "rx0", "ry0", "rx1", "ry1"])
    bfh = open(os.path.join(blink_dir, f"subid_{sub}.csv"), "w", newline="")
    bw = csv.writer(bfh)
    bw.writerow(["frame", "method", "is_blink", "score_l", "score_r"])

    eye_stats = {m: dict(found=0, n=0, lareas=[], prevc=None, jit=[]) for m in EYE_METHODS}
    blink_stats = {m: dict(blink=0, n=0) for m in BLINK_METHODS}
    eye_iou = defaultdict(lambda: {"L": [], "R": []})   # pair -> iou lists
    blink_agree = defaultdict(lambda: [0, 0])           # pair -> [agree, total]

    read = 0
    t0 = time.perf_counter()
    while read < window:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, ROT)
        fr = begin + read
        read += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = mesh.detect(rgb)
        face_bbox = bbox_from_points(lm.face_oval) if lm is not None else None

        # --- eye detectors ---
        eye_boxes = {}
        for m in EYE_METHODS:
            res = eyes[m].detect(rgb, face_bbox, lm)
            s = eye_stats[m]; s["n"] += 1
            if res is None:
                eye_boxes[m] = None
                s["prevc"] = None
                ew.writerow([fr, m, 0, "", "", "", "", "", "", "", ""])
            else:
                L, R = res
                eye_boxes[m] = (L, R)
                s["found"] += 1
                s["lareas"].append((L[2]-L[0])*(L[3]-L[1]))
                c = center(L)
                if s["prevc"] is not None:
                    s["jit"].append(float(np.hypot(c[0]-s["prevc"][0], c[1]-s["prevc"][1])))
                s["prevc"] = c
                ew.writerow([fr, m, 1, f"{L[0]:.1f}", f"{L[1]:.1f}", f"{L[2]:.1f}", f"{L[3]:.1f}",
                             f"{R[0]:.1f}", f"{R[1]:.1f}", f"{R[2]:.1f}", f"{R[3]:.1f}"])
        for a, b in combinations(EYE_METHODS, 2):
            if eye_boxes[a] and eye_boxes[b]:
                vL = iou(eye_boxes[a][0], eye_boxes[b][0])
                vR = iou(eye_boxes[a][1], eye_boxes[b][1])
                eye_iou[(a, b)]["L"].append(vL)
                eye_iou[(a, b)]["R"].append(vR)

        # --- blink detectors ---
        # template_match needs the eye boxes; the others use mesh only. Pass the
        # facemesh_contour boxes (fall back to None if that detector missed).
        cb = eye_boxes.get("facemesh_contour")
        L_box = cb[0] if cb else None
        R_box = cb[1] if cb else None
        flags = {}
        for m in BLINK_METHODS:
            is_blink, sl, sr = blinks[m].detect(rgb, L_box, R_box, lm)
            flags[m] = bool(is_blink)
            bs = blink_stats[m]; bs["n"] += 1
            if is_blink:
                bs["blink"] += 1
            bw.writerow([fr, m, int(bool(is_blink)),
                         f"{sl:.4f}" if sl == sl else "",
                         f"{sr:.4f}" if sr == sr else ""])
        for a, b in combinations(BLINK_METHODS, 2):
            ag = blink_agree[(a, b)]
            ag[1] += 1
            if flags[a] == flags[b]:
                ag[0] += 1

        # --- sample overlay ---
        if read % sample_stride == 0 and lm is not None:
            vis = frame.copy()
            for m in EYE_METHODS:
                if eye_boxes[m]:
                    for bb in eye_boxes[m]:
                        x0, y0, x1, y1 = (int(round(v)) for v in bb)
                        cv2.rectangle(vis, (x0, y0), (x1, y1), EYE_COL[m], 2)
            y = 30
            for m in BLINK_METHODS:
                txt = f"{m}: {'BLINK' if flags[m] else 'open'}"
                cv2.putText(vis, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if flags[m] else (200, 200, 200), 2, cv2.LINE_AA)
                y += 32
            for i, m in enumerate(EYE_METHODS):
                cv2.putText(vis, m, (20, y + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            EYE_COL[m], 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(samp_dir, f"{fr:06d}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        if read % 300 == 0:
            print(f"[sub {sub}] {read}/{window}  {read/(time.perf_counter()-t0):.1f} fps", flush=True)

    efh.close(); bfh.close(); mesh.close()
    for d in list(eyes.values()) + list(blinks.values()):
        d.close()
    cap.release()

    eye_rows, blink_rows, eye_ag_rows, blink_ag_rows = [], [], [], []
    for m in EYE_METHODS:
        s = eye_stats[m]; n = max(1, s["n"])
        eye_rows.append(dict(sub_id=sub, method=m, frames=s["n"],
                            det_rate=s["found"]/n,
                            mean_lefteye_area=float(np.mean(s["lareas"])) if s["lareas"] else float("nan"),
                            mean_jitter_px=float(np.mean(s["jit"])) if s["jit"] else float("nan")))
    for m in BLINK_METHODS:
        s = blink_stats[m]; n = max(1, s["n"])
        blink_rows.append(dict(sub_id=sub, method=m, frames=s["n"],
                              blink_frames=s["blink"], blink_rate=s["blink"]/n))
    for (a, b), d in eye_iou.items():
        eye_ag_rows.append(dict(sub_id=sub, method_a=a, method_b=b,
                               n=len(d["L"]),
                               mean_iou_left=float(np.mean(d["L"])) if d["L"] else float("nan"),
                               mean_iou_right=float(np.mean(d["R"])) if d["R"] else float("nan")))
    for (a, b), (ag, tot) in blink_agree.items():
        blink_ag_rows.append(dict(sub_id=sub, method_a=a, method_b=b,
                                 n=tot, agreement=ag/max(1, tot)))
    print(f"[sub {sub}] DONE frames={read} | eye "
          + " ".join(f"{r['method']}={r['det_rate']*100:.0f}%" for r in eye_rows)
          + " | blink "
          + " ".join(f"{r['method']}={r['blink_rate']*100:.1f}%" for r in blink_rows), flush=True)
    return eye_rows, blink_rows, eye_ag_rows, blink_ag_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/springbrook/share/eng/esrpxk/datasets/videos")
    ap.add_argument("--out-root", default="/springbrook/share/eng/esrpxk/datasets/eye_blink_comparison")
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--sample-stride", type=int, default=100)
    ap.add_argument("--subjects", default="all")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    subjects = SUB_IDS if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    EYE, BLINK, EAG, BAG = [], [], [], []
    for sub in subjects:
        res = process_subject(sub, args.videos_dir, args.out_root, args.window, args.sample_stride)
        if res:
            EYE += res[0]; BLINK += res[1]; EAG += res[2]; BAG += res[3]
    write_csv(os.path.join(args.out_root, "eye_summary.csv"), EYE)
    write_csv(os.path.join(args.out_root, "blink_summary.csv"), BLINK)
    write_csv(os.path.join(args.out_root, "eye_agreement.csv"), EAG)
    write_csv(os.path.join(args.out_root, "blink_agreement.csv"), BAG)
    print(f"\nWrote summaries to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
