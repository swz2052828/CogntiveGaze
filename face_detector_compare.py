"""Compare the three video_preprocess face detectors on the calibration window.

For each subject we read frames [calib_begins, calib_begins+WINDOW) from
datasets/videos/subid_<id>.mp4 and run all three detectors from
video_preprocess.face_detectors:

  mediapipe_facemesh    bbox from FaceMesh face_oval (478-landmark mesh)
  mediapipe_facedetect  BlazeFace bounding box
  opencv_haar           Viola-Jones Haar cascade

calib_begins comes from Experiment 2/code/GazeDataLoader_swz.py.

Outputs (under --out-root):
  per_frame/subid_<id>.csv   one row per (frame, detector): found, bbox
  samples/subid_<id>/*.jpg   a few frames with all three boxes overlaid
  summary.csv                per-(subject,detector) detection rate / speed / size / jitter
  agreement.csv              per-subject pairwise mean IoU between detectors
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# --- shim so the code's `from mediapipe.solutions.face_X import ...` resolves
# on mediapipe builds that expose solutions only as mediapipe.python.solutions.
import mediapipe  # noqa: E402
import mediapipe.python.solutions as _sol  # noqa: E402

sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
sys.modules.setdefault("mediapipe.solutions.face_detection", _sol.face_detection)

import cv2  # noqa: E402

from video_preprocess.face_detectors import build_face_detector  # noqa: E402
from video_preprocess.detector import VideoFaceDetector  # noqa: E402


# subject -> calib_begins, from GazeDataLoader_swz.py
SUB_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB_BEGINS = [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
                1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]
CALIB = dict(zip(SUB_IDS, CALIB_BEGINS))

DETECTORS = ["mediapipe_facemesh", "mediapipe_facedetect", "opencv_haar"]
COLORS = {  # BGR for cv2 overlay
    "mediapipe_facemesh": (0, 255, 0),
    "mediapipe_facedetect": (0, 165, 255),
    "opencv_haar": (255, 0, 0),
}


def iou(a, b):
    if a is None or b is None:
        return None
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


ROT_CODE = None  # set in main() from --rotate; cv2 rotate code or None


def process_subject(sub_id, videos_dir, out_root, window, sample_stride):
    video_path = Path(videos_dir) / f"subid_{sub_id}.mp4"
    if not video_path.exists():
        print(f"[sub {sub_id}] MISSING video {video_path}", flush=True)
        return []

    begin = CALIB[sub_id]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[sub {sub_id}] could not open {video_path}", flush=True)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)

    # FaceMesh session shared (tracking mode) — facemesh detector consumes its mesh
    mesh_session = VideoFaceDetector()
    dets = {name: build_face_detector(name) for name in DETECTORS}

    per_frame_dir = out_root / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_root / "samples" / f"subid_{sub_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_frame_dir / f"subid_{sub_id}.csv"
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["frame", "detector", "found", "x0", "y0", "x1", "y1", "ms"])

    # accumulators
    stats = {name: dict(found=0, n=0, time=0.0, areas=[], widths=[], heights=[],
                        prev_center=None, jitter=[]) for name in DETECTORS}
    iou_acc = {("mediapipe_facemesh", "mediapipe_facedetect"): [],
               ("mediapipe_facemesh", "opencv_haar"): [],
               ("mediapipe_facedetect", "opencv_haar"): []}

    read = 0
    t_start = time.perf_counter()
    while read < window:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if ROT_CODE is not None:
            frame_bgr = cv2.rotate(frame_bgr, ROT_CODE)
        frame_idx = begin + read
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t_mesh = time.perf_counter()
        mesh_landmarks = mesh_session.detect(frame_rgb)
        mesh_ms = (time.perf_counter() - t_mesh) * 1000.0
        boxes = {}
        for name in DETECTORS:
            t0 = time.perf_counter()
            bbox = dets[name].detect(frame_rgb, mesh_landmarks)
            ms = (time.perf_counter() - t0) * 1000.0
            # facemesh's real cost is the shared FaceMesh inference; attribute it
            if name == "mediapipe_facemesh":
                ms += mesh_ms
            stats[name]["n"] += 1
            stats[name]["time"] += ms
            boxes[name] = bbox
            if bbox is not None:
                stats[name]["found"] += 1
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                stats[name]["areas"].append(w * h)
                stats[name]["widths"].append(w)
                stats[name]["heights"].append(h)
                c = center(bbox)
                if stats[name]["prev_center"] is not None:
                    pc = stats[name]["prev_center"]
                    stats[name]["jitter"].append(
                        float(np.hypot(c[0] - pc[0], c[1] - pc[1])))
                stats[name]["prev_center"] = c
                writer.writerow([frame_idx, name, 1, f"{bbox[0]:.1f}", f"{bbox[1]:.1f}",
                                 f"{bbox[2]:.1f}", f"{bbox[3]:.1f}", f"{ms:.2f}"])
            else:
                stats[name]["prev_center"] = None
                writer.writerow([frame_idx, name, 0, "", "", "", "", f"{ms:.2f}"])

        for (a, b) in iou_acc:
            v = iou(boxes[a], boxes[b])
            if v is not None:
                iou_acc[(a, b)].append(v)

        if read % sample_stride == 0:
            vis = frame_bgr.copy()
            for name in DETECTORS:
                bb = boxes[name]
                if bb is not None:
                    x0, y0, x1, y1 = (int(round(v)) for v in bb)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), COLORS[name], 2)
                    cv2.putText(vis, name, (x0, max(15, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[name], 1, cv2.LINE_AA)
            cv2.imwrite(str(sample_dir / f"{frame_idx:06d}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        read += 1
        if read % 200 == 0:
            el = time.perf_counter() - t_start
            print(f"[sub {sub_id}] {read}/{window} frames  {read/el:.1f} fps", flush=True)

    fh.close()
    mesh_session.close()
    for d in dets.values():
        d.close()
    cap.release()

    rows = []
    for name in DETECTORS:
        s = stats[name]
        n = max(1, s["n"])
        rows.append(dict(
            sub_id=sub_id, begin=begin, frames=read, detector=name,
            found=s["found"], det_rate=s["found"] / n,
            ms_per_frame=s["time"] / n,
            fps=1000.0 * n / max(1e-6, s["time"]),
            mean_area=float(np.mean(s["areas"])) if s["areas"] else float("nan"),
            mean_w=float(np.mean(s["widths"])) if s["widths"] else float("nan"),
            mean_h=float(np.mean(s["heights"])) if s["heights"] else float("nan"),
            mean_jitter_px=float(np.mean(s["jitter"])) if s["jitter"] else float("nan"),
        ))
    iou_rows = []
    for (a, b), vals in iou_acc.items():
        iou_rows.append(dict(sub_id=sub_id, det_a=a, det_b=b,
                             n_both=len(vals),
                             mean_iou=float(np.mean(vals)) if vals else float("nan")))
    print(f"[sub {sub_id}] DONE frames={read} "
          + " | ".join(f"{r['detector']}={r['det_rate']*100:.1f}%@{r['ms_per_frame']:.1f}ms"
                       for r in rows), flush=True)
    return rows, iou_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/springbrook/share/eng/esrpxk/datasets/videos")
    ap.add_argument("--out-root",
                    default="/springbrook/share/eng/esrpxk/datasets/face_detector_comparison")
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--sample-stride", type=int, default=200,
                    help="save an annotated frame every N frames")
    ap.add_argument("--subjects", default="all",
                    help="comma list of subject ids, or 'all'")
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none",
                    help="rotate frames before detection ('cw' for this dataset)")
    args = ap.parse_args()

    global ROT_CODE
    ROT_CODE = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
                "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[args.rotate]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.subjects == "all":
        subjects = SUB_IDS
    else:
        subjects = [int(x) for x in args.subjects.split(",")]

    all_summary, all_iou = [], []
    for sub_id in subjects:
        res = process_subject(sub_id, args.videos_dir, out_root,
                              args.window, args.sample_stride)
        if res:
            srows, irows = res
            all_summary.extend(srows)
            all_iou.extend(irows)

    sum_path = out_root / "summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
        w.writeheader()
        w.writerows(all_summary)
    iou_path = out_root / "agreement.csv"
    with open(iou_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_iou[0].keys()))
        w.writeheader()
        w.writerows(all_iou)
    print(f"\nWrote {sum_path} and {iou_path}", flush=True)


if __name__ == "__main__":
    main()
