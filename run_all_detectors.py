"""Run ALL detectors on a subject's full video, record disagreements, save marked
overlay examples at reasonable intervals.

Per frame (upright, rotate cw), FaceMesh runs once and feeds the mesh-based
detectors; then:
  face  : mediapipe_facemesh, mediapipe_facedetect, opencv_haar
  eye   : facemesh_contour, facemesh_iris, opencv_haar
  blink : ear, contour_ratio, iris_visibility, template_match (per-subject tmpl)

Disagreement definitions (meaningful, convention-aware):
  face  : the 3 detectors don't all agree on found/not-found, OR (all found but
          their box centres span > 12% of the frame diagonal -> someone is on
          the wrong thing).
  eye   : same idea on the left-eye centre across the 3 eye detectors.
  blink : the 4 blink flags are not unanimous.

Outputs under <out-root>/subid_<id>/:
  face_disagreements.csv, eye_disagreements.csv, blink_disagreements.csv
  overlays/face/<frame>.jpg, overlays/eye/<frame>.jpg, overlays/blink/<frame>.jpg
    (marked with every method's result; saved >= --save-interval frames apart
     per category, capped at --max-examples per category)

Runs on a SLURM compute node (array: one task per subject).
"""
import argparse, csv, os, sys
import numpy as np
import mediapipe, mediapipe.python.solutions as _sol
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
sys.modules.setdefault("mediapipe.solutions.face_detection", _sol.face_detection)
import cv2
from video_preprocess.detector import VideoFaceDetector
from video_preprocess.bbox import bbox_from_points
from video_preprocess.face_detectors import build_face_detector
from video_preprocess.eye_detectors import build_eye_detector
from video_preprocess.blink_detectors import build_blink_detector

TEMPLATES = "/springbrook/share/eng/esrpxk/datasets/eye_templates"
ROT = cv2.ROTATE_90_CLOCKWISE
FACE_M = ["mediapipe_facemesh", "mediapipe_facedetect", "opencv_haar"]
EYE_M = ["facemesh_contour", "facemesh_iris", "opencv_haar"]
BLINK_M = ["ear", "contour_ratio", "iris_visibility", "template_match"]
FACE_COL = {"mediapipe_facemesh": (0, 255, 0), "mediapipe_facedetect": (0, 165, 255), "opencv_haar": (255, 0, 0)}
EYE_COL = {"facemesh_contour": (0, 255, 0), "facemesh_iris": (255, 255, 0), "opencv_haar": (255, 0, 255)}


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def max_center_spread(boxes):
    cs = [center(b) for b in boxes if b is not None]
    if len(cs) < 2:
        return 0.0
    return max(np.hypot(a[0] - b[0], a[1] - b[1]) for a in cs for b in cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out-root", default="/springbrook/share/eng/esrpxk/datasets/all_detectors")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--save-interval", type=int, default=150,
                    help="min frames between saved overlay examples per category")
    ap.add_argument("--max-examples", type=int, default=120, help="cap saved overlays per category")
    args = ap.parse_args()

    sub = args.subject
    video = f"/springbrook/share/eng/esrpxk/datasets/videos/subid_{sub}.mp4"
    base = os.path.join(args.out_root, f"subid_{sub}")
    for c in ("face", "eye", "blink"):
        os.makedirs(os.path.join(base, "overlays", c), exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[sub {sub}] cannot open {video}", flush=True); return
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diag = np.hypot(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # rotated swaps but ~same
    spread_thr = 0.12 * diag

    mesh = VideoFaceDetector()
    faces = {m: build_face_detector(m) for m in FACE_M}
    eyes = {m: build_eye_detector(m) for m in EYE_M}
    blinks = {}
    for m in BLINK_M:
        blinks[m] = (build_blink_detector(m, templates_dir=TEMPLATES, subject_id=sub)
                     if m == "template_match" else build_blink_detector(m))

    fcsv = open(os.path.join(base, "face_disagreements.csv"), "w", newline=""); fw = csv.writer(fcsv)
    fw.writerow(["frame", "n_found"] + [f"{m}_found" for m in FACE_M] + ["center_spread_px", "reason"])
    ecsv = open(os.path.join(base, "eye_disagreements.csv"), "w", newline=""); ew = csv.writer(ecsv)
    ew.writerow(["frame", "n_found"] + [f"{m}_found" for m in EYE_M] + ["L_center_spread_px", "reason"])
    bcsv = open(os.path.join(base, "blink_disagreements.csv"), "w", newline=""); bw = csv.writer(bcsv)
    bw.writerow(["frame"] + [f"{m}" for m in BLINK_M] + ["n_blink_votes"])

    last_saved = {"face": -10**9, "eye": -10**9, "blink": -10**9}
    saved = {"face": 0, "eye": 0, "blink": 0}
    counts = {"face": 0, "eye": 0, "blink": 0}
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = fr
        fr += 1
        if idx % args.stride:
            continue
        img = cv2.rotate(frame, ROT)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm = mesh.detect(rgb)
        face_bbox = bbox_from_points(lm.face_oval) if lm is not None else None

        # face detectors
        fboxes = {m: faces[m].detect(rgb, lm) for m in FACE_M}
        found = [fboxes[m] is not None for m in FACE_M]
        nf = sum(found)
        present = [fboxes[m] for m in FACE_M if fboxes[m] is not None]
        spread = max_center_spread(present)
        face_dis = (0 < nf < 3) or (nf == 3 and spread > spread_thr)
        if face_dis:
            counts["face"] += 1
            reason = "detection_mismatch" if 0 < nf < 3 else "box_location_disagree"
            fw.writerow([idx, nf] + [int(x) for x in found] + [f"{spread:.0f}", reason])

        # eye detectors
        eboxes = {m: eyes[m].detect(rgb, face_bbox, lm) for m in EYE_M}
        efound = [eboxes[m] is not None for m in EYE_M]
        nef = sum(efound)
        lefts = [eboxes[m][0] for m in EYE_M if eboxes[m] is not None]
        espread = max_center_spread(lefts)
        eye_dis = (0 < nef < 3) or (nef == 3 and espread > spread_thr)
        if eye_dis:
            counts["eye"] += 1
            reason = "detection_mismatch" if 0 < nef < 3 else "box_location_disagree"
            ew.writerow([idx, nef] + [int(x) for x in efound] + [f"{espread:.0f}", reason])

        # blink detectors (template_match needs eye boxes -> facemesh_contour)
        cb = eboxes.get("facemesh_contour")
        Lb = cb[0] if cb else None
        Rb = cb[1] if cb else None
        bflags = {}
        for m in BLINK_M:
            isb, _, _ = blinks[m].detect(rgb, Lb, Rb, lm)
            bflags[m] = int(bool(isb))
        votes = sum(bflags.values())
        blink_dis = 0 < votes < len(BLINK_M)
        if blink_dis:
            counts["blink"] += 1
            bw.writerow([idx] + [bflags[m] for m in BLINK_M] + [votes])

        # save marked overlays at intervals
        def save_face():
            vis = img.copy()
            for m in FACE_M:
                b = fboxes[m]
                if b is not None:
                    x0, y0, x1, y1 = (int(round(v)) for v in b)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), FACE_COL[m], 2)
                    cv2.putText(vis, m, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FACE_COL[m], 1, cv2.LINE_AA)
                else:
                    cv2.putText(vis, f"{m}: NONE", (10, 25 + 22 * FACE_M.index(m)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FACE_COL[m], 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(base, "overlays/face", f"{idx:06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        def save_eye():
            vis = img.copy()
            for m in EYE_M:
                bx = eboxes[m]
                if bx is not None:
                    for bb in bx:
                        x0, y0, x1, y1 = (int(round(v)) for v in bb)
                        cv2.rectangle(vis, (x0, y0), (x1, y1), EYE_COL[m], 2)
                    cv2.putText(vis, m, (10, 25 + 22 * EYE_M.index(m)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_COL[m], 2, cv2.LINE_AA)
                else:
                    cv2.putText(vis, f"{m}: NONE", (10, 25 + 22 * EYE_M.index(m)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_COL[m], 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(base, "overlays/eye", f"{idx:06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        def save_blink():
            vis = img.copy()
            if cb:
                for bb in (Lb, Rb):
                    x0, y0, x1, y1 = (int(round(v)) for v in bb)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 200), 2)
            y = 30
            for m in BLINK_M:
                col = (0, 0, 255) if bflags[m] else (180, 220, 180)
                cv2.putText(vis, f"{m}: {'BLINK' if bflags[m] else 'open'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                y += 30
            cv2.imwrite(os.path.join(base, "overlays/blink", f"{idx:06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        for cat, dis, saver in (("face", face_dis, save_face), ("eye", eye_dis, save_eye), ("blink", blink_dis, save_blink)):
            if dis and saved[cat] < args.max_examples and (idx - last_saved[cat]) >= args.save_interval:
                saver(); last_saved[cat] = idx; saved[cat] += 1

        if idx % 4000 == 0:
            print(f"[sub {sub}] frame {idx}/{nframes} disagreements face={counts['face']} eye={counts['eye']} blink={counts['blink']}", flush=True)

    fcsv.close(); ecsv.close(); bcsv.close(); mesh.close()
    for d in list(faces.values()) + list(eyes.values()) + list(blinks.values()):
        d.close()
    cap.release()
    # summary
    with open(os.path.join(base, "SUMMARY.txt"), "w") as f:
        f.write(f"subject {sub}: {nframes} frames, stride {args.stride}\n")
        for cat in ("face", "eye", "blink"):
            f.write(f"  {cat}: {counts[cat]} disagreement frames, {saved[cat]} overlay examples saved\n")
    print(f"[sub {sub}] DONE  face={counts['face']} eye={counts['eye']} blink={counts['blink']} "
          f"(examples saved: face={saved['face']} eye={saved['eye']} blink={saved['blink']})", flush=True)


if __name__ == "__main__":
    main()
