"""Aggregate the FAMS-vs-SimSwap like-for-like checks (2026-10-16) into one summary.

  A  face-crop A1, every family, ArcFace + FaceNet      results/face_a1_all
  B  model-free pupil preservation, every family         results/iris_preservation/methods.json
  C  FAMS anonymisation-strength sweep: privacy + pupil  results/fams_sweep, iris_preservation/fams_sweep.json
  D  stream isolation: eyes-only / face-only retrained   runs/anon_train/eval/at_{none,swap1,swap2,famsF}_{bb}_*
Each section prints PENDING when its inputs are incomplete. Writes
results/fams_checks/summary.json and SUMMARY.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/springbrook/share/eng/esrpxk")
RES, RUNS = ROOT / "results", ROOT / "runs"
OUT = RES / "fams_checks"
out, md = {}, ["# FAMS vs SimSwap: like-for-like checks", ""]


def j(p):
    p = Path(p)
    return json.load(open(p)) if p.is_file() else None


# ---- A face A1
A_ARMS = ["ProcessedSwap", "ProcessedSwap2", "ProcessedDP2t1_per_subject", "ProcessedDP2t1_per_frame",
          "ProcessedDP2fb_per_subject", "ProcessedDP2fb_per_frame", "ProcessedFAMS_per_subject", "ProcessedFAMS_per_frame"]
rows = []
for a in A_ARMS:
    r = {"arm": a}
    for rec in ("arcface", "facenet"):
        d = j(RES / "face_a1_all" / f"{rec}_{a}.json")
        if d:
            r.update({f"{rec}_rank1": d["rank1"], f"{rec}_tar3": d["TAR@FAR=0.001"], f"{rec}_auc": d["roc_auc"]})
    rows.append(r)
out["A_face_a1"] = rows
md += ["## A. Face-crop A1 (gallery = original photos)", "",
       "| arm | ArcFace Rank-1 | ArcFace TAR@1e-3 | AUC | FaceNet Rank-1 | FaceNet TAR@1e-3 |", "|---|---|---|---|---|---|"]
for r in rows:
    f = lambda k: f"{100 * r[k]:.1f}%" if k in r else "PENDING"
    md.append(f"| {r['arm']} | {f('arcface_rank1')} | {f('arcface_tar3')} | "
              f"{r.get('arcface_auc', float('nan')):.3f} | {f('facenet_rank1')} | {f('facenet_tar3')} |")

# ---- B pupil preservation
B = j(RES / "iris_preservation" / "methods.json")
out["B_iris"] = B
md += ["", "## B. Pupil preservation (identical 4,680 frames)", ""]
if B:
    md += ["| arm | r_u | r_v | gt r x | gt r y | mean abs du |", "|---|---|---|---|---|---|"]
    for k, v in B.items():
        md.append(f"| {k} | {v['r_u']:.3f} | {v['r_v']:.3f} | {v['gt_r_x']:.3f} | {v['gt_r_y']:.3f} | {v['mae_u']:.3f} |")
else:
    md.append("PENDING")

# ---- C sweep
C = []
IS = j(RES / "iris_preservation" / "fams_sweep.json") or {}
for deg in ("1.25", "1.35", "1.5", "1.6", "1.75", "2.0", "3.0"):
    if deg == "1.25":
        src = {"face_arc": RES / "face_a1_all/arcface_ProcessedFAMS_per_frame.json",
               "face_fn": RES / "face_a1_all/facenet_ProcessedFAMS_per_frame.json",
               "t1": RES / "anon_fams/t1_fams_per_frame.json",
               "a1L": RES / "anon_fams/a1_famsper_frame_appleLeftEye.json",
               "periL": RES / "anon_fams/peri_famsper_frame_appleLeftEye.json",
               "ariL": RES / "anon_fams/ari_fams_per_frame_appleLeftEye.json"}
        iris = IS.get("ProcessedFAMS_per_frame")
    else:
        S = RES / "fams_sweep"; T = f"d{deg}"
        src = {"face_arc": S / f"face_a1_arcface_{T}.json", "face_fn": S / f"face_a1_facenet_{T}.json",
               "t1": S / f"t1_{T}.json", "a1L": S / f"a1_{T}_appleLeftEye.json",
               "periL": S / f"peri_{T}_appleLeftEye.json", "ariL": S / f"ari_{T}_appleLeftEye.json"}
        iris = IS.get(f"ProcessedFAMSd{deg}_per_frame")
    d = {k: j(p) for k, p in src.items()}
    row = {"degree": deg, "complete": all(d.values()) and iris is not None}
    if d["t1"]: row["collapsed"] = d["t1"]["det_fail"] / (d["t1"]["det_fail"] + d["t1"]["n_frames"])
    if d["face_arc"]: row.update(face_arc_rank1=d["face_arc"]["rank1"], face_arc_tar3=d["face_arc"]["TAR@FAR=0.001"])
    if d["face_fn"]: row.update(face_fn_rank1=d["face_fn"]["rank1"], face_fn_tar3=d["face_fn"]["TAR@FAR=0.001"])
    if d["t1"]: row.update(face_ari=d["t1"]["ari_k18"])
    if d["a1L"]: row.update(eyeL_a1_tar3=d["a1L"]["TAR@FAR=0.001"])
    if d["periL"]: row.update(eyeL_a2_tar3=d["periL"]["direct_TAR@FAR=0.001"])
    if d["ariL"]: row.update(eyeL_ari=d["ariL"]["ari_point"])
    if iris: row.update(iris_r_u=iris["r_u"], iris_gt_r_x=iris["gt_r_x"], iris_gt_r_y=iris["gt_r_y"])
    C.append(row)
if "ProcessedSwap" in IS:
    C.append({"degree": "SimSwap A (reference)", "complete": True, "iris_r_u": IS["ProcessedSwap"]["r_u"],
              "iris_gt_r_x": IS["ProcessedSwap"]["gt_r_x"], "iris_gt_r_y": IS["ProcessedSwap"]["gt_r_y"]})
out["C_sweep"] = C
md += ["", "## C. FAMS strength sweep (260 frames/participant, per-frame noise)", "",
       "| degree | collapsed | face R1 Arc | face TAR Arc | face R1 FaceNet | face ARI | eyeL A1 TAR | eyeL A2 TAR | eyeL ARI | pupil r_u | pupil gt r x | complete |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in C:
    g = lambda k, pct=False: ("—" if r.get(k) is None else (f"{100 * r[k]:.1f}%" if pct else f"{r[k]:.3f}"))
    md.append(f"| {r['degree']} | {g('collapsed', 1)} | {g('face_arc_rank1', 1)} | {g('face_arc_tar3', 1)} | {g('face_fn_rank1', 1)} | "
              f"{g('face_ari')} | {g('eyeL_a1_tar3', 1)} | {g('eyeL_a2_tar3', 1)} | {g('eyeL_ari')} | "
              f"{g('iris_r_u')} | {g('iris_gt_r_x')} | {r['complete']} |")


# ---- D stream isolation
def fold_mean(p):
    if not p.is_file(): return None
    d = pd.read_csv(p); d = d[pd.to_numeric(d["fold"], errors="coerce").notna()].copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates("fold", keep="last")
    return float(d["base"].mean()) if sorted(d["fold"].astype(int)) == [0, 1, 2, 3, 4] else None


E = RUNS / "anon_train" / "eval"
D = {}
md += ["", "## D. Stream isolation (retrained; penalty vs same-backbone real-data model, paired within draw)", ""]
md += ["| backbone | real-data error | SimSwap A test real | SimSwap B test real | FAMS test real | SimSwap A test anon | SimSwap B test anon | FAMS test anon | draws |",
       "|---|---|---|---|---|---|---|---|---|"]
for bb in ("eyes_only_mobilenet_v3", "eyes_only_mgazenet", "face_only_mobile_vit"):
    cell, nd = {}, []
    base = [fold_mean(E / f"at_none_{bb}_testreal{'' if r == 1 else f'_rep{r}'}.csv") for r in range(1, 6)]
    for arm in ("swap1", "swap2", "famsF"):
        for t in ("real", "anon"):
            pen = []
            for r in range(1, 6):
                a = fold_mean(E / f"at_{arm}_{bb}_test{t}{'' if r == 1 else f'_rep{r}'}.csv")
                if a is not None and base[r - 1] is not None: pen.append(a - base[r - 1])
            cell[f"{arm}_{t}"] = dict(mean=float(np.mean(pen)) if pen else None,
                                      sd=float(np.std(pen, ddof=1)) if len(pen) > 1 else None, n=len(pen), draws=pen)
            nd.append(len(pen))
    b = [x for x in base if x is not None]
    D[bb] = dict(baseline=float(np.mean(b)) if b else None, n_base=len(b), **cell)
    c = lambda k: ("PENDING" if cell[k]["mean"] is None else f"{cell[k]['mean']:+.2f}" + (f" ± {cell[k]['sd']:.2f}" if cell[k]["sd"] else ""))
    md.append(f"| {bb} | {D[bb]['baseline'] if D[bb]['baseline'] is None else round(D[bb]['baseline'], 2)} | "
              f"{c('swap1_real')} | {c('swap2_real')} | {c('famsF_real')} | {c('swap1_anon')} | {c('swap2_anon')} | "
              f"{c('famsF_anon')} | {min(nd)}–{max(nd)} |")
out["D_streams"] = D
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "summary.json", "w"), indent=2)
open(OUT / "SUMMARY.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
