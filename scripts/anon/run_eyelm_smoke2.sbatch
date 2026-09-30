#!/bin/bash
#SBATCH --job-name=eyelm2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=6
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/eyelm2_%j.log
set -uo pipefail
export OMP_NUM_THREADS=2
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
L=/springbrook/share/eng/esrpxk/datasets/anon_eye_landmarks
mkdir -p "$L"
for REC in 00006 00013 00020; do
  $PY scripts/anon/eye_landmarks.py --rec "$REC" --limit 200 --out "$L/${REC}.npz" &
done
wait
echo "--- P1/P2 生成 smoke ---"
for REC in 00006; do
  for LV in P1 P2; do
    $PY scripts/anon/make_anon_eyes.py --rec "$REC" --level $LV --limit 40 \
        --dst-root /springbrook/share/eng/esrpxk/datasets/anon_paxis_smoke/$LV
  done
done
echo "--- 验证保留区域 ---"
$PY - <<'EOF'
import cv2, numpy as np, glob, os
base='/springbrook/share/eng/esrpxk/datasets'
for lv in ('P1','P2'):
    ps=sorted(glob.glob(f'{base}/anon_paxis_smoke/{lv}/00006/appleLeftEye/*.jpg'))[:8]
    if not ps: print(f'{lv}: MISSING'); continue
    kept=[];mae=[]
    for p in ps:
        n=os.path.basename(p)
        a=cv2.imread(f'{base}/ProcessedData/00006/appleLeftEye/{n}').astype(np.float32)
        b=cv2.imread(p).astype(np.float32)
        d=np.abs(a-b).mean(2)
        kept.append((d<3).mean()*100); mae.append(d.mean())
    print(f'{lv}: 眼图中与原图一致的像素占比 {np.mean(kept):5.1f}%  整体MAE {np.mean(mae):6.2f}')
EOF
