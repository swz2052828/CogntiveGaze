#!/bin/bash
#SBATCH --job-name=anon_s3_smoke
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s3_smoke_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
S=/springbrook/share/eng/esrpxk/datasets/anon_s3_smoke
for OP in blur0.10 blur0.50 pixel0.12 blackbox; do
  $PY scripts/anon/make_anon_frames.py --rec 00006 --op $OP --dst-root "$S/$OP" --limit 200
done
$PY scripts/anon/make_anon_frames.py --rec 00006 --op blur0.50 --dst-root "$S/blur0.50_unmasked" \
    --protect-eyes 0 --limit 200
echo "===== 验证 ====="
$PY - <<'EOF'
import cv2, numpy as np, glob, os
base='/springbrook/share/eng/esrpxk/datasets'
S=base+'/anon_s3_smoke'
for op in ('blur0.10','blur0.50','pixel0.12','blackbox','blur0.50_unmasked'):
    row=[]
    for fld in ('appleFace','appleLeftEye','appleRightEye'):
        ps=sorted(glob.glob(f'{S}/{op}/00006/{fld}/*.jpg'))[:60]
        if not ps: row.append((fld,'MISSING')); continue
        d=[]
        for p in ps:
            n=os.path.basename(p)
            a=cv2.imread(f'{base}/ProcessedData/00006/{fld}/{n}')
            b=cv2.imread(p)
            if a is None or b is None or a.shape!=b.shape: continue
            d.append(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean())
        row.append((fld, f'{np.mean(d):6.2f}' if d else 'n/a'))
    print(f'{op:22s} ' + '  '.join(f'{f}={v}' for f,v in row))
EOF
echo DONE
