#!/bin/bash
# DeepPrivacy2's FULL-BODY pipeline, against the face-mode arms already run.
#
# Why. DP2's own paper is about full-body anonymisation; its stated motivation
# is that face anonymisation "does not prevent recognition through identifiers
# outside the face". Our DP2 arms used the face mode only. A reviewer will ask
# for the strongest mode. Full-body repaints hair, ears, neck and clothing --
# exactly the channels we credit with face-crop linkage -- so it is the direct
# test of condition (b): can widening the edit domain to the whole person break
# linkage, or do background / pose / geometry / illumination suffice?
#
# FB_cse_mask_face with person_G=None (styleganL_nocse is 410 Gone upstream).
# Per frame, the report records which detection path fired; persons detected
# without CSE are left UNTOUCHED and listed as such.
#
# Same frame sample (260/participant, rng seed 0), same truncation (1.0), same two
# identity regimes, same attacks and null control as run_dp2_trunc1.sbatch.
#SBATCH --job-name=dp2_fb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_fb_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
export FVCORE_CACHE=/springbrook/share/eng/esrpxk/home_caches/fvcore
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_dp2/fullbody
mkdir -p "$R"
PY_DP2=/springbrook/share/eng/esrpxk/envs/dp2/bin/python
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
ARC_LIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')

for M in per_subject per_frame; do
  echo "########## generate FULL-BODY identity=$M truncation=1.0"
  $PY_DP2 scripts/anon/run_deepprivacy2.py --mode fullbody --identity "$M" --truncation 1.0 \
      --out-root "$D/ProcessedDP2fb_$M" --n-per-subject 260 --resume \
      --report "$R/edit_domain_fb_$M.json" 2>&1 | grep -v "Calling face detector" \
      || { echo "FAILED gen $M"; exit 1; }
  [ -s "$R/edit_domain_fb_$M.json" ] || { echo "FAILED gen $M (no report)"; exit 1; }
done

echo "########## NULL CONTROLS"
$PY - <<'PYEOF' || exit 1
import cv2, numpy as np, glob, os, sys
D='/springbrook/share/eng/esrpxk/datasets'
A=f'{D}/ProcessedDP2fb_per_subject'; B=f'{D}/ProcessedDP2fb_per_frame'
F=f'{D}/ProcessedDP2t1_per_subject'; O=f'{D}/ProcessedData'
def md(x, y):
    d=[]
    for rec in sorted(os.listdir(x))[:5]:
        for f in sorted(glob.glob(f'{x}/{rec}/appleFace/*.jpg'))[:12]:
            g=f.replace(x,y)
            if not os.path.exists(g): continue
            a,b=cv2.imread(f),cv2.imread(g)
            if a.shape!=b.shape: continue
            d.append(np.abs(a.astype(int)-b.astype(int)).mean())
    return float(np.mean(d)), len(d)
for name,(x,y) in {'fb per_subject vs per_frame':(A,B),
                   'fb per_subject vs face-mode per_subject':(A,F),
                   'fb per_subject vs original':(A,O)}.items():
    m,n=md(x,y); print(f'  {name}: mean |delta| = {m:.3f} over {n} face crops')
    if m < 1.0:
        print('  FATAL: arms are (near-)identical; refusing to run the attacks'); sys.exit(1)
print('  PASS')
PYEOF

for M in per_subject per_frame; do
  for F in appleLeftEye appleRightEye; do
    echo "########## periocular A2  FB_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/periocular_attack.py --ctx 0 \
        --root "$D/ProcessedDP2fb_$M" --folder "$F" --tag "dp2fb${M}_$F" \
        --emb-out "$R/emb_dp2fb${M}_$F.npz" \
        --json-out "$R/peri_dp2fb${M}_$F.json" || echo "FAILED peri $M $F"
    echo "########## A1 cross-domain  FB_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/id_attack.py --ctx 0 --no-detect \
        --gallery-root "$D/ProcessedData" --probe-root "$D/ProcessedDP2fb_$M" \
        --folder "$F" --n-gallery 60 --n-probe 60 --tag "a1_dp2fb${M}_$F" \
        --json-out "$R/a1_dp2fb${M}_$F.json" || echo "FAILED a1 $M $F"
    echo "########## ARI bootstrap  FB_$M  $F"
    $PY scripts/anon/ari_bootstrap.py "$R/emb_dp2fb${M}_$F.npz" --tag "dp2fb_${M}_$F" \
        --json-out "$R/ari_dp2fb_${M}_$F.json" || echo "FAILED boot $M $F"
  done
  echo "########## T1 self-aligned linkage, face crop, FB_$M"
  LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
  $PY_ARC scripts/anon/linkage_attack.py --ctx 0 --self-align \
      --probe-root "$D/ProcessedDP2fb_$M" --folder appleFace \
      --n-per-subject 60 --tag "t1_dp2fb_$M" \
      --json-out "$R/t1_dp2fb_$M.json" || echo "FAILED t1 $M"
done
echo ALL_DONE
