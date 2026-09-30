#!/bin/bash
# DeepPrivacy2 at truncation 1.0, so the identity latent actually matters.
#
# The first run used anonymize.py's default truncation_value=0. The generator
# computes w = w_avg.lerp(w(z), truncation), so at 0 the latent is discarded and
# every face is the mean face -- which made --identity a no-op and produced
# bit-identical output for the two arms. Those results stand as "DP2 at its
# default setting" but cannot answer the consistency question.
#
# Here both arms run at truncation 1.0:
#   per_subject  one fixed latent per participant -> SimSwap-equivalent
#                consistency, differing only in edit domain (M5)
#   per_frame    a fresh latent per frame -> the fix our falsifiable claim
#                predicts (M7). Linkage should collapse.
#
# A NULL CONTROL runs before the attacks: if the two arms are not actually
# different images, the job fails instead of producing a comparison of an arm
# with itself.
#SBATCH --job-name=dp2_t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_t1_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
export HF_HUB_OFFLINE=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_dp2
mkdir -p "$R"
PY_DP2=/springbrook/share/eng/esrpxk/envs/dp2/bin/python
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
ARC_LIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')

for M in per_subject per_frame; do
  echo "########## generate identity=$M truncation=1.0"
  $PY_DP2 scripts/anon/run_deepprivacy2.py --identity "$M" --truncation 1.0 \
      --out-root "$D/ProcessedDP2t1_$M" --n-per-subject 260 \
      --report "$R/edit_domain_t1_$M.json" || { echo "FAILED gen $M"; exit 1; }
done

# ----------------------------------------------------- null control, first
echo "########## NULL CONTROL: do the two identity regimes differ at all?"
$PY - <<'EOF' || exit 1
import cv2, numpy as np, glob, os, sys
A='/springbrook/share/eng/esrpxk/datasets/ProcessedDP2t1_per_subject'
B='/springbrook/share/eng/esrpxk/datasets/ProcessedDP2t1_per_frame'
d=[]
for rec in sorted(os.listdir(A))[:5]:
    for f in sorted(glob.glob(f'{A}/{rec}/appleFace/*.jpg'))[:12]:
        b=f.replace(A,B)
        if not os.path.exists(b): continue
        x,y=cv2.imread(f),cv2.imread(b)
        d.append(np.abs(x.astype(int)-y.astype(int)).mean())
m=float(np.mean(d))
print(f"  mean |delta| between arms over {len(d)} pairs = {m:.3f}")
if m < 1.0:
    print("  FATAL: the two identity regimes produced (near-)identical images. "
          "The contrast is not being applied; refusing to run the attacks.")
    sys.exit(1)
print("  PASS: the arms are genuinely different.")

# and within per_frame, consecutive frames of one participant must differ
rec=sorted(os.listdir(B))[0]
fs=sorted(glob.glob(f'{B}/{rec}/appleFace/*.jpg'))[:2]
w=np.abs(cv2.imread(fs[0]).astype(int)-cv2.imread(fs[1]).astype(int)).mean()
print(f"  within per_frame, two frames of {rec}: mean |delta| = {w:.2f}")
EOF

# ------------------------------------------------------------- attacks
for M in per_subject per_frame; do
  for F in appleLeftEye appleRightEye; do
    echo "########## periocular A2  DP2t1_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/periocular_attack.py --ctx 0 \
        --root "$D/ProcessedDP2t1_$M" --folder "$F" --tag "dp2t1${M}_$F" \
        --emb-out "$R/emb_dp2t1${M}_$F.npz" \
        --json-out "$R/peri_dp2t1${M}_$F.json" || echo "FAILED peri $M $F"

    echo "########## A1 cross-domain  DP2t1_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/id_attack.py --ctx 0 --no-detect \
        --gallery-root "$D/ProcessedData" --probe-root "$D/ProcessedDP2t1_$M" \
        --folder "$F" --n-gallery 60 --n-probe 60 --tag "a1_dp2t1${M}_$F" \
        --json-out "$R/a1_dp2t1${M}_$F.json" || echo "FAILED a1 $M $F"
  done

  echo "########## T1 self-aligned linkage, face crop, DP2t1_$M"
  LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
  $PY_ARC scripts/anon/linkage_attack.py --ctx 0 --self-align \
      --probe-root "$D/ProcessedDP2t1_$M" --folder appleFace \
      --n-per-subject 60 --tag "t1_dp2t1_$M" \
      --json-out "$R/t1_dp2t1_$M.json" || echo "FAILED t1 $M"
done

echo ALL_DONE
