#!/bin/bash
# FAMS (diffusion) privacy arm: 260 frames/participant, the same frames DeepPrivacy2
# used, both identity regimes; then the null control and the same attacks.
#
#   MODE=smoke  2 participants x 4 frames x 2 regimes, prints timing   (array 0)
#   MODE=gen    one task per (regime, participant)                    (array 0-35)
#   MODE=attack null control + A2 periocular + A1 + T1 + bootstraps   (array 0)
#   MODE=utilgen every 10th frame, per_frame identity, one task per rec (array 0-17)
#SBATCH --job-name=fams
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/fams_%x_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
export HF_HOME=/springbrook/share/eng/esrpxk/home_caches/huggingface HF_HUB_OFFLINE=1
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_fams
mkdir -p "$R"
PYF=/springbrook/share/eng/esrpxk/envs/fams/bin/python
MODES=(per_subject per_frame)
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)

case "${MODE:?smoke|gen|utilgen|attack}" in
smoke)
  for M in per_subject per_frame; do
    SECONDS=0
    $PYF scripts/anon/run_fams.py --identity $M --recs 00006 00013 --n-per-subject 4 \
        --out-root "$R/smoke/$M" --report "$R/smoke/rep_$M.json" || { echo FAILED; exit 1; }
    echo "TIMING $M 8 frames ${SECONDS}s (includes model load)"
  done ;;
gen)
  M=${MODES[$(( SLURM_ARRAY_TASK_ID / 18 ))]}; REC=${RECS[$(( SLURM_ARRAY_TASK_ID % 18 ))]}
  $PYF scripts/anon/run_fams.py --identity $M --recs $REC --n-per-subject 260 --resume \
      --out-root "$D/ProcessedFAMS_$M" --report "$R/edit_${M}_${REC}.json" \
      || { echo "FAILED $M $REC"; exit 1; }
  echo "ALL_DONE $M $REC" ;;
utilgen)
  # Utility subsample: every 10th frame of every recording, fresh identity per
  # frame (privacy showed the two regimes indistinguishable for FAMS, whose output
  # identity follows the source frame). ~1,170 frames per participant.
  REC=${RECS[$SLURM_ARRAY_TASK_ID]}
  $PYF scripts/anon/run_fams.py --identity per_frame --recs $REC --all-frames --every 10 --resume \
      --out-root "$D/ProcessedFAMSsub10_per_frame" --report "$R/edit_util_${REC}.json" \
      || { echo "FAILED util $REC"; exit 1; }
  echo "ALL_DONE util $REC" ;;
attack)
  PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
  PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
  SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
  export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
  echo "########## NULL CONTROL"
  $PY - <<'EOF' || exit 1
import cv2, numpy as np, glob, os, sys
D='/springbrook/share/eng/esrpxk/datasets'
A, B, O = f'{D}/ProcessedFAMS_per_subject', f'{D}/ProcessedFAMS_per_frame', f'{D}/ProcessedData'
def md(x, y):
    d = []
    for rec in sorted(os.listdir(x))[:5]:
        for f in sorted(glob.glob(f'{x}/{rec}/appleFace/*.jpg'))[:12]:
            g = f.replace(x, y)
            if os.path.exists(g):
                a, b = cv2.imread(f), cv2.imread(g)
                if a.shape == b.shape: d.append(np.abs(a.astype(int) - b.astype(int)).mean())
    return float(np.mean(d)), len(d)
for name, (x, y) in {'per_subject vs per_frame': (A, B), 'per_subject vs original': (A, O)}.items():
    m, n = md(x, y); print(f'  {name}: mean |delta| {m:.3f} over {n}')
    if m < 1.0: print('  FATAL: arms identical'); sys.exit(1)
print('  PASS')
EOF
  for M in per_subject per_frame; do
    for F in appleLeftEye appleRightEye; do
      $PY_ARC scripts/anon/periocular_attack.py --ctx 0 --root "$D/ProcessedFAMS_$M" --folder $F \
          --tag "fams${M}_$F" --emb-out "$R/emb_fams${M}_$F.npz" --json-out "$R/peri_fams${M}_$F.json" \
          || echo "FAILED peri $M $F"
      $PY_ARC scripts/anon/id_attack.py --ctx 0 --no-detect --gallery-root "$D/ProcessedData" \
          --probe-root "$D/ProcessedFAMS_$M" --folder $F --n-gallery 60 --n-probe 60 \
          --tag "a1_fams${M}_$F" --json-out "$R/a1_fams${M}_$F.json" || echo "FAILED a1 $M $F"
      $PY scripts/anon/ari_bootstrap.py "$R/emb_fams${M}_$F.npz" --tag "fams_${M}_$F" \
          --json-out "$R/ari_fams_${M}_$F.json" || echo "FAILED boot $M $F"
    done
    $PY_ARC scripts/anon/linkage_attack.py --ctx 0 --self-align --probe-root "$D/ProcessedFAMS_$M" \
        --folder appleFace --n-per-subject 60 --tag "t1_fams_$M" --json-out "$R/t1_fams_$M.json" \
        || echo "FAILED t1 $M"
  done ;;
esac
echo ALL_DONE
