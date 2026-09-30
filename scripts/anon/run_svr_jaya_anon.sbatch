#!/bin/bash
#SBATCH --job-name=jaya_anon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-24%4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/jaya_anon_%A_%a.log
set -uo pipefail
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || exit 1
export HF_HUB_OFFLINE=1
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/runs/anon_svr_jaya
mkdir -p "$OUT"

# TUNED svr_embed on the anonymisation arms.
#
# Every anon result so far used svr_embed at FIXED (C=1, gamma=scale, eps=0.1).
# The optimizer sweep (2026-07-19) showed that configuration is the weak one:
# cohort @K72 fixed 5.41 > pso 5.02 > mvo 4.80 > JAYA 4.69, and tuned svr_embed
# reaches the fc_ft/meta podium. So the svr_embed column in Stage 4 measures a
# deliberately handicapped calibrator, and any claim resting on it -- including
# "svr_embed absorbs the anonymisation cost" -- is about fixed HPs, not svr_embed.
#
# Enrolment is matched to the operator (condition iii): the tuner fits
# (C,gamma,eps) ON the support, so a clean support with anonymised deployment
# would tune for the wrong distribution.
#
# The script emits fixed/pso/jaya/mvo rows per fold, so fixed-vs-tuned is
# measured inside the same run rather than across jobs.
BBS=(mobile_vit convnextv2 itracker mobilenet_v3 mgazenet)
OPS=(none blackbox)
I=$SLURM_ARRAY_TASK_ID
BB=${BBS[$(( I / 5 ))]}
FOLD=$(( I % 5 ))
for OP in "${OPS[@]}"; do
  DP=$D/anon/$OP
  CS=$D/calib_support_K72_$OP
  CK=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_${BB}/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  [ -f "$CS/meanno7_clean/metadata.mat" ] || { echo "FATAL no manifest in $CS"; continue; }
  [ -f "$CK" ] || { echo "FATAL no checkpoint $CK"; continue; }
  echo "########## JAYA $BB fold$FOLD op=$OP $(date +%H:%M:%S)"
  python scripts/svr_optimizer_sweep.py \
    --backbone $BB --fold-index $FOLD \
    --data-root "$DP" --calib-root "$CS" --checkpoint "$CK" \
    --tag "anon_${OP}" \
    --batch-size 128 --num-workers 16 \
    --csv-out "$OUT/jaya_${BB}_${OP}.csv" || echo "FAILED $BB $OP f$FOLD"
done
echo "ALL_DONE $BB fold$FOLD"
