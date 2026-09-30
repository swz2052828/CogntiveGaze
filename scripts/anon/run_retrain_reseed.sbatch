#!/bin/bash
#SBATCH --job-name=retrain_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/retrain_seed_%A_%a.log
# convnextv2 retrain, seeds 1 and 7.
#
# Seed 42 collapsed on all 10 cells: train_loss_mean frozen at 0.400 from epoch
# 1, val_coord_error 9.0-9.7 cm against a 4.33 clean baseline, and IDENTICAL
# between blackbox and blur0.20 (9.6545 vs 9.6537) -- which is what rules out an
# anonymisation cause. convnextv2 is the backbone already documented as
# init-fragile (init-variance study: std 0.312, 2.24x mobile_vit).
#
# Two extra seeds per cell. The selection rule is declared HERE, before the
# results exist: a run counts as trained if its best val_coord_error is below
# 7.0 cm (midway between the 4.33 clean baseline and the ~9.5 predict-the-mean
# floor); among trained runs take the LOWEST seed index, never the best score,
# so this is not a best-of-3 that biases the arm downward. If no seed trains for
# a cell, that cell is reported as failed-to-train, not as an anonymisation cost.
set -uo pipefail
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
SEEDS=(1 7)
OPS=(blackbox blur0.20)
I=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( I / 10 ))]}
OP=${OPS[$(( (I % 10) / 5 ))]}
FOLD=$(( I % 5 ))
BB=convnextv2
DP=$D/anon/$OP
OUT=/springbrook/share/eng/esrpxk/runs/anon_retrain/${BB}_${OP}/seed${SEED}
mkdir -p "$OUT" /springbrook/share/eng/esrpxk/runs/anon_retrain/logs

SHORT=""
for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
  W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
  [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
done
[ -n "$SHORT" ] && { echo "FATAL $OP: short recordings:$SHORT"; exit 1; }

echo "########## reseed BB=$BB OP=$OP fold=$FOLD seed=$SEED $(date)"
python vit_gaze_segmenter.py train \
  --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
  --input-mode multistream --backbone $BB --use-grid \
  --weights imagenet --epochs 20 --batch-size 128 --num-workers 16 \
  --lr 8e-4 --lr-scheduler cosine --augment light \
  --patience 5 --min-delta 0.005 --amp --no-accelerate --temporal-window 4 \
  --output-activation none --gaze-range 4.0 \
  --fold-index $FOLD --seed $SEED --out-path "$OUT" \
  --log-file "/springbrook/share/eng/esrpxk/runs/anon_retrain/logs/${BB}_${OP}_f${FOLD}_s${SEED}.log"
echo "DONE $BB $OP fold$FOLD seed$SEED"
