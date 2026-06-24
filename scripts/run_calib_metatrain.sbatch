#!/bin/bash
# scripts/run_calib_metatrain.sbatch
# Re-meta-train the FiLM adapter so its inner-loop SUPPORT is each subject's
# PRE-TASK calibration frames (deploy-faithful) instead of random in-task frames.
# This fixes the near-no-op meta/meta_adv seen in calib_metacompare (the adapter
# was previously meta-trained on in-task support, a distribution it never sees at
# deploy). Query stays on in-task frames (labels available at train time).
# Writes new checkpoints to <RUN_ROOT>/meta_on_base_calib and meta_on_adv_calib;
# the original meta_on_base / meta_on_adv are left untouched for comparison.
#
#   RUN_ROOT=.../meta_pipeline_mobile_vit_rep1 BACKBONE=mobile_vit ADAPTER=film \
#     sbatch --array=0-4 scripts/run_calib_metatrain.sbatch
#SBATCH --job-name=calib_mtrain
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_mtrain_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/calib_mtrain_%A_%a.err

set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze

module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || { echo "[ERR] activate gaze env failed"; exit 1; }

: "${DATA_PATH:=/springbrook/share/eng/esrpxk/datasets/ProcessedData}"
: "${EYE_PATH:=$DATA_PATH}"
: "${MEAN_PATH:=meanno7}"
: "${BACKBONE:=mobile_vit}"
: "${ADAPTER:=film}"
: "${SEED:=42}"
# densest calibration set -> variable support in [4, min(META_SUPPORT, n_calib)]
: "${CALIB_ROOT:=/springbrook/share/eng/esrpxk/datasets/calib_support_K72}"
: "${RUN_ROOT:=/springbrook/share/eng/esrpxk/runs/meta_pipeline_mobile_vit_rep1}"
: "${META_ITERS:=2000}"
: "${INNER_STEPS:=20}"
: "${INNER_LR:=1.0}"
: "${META_SUPPORT:=72}"   # cap on support size (variable per episode)
: "${META_QUERY:=32}"
: "${TEMPORAL_WINDOW:=4}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=16}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

BASE_CKPT="$RUN_ROOT/base/seed${SEED}/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
ADV_CKPT="$RUN_ROOT/adv/seed${SEED}/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
SMD="$RUN_ROOT/meta_on_base_calib/seed${SEED}"
SAMD="$RUN_ROOT/meta_on_adv_calib/seed${SEED}"
PIPE_LOGS="$RUN_ROOT/logs"
mkdir -p "$SMD" "$SAMD" "$PIPE_LOGS"

CLI="python vit_gaze_segmenter.py"
COMMON="--data-path $DATA_PATH --eye-path $EYE_PATH --mean-path $MEAN_PATH \
  --input-mode multistream --backbone $BACKBONE --use-grid \
  --weights imagenet --temporal-window $TEMPORAL_WINDOW \
  --adapter $ADAPTER --meta-support $META_SUPPORT --meta-query $META_QUERY \
  --inner-steps $INNER_STEPS --inner-lr $INNER_LR --meta-iters $META_ITERS \
  --calib-support-root $CALIB_ROOT \
  --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
  --fold-index $FOLD --seed $SEED"

echo "Host $(hostname)  fold=$FOLD  backbone=$BACKBONE  calib=$CALIB_ROOT  $(date)"

# Stage A: meta on BASE (calib support)
META_CKPT="$SMD/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
if [[ -f "$META_CKPT" ]]; then
  echo "[fold $FOLD] meta_on_base_calib: skip ($META_CKPT exists)"
else
  $CLI metatrain $COMMON --init-checkpoint "$BASE_CKPT" --out-path "$SMD" \
    --log-file "$PIPE_LOGS/meta_on_base_calib_seed${SEED}_fold${FOLD}.log"
fi

# Stage B: meta on ADV (calib support)
META_ADV_CKPT="$SAMD/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
if [[ -f "$META_ADV_CKPT" ]]; then
  echo "[fold $FOLD] meta_on_adv_calib: skip ($META_ADV_CKPT exists)"
else
  $CLI metatrain $COMMON --init-checkpoint "$ADV_CKPT" --out-path "$SAMD" \
    --log-file "$PIPE_LOGS/meta_on_adv_calib_seed${SEED}_fold${FOLD}.log"
fi
echo "Done fold $FOLD: $(date)"
