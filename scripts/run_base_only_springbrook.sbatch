#!/bin/bash
# scripts/run_base_only_springbrook.sbatch
#
# Base-training-ONLY pipeline (stage 1 of run_meta_pipeline_springbrook.sbatch).
# For backbones that deliberately opt out of the meta-learned / SVR-embed
# calibration path -- e.g. --backbone cnn_transformer_raw, whose CLS-token
# Transformer fusion does not expose forward_features, so the subject-adv /
# metatrain / svr-embed stages do not apply. This runs only `train` per fold.
#
# Submit:
#   sbatch --export=ALL,BACKBONE=cnn_transformer_raw scripts/run_base_only_springbrook.sbatch

#SBATCH --job-name=vit_gaze_base
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/base_%A_fold%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/base_%A_fold%a.err

# ==============================================================================
# Config -- same defaults as run_meta_pipeline_springbrook.sbatch
# ==============================================================================
: "${DATA_PATH:=/springbrook/share/eng/esrpxk/datasets/ProcessedData}"
: "${EYE_PATH:=/springbrook/share/eng/esrpxk/datasets/ProcessedData}"
: "${MEAN_PATH:=meanno7}"
: "${OUT_ROOT:=/springbrook/share/eng/esrpxk/runs/meta_pipeline}"
: "${BACKBONE:=cnn_transformer_raw}"
: "${SEEDS:=42}"
: "${EPOCHS:=20}"
: "${TEMPORAL_WINDOW:=4}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=16}"
: "${LR:=8e-4}"
: "${LR_SCHEDULER:=cosine}"
: "${AUGMENT:=light}"
: "${PATIENCE:=5}"
: "${MIN_DELTA:=0.005}"

echo "Running on: $(hostname)"
echo "Fold index: $SLURM_ARRAY_TASK_ID  Backbone: $BACKBONE  Seeds: $SEEDS"
nvidia-smi -L

# ==============================================================================
# Modules + conda env (Springbrook-specific)
# ==============================================================================
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || { echo "[ERR] failed to activate gaze env"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /springbrook/share/eng/esrpxk/logs

echo "[repo] running from $(pwd)"
echo "[repo] HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

set -euo pipefail
_fmt() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
BASE_DIR="$OUT_ROOT/base"
PIPE_LOGS="$OUT_ROOT/logs"
mkdir -p "$BASE_DIR" "$PIPE_LOGS"

CLI="srun --cpus-per-task=$SLURM_CPUS_PER_TASK python vit_gaze_segmenter.py"
COMMON_DATA="--data-path $DATA_PATH --eye-path $EYE_PATH --mean-path $MEAN_PATH \
  --input-mode multistream --backbone $BACKBONE --use-grid"

for SEED in $SEEDS; do
  SBD="$BASE_DIR/seed${SEED}"; mkdir -p "$SBD"
  BASE_CKPT="$SBD/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"

  COMMON_TRAIN="$COMMON_DATA --weights imagenet --epochs $EPOCHS --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS --lr $LR --lr-scheduler $LR_SCHEDULER --augment $AUGMENT \
    --patience $PATIENCE --min-delta $MIN_DELTA --amp --temporal-window $TEMPORAL_WINDOW \
    --fold-index $FOLD --seed $SEED"

  echo "===================================================="
  echo "[fold $FOLD seed $SEED] base-only training ($BACKBONE)"
  echo "===================================================="
  _t=$SECONDS
  if [[ -f "$BASE_CKPT" ]]; then
    echo "[fold $FOLD seed $SEED] stage 1 base: skip ($BASE_CKPT exists)"
  else
    $CLI train $COMMON_TRAIN --out-path "$SBD" \
      --log-file "$PIPE_LOGS/base_seed${SEED}_fold${FOLD}.log"
  fi
  echo "[fold $FOLD seed $SEED] TIMING stage1_base took $(_fmt $(( SECONDS - _t )))"
  echo "[fold $FOLD seed $SEED] done."
done
