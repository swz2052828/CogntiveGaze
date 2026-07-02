#!/bin/bash
# Unified eval-side extras on clean checkpoints: fc_ft tuning (#1), LOO auto-selection
# (#2), per-frame prediction dump for temporal smoothing (#5). One array task per fold.
#
#   RUN_ROOT=... BACKBONE=... sbatch --array=0-4 scripts/run_calib_extras.sbatch
#SBATCH --job-name=calib_extras
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_extras_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/calib_extras_%A_%a.err

set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze

: "${BACKBONE:?}"
: "${RUN_ROOT:?}"
: "${OUT_ROOT:=/springbrook/share/eng/esrpxk/runs/calib_extras}"
: "${KS:=9 72}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$OUT_ROOT"

srun python scripts/calib_eval_extras.py \
  --run-root "$RUN_ROOT" --backbone "$BACKBONE" --fold-index "$FOLD" --ks $KS \
  --svr-hp-dir "/springbrook/share/eng/esrpxk/runs/calib_metacompare/svr_hp_${BACKBONE}_clean" \
  --csv-out "$OUT_ROOT/extras_${BACKBONE}.csv" \
  --dump-dir "$OUT_ROOT/dumps" \
  --hp-out "$OUT_ROOT/fcft_hp_${BACKBONE}.json" \
  --batch-size 128 --num-workers "$SLURM_CPUS_PER_TASK"
echo "done $BACKBONE fold $FOLD"
