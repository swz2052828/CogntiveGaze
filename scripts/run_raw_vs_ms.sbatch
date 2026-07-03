#!/bin/bash
# Raw-frame ViT vs facemesh+multistream head-to-head (task #6). Array = folds.
#SBATCH --job-name=raw_vs_ms
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/raw_vs_ms_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/raw_vs_ms_%A_%a.err
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface HF_HUB_OFFLINE=1
: "${MS_BACKBONE:=eyes_only_convnextv2_atto_binocular}"
srun python scripts/raw_vs_multistream_eval.py \
  --fold-index "${SLURM_ARRAY_TASK_ID:-0}" --ms-backbone "$MS_BACKBONE" \
  --csv-out "/springbrook/share/eng/esrpxk/runs/raw_vs_ms/raw_vs_${MS_BACKBONE}.csv"
