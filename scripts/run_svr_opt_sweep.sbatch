#!/bin/bash
#SBATCH --job-name=svr_opt
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/svr_opt_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/svr_opt_%A_%a.err
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface HF_HUB_OFFLINE=1
: "${BACKBONE:?}"
srun python scripts/svr_optimizer_sweep.py --backbone "$BACKBONE" \
  --fold-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --csv-out "/springbrook/share/eng/esrpxk/runs/svr_opt_sweep/svropt_${BACKBONE}.csv"
