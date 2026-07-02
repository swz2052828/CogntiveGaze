#!/bin/bash
# Auto-aggregation for the ADV clean-retrain fleet. Submitted with a SLURM
# dependency (afterany) on the last still-running backbone arrays so it fires
# on the cluster once every fold has terminated -- independent of any Claude
# session / login node being alive. CPU-only (mmap torch.load of val_error).
#
#   sbatch --dependency=afterany:<jobA>:<jobB> scripts/run_aggregate_clean_retrain_adv.sbatch
#SBATCH --job-name=agg_clean_adv
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/agg_clean_adv_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/agg_clean_adv_%j.err

set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
mkdir -p /springbrook/share/eng/esrpxk/logs

module load Miniconda3/24.7.1-0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze

cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
srun python scripts/aggregate_clean_retrain_adv.py
echo "aggregation done -> results/clean_retrain_adv_summary.md  end=$(date -Is)"
