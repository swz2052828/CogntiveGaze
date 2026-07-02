#!/bin/bash
# Auto-aggregation for the CLEAN metacompare (Stage B). Submitted with an afterany
# dependency on all 33 metacompare arrays so it fires on the cluster once every fold
# has terminated -- independent of any Claude session. CPU-only.
#
#   sbatch --dependency=afterany:<j1>:<j2>:... scripts/run_aggregate_clean_metacompare.sbatch
#SBATCH --job-name=agg_clean_mcmp
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/agg_clean_mcmp_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/agg_clean_mcmp_%j.err

set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
mkdir -p /springbrook/share/eng/esrpxk/logs

module load Miniconda3/24.7.1-0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze

cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
srun python scripts/aggregate_clean_metacompare.py
echo "aggregation done -> results/clean_metacompare_summary.md  end=$(date -Is)"
