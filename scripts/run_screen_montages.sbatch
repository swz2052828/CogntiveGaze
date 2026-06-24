#!/bin/bash
# scripts/run_screen_montages.sbatch
# Build per-subject eye-crop montages by linked screen target (cycle1 | cycle2)
# from the updated calib-frame link, on a COMPUTE node (video decode + FaceMesh).
#SBATCH --job-name=screen_mont
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/screen_mont_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/screen_mont_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" generate_cluster_montages.py
echo "Done: $(date)"
