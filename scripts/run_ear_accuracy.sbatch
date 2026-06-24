#!/bin/bash
#SBATCH --job-name=ear_acc
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/ear_acc_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/ear_acc_%j.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" test_ear_accuracy.py
echo "Done: $(date)"
