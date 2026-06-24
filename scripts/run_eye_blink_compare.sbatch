#!/bin/bash
# scripts/run_eye_blink_compare.sbatch
# Compare all eye + all blink detectors on upright calib windows (COMPUTE node).
#SBATCH --job-name=eyeblink
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/eyeblink_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/eyeblink_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" eye_blink_compare.py --subjects all --window 1200 --sample-stride 100 \
  --out-root /springbrook/share/eng/esrpxk/datasets/eye_blink_comparison
echo "Done: $(date)"
