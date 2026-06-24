#!/bin/bash
# scripts/run_calib_processing.sbatch
# Final preprocessing of each subject's 1200-frame calibration window on a
# COMPUTE node: facemesh face + facemesh_contour eye + ear blink, upright (cw).
#SBATCH --job-name=calib_proc
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_proc_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/calib_proc_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" process_calib_windows.py \
  --subjects all --window 1200 --rotate cw \
  --face-method mediapipe_facemesh \
  --eye-method facemesh_contour \
  --blink-method ear \
  --vis-stride 30 \
  --out-root /springbrook/share/eng/esrpxk/datasets/calib_processed
echo "Done: $(date)"
