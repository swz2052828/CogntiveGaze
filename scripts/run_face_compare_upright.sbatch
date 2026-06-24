#!/bin/bash
# scripts/run_face_compare_upright.sbatch
# Re-run the 3-detector comparison on UPRIGHT frames (rotate 90 deg CW) on a
# COMPUTE node.
#SBATCH --job-name=facecmp_up
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/facecmp_up_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/facecmp_up_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" face_detector_compare.py --subjects all --window 1200 \
  --rotate cw --sample-stride 200 \
  --out-root /springbrook/share/eng/esrpxk/datasets/face_detector_comparison_upright
echo "Done: $(date)"
