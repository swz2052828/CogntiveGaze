#!/bin/bash
# scripts/run_haar_rotated.sbatch  — Haar de-rotation re-run on a COMPUTE node.
#SBATCH --job-name=haar_rot
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/haar_rot_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/haar_rot_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" haar_rotated_rerun.py --stride 25 \
  --out /springbrook/share/eng/esrpxk/datasets/face_detector_comparison/haar_rotated
echo "Done: $(date)"
