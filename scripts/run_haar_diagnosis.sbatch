#!/bin/bash
# scripts/run_haar_diagnosis.sbatch
#
# CPU-only Haar face-detector diagnosis on a COMPUTE node (never the login node).
# Submit:
#   sbatch scripts/run_haar_diagnosis.sbatch
#
#SBATCH --job-name=haar_diag
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/haar_diag_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/haar_diag_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs

PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze

# Keep OpenCV/OpenMP within the allocated cpus-per-task.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENCV_FFMPEG_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "Host: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Start: $(date)"
"$PY" haar_diagnosis.py \
  --stride 25 \
  --out /springbrook/share/eng/esrpxk/datasets/face_detector_comparison/haar_diagnosis
echo "Done: $(date)"
