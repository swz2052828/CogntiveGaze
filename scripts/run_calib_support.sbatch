#!/bin/bash
# scripts/run_calib_support.sbatch
# Build per-K calibration SUPPORT datasets (K=4/9/18/36/72) for the meta pipeline,
# from calib_selection.csv, on a COMPUTE node (video decode + FaceMesh + crops).
#SBATCH --job-name=calib_supp
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_supp_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/calib_supp_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" generate_calib_support.py
echo "Done: $(date)"
