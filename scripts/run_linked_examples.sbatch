#!/bin/bash
# scripts/run_linked_examples.sbatch
# Save example linked eye crops for the 9 calib points (interval 10) + filmstrips,
# on a COMPUTE node (video decode + FaceMesh).
#SBATCH --job-name=linked_ex
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/linked_ex_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/linked_ex_%j.err

set -euo pipefail
mkdir -p /springbrook/share/eng/esrpxk/logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  Start: $(date)"
"$PY" save_linked_examples.py
echo "Done: $(date)"
