#!/bin/bash
#SBATCH --job-name=seek_test
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/seek_test_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/seek_test_%j.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname) start $(date)"
"$PY" test_seek_offset.py 6 7 8 16 21 23 11 17
echo "done $(date)"
