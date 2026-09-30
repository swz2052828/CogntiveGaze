#!/bin/bash
#SBATCH --job-name=calibcrop
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calibcrop_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
for REC in 00006 00013 00020; do
  $PY scripts/anon/calib_crop_geometry.py --rec $REC --limit 40
done
