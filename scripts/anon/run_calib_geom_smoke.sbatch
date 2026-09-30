#!/bin/bash
#SBATCH --job-name=calib_geom
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_geom_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
for REC in 00006 00013 00020; do
  $PY scripts/anon/recover_calib_geometry.py --rec $REC --k 72
done
