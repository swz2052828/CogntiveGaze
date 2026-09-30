#!/bin/bash
#SBATCH --job-name=calib_verify
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_verify_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
for REC in 00006 00013; do
  $PY scripts/anon/extract_calib_frames.py --rec $REC --verify
done
