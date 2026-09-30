#!/bin/bash
#SBATCH --job-name=eyelm_smoke
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/eyelm_smoke_%j.log
set -uo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
for REC in 00006 00013 00020; do
  $PY scripts/anon/eye_landmarks.py --rec $REC --limit 120
done
