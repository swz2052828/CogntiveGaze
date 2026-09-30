#!/bin/bash
#SBATCH --job-name=anon_geomtm
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_geomtm_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
for REC in 00006 00013 00020; do
  $PY scripts/anon/recover_geometry_tm.py --rec $REC --limit 400
done
