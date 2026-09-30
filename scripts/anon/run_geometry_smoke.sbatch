#!/bin/bash
#SBATCH --job-name=anon_geom_smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_geom_smoke_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
echo "== tracking mode (as original pipeline) =="
$PY scripts/anon/recover_geometry.py --rec 00006 --limit 300 --report-only
echo
echo "== static mode (per-frame, no history) =="
$PY scripts/anon/recover_geometry.py --rec 00006 --limit 300 --report-only --static
echo
echo "== second subject, tracking =="
$PY scripts/anon/recover_geometry.py --rec 00013 --limit 300 --report-only
