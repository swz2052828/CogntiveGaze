#!/bin/bash
#SBATCH --job-name=anon_dprime
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_dprime_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
RES=/springbrook/share/eng/esrpxk/results/anon_smoke
for F in appleFace appleLeftEye appleRightEye; do
  $PY scripts/anon/periocular_attack.py --folder $F --tag "dp_$F" --json-out "$RES/dp_$F.json"
done
$PY scripts/anon/nuisance_control.py --json-out "$RES/nuisance_control.json"
echo ALL_DONE
