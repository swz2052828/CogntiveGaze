#!/bin/bash
#SBATCH --job-name=anon_periocular
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_periocular_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
RES=/springbrook/share/eng/esrpxk/results/anon_smoke
for F in appleLeftEye appleRightEye appleFace; do
  $PY scripts/anon/periocular_attack.py --folder $F --tag "peri_$F" \
      --json-out "$RES/peri_$F.json"
done
echo ALL_DONE
