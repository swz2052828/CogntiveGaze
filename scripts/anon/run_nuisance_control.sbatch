#!/bin/bash
#SBATCH --job-name=anon_nuisance
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_nuisance_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
/springbrook/share/eng/esrpxk/envs/idattack/bin/python scripts/anon/nuisance_control.py \
  --json-out /springbrook/share/eng/esrpxk/results/anon_smoke/nuisance_control.json
echo ALL_DONE
