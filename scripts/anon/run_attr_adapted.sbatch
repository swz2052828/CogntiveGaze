#!/bin/bash
#SBATCH --job-name=anon_attradapt
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_attradapt_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_attr_adapted2
mkdir -p "$RES"
for OP in none blur0.10 blur0.20 blur0.35 blur0.50 pixel0.40 blackbox; do
  echo "########## $OP"
  $PY scripts/anon/attribute_adapted.py --ctx 0 --probe-root "$A/$OP" \
      --tag "adapt_$OP" --json-out "$RES/adapt_$OP.json" || echo "CELL FAILED: $OP"
done
echo ALL_DONE
