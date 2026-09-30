#!/bin/bash
#SBATCH --job-name=anon_attr
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_attr_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_attr
mkdir -p "$RES"
# Tests the marginal-shift hypothesis: does blur destroy the gender signal, or
# does it push the classifier systematically toward one class? The predicted
# marginal on original vs anonymised is the discriminating quantity.
for OP in none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 pixel0.12 pixel0.25 pixel0.40 blackbox; do
  echo "########## $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-gallery 20 --n-probe 60 --attributes --tag "attr_$OP" \
      --json-out "$RES/attr_$OP.json" || echo "CELL FAILED: attr_$OP"
done
echo ALL_DONE
