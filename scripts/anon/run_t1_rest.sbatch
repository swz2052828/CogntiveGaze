#!/bin/bash
#SBATCH --job-name=anon_t1rest
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_t1rest_%j.log
set -uo pipefail          # NOT -e: a failing cell must not abort the sweep
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
# cells the crashed sweep never produced
for OP in pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40; do
  for SA in "" "--self-align"; do
    TAG=$([ -z "$SA" ] && echo "t1_$OP" || echo "t1sa_$OP")
    [ -f "$RES/$TAG.json" ] && { echo "SKIP $TAG (exists)"; continue; }
    echo "########## $TAG"
    $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/$OP" \
        --n-per-subject 60 $SA --tag "$TAG" --json-out "$RES/$TAG.json" || \
        echo "CELL FAILED: $TAG"
  done
done
echo ALL_DONE
