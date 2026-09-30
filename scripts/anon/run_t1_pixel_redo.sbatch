#!/bin/bash
#SBATCH --job-name=t1_pixel
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/t1_pixel_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
for OP in pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40; do
  for MODE in "" "--self-align" "--adapted-align"; do
    case "$MODE" in "") TAG="t1_$OP";; "--self-align") TAG="t1sa_$OP";; *) TAG="t1ad_$OP";; esac
    [ -f "$RES/$TAG.json" ] && { echo "SKIP $TAG"; continue; }
    echo "########## $TAG"
    $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/$OP" \
        --n-per-subject 60 $MODE --tag "$TAG" --json-out "$RES/$TAG.json" \
        || echo "CELL FAILED: $TAG"
  done
done
echo ALL_DONE
