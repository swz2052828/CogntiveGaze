#!/bin/bash
#SBATCH --job-name=anon_t1adapt
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_t1adapt_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
# The third regime: release-only, but alignment solved from the release itself.
# This is the cell that decides whether the conservative column's assumption is
# doing the work, so it runs on the operators where self-align appeared to protect.
for OP in blur0.10 blur0.20 blur0.35 blur0.50 blackbox pixel0.25 pixel0.40 none; do
  echo "########## t1ad_$OP"
  $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-per-subject 60 --adapted-align --tag "t1ad_$OP" \
      --json-out "$RES/t1ad_$OP.json" || echo "CELL FAILED: t1ad_$OP"
done
echo ALL_DONE
