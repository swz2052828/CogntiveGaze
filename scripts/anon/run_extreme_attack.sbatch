#!/bin/bash
#SBATCH --job-name=anon_extatk
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_extatk_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets/anon_extreme
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
# Read these two before the root is deleted for inode reasons: they are the only
# unattacked arms in anon_extreme, and regenerating them costs ~122 core-hours
# whereas attacking them costs GPU minutes. They confirm (or refute) the flat
# prediction the meanfill_skin control implies for sigma past the uniform
# transition: d' ~3.9, not a turn upward.
for OP in blur0.75 blur1.00; do
  echo "########## $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/$OP" \
      --n-gallery 60 --n-probe 60 --tag "$OP" --attributes \
      --json-out "$RES/t2_${OP}.json" || echo "FAILED $OP"
done
echo ALL_DONE
