#!/bin/bash
#SBATCH --job-name=anon_mfatk
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_mfatk_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
E=/springbrook/share/eng/esrpxk/datasets/anon_extreme
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
for OP in meanfill_skin meanfill; do
  echo "########## $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$E/$OP" --folder appleFace \
      --n-gallery 60 --n-probe 60 --tag "$OP" --attributes \
      --json-out "$RES/t2_${OP}.json"
done
echo ALL_DONE
