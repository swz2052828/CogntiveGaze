#!/bin/bash
#SBATCH --job-name=anon_flatatk
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_flatatk_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
F=/springbrook/share/eng/esrpxk/datasets/anon_eyeflat
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
# --folder appleFace: in the flat arm the released eye files are constant grey,
# so an eye-crop attack would score nothing and look like a spurious collapse.
# 60/60 matches the Stage 3b run so d' is comparable.
for OP in blackbox blur0.50; do
  echo "########## ${OP} eye_mode=flat"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$F/$OP" --folder appleFace \
      --n-gallery 60 --n-probe 60 --tag "${OP}_eyeflat" --attributes \
      --json-out "$RES/t2_${OP}_eyeflat.json"
done
echo ALL_DONE
