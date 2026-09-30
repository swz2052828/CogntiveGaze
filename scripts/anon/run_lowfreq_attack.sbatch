#!/bin/bash
#SBATCH --job-name=anon_lowfreq
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_lowfreq_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
E=/springbrook/share/eng/esrpxk/datasets/anon_extreme
RES=/springbrook/share/eng/esrpxk/results/anon_lowfreq
mkdir -p "$RES"
# Is the ArcFace-only attacker under-powered on the low-frequency field that every
# blur arm leaves? Fuse an 8x8 thumbnail descriptor with the embedding and sweep
# the weight; the attacker is free to pick the best w, so the max is the right
# figure for a privacy claim.
for OP in blur0.50 blur0.35 blur0.20 blackbox; do
  echo "########## $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$A/$OP" --folder appleFace \
      --n-gallery 60 --n-probe 60 --lowfreq --tag "${OP}_lf" \
      --json-out "$RES/lf_${OP}.json"
done
echo "########## meanfill_skin"
$PY scripts/anon/id_attack.py --ctx 0 --probe-root "$E/meanfill_skin" --folder appleFace \
    --n-gallery 60 --n-probe 60 --lowfreq --tag "meanfill_skin_lf" \
    --json-out "$RES/lf_meanfill_skin.json"
echo ALL_DONE
