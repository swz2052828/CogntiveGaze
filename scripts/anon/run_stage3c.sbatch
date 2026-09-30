#!/bin/bash
#SBATCH --job-name=anon_s3c
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s3c_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
mkdir -p "$RES"

OPS="none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
     pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox"

# --- T2 rerun -----------------------------------------------------------------
# Identical protocol and seed to job 1271444, so every number reproduces; the
# rerun exists to fill d' (the pre-registered primary metric, §3), which the
# original script did not emit, and to persist the score matrices so any further
# metric is a CPU rescore rather than another GPU pass.
for OP in $OPS; do
  echo "########## T2 $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-gallery 60 --n-probe 60 --tag "$OP" --attributes \
      --json-out "$RES/t2_$OP.json"
done
echo ALL_DONE
