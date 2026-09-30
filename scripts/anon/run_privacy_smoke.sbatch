#!/bin/bash
#SBATCH --job-name=anon_priv_smoke
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_priv_smoke_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
SMOKE=/springbrook/share/eng/esrpxk/datasets/anon_smoke
RES=/springbrook/share/eng/esrpxk/results/anon_smoke
mkdir -p "$RES"
nvidia-smi --query-gpu=name --format=csv,noheader

# 1. upper anchor: identity leakage of the UNMODIFIED release
$PY scripts/anon/id_attack.py --ctx 0 --probe-root /springbrook/share/eng/esrpxk/datasets/ProcessedData \
    --tag real --attributes --json-out "$RES/priv_real.json"

# 3. cheap operator bank
for OP in blur5 blur15 blur30 pixel16 blackbox; do
  echo "===== $OP ====="
  $PY scripts/anon/make_anon_crops.py --dst-root "$SMOKE/$OP" --op "$OP" --limit 400
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$SMOKE/$OP" --tag "$OP" \
      --attributes --json-out "$RES/priv_$OP.json"
done
echo ALL_DONE
