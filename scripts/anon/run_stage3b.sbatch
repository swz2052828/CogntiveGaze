#!/bin/bash
#SBATCH --job-name=anon_s3b
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s3b_%j.log
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

# T2: external-reference attacker. Gallery = originals (stand-in for a photo the
# attacker obtained elsewhere); probe = the released, anonymised crops.
for OP in none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
          pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox; do
  echo "########## $OP"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-gallery 60 --n-probe 60 --tag "$OP" --attributes \
      --json-out "$RES/t2_$OP.json"
done

# background floor, recomputed at the same sample size for comparability
$PY scripts/anon/nuisance_control.py --n-per-half 60 --json-out "$RES/floor_background.json"
echo ALL_DONE
