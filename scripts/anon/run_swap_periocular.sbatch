#!/bin/bash
#SBATCH --job-name=anon_swpe
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_swpe_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets
RES=/springbrook/share/eng/esrpxk/results/anon_swap
mkdir -p "$RES"

# The face-crop attack CANNOT separate ProcessedSwap from ProcessedSwap_oldeye:
# the two ship the identical swapped face crop and differ only in the eye crops.
# The privacy price of preserving the eyes therefore has to be read off the eye
# crops themselves, with detection bypassed (a FACE detector finds nothing in a
# 120 px eye crop, and detector failure must not be scored as privacy).
for FLD in appleLeftEye appleRightEye; do
  for COND in ProcessedSwap ProcessedData; do
    echo "########## periocular $COND $FLD"
    $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/$COND" --folder "$FLD" \
        --no-detect --n-gallery 60 --n-probe 60 --tag "${COND}_${FLD}" \
        --json-out "$RES/peri_${COND}_${FLD}.json" || echo "FAILED $COND $FLD"
  done
done
echo ALL_DONE
