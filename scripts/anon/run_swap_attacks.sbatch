#!/bin/bash
#SBATCH --job-name=anon_swap
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_swap_%j.log
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

# The two configurations the deployment actually chooses between:
#   ProcessedSwap        - SimSwap++ applied to face AND eye crops (full swap)
#   ProcessedSwap_oldeye - swapped face, ORIGINAL eye crops restored byte-exact
# The difference between them is the privacy price of preserving the eyes, which
# is the trade-off the whole method rests on and has never been measured.
for COND in ProcessedSwap ProcessedSwap_oldeye; do
  echo "########## T2 $COND"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/$COND" \
      --n-gallery 60 --n-probe 60 --tag "$COND" --attributes \
      --json-out "$RES/t2_${COND}.json" || echo "FAILED T2 $COND"
done
echo "########## T1 linkage"
for COND in ProcessedSwap ProcessedSwap_oldeye; do
  $PY scripts/anon/linkage_attack.py --probe-root "$D/$COND" \
      --tag "t1_${COND}" --json-out "$RES/t1_${COND}.json" || echo "FAILED T1 $COND"
  $PY scripts/anon/linkage_attack.py --probe-root "$D/$COND" --self-align \
      --tag "t1_${COND}_selfalign" --json-out "$RES/t1_${COND}_selfalign.json" \
      || echo "FAILED T1self $COND"
done
echo ALL_DONE
