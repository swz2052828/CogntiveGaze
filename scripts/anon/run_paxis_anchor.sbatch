#!/bin/bash
#SBATCH --job-name=paxis_anch
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_anch_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets
RES=/springbrook/share/eng/esrpxk/results/anon_paxis

# The P0 anchor for the periocular axis, measured with the IDENTICAL harness as
# every other point on it: id_attack.py, --folder appleLeftEye, --no-detect.
#
# It was missing, and the gap got filled with the wrong number. The figure had
# been anchored at d' 5.122 / Rank-1 100%, which is t2_blackbox.json -- the T2
# attack on the FACE crop, a different crop and a different attack
# configuration. Points on one curve must share a harness; an anchor borrowed
# from another measurement exaggerated the axis by more than a factor of two.
#
# ProcessedData eye crops are the true P0: P-axis operators only ever remove
# pixels from them, and blackbox/ProcessedSwap_oldeye restore them byte-exact.
for F in appleLeftEye appleRightEye; do
  echo "########## P0 anchor $F"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/ProcessedData" --folder $F \
      --no-detect --n-gallery 60 --n-probe 60 --tag "paxis_P0_${F}" \
      --json-out "$RES/paxis_P0_${F}.json" || echo "FAILED $F"
done
# and the right-eye arm for every P level, so the axis is not a single-eye claim
for LV in P1 P2 P1d5 P1d10 P1d20 P1d40; do
  echo "########## $LV appleRightEye"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/anon_paxis/$LV" \
      --folder appleRightEye --no-detect --n-gallery 60 --n-probe 60 \
      --tag "paxis_${LV}_reye" --json-out "$RES/paxis_${LV}_reye.json" \
      || echo "FAILED $LV reye"
done
echo ALL_DONE
