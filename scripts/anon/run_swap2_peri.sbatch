#!/bin/bash
# Swap2 eye crops (now repaired) + T4 on them, and a rerun of every earlier T4 arm
# with the TAR threshold aligned to id_attack.tar_at_far (np.quantile). The
# earlier T4 JSONs used the k-th-highest-impostor threshold; the rerun makes every
# T4 number in the paper come from one definition.
#SBATCH --job-name=swap2_peri
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/swap2_peri_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results
A=$R/anon_adaptive; S2=$R/anon_swap2
mkdir -p "$S2"

for F in appleLeftEye appleRightEye; do
  SHORT=""
  for REC in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$D/ProcessedSwap2/$REC/$F" | wc -l); W=$(ls "$D/ProcessedData/$REC/$F" | wc -l)
    [ "$H" -lt "$W" ] && SHORT="$SHORT $REC($H/$W)"
  done
  [ -n "$SHORT" ] && { echo "FATAL $F short:$SHORT"; exit 1; }
  echo "########## periocular ProcessedSwap2 $F"
  $PY scripts/anon/periocular_attack.py --root "$D/ProcessedSwap2" --folder "$F" \
      --tag "ProcessedSwap2_$F" --emb-out "$S2/emb_ProcessedSwap2_$F.npz" \
      --json-out "$S2/peri_ProcessedSwap2_$F.json" || echo "FAILED peri $F"
  echo "########## T4 swap2 $F"
  $PY scripts/anon/adaptive_attack.py --orig "$A/emb_ProcessedData_$F.npz" \
      --anon "$S2/emb_ProcessedSwap2_$F.npz" --tag "swap2_$F" --out "$S2/t4_swap2_$F.json" || echo "FAILED T4 $F"
  echo "########## T4 rerun swap (template 1) $F"
  $PY scripts/anon/adaptive_attack.py --orig "$A/emb_ProcessedData_$F.npz" \
      --anon "$A/emb_ProcessedSwap_$F.npz" --tag "swap_$F" --out "$A/t4_swap_$F.json" || echo "FAILED T4r $F"
done
ORIG=$R/anon_stage3b/t1_none.emb.npz
for ARM in swap:$R/anon_swap/t1_ProcessedSwap.emb.npz blackbox:$R/anon_stage3b/t1_blackbox.emb.npz \
           blur0.50:$R/anon_stage3b/t1_blur0.50.emb.npz blur0.20:$R/anon_stage3b/t1_blur0.20.emb.npz \
           pixel0.25:$R/anon_stage3b/t1_pixel0.25.emb.npz; do
  T=${ARM%%:*}; E=${ARM#*:}
  echo "########## T4 rerun face $T"
  $PY scripts/anon/adaptive_attack.py --orig "$ORIG" --anon "$E" --tag "$T" --out "$A/t4_${T}.json" || echo "FAILED T4 face $T"
done
echo ALL_DONE
