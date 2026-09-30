#!/bin/bash
# Periocular T4: does the swap keep the EYE region at the nuisance floor when the
# attacker adapts? The eye crops are where the utility lives, so this is the arm
# that matters for the method. One small GPU pass (about 8.6k 112x112 embeddings)
# then a CPU fit.
#SBATCH --job-name=peri_t4
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/peri_t4_%j.log
set -uo pipefail
# The insightface/ArcFace stack lives in envs/idattack, not envs/gaze, and needs
# the bundled CUDA libraries on LD_LIBRARY_PATH or onnxruntime-gpu silently falls
# back to CPU (or fails to import at all).
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_adaptive
mkdir -p "$R"
for FOLD in appleLeftEye appleRightEye; do
  for RT in ProcessedData ProcessedSwap; do
    E=$R/emb_${RT}_${FOLD}.npz
    [ -f "$E" ] || $PY scripts/anon/periocular_attack.py --root "$D/$RT" \
        --folder "$FOLD" --tag "${RT}_${FOLD}" --emb-out "$E" \
        --json-out "$R/peri_${RT}_${FOLD}.json"
  done
  echo "=== $FOLD"
  $PY scripts/anon/adaptive_attack.py \
      --orig "$R/emb_ProcessedData_${FOLD}.npz" \
      --anon "$R/emb_ProcessedSwap_${FOLD}.npz" \
      --tag "swap_${FOLD}" --out "$R/t4_swap_${FOLD}.json"
done
