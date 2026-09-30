#!/bin/bash
# T4 adaptive attacker. Pure numpy on cached embeddings -- seconds, one core, no
# GPU. Runs on `compute` rather than the login node per the usage policy.
#SBATCH --job-name=adapt_t4
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/adapt_t4_%j.log
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
R=/springbrook/share/eng/esrpxk/results
ORIG=$R/anon_stage3b/t1_none.emb.npz
for A in swap:$R/anon_swap/t1_ProcessedSwap.emb.npz \
         blackbox:$R/anon_stage3b/t1_blackbox.emb.npz \
         blur0.50:$R/anon_stage3b/t1_blur0.50.emb.npz \
         blur0.20:$R/anon_stage3b/t1_blur0.20.emb.npz \
         pixel0.25:$R/anon_stage3b/t1_pixel0.25.emb.npz; do
  TAG=${A%%:*}; F=${A#*:}
  [ -f "$F" ] || { echo "SKIP missing $F"; continue; }
  echo "=== $TAG"
  $PY scripts/anon/adaptive_attack.py --orig "$ORIG" --anon "$F" \
      --tag "$TAG" --out "$R/anon_adaptive/t4_${TAG}.json"
done
