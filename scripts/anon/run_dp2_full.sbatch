#!/bin/bash
# DeepPrivacy2: regenerate at 260 frames/participant, then attack.
#
# 260 because periocular_attack requires 4 x n_per_half frames and every other
# arm in this paper uses n_per_half=60. Matching it exactly is the difference
# between a comparison and an anecdote.
#
# Two identity regimes (see run_dp2_generate.sbatch for the reasoning):
#   per_subject  fixed latent per participant  -> differs from SimSwap in EDIT
#                DOMAIN only; the M5 comparison
#   per_frame    fresh latent per frame        -> the fix our falsifiable claim
#                predicts; linkage should collapse
#SBATCH --job-name=dp2_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_full_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
export HF_HUB_OFFLINE=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_dp2
mkdir -p "$R"
PY_DP2=/springbrook/share/eng/esrpxk/envs/dp2/bin/python
PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
ARC_LIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')

# ------------------------------------------------------------- generation
for M in per_subject per_frame; do
  echo "########## generate identity=$M"
  $PY_DP2 scripts/anon/run_deepprivacy2.py --identity "$M" \
      --out-root "$D/ProcessedDP2_$M" --n-per-subject 260 \
      --report "$R/edit_domain_$M.json" || { echo "FAILED gen $M"; exit 1; }
done

# ------------------------------------------------------------- attacks
for M in per_subject per_frame; do
  for F in appleLeftEye appleRightEye; do
    echo "########## periocular A2  DP2_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/periocular_attack.py --ctx 0 \
        --root "$D/ProcessedDP2_$M" --folder "$F" --tag "dp2${M}_$F" \
        --emb-out "$R/emb_dp2${M}_$F.npz" \
        --json-out "$R/peri_dp2${M}_$F.json" || echo "FAILED peri $M $F"

    echo "########## A1 cross-domain  DP2_$M  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/id_attack.py --ctx 0 --no-detect \
        --gallery-root "$D/ProcessedData" --probe-root "$D/ProcessedDP2_$M" \
        --folder "$F" --n-gallery 60 --n-probe 60 --tag "a1_dp2${M}_$F" \
        --json-out "$R/a1_dp2${M}_$F.json" || echo "FAILED a1 $M $F"
  done

  echo "########## T1 self-aligned linkage, face crop, DP2_$M"
  LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
  $PY_ARC scripts/anon/linkage_attack.py --ctx 0 --self-align \
      --probe-root "$D/ProcessedDP2_$M" --folder appleFace \
      --n-per-subject 60 --tag "t1_dp2_$M" \
      --json-out "$R/t1_dp2_$M.json" || echo "FAILED t1 $M"
done

echo ALL_DONE
