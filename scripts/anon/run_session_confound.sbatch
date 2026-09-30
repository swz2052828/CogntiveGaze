#!/bin/bash
# Attacking the single-session confound with the data we have.
#
# A second recording is the decisive experiment and we cannot run one. What we
# can do is test the specific channel the objection names -- session imaging
# conditions, chiefly illumination -- three ways. See session_confound.py for
# the reasoning.
#
#   illum  photometric-only clustering: can brightness statistics ALONE recover
#          the participants? (positive control on the suspected confound)
#   norm   re-embed after removing per-crop brightness/contrast (zscore) and
#          after local histogram equalisation (clahe)
#   block  gallery from dark-screen stimulus blocks, probe from natural-image
#          blocks: a large within-session illumination change
#
# ArcFace, so envs/idattack plus its nvidia libs. The illum probe is pure numpy
# and runs under envs/gaze.
#SBATCH --job-name=sess_conf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/sess_conf_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_session_confound
mkdir -p "$R"
PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
ARC_LIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')

# ---------------------------------------------- 1. illumination-only control
for F in appleLeftEye appleRightEye; do
  for ROOT in ProcessedData ProcessedSwap; do
    echo "########## illum-only  $ROOT  $F"
    $PY scripts/anon/session_confound.py --root "$D/$ROOT" --folder "$F" \
        --tag "illum_${ROOT}_$F" \
        --json-out "$R/illum_${ROOT}_$F.json" || echo "FAILED illum $ROOT $F"
  done
done

# ---------------------------------------------- 2. photometric normalisation
for F in appleLeftEye appleRightEye; do
  for ROOT in ProcessedData ProcessedSwap; do
    for N in zscore clahe; do
      echo "########## normalise=$N  $ROOT  $F"
      LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
      $PY_ARC scripts/anon/periocular_attack.py --ctx 0 --normalise "$N" \
          --root "$D/$ROOT" --folder "$F" --tag "norm${N}_${ROOT}_$F" \
          --emb-out "$R/emb_norm${N}_${ROOT}_$F.npz" \
          --json-out "$R/peri_norm${N}_${ROOT}_$F.json" \
          || echo "FAILED norm $N $ROOT $F"
    done
  done
done

# ---------------------------------------------- 3. cross-task-block split
for F in appleLeftEye appleRightEye; do
  for ROOT in ProcessedData ProcessedSwap; do
    echo "########## taskblock split  $ROOT  $F"
    LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
    $PY_ARC scripts/anon/periocular_attack.py --ctx 0 --split taskblock \
        --root "$D/$ROOT" --folder "$F" --tag "block_${ROOT}_$F" \
        --emb-out "$R/emb_block_${ROOT}_$F.npz" \
        --json-out "$R/peri_block_${ROOT}_$F.json" \
        || echo "FAILED block $ROOT $F"
  done
done

echo ALL_DONE
