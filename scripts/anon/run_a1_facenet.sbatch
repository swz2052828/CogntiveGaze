#!/bin/bash
# The one cell of the privacy table that was still single-recogniser.
#
# The release-only (A2) periocular figures are now confirmed by FaceNet
# (results/anon_peri_controls). The A1 claim -- an attacker holding ORIGINAL
# enrolment imagery loses the swapped eye ROIs to the floor, 0.2% / 1.8% -- runs
# through id_attack (gallery = ProcessedData, probe = the swapped root) and is
# still ArcFace-only. That is the claim a reviewer is most likely to challenge,
# because it is the one that sounds like good news for the method.
#
# --no-detect throughout: a FACE detector finds nothing in a 120 px eye crop, and
# detector failure must not be scored as privacy.
#
# FaceNet runs under envs/gaze (torch in envs/idattack is built against cudnn 8
# and only cudnn 9 is installed there). It needs nothing from insightface on the
# --no-detect path.
#SBATCH --job-name=a1_facenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/a1_facenet_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_peri_controls
mkdir -p "$R"
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python

for F in appleLeftEye appleRightEye; do
  for COND in ProcessedData ProcessedSwap ProcessedSwap2; do
    echo "########## A1 facenet  gallery=ProcessedData  probe=$COND  $F"
    $PY scripts/anon/id_attack.py --ctx 0 --recogniser facenet --no-detect \
        --gallery-root "$D/ProcessedData" --probe-root "$D/$COND" --folder "$F" \
        --n-gallery 60 --n-probe 60 --tag "a1_facenet_${COND}_${F}" \
        --json-out "$R/a1_facenet_${COND}_${F}.json" \
        || echo "FAILED a1 facenet $COND $F"
  done
done
echo ALL_DONE
