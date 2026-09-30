#!/bin/bash
# Two controls the privacy axis is missing, both cheap, both asked for by the
# review of our own draft:
#
#  A. A PERIOCULAR-SPECIFIC FLOOR. Every ARI/TAR floor quoted so far comes from
#     t1bg_p120, which runs the FACE-crop pipeline on a face-free patch. That
#     controls for room and session, but it is not strictly an eye-crop number.
#     --background 120 runs the periocular pipeline itself on a 120 px face-free
#     corner of the same frames, so the floor and the eye-crop figures differ in
#     pixels alone.
#
#  B. A SECOND RECOGNISER FAMILY. Every identity figure comes from ArcFace
#     (ResNet-50 / WebFace600K / margin softmax). FaceNet (Inception-ResNet-v1 /
#     VGGFace2 / triplet) is architecturally unrelated and trained on a different
#     corpus. Agreement is evidence about the imagery; disagreement bounds the
#     claim. Run on the arms the paper's headline rests on.
#
# ArcFace needs envs/idattack (onnxruntime + its nvidia libs). torch in that env
# is broken (built against cudnn 8, only cudnn 9 is installed), so FaceNet runs
# under envs/gaze instead. The attacks never detect on eye crops, so FaceNet
# needs nothing from insightface.
#SBATCH --job-name=peri_ctrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/peri_ctrl_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_peri_controls
mkdir -p "$R"

PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
PY_FN=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
ARC_LIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')

# ---------------------------------------------------------------- A: floor
for F in appleLeftEye appleRightEye; do
  echo "########## A  periocular floor (arcface, 120px face-free patch)  $F"
  LD_LIBRARY_PATH="${ARC_LIBS}${LD_LIBRARY_PATH:-}" \
  $PY_ARC scripts/anon/periocular_attack.py --ctx 0 \
      --root "$D/ProcessedData" --folder "$F" --background 120 \
      --tag "floor_arcface_$F" \
      --emb-out "$R/emb_floor_arcface_$F.npz" \
      --json-out "$R/peri_floor_arcface_$F.json" || echo "FAILED floor arcface $F"
done

# ------------------------------------------------- B: FaceNet second opinion
for F in appleLeftEye appleRightEye; do
  for ROOT in ProcessedData ProcessedSwap ProcessedSwap2; do
    echo "########## B  facenet  $ROOT  $F"
    $PY_FN scripts/anon/periocular_attack.py --ctx 0 --recogniser facenet \
        --root "$D/$ROOT" --folder "$F" \
        --tag "facenet_${ROOT}_$F" \
        --emb-out "$R/emb_facenet_${ROOT}_$F.npz" \
        --json-out "$R/peri_facenet_${ROOT}_$F.json" || echo "FAILED facenet $ROOT $F"
  done
  echo "########## B  facenet FLOOR  $F"
  $PY_FN scripts/anon/periocular_attack.py --ctx 0 --recogniser facenet \
      --root "$D/ProcessedData" --folder "$F" --background 120 \
      --tag "floor_facenet_$F" \
      --emb-out "$R/emb_floor_facenet_$F.npz" \
      --json-out "$R/peri_floor_facenet_$F.json" || echo "FAILED floor facenet $F"
done

echo ALL_DONE
