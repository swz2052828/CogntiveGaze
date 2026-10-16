#!/bin/bash
# A1 on the FACE crop for every de-identification family: gallery = original
# frames (first half), probe = released face crops (second half). Asks whether the
# synthetic face still matches the participant's real face -- the question raised
# by visual inspection of FAMS outputs. ArcFace (as Table 2) and FaceNet.
#SBATCH --job-name=face_a1_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/face_a1_all_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/face_a1_all
mkdir -p "$R"
ARMS="${ARMS_OVERRIDE:-ProcessedSwap ProcessedSwap2 ProcessedDP2t1_per_subject ProcessedDP2t1_per_frame ProcessedDP2fb_per_subject ProcessedDP2fb_per_frame ProcessedFAMS_per_subject ProcessedFAMS_per_frame}"
for A in $ARMS; do
  echo "########## arcface $A"
  /springbrook/share/eng/esrpxk/envs/idattack/bin/python scripts/anon/id_attack.py --ctx 0 \
      --probe-root "$D/$A" --folder appleFace --n-gallery 20 --n-probe 60 --attributes \
      --tag "face_a1_arcface_$A" --json-out "$R/arcface_$A.json" || echo "FAILED arcface $A"
  echo "########## facenet $A"
  /springbrook/share/eng/esrpxk/envs/gaze/bin/python scripts/anon/id_attack.py --ctx 0 --recogniser facenet --no-detect \
      --probe-root "$D/$A" --folder appleFace --n-gallery 60 --n-probe 60 \
      --tag "face_a1_facenet_$A" --json-out "$R/facenet_$A.json" || echo "FAILED facenet $A"
done
echo ALL_DONE
