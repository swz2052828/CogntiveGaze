#!/bin/bash
# FAMS anonymisation-strength sweep on the 260-frame privacy sample, per-frame
# noise (as the utility run). Degree 1.25 is the authors' setting and already
# exists (ProcessedFAMS_per_frame). The degree extrapolates the source-identity
# conditioning toward its nullified version (pipeline_referencenet.py,
# nullify_referencenet_states): larger = further from the source face.
#   MODE=gen     array 0-$((18*NDEG-1)): one task per (degree, participant)
#   MODE=attack  array 0-$((NDEG-1)):    face A1 (ArcFace+FaceNet), face linkage,
#                                         eye A1, eye A2 + bootstrap, per degree
#SBATCH --job-name=fams_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/fams_sweep_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
export HF_HOME=/springbrook/share/eng/esrpxk/home_caches/huggingface HF_HUB_OFFLINE=1
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/fams_sweep
mkdir -p "$R"
read -ra DEGS <<< "${DEGS:-1.5 2.0 3.0}"
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)
case "${MODE:?gen|attack}" in
gen)
  DEG=${DEGS[$(( SLURM_ARRAY_TASK_ID / 18 ))]}; REC=${RECS[$(( SLURM_ARRAY_TASK_ID % 18 ))]}
  /springbrook/share/eng/esrpxk/envs/fams/bin/python scripts/anon/run_fams.py --identity per_frame \
      --degree "$DEG" --recs $REC --n-per-subject 260 --resume \
      --out-root "$D/ProcessedFAMSd${DEG}_per_frame" --report "$R/edit_d${DEG}_${REC}.json" \
      || { echo "FAILED $DEG $REC"; exit 1; } ;;
attack)
  DEG=${DEGS[$SLURM_ARRAY_TASK_ID]}; ROOT="$D/ProcessedFAMSd${DEG}_per_frame"; T="d${DEG}"
  PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
  PY_ARC=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
  SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
  export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
  $PY_ARC scripts/anon/id_attack.py --ctx 0 --probe-root "$ROOT" --folder appleFace --n-gallery 20 \
      --n-probe 60 --attributes --tag "face_a1_arcface_$T" --json-out "$R/face_a1_arcface_$T.json" || echo "FAILED face a1 arcface"
  $PY scripts/anon/id_attack.py --ctx 0 --recogniser facenet --no-detect --probe-root "$ROOT" --folder appleFace \
      --n-gallery 60 --n-probe 60 --tag "face_a1_facenet_$T" --json-out "$R/face_a1_facenet_$T.json" || echo "FAILED face a1 facenet"
  $PY_ARC scripts/anon/linkage_attack.py --ctx 0 --self-align --probe-root "$ROOT" --folder appleFace \
      --n-per-subject 60 --tag "t1_$T" --json-out "$R/t1_$T.json" || echo "FAILED t1"
  for F in appleLeftEye appleRightEye; do
    $PY_ARC scripts/anon/periocular_attack.py --ctx 0 --root "$ROOT" --folder $F \
        --tag "peri_${T}_$F" --emb-out "$R/emb_${T}_$F.npz" --json-out "$R/peri_${T}_$F.json" || echo "FAILED peri $F"
    $PY_ARC scripts/anon/id_attack.py --ctx 0 --no-detect --gallery-root "$D/ProcessedData" --probe-root "$ROOT" \
        --folder $F --n-gallery 60 --n-probe 60 --tag "a1_${T}_$F" --json-out "$R/a1_${T}_$F.json" || echo "FAILED a1 $F"
    $PY scripts/anon/ari_bootstrap.py "$R/emb_${T}_$F.npz" --tag "${T}_$F" --json-out "$R/ari_${T}_$F.json" || echo "FAILED boot $F"
  done ;;
esac
echo ALL_DONE
