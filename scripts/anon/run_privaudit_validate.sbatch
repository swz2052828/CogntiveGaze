#!/bin/bash
# Run privaudit on four CognitiveGaze releases and check it reproduces the study.
#   0 ProcessedData              unprocessed; with the face-free floor
#   1 ProcessedSwap              SimSwap identity A; with original (R8) and enrolment (A1)
#   2 ProcessedDP2t1_per_subject DeepPrivacy2 face mode, fixed identity
#   3 ProcessedDP2fb_per_frame   DeepPrivacy2 full body, fresh identity per frame
# Then: python -m privaudit.validate_cognitivegaze $OUT
#SBATCH --job-name=privaudit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/privaudit_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/results/privaudit_validate
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

ROOTS=(ProcessedData ProcessedSwap ProcessedDP2t1_per_subject ProcessedDP2fb_per_frame)
R=${ROOTS[$SLURM_ARRAY_TASK_ID]}
EXTRA=()
case $R in
  ProcessedData) EXTRA=(--floor-frames "$D/OriginalData" --floor-patch 120) ;;
  ProcessedSwap) EXTRA=(--original "$D/ProcessedData" --enrolment "$D/ProcessedData") ;;
esac
echo "########## privaudit $R $(date +%T)"
$PY -m privaudit --release "$D/$R" --streams appleLeftEye appleRightEye appleFace \
    --gpu 0 --out "$OUT/$R" "${EXTRA[@]}" || { echo "FAILED $R"; exit 1; }
echo "ALL_DONE $R $(date +%T)"
