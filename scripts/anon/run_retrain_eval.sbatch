#!/bin/bash
# Evaluate the Plan D training protocols on anonymised data.
#
# Three ways to get a gaze model onto anonymised imagery, on identical cells
# (mobile_vit / convnextv2 x blur0.20 / blackbox x 5 folds):
#   zeroshot  clean-trained model, deployed on anonymised data     (Stage 4, exists)
#   retrain   trained from scratch on anonymised data              (seed 42; convnextv2 also 1, 7)
#   finetune  clean model fine-tuned on anonymised data            (seed 42)
#
# Enrolment is MATCHED (calib_support_K{K}_<op>) -- condition (iii), the only
# deployable one -- at K=72 and K=9, so every number here lines up with
# anon_cond3_RESULT.md and anon_cond3_K9_RESULT.md.
#
# Only base / fc_ft / svr_embed read --base-checkpoint. The meta column is the
# CLEAN meta model evaluated on anonymised data and means nothing here; it is
# carried only because the CLI requires the argument.
#
# The `zeroshot` protocol exists for ONE purpose: the smoke run must reproduce the
# existing condition (iii) CSV for the same cell to the digit, which proves this
# harness is the same harness before any retrain number is read against it.
#SBATCH --job-name=rt_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/rt_eval_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/runs
OUT=$R/anon_retrain_eval
mkdir -p "$OUT/logs"

# task list: proto:seed:bb:op:fold
TASKS=()
for BB in mobile_vit convnextv2; do for OP in blackbox blur0.20; do for F in 0 1 2 3 4; do
  TASKS+=("retrain:42:$BB:$OP:$F" "finetune:42:$BB:$OP:$F")
done; done; done
for S in 1 7; do for OP in blackbox blur0.20; do for F in 0 1 2 3 4; do
  TASKS+=("retrain:$S:convnextv2:$OP:$F")
done; done; done
TASKS+=("zeroshot:42:mobile_vit:blur0.20:0")          # index 60: harness anchor
: "${ONLY_TASK:=}"
T=${ONLY_TASK:-${TASKS[$SLURM_ARRAY_TASK_ID]}}
IFS=: read PROTO SEED BB OP FOLD <<< "$T"
: "${KS:=72 9}"

case $PROTO in
  retrain)  CK=$R/anon_retrain/${BB}_${OP}/seed${SEED}/fold${FOLD}_best_${BB}_gaze_segmenter.pth ;;
  finetune) CK=$R/anon_finetune/${BB}_${OP}/seed${SEED}/fold${FOLD}_best_${BB}_gaze_segmenter.pth ;;
  zeroshot) CK=$R/meta_pipeline_clean_metacmp_${BB}/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth ;;
esac
META=$R/meta_pipeline_clean_metacmp_${BB}/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth
DP=$D/anon/$OP
[ -f "$CK" ] || { echo "FATAL missing checkpoint $CK"; exit 1; }
[ -d "$DP" ] || { echo "FATAL missing deployment root $DP"; exit 1; }

for K in $KS; do
  CS=$D/calib_support_K${K}_$OP
  [ -f "$CS/meanno7_clean/metadata.mat" ] || { echo "FATAL no manifest in $CS"; continue; }
  TAG=${PROTO}_s${SEED}_${BB}_${OP}_K${K}
  echo "########## $TAG fold$FOLD $(date +%T)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$CK" --meta-checkpoint "$META" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k "$K" --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/${TAG}_f${FOLD}.csv" \
    --log-file "$OUT/logs/${TAG}_f${FOLD}.log" \
    || echo "FAILED $TAG fold$FOLD"
done
echo "ALL_DONE $T"
