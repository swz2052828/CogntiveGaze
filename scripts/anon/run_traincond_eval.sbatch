#!/bin/bash
#SBATCH --job-name=tc_eval
#SBATCH --partition=gpu
# NOT gpu,vis: the vis QOS caps a user at ONE gpu, so a multi-task array that
# spills onto vis loses tasks to QOSMaxGRESPerUser mid-array (cost job 1272012
# its itracker folds).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/tc_eval_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/runs/anon_traincond
mkdir -p "$OUT/logs"

# Evaluate the condition (i) retrain and the fine-tune arms through the SAME
# metacompare pipeline as conditions (ii)/(iii), so all four protocols are
# directly comparable. Enrolment is the matched anonymised support, i.e. the
# condition (iii) setting -- a model trained on anonymised data being enrolled
# on clean frames would be an incoherent deployment.
#
# convnextv2 RETRAIN is deliberately absent: all 10 of its runs collapsed to
# predict-the-mean (train_loss frozen at 0.400, val_coord_error 9.0-9.7 against
# a 4.33 clean baseline, identical across both operators). That is the known
# convnextv2 init instability, not an anonymisation effect, and it is being
# re-run under other seeds before it is evaluated.
ARMS=(finetune_mobile_vit finetune_convnextv2 retrain_mobile_vit)
A=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
PROTO=${A%%_*}                 # finetune | retrain
BB=${A#*_}                     # mobile_vit | convnextv2
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB

for OP in blackbox blur0.20; do
  DP=$D/anon/$OP
  CS=$D/calib_support_K72_$OP
  CK=/springbrook/share/eng/esrpxk/runs/anon_${PROTO}/${BB}_${OP}/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth
  [ -f "$CK" ] || { echo "FATAL no checkpoint $CK"; continue; }
  [ -d "$DP" ] || { echo "FATAL missing $DP"; continue; }
  [ -f "$CS/meanno7_clean/metadata.mat" ] || { echo "FATAL no manifest in $CS"; continue; }
  SHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
    W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
    C=$(ls "$CS/$R/appleFace" 2>/dev/null | wc -l)
    [ "$C" -lt 72 ] && SHORT="$SHORT calib:$R($C)"
  done
  [ -n "$SHORT" ] && { echo "FATAL $OP: short:$SHORT"; continue; }

  echo "########## $PROTO $BB fold$FOLD op=$OP $(date +%H:%M:%S)"
  # --meta-checkpoint is required by the CLI but the adapter was fitted over the
  # CLEAN base features; against a retrained base its column is meaningless.
  # Pass the clean one to satisfy the argument and DISCARD `meta` downstream.
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$CK" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/tc_${PROTO}_${BB}_${OP}.csv" \
    --log-file "$OUT/logs/tc_${PROTO}_${BB}_${OP}_f${FOLD}.log" \
    || echo "FAILED $PROTO $BB fold$FOLD $OP"
done
echo "ALL_DONE $A fold$FOLD"
