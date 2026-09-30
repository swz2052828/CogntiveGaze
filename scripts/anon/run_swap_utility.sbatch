#!/bin/bash
# SimSwap utility in THIS paper's harness, condition (ii): clean enrolment,
# swapped deployment.
#
# Why this exists. Utility figures for these arms already exist from the earlier
# MGazeNet/AFFNet/MobileNet pipeline, but they are mean SQUARED error ("not the
# actual linear error", as those logs say), from different models, without this
# paper's per-subject calibration, and with a fold protocol that the surviving
# logs do not record. They cannot share an axis with the cm values here. This
# re-measures the same four arms with the identical metacompare call used for
# every other operator, so SimSwap can be placed on the same frontier.
#
# Condition (iii) is NOT possible for these arms: SimSwap was applied to the task
# frames only, so no swapped calibration frames exist. Reported as a known gap.
#
# Arms:
#   ProcessedSwap / ProcessedSwap2          face AND eyes replaced
#   *_oldeye                                face replaced, ORIGINAL eyes restored
#   plus a `none` control through the identical path, which must reproduce the
#   published clean number or the harness is not the harness.
#SBATCH --job-name=swap_util
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/swap_util_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/runs
OUT=$R/anon_swap_utility
mkdir -p "$OUT/logs"

BBS=(face_only_mobile_vit mobile_vit convnextv2 itracker)
BB=${BBS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
RR=$R/meta_pipeline_clean_metacmp_$BB
CS=$D/calib_support_K72          # clean enrolment = condition (ii)

for ARM in none:ProcessedData swap1:ProcessedSwap swap1_oldeye:ProcessedSwap_oldeye \
           swap2:ProcessedSwap2 swap2_oldeye:ProcessedSwap2_oldeye; do
  TAG=${ARM%%:*}; ROOT=${ARM#*:}
  DP=$D/$ROOT
  SHORT=""
  for rec in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    for F in appleFace appleLeftEye appleRightEye; do
      H=$(ls "$DP/$rec/$F" 2>/dev/null | wc -l); W=$(ls "$D/ProcessedData/$rec/$F" | wc -l)
      [ "$H" -lt "$W" ] && SHORT="$SHORT $rec/$F($H/$W)"
    done
  done
  [ -n "$SHORT" ] && { echo "FATAL $TAG short:$SHORT"; continue; }
  [ -f "$DP/meanno7_clean/metadata.mat" ] || { echo "FATAL $TAG: no manifest"; continue; }
  echo "########## $TAG $BB fold$FOLD $(date +%T)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/su_${BB}_${TAG}.csv" \
    --log-file "$OUT/logs/su_${BB}_${TAG}_f${FOLD}.log" \
    || echo "FAILED $TAG $BB fold$FOLD"
done
echo "ALL_DONE $BB fold$FOLD"
