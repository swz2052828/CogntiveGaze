#!/bin/bash
# DP2 on the utility axis: all five initialisation draws x 4 backbones x 5 folds,
# the exact metacompare call of run_initvar_eval.sbatch, so the DP2 rows join
# Table 3 without any change of harness.
#
# `none` is re-run through the same path as a HARNESS CHECK: its `base` column
# must reproduce the existing none CSV for that draw (check_dp2_util.py gates it).
# The `meta` column pairs draws 1-4 with seed42's adapter and must not be read.
#
# array index -> (draw, backbone, fold): draw = id/20, bb = (id%20)/5, fold = id%5
# draw 0 = runs/meta_pipeline_clean_metacmp_<BB>, draws 1-4 = runs/initvar/rep<N>
#SBATCH --job-name=dp2_util
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-99%10
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_util_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/runs
OUT=$R/dp2_utility
mkdir -p "$OUT/logs"

BBS=(itracker affnet mgazenet mobilenet_v3)
ID=$SLURM_ARRAY_TASK_ID
DRAW=$(( ID / 20 ))
BB=${BBS[$(( (ID % 20) / 5 ))]}
FOLD=$(( ID % 5 ))
if [ "$DRAW" -eq 0 ]; then B=$R/meta_pipeline_clean_metacmp_${BB}; else B=$R/initvar/rep${DRAW}/${BB}; fi
BASE_CKPT="$B/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth"
META_CKPT="$R/meta_pipeline_clean_metacmp_${BB}/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth"
[ -f "$BASE_CKPT" ] || { echo "FATAL missing $BASE_CKPT"; exit 1; }
[ -f "$META_CKPT" ] || { echo "FATAL missing $META_CKPT"; exit 1; }

# ARMS (env) overrides the arm list, e.g. for the full-body roots.
ARMS=${ARMS:-"none:ProcessedData dp2s:ProcessedDP2full_per_subject dp2s_oldeye:ProcessedDP2full_per_subject_oldeye dp2f:ProcessedDP2full_per_frame dp2f_oldeye:ProcessedDP2full_per_frame_oldeye"}
for ARM in $ARMS; do
  TAG=${ARM%%:*}; DP=$D/${ARM#*:}
  [ -f "$DP/meanno7_clean/metadata.mat" ] || { echo "FATAL $TAG: no manifest"; continue; }
  echo "########## draw$DRAW $BB fold$FOLD $TAG $(date +%T)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$BASE_CKPT" --meta-checkpoint "$META_CKPT" \
    --calib-support-root "$D/calib_support_K72" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/dp2_d${DRAW}_${BB}_${TAG}.csv" \
    --log-file "$OUT/logs/dp2_d${DRAW}_${BB}_${TAG}_f${FOLD}.log" \
    || echo "FAILED draw$DRAW $BB fold$FOLD $TAG"
done
echo "ALL_DONE draw$DRAW $BB fold$FOLD"
