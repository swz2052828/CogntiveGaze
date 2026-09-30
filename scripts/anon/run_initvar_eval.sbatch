#!/bin/bash
# Stage 2 of the initialisation-variance replication: evaluate every repeat's
# base checkpoints on the five de-identification arms, exactly as
# run_swap_utility.sbatch does for the seed42 draw.
#
# The meta checkpoint is REUSED from the seed42 run. It is a self-contained
# model, loaded only to fill the `meta` column, and it does not touch `base`,
# `svr`, `svr_embed` or `fc_ft`, which all derive from --base-checkpoint. The
# `meta` column of these CSVs is therefore MEANINGLESS (it pairs a repeat's base
# with seed42's adapter) and must not be read. We report `base`, which is the
# no-per-subject-calibration number the manuscript's Table 3 reports.
#
# array index -> (rep, backbone, fold):  rep = id/20 + 1, bb = (id%20)/5, fold = id%5
#SBATCH --job-name=iv_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-79%6
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/iv_eval_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/runs
OUT=$R/initvar_utility
mkdir -p "$OUT/logs"

BBS=(itracker affnet mgazenet mobilenet_v3)
ID=$SLURM_ARRAY_TASK_ID
REP=$(( ID / 20 + 1 ))
BB=${BBS[$(( (ID % 20) / 5 ))]}
FOLD=$(( ID % 5 ))

BASE_CKPT="$R/initvar/rep${REP}/${BB}/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth"
META_CKPT="$R/meta_pipeline_clean_metacmp_${BB}/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth"
CS=$D/calib_support_K72

[ -f "$BASE_CKPT" ] || { echo "FATAL missing base ckpt $BASE_CKPT"; exit 1; }
[ -f "$META_CKPT" ] || { echo "FATAL missing meta ckpt $META_CKPT"; exit 1; }

for ARM in none:ProcessedData swap1:ProcessedSwap swap1_oldeye:ProcessedSwap_oldeye \
           swap2:ProcessedSwap2 swap2_oldeye:ProcessedSwap2_oldeye; do
  TAG=${ARM%%:*}; ROOT=${ARM#*:}
  DP=$D/$ROOT
  [ -f "$DP/meanno7_clean/metadata.mat" ] || { echo "FATAL $TAG: no manifest"; continue; }
  echo "########## rep$REP $BB fold$FOLD $TAG $(date +%T)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$BASE_CKPT" \
    --meta-checkpoint "$META_CKPT" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/iv_rep${REP}_${BB}_${TAG}.csv" \
    --log-file "$OUT/logs/iv_rep${REP}_${BB}_${TAG}_f${FOLD}.log" \
    || echo "FAILED rep$REP $BB fold$FOLD $TAG"
done
echo "ALL_DONE rep$REP $BB fold$FOLD"
