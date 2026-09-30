#!/bin/bash
#SBATCH --job-name=anon_s4c3
#SBATCH --partition=gpu
# NOT gpu,vis. The `vis` QOS caps a user at ONE gpu (`gpu` allows 15), so any
# array with concurrency >1 that spills onto vis has its extra tasks cancelled
# with QOSMaxGRESPerUser -- silently, mid-array. That is what truncated job
# 1272012 (itracker folds 1-4). Multi-task arrays: gpu only.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s4c3_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"

# Stage 4c: condition (iii) -- anonymised enrolment AND anonymised deployment.
#
# Condition (ii) (run_stage4_full.sbatch) enrols on ORIGINAL calibration frames
# and deploys on anonymised ones. That is a real deployment story, but it is not
# the one a release-only pipeline can offer: if the originals never leave the
# lab, the calibration support cannot be original either. Per-subject calibration
# works by fitting each subject's systematic bias, so feeding it clean faces
# while deployment sees anonymised ones is a mismatch that flatters the result.
#
# The only change from Stage 4a is --calib-support-root: K72_$OP instead of K72.
# `none` is NOT the untouched original set -- it was regenerated through the
# identical decode/encode path (build_anon_calib_support.py), so the control
# carries the same interpolation count as the operator arms and the comparison
# isolates the operator rather than the re-encode.
BBS=(face_only_mobile_vit mobile_vit convnextv2 itracker)
BB=${BBS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB
: "${K:=72}"
OUT=/springbrook/share/eng/esrpxk/runs/anon_stage4_cond3${KTAG:-}
D=/springbrook/share/eng/esrpxk/datasets
mkdir -p "$OUT/logs"

for OP in none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
          pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox; do
  DP=$D/anon/$OP
  CS=$D/calib_support_K${K}_$OP
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  [ -d "$CS" ] || { echo "SKIP missing $CS"; continue; }
  # Coverage guard on the deployment root: a --limit root has ~10% of the frames
  # the manifest expects and the evaluator warns per frame then proceeds,
  # producing plausible-looking wrong numbers. Check EVERY recording.
  SHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
    W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
  done
  [ -n "$SHORT" ] && { echo "FATAL $OP: short recordings:$SHORT"; continue; }
  # Enrolment guard: the support root must resolve the manifest name AND hold
  # K frames for every recording, or calibration silently runs on fewer.
  [ -f "$CS/meanno7_clean/metadata.mat" ] || { echo "FATAL $OP: no manifest in $CS"; continue; }
  CSHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$CS/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt "$K" ] && CSHORT="$CSHORT $R($H)"
  done
  [ -n "$CSHORT" ] && { echo "FATAL $OP: short enrolment:$CSHORT"; continue; }

  echo "########## $BB fold$FOLD op=$OP cond=iii $(date +%H:%M:%S)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k "$K" --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/s4c3_${BB}_${OP}.csv" \
    --log-file "$OUT/logs/s4c3_${BB}_${OP}_f${FOLD}.log" \
    || echo "FAILED $BB fold$FOLD $OP"
done
echo "ALL_DONE $BB fold$FOLD"
