#!/bin/bash
#SBATCH --job-name=anon_s4
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s4_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"

# Stage 4a: zero-shot utility, condition (ii) -- clean enrolment, anonymised
# deployment. Models are the clean-trained checkpoints, unmodified; the
# calibration support is the ORIGINAL K72 set. This is a legitimate deployment
# story in its own right (enrol in a trusted setting, deploy anonymised), not an
# approximation to condition (iii), which needs anonymised enrolment.
# The `base` column is condition-independent: it never touches the support.
BBS=(face_only_mobile_vit mobile_vit convnextv2 itracker)
BB=${BBS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB
: "${K:=72}"
OUT=/springbrook/share/eng/esrpxk/runs/anon_stage4${KTAG:-}
mkdir -p "$OUT/logs"

for OP in none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
          pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox; do
  DP=/springbrook/share/eng/esrpxk/datasets/anon/$OP
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  # Coverage guard. A root generated at --limit for the attack side has ~10% of
  # the frames the utility manifest expects; the evaluator warns per frame and
  # proceeds, producing plausible-looking numbers. Abort instead.
  # Check EVERY recording, not just the first: a root can be complete at 00006
  # and truncated later, which is exactly how a timed-out generation job fails.
  SHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
    W=$(ls "/springbrook/share/eng/esrpxk/datasets/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
  done
  if [ -n "$SHORT" ]; then
    echo "FATAL $OP: short recordings:$SHORT -- refusing to evaluate a partial root"
    continue
  fi
  echo "########## $BB fold$FOLD op=$OP $(date +%H:%M:%S)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root /springbrook/share/eng/esrpxk/datasets/calib_support_K$K \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k "$K" --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/s4_${BB}_${OP}.csv" \
    --log-file "$OUT/logs/s4_${BB}_${OP}_f${FOLD}.log" \
    || echo "FAILED $BB fold$FOLD $OP"
done
echo "ALL_DONE $BB fold$FOLD"
