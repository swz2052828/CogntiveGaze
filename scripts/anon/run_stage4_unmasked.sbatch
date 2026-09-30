#!/bin/bash
#SBATCH --job-name=anon_s4un
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-19%10
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s4un_%A_%a.log
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
OUT=/springbrook/share/eng/esrpxk/runs/anon_stage4
mkdir -p "$OUT/logs"

for OP in blur0.10 blur0.20 blur0.35 blur0.50 \
          pixel0.25 pixel0.40 blackbox; do
  DP=/springbrook/share/eng/esrpxk/datasets/anon_unmasked/$OP
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  echo "########## $BB fold$FOLD op=$OP $(date +%H:%M:%S)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root /springbrook/share/eng/esrpxk/datasets/calib_support_K72 \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/s4un_${BB}_${OP}.csv" \
    --log-file "$OUT/logs/s4un_${BB}_${OP}_f${FOLD}.log" \
    || echo "FAILED $BB fold$FOLD $OP"
done
echo "ALL_DONE $BB fold$FOLD"
