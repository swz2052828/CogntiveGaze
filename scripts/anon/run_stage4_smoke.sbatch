#!/bin/bash
#SBATCH --job-name=anon_s4_smoke
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s4_smoke_%j.log
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1

# Stage 4a smoke: zero-shot utility. Models trained on clean data, evaluated on
# anonymised data with NO retraining. The `base` column is the number of interest
# and it does not depend on the calibration support at all; the calibrated columns
# here use the ORIGINAL (un-anonymised) enrolment, which is a legitimate but
# DIFFERENT scenario -- clean enrolment, anonymised deployment -- and must be
# labelled as such, not as the deployable figure. Fully-anonymised enrolment
# needs anonymised calib support, which is blocked on calibration geometry.
BB=face_only_mobile_vit
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB
OUT=/springbrook/share/eng/esrpxk/runs/anon_stage4
mkdir -p "$OUT/logs"
# Call the venv interpreter directly; `source activate` did not take under
# sbatch and the job ran against a system python with no numpy.
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"

for OP in none blur0.50; do
  DP=/springbrook/share/eng/esrpxk/datasets/anon/$OP
  echo "########## $BB fold0 op=$OP"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold0_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold0_meta_film_${BB}_gaze.pth" \
    --calib-support-root /springbrook/share/eng/esrpxk/datasets/calib_support_K72 \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index 0 --seed 42 \
    --csv-out "$OUT/s4_smoke_${BB}.csv" \
    --log-file "$OUT/logs/s4_smoke_${BB}_${OP}.log"
done
echo ALL_DONE
