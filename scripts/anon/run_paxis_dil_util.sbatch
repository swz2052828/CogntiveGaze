#!/bin/bash
#SBATCH --job-name=paxis_dilu
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_dilu_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/runs/anon_paxis_dil
mkdir -p "$OUT/logs"
# THE P-axis experiment. Privacy is already known along this axis:
#   P0 (whole eye crop)  d' 5.12, TAR@1e-3 99.3%
#   P1 (opening only)    d' 0.27, TAR@1e-3  0.0%   <- below the background floor
#   P2 (iris disc only)  d' 0.07, TAR@1e-3  0.0%
# So the open question is entirely utility: does gaze survive at P1?
# All three arms use the SAME subset manifest (meanno7_clean_paxis, 18,963 rows
# from the 1200-frames-per-recording P-axis roots) so the comparison is
# like-for-like. P0 is anon/blackbox, which is full-count and therefore a
# superset of the subset -- valid, and the guard below checks it.
MEAN=meanno7_clean_paxis
CS=/springbrook/share/eng/esrpxk/datasets/calib_support_K72
# Two backbones, and the second one is a FALSIFICATION control, not a second
# measurement. face_only_mobile_vit has no eye stream: it never reads the pixels
# the P-axis operators modify, so its P0/P1/P2 columns MUST agree to within the
# init-noise floor. If they move, the effect being attributed to the eye
# operators is coming from somewhere else (manifest, ordering, calibration
# support) and the mobile_vit numbers cannot be trusted either.
BB=mobile_vit
FOLD=$SLURM_ARRAY_TASK_ID
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_${BB}
for ARM in P1d5:$D/anon_paxis/P1d5 P1d10:$D/anon_paxis/P1d10 P1d20:$D/anon_paxis/P1d20 P1d40:$D/anon_paxis/P1d40; do
  NAME=${ARM%%:*}; DP=${ARM#*:}
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  SHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt 1200 ] && SHORT="$SHORT $R($H)"
  done
  [ -n "$SHORT" ] && { echo "FATAL $NAME: short recordings:$SHORT"; continue; }
  # Manifest guard. --mean-path resolves RELATIVE TO --data-path, so a root that
  # is complete on frames can still be unevaluable because the manifest was
  # created after the root and never linked into it. That is how the first
  # attempt at this job died (1271776: three arms, FileNotFoundError, 7s).
  [ -f "$DP/$MEAN/metadata.mat" ] || { echo "FATAL $NAME: no $MEAN/metadata.mat in $DP"; continue; }
  # --mean-path is ALSO resolved inside --calib-support-root, whose manifest
  # describes the calibration frames and is therefore independent of which task
  # manifest is in use -- every name there is an alias of meanno7. Missing alias
  # = FileNotFoundError after the task set has already loaded (job 1271857).
  [ -f "$CS/$MEAN/metadata.mat" ] || { echo "FATAL: no $MEAN alias in $CS"; continue; }
  echo "########## $BB fold$FOLD arm=$NAME $(date +%H:%M:%S)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path "$MEAN" \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/pu_${BB}_${NAME}.csv" \
    --log-file "$OUT/logs/pu_${BB}_${NAME}_f${FOLD}.log" \
    || echo "FAILED $BB fold$FOLD $NAME"
done
echo "ALL_DONE fold$FOLD"
