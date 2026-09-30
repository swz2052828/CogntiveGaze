#!/bin/bash
#SBATCH --job-name=anon_subval
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_subval_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
# Does a 1200-frame-per-recording subset reproduce full-count utility? If yes,
# every future arm can be generated at a tenth of the inodes and evaluated
# validly, which is the structural fix for the quota problem rather than a
# workaround. The `none` control has known full-manifest 5-fold means
# (face_only 5.428, mobile_vit 5.121, convnextv2 4.331, itracker 5.927) so the
# comparison is against an established number, not a self-consistency check.
BBS=(face_only_mobile_vit mobile_vit convnextv2 itracker)
BB=${BBS[$SLURM_ARRAY_TASK_ID]}
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB
OUT=/springbrook/share/eng/esrpxk/runs/anon_stage4
DP=/springbrook/share/eng/esrpxk/datasets/anon/none
for FOLD in 0 1 2 3 4; do
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean_sub1200 \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root /springbrook/share/eng/esrpxk/datasets/calib_support_K72 \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/sub1200_${BB}_none.csv" \
    --log-file "$OUT/logs/sub1200_${BB}_f${FOLD}.log" || echo "FAILED $BB f$FOLD"
done
echo "ALL_DONE $BB"
