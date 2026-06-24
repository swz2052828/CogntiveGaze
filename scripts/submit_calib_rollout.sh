#!/bin/bash
# Deploy-faithful calibration rollout across all backbones. For each trained run:
#   Stage A: run_calib_metatrain.sbatch  -> retrain base+adv FiLM adapters with
#            pre-task calibration-frame support (meta_on_base_calib/meta_on_adv_calib).
#   Stage B: run_calib_metacompare.sbatch with SVR_TUNE=1 + the *_calib adapters
#            -> calib_metacompare_<bb>_calib.csv (tuned SVR + calib-trained meta).
# Stage B depends on Stage A per fold (--dependency=aftercorr), so fold k's
# metacompare waits for fold k's metatrain. All jobs pinned to --partition=gpu
# (gpu first; QOS cap 15 GPUs schedules them in waves).
#
#   bash scripts/submit_calib_rollout.sh            # all backbones
#   DRYRUN=1 bash scripts/submit_calib_rollout.sh   # print, don't submit
set -euo pipefail
RUNS_DIR=/springbrook/share/eng/esrpxk/runs
cd /springbrook/share/eng/esrpxk/CogntiveGaze

# mobile_vit_rep1 already has _calib adapters + a finished/running _calib sweep.
RUNS="convnext_lr1e4 convnextv2_lr1e4 dinov2_lr1e4 eva02_lr1e4
eyes_only_convnextv2_lr1e4 eyes_only_eva02_tiny_lr1e4 eyes_only_fastvit_lr1e4
eyes_only_mgazenet_lr1e4 eyes_only_mobilenet_v3_lr1e4 eyes_only_mobilenet_v4_lr1e4
eyes_only_mobilevitv2_lr1e4 eyes_only_vit_lr1e4 face_only_mobile_vit_lr1e4
foveal_vit_lr1e4 itracker_lr1e4 mobilevitv2_lr1e4 normface_convnext_lr1e4 repvit_lr1e4"

for r in $RUNS; do
  RD="$RUNS_DIR/meta_pipeline_$r"
  ck=$(ls "$RD"/base/seed42/fold0_best_*_gaze_segmenter.pth 2>/dev/null | head -1 || true)
  adv=$(ls "$RD"/adv/seed42/fold0_best_*_gaze_segmenter.pth 2>/dev/null | head -1 || true)
  mck=$(ls "$RD"/meta_on_base/seed42/fold0_meta_*_gaze.pth 2>/dev/null | head -1 || true)
  if [ -z "$ck" ] || [ -z "$adv" ] || [ -z "$mck" ]; then echo "SKIP $r (missing base/adv/meta ckpt)"; continue; fi
  BB=$(basename "$ck" | sed -E 's/^fold0_best_(.*)_gaze_segmenter.pth$/\1/')
  AD=$(basename "$mck" | sed -E "s/^fold0_meta_([a-z0-9]+)_${BB}_gaze.pth$/\1/")
  echo "=== $r  BACKBONE=$BB ADAPTER=$AD ==="
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "  would: metatrain (array 0-4, gpu) -> metacompare (aftercorr, SVR_TUNE=1, _calib)"
    continue
  fi
  # Stage A: calib metatrain (retrain adapters on calib support)
  AJID=$(RUN_ROOT="$RD" BACKBONE="$BB" ADAPTER="$AD" \
    sbatch --parsable --array=0-4 --partition=gpu scripts/run_calib_metatrain.sbatch)
  echo "  metatrain job $AJID"
  # Stage B: tuned _calib metacompare, fold k waits for metatrain fold k
  BJID=$(SVR_TUNE=1 META_BASE_SUBDIR=meta_on_base_calib META_ADV_SUBDIR=meta_on_adv_calib \
    CSV_TAG=_calib RUN_ROOT="$RD" BACKBONE="$BB" ADAPTER="$AD" \
    sbatch --parsable --array=0-4 --partition=gpu --dependency=aftercorr:"$AJID" \
    scripts/run_calib_metacompare.sbatch)
  echo "  metacompare job $BJID (depends on $AJID)"
done
echo "rollout submitted. Watch: squeue -u \$USER ; results runs/calib_metacompare/calib_metacompare_<bb>_calib.csv"
