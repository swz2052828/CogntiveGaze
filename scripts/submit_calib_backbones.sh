#!/bin/bash
# Submit the deterministic-calibration metacompare sweep (K=4/9/18/36/72, all
# methods incl base_adv/fc_ft/meta_adv/svr_embed) for a curated set of trained
# backbones. Backbone + adapter names are read from each run's checkpoint files.
#   bash scripts/submit_calib_backbones.sh
set -euo pipefail
RUNS_DIR=/springbrook/share/eng/esrpxk/runs
cd /springbrook/share/eng/esrpxk/CogntiveGaze

RUNS="convnext_lr1e4 convnextv2_lr1e4 dinov2_lr1e4 eva02_lr1e4
eyes_only_convnextv2_lr1e4 eyes_only_eva02_tiny_lr1e4 eyes_only_fastvit_lr1e4
eyes_only_mgazenet_lr1e4 eyes_only_mobilenet_v3_lr1e4 eyes_only_mobilenet_v4_lr1e4
eyes_only_mobilevitv2_lr1e4 eyes_only_vit_lr1e4 face_only_mobile_vit_lr1e4
foveal_vit_lr1e4 itracker_lr1e4 mobile_vit_rep1 mobilevitv2_lr1e4
normface_convnext_lr1e4 repvit_lr1e4"

for r in $RUNS; do
  RD="$RUNS_DIR/meta_pipeline_$r"
  ck=$(ls "$RD"/base/seed42/fold0_best_*_gaze_segmenter.pth 2>/dev/null | head -1 || true)
  mck=$(ls "$RD"/meta_on_base/seed42/fold0_meta_*_gaze.pth 2>/dev/null | head -1 || true)
  mack=$(ls "$RD"/meta_on_adv/seed42/fold0_meta_*_gaze.pth 2>/dev/null | head -1 || true)
  if [ -z "$ck" ] || [ -z "$mck" ] || [ -z "$mack" ]; then echo "SKIP $r (missing ckpt)"; continue; fi
  BB=$(basename "$ck" | sed -E 's/^fold0_best_(.*)_gaze_segmenter.pth$/\1/')
  AD=$(basename "$mck" | sed -E "s/^fold0_meta_([a-z0-9]+)_${BB}_gaze.pth$/\1/")
  echo "submit $r  BACKBONE=$BB ADAPTER=$AD"
  RUN_ROOT="$RD" BACKBONE="$BB" ADAPTER="$AD" \
    sbatch --array=0-4 scripts/run_calib_metacompare.sbatch
done
