#!/bin/bash
# Step 2 (adv): retrain the subject-adversarial (DANN) variant of every canonical
# backbone on blink-cleaned metadata. Same recipe as the base clean-retrain plus
# --subject-adv --adv-weight 0.1 --adv-warmup-frac 1.0 (the uniform adv recipe
# read from runs/meta_pipeline_*/adv saved args). Reuses run_base_only_springbrook.sbatch
# via EXTRA_TRAIN_ARGS. Skips cnn_transformer_raw (no adv variant by design).
# Outputs to runs/meta_pipeline_clean_adv_<run>/base/seed42/.
#
#   bash scripts/submit_clean_retrain_adv.sh            # submit all 33
#   DRYRUN=1 bash scripts/submit_clean_retrain_adv.sh   # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
MANIFEST="$REPO/scripts/clean_canonical_runs.tsv"
JOBLIST="$REPO/scripts/clean_retrain_adv_jobids.txt"
ADV_ARGS="--subject-adv --adv-weight 0.1 --adv-warmup-frac 1.0"
cd "$REPO"
: > "$JOBLIST"

n=0
while IFS=$'\t' read -r RUN BB OA GR LR EP UG NF; do
  [ "$RUN" = "run_subdir" ] && continue                 # header
  [ "$BB" = "cnn_transformer_raw" ] && continue          # no adv variant (CLS fusion opts out)
  OUT_ROOT="$RUNS/meta_pipeline_clean_adv_${RUN#meta_pipeline_}"
  [ "$RUN" = "meta_pipeline" ] && OUT_ROOT="$RUNS/meta_pipeline_clean_adv__shared_${BB}"
  echo "=== adv $BB  (from $RUN)  oa=$OA gr=$GR lr=$LR ep=$EP -> $(basename "$OUT_ROOT") ==="
  if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
  JID=$(MEAN_PATH=meanno7_clean OUT_ROOT="$OUT_ROOT" BACKBONE="$BB" \
        OUTPUT_ACTIVATION="$OA" GAZE_RANGE="$GR" LR="$LR" EPOCHS="$EP" SEEDS=42 \
        EXTRA_TRAIN_ARGS="$ADV_ARGS" \
        sbatch --parsable --array=0-4 --partition=gpu \
        scripts/run_base_only_springbrook.sbatch)
  echo "  job $JID"
  echo "$JID $BB $(basename "$OUT_ROOT")" >> "$JOBLIST"
  n=$((n+1))
done < "$MANIFEST"
echo "submitted $n adv backbone arrays (5 folds each). job ids -> $JOBLIST"
