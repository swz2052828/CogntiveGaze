#!/bin/bash
# Step 2: retrain every canonical backbone on the blink-cleaned metadata
# (meanno7_clean). One sbatch array (folds 0-4) per backbone, reusing the
# committed run_base_only_springbrook.sbatch recipe with per-backbone env from
# scripts/clean_canonical_runs.tsv. convnextv2_dualenc is skipped (dead-end).
# Outputs go to runs/meta_pipeline_clean_<run>/base/seed42/.
#
#   bash scripts/submit_clean_retrain.sh            # submit all
#   DRYRUN=1 bash scripts/submit_clean_retrain.sh   # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
MANIFEST="$REPO/scripts/clean_canonical_runs.tsv"
JOBLIST="$REPO/scripts/clean_retrain_jobids.txt"
cd "$REPO"
: > "$JOBLIST"

n=0
while IFS=$'\t' read -r RUN BB OA GR LR EP UG NF; do
  [ "$RUN" = "run_subdir" ] && continue          # header
  # meta_pipeline_<x> -> meta_pipeline_clean_<x> ; shared meta_pipeline -> per-backbone
  OUT_ROOT="$RUNS/meta_pipeline_clean_${RUN#meta_pipeline_}"
  [ "$RUN" = "meta_pipeline" ] && OUT_ROOT="$RUNS/meta_pipeline_clean__shared_${BB}"
  echo "=== $BB  (from $RUN)  oa=$OA gr=$GR lr=$LR ep=$EP -> $(basename "$OUT_ROOT") ==="
  if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
  JID=$(MEAN_PATH=meanno7_clean OUT_ROOT="$OUT_ROOT" BACKBONE="$BB" \
        OUTPUT_ACTIVATION="$OA" GAZE_RANGE="$GR" LR="$LR" EPOCHS="$EP" SEEDS=42 \
        sbatch --parsable --array=0-4 --partition=gpu \
        scripts/run_base_only_springbrook.sbatch)
  echo "  job $JID"
  echo "$JID $BB $(basename "$OUT_ROOT")" >> "$JOBLIST"
  n=$((n+1))
done < "$MANIFEST"
echo "submitted $n backbone arrays (5 folds each). job ids -> $JOBLIST"
