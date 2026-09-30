#!/bin/bash
# Initialisation-variance replication for the privacy-utility utility axis.
#
# WHY THE SEED IS HELD FIXED. `--seed` in this codebase reaches exactly one
# place: recording_kfolds(), which permutes which SUBJECTS land in which fold.
# There is no torch.manual_seed / np.random.seed anywhere in vit_gaze, so weight
# initialisation, DataLoader order and dropout are unseeded. Re-running training
# at the SAME seed therefore holds the subject partition fixed and redraws the
# initialisation -- which is exactly the variance component we need, with the
# split held constant. Changing the seed would confound the two.
#
# Four independent repeats per (backbone, fold), reusing the exact recipe that
# produced the seed42 checkpoints (run_base_only_springbrook.sbatch, driven by
# clean_canonical_runs.tsv: lr 1e-4, 20 epochs, meanno7_clean, use-grid).
# Together with the existing run that gives 5 draws.
#
# Repeats are chained so only one is ever in flight: peak is 4 backbones x 5
# folds, but three of the four finish in ~20-35 min, so after the first half hour
# only the five iTracker tasks remain (iTracker is ~8.5 h/fold -- LocalResponse
# Normalisation has no fast cuDNN path). Expect ~9 h per repeat, ~36 h total.
#
#   bash scripts/submit_initvar.sh            # submit
#   DRYRUN=1 bash scripts/submit_initvar.sh   # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
JOBLIST="$REPO/scripts/initvar_jobids.txt"
cd "$REPO"
: > "$JOBLIST"

# Slowest first so it starts earliest within each repeat.
BACKBONES=(itracker affnet mgazenet mobilenet_v3)
REPS="${REPS:-1 2 3 4}"

PREV=""
for REP in $REPS; do
  STAGE_IDS=()
  for BB in "${BACKBONES[@]}"; do
    OUT_ROOT="$RUNS/initvar/rep${REP}/${BB}"
    DEP=""
    [ -n "$PREV" ] && DEP="--dependency=afterany:${PREV}"
    echo "=== rep$REP $BB -> $OUT_ROOT ${DEP:+(after $PREV)} ==="
    if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
    mkdir -p "$OUT_ROOT"
    JID=$(MEAN_PATH=meanno7_clean OUT_ROOT="$OUT_ROOT" BACKBONE="$BB" \
          OUTPUT_ACTIVATION=none GAZE_RANGE=4.0 LR=1e-4 EPOCHS=20 SEEDS=42 \
          sbatch --parsable --array=0-4 --partition=gpu \
          --job-name="iv${REP}_${BB}" \
          $DEP \
          scripts/run_base_only_springbrook.sbatch)
    echo "  job $JID"
    echo "$JID rep$REP $BB" >> "$JOBLIST"
    STAGE_IDS+=("$JID")
  done
  [ "${DRYRUN:-0}" = "1" ] || PREV=$(IFS=:; echo "${STAGE_IDS[*]}")
done
echo
echo "submitted. job ids -> $JOBLIST"
echo "when training is done:  sbatch scripts/anon/run_initvar_eval.sbatch"
