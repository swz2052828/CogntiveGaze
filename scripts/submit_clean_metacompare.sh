#!/bin/bash
# Stage B of the CLEAN metacompare: full 6-way metacompare (base / svr / svr_embed /
# fc_ft / meta / meta_adv) on the clean checkpoints, using the calib-support adapters
# from Stage A. Each backbone's array is gated afterok on its Stage-A metatrain array
# (clean_metatrain_jobids.txt). SVR_TUNE=1 (leak-free swarm search), K sweep 4..72,
# MEAN_PATH=meanno7_clean, deploy-faithful calib support. Writes per-backbone CSVs
# calib_metacompare_<bb>_clean.csv under runs/calib_metacompare/.
#
#   bash scripts/submit_clean_metacompare.sh            # submit all
#   DRYRUN=1 bash scripts/submit_clean_metacompare.sh   # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
MTJOBS="$REPO/scripts/clean_metatrain_jobids.txt"       # "<jobid> <bb> <root>"
OUT="$REPO/scripts/clean_metacompare_jobids.txt"
cd "$REPO"
: > "$OUT"

n=0
while read -r MTJID BB ROOTNAME; do
  [ -z "$BB" ] && continue
  ROOT="$RUNS/$ROOTNAME"
  echo "=== metacompare $BB  root=$ROOTNAME  afterok:$MTJID ==="
  if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
  JID=$(RUN_ROOT="$ROOT" BACKBONE="$BB" MEAN_PATH=meanno7_clean \
        META_BASE_SUBDIR=meta_on_base_calib META_ADV_SUBDIR=meta_on_adv_calib \
        CSV_TAG=_clean SVR_TUNE=1 \
        sbatch --parsable --array=0-4 --partition=gpu,vis --dependency=afterok:$MTJID \
        scripts/run_calib_metacompare.sbatch)
  echo "  job $JID"
  echo "$JID $BB $ROOTNAME" >> "$OUT"
  n=$((n+1))
done < "$MTJOBS"
echo "submitted $n metacompare arrays (5 folds each). job ids -> $OUT"
