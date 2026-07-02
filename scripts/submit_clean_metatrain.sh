#!/bin/bash
# Stage A of the CLEAN metacompare: re-meta-train the FiLM adapter (deploy-faithful
# calib support) on BOTH the clean base and clean adv checkpoints of every backbone,
# writing meta_on_base_calib/ and meta_on_adv_calib/ into each unified metacmp root
# (built by build_clean_metacmp_roots.sh). Each backbone's array is gated afterany
# on ITS OWN adv clean-retrain array (from clean_retrain_adv_jobids.txt) so the two
# still-training backbones (itracker, vit) wait for their final adv checkpoint while
# the 31 already-complete ones start immediately.
#
#   bash scripts/submit_clean_metatrain.sh            # submit all 33
#   DRYRUN=1 bash scripts/submit_clean_metatrain.sh   # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
LIST="$REPO/scripts/clean_metacmp_backbones.txt"
ADVJOBS="$REPO/scripts/clean_retrain_adv_jobids.txt"     # "<jobid> <bb> <dir>"
OUT="$REPO/scripts/clean_metatrain_jobids.txt"
cd "$REPO"
: > "$OUT"

n=0
while read -r BB ROOTNAME; do
  [ -z "$BB" ] && continue
  ROOT="$RUNS/$ROOTNAME"
  ADVJID=$(awk -v b="$BB" '$2==b{print $1; exit}' "$ADVJOBS")
  DEP=""
  [ -n "$ADVJID" ] && DEP="--dependency=afterany:$ADVJID"
  echo "=== metatrain $BB  root=$ROOTNAME  advdep=${ADVJID:-none} ==="
  if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
  JID=$(RUN_ROOT="$ROOT" BACKBONE="$BB" MEAN_PATH=meanno7_clean \
        CALIB_ROOT=/springbrook/share/eng/esrpxk/datasets/calib_support_K72 \
        sbatch --parsable --array=0-4 --partition=gpu,vis $DEP \
        scripts/run_calib_metatrain.sbatch)
  echo "  job $JID"
  echo "$JID $BB $ROOTNAME" >> "$OUT"
  n=$((n+1))
done < "$LIST"
echo "submitted $n metatrain arrays (5 folds each, base+adv adapters). job ids -> $OUT"
