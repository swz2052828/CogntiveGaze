#!/bin/bash
# Build unified per-backbone run roots for the CLEAN metacompare so the existing
# run_calib_metatrain.sbatch / run_calib_metacompare.sbatch work unchanged.
# clean base and clean adv checkpoints live in SEPARATE run dirs; here we symlink
# them into one root:
#   meta_pipeline_clean_metacmp_<BB>/base -> meta_pipeline_clean_<x>/base   (plain clean-retrain)
#   meta_pipeline_clean_metacmp_<BB>/adv  -> meta_pipeline_clean_adv_<x>/base (adv clean-retrain)
# meta_on_base_calib/ and meta_on_adv_calib/ are created as REAL dirs by Stage A.
#
#   bash scripts/build_clean_metacmp_roots.sh          # build
#   DRYRUN=1 bash scripts/build_clean_metacmp_roots.sh # print only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
MANIFEST="$REPO/scripts/clean_canonical_runs.tsv"
LIST="$REPO/scripts/clean_metacmp_backbones.txt"
: > "$LIST"

n=0; skip=0
while IFS=$'\t' read -r RUN BB OA GR LR EP UG NF; do
  [ "$RUN" = "run_subdir" ] && continue                     # header
  [ "$BB" = "cnn_transformer_raw" ] && { echo "skip $BB (no adv variant)"; skip=$((skip+1)); continue; }
  SUF="${RUN#meta_pipeline_}"
  if [ "$RUN" = "meta_pipeline" ]; then
    BASE_RUN="$RUNS/meta_pipeline_clean__shared_${BB}"
    ADV_RUN="$RUNS/meta_pipeline_clean_adv__shared_${BB}"
  else
    BASE_RUN="$RUNS/meta_pipeline_clean_${SUF}"
    ADV_RUN="$RUNS/meta_pipeline_clean_adv_${SUF}"
  fi
  ROOT="$RUNS/meta_pipeline_clean_metacmp_${BB}"
  # require both base and adv checkpoint dirs to exist
  if [ ! -d "$BASE_RUN/base/seed42" ] || [ ! -d "$ADV_RUN/base/seed42" ]; then
    echo "MISSING ckpts for $BB (base=$BASE_RUN adv=$ADV_RUN) -- skipping"; skip=$((skip+1)); continue
  fi
  echo "=== $BB -> $(basename "$ROOT")  base<-$(basename "$BASE_RUN")  adv<-$(basename "$ADV_RUN") ==="
  if [ "${DRYRUN:-0}" = "1" ]; then continue; fi
  mkdir -p "$ROOT"
  ln -sfnT "$BASE_RUN/base" "$ROOT/base"
  ln -sfnT "$ADV_RUN/base"  "$ROOT/adv"
  mkdir -p "$ROOT/meta_on_base_calib" "$ROOT/meta_on_adv_calib" "$ROOT/logs"
  echo "$BB $(basename "$ROOT")" >> "$LIST"
  n=$((n+1))
done < "$MANIFEST"
echo "built $n metacmp roots ($skip skipped). list -> $LIST"
