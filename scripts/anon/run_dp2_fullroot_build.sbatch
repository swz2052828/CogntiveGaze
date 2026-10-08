#!/bin/bash
# Assemble the DP2 evaluation roots the way the SimSwap ones are assembled:
#   ProcessedDP2full_<M>          DP2 face AND DP2 eyes          (= Full Synthetic)
#   ProcessedDP2full_<M>_oldeye   DP2 face, ORIGINAL eye crops   (= Hybrid)
# both sharing ProcessedData's manifest and mean images through meanno7_clean.
# Fails unless every recording/stream is complete.
#SBATCH --job-name=dp2_build
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_build_%j.log
set -euo pipefail
PFX=${PFX:-ProcessedDP2full}   # ProcessedDP2fbfull for the full-body roots
D=/springbrook/share/eng/esrpxk/datasets
RECS="00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023"
BAD=""
for M in per_subject per_frame; do
  A=$D/${PFX}_$M
  for rec in $RECS; do
    for F in appleFace appleLeftEye appleRightEye; do
      H=$(ls "$A/$rec/$F" 2>/dev/null | wc -l); W=$(ls "$D/ProcessedData/$rec/$F" | wc -l)
      [ "$H" -lt "$W" ] && BAD="$BAD $M/$rec/$F($H/$W)"
    done
  done
done
[ -n "$BAD" ] && { echo "FATAL short:$BAD"; exit 1; }

for M in per_subject per_frame; do
  A=$D/${PFX}_$M; H=${A}_oldeye
  ln -sfn "$D/ProcessedData/meanno7_clean" "$A/meanno7_clean"
  mkdir -p "$H"; ln -sfn "$D/ProcessedData/meanno7_clean" "$H/meanno7_clean"
  for rec in $RECS; do
    mkdir -p "$H/$rec"
    ln -sfn "$A/$rec/appleFace"                   "$H/$rec/appleFace"
    ln -sfn "$D/ProcessedData/$rec/appleLeftEye"  "$H/$rec/appleLeftEye"
    ln -sfn "$D/ProcessedData/$rec/appleRightEye" "$H/$rec/appleRightEye"
  done
done
echo BUILD_OK
