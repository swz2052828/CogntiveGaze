#!/bin/bash
#SBATCH --job-name=anon_blurext
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_blurext_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
FD=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
RECS="00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023"

# Two questions, two kinds of arm.
#   meanfill  -- MECHANISM. Face box replaced by its own mean colour: identical to
#                blackbox in geometry, eye preservation and uniformity, differing
#                only in that the surround carries the subject's mean face colour.
#                meanfill vs blackbox isolates the skin-tone channel exactly.
#   0.75/1.00 -- SHAPE. Whether the measured curve turns up past its minimum
#                (0.35 -> 4.11, 0.50 -> 4.02). Per-channel spatial std of the
#                blurred surround goes 4.46 at 0.50 -> 1.29 at 0.75 -> 0.54 at 1.00,
#                so these two arms straddle the uniform transition; the existing
#                sweep stopped just before it.
# --limit 1200 matches the other control arms; the attack draws 60/60 per subject
# either way, so full frame count buys nothing and costs ~10x.
for OP in blur1.00; do   # meanfill/meanfill_skin/blur0.75 already complete
  echo "########## $OP masked (limit 1200)"
  for REC in $RECS; do
    $FD scripts/anon/make_anon_frames.py --rec "$REC" --op "$OP" \
        --dst-root "/springbrook/share/eng/esrpxk/datasets/anon_extreme/${OP}" --limit 1200 &
  done
  wait
done
echo ALL_DONE
