#!/bin/bash
#SBATCH --job-name=anon_mfskin
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_mfskin_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
FD=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
E=/springbrook/share/eng/esrpxk/datasets/anon_extreme
RECS="00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023"
# Unbiased companion to the peer's meanfill arm: same limit (1200) and same root,
# so the pair is directly comparable and their difference measures how much the
# eye-region pixels contaminate the plain-meanfill fill colour.
for REC in $RECS; do
  $FD scripts/anon/make_anon_frames.py --rec "$REC" --op meanfill_skin \
      --dst-root "$E/meanfill_skin" --limit 1200 &
done
wait
echo ALL_DONE
