#!/bin/bash
#SBATCH --job-name=anon_eyeflat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_eyeflat_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
FD=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
F=/springbrook/share/eng/esrpxk/datasets/anon_eyeflat
RECS="00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023"

# Layout-vs-texture control: eye boxes keep position and size, lose their pixels.
# Frame count matches the unmasked ablation (1200/recording) so the three arms
# -- exact / flat / unmasked -- are directly comparable.
for OP in blackbox blur0.50 blur0.20; do
  echo "########## $OP eye_mode=flat"
  for REC in $RECS; do
    $FD scripts/anon/make_anon_frames.py --rec "$REC" --op "$OP" \
        --dst-root "$F/$OP" --eye-mode flat --limit 1200 &
  done
  wait
done
echo ALL_DONE
