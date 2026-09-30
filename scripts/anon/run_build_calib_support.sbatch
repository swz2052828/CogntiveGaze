#!/bin/bash
#SBATCH --job-name=anon_calsup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_calsup_%j.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
# Anonymised enrolment sets for condition (iii). `none` is regenerated through the
# identical path, so condition (i) carries the same interpolation count.
for OP in none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
          pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox; do
  for K in 9 18 36 72; do
    echo "########## $OP K=$K"
    $PY scripts/anon/build_anon_calib_support.py --op "$OP" --k "$K" \
        --allow-median-geometry || echo "FAILED $OP K=$K"
  done
done
echo ALL_DONE
