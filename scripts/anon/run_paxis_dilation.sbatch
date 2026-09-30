#!/bin/bash
#SBATCH --job-name=paxis_dil
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_dil_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
P=/springbrook/share/eng/esrpxk/datasets/anon_paxis
# The Pareto sweep for the periocular axis. The endpoints are already measured
# and they do not meet in the middle:
#   P0 (whole eye crop)   d' 5.122, TAR@1e-3 99.3%  | fc_ft 3.872 cm
#   P1 (eye opening only) d' 0.272, TAR@1e-3  0.0%  | fc_ft 7.131 cm
# P1 buys essentially perfect privacy -- below the 120px background nuisance
# floor (d' 0.361) -- and costs +3.26 cm, 8x the pre-declared 0.4 cm
# non-inferiority margin. So P1 is not deployable and the question is whether
# ANY point on the axis is: these dilate the P1 mask outward by N px, sweeping
# the preserved region from the eye opening back toward the full crop.
LEVELS=(P1d5 P1d10 P1d20 P1d40)
LV=${LEVELS[$SLURM_ARRAY_TASK_ID]}
# Subsampled to 1200 frames per recording -- the group inode quota hit its hard
# limit at 10,497,011 files, and full-count generation for two levels would be
# another 1.26M. Utility on a subsampled root is valid only against a matching
# subset manifest (make_subset_manifest.py); both arms of any comparison must
# use the same one.
for REC in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  $PY scripts/anon/make_anon_eyes.py --rec "$REC" --level "$LV" --limit 1200 \
      --dst-root "$P/$LV" &
done
wait
echo "ALL_DONE $LV"
