#!/bin/bash
#SBATCH --job-name=paxis_gen
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_gen_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
P=/springbrook/share/eng/esrpxk/datasets/anon_paxis
LEVELS=(P1 P2)
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
