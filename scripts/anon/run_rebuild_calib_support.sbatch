#!/bin/bash
# Rebuild calib_support_K72_<op> after the calib_processed source bug.
#
# The builder now reads pixels from calib_support_K72 itself and takes IOD from
# anon_calib_geometry_k72, both of which carry the task-data face-box convention.
# Only the operators whose DEPLOYMENT roots still exist under datasets/anon are
# rebuilt; the rest were removed in the storage cleanup and would have no arm.
#SBATCH --job-name=rebuild_cs
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/rebuild_cs_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
OPS=(none blur0.10 blur0.20 blur0.50 blackbox)
OP=${OPS[$SLURM_ARRAY_TASK_ID]}
: "${K:=72}"
OUT=/springbrook/share/eng/esrpxk/datasets/calib_support_K${K}_$OP
rm -rf "$OUT"
# --allow-median-geometry: 19 of 1296 enrolment frames (1.5%) sit below the 0.95
# match threshold, and all of them are structurally truncated -- the eye region
# extends past the face-crop edge (00012 left eye, 17 frames; 00007 and 00009,
# 1 each), so part of the template necessarily lands on replicated border and the
# correlation is capped below 1 even at the correct position. The per-subject box
# is stable to a few px (sd 0.7-3.4), so the median box is the right answer for
# those frames; rejecting them instead would have dropped 00012 to 55 of 72
# enrolment frames and tripped the cond (iii) enrolment guard, silently removing
# the subject with the tightest face crop.
$PY scripts/anon/build_anon_calib_support.py --op "$OP" --k "$K" --out-root "$OUT" \
    --allow-median-geometry
