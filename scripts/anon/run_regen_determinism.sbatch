#!/bin/bash
#SBATCH --job-name=regen_det
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/regen_det_%j.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
T=/springbrook/share/eng/esrpxk/datasets/_regen_check
rm -rf "$T"; mkdir -p "$T"
# Regenerate ONE recording of a SURVIVING operator and compare byte-for-byte
# against what is already on disk. If it reproduces exactly, the seven deleted
# operators can be rebuilt identically and the 12-point table is recoverable
# cell by cell. If it does not, the deletion is genuinely lossy and the table
# has to be restricted -- so this check decides the question.
$PY scripts/anon/make_anon_frames.py --rec 00006 --op blur0.20 --dst-root "$T"
echo "### byte comparison against datasets/anon/blur0.20/00006"
A=/springbrook/share/eng/esrpxk/datasets/anon/blur0.20/00006
B=$T/00006
tot=0; diff=0
for F in appleFace appleLeftEye appleRightEye; do
  for f in $(ls "$B/$F" | head -400); do
    tot=$((tot+1))
    cmp -s "$A/$F/$f" "$B/$F/$f" || diff=$((diff+1))
  done
done
echo "compared $tot files, $diff differ"
echo -n "file counts: on-disk="; ls $A/appleFace | wc -l; echo -n " regenerated="; ls $B/appleFace | wc -l
rm -rf "$T"
echo CLEANED
