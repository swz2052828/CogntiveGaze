#!/bin/bash
#SBATCH --job-name=eyelm_all
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/eyelm_all_%j.log
set -uo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
L=/springbrook/share/eng/esrpxk/datasets/anon_eye_landmarks
mkdir -p "$L"
# Landmarks for the P-axis. 1200 frames per recording, evenly subsampled, to
# match the subsampled generation the inode budget now requires. One npz per
# recording -- 18 files total, negligible against the quota that forced this.
for REC in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  $PY scripts/anon/eye_landmarks.py --rec "$REC" --limit 1200 --out "$L/${REC}.npz" &
done
wait
echo ALL_DONE
