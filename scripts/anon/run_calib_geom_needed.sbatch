#!/bin/bash
#SBATCH --job-name=calibgeom_need
#SBATCH --partition=compute
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calibgeom_need_%j.log
set -uo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
G=/springbrook/share/eng/esrpxk/datasets/anon_calib_geometry
for REC in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  $PY scripts/anon/calib_crop_geometry.py --rec "$REC" \
      --frames-json "$G/NEEDED_FRAMES.json" --out "$G/needed_${REC}.npz" &
done
wait
echo ALL_DONE
