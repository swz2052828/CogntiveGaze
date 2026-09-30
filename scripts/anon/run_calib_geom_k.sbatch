#!/bin/bash
# Recover eye geometry INSIDE the released calibration face crops.
#
# --src-root is calib_support_K72, NOT calib_processed. The two hold different
# face boxes for the same frames (1.40x, MAE 39.4 levels); only the support set
# matches the task-data convention the models were trained on. Getting this wrong
# is silent -- it cost the first condition (iii) run (job 1272012, quarantined) --
# so calib_crop_geometry.py now refuses to write if the recovered
# IOD/face-crop-width falls outside the task range.
#SBATCH --job-name=calib_geom_k
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_geom_k${K:-72}_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 \
      00018 00019 00020 00021 00022 00023)
REC=${RECS[$SLURM_ARRAY_TASK_ID]}
: "${K:=72}"
OUT=/springbrook/share/eng/esrpxk/datasets/anon_calib_geometry_k$K/$REC.npz
$PY scripts/anon/calib_crop_geometry.py --rec "$REC" \
    --src-root "calib_support_K$K" --out "$OUT"
