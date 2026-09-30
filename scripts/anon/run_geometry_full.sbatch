#!/bin/bash
#SBATCH --job-name=anon_geom
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-17
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_geom_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)
REC=${RECS[$SLURM_ARRAY_TASK_ID]}
/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/recover_geometry_tm.py \
  --rec "$REC" --out /springbrook/share/eng/esrpxk/datasets/anon_geometry/${REC}.npz
