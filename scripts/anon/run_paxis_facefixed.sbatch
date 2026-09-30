#!/bin/bash
#SBATCH --job-name=paxis_ff
#SBATCH --partition=compute,hmem
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-35
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_ff_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)
LV=(P1 P2)
L=${LV[$((SLURM_ARRAY_TASK_ID / 18))]}
R=${RECS[$((SLURM_ARRAY_TASK_ID % 18))]}
$PY scripts/anon/make_paxis_facefixed.py --level $L --rec $R
