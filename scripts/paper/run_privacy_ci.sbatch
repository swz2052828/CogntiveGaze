#!/bin/bash
#SBATCH --job-name=priv_ci
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/priv_ci_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
/springbrook/share/eng/esrpxk/envs/gaze/bin/python scripts/paper/privacy_ci.py \
    --boot "${BOOT:-2000}" --out /springbrook/share/eng/esrpxk/results/paper/privacy_ci.csv
