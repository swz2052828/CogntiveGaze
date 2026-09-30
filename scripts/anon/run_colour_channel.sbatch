#!/bin/bash
#SBATCH --job-name=anon_colour
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_colour_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
cd /springbrook/share/eng/esrpxk/CogntiveGaze
/springbrook/share/eng/esrpxk/envs/idattack/bin/python scripts/anon/colour_channel.py
