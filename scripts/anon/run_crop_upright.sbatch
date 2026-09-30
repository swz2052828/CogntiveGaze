#!/bin/bash
#SBATCH --job-name=cropup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/cropup_%j.log
set -euo pipefail
/springbrook/share/eng/esrpxk/envs/facedet/bin/python /springbrook/share/eng/esrpxk/CogntiveGaze/scripts/anon/crop_upright_check.py
