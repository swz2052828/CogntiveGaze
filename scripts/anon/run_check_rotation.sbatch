#!/bin/bash
#SBATCH --job-name=rotchk
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/rotchk_%j.log
set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/check_rotation.py "$@"
