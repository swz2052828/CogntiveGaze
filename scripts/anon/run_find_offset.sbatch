#!/bin/bash
#SBATCH --job-name=find_offset
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/find_offset_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
$PY scripts/anon/find_video_offset.py --rec 00006 --src OriginalCalib
$PY scripts/anon/find_video_offset.py --rec 00006 --src OriginalData --max-frames 12000
