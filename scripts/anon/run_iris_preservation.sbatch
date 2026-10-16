#!/bin/bash
# Model-free pupil-position preservation, every family on identical frames.
# ARMS env: dataset roots, ProcessedData first. OUT env: json path.
#SBATCH --job-name=iris_pres
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/iris_pres_%j.log
set -uo pipefail
export OMP_NUM_THREADS=1
cd /springbrook/share/eng/esrpxk/CogntiveGaze
/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/iris_preservation.py \
    --arms ${ARMS:?} --out "${OUT:?}" --procs 16
echo ALL_DONE
