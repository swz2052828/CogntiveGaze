#!/bin/bash
#SBATCH --job-name=dump_ev
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dump_ev_%j.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/dump_ev_%j.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
SUB=${1:-6}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname) sub=$SUB start $(date)"
"$PY" dump_sub_events.py --subject "$SUB"
echo "done sub$SUB $(date)"
