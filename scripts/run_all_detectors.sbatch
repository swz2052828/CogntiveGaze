#!/bin/bash
#SBATCH --job-name=alldet
#SBATCH --partition=compute
#SBATCH --array=0-17
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/alldet_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/alldet_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
SUBS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
echo "Host: $(hostname)  subject=$SUB  Start: $(date)"
"$PY" run_all_detectors.py --subject "$SUB" --stride 2 \
  --save-interval 150 --max-examples 120 \
  --out-root /springbrook/share/eng/esrpxk/datasets/all_detectors
echo "Done sub$SUB: $(date)"
