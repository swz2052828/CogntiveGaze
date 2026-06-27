#!/bin/bash
#SBATCH --job-name=disagr_fr
#SBATCH --partition=compute
#SBATCH --array=0-17
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/disagr_fr_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/disagr_fr_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
SUBS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
SUB=${SUBS[$SLURM_ARRAY_TASK_ID]}
PY=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
cd /springbrook/share/eng/esrpxk/CogntiveGaze
# clear stale crops (FP events changed under the corrected mask) then re-render
rm -rf "/springbrook/share/eng/esrpxk/datasets/ear_vs_gt/disagreement_frames/subid_${SUB}"
echo "Host: $(hostname) sub=$SUB start $(date)"
"$PY" save_disagreement_frames.py --subject "$SUB"
echo "done sub$SUB $(date)"
