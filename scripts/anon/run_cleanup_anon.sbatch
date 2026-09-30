#!/bin/bash
#SBATCH --job-name=anon_cleanup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_cleanup_%A_%a.log
set -uo pipefail
# Group inode quota (eng, share-eng) hit its hard limit of 10,497,011 files.
# datasets/anon/ holds 12 operators x 18 x 11,718 x 3 = 7.59M of them, and its
# purpose is served: Stage 4a masked utility is in runs/anon_stage4/*.csv and the
# T1/T2/T3 attacks are complete. Any operator is ~13 CPU-min to regenerate from
# datasets/anon_geometry, so keeping the pixels costs far more than recreating
# them would. `none` and blur0.50 are retained as spot-check references.
OPS=(blur0.03 blur0.10 blur0.20 blur0.35 pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox)
OP=${OPS[$SLURM_ARRAY_TASK_ID]}
D=/springbrook/share/eng/esrpxk/datasets/anon/$OP
[ -d "$D" ] || { echo "$OP already gone"; exit 0; }
N=$(find "$D" -type f | wc -l)
echo "$OP: removing $N files  $(date)"
rm -rf "$D"
echo "$OP: done  $(date)"
