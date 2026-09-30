#!/bin/bash
#SBATCH --job-name=anon_repair
#SBATCH --partition=compute
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_repair_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=2
cd /springbrook/share/eng/esrpxk/CogntiveGaze
# Recording 00021 was deleted from these three operators before I cancelled the
# cleanup array. They are needed for condition (iii), so the roots must be whole
# again -- a root missing one subject is the partial-data hazard, not a saving.
OPS=(blur0.10 blur0.20 blackbox)
OP=${OPS[$SLURM_ARRAY_TASK_ID]}
/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/make_anon_frames.py \
  --rec 00021 --op "$OP" --dst-root "/springbrook/share/eng/esrpxk/datasets/anon/${OP}"
