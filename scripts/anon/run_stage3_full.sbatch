#!/bin/bash
#SBATCH --job-name=anon_s3
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --array=0-215%60
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_s3_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=2
cd /springbrook/share/eng/esrpxk/CogntiveGaze

# 'none' is a pass-through that still re-encodes: it is the CONTROL arm. Using the
# original ProcessedData as control would give the control one fewer JPEG
# generation than every anonymised arm, confounding utility with re-encoding loss.
OPS=(none blur0.03 blur0.10 blur0.20 blur0.35 blur0.50 \
     pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40 blackbox)
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)

OP=${OPS[$((SLURM_ARRAY_TASK_ID / 18))]}
REC=${RECS[$((SLURM_ARRAY_TASK_ID % 18))]}
echo "task=$SLURM_ARRAY_TASK_ID op=$OP rec=$REC $(date)"

/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/make_anon_frames.py \
  --rec "$REC" --op "$OP" \
  --dst-root "/springbrook/share/eng/esrpxk/datasets/anon/${OP}"
