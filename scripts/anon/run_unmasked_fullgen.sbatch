#!/bin/bash
#SBATCH --job-name=anon_unfull
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-71%40
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_unfull_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
# Regenerate the unmasked arm at FULL frame count for the four operators that
# carry the claim. The utility manifest reads ~11,718 frames per recording; the
# existing --limit 1200 roots are for the attack side, where 60/60 per subject is
# ample. Only these four are needed: they bracket the privacy collapse
# (blur0.20 TAR 52.5% -> blur0.35 1.3% -> blur0.50 0.0%, blackbox 5.3%), which is
# where the paper claims a strength exists that defeats the attacker but must
# touch the eyes. That claim asserts a utility cost it has not yet measured.
OPS=(blur0.20 blur0.35 blur0.50 blackbox)
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)
OP=${OPS[$((SLURM_ARRAY_TASK_ID / 18))]}
REC=${RECS[$((SLURM_ARRAY_TASK_ID % 18))]}
echo "task=$SLURM_ARRAY_TASK_ID op=$OP rec=$REC $(date +%H:%M:%S)"
/springbrook/share/eng/esrpxk/envs/facedet/bin/python scripts/anon/make_anon_frames.py \
  --rec "$REC" --op "$OP" --protect-eyes 0 \
  --dst-root "/springbrook/share/eng/esrpxk/datasets/anon_unmasked_full/${OP}"
