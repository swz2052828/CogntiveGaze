#!/bin/bash
# DeepPrivacy2 over EVERY task frame, so the gaze models can be evaluated on it
# exactly as Table 3 evaluates the SimSwap arms (clean-trained checkpoints,
# anonymised deployment, identical metacompare call).
#
# Until now DP2 existed only as 260 frames/participant for the attacks, so the
# claim "DP2 destroys the gaze signal" rested on inspection of repainted eyes.
# This turns it into a number on the same axis as Table 3.
#
# truncation 1.0 (see run_dp2_trunc1.sbatch for why 0 is unusable), two regimes:
#   per_subject  the SimSwap-comparable arm
#   per_frame    the linkage-breaking fix -- what does it cost in utility?
#
# array index -> (regime, recording): regime = id/18, rec = id%18
#SBATCH --job-name=dp2_fullroot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-35%15
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/dp2_fullroot_%A_%a.log
set -uo pipefail
export OMP_NUM_THREADS=4
export TORCH_HOME=/springbrook/share/eng/esrpxk/home_caches/torch
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
cd /springbrook/share/eng/esrpxk/CogntiveGaze

MODES=(per_subject per_frame)
RECS=(00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023)
M=${MODES[$(( SLURM_ARRAY_TASK_ID / 18 ))]}
REC=${RECS[$(( SLURM_ARRAY_TASK_ID % 18 ))]}
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/results/anon_dp2/fullroot
mkdir -p "$R"

echo "########## DP2 full root  identity=$M  rec=$REC  $(date +%T)"
/springbrook/share/eng/esrpxk/envs/dp2/bin/python scripts/anon/run_deepprivacy2.py \
    --identity "$M" --truncation 1.0 --all-frames --resume --recs "$REC" \
    --out-root "$D/ProcessedDP2full_$M" \
    --report "$R/edit_domain_${M}_${REC}.json" || { echo "FAILED $M $REC"; exit 1; }
echo "ALL_DONE $M $REC $(date +%T)"
