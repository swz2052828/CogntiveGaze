#!/bin/bash
#SBATCH --job-name=anon_retrain
#SBATCH --partition=gpu
# NOT gpu,vis. The `vis` QOS caps a user at ONE gpu (`gpu` allows 15), so any
# array with concurrency >1 that spills onto vis has its extra tasks cancelled
# with QOSMaxGRESPerUser -- silently, mid-array. That is what truncated job
# 1272012 (itracker folds 1-4). Multi-task arrays: gpu only.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-19%5
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_retrain_%A_%a.log
# Plan D condition (i): TRAIN on anonymised data, not just deploy on it.
#
# Scoped deliberately. The full grid is 12 operators x 4 backbones x 5 folds =
# 240 training runs, which is not affordable. This takes the two operators that
# bracket the privacy axis (blackbox = the strongest, blur0.20 = mid-strength)
# and the two backbones that disagree under zero-shot (convnextv2, which
# IMPROVES under anonymisation, and mobile_vit, which is the only multistream
# model that degrades). 2 x 2 x 5 = 20 runs.
#
# `none` is NOT retrained here: the clean-trained checkpoints in
# meta_pipeline_clean_metacmp_* already are that arm, trained with these exact
# hyperparameters. Retraining it would only add init noise to the control.
set -uo pipefail
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface
cd /springbrook/share/eng/esrpxk/CogntiveGaze

D=/springbrook/share/eng/esrpxk/datasets
OPS=(blackbox blur0.20)
BBS=(mobile_vit convnextv2)
I=$SLURM_ARRAY_TASK_ID
OP=${OPS[$(( I / 10 ))]}
BB=${BBS[$(( (I % 10) / 5 ))]}
FOLD=$(( I % 5 ))
DP=$D/anon/$OP
OUT=/springbrook/share/eng/esrpxk/runs/anon_retrain/${BB}_${OP}/seed42
mkdir -p "$OUT" /springbrook/share/eng/esrpxk/runs/anon_retrain/logs

# Coverage guard: training silently on a partial root is the same failure that
# cost 28 result files on the utility side, and it is worse here because the
# damage is baked into a checkpoint.
SHORT=""
for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
  W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
  [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
done
[ -n "$SHORT" ] && { echo "FATAL $OP: short recordings:$SHORT"; exit 1; }

echo "########## retrain BB=$BB OP=$OP fold=$FOLD $(date)"
python vit_gaze_segmenter.py train \
  --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
  --input-mode multistream --backbone $BB --use-grid \
  --weights imagenet --epochs 20 --batch-size 128 --num-workers 16 \
  --lr 8e-4 --lr-scheduler cosine --augment light \
  --patience 5 --min-delta 0.005 --amp --no-accelerate --temporal-window 4 \
  --output-activation none --gaze-range 4.0 \
  --fold-index $FOLD --seed 42 --out-path "$OUT" \
  --log-file "/springbrook/share/eng/esrpxk/runs/anon_retrain/logs/${BB}_${OP}_f${FOLD}.log"
echo "DONE $BB $OP fold$FOLD"
