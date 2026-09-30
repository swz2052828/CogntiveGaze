#!/bin/bash
#SBATCH --job-name=anon_ft
#SBATCH --partition=gpu
# NOT gpu,vis. The `vis` QOS caps a user at ONE gpu (`gpu` allows 15), so any
# array with concurrency >1 that spills onto vis has its extra tasks cancelled
# with QOSMaxGRESPerUser -- silently, mid-array. That is what truncated job
# 1272012 (itracker folds 1-4). Multi-task arrays: gpu only.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%5
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_ft_%A_%a.log
# Plan D condition (iii)-as-adaptation: FINE-TUNE real -> anon.
#
# Same 2x2x5 scope as the retrain arm so the three protocols are directly
# comparable on identical cells: zero-shot (Stage 4), retrain-from-scratch, and
# fine-tune. Starts from the clean-trained checkpoint for the SAME fold via the
# new --init-checkpoint {fold} substitution, with a reduced LR and fewer epochs
# because the model starts near a good solution.
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
SRC=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_${BB}/base/seed42/fold{fold}_best_${BB}_gaze_segmenter.pth
OUT=/springbrook/share/eng/esrpxk/runs/anon_finetune/${BB}_${OP}/seed42
mkdir -p "$OUT" /springbrook/share/eng/esrpxk/runs/anon_finetune/logs

SHORT=""
for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
  W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
  [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
done
[ -n "$SHORT" ] && { echo "FATAL $OP: short recordings:$SHORT"; exit 1; }
# {fold} is substituted by the trainer; check the resolved file exists first.
RESOLVED=${SRC/\{fold\}/$FOLD}
[ -f "$RESOLVED" ] || { echo "FATAL: no source checkpoint $RESOLVED"; exit 1; }

echo "########## finetune BB=$BB OP=$OP fold=$FOLD from $RESOLVED $(date)"
python vit_gaze_segmenter.py train \
  --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
  --input-mode multistream --backbone $BB --use-grid \
  --init-checkpoint "$SRC" \
  --epochs 8 --batch-size 128 --num-workers 16 \
  --lr 5e-5 --lr-scheduler cosine --augment light \
  --patience 3 --min-delta 0.005 --amp --no-accelerate --temporal-window 4 \
  --output-activation none --gaze-range 4.0 \
  --fold-index $FOLD --seed 42 --out-path "$OUT" \
  --log-file "/springbrook/share/eng/esrpxk/runs/anon_finetune/logs/${BB}_${OP}_f${FOLD}.log"
echo "DONE $BB $OP fold$FOLD"
