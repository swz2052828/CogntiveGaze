#!/bin/bash
#SBATCH --job-name=refill_op
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/refill_%A_%a.log
#SBATCH --array=0-6%1
# Refill the seven operator cells that the inode cleanup removed, one operator
# at a time: regenerate -> build enrolment -> evaluate -> VERIFY -> delete.
#
# Regeneration is exactly reproducible: the operators contain no randomness and
# a control rebuild of blur0.20/00006 matched the surviving root byte-for-byte
# on 1200/1200 files with identical frame counts (job 1273664). The source data
# the cleanup did NOT touch -- OriginalData 18/18 and anon_geometry 18/18 -- is
# what makes this possible.
#
# Each operator root is 632,772 files / 7.3 GB against 4.6M free inodes, so one
# at a time (%1) is never close to the limit even if a delete is missed. The
# calibration support is kept permanently: it is only 3,888 files per operator.
#
# ORDER MATTERS: the delete happens only after the expected CSVs are confirmed
# present and non-empty. A failed evaluation leaves the data in place for a
# retry rather than destroying the only copy.
set -uo pipefail
module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || exit 1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PYF=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
CLI="python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/runs/anon_refill
mkdir -p "$OUT/logs"

OPS=(blur0.03 blur0.35 pixel0.03 pixel0.06 pixel0.12 pixel0.25 pixel0.40)
OP=${OPS[$SLURM_ARRAY_TASK_ID]}
DP=$D/anon/$OP
CS=$D/calib_support_K72_$OP

# ---- 1. regenerate the data root -------------------------------------------
if [ -d "$DP" ] && [ "$(ls "$DP/00023/appleFace" 2>/dev/null | wc -l)" -ge 11718 ]; then
  echo "[$OP] root already complete, skipping generation"
else
  echo "### [$OP] generating $(date +%H:%M:%S)"
  for REC in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    $PYF scripts/anon/make_anon_frames.py --rec "$REC" --op "$OP" --dst-root "$DP" &
  done
  wait
fi
# coverage guard before anything reads it
SHORT=""
for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
  W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
  [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
done
[ -n "$SHORT" ] && { echo "FATAL [$OP] generation short:$SHORT -- keeping data, not deleting"; exit 1; }
echo "### [$OP] generation verified complete"

# ---- 2. matched enrolment set (peer's fixed builder) ------------------------
if [ ! -f "$CS/meanno7_clean/metadata.mat" ]; then
  echo "### [$OP] building enrolment $(date +%H:%M:%S)"
  $PYF scripts/anon/build_anon_calib_support.py --op "$OP" --k 72 --allow-median-geometry \
    || { echo "FATAL [$OP] enrolment build failed -- keeping data"; exit 1; }
fi
for R in 00006 00023; do
  C=$(ls "$CS/$R/appleFace" 2>/dev/null | wc -l)
  [ "$C" -lt 72 ] && { echo "FATAL [$OP] enrolment short at $R ($C) -- keeping data"; exit 1; }
done

# ---- 3. evaluate the cells this operator is missing -------------------------
# cond (ii) exists already for the four original backbones (Stage 4a), so only
# the three new comparison architectures need it. cond (iii) needs all seven.
run_cell () {  # $1=cond $2=backbone $3=support
  local COND=$1 BB=$2 SUP=$3
  local RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB
  for FOLD in 0 1 2 3 4; do
    $CLI metacompare \
      --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
      --input-mode multistream --backbone "$BB" --use-grid \
      --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
      --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
      --calib-support-root "$SUP" \
      --svr-embed --svr-embed-gamma scale \
      --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
      --k 72 --trials 1 --batch-size 128 --num-workers 8 \
      --fold-index $FOLD --seed 42 \
      --csv-out "$OUT/${COND}_${BB}_${OP}.csv" \
      --log-file "$OUT/logs/s4$([ "$COND" = "iii" ] && echo c3)_${BB}_${OP}_f${FOLD}.log" \
      || echo "FAILED $COND $BB $OP f$FOLD"
  done
}
for BB in mobilenet_v3 affnet mgazenet; do
  echo "### [$OP] cond ii $BB $(date +%H:%M:%S)"; run_cell ii "$BB" "$D/calib_support_K72"
done
for BB in face_only_mobile_vit mobile_vit convnextv2 itracker mobilenet_v3 affnet mgazenet; do
  echo "### [$OP] cond iii $BB $(date +%H:%M:%S)"; run_cell iii "$BB" "$CS"
done

# ---- 4. verify before destroying anything ----------------------------------
MISSING=""
for BB in mobilenet_v3 affnet mgazenet; do
  f="$OUT/ii_${BB}_${OP}.csv";  [ -s "$f" ] && [ "$(wc -l < "$f")" -ge 6 ] || MISSING="$MISSING ii/$BB"
done
for BB in face_only_mobile_vit mobile_vit convnextv2 itracker mobilenet_v3 affnet mgazenet; do
  f="$OUT/iii_${BB}_${OP}.csv"; [ -s "$f" ] && [ "$(wc -l < "$f")" -ge 6 ] || MISSING="$MISSING iii/$BB"
done
if [ -n "$MISSING" ]; then
  echo "INCOMPLETE [$OP]:$MISSING -- KEEPING $DP for a retry"
  exit 1
fi

# ---- 5. reclaim -------------------------------------------------------------
echo "### [$OP] all 10 cells x5 folds present; removing $DP $(date +%H:%M:%S)"
rm -rf "$DP"
echo "ALL_DONE $OP"
