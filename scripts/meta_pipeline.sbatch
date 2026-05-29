#!/bin/bash
# scripts/meta_pipeline.sbatch
#
# Per-fold driver for the full meta-learned-calibration experiment. Submit as
# a Slurm array (one task per CV fold) and each task runs, *sequentially within
# the same job*, the five stages needed to produce the four-way
# base / svr / meta / meta_adv comparison + the K-sweep CSV:
#
#   stage 1: train             -> base encoder+head     (BASE)
#   stage 2: train --subject-adv -> subject-invariant encoder+head (ADV)
#   stage 3: metatrain --init-checkpoint BASE -> meta adapter on plain features  (META)
#   stage 4: metatrain --init-checkpoint ADV  -> meta adapter on adv features    (META_ADV)
#   stage 5: metacompare across K in {4, 8, 16, 32, 64}, appending to ONE CSV
#
# Submit:
#   sbatch --array=0-4 scripts/meta_pipeline.sbatch
#
# After all array tasks finish, render the K-sweep figure with
#   bash scripts/plot_metacompare.sh
#
# Stage-skip: re-running this script will skip stages whose final checkpoint
# already exists, so re-submitting after a partial failure resumes cleanly.

#SBATCH --job-name=meta_pipeline
#SBATCH --array=0-4
#SBATCH --output=logs/meta_pipeline_%A_%a.out
#SBATCH --error=logs/meta_pipeline_%A_%a.err
# Fill in your cluster's directives below (partition, account, time, GPU):
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --time=24:00:00
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=32G

set -euo pipefail

# ---- config (override on the sbatch command line via --export=VAR=val,...) ----
: "${DATA_PATH:=./datasets/ProcessedData}"
: "${EYE_PATH:=$DATA_PATH}"
: "${OUT_ROOT:=./runs/meta_pipeline}"
: "${BACKBONE:=vit}"
: "${EPOCHS:=20}"
: "${BATCH_SIZE:=8}"
: "${NUM_WORKERS:=8}"
: "${LR:=1e-4}"
: "${LR_SCHEDULER:=cosine}"
: "${AUGMENT:=light}"
: "${PATIENCE:=8}"
: "${MIN_DELTA:=0.005}"
: "${ADV_WEIGHT:=0.1}"
: "${META_ITERS:=2000}"
: "${INNER_STEPS:=5}"
: "${INNER_LR:=1e-2}"
: "${META_SUPPORT:=16}"
: "${META_QUERY:=32}"
: "${ADAPTER:=film}"
: "${K_SWEEP:=4 8 16 32 64}"
: "${TRIALS:=5}"
: "${SVR_C:=1.0}"
: "${SVR_GAMMA:=scale}"
: "${SVR_EPS:=0.1}"
: "${SVR_TUNE:=1}"        # 1 = run PSO HP search, paste tuned (C, gamma, epsilon)
: "${SVR_POP:=30}"
: "${SVR_ITERS:=50}"
: "${FC_FT:=1}"           # 1 = add the fc-only fine-tune baseline to metacompare
: "${FC_FT_STEPS:=20}"
: "${FC_FT_LR:=5e-5}"
: "${FC_FT_WD:=5e-4}"

# ---- per-fold paths ----
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
BASE_DIR="$OUT_ROOT/base"
ADV_DIR="$OUT_ROOT/adv"
META_DIR="$OUT_ROOT/meta_on_base"
META_ADV_DIR="$OUT_ROOT/meta_on_adv"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$BASE_DIR" "$ADV_DIR" "$META_DIR" "$META_ADV_DIR" "$LOG_DIR" logs

BASE_CKPT="$BASE_DIR/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
ADV_CKPT="$ADV_DIR/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
META_CKPT="$META_DIR/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
META_ADV_CKPT="$META_ADV_DIR/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
CSV_OUT="$OUT_ROOT/metacompare.csv"   # one shared CSV, all folds * all K append

CLI="python vit_gaze_segmenter.py"
COMMON_DATA="--data-path $DATA_PATH --eye-path $EYE_PATH --input-mode multistream --backbone $BACKBONE --use-grid"
COMMON_TRAIN="$COMMON_DATA --weights imagenet --epochs $EPOCHS --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS --lr $LR --lr-scheduler $LR_SCHEDULER --augment $AUGMENT \
  --patience $PATIENCE --min-delta $MIN_DELTA --amp --fold-index $FOLD"

# ---- stage 1: base train -----------------------------------------------------
if [[ -f "$BASE_CKPT" ]]; then
  echo "[fold $FOLD] stage 1 base train: skip ($BASE_CKPT exists)"
else
  echo "[fold $FOLD] stage 1 base train"
  $CLI train $COMMON_TRAIN --out-path "$BASE_DIR" \
    --log-file "$LOG_DIR/base_fold${FOLD}.log"
fi

# ---- stage 2: subject-adv train ---------------------------------------------
if [[ -f "$ADV_CKPT" ]]; then
  echo "[fold $FOLD] stage 2 adv train: skip ($ADV_CKPT exists)"
else
  echo "[fold $FOLD] stage 2 adv train"
  $CLI train $COMMON_TRAIN --out-path "$ADV_DIR" \
    --subject-adv --adv-weight $ADV_WEIGHT \
    --log-file "$LOG_DIR/adv_fold${FOLD}.log"
fi

# ---- stage 3: metatrain on BASE ---------------------------------------------
if [[ -f "$META_CKPT" ]]; then
  echo "[fold $FOLD] stage 3 metatrain (on base): skip ($META_CKPT exists)"
else
  echo "[fold $FOLD] stage 3 metatrain (on base)"
  $CLI metatrain $COMMON_DATA --weights imagenet \
    --init-checkpoint "$BASE_CKPT" \
    --adapter $ADAPTER \
    --meta-support $META_SUPPORT --meta-query $META_QUERY \
    --inner-steps $INNER_STEPS --inner-lr $INNER_LR \
    --meta-iters $META_ITERS \
    --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
    --fold-index $FOLD --out-path "$META_DIR" \
    --log-file "$LOG_DIR/meta_on_base_fold${FOLD}.log"
fi

# ---- stage 4: metatrain on ADV ----------------------------------------------
if [[ -f "$META_ADV_CKPT" ]]; then
  echo "[fold $FOLD] stage 4 metatrain (on adv): skip ($META_ADV_CKPT exists)"
else
  echo "[fold $FOLD] stage 4 metatrain (on adv)"
  $CLI metatrain $COMMON_DATA --weights imagenet \
    --init-checkpoint "$ADV_CKPT" \
    --adapter $ADAPTER \
    --meta-support $META_SUPPORT --meta-query $META_QUERY \
    --inner-steps $INNER_STEPS --inner-lr $INNER_LR \
    --meta-iters $META_ITERS \
    --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
    --fold-index $FOLD --out-path "$META_ADV_DIR" \
    --log-file "$LOG_DIR/meta_on_adv_fold${FOLD}.log"
fi

# ---- stage 5a: PSO hyperparameter search for the SVR baseline ---------------
SVR_HP_JSON="$OUT_ROOT/svr_hp_fold${FOLD}.json"
if [[ "$SVR_TUNE" == "1" ]]; then
  if [[ -f "$SVR_HP_JSON" ]]; then
    echo "[fold $FOLD] stage 5a svrsearch: skip ($SVR_HP_JSON exists)"
  else
    echo "[fold $FOLD] stage 5a svrsearch (PSO over (C, gamma, epsilon))"
    $CLI svrsearch $COMMON_DATA \
      --base-checkpoint "$BASE_CKPT" \
      --k $META_SUPPORT --trials 3 --pop $SVR_POP --iters $SVR_ITERS \
      --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
      --fold-index $FOLD --json-out "$SVR_HP_JSON" \
      --log-file "$LOG_DIR/svrsearch_fold${FOLD}.log"
  fi
  # Extract tuned (C, gamma, epsilon) from the JSON for the current fold.
  read SVR_C SVR_GAMMA SVR_EPS < <(python -c "
import json
hp = json.load(open('$SVR_HP_JSON'))['$FOLD']
print(hp['C'], hp['gamma'], hp['epsilon'])
")
  echo "[fold $FOLD] tuned SVR: C=$SVR_C gamma=$SVR_GAMMA epsilon=$SVR_EPS"
fi

# ---- stage 5b: metacompare K-sweep ------------------------------------------
echo "[fold $FOLD] stage 5b metacompare K-sweep"
FC_FT_ARGS=""
if [[ "$FC_FT" == "1" ]]; then
  FC_FT_ARGS="--fc-ft --fc-ft-steps $FC_FT_STEPS --fc-ft-lr $FC_FT_LR --fc-ft-weight-decay $FC_FT_WD"
fi
for K in $K_SWEEP; do
  echo "[fold $FOLD]   K=$K"
  $CLI metacompare $COMMON_DATA \
    --base-checkpoint "$BASE_CKPT" \
    --meta-checkpoint "$META_CKPT" \
    --meta-adv-checkpoint "$META_ADV_CKPT" \
    --k $K --trials $TRIALS \
    --inner-steps $INNER_STEPS --inner-lr $INNER_LR \
    --svr-C $SVR_C --svr-gamma $SVR_GAMMA --svr-eps $SVR_EPS \
    $FC_FT_ARGS \
    --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
    --fold-index $FOLD --csv-out "$CSV_OUT" \
    --log-file "$LOG_DIR/metacompare_K${K}_fold${FOLD}.log"
done

echo "[fold $FOLD] done. CSV: $CSV_OUT"
