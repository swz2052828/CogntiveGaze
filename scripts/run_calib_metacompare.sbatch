#!/bin/bash
# scripts/run_calib_metacompare.sbatch
# Meta-calibration comparison using the DETERMINISTIC pre-task calibration frames
# (calib_support_K<K>) instead of random K-from-test sampling. One array task per
# fold; loops K in CALIB_KS. base/meta checkpoints from an existing trained run.
#
#   sbatch --array=0-4 scripts/run_calib_metacompare.sbatch                # full sweep
#   CALIB_KS=18 sbatch --array=0 scripts/run_calib_metacompare.sbatch      # smoke (fold0,K18)
#SBATCH --job-name=calib_mcmp
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/calib_mcmp_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/calib_mcmp_%A_%a.err

set -euo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze

module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze || { echo "[ERR] activate gaze env failed"; exit 1; }

: "${DATA_PATH:=/springbrook/share/eng/esrpxk/datasets/ProcessedData}"
: "${MEAN_PATH:=meanno7}"
: "${BACKBONE:=eyes_only_convnextv2_atto}"
: "${ADAPTER:=film}"
: "${SEED:=42}"
: "${CALIB_KS:=4 9 18 36 72}"
: "${RUN_ROOT:=/springbrook/share/eng/esrpxk/runs/meta_pipeline_eyes_only_convnextv2_atto_lr1e4}"
: "${SUPP_ROOT:=/springbrook/share/eng/esrpxk/datasets}"
: "${OUT_ROOT:=/springbrook/share/eng/esrpxk/runs/calib_metacompare}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=16}"
: "${INNER_STEPS:=20}"
: "${INNER_LR:=1.0}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
# Which meta checkpoints to evaluate. Default = original (in-task-support) meta;
# set META_BASE_SUBDIR=meta_on_base_calib META_ADV_SUBDIR=meta_on_adv_calib
# (and CSV_TAG=_calib) to evaluate the deploy-faithful calibration-trained adapter.
: "${META_BASE_SUBDIR:=meta_on_base}"
: "${META_ADV_SUBDIR:=meta_on_adv}"
: "${CSV_TAG:=}"
mkdir -p "$OUT_ROOT/logs"
CSV_OUT="$OUT_ROOT/calib_metacompare_${BACKBONE}${CSV_TAG}.csv"

BASE_CKPT="$RUN_ROOT/base/seed${SEED}/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
BASE_ADV_CKPT="$RUN_ROOT/adv/seed${SEED}/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
META_CKPT="$RUN_ROOT/${META_BASE_SUBDIR}/seed${SEED}/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
META_ADV_CKPT="$RUN_ROOT/${META_ADV_SUBDIR}/seed${SEED}/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
CLI="python vit_gaze_segmenter.py"
COMMON="--data-path $DATA_PATH --eye-path $DATA_PATH --mean-path $MEAN_PATH \
  --input-mode multistream --backbone $BACKBONE --use-grid"

# SVR hyperparameters: SVR_TUNE=1 runs a leak-free swarm search (per fold, per K,
# per space) on the TRAINING subjects' calibration support before each metacompare
# (deploy-faithful: fit on calib frames, score on in-task frames). SVR_TUNE=0 keeps
# fixed literature defaults C=1, gamma=scale, eps=0.1.
: "${SVR_TUNE:=1}"
: "${SVR_OPTIMIZER:=pso}"
: "${SVR_POP:=20}"
: "${SVR_ITERS:=40}"
HPDIR="$OUT_ROOT/svr_hp_${BACKBONE}${CSV_TAG}"
mkdir -p "$HPDIR"

echo "Host $(hostname)  fold=$FOLD  KS=$CALIB_KS  mean=$MEAN_PATH  SVR_TUNE=$SVR_TUNE  $(date)"
for K in $CALIB_KS; do
  echo "==== fold $FOLD  K=$K ===="
  CALIB_ROOT="$SUPP_ROOT/calib_support_K${K}"
  SVR_HP_ARGS="--svr-embed --svr-embed-gamma scale"   # default: fixed HPs
  if [[ "$SVR_TUNE" == "1" ]]; then
    PRED_HP="$HPDIR/pred_K${K}_fold${FOLD}.json"
    EMB_HP="$HPDIR/embed_K${K}_fold${FOLD}.json"
    SVR_COMMON="$COMMON --base-checkpoint $BASE_CKPT --calib-support-root $CALIB_ROOT \
      --optimizer $SVR_OPTIMIZER --pop $SVR_POP --iters $SVR_ITERS \
      --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --fold-index $FOLD --seed $SEED"
    [[ -f "$PRED_HP" ]] || $CLI svrsearch $SVR_COMMON --space prediction --json-out "$PRED_HP" \
      --log-file "$OUT_ROOT/logs/svrtune_pred_K${K}_fold${FOLD}.log"
    [[ -f "$EMB_HP" ]]  || $CLI svrsearch $SVR_COMMON --space embedding  --json-out "$EMB_HP" \
      --log-file "$OUT_ROOT/logs/svrtune_embed_K${K}_fold${FOLD}.log"
    read PC PG PE < <(python -c "import json;h=json.load(open('$PRED_HP'))['$FOLD'];print(h['C'],h['gamma'],h['epsilon'])")
    read EC EG EE < <(python -c "import json;h=json.load(open('$EMB_HP'))['$FOLD'];print(h['C'],h['gamma'],h['epsilon'])")
    SVR_HP_ARGS="--svr-C $PC --svr-gamma $PG --svr-eps $PE \
      --svr-embed --svr-embed-C $EC --svr-embed-gamma $EG --svr-embed-eps $EE"
    echo "fold $FOLD K=$K tuned SVR: pred(C=$PC g=$PG e=$PE) embed(C=$EC g=$EG e=$EE)"
  fi
  ADV_ARGS='--base-adv-checkpoint '"$BASE_ADV_CKPT"' --meta-adv-checkpoint '"$META_ADV_CKPT"
  [ "${NO_ADV:-0}" = "1" ] && ADV_ARGS=""   # vivit: adv aliases base -> columns carry no info
  $CLI metacompare $COMMON \
    --base-checkpoint "$BASE_CKPT" --meta-checkpoint "$META_CKPT" \
    $ADV_ARGS \
    --calib-support-root "$CALIB_ROOT" \
    $SVR_HP_ARGS \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k $K --trials 1 --inner-steps $INNER_STEPS --inner-lr $INNER_LR \
    --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
    --fold-index $FOLD --seed $SEED --csv-out "$CSV_OUT" \
    --log-file "$OUT_ROOT/logs/calib_mcmp_K${K}_fold${FOLD}.log"
done
echo "Done fold $FOLD: $(date)"
