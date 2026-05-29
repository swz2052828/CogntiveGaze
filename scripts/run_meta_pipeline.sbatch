#!/bin/bash
# scripts/run_meta_pipeline.sbatch
#
# Site-adapted version of scripts/meta_pipeline.sbatch (Bletchley HPC). Adds
# the cluster's module loading + conda activation, system-monitor CSV, and a
# fresh-clone-or-pull gate so repeat runs don't re-download the repo. All
# pipeline knobs are overridable via --export=ALL,VAR=val,...
#
# Submit:
#   sbatch scripts/run_meta_pipeline.sbatch                              # full sweep, defaults
#   sbatch --array=2 --export=ALL,SEEDS=42 scripts/run_meta_pipeline.sbatch  # smoke test (fold 2)

#SBATCH --job-name=vit_gaze_meta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/meta_%A_fold%a.out
#SBATCH --error=logs/meta_%A_fold%a.err

# ==============================================================================
# Pipeline config -- override via --export=ALL,VAR=val,...
# ==============================================================================
: "${BRANCH:=claude/serene-goodall-ld3iR}"
: "${REPO_URL:=https://github.com/swz2052828/CogntiveGaze.git}"
: "${DATA_PATH:=../datasets/ProcessedData}"
: "${EYE_PATH:=../datasets/ProcessedData}"
: "${MEAN_PATH:=meanno7}"
: "${OUT_ROOT:=../runs/meta_pipeline}"
: "${BACKBONE:=mobilenet_v3}"
: "${SEEDS:=42 123 7}"
: "${EPOCHS:=20}"
: "${BATCH_SIZE:=256}"
: "${NUM_WORKERS:=8}"
: "${LR:=1e-4}"
: "${LR_SCHEDULER:=cosine}"
: "${AUGMENT:=light}"
: "${PATIENCE:=5}"
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
: "${SVR_TUNE:=1}"
: "${SVR_POP:=30}"
: "${SVR_ITERS:=50}"
: "${SVR_EMBED:=1}"
: "${FC_FT:=1}"
: "${FC_FT_STEPS:=20}"
: "${FC_FT_LR:=5e-5}"
: "${FC_FT_WD:=5e-4}"

echo "Running on: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "Memory allocated (MB): $SLURM_MEM_PER_NODE"
echo "Fold index: $SLURM_ARRAY_TASK_ID"
echo "Seeds: $SEEDS"
echo "GPU info:"
nvidia-smi -L

# ==============================================================================
# Modules + conda env (Bletchley-specific)
# ==============================================================================
ml load conda/miniforge3 cuda/13.0
eval "$(conda shell.bash hook)"
conda activate gaze || { echo "[ERR] failed to activate gaze"; exit 1; }

mkdir -p logs

# ==============================================================================
# System monitor (background; CSV + summary)
# ==============================================================================
LOG_DIR="./system_logs"
mkdir -p "$LOG_DIR"
COMBINED_LOG="${LOG_DIR}/system_monitor_${SLURM_JOB_ID}.csv"
SUMMARY_FILE="${LOG_DIR}/system_summary_${SLURM_JOB_ID}.txt"
echo "Timestamp,GPU_Util(%),GPU_Mem_Used(MB),GPU_Mem_Total(MB),GPU_Temp(C),GPU_Power(W),CPU_Usage(%),Mem_Used(MB),Mem_Free(MB),Mem_Available(MB)" > "$COMBINED_LOG"

(
while true; do
    TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | head -n1)
    CPU_USAGE=$(top -b -n1 | awk '/Cpu\(s\)/ {print 100 - $8}')
    read MEM_USED MEM_FREE MEM_AVAIL <<< $(free -m | awk 'NR==2{print $3, $4, $7}')
    echo "${TIMESTAMP},${GPU_STATS},${CPU_USAGE},${MEM_USED},${MEM_FREE},${MEM_AVAIL}" >> "$COMBINED_LOG"
    sleep 5
done
) &
MONITOR_PID=$!
trap 'kill -0 $MONITOR_PID 2>/dev/null && kill $MONITOR_PID' EXIT

# ==============================================================================
# Repo: clone on first run, pull-and-checkout on repeats
# ==============================================================================
if [[ -d CogntiveGaze/.git ]]; then
  echo "[repo] CogntiveGaze/ exists; fetching $BRANCH"
  cd CogntiveGaze
  git fetch --quiet origin "$BRANCH"
  git checkout --quiet "$BRANCH"
  git pull --quiet --ff-only origin "$BRANCH"
else
  echo "[repo] cloning $REPO_URL"
  git clone --quiet "$REPO_URL"
  cd CogntiveGaze
  git checkout --quiet "$BRANCH"
fi
echo "[repo] HEAD: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

set -euo pipefail

# ==============================================================================
# Per (fold, seed) pipeline
# ==============================================================================
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
BASE_DIR="$OUT_ROOT/base"
ADV_DIR="$OUT_ROOT/adv"
META_DIR="$OUT_ROOT/meta_on_base"
META_ADV_DIR="$OUT_ROOT/meta_on_adv"
PIPE_LOGS="$OUT_ROOT/logs"
mkdir -p "$BASE_DIR" "$ADV_DIR" "$META_DIR" "$META_ADV_DIR" "$PIPE_LOGS"
CSV_OUT="$OUT_ROOT/metacompare.csv"

CLI="srun python vit_gaze_segmenter.py"
COMMON_DATA="--data-path $DATA_PATH --eye-path $EYE_PATH --mean-path $MEAN_PATH \
  --input-mode multistream --backbone $BACKBONE --use-grid"

for SEED in $SEEDS; do
  SBD="$BASE_DIR/seed${SEED}";       mkdir -p "$SBD"
  SAD="$ADV_DIR/seed${SEED}";        mkdir -p "$SAD"
  SMD="$META_DIR/seed${SEED}";       mkdir -p "$SMD"
  SAMD="$META_ADV_DIR/seed${SEED}";  mkdir -p "$SAMD"

  BASE_CKPT="$SBD/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
  ADV_CKPT="$SAD/fold${FOLD}_best_${BACKBONE}_gaze_segmenter.pth"
  META_CKPT="$SMD/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"
  META_ADV_CKPT="$SAMD/fold${FOLD}_meta_${ADAPTER}_${BACKBONE}_gaze.pth"

  COMMON_TRAIN="$COMMON_DATA --weights imagenet --epochs $EPOCHS --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS --lr $LR --lr-scheduler $LR_SCHEDULER --augment $AUGMENT \
    --patience $PATIENCE --min-delta $MIN_DELTA --amp --compile \
    --fold-index $FOLD --seed $SEED"

  echo "===================================================="
  echo "[fold $FOLD seed $SEED] starting full pipeline"
  echo "===================================================="

  # Stage 1: base train
  if [[ -f "$BASE_CKPT" ]]; then
    echo "[fold $FOLD seed $SEED] stage 1 base: skip ($BASE_CKPT exists)"
  else
    $CLI train $COMMON_TRAIN --out-path "$SBD" \
      --log-file "$PIPE_LOGS/base_seed${SEED}_fold${FOLD}.log"
  fi

  # Stage 2: subject-adv train
  if [[ -f "$ADV_CKPT" ]]; then
    echo "[fold $FOLD seed $SEED] stage 2 adv: skip ($ADV_CKPT exists)"
  else
    $CLI train $COMMON_TRAIN --out-path "$SAD" \
      --subject-adv --adv-weight $ADV_WEIGHT \
      --log-file "$PIPE_LOGS/adv_seed${SEED}_fold${FOLD}.log"
  fi

  # Stage 3: metatrain on BASE
  if [[ -f "$META_CKPT" ]]; then
    echo "[fold $FOLD seed $SEED] stage 3 meta(base): skip"
  else
    $CLI metatrain $COMMON_DATA --weights imagenet \
      --init-checkpoint "$BASE_CKPT" --adapter $ADAPTER \
      --meta-support $META_SUPPORT --meta-query $META_QUERY \
      --inner-steps $INNER_STEPS --inner-lr $INNER_LR --meta-iters $META_ITERS \
      --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
      --fold-index $FOLD --seed $SEED --out-path "$SMD" \
      --log-file "$PIPE_LOGS/meta_on_base_seed${SEED}_fold${FOLD}.log"
  fi

  # Stage 4: metatrain on ADV
  if [[ -f "$META_ADV_CKPT" ]]; then
    echo "[fold $FOLD seed $SEED] stage 4 meta(adv): skip"
  else
    $CLI metatrain $COMMON_DATA --weights imagenet \
      --init-checkpoint "$ADV_CKPT" --adapter $ADAPTER \
      --meta-support $META_SUPPORT --meta-query $META_QUERY \
      --inner-steps $INNER_STEPS --inner-lr $INNER_LR --meta-iters $META_ITERS \
      --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
      --fold-index $FOLD --seed $SEED --out-path "$SAMD" \
      --log-file "$PIPE_LOGS/meta_on_adv_seed${SEED}_fold${FOLD}.log"
  fi

  # Stage 5a: PSO svrsearch (prediction space)
  SVR_HP="$OUT_ROOT/svr_hp_seed${SEED}_fold${FOLD}.json"
  if [[ "$SVR_TUNE" == "1" ]]; then
    if [[ ! -f "$SVR_HP" ]]; then
      $CLI svrsearch $COMMON_DATA --base-checkpoint "$BASE_CKPT" --space prediction \
        --k $META_SUPPORT --trials 3 --pop $SVR_POP --iters $SVR_ITERS \
        --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
        --fold-index $FOLD --seed $SEED --json-out "$SVR_HP" \
        --log-file "$PIPE_LOGS/svrsearch_seed${SEED}_fold${FOLD}.log"
    fi
    read SVR_C SVR_GAMMA SVR_EPS < <(python -c "
import json; hp=json.load(open('$SVR_HP'))['$FOLD']
print(hp['C'], hp['gamma'], hp['epsilon'])")
  else
    SVR_C=1.0; SVR_GAMMA=scale; SVR_EPS=0.1
  fi

  # Stage 5b: PSO svrsearch (embedding space)
  SVR_EMBED_HP="$OUT_ROOT/svr_embed_hp_seed${SEED}_fold${FOLD}.json"
  if [[ "$SVR_EMBED" == "1" && "$SVR_TUNE" == "1" ]]; then
    if [[ ! -f "$SVR_EMBED_HP" ]]; then
      $CLI svrsearch $COMMON_DATA --base-checkpoint "$BASE_CKPT" --space embedding \
        --k $META_SUPPORT --trials 3 --pop $SVR_POP --iters $SVR_ITERS \
        --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
        --fold-index $FOLD --seed $SEED --json-out "$SVR_EMBED_HP" \
        --log-file "$PIPE_LOGS/svrsearch_embed_seed${SEED}_fold${FOLD}.log"
    fi
    read SVR_E_C SVR_E_G SVR_E_E < <(python -c "
import json; hp=json.load(open('$SVR_EMBED_HP'))['$FOLD']
print(hp['C'], hp['gamma'], hp['epsilon'])")
  else
    SVR_E_C=1.0; SVR_E_G=scale; SVR_E_E=0.1
  fi

  # Stage 5c: metacompare K-sweep
  EXTRA=""
  [[ "$FC_FT" == "1" ]] && EXTRA="$EXTRA --fc-ft --fc-ft-steps $FC_FT_STEPS --fc-ft-lr $FC_FT_LR --fc-ft-weight-decay $FC_FT_WD"
  [[ "$SVR_EMBED" == "1" ]] && EXTRA="$EXTRA --svr-embed --svr-embed-C $SVR_E_C --svr-embed-gamma $SVR_E_G --svr-embed-eps $SVR_E_E"

  for K in $K_SWEEP; do
    $CLI metacompare $COMMON_DATA \
      --base-checkpoint "$BASE_CKPT" \
      --meta-checkpoint "$META_CKPT" \
      --meta-adv-checkpoint "$META_ADV_CKPT" \
      --k $K --trials $TRIALS \
      --inner-steps $INNER_STEPS --inner-lr $INNER_LR \
      --svr-C $SVR_C --svr-gamma $SVR_GAMMA --svr-eps $SVR_EPS \
      $EXTRA \
      --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS \
      --fold-index $FOLD --seed $SEED --csv-out "$CSV_OUT" \
      --log-file "$PIPE_LOGS/metacompare_K${K}_seed${SEED}_fold${FOLD}.log"
  done

  echo "[fold $FOLD seed $SEED] done."
done   # end seed loop

# ==============================================================================
# System summary (unchanged from prior driver)
# ==============================================================================
kill $MONITOR_PID 2>/dev/null || true

echo "Generating system summary..." > "$SUMMARY_FILE"
awk -F',' '
NR>1 {
    gu=$2; gm=$3; gt=$5; gp=$6; cu=$7; mu=$8
    sum_gu+=gu; sum_gm+=gm; sum_gt+=gt; sum_gp+=gp; sum_cu+=cu; sum_mu+=mu
    count++
    if(min_gu=="" || gu<min_gu) min_gu=gu; if(gu>max_gu) max_gu=gu
    if(min_gm=="" || gm<min_gm) min_gm=gm; if(gm>max_gm) max_gm=gm
    if(min_gt=="" || gt<min_gt) min_gt=gt; if(gt>max_gt) max_gt=gt
    if(min_gp=="" || gp<min_gp) min_gp=gp; if(gp>max_gp) max_gp=gp
    if(min_cu=="" || cu<min_cu) min_cu=cu; if(cu>max_cu) max_cu=cu
    if(min_mu=="" || mu<min_mu) min_mu=mu; if(mu>max_mu) max_mu=mu
}
END {
    printf "Metric,Min,Max,Avg\n" >> "'"$SUMMARY_FILE"'"
    printf "GPU Util(%%),%.1f,%.1f,%.1f\n", min_gu, max_gu, sum_gu/count >> "'"$SUMMARY_FILE"'"
    printf "GPU Mem Used(MB),%.0f,%.0f,%.0f\n", min_gm, max_gm, sum_gm/count >> "'"$SUMMARY_FILE"'"
    printf "GPU Temp(C),%.1f,%.1f,%.1f\n", min_gt, max_gt, sum_gt/count >> "'"$SUMMARY_FILE"'"
    printf "GPU Power(W),%.1f,%.1f,%.1f\n", min_gp, max_gp, sum_gp/count >> "'"$SUMMARY_FILE"'"
    printf "CPU Usage(%%),%.1f,%.1f,%.1f\n", min_cu, max_cu, sum_cu/count >> "'"$SUMMARY_FILE"'"
    printf "Mem Used(MB),%.0f,%.0f,%.0f\n", min_mu, max_mu, sum_mu/count >> "'"$SUMMARY_FILE"'"
}' "$COMBINED_LOG"

echo "Combined log saved to: $COMBINED_LOG"
echo "Summary saved to:      $SUMMARY_FILE"
echo "Pipeline CSV:          $CSV_OUT"
