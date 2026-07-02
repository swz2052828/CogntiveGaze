#!/bin/bash
# Step 1: re-evaluate EXISTING base checkpoints on clean vs dirty val.
# Array over the canonical backbones in scripts/clean_canonical_runs.tsv
# (convnextv2_dualenc skipped). One GPU task per backbone; per-backbone CSV.
#
#   sbatch --array=0-32 scripts/run_clean_eval.sbatch
#SBATCH --job-name=clean_eval
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/clean_eval_%A_%a.out
#SBATCH --error=/springbrook/share/eng/esrpxk/logs/clean_eval_%A_%a.err

set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
RUNS=/springbrook/share/eng/esrpxk/runs
SUBDIR="${SUBDIR:-base}"           # base | adv
OUTDIR=/springbrook/share/eng/esrpxk/results/clean_eval${SUBDIR:+_$SUBDIR}
[ "$SUBDIR" = "base" ] && OUTDIR=/springbrook/share/eng/esrpxk/results/clean_eval
MANIFEST="$REPO/scripts/clean_canonical_runs.tsv"
mkdir -p "$OUTDIR" /springbrook/share/eng/esrpxk/logs

module load Miniconda3/24.7.1-0 CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate /springbrook/share/eng/esrpxk/envs/gaze
export HF_HUB_OFFLINE=1 HF_HOME=/springbrook/home/e/esrpxk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN_OVERRIDE/BB_OVERRIDE let us submit a single backbone (e.g. a late add)
# without depending on the manifest line order; otherwise index into the manifest.
if [ -n "${RUN_OVERRIDE:-}" ] && [ -n "${BB_OVERRIDE:-}" ]; then
  RUN="$RUN_OVERRIDE"; BB="$BB_OVERRIDE"
else
  mapfile -t LINES < <(tail -n +2 "$MANIFEST")
  LINE="${LINES[$SLURM_ARRAY_TASK_ID]}"
  RUN=$(echo "$LINE" | cut -f1); BB=$(echo "$LINE" | cut -f2)
fi
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID run=$RUN backbone=$BB"
nvidia-smi -L

cd "$REPO"
srun python eval_clean_vs_dirty.py \
  --run-dir "$RUNS/$RUN" --backbone "$BB" --subdir "$SUBDIR" \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --out-csv "$OUTDIR/${BB}.csv"
echo "done $BB ($SUBDIR)"
