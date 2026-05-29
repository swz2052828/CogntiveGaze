#!/bin/bash
# scripts/plot_metacompare.sh
#
# Render the K-sweep figure from the metacompare CSV produced by
# scripts/meta_pipeline.sbatch. Cheap (CPU, ~seconds), so it usually runs
# straight on the login node rather than as a Slurm job.
#
# Usage:
#   bash scripts/plot_metacompare.sh                                  # defaults
#   CSV=./runs/meta_pipeline/metacompare.csv OUT=kcurve.png \
#     bash scripts/plot_metacompare.sh
set -euo pipefail
: "${CSV:=./runs/meta_pipeline/metacompare.csv}"
: "${OUT:=./runs/meta_pipeline/kcurve.png}"
: "${TITLE:=Calibration error vs K (5-fold CV)}"

if [[ ! -f "$CSV" ]]; then
  echo "No CSV at $CSV. Run scripts/meta_pipeline.sbatch first." >&2
  exit 1
fi

python -m vit_gaze.plot_metacompare --csv "$CSV" --out "$OUT" --title "$TITLE"
echo "wrote $OUT"
