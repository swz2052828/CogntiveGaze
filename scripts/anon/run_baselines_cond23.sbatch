#!/bin/bash
#SBATCH --job-name=base_c23
#SBATCH --partition=gpu
# NOT gpu,vis -- vis QOS caps a user at one GPU and silently cancels the rest of
# a multi-task array (QOSMaxGRESPerUser).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-29%4
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/base_c23_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
export PYTHONUNBUFFERED=1
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
OUT=/springbrook/share/eng/esrpxk/runs/anon_baselines
mkdir -p "$OUT/logs"

# The comparison architectures for this stage: mobilenet_v3, affnet, mgazenet.
# All three are CNN baselines that require --use-grid, all have complete clean
# metacmp checkpoints (5 base + 5 meta each), so this is evaluation only -- no
# training. itracker is NOT re-run here: it stays as the existing baseline.
#
# Both enrolment conditions, because the mobile_vit result showed the choice is
# not neutral and is backbone-dependent (clean enrolment was pessimistic for
# mobile_vit, optimistic for itracker). Reporting one condition only would
# under-determine the comparison.
#
# Only 5 operators exist: blur0.03/blur0.35 and all five pixel levels were
# deleted in the inode cleanup and cannot be regenerated cheaply.
BBS=(mobilenet_v3 affnet mgazenet)
CONDS=(ii iii)
I=$SLURM_ARRAY_TASK_ID
COND=${CONDS[$(( I / 15 ))]}
BB=${BBS[$(( (I % 15) / 5 ))]}
FOLD=$(( I % 5 ))
RR=/springbrook/share/eng/esrpxk/runs/meta_pipeline_clean_metacmp_$BB

for OP in none blur0.10 blur0.20 blur0.50 blackbox; do
  DP=$D/anon/$OP
  if [ "$COND" = "ii" ]; then CS=$D/calib_support_K72; else CS=$D/calib_support_K72_$OP; fi
  [ -d "$DP" ] || { echo "SKIP missing $DP"; continue; }
  [ -f "$CS/meanno7_clean/metadata.mat" ] || { echo "FATAL no manifest in $CS"; continue; }
  SHORT=""
  for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
    H=$(ls "$DP/$R/appleFace" 2>/dev/null | wc -l)
    W=$(ls "$D/ProcessedData/$R/appleFace" 2>/dev/null | wc -l)
    [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
    C=$(ls "$CS/$R/appleFace" 2>/dev/null | wc -l)
    [ "$C" -lt 72 ] && SHORT="$SHORT calib:$R($C)"
  done
  [ -n "$SHORT" ] && { echo "FATAL $OP cond=$COND: short:$SHORT"; continue; }

  echo "########## $BB fold$FOLD op=$OP cond=$COND $(date +%H:%M:%S)"
  # NOTE for analysis: affnet's `meta` column is known-unstable on clean data
  # (69.6 / 106.3 cm in the clean metacompare fleet). Discard meta for affnet;
  # base / svr_embed / fc_ft are sound.
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$RR/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth" \
    --meta-checkpoint "$RR/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth" \
    --calib-support-root "$CS" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/${COND}_${BB}_${OP}.csv" \
    --log-file "$OUT/logs/s4$([ "$COND" = "iii" ] && echo c3)_${BB}_${OP}_f${FOLD}.log" \
    || echo "FAILED $BB fold$FOLD $OP cond=$COND"
done
echo "ALL_DONE $BB fold$FOLD cond=$COND"
