#!/bin/bash
# Condition (i): TRAIN on de-identified data, for every arm of Table 3 plus DP2.
#
# Why. Table 3 is condition (ii): clean-trained checkpoints deployed on
# de-identified crops. Its penalty mixes information loss with plain domain shift
# (the model never saw synthetic texture). Retraining removes the shift, so what
# is left is what the release actually costs -- and it separates SimSwap (gaze
# roughly preserved, penalty may be mostly shift) from DP2 (eyes repainted, gaze
# direction altered: image and label disagree, retraining cannot fix that).
#
# Recipe is IDENTICAL to the clean checkpoints (run_base_only_springbrook.sbatch
# via clean_canonical_runs.tsv: lr 1e-4, 20 epochs, meanno7_clean, use-grid,
# output none, range 4.0, seed 42 = same participant partition). Only the data
# root changes. Mean images stay the real-data ones (shared via the meanno7_clean
# symlink): a fixed normalisation constant, identical across arms.
#
# The retrain `none` arm is not rerun: the five clean draws already are it, and
# they give the init noise floor (0.04-0.20 cm) to read these single draws against.
#
# Phase 1: the three fast backbones only. iTracker (~8.5 h/fold) is phase 2.
#   bash scripts/anon/submit_anon_train.sh swap            # SimSwap arms, now
#   DEP=<jobid> bash scripts/anon/submit_anon_train.sh dp2 # DP2 arms, after build
#   ... dp2fb                                              # DP2 full-body arms
#
# Environment knobs (all optional):
#   REP=1..5      initialisation draw. 1 is the original run (runs/anon_train/<arm>);
#                 2-5 go to runs/anon_train/rep<N>/<arm>. Same seed 42 = same
#                 participant partition; weights are unseeded, so a repeat is a
#                 new initialisation draw (see run_initvar_eval.sbatch).
#   BACKBONES     default "affnet mgazenet mobilenet_v3"
#   CPUS          cpus-per-task and DataLoader workers (default 16). Workers do
#                 not change the recipe; 10 lets 15 jobs fit the 160-CPU QOS cap.
#   NICE, TIME    scheduling only
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
D=/springbrook/share/eng/esrpxk/datasets
RUNS=/springbrook/share/eng/esrpxk/runs/anon_train
cd "$REPO"
case "${1:?swap|dp2}" in
  swap) ARMS="swap1:ProcessedSwap swap1_oldeye:ProcessedSwap_oldeye swap2:ProcessedSwap2 swap2_oldeye:ProcessedSwap2_oldeye" ;;
  dp2)  ARMS="dp2s:ProcessedDP2full_per_subject dp2s_oldeye:ProcessedDP2full_per_subject_oldeye dp2f:ProcessedDP2full_per_frame dp2f_oldeye:ProcessedDP2full_per_frame_oldeye" ;;
  dp2fb) ARMS="dp2fbs:ProcessedDP2fbfull_per_subject dp2fbs_oldeye:ProcessedDP2fbfull_per_subject_oldeye dp2fbf:ProcessedDP2fbfull_per_frame dp2fbf_oldeye:ProcessedDP2fbfull_per_frame_oldeye" ;;
  *) echo "usage: $0 swap|dp2|dp2fb"; exit 1 ;;
esac
BACKBONES="${BACKBONES:-affnet mgazenet mobilenet_v3}"
REP="${REP:-1}"
CPUS="${CPUS:-16}"
if [ "$REP" = 1 ]; then BASE_OUT="$RUNS"; else BASE_OUT="$RUNS/rep$REP"; fi
JOBLIST="$REPO/scripts/anon/anon_train_jobids.txt"
IDS=()
for ARM in $ARMS; do
  TAG=${ARM%%:*}; ROOT=$D/${ARM#*:}
  for BB in $BACKBONES; do
    OUT_ROOT="$BASE_OUT/$TAG/$BB"; mkdir -p "$OUT_ROOT"
    JID=$(DATA_PATH="$ROOT" EYE_PATH="$ROOT" MEAN_PATH=meanno7_clean OUT_ROOT="$OUT_ROOT" \
          BACKBONE="$BB" OUTPUT_ACTIVATION=none GAZE_RANGE=4.0 LR=1e-4 EPOCHS=20 SEEDS=42 \
          NUM_WORKERS="$CPUS" \
          sbatch --parsable --array=0-4 --partition=gpu --cpus-per-task="$CPUS" \
          --job-name="at${REP}_${TAG}_${BB}" ${NICE:+--nice=$NICE} ${TIME:+--time=$TIME} \
          ${DEP:+--dependency=afterok:$DEP} scripts/run_base_only_springbrook.sbatch)
    echo "$JID $1 rep$REP $TAG $BB" | tee -a "$JOBLIST"
    IDS+=("$JID")
  done
done
echo "TRAIN_IDS=$(IFS=:; echo "${IDS[*]}")"
