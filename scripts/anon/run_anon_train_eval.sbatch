#!/bin/bash
# Evaluate the condition-(i) checkpoints on BOTH test sets, with the exact
# metacompare call used for Table 3:
#   test_real  trained on de-identified, deployed on real faces
#              (a third party trains on the release and ships it)
#   test_anon  trained and tested on de-identified data
#              (benchmarking on the release itself)
# Condition (ii) (clean-trained, anonymised test) stays as Table 3's third column.
# `meta` is meaningless here (seed42 clean adapter); read `base`.
#
# ARMS env selects the family; BBS env the backbones (default the three fast
# ones); REP the initialisation draw (1 = runs/anon_train/<arm>, N>1 =
# runs/anon_train/rep<N>/<arm>). Array index -> (arm, backbone, fold); submit
# with --array=0-$((n_arms*n_bbs*5-1)).
#SBATCH --job-name=at_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-59%10
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/at_eval_%A_%a.log
set -uo pipefail
cd /springbrook/share/eng/esrpxk/CogntiveGaze
CLI="/springbrook/share/eng/esrpxk/envs/gaze/bin/python vit_gaze_segmenter.py"
D=/springbrook/share/eng/esrpxk/datasets
R=/springbrook/share/eng/esrpxk/runs
OUT=$R/anon_train/eval; mkdir -p "$OUT/logs"
read -ra A <<< "${ARMS:?}"
read -ra BBS <<< "${BBS:-affnet mgazenet mobilenet_v3}"
REP=${REP:-1}
NB=${#BBS[@]}
ID=$SLURM_ARRAY_TASK_ID
ARM=${A[$(( ID / (NB * 5) ))]}; TAG=${ARM%%:*}; ROOT=$D/${ARM#*:}
BB=${BBS[$(( (ID % (NB * 5)) / 5 ))]}; FOLD=$(( ID % 5 ))
if [ "$REP" = 1 ]; then TR=$R/anon_train; SFX=""; else TR=$R/anon_train/rep$REP; SFX="_rep$REP"; fi
BASE_CKPT="$TR/$TAG/$BB/base/seed42/fold${FOLD}_best_${BB}_gaze_segmenter.pth"
META_CKPT="$R/meta_pipeline_clean_metacmp_${BB}/meta_on_base_calib/seed42/fold${FOLD}_meta_film_${BB}_gaze.pth"
[ -f "$BASE_CKPT" ] || { echo "FATAL missing $BASE_CKPT"; exit 1; }
for TEST in real:$D/ProcessedData anon:$ROOT; do
  TT=${TEST%%:*}; DP=${TEST#*:}
  echo "########## train=$TAG test=$TT $BB fold$FOLD $(date +%T)"
  $CLI metacompare \
    --data-path "$DP" --eye-path "$DP" --mean-path meanno7_clean \
    --input-mode multistream --backbone $BB --use-grid \
    --base-checkpoint "$BASE_CKPT" --meta-checkpoint "$META_CKPT" \
    --calib-support-root "$D/calib_support_K72" \
    --svr-embed --svr-embed-gamma scale \
    --fc-ft --fc-ft-steps 20 --fc-ft-lr 5e-5 --fc-ft-weight-decay 5e-4 \
    --k 72 --trials 1 --batch-size 128 --num-workers 8 \
    --fold-index $FOLD --seed 42 \
    --csv-out "$OUT/at_${TAG}_${BB}_test${TT}${SFX}.csv" \
    --log-file "$OUT/logs/at_${TAG}_${BB}_test${TT}${SFX}_f${FOLD}.log" \
    || echo "FAILED $TAG $BB fold$FOLD test=$TT"
done
echo "ALL_DONE $TAG $BB fold$FOLD"
