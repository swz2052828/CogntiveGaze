#!/bin/bash
# Round 3 (2026-10-08), in the order the author set, enforced with priorities
# (nice) so later stages only use GPUs the earlier ones leave idle:
#
#   stage 1  nice 0    DP2 FULL-BODY utility: full roots (both identity regimes)
#                      -> condition (ii) on all 5 init draws x 4 backbones
#                      -> condition (i) retrain, 3 fast backbones, draw 1
#   stage 2  nice 50   condition (i) draws 2-5 for every arm (SimSwap, DP2 face,
#                      DP2 full body), 3 fast backbones
#   stage 3  nice 100  condition (i) on iTracker, every arm, draw 1
#
# Fast backbones run at 10 CPUs so 15 jobs fit the 160-CPU / 15-GPU QOS cap
# (DataLoader workers do not change the recipe). Every job id is appended to
# scripts/anon/round3_jobids.txt.
#
#   bash scripts/anon/submit_round3.sh            # submit
set -euo pipefail
REPO=/springbrook/share/eng/esrpxk/CogntiveGaze
cd "$REPO"
LOG=scripts/anon/round3_jobids.txt
: > "$LOG"
S=scripts/anon

FB_ARMS="dp2fbs:ProcessedDP2fbfull_per_subject dp2fbs_oldeye:ProcessedDP2fbfull_per_subject_oldeye dp2fbf:ProcessedDP2fbfull_per_frame dp2fbf_oldeye:ProcessedDP2fbfull_per_frame_oldeye"
declare -A FAM_ARMS=(
  [swap]="swap1:ProcessedSwap swap1_oldeye:ProcessedSwap_oldeye swap2:ProcessedSwap2 swap2_oldeye:ProcessedSwap2_oldeye"
  [dp2]="dp2s:ProcessedDP2full_per_subject dp2s_oldeye:ProcessedDP2full_per_subject_oldeye dp2f:ProcessedDP2full_per_frame dp2f_oldeye:ProcessedDP2full_per_frame_oldeye"
  [dp2fb]="$FB_ARMS"
)
rec() { echo "$*" | tee -a "$LOG"; }

# train one family; prints the colon-joined job ids
train() {   # fam rep nice backbones cpus time dep
  DEP="$7" REP="$2" NICE="$3" BACKBONES="$4" CPUS="$5" TIME="$6" \
    bash $S/submit_anon_train.sh "$1" | sed -n 's/^TRAIN_IDS=//p'
}
evaluate() {   # fam rep nice backbones dep(afterany)
  local nb; nb=$(wc -w <<< "$4")
  local n; n=$(( $(wc -w <<< "${FAM_ARMS[$1]}") * nb * 5 - 1 ))
  ARMS="${FAM_ARMS[$1]}" BBS="$4" REP="$2" sbatch --parsable --array=0-$n%10 \
    --nice="$3" --job-name="ate$2_$1" --dependency=afterany:"$5" $S/run_anon_train_eval.sbatch
}
FAST="affnet mgazenet mobilenet_v3"

# ------------------------------------------------------------------ stage 1
G=$(MODE=fullbody sbatch --parsable --mem=48G --job-name=dp2fb_root $S/run_dp2_fullroot_gen.sbatch)
rec "stage1 gen $G"
B=$(PFX=ProcessedDP2fbfull sbatch --parsable --dependency=afterok:$G $S/run_dp2_fullroot_build.sbatch)
rec "stage1 build $B"
E2=$(ARMS="none:ProcessedData $FB_ARMS" sbatch --parsable --dependency=afterok:$B \
      --job-name=dp2fb_util $S/run_dp2_fullroot_eval.sbatch)
rec "stage1 cond-ii eval $E2"
T=$(train dp2fb 1 0 "$FAST" 10 "" "$B");          rec "stage1 train dp2fb rep1 $T"
rec "stage1 eval dp2fb rep1 $(evaluate dp2fb 1 0 "$FAST" "$T")"

# ------------------------------------------------------------------ stage 2
for REP in 2 3 4 5; do
  for FAM in swap dp2 dp2fb; do
    D=""; [ "$FAM" = dp2fb ] && D="$B"
    T=$(train $FAM $REP 50 "$FAST" 10 "" "$D");   rec "stage2 train $FAM rep$REP $T"
    rec "stage2 eval $FAM rep$REP $(evaluate $FAM $REP 50 "$FAST" "$T")"
  done
done

# ------------------------------------------------------------------ stage 3
for FAM in swap dp2 dp2fb; do
  D=""; [ "$FAM" = dp2fb ] && D="$B"
  T=$(train $FAM 1 100 itracker 10 36:00:00 "$D"); rec "stage3 train $FAM itracker $T"
  rec "stage3 eval $FAM itracker $(evaluate $FAM 1 100 itracker "$T")"
done
echo "submitted; ids in $LOG"
