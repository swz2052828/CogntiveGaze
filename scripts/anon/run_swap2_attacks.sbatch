#!/bin/bash
# SimSwap second template (ProcessedSwap2): the face-crop half of the attack suite.
#
# Purpose: every SimSwap number so far rests on ONE template identity. A second
# template separates "this is what SimSwap does" from "this is what that one
# template face does". Same commands as run_swap_attacks.sbatch, same n, same
# seeds, so each row is directly comparable with its ProcessedSwap counterpart.
#
# Only ProcessedSwap2, not ProcessedSwap2_oldeye: the two ship byte-identical
# face crops, so any attack reading appleFace returns identical numbers for both
# (the tautology found on template 1). The eyes-preserved price is read off the
# eye crops, which for this template are still being repaired (00020 and 00023
# were truncated by the 2026-09-10 inode-quota stop) and run separately.
#SBATCH --job-name=swap2_face
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/swap2_face_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
D=/springbrook/share/eng/esrpxk/datasets
RES=/springbrook/share/eng/esrpxk/results/anon_swap2
mkdir -p "$RES"
COND=ProcessedSwap2

# face-crop coverage guard: the eye folders are known-short, the face must not be
SHORT=""
for R in 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023; do
  H=$(ls "$D/$COND/$R/appleFace" 2>/dev/null | wc -l); W=$(ls "$D/ProcessedData/$R/appleFace" | wc -l)
  [ "$H" -lt "$W" ] && SHORT="$SHORT $R($H/$W)"
done
[ -n "$SHORT" ] && { echo "FATAL appleFace short:$SHORT"; exit 1; }

echo "########## T2 $COND"
$PY scripts/anon/id_attack.py --ctx 0 --probe-root "$D/$COND" \
    --n-gallery 60 --n-probe 60 --tag "$COND" --attributes \
    --json-out "$RES/t2_${COND}.json" || echo "FAILED T2"
echo "########## T1 conservative"
$PY scripts/anon/linkage_attack.py --probe-root "$D/$COND" \
    --tag "t1_${COND}" --json-out "$RES/t1_${COND}.json" || echo "FAILED T1"
echo "########## T1 self-align"
$PY scripts/anon/linkage_attack.py --probe-root "$D/$COND" --self-align \
    --tag "t1_${COND}_selfalign" --json-out "$RES/t1_${COND}_selfalign.json" || echo "FAILED T1self"
echo "########## T4 linear, face crop"
$PY scripts/anon/adaptive_attack.py \
    --orig /springbrook/share/eng/esrpxk/results/anon_stage3b/t1_none.emb.npz \
    --anon "$RES/t1_${COND}.emb.npz" --tag swap2 --out "$RES/t4_swap2.json" || echo "FAILED T4"
echo ALL_DONE
