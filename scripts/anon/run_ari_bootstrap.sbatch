#!/bin/bash
# Review point M6: every ARI in the draft is one KMeans fit at random_state=0,
# with no clustering variability and no interval. Supply both, on the arms the
# paper's argument rests on. CPU only.
#SBATCH --job-name=ari_boot
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/ari_boot_%j.log
set -uo pipefail
export OMP_NUM_THREADS=2
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/gaze/bin/python
R=/springbrook/share/eng/esrpxk/results/anon_ari_ci
mkdir -p "$R"

A=/springbrook/share/eng/esrpxk/results/anon_adaptive
S2=/springbrook/share/eng/esrpxk/results/anon_swap2
DP=/springbrook/share/eng/esrpxk/results/anon_dp2

run () {   # tag  embfile
  [ -f "$2" ] || { echo "MISSING $2"; return; }
  echo "########## $1"
  $PY scripts/anon/ari_bootstrap.py "$2" --tag "$1" \
      --json-out "$R/ari_$1.json" || echo "FAILED $1"
}

for F in appleLeftEye appleRightEye; do
  run "original_$F"        "$A/emb_ProcessedData_$F.npz"
  run "swap_t1_$F"         "$A/emb_ProcessedSwap_$F.npz"
  run "swap_t2_$F"         "$S2/emb_ProcessedSwap2_$F.npz"
  run "dp2_persubject_$F"  "$DP/emb_dp2t1per_subject_$F.npz"
  run "dp2_perframe_$F"    "$DP/emb_dp2t1per_frame_$F.npz"
done

echo ALL_DONE
