#!/bin/bash
#SBATCH --job-name=t1_bg
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/t1_bg_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b

# T1 NUISANCE FLOOR. --background ignores the released crop entirely and clusters
# a face-free top-left corner patch of the original frame. It is the same
# construction as nuisance_control.py (which gave T2 Rank-1 50.5%, d' 0.46),
# scored through the T1 metric instead.
#
# The question it answers: SimSwap reports ARI 1.000 / purity 1.000 / k=18 in BOTH
# regimes. Clustering is an easier task than identification, so single-session
# nuisance -- lighting, camera gain, background -- might reach ARI 1.000 on its
# own. If it does, ARI cannot discriminate operators in this dataset at all, and
# the "every operator leaves the partition exactly recoverable" line in SS2.4 is
# a statement about the recording sessions, not about the operators.
#
# The probe-root argument is required by the CLI but unused under --background
# (the patch is read from OriginalData); pass `none` so the recording list and
# the sampled frame NAMES match the operator runs exactly.
for PATCH in 120 224; do
  echo "########## T1-background patch=${PATCH} (conservative-equivalent)"
  $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/none" \
      --background --patch $PATCH --n-per-subject 60 \
      --tag "t1bg_p${PATCH}" --json-out "$RES/t1bg_p${PATCH}.json"
done
echo ALL_DONE
