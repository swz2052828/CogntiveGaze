#!/bin/bash
#SBATCH --job-name=anon_t1
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_t1_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
A=/springbrook/share/eng/esrpxk/datasets/anon
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
mkdir -p "$RES"

# Ordered so the cells that carry the unique T1 contribution run FIRST: the
# conservative-vs-self-align gap at the extreme end. That gap isolates protection
# that comes from breaking face DETECTION rather than from removing identity,
# which no other experiment in the suite measures. If the job is cut short, the
# comparison is already secured.
OPS="none blackbox blur0.50 blur0.35 blur0.20 blur0.10 blur0.03 \
     pixel0.40 pixel0.25 pixel0.12 pixel0.06 pixel0.03"

# --- T1 linkage within the release (primary, §3) ------------------------------
# Two alignment regimes: the conservative §3 rule (landmarks from the original)
# and the strict release-only attacker. The gap between them is the protection
# that comes from breaking face detection rather than from removing identity.
for OP in $OPS; do
  echo "########## T1 $OP"
  $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-per-subject 60 --tag "t1_$OP" \
      --json-out "$RES/t1_$OP.json"
  echo "########## T1-selfalign $OP"
  $PY scripts/anon/linkage_attack.py --ctx 0 --probe-root "$A/$OP" \
      --n-per-subject 60 --self-align --tag "t1sa_$OP" \
      --json-out "$RES/t1sa_$OP.json"
done
echo ALL_DONE
