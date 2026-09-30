#!/bin/bash
#SBATCH --job-name=anon_unmask
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/anon_unmask_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
FD=/springbrook/share/eng/esrpxk/envs/facedet/bin/python
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
U=/springbrook/share/eng/esrpxk/datasets/anon_unmasked
RES=/springbrook/share/eng/esrpxk/results/anon_stage3b
RECS="00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019 00020 00021 00022 00023"

# Decisive ablation: identical operators, eye protection OFF. If masked stays at
# ~98% TAR while unmasked collapses, the residual leakage is carried by the
# byte-exact eye regions, not by the processed face region.
for OP in blur0.10 blur0.20 blur0.35 blur0.50 pixel0.25 pixel0.40 blackbox; do
  echo "########## generate $OP (unmasked)"
  for REC in $RECS; do
    $FD scripts/anon/make_anon_frames.py --rec "$REC" --op "$OP" \
        --dst-root "$U/$OP" --protect-eyes 0 --limit 1200 &
  done
  wait
  echo "########## attack $OP (unmasked)"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$U/$OP" \
      --n-gallery 60 --n-probe 60 --tag "${OP}_unmasked" --attributes \
      --json-out "$RES/t2_${OP}_unmasked.json"
done
echo ALL_DONE
