#!/bin/bash
#SBATCH --job-name=paxis_atk
#SBATCH --partition=gpu,vis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/springbrook/share/eng/esrpxk/logs/paxis_atk_%j.log
set -uo pipefail
export OMP_NUM_THREADS=4
SP=/springbrook/share/eng/esrpxk/envs/idattack/lib/python3.11/site-packages
NVLIBS=$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
cd /springbrook/share/eng/esrpxk/CogntiveGaze
PY=/springbrook/share/eng/esrpxk/envs/idattack/bin/python
P=/springbrook/share/eng/esrpxk/datasets/anon_paxis
RES=/springbrook/share/eng/esrpxk/results/anon_paxis
mkdir -p "$RES"
# The P-axis question is where the eye-crop leak lives: in the ~5% that is the
# palpebral opening, or the ~95% periocular surround. T2 on the FACE crop (the
# released artefact, alignment from the original per SS3) and on the EYE crops
# directly with --no-detect, since a face detector finds nothing in these.
for LV in P1 P2; do
  echo "########## $LV face"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$P/$LV" --folder appleFace \
      --n-gallery 60 --n-probe 60 --tag "paxis_${LV}_face" \
      --json-out "$RES/paxis_${LV}_face.json" || echo "FAILED $LV face"
  echo "########## $LV eye"
  $PY scripts/anon/id_attack.py --ctx 0 --probe-root "$P/$LV" --folder appleLeftEye \
      --no-detect --n-gallery 60 --n-probe 60 --tag "paxis_${LV}_eye" \
      --json-out "$RES/paxis_${LV}_eye.json" || echo "FAILED $LV eye"
done
echo ALL_DONE
